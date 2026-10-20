import itertools
import random
import unittest

from study.generate_case5_confirmatory import generate as generate_case5
from study.uc_elicit import classify_answer, parse_profile, task_messages, two_stage_messages
from study.uc_items import GRID, TAIL, build, case5_bases
from study.uc_measures import (cdf_bounds_intervals, cdf_bounds_possibility, mmi_bound_intervals,
                               mmi_intervals, mmi_possibility, reachable, summarize,
                               wasserstein_diameter)
from study.uc_run import plan


def _vertices(lower, upper):
    """Vertices of {p : l <= p <= u, sum p = 1}: all but one coordinate at a bound."""
    n, found = len(lower), []
    for free in range(n):
        for bounds in itertools.product(*[(lower[i], upper[i]) for i in range(n) if i != free]):
            rest = list(bounds)
            value = 1 - sum(rest)
            if lower[free] - 1e-12 <= value <= upper[free] + 1e-12:
                found.append(rest[:free] + [value] + rest[free:])
    return found


def _brute(lower, upper):
    vs = _vertices(lower, upper)
    tv = max(0.5 * sum(abs(a - b) for a, b in zip(p, q)) for p in vs for q in vs)
    w1 = max(sum(abs(sum(p[:k + 1]) - sum(q[:k + 1])) for k in range(len(p) - 1))
             for p in vs for q in vs)
    return tv, w1


def _random_intervals(rng, n):
    while True:
        lower = [rng.choice([0, 0, rng.random() * 0.4]) for _ in range(n)]
        upper = [min(1.0, l + rng.random() * 0.7) for l in lower]
        if sum(lower) <= 1 <= sum(upper):
            return lower, upper


class ItemTests(unittest.TestCase):
    problems = build(case5_bases())

    def test_counts_and_case5_wording(self):
        self.assertEqual(len(case5_bases()), 36)
        self.assertEqual(len(self.problems), 36 * 10)
        case5 = {(i["base_id"], i["salience"]): i["problem"] for i in generate_case5()[0]
                 if i["kind"] == "factorial" and i["slot"] == "initial" and i["chain_length"] == 2}
        for p in self.problems:
            if p["content"] == "unconstrained" and p["presentation"] in ("omitted", "explicit"):
                salience = "implicit" if p["presentation"] == "omitted" else "explicit"
                self.assertEqual(f"{p['facts']} {p['question']}", case5[(p["base_id"], salience)])

    def test_true_sets(self):
        by = {(p["base_id"], p["content"], p["presentation"]): p for p in self.problems}
        base = case5_bases()[0]
        d, low = base["received"] - base["given"], base["start"] - 2
        self.assertEqual(by[(0, "determined", "stated")]["true_set"], [str(base["start"] + d)])
        bounded = by[(0, "bounded", "passing")]["true_set"]
        spread = by[(0, "spread", "prominent")]["true_set"]
        self.assertEqual(bounded, [str(low + d + k) for k in range(4)])
        self.assertEqual(spread, [str(low + d + 5 * k) for k in range(4)])
        self.assertEqual(by[(0, "one_sided", "prominent")]["true_set"][0], str(low + d))
        unconstrained = by[(0, "unconstrained", "omitted")]["true_set"]
        self.assertEqual((unconstrained[0], unconstrained[-1]), (str(d), TAIL))
        self.assertIn("who started out with between", by[(0, "bounded", "passing")]["facts"])


class MeasureTests(unittest.TestCase):
    def test_interval_mmi_and_wasserstein_match_vertex_enumeration(self):
        rng = random.Random(7)
        for _ in range(300):
            lower, upper = _random_intervals(rng, rng.randint(2, 6))
            tv, w1 = _brute(lower, upper)
            self.assertAlmostEqual(mmi_intervals(lower, upper), tv, places=4)
            self.assertAlmostEqual(wasserstein_diameter(*cdf_bounds_intervals(lower, upper)), w1,
                                   places=6)
            self.assertLessEqual(mmi_intervals(lower, upper),
                                 mmi_bound_intervals(reachable(lower, upper)[0]) + 1e-9)

    def test_possibility_measures(self):
        def indicator(values):
            return [1.0 if i in values else 0.0 for i in range(len(GRID))]
        wide, ends = indicator(range(5, 21)), indicator({5, 20})
        for pi in (wide, ends):  # range and count are different axes
            self.assertAlmostEqual(wasserstein_diameter(*cdf_bounds_possibility(pi)), 15)
        self.assertEqual(mmi_possibility(indicator({7})), 0.0)
        self.assertEqual(mmi_possibility([0.2, 0.4, 0.1]), 0.5)
        self.assertAlmostEqual(wasserstein_diameter(*cdf_bounds_possibility([0.5, 1.0, 0.25])),
                               0.75)

    def test_summary_flags(self):
        point = {y: (1.0 if y == "5" else 0.0) for y in GRID}
        s = summarize("pos", point, GRID, [str(v) for v in range(5, 51)] + [TAIL], own_answer="5")
        self.assertEqual((s["set_size"], s["mmi"], s["echo"], s["contains_true"], s["within_true"]),
                         (1, 0.0, True, False, True))
        vacuous = {y: [0.0, 1.0] for y in GRID}
        s = summarize("probint", vacuous, GRID, ["7"])
        self.assertTrue(s["all_ones"] and s["contains_true"])
        self.assertAlmostEqual(s["mmi"], 1.0)


class ElicitTests(unittest.TestCase):
    problem = next(p for p in build(case5_bases()[:1])
                   if p["content"] == "unconstrained" and p["presentation"] == "omitted")

    def test_parse_profiles(self):
        raw = '</think>Reasoning.\nFinal answer: {"5": [0.2, 0.6], "5-50": [0.0, 0.05], "more than 50": [0, 0.1]}'
        r = parse_profile("probint", raw, "stop", "thinking")
        self.assertTrue(r["valid"])
        self.assertEqual((r["profile"]["5"], r["profile"]["6"], r["profile"]["4"]),
                         ([0.2, 0.6], [0.0, 0.05], [0.0, 0.0]))
        self.assertEqual(parse_profile("probint", '</think>Final answer: {"5": [0.7, 0.9], "6": [0.5, 0.9]}',
                                       "stop", "thinking")["error"], "lower_sum_above_1")
        self.assertFalse(parse_profile("pos", '{"5": 1.0}', "stop", "thinking")["valid"])
        self.assertTrue(parse_profile("pos", 'Final answer: {"5-50": 1, "more than 50": 1}', "stop",
                                      "non_thinking")["valid"])
        nested = parse_profile("pos", 'Final answer: {"14-17": {"14": 0.8, "17": 0.5}}', "stop", "non_thinking")
        self.assertEqual((nested["profile"]["14"], nested["profile"]["15"]), (0.8, 0.0))
        self.assertEqual(parse_profile("pos", 'Final answer: {"answer": "5"}', "stop", "non_thinking")["error"],
                         "bad_key:answer")
        self.assertEqual(parse_profile("pos", 'Final answer: {"x": 1}', "stop", "non_thinking")["error"],
                         "bad_key:x")

    def test_classify_answers(self):
        def outcome(line, task="T1"):
            return classify_answer(self.problem, task, f"</think>Final answer: {line}", "stop", "thinking")
        d = self.problem["zero_default"]
        self.assertEqual((outcome(str(d))["outcome"], outcome(str(d))["zero_default"]), ("single", True))
        self.assertEqual(outcome("cannot be determined")["outcome"], "abstain")
        self.assertTrue(outcome(f"x + {d}")["correct"])
        self.assertEqual(outcome(f"at least {d}")["parsed"], self.problem["true_set"])
        self.assertTrue(outcome(f"at least {d}")["correct"])
        self.assertEqual(outcome("7, 12, 17, or 22")["parsed"], ["7", "12", "17", "22"])
        self.assertEqual(outcome("no", "T4")["correct"], True)

    def test_prompts_and_plan(self):
        t1 = task_messages(self.problem, "T1")
        self.assertIn("cannot be determined", t1[0]["content"])
        self.assertNotIn("cannot be determined", task_messages(self.problem, "T0")[0]["content"])
        merged = task_messages(self.problem, "T0", system_in_user=True)
        self.assertEqual([m["role"] for m in merged], ["user"])
        convo = two_stage_messages(self.problem, "T1", "Final answer: 5", "pos")
        self.assertEqual([m["role"] for m in convo], ["system", "user", "assistant", "user"])
        with self.assertRaises(ValueError):
            plan([self.problem], ["T4"], ["two_stage"], ["pos"])


if __name__ == "__main__":
    unittest.main()
