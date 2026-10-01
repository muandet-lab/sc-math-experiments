import json
import tempfile
import unittest
from pathlib import Path

from study.generate_case5_confirmatory import COUNT, generate, validate
from study.generate_shortcut_cases import UNDETERMINED
from study.analyze_case5_confirmatory import analyze
from study.run_case5_confirmatory import conversation, load_items, load_manifest
from study.score_case5_confirmatory import SCORER_VERSION, score


HERE = Path(__file__).parent


class ConfirmatoryCase5Tests(unittest.TestCase):
    def test_frozen_grid_and_arithmetic(self):
        items, probes = generate()
        validate(items, probes)
        self.assertEqual(len(items), COUNT * 23)
        self.assertEqual(len(probes), COUNT * 2)
        with tempfile.TemporaryDirectory() as directory:
            item_path = Path(directory) / "items.jsonl"
            item_path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
            self.assertEqual(items, load_items(item_path))
        first = {r["kind"]: r for r in items[:23]}
        self.assertEqual(first["complete"]["gold"], 16)
        self.assertEqual(first["zero_start"]["gold"], 5)
        self.assertEqual(first["net_change"]["gold"], 5)
        self.assertEqual(first["infeasible_start"]["zero_default"], -5)
        self.assertEqual(first["marked_pocket"]["symbolic_gold"], [1, 16])
        for item in items:
            if item["kind"] not in ("zero_start", "net_change"):
                self.assertNotEqual(item["gold"], item["zero_default"])

    def test_all_models_pinned(self):
        models = load_manifest(HERE / "case5_confirmatory_models.json")
        self.assertEqual(len(models), 8)
        for model in models.values():
            self.assertEqual(len(model["revision"]), 40)
            self.assertIn(model["samples"], (2, 4))

    def test_prompts_have_fresh_context_and_instruction_placement(self):
        items, _ = generate()
        for kind in ("factorial", "marked_pocket", "net_change"):
            item = next(r for r in items if r["kind"] == kind)
            for task in ("solve", "answerability"):
                no = conversation(item, task, "none")
                yes = conversation(item, task, "user")
                self.assertEqual(no[0], yes[0])
                self.assertTrue(yes[1]["content"].startswith(no[1]["content"]))
                self.assertIn("Do not assign values", yes[1]["content"])
                if task == "answerability":
                    self.assertIn("uniquely determine", no[1]["content"])

    def test_frozen_scorer_categories(self):
        items, _ = generate()
        omitted = items[0]
        complete = next(r for r in items if r["kind"] == "complete")
        self.assertEqual(score(omitted, "solve", "</think>Final answer: 5", "stop")["outcome"], "zero_default")
        self.assertEqual(score(omitted, "solve", "</think>Final answer: x + 5", "stop")["outcome"], "correct")
        self.assertEqual(score(omitted, "solve", "</think>Final answer: cannot be determined", "stop")["outcome"], "correct")
        self.assertEqual(score(complete, "solve", "</think>Final answer: 16 shells", "stop")["outcome"], "correct")
        self.assertEqual(score(complete, "solve", "</think>Final answer: cannot be determined", "stop")["outcome"], "other_error")
        self.assertEqual(score(omitted, "answerability", "</think>Final answer: no", "stop")["outcome"], "correct")
        self.assertEqual(score(complete, "answerability", "</think>Final answer: yes", "stop")["outcome"], "correct")
        self.assertEqual(score(omitted, "solve", "</think>Final answer: 5", "length")["outcome"], "truncated")
        self.assertEqual(score(omitted, "solve", "<think>unfinished", "stop")["outcome"], "unparseable")

    def test_analysis_accepts_a_complete_synthetic_grid(self):
        items, probes = generate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"fake": {"repository": "test", "revision": "a" * 40,
                                 "billions": 1, "samples": 2}}
            (root / "models.json").write_text(json.dumps(manifest))
            rows = []
            for item in items:
                for task in ("solve", "answerability"):
                    for placement in ("none", "user"):
                        for sample in range(2):
                            parsed = ("no" if item["gold"] == UNDETERMINED else "yes") if task == "answerability" else item["gold"]
                            rows.append({"item_id": item["item_id"], "base_id": item["base_id"],
                                         "task": task, "instruction_placement": placement,
                                         "sample_index": sample, "scorer_version": SCORER_VERSION,
                                         "revision": "a" * 40, "correct": True,
                                         "outcome": "correct", "parsed": parsed})
            (root / "fake.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "fake.prefill.jsonl").write_text("".join(json.dumps({"probe_id": p["probe_id"], "revision": "a" * 40,
                "condition": p["condition"], "log_odds_zero_vs_received": 0}) + "\n" for p in probes))
            items_path = root / "items.jsonl"
            items_path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
            metrics, conditions, regression = analyze(items_path,
                                                       root / "models.json", root)
            self.assertEqual(metrics["fake"]["zero_default_rate"], 0)
            self.assertEqual(metrics["fake"]["over_abstention_rate"], 0)
            self.assertEqual(len(conditions), 36)
            self.assertEqual(len(regression), COUNT * 19 * 2 * 2)


if __name__ == "__main__":
    unittest.main()
