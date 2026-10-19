import json
import tempfile
import unittest
from pathlib import Path

from study.generate_case5_confirmatory import generate as generate_case5
from study.generate_shortcut_cases import UNDETERMINED
from study.shortcut_rep_analyze import analyze
from study.shortcut_rep_items import (build_distractor, build_keyword, build_missing_start,
                                      build_option_position, validate)
from study.shortcut_rep_run import conversation, load_manifest, run
from study.shortcut_rep_score import score

MMLU = [{"question": "Which is a prime?", "subject": "math", "choices": ["4", "6", "7", "9"],
         "answer": 2, "source_row": 11}]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class ItemTests(unittest.TestCase):
    def test_keyword_bases_are_matched_and_signatures_follow_the_keyword(self):
        items = build_keyword(3, count=8)
        validate(items, "keyword")
        for base in range(8):
            group = {i["cell"]: i for i in items if i["base_id"] == base}
            self.assertEqual(len(group), 5)
            self.assertEqual(len({i["gold"] for i in group.values()}), 1)
            self.assertTrue(group["arithmetic"]["user"].startswith("Compute "))
            self.assertIsNone(group["verbal_consistent"]["signature"])
            self.assertEqual(group["verbal_inconsistent"]["signature"],
                             group["symbolic_reversed"]["signature"])
        # Applying the keyword's operation shifts an additive answer by twice the amount.
        additive = [i for i in items if i["cell"] == "verbal_inconsistent"
                    and i["operation"] in ("add", "subtract")]
        self.assertTrue(additive)
        for item in additive:
            sign = -2 if item["operation"] == "add" else 2
            self.assertEqual(item["signature"] - item["gold"], sign * item["amount"])

    def test_distractor_changes_only_the_inserted_clause(self):
        items = build_distractor(5, count=6)
        validate(items, "distractor")
        for base in range(6):
            group = {i["cell"]: i for i in items if i["base_id"] == base}
            none = group["none"]["user"].split(". ")
            d = group["plain"]["distractor_amount"]
            for cell in ("plain", "keyword_fewer", "keyword_more"):
                sentences = group[cell]["user"].split(". ")
                self.assertEqual(sentences[:1] + sentences[2:], none)
            self.assertEqual(group["keyword_fewer"]["signature"], group["none"]["gold"] - d)
            self.assertEqual(group["keyword_more"]["signature"], group["none"]["gold"] + d)

    def test_option_position_moves_gold_through_every_letter(self):
        items = build_option_position(1, MMLU)
        validate(items, "option_position")
        mmlu = [i for i in items if i["source"] == "mmlu"]
        self.assertEqual([i["gold"] for i in mmlu], list("ABCD"))
        self.assertEqual([i["options"] for i in mmlu][1], ["4", "7", "6", "9"])
        noids = [i for i in items if i["arm"] == "no_ids" and i["base_id"] == 0]
        self.assertEqual(len(noids), 4)
        self.assertNotIn("A.", noids[0]["user"])

    def test_validate_rejects_signature_equal_to_gold(self):
        items = build_keyword(3, count=2)
        broken = [dict(i) for i in items]
        target = next(i for i in broken if i["signature"] is not None)
        target["signature"] = target["gold"]
        with self.assertRaises(ValueError):
            validate(broken, "keyword")

    def test_missing_start_marks_only_underdetermined_items(self):
        with tempfile.TemporaryDirectory() as directory:
            case5, _ = generate_case5()
            path = _write(Path(directory) / "case5.jsonl", case5)
            items = build_missing_start(path)
        validate(items, "missing_start")
        self.assertEqual(len(items), 2 * len(case5))
        for item in items:
            if item["gold"] == UNDETERMINED:
                self.assertEqual(item["signature"], item["case5"]["zero_default"])
            else:
                self.assertIsNone(item["signature"])
        self.assertTrue(any("Use only the stated facts" in i["user"] for i in items))


class ScoreTests(unittest.TestCase):
    item = {"answer_type": "integer", "gold": 8, "signature": 2}

    def test_integer_outcomes(self):
        self.assertEqual(score(self.item, "</think>Final answer: 8", "stop", "thinking")["outcome"],
                         "correct")
        self.assertEqual(score(self.item, "</think>**Final answer:** $\\boxed{2}$", "stop",
                               "thinking")["outcome"], "signature")
        self.assertEqual(score(self.item, "</think>Final answer: 9 apples.", "stop",
                               "thinking")["outcome"], "other_error")
        self.assertEqual(score(self.item, "Final answer: 8", "length", "thinking")["outcome"],
                         "truncated")

    def test_thinking_requires_a_closed_block_but_non_thinking_does_not(self):
        self.assertEqual(score(self.item, "Final answer: 8", "stop", "thinking")["outcome"],
                         "unparseable")
        self.assertEqual(score(self.item, "Final answer: 8", "stop", "non_thinking")["outcome"],
                         "correct")
        self.assertEqual(score(self.item, "<think>Final answer: 2</think>Final answer: 8",
                               "stop", "thinking")["outcome"], "correct")

    def test_large_numbers_with_commas(self):
        item = {"answer_type": "integer", "gold": 1200, "signature": None}
        self.assertEqual(score(item, "Final answer: 1,200", "stop", "non_thinking")["outcome"],
                         "correct")

    def test_choice_outcomes(self):
        letter = {"answer_type": "letter", "gold": "C", "signature": None}
        result = score(letter, "Final answer: C. 7", "stop", "non_thinking")
        self.assertEqual((result["outcome"], result["chosen_index"]), ("correct", 2))
        self.assertEqual(score(letter, "Final answer: (B)", "stop", "non_thinking")["chosen_index"], 1)
        self.assertEqual(score(letter, "Final answer: Because", "stop", "non_thinking")["outcome"],
                         "unparseable")
        text = {"answer_type": "option_text", "gold": "7", "gold_index": 1, "signature": None,
                "options": ["4", "7", "6", "9"]}
        result = score(text, "Final answer: 6", "stop", "non_thinking")
        self.assertEqual((result["outcome"], result["chosen_index"]), ("other_error", 2))

    def test_case5_uses_frozen_scorer_in_both_modes(self):
        case5, _ = generate_case5()
        source = next(i for i in case5 if i["gold"] == UNDETERMINED and i.get("zero_default"))
        item = {"answer_type": "case5", "case5": source, "gold": source["gold"],
                "signature": source["zero_default"]}
        raw = f"Final answer: {source['zero_default']}"
        self.assertEqual(score(item, raw, "stop", "non_thinking")["outcome"], "signature")
        self.assertEqual(score(item, raw, "stop", "thinking")["outcome"], "unparseable")
        self.assertEqual(score(item, "Final answer: cannot be determined", "stop",
                               "non_thinking")["outcome"], "correct")


class RunTests(unittest.TestCase):
    def test_manifest_modes_and_r1_prompt(self):
        manifest = load_manifest()
        self.assertEqual(set(manifest["qwen_8b"]["modes"]), {"thinking", "non_thinking"})
        self.assertEqual(manifest["qwen_8b"]["modes"]["non_thinking"]["chat_template_kwargs"],
                         {"enable_thinking": False})
        item = {"system": "S", "user": "U"}
        self.assertEqual(conversation(item, manifest["r1_llama_8b"]),
                         [{"role": "user", "content": "S\n\nU"}])
        self.assertEqual(len(conversation(item, manifest["qwen_8b"])), 2)

    def test_refusals_before_loading_vllm(self):
        with tempfile.TemporaryDirectory() as directory:
            items = _write(Path(directory) / "kw.jsonl", build_keyword(3, count=2))
            output = Path(directory) / "out.jsonl"
            with self.assertRaises(ValueError):
                run("llama31_8b", "non_thinking", items, output)
            with self.assertRaises(ValueError):
                run("olmo_7b_think", "non_thinking", items, output)
            output.write_text(json.dumps({"item_id": "not-an-item", "sample_index": 0}) + "\n")
            with self.assertRaises(ValueError):
                run("qwen_0_6b", "thinking", items, output)


class ImportTests(unittest.TestCase):
    def test_case5_import_checks_prompts_and_coverage(self):
        from study.shortcut_rep_import_case5 import import_case5

        model = load_manifest()["olmo_7b_think"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case5, _ = generate_case5()
            items = build_missing_start(_write(root / "case5.jsonl", case5))
            items_path = _write(root / "ms.jsonl", items)
            raw = [{"item_id": i["case5"]["item_id"], "base_id": i["base_id"], "task": "solve",
                    "instruction_placement": i["instruction_placement"], "sample_index": n,
                    "model_key": "olmo_7b", "revision": model["revision"],
                    "benchmark_sha256": "b", "model_manifest_sha256": "m",
                    "sampling": {"n": 4, "temperature": 0.6, "top_p": 0.95,
                                 "max_tokens": 16384, "seed": 20261001},
                    "max_model_len": 20480, "system_prompt": i["system"],
                    "user_prompt": i["user"], "raw_response": "</think>Final answer: 1",
                    "finish_reason": "stop"} for i in items for n in range(4)]
            raw_path = _write(root / "raw.jsonl", raw)
            self.assertEqual(import_case5(raw_path, items_path, root / "out.jsonl"), len(raw))
            raw[0]["user_prompt"] += " changed"
            with self.assertRaises(ValueError):
                import_case5(_write(root / "bad.jsonl", raw), items_path, root / "bad_out.jsonl")
            with self.assertRaises(ValueError):
                import_case5(_write(root / "short.jsonl", raw[1:]), items_path,
                             root / "short_out.jsonl")


class AnalyzeTests(unittest.TestCase):
    def test_keyword_metrics_from_synthetic_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = build_keyword(3, count=40)
            _write(root / "keyword.jsonl", items)
            rows = []
            for item in items:
                # Verbal inconsistent items always fail with the keyword answer when it exists.
                failing = item["cell"] == "verbal_inconsistent" and item["signature"] is not None
                outcome = "signature" if failing else "correct"
                rows.append({"item_id": item["item_id"], "model_key": "m", "mode": "thinking",
                             "family": "keyword", "outcome": outcome,
                             "parsed": item["signature"] if failing else item["gold"]})
            scored = _write(root / "scored.jsonl", rows)
            results = {r["metric"]: r for r in analyze([scored], root)}
        self.assertEqual(results["accuracy:symbolic_reversed"]["estimate"], 1.0)
        self.assertGreater(results["cate_verbal"]["estimate"], 0.5)
        self.assertEqual(results["delta_kw"]["estimate"], results["cate_verbal"]["estimate"])
        self.assertEqual(results["cae_verbal"]["estimate"], 1.0)
        self.assertEqual(results["attributable_share"]["estimate"], 1.0)
        self.assertEqual(results["attributable_share_capable"]["estimate"], 1.0)
        self.assertLessEqual(results["cate_verbal"]["ci_low"], results["cate_verbal"]["estimate"])


if __name__ == "__main__":
    unittest.main()
