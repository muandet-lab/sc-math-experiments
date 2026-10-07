import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from study.case5_relational import CONDITIONS, generate, load_items, score_file, score_row
from study.generate_case5_confirmatory import COUNT


class RelationalCase5Tests(unittest.TestCase):
    def test_matched_items(self):
        items = generate()
        self.assertEqual(len(items), COUNT * len(CONDITIONS))
        first = {item["condition"]: item for item in items if item["base_id"] == 0}
        self.assertEqual(first["direct"]["problem"],
                         "Maya starts with 11 shells. Maya receives 9 shells. "
                         "Maya gives away 4 shells. How many shells does Maya have now?")
        self.assertIn("Eli starts with 11 shells", first["same_as"]["problem"])
        self.assertIn("Maya starts with 7 more shells than Maya later gives away",
                      first["more_than_given"]["problem"])
        self.assertIn("Eli starts with 22 shells", first["half_as_many"]["problem"])
        self.assertEqual({item["gold"] for item in first.values()}, {16})
        self.assertEqual({item["zero_default"] for item in first.values()}, {5})

    def test_scoring_separates_zero_default_from_other_errors(self):
        item = generate()[1]
        def row(answer, finish="stop"):
            return {"raw_response": "</think>Final answer: " + answer,
                    "finish_reason": finish}
        self.assertEqual(score_row(item, row("16"))["outcome"], "correct")
        self.assertEqual(score_row(item, row("5"))["outcome"], "zero_default")
        self.assertEqual(score_row(item, row("cannot be determined"))["outcome"], "other_error")
        self.assertEqual(score_row(item, row("16", "length"))["outcome"], "truncated")

    def test_score_file_requires_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = generate()
            item_path = root / "items.jsonl"
            item_path.write_text("".join(json.dumps(item) + "\n" for item in items))
            self.assertEqual(load_items(item_path), items)
            manifest_path = root / "models.json"
            revision = "a" * 40
            manifest_path.write_text(json.dumps({"fake": {"revision": revision, "samples": 2}}))
            item_hash = hashlib.sha256(item_path.read_bytes()).hexdigest()
            raw = [{"item_id": item["item_id"], "condition": item["condition"],
                    "sample_index": sample, "model_key": "fake", "revision": revision,
                    "benchmark_sha256": item_hash,
                    "raw_response": "</think>Final answer: " + str(item["zero_default"]),
                    "finish_reason": "stop"}
                   for item in items for sample in range(2)]
            raw_path = root / "raw.jsonl"
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw))
            counts = score_file(item_path, raw_path, root / "scored.jsonl", manifest_path)
            self.assertEqual(counts["direct"]["zero_default"], COUNT * 2)
            self.assertEqual(sum(counts[condition]["zero_default"] for condition in CONDITIONS),
                             len(raw))
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw[:-1]))
            with self.assertRaises(ValueError):
                score_file(item_path, raw_path, root / "incomplete.jsonl", manifest_path)


if __name__ == "__main__":
    unittest.main()
