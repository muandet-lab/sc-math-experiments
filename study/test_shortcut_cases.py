import json
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from study.generate_shortcut_cases import CASES, UNDETERMINED, generate, validate_pair
from study.run_shortcut_cases import load_items, parse_final_answer, run
from study.summarize_shortcut_cases import summarize


class ShortcutCaseGenerationTests(unittest.TestCase):
    def test_pairs_are_reproducible_and_have_intended_manipulation(self):
        for case in CASES:
            with self.subTest(case=case):
                rows = generate(case, 24, 1739)
                self.assertEqual(rows, generate(case, 24, 1739))
                self.assertEqual(len(rows), 48)
                self.assertEqual(len({row["item_id"] for row in rows}), 48)
                for index in range(0, len(rows), 2):
                    left, right = rows[index:index + 2]
                    validate_pair([left, right])
                    self.assertEqual(left["generator_seed"], 1739)
                    if case in (2, 3, 4):
                        self.assertEqual(left["answer"], right["answer"])
                    if case == 2:
                        self.assertEqual(left["operation"], right["operation"])
                        self.assertIn(left["operation"], ("multiply", "divide"))
                    if case == 3:
                        self.assertEqual(left["problem"].split(". ")[-1],
                                         right["problem"].split(". ")[-1])
                    if case == 4:
                        left_sentences = left["problem"].split(". ")
                        right_sentences = right["problem"].split(". ")
                        self.assertEqual(right_sentences,
                                         [left_sentences[0], left_sentences[2],
                                          left_sentences[1], left_sentences[3]])
                    if case == 5:
                        self.assertEqual(right["answer"], UNDETERMINED)
                        self.assertTrue(left["problem"].endswith(right["problem"]))
                    if case == 6:
                        self.assertNotEqual(left["answer"], right["answer"])
                        self.assertEqual(left["problem"].rsplit(". ", 1)[0],
                                         right["problem"].rsplit(". ", 1)[0])

    def test_loader_requires_complete_unique_pairs(self):
        rows = generate(5, 2, 937)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(load_items(path), rows)
            path.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "complete matched pairs"):
                load_items(path)


class AnswerParsingTests(unittest.TestCase):
    def test_numeric_and_unsolvable_final_answers(self):
        self.assertEqual(parse_final_answer("<think>reason</think>\nFinal answer: 12"), 12)
        self.assertEqual(parse_final_answer("**Final answer:** 7 coins."), 7)
        self.assertEqual(parse_final_answer("Final answer: cannot be determined"),
                         UNDETERMINED)
        self.assertEqual(parse_final_answer("Final answer: insufficient information"),
                         UNDETERMINED)
        self.assertEqual(parse_final_answer(
            "</think> Maya has 16 coins. Final answer: 16"), 16)
        self.assertEqual(parse_final_answer(
            "<think>Final answer: cannot be determined</think> "
            "The facts suffice. Final answer: 5"), 5)
        self.assertEqual(parse_final_answer(
            "</think> Final answer: <integer> 8"), 8)
        self.assertIsNone(parse_final_answer(
            "<think>Final answer: 8", finish_reason="length"))
        self.assertIsNone(parse_final_answer(
            "<think>Final answer: 8</think> Still explaining...", finish_reason="length"))
        self.assertIsNone(parse_final_answer("I think the answer is 7."))
        self.assertIsNone(parse_final_answer("Final answer: 7.5"))

    def test_summary_counts_case_specific_failures(self):
        rows = generate(6, 1, 42)
        for row in rows:
            row.update(model="Qwen/Qwen3-0.6B", revision="example-revision",
                       mode="thinking", finish_reason="stop")
        rows[0].update(parsed_answer=rows[0]["answer"], correct=True)
        rows[1].update(parsed_answer=rows[0]["answer"], correct=False)
        for row in rows:
            row["raw_response"] = f"</think> Final answer: {row['parsed_answer']}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                summarize(path)
        self.assertIn("changed-query answers repeating original gold=1/1",
                      printed.getvalue())

    def test_summary_regrades_saved_raw_responses_without_rewriting_file(self):
        rows = generate(3, 1, 42)
        for row in rows:
            row.update(model="Qwen/Qwen3-0.6B", revision="example-revision",
                       mode="thinking", finish_reason="stop",
                       parsed_answer=None, correct=False,
                       raw_response=f"</think> The result follows. Final answer: {row['answer']}")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            original = "\n".join(json.dumps(row) for row in rows) + "\n"
            path.write_text(original)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                summarize(path)
            self.assertEqual(path.read_text(), original)
        self.assertIn("transfer: 1/1; comparison: 1/1", printed.getvalue())
        self.assertIn("regraded_rows=2", printed.getvalue())


class ThinkingRunnerTests(unittest.TestCase):
    def test_both_model_paths_preserve_thinking_and_score_solvability(self):
        items = generate(5, 1, 42)
        seen = {}

        class FakeEngine:
            def chat(self, messages, sampling, chat_template_kwargs):
                seen["messages"] = messages
                seen["sampling"] = sampling
                seen["template_kwargs"] = chat_template_kwargs
                responses = []
                for item in items:
                    answer = (str(item["answer"]) if isinstance(item["answer"], int)
                              else "cannot be determined")
                    candidate = types.SimpleNamespace(
                        text=f"<think>work</think>\nFinal answer: {answer}",
                        token_ids=[1, 2], finish_reason="stop")
                    responses.append(types.SimpleNamespace(
                        outputs=[candidate], prompt_token_ids=[1, 2, 3]))
                return responses

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(device_count=lambda: 1)
        torch.version = types.SimpleNamespace(cuda="13.0")
        torch.__version__ = "test"
        vllm = types.ModuleType("vllm")
        vllm.__version__ = "test"
        vllm.LLM = lambda **kwargs: FakeEngine()
        vllm.SamplingParams = lambda **kwargs: kwargs
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: types.SimpleNamespace(
            model_info=lambda repository: types.SimpleNamespace(sha="test-revision"))

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "items.jsonl"
            input_path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
            with patch.dict(sys.modules, {"torch": torch, "vllm": vllm,
                                          "huggingface_hub": hub}):
                for model in ("qwen", "olmo"):
                    output = Path(directory) / f"{model}.jsonl"
                    with contextlib.redirect_stdout(io.StringIO()):
                        run(model, input_path, output)
                    rows = [json.loads(line) for line in output.read_text().splitlines()]
                    self.assertEqual([row["correct"] for row in rows], [True, True])
                    self.assertEqual(rows[1]["parsed_answer"], UNDETERMINED)
                    self.assertTrue(all(row["mode"].endswith("thinking") for row in rows))
                    self.assertEqual(seen["template_kwargs"],
                                     {"enable_thinking": True} if model == "qwen" else None)


if __name__ == "__main__":
    unittest.main()
