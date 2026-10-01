import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from study.generate_case5_diagnostics import VARIANTS, generate, validate_items
from study.generate_shortcut_cases import UNDETERMINED
from study.run_case5_diagnostics import _affine, grade, run
from study.summarize_case5_diagnostics import summarize


class Case5DiagnosticTests(unittest.TestCase):
    def test_generated_variants_keep_distinct_numbers_and_correct_gold(self):
        rows = generate(30, 615)
        self.assertEqual(rows, generate(30, 615))
        validate_items(rows)
        self.assertEqual(len(rows), 30 * len(VARIANTS))
        for offset in range(0, len(rows), len(VARIANTS)):
            block = {row["variant"]: row for row in rows[offset:offset + len(VARIANTS)]}
            start, received, given = (block["complete_solve"][key]
                                      for key in ("start", "received", "given"))
            self.assertEqual(len({start, received, given}), 3)
            self.assertEqual(block["complete_solve"]["answer"], start + received - given)
            self.assertEqual(block["zero_control_solve"]["answer"], received - given)
            self.assertEqual(block["unknown_delta_solve"]["answer"], received - given)
            self.assertEqual(block["missing_initial_solve"]["answer"], UNDETERMINED)
            self.assertEqual(block["missing_received_solve"]["answer"], UNDETERMINED)
            self.assertEqual(block["missing_given_solve"]["answer"], UNDETERMINED)
            self.assertEqual(block["missing_initial_solve"]["symbolic_gold"],
                             (1, received - given))
            self.assertEqual(block["missing_given_solve"]["symbolic_gold"],
                             (-1, start + received))
            self.assertEqual(block["complete_answerability"]["answer"], "yes")
            self.assertEqual(block["missing_initial_answerability"]["answer"], "no")
            self.assertNotIn(str(start), block["missing_initial_solve"]["problem"])

    def test_distinguishes_guess_abstention_expression_conditional_and_truncation(self):
        block = {row["variant"]: row for row in generate(1, 42)}
        missing = block["missing_initial_solve"]
        delta = missing["delta"]
        self.assertEqual(_affine(f"x + {missing['received']} - {missing['given']}"),
                         (1, delta))
        cases = (
            (missing, "Final answer: cannot be determined", "correct_underdetermined"),
            (missing, f"Final answer: x + {delta}",
             "correct_symbolic_underdetermined"),
            (missing, "Final answer: 1", "unsupported_numeric"),
            (missing, "Final answer: If she started at zero, then 1",
             "conditional_answer"),
            (block["complete_solve"],
             f"Final answer: {block['complete_solve']['answer']}", "correct_numeric"),
            (block["unknown_delta_solve"], f"Final answer: +{delta}", "correct_numeric"),
            (block["missing_initial_answerability"], "Final answer: no",
             "correct_answerability"),
            (block["complete_answerability"], "Final answer: yes",
             "correct_answerability"),
        )
        for row, answer, category in cases:
            with self.subTest(variant=row["variant"], answer=answer):
                result = grade(row, f"<think>work</think>\n{answer}", "stop")
                self.assertEqual(result["classification"], category)
        zero_guess = grade(missing, f"</think>Final answer: {delta}", "stop")
        self.assertEqual(zero_guess["classification"], "unsupported_numeric")
        self.assertTrue(zero_guess["zero_equivalent_guess"])
        self.assertEqual(grade(missing, "<think>Final answer: 1", "length")
                         ["classification"], "truncation")

    def test_runner_uses_fresh_conversations_and_summary_regrades(self):
        items = generate(1, 42)
        seen = {}

        class FakeEngine:
            def chat(self, messages, sampling, chat_template_kwargs):
                seen["messages"] = messages
                seen["sampling"] = sampling
                seen["template_kwargs"] = chat_template_kwargs
                responses = []
                for item in items:
                    answer = (item["answer"] if item["answer"] != UNDETERMINED
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
                output = Path(directory) / "responses.jsonl"
                with contextlib.redirect_stdout(io.StringIO()):
                    run("qwen", input_path, output)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(rows), len(VARIANTS))
                self.assertTrue(all(len(messages) == 2 for messages in seen["messages"]))
                self.assertEqual(seen["template_kwargs"], {"enable_thinking": True})
                self.assertEqual(seen["sampling"]["max_tokens"], 4096)
                self.assertEqual(rows[0]["max_model_len"], 8192)
                self.assertEqual(rows[1]["classification"], "correct_underdetermined")
                self.assertEqual(rows[7]["classification"], "correct_answerability")
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    summarize(output)
                self.assertIn("missing_initial_solve: correct=1/1", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
