"""Run case-5 behavioral diagnostics with thinking enabled on one GPU."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from study.generate_case5_diagnostics import load_items
from study.generate_shortcut_cases import UNDETERMINED
from study.run_shortcut_cases import (FINAL_MARKER, MODELS, PARSER_VERSION,
                                      parse_final_answer_detailed)

SCORER_VERSION = 2
SOLVE_SYSTEM = ("Solve carefully. End with one line beginning 'Final answer:' "
                "followed by your actual result: a whole number, an expression "
                "using x for an unknown quantity, or 'cannot be determined' as "
                "appropriate. Never write a placeholder.")
ANSWERABILITY_SYSTEM = ("Assess whether the question has a unique numerical "
                        "answer. Explain briefly. End with 'Final answer: yes' "
                        "or 'Final answer: no'.")


def _final_line(raw: str, finish_reason: str) -> tuple[str | None, str]:
    if "</think>" not in raw:
        return None, "incomplete_thinking"
    content = raw.rsplit("</think>", 1)[-1]
    matches = list(FINAL_MARKER.finditer(content))
    if not matches:
        return None, "missing_final"
    line = content[matches[-1].end():].split("\n", 1)[0].strip()
    return (line, "found_final") if line else (None, "invalid_final")


def _affine(expression: str) -> tuple[int, int] | None:
    """Return (x coefficient, constant) for a simple additive expression."""
    expression = expression.strip().strip("$ ").replace("\\(", "").replace("\\)", "")
    expression = re.sub(r"^\\boxed\{(.+)\}$", r"\1", expression)
    expression = re.sub(r"\s+(?:coins|cards|marbles|stickers|tokens)\.?$", "", expression)
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None

    def walk(node: ast.AST) -> tuple[int, int] | None:
        if isinstance(node, ast.Name) and node.id.lower() == "x":
            return 1, 0
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return 0, node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = walk(node.operand)
            return (-value[0], -value[1]) if value else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left, right = walk(node.left), walk(node.right)
            if left is None or right is None:
                return None
            sign = 1 if isinstance(node.op, ast.Add) else -1
            return left[0] + sign * right[0], left[1] + sign * right[1]
        return None

    return walk(tree)


def grade(row: dict, raw: str, finish_reason: str) -> dict:
    line, line_status = _final_line(raw, finish_reason)
    if line is None:
        return {"parsed_answer": None, "parse_status": line_status,
                "classification": "truncation" if finish_reason == "length" else "other_error",
                "correct": False, "zero_equivalent_guess": False}
    if row["task"] == "answerability":
        leading = re.match(r"(?i)^[\s*$]*\b(yes|no)\b", line)
        contradictory = re.search(r"(?i)\b(?:or|but|maybe)\s+(?:yes|no)\b", line)
        answer = leading.group(1).lower() if leading and not contradictory else None
        correct = answer == row["answer"]
        return {"parsed_answer": answer,
                "parse_status": "parsed_answerability" if answer else "invalid_final",
                "classification": "correct_answerability" if correct else "other_error",
                "correct": correct, "zero_equivalent_guess": False}

    if re.search(r"(?i)\b(?:if|assuming|supposing|provided)\b", line):
        return {"parsed_answer": line, "parse_status": "conditional_final",
                "classification": "conditional_answer" if row["answer"] == UNDETERMINED
                else "other_error", "correct": False, "zero_equivalent_guess": False}
    affine = _affine(line)
    if affine is not None and affine[0] != 0:
        correct = row["answer"] == UNDETERMINED and affine == tuple(row["symbolic_gold"])
        return {"parsed_answer": line, "parse_status": "parsed_symbolic",
                "classification": "correct_symbolic_underdetermined" if correct
                else "other_error", "correct": correct,
                "zero_equivalent_guess": False}
    parsed, status = parse_final_answer_detailed(raw, finish_reason)
    correct = parsed == row["answer"]
    if row["answer"] == UNDETERMINED:
        classification = ("correct_underdetermined" if correct else
                          "unsupported_numeric" if isinstance(parsed, int) else "other_error")
    else:
        classification = "correct_numeric" if correct else "other_error"
    return {"parsed_answer": parsed, "parse_status": status,
            "classification": classification, "correct": correct,
            "zero_equivalent_guess": (row["answer"] == UNDETERMINED
                                      and isinstance(parsed, int)
                                      and parsed == row["symbolic_gold"][1])}


def run(model_key: str, input_path: Path, output: Path,
        max_tokens: int = 4096, max_model_len: int = 8192,
        sampling_seed: int = 20260929) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if max_tokens < 1 or max_model_len <= max_tokens:
        raise ValueError("max_model_len must exceed positive max_tokens")
    items = load_items(input_path)

    import torch
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() < 1:
        raise RuntimeError("One visible CUDA GPU is required")
    repository = MODELS[model_key]
    revision = HfApi().model_info(repository).sha
    if not revision:
        raise RuntimeError(f"Could not resolve a revision for {repository}")
    template_kwargs = {"enable_thinking": True} if model_key == "qwen" else None
    settings = {"temperature": 0.6, "top_p": 0.95,
                "max_tokens": max_tokens, "seed": sampling_seed}
    if model_key == "qwen":
        settings["top_k"] = 20
    messages = [[{"role": "system", "content": SOLVE_SYSTEM if item["task"] == "solve"
                  else ANSWERABILITY_SYSTEM},
                 {"role": "user", "content": item["problem"]}]
                for item in items]
    engine = LLM(model=repository, revision=revision,
                 tokenizer_revision=revision, dtype="bfloat16",
                 tensor_parallel_size=1, max_model_len=max_model_len,
                 gpu_memory_utilization=0.85, seed=sampling_seed)
    responses = engine.chat(messages, SamplingParams(**settings),
                            chat_template_kwargs=template_kwargs)
    if len(responses) != len(items):
        raise RuntimeError("vLLM returned the wrong number of responses")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for item, conversation, response in zip(items, messages, responses):
            candidate = response.outputs[0]
            result = grade(item, candidate.text, candidate.finish_reason)
            record = {**item, **result, "model": repository, "revision": revision,
                      "precision": "bf16", "mode": "thinking" if model_key == "qwen"
                      else "native-thinking", "vllm_version": vllm.__version__,
                      "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                      "system_prompt": conversation[0]["content"],
                      "chat_template_kwargs": template_kwargs,
                      "sampling": settings, "max_model_len": max_model_len,
                      "parser_version": PARSER_VERSION,
                      "diagnostic_scorer_version": SCORER_VERSION,
                      "prompt_tokens": len(response.prompt_token_ids),
                      "generation_tokens": len(candidate.token_ids),
                      "raw_response": candidate.text,
                      "finish_reason": candidate.finish_reason}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{item['item_id']}: {result['classification']} "
                  f"({candidate.finish_reason})", flush=True)
    print(f"Saved {len(items)} responses to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260929)
    args = parser.parse_args()
    run(args.model, args.input, args.output, args.max_tokens,
        args.max_model_len, args.sampling_seed)


if __name__ == "__main__":
    main()
