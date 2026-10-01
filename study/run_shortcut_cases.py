"""Run matched shortcut cases 2–6 with thinking always enabled.

The Qwen hybrid checkpoint uses its thinking chat template; OLMo Think uses
its native reasoning template. Run one model and input file per process.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from study.generate_shortcut_cases import CASES, UNDETERMINED, validate_pair
MODELS = {"qwen": "Qwen/Qwen3-0.6B", "olmo": "allenai/Olmo-3-7B-Think"}
PARSER_VERSION = 2
SYSTEM = ("Solve the problem carefully. If a necessary quantity is missing, "
          "do not guess it. End with one line beginning 'Final answer:' followed "
          "by either a whole number or the exact words 'cannot be determined'. "
          "Write the actual answer, never a placeholder.")
FINAL_MARKER = re.compile(r"(?i)\bfinal\s+answer(?:\*\*)?\s*[:：][ \t]*")
UNDETERMINED_PHRASE = re.compile(
    r"(?i)\b(?:cannot|can't|can not)\s+be\s+determined\b|"
    r"\b(?:not enough|insufficient)\s+information\b|\bunderdetermined\b"
)
INTEGER = re.compile(r"(?<![\w.])[-+]?\d+(?!\w|\.\d)")


def parse_final_answer_detailed(raw: str, finish_reason: str | None = None
                                ) -> tuple[int | str | None, str]:
    if "</think>" not in raw and ("<think>" in raw or finish_reason is not None):
        return None, "incomplete_thinking"
    final_content = raw.rsplit("</think>", 1)[-1]
    matches = list(FINAL_MARKER.finditer(final_content))
    if not matches:
        return None, "missing_final"
    final = final_content[matches[-1].end():].split("\n", 1)[0].replace("**", "").strip()
    final = re.sub(r"(?i)^<integer>\s*", "", final)
    numbers = INTEGER.findall(final)
    abstains = bool(UNDETERMINED_PHRASE.search(final))
    if abstains and numbers or len(numbers) > 1:
        return None, "ambiguous_final"
    if abstains:
        return UNDETERMINED, "parsed_abstention"
    if len(numbers) == 1:
        return int(numbers[0]), "parsed_numeric"
    return None, "invalid_final"


def parse_final_answer(raw: str, finish_reason: str | None = None) -> int | str | None:
    return parse_final_answer_detailed(raw, finish_reason)[0]


def load_items(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or len(rows) % 2:
        raise ValueError("Input must contain complete matched pairs")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate item IDs")
    if len({row["case"] for row in rows}) != 1 or rows[0]["case"] not in CASES:
        raise ValueError("Input must contain exactly one supported case")
    for index in range(0, len(rows), 2):
        validate_pair(rows[index:index + 2])
    return rows


def run(model_key: str, input_path: Path, output: Path,
        max_tokens: int = 2048, max_model_len: int = 4096,
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
    messages = [[{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": item["problem"]}]
                for item in items]
    engine = LLM(model=repository, revision=revision,
                 tokenizer_revision=revision, dtype="bfloat16",
                 tensor_parallel_size=1, max_model_len=max_model_len,
                 gpu_memory_utilization=0.85, seed=settings["seed"])
    responses = engine.chat(messages, SamplingParams(**settings),
                            chat_template_kwargs=template_kwargs)
    if len(responses) != len(items):
        raise RuntimeError("vLLM returned the wrong number of responses")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for item, response in zip(items, responses):
            candidate = response.outputs[0]
            parsed, parse_status = parse_final_answer_detailed(
                candidate.text, candidate.finish_reason)
            row = {**item, "model": repository, "revision": revision,
                   "precision": "bf16",
                   "mode": "thinking" if model_key == "qwen" else "native-thinking",
                   "vllm_version": vllm.__version__,
                   "torch_version": torch.__version__,
                   "torch_cuda": torch.version.cuda,
                   "system_prompt": SYSTEM,
                   "chat_template_kwargs": template_kwargs,
                   "sampling": settings, "max_model_len": max_model_len,
                   "prompt_tokens": len(response.prompt_token_ids),
                   "generation_tokens": len(candidate.token_ids),
                   "raw_response": candidate.text, "parsed_answer": parsed,
                   "parse_status": parse_status, "parser_version": PARSER_VERSION,
                   "correct": parsed == item["answer"],
                   "finish_reason": candidate.finish_reason}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{item['item_id']}: {parsed} / {item['answer']} "
                  f"({candidate.finish_reason})", flush=True)
    print(f"Saved {len(items)} responses to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--sampling-seed", type=int, default=20260929)
    args = parser.parse_args()
    run(args.model, args.input, args.output, args.max_tokens,
        args.max_model_len, args.sampling_seed)


if __name__ == "__main__":
    main()
