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
from study.run_local_pilot import extract_answer

MODELS = {"qwen": "Qwen/Qwen3-0.6B", "olmo": "allenai/Olmo-3-7B-Think"}
SYSTEM = ("Solve the problem carefully. If a necessary quantity is missing, "
          "do not guess it. End with exactly one line: Final answer: <integer> "
          "or Final answer: cannot be determined.")
FINAL_LINE = re.compile(r"(?im)^\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*[:：]\s*(.+?)\s*$")
UNDETERMINED_PHRASE = re.compile(
    r"(?i)\b(?:cannot|can't|can not)\s+be\s+determined\b|"
    r"\b(?:not enough|insufficient)\s+information\b|\bunderdetermined\b"
)


def parse_final_answer(raw: str) -> int | str | None:
    matches = FINAL_LINE.findall(raw)
    if not matches:
        return None
    final = matches[-1].replace("**", "").strip()
    if UNDETERMINED_PHRASE.search(final):
        return UNDETERMINED
    return extract_answer("Final answer: " + final)


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
            parsed = parse_final_answer(candidate.text)
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
