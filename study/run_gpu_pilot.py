"""Exploratory bf16 A100 pilot on the same five bases as the local MLX pilot.

Run one model and mode at a time. This pilot is excluded from the final study.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from study.generate import generate
from study.run_local_pilot import MAX_STEPS, N_BASE, SEED, SYSTEM, extract_answer

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b": "Qwen/Qwen3-4B",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen14b": "Qwen/Qwen3-14B",
    "qwen32b": "Qwen/Qwen3-32B",
    "olmo": "allenai/Olmo-3-7B-Think",
}


def pilot_items() -> list[dict]:
    items = [item for item in generate(N_BASE, SEED)
             if item["n_steps"] <= MAX_STEPS]
    random.Random(SEED + 1).shuffle(items)
    if len(items) != 20 or len({item["base_id"] for item in items}) != 5:
        raise ValueError("Pilot selection changed; check the generator")
    return items


def sampling_settings(model_key: str, mode: str) -> dict:
    if model_key.startswith("qwen"):
        if mode == "thinking":
            return {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                    "max_tokens": 2048, "seed": SEED}
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "max_tokens": 2048, "seed": SEED}
    if mode != "native-thinking":
        raise ValueError("OLMo Think only has its native thinking mode")
    return {"temperature": 0.6, "top_p": 0.95,
            "max_tokens": 2048, "seed": SEED}


def run(model_key: str, mode: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if model_key.startswith("qwen") and mode not in ("thinking", "non-thinking"):
        raise ValueError("Qwen mode must be thinking or non-thinking")
    if model_key not in ("qwen", "olmo") and mode != "thinking":
        raise ValueError("The remaining Qwen3 size pilots require thinking mode")
    settings = sampling_settings(model_key, mode)

    import torch
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams

    repository = MODELS[model_key]
    revision = HfApi().model_info(repository).sha
    if not revision:
        raise RuntimeError(f"Could not resolve a revision for {repository}")
    items = pilot_items()
    messages = [
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": item["problem"]}]
        for item in items
    ]
    template_kwargs = {"enable_thinking": mode == "thinking"} if model_key.startswith("qwen") else None
    tensor_parallel_size = 2 if model_key == "qwen32b" else 1
    gpu_memory_utilization = 0.95 if model_key == "qwen32b" else 0.85
    if torch.cuda.device_count() < tensor_parallel_size:
        raise RuntimeError(f"{repository} requires {tensor_parallel_size} visible GPU(s)")
    engine = LLM(model=repository, revision=revision,
                 tokenizer_revision=revision, dtype="bfloat16",
                 tensor_parallel_size=tensor_parallel_size, max_model_len=4096,
                 gpu_memory_utilization=gpu_memory_utilization, seed=SEED)
    responses = engine.chat(messages, SamplingParams(**settings),
                            chat_template_kwargs=template_kwargs)
    if len(responses) != len(items):
        raise RuntimeError("vLLM returned the wrong number of responses")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for item, response in zip(items, responses):
            candidate = response.outputs[0]
            raw = candidate.text
            parsed = extract_answer(raw)
            row = {
                "item_id": item["item_id"], "base_id": item["base_id"],
                "form": item["form"], "reversal": item["reversal"],
                "operation": item["operation"], "keyword": item["keyword"],
                "problem": item["problem"], "answer": item["answer"],
                "model": repository, "revision": revision,
                "precision": "bf16", "mode": mode,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "vllm_version": vllm.__version__,
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "system_prompt": SYSTEM,
                "chat_template_kwargs": template_kwargs,
                "sampling": settings,
                "prompt_tokens": len(response.prompt_token_ids),
                "generation_tokens": len(candidate.token_ids),
                "raw_response": raw, "parsed_answer": parsed,
                "correct": parsed == item["answer"],
                "finish_reason": candidate.finish_reason,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{item['item_id']}: {parsed} / {item['answer']} "
                  f"({candidate.finish_reason})", flush=True)
    print(f"Saved {len(items)} responses to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--mode", required=True,
                        choices=("thinking", "non-thinking", "native-thinking"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.mode, args.output)


if __name__ == "__main__":
    main()
