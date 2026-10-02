"""Run pinned Case 5 confirmatory cells with fresh contexts and resumable JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from study.generate_case5_confirmatory import COUNT, validate_items
from study.run_case5_assumption_pilot import INSTRUCTION

SYSTEM_SOLVE = ("Solve the problem. End with one line: Final answer: followed by "
                "a whole number, an expression in x for an unknown amount, or "
                "cannot be determined.")
SYSTEM_CHECK = ("Does the question have a unique numerical answer? Explain briefly. "
                "End with one line: Final answer: yes or Final answer: no.")
TASKS = ("solve", "answerability")
PLACEMENTS = ("none", "user")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_items(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    validate_items(rows, COUNT)
    return rows


def conversation(item: dict, task: str, placement: str) -> list[dict]:
    if task not in TASKS or placement not in PLACEMENTS:
        raise ValueError("Unknown cell")
    system = SYSTEM_SOLVE if task == "solve" else SYSTEM_CHECK
    question = item["problem"]
    if task == "answerability":
        question = question[:question.rfind(" How many ")].rstrip(" .") + ". " + (
            f"Does the information uniquely determine how many {item['noun']} "
            f"{item['name']} has now?") if item["kind"] != "net_change" else (
            question[:question.rfind(" By how much ")].rstrip(" .") + ". " +
            f"Does the information uniquely determine how much {item['name']}'s "
            f"number of {item['noun']} changed?")
    if placement == "user":
        question += "\n\n" + INSTRUCTION
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def run(model_key: str, items_path: Path, manifest_path: Path, output_path: Path,
        batch_size: int = 32, max_tokens: int = 16384,
        max_model_len: int = 20480, tensor_parallel_size: int = 1,
        seed: int = 20261001, gpu_memory_utilization: float = 0.85) -> None:
    if (max_model_len <= max_tokens or batch_size < 1 or tensor_parallel_size < 1
            or not 0 < gpu_memory_utilization <= 1):
        raise ValueError("Invalid generation configuration")
    models = load_manifest(manifest_path)
    model = models[model_key]
    items = load_items(items_path)
    item_hash = hashlib.sha256(items_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected = {(item["item_id"], task, placement, n)
                for item in items for task in TASKS for placement in PLACEMENTS
                for n in range(model["samples"])}
    completed = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["item_id"], row["task"], row["instruction_placement"], row["sample_index"])
            if (key not in expected or key in completed or row["revision"] != model["revision"]
                    or row["benchmark_sha256"] != item_hash or row["max_tokens"] != max_tokens
                    or row["model_manifest_sha256"] != manifest_hash
                    or row["max_model_len"] != max_model_len
                    or row["tensor_parallel_size"] != tensor_parallel_size
                    or row["sampling"]["seed"] != seed):
                raise ValueError("Existing output is incompatible or has duplicate records")
            completed.add(key)
    pending = [(item, task, placement) for item in items
               for task in TASKS for placement in PLACEMENTS
               if any((item["item_id"], task, placement, n) not in completed
                      for n in range(model["samples"]))]
    if not pending:
        print(f"Already complete: {len(completed)} responses")
        return
    # A partial group is rerun in full. Existing sample indices remain fixed.
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() < tensor_parallel_size:
        raise RuntimeError(f"Need {tensor_parallel_size} visible CUDA GPUs")
    repository, revision = model["repository"], model["revision"]
    kwargs = {"enable_thinking": True} if model_key.startswith("qwen") else None
    engine = LLM(model=repository, revision=revision, tokenizer_revision=revision,
                 dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
                 max_model_len=max_model_len,
                 gpu_memory_utilization=gpu_memory_utilization, seed=seed)
    settings = {"n": model["samples"], "temperature": 0.6, "top_p": 0.95,
                "max_tokens": max_tokens, "seed": seed}
    if model_key.startswith("qwen"):
        settings["top_k"] = 20
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            messages = [conversation(*cell) for cell in batch]
            responses = engine.chat(messages, SamplingParams(**settings),
                                    chat_template_kwargs=kwargs)
            if len(responses) != len(batch):
                raise RuntimeError("vLLM response count mismatch")
            for (item, task, placement), message, response in zip(batch, messages, responses):
                if len(response.outputs) != model["samples"]:
                    raise RuntimeError("vLLM sample count mismatch")
                for sample_index, candidate in enumerate(response.outputs):
                    key = (item["item_id"], task, placement, sample_index)
                    if key in completed:
                        continue
                    row = {"item_id": item["item_id"], "base_id": item["base_id"],
                           "task": task, "instruction_placement": placement,
                           "sample_index": sample_index, "model_key": model_key,
                           "model": repository, "revision": revision,
                           "benchmark_sha256": item_hash,
                           "model_manifest_sha256": manifest_hash, "precision": "bf16",
                           "mode": "thinking", "vllm_version": vllm.__version__,
                           "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                           "sampling": settings, "max_tokens": max_tokens,
                           "max_model_len": max_model_len,
                           "tensor_parallel_size": tensor_parallel_size,
                           "chat_template_kwargs": kwargs, "system_prompt": message[0]["content"],
                           "user_prompt": message[1]["content"],
                           "prompt_tokens": len(response.prompt_token_ids),
                           "generation_tokens": len(candidate.token_ids),
                           "raw_response": candidate.text,
                           "finish_reason": candidate.finish_reason}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed.add(key)
            handle.flush()
            print(f"Saved {len(completed)}/{len(expected)} responses", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=load_manifest(Path(__file__).with_name("case5_confirmatory_models.json")))
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("case5_confirmatory_models.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    run(args.model, args.items, args.manifest, args.output, args.batch_size,
        args.max_tokens, args.max_model_len, args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
