"""Run one model and mode on one Phase 1 family file, with resumable JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from study.shortcut_rep_items import load_family

MANIFEST = Path(__file__).with_name("shortcut_rep_models.json")
SEED = 20261001
# Output and context limits per mode; non-thinking answers are short.
LIMITS = {"thinking": (16384, 20480), "non_thinking": (4096, 8192)}


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def conversation(item: dict, model: dict) -> list[dict]:
    if model.get("system_in_user"):
        # DeepSeek-R1 distills are documented to work best without a system prompt.
        return [{"role": "user", "content": item["system"] + "\n\n" + item["user"]}]
    return [{"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]}]


def _completed(output_path: Path, expected: set, checks: dict) -> set:
    completed = set()
    if not output_path.exists():
        return completed
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["item_id"], row["sample_index"])
        if key not in expected or key in completed or any(
                row.get(name) != value for name, value in checks.items()):
            raise ValueError("Existing output is incompatible or has duplicate records")
        completed.add(key)
    return completed


def run(model_key: str, mode: str, items_path: Path, output_path: Path,
        manifest_path: Path = MANIFEST, batch_size: int = 32, max_tokens: int | None = None,
        max_model_len: int | None = None, tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85) -> None:
    model = load_manifest(manifest_path)[model_key]
    if mode not in model["modes"]:
        raise ValueError(f"{model_key} has no {mode} mode")
    if model.get("sampling_verified") is False:
        raise ValueError(f"Verify {model_key}'s generation_config.json sampling, then set "
                         "sampling_verified to true in the manifest")
    max_tokens = max_tokens or LIMITS[mode][0]
    max_model_len = max_model_len or LIMITS[mode][1]
    if (max_model_len <= max_tokens or batch_size < 1 or tensor_parallel_size < 1
            or not 0 < gpu_memory_utilization <= 1):
        raise ValueError("Invalid generation configuration")
    items = load_family(items_path)
    item_hash = hashlib.sha256(items_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    settings = {"n": model["samples"], **model["modes"][mode]["sampling"],
                "max_tokens": max_tokens, "seed": SEED}
    kwargs = model["modes"][mode]["chat_template_kwargs"]
    expected = {(item["item_id"], n) for item in items for n in range(model["samples"])}
    completed = _completed(output_path, expected, {
        "model_key": model_key, "mode": mode, "revision": model["revision"],
        "benchmark_sha256": item_hash, "model_manifest_sha256": manifest_hash,
        "max_model_len": max_model_len, "tensor_parallel_size": tensor_parallel_size,
        "sampling": settings})
    pending = [item for item in items if any((item["item_id"], n) not in completed
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
    engine = LLM(model=repository, revision=revision, tokenizer_revision=revision,
                 dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
                 max_model_len=max_model_len,
                 gpu_memory_utilization=gpu_memory_utilization, seed=SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            messages = [conversation(item, model) for item in batch]
            responses = engine.chat(messages, SamplingParams(**settings),
                                    chat_template_kwargs=kwargs)
            if len(responses) != len(batch):
                raise RuntimeError("vLLM response count mismatch")
            for item, message, response in zip(batch, messages, responses):
                if len(response.outputs) != model["samples"]:
                    raise RuntimeError("vLLM sample count mismatch")
                for sample_index, candidate in enumerate(response.outputs):
                    if (item["item_id"], sample_index) in completed:
                        continue
                    row = {"item_id": item["item_id"], "base_id": item["base_id"],
                           "family": item["family"], "cell": item["cell"],
                           "sample_index": sample_index, "model_key": model_key, "mode": mode,
                           "model": repository, "revision": revision,
                           "benchmark_sha256": item_hash,
                           "model_manifest_sha256": manifest_hash, "precision": "bf16",
                           "vllm_version": vllm.__version__,
                           "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                           "sampling": settings, "max_tokens": max_tokens,
                           "max_model_len": max_model_len,
                           "tensor_parallel_size": tensor_parallel_size,
                           "gpu_memory_utilization": gpu_memory_utilization,
                           "batch_size": batch_size, "chat_template_kwargs": kwargs,
                           "messages": message,
                           "prompt_tokens": len(response.prompt_token_ids),
                           "generation_tokens": len(candidate.token_ids),
                           "raw_response": candidate.text,
                           "finish_reason": candidate.finish_reason}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed.add((item["item_id"], sample_index))
            handle.flush()
            print(f"Saved {len(completed)}/{len(expected)} responses", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=load_manifest())
    parser.add_argument("--mode", required=True, choices=LIMITS)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    run(args.model, args.mode, args.items, args.output, args.manifest, args.batch_size,
        args.max_tokens, args.max_model_len, args.tensor_parallel_size,
        args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
