"""Run matched direct and relational starting-state Case 5 problems."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from study.generate_case5_confirmatory import COUNT, DOMAINS, NAMES, generate as generate_confirmatory
from study.run_case5_confirmatory import SYSTEM_SOLVE, load_manifest
from study.score_case5_confirmatory import score as score_answer

CONDITIONS = ("direct", "same_as", "more_than_given", "half_as_many")
MANIFEST = Path(__file__).with_name("case5_confirmatory_models.json")
SCORER_VERSION = "case5-relational-v1"


def generate() -> list[dict]:
    confirmatory, _ = generate_confirmatory()
    controls = [item for item in confirmatory if item["kind"] == "complete"]
    items = []
    for control in controls:
        name, noun, start = control["name"], control["noun"], control["start"]
        other = NAMES[(control["base_id"] + 1) % len(NAMES)]
        give_verb = DOMAINS[control["base_id"] % len(DOMAINS)][2]
        actions = control["problem"].split(". ", 1)[1]
        openings = {
            "direct": f"{name} starts with {start} {noun}.",
            "same_as": (f"{name} starts with as many {noun} as {other}. "
                        f"{other} starts with {start} {noun}."),
            "more_than_given": (f"{name} starts with {start - control['given']} more {noun} "
                                f"than {name} later {give_verb}."),
            "half_as_many": (f"{name} starts with half as many {noun} as {other}. "
                             f"{other} starts with {2 * start} {noun}."),
        }
        for condition in CONDITIONS:
            items.append({
                "item_id": f"c5rel-{control['base_id']:02d}-{condition}",
                "base_id": control["base_id"], "condition": condition,
                "parent_item_id": control["item_id"],
                "problem": openings[condition] + " " + actions,
                "gold": control["gold"], "zero_default": control["received"] - control["given"],
                "start": start, "received": control["received"], "given": control["given"],
            })
    if len(items) != COUNT * len(CONDITIONS) or len({x["item_id"] for x in items}) != len(items):
        raise ValueError("Incomplete relational item set")
    for base_id in range(COUNT):
        block = [x for x in items if x["base_id"] == base_id]
        if {x["condition"] for x in block} != set(CONDITIONS):
            raise ValueError(f"Missing relational condition for base {base_id}")
        if len({x["gold"] for x in block}) != 1 or len({x["zero_default"] for x in block}) != 1:
            raise ValueError(f"Unmatched relational base {base_id}")
        if block[0]["gold"] == block[0]["zero_default"]:
            raise ValueError(f"Gold equals zero-default for base {base_id}")
    return items


def load_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if items != generate():
        raise ValueError("Relational items differ from the generated matched set")
    return items


def run(model_key: str, items_path: Path, output_path: Path, manifest_path: Path = MANIFEST,
        batch_size: int = 32, max_tokens: int = 16384, max_model_len: int = 20480,
        tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85) -> None:
    if batch_size < 1 or max_model_len <= max_tokens or tensor_parallel_size < 1 or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("Invalid generation configuration")
    model = load_manifest(manifest_path)[model_key]
    items = load_items(items_path)
    item_hash = hashlib.sha256(items_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    seed = 20261001
    expected = {(item["item_id"], sample) for item in items for sample in range(model["samples"])}
    completed = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["item_id"], row["sample_index"])
            if (key not in expected or key in completed or row["model_key"] != model_key
                    or row["revision"] != model["revision"] or row["benchmark_sha256"] != item_hash
                    or row["model_manifest_sha256"] != manifest_hash or row["max_tokens"] != max_tokens
                    or row["max_model_len"] != max_model_len or row["tensor_parallel_size"] != tensor_parallel_size
                    or row["sampling"]["seed"] != seed):
                raise ValueError("Existing output is incompatible or has duplicate records")
            completed.add(key)
    pending = [item for item in items if any((item["item_id"], sample) not in completed
                                                for sample in range(model["samples"]))]
    if not pending:
        print(f"Already complete: {len(completed)} responses")
        return

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() < tensor_parallel_size:
        raise RuntimeError(f"Need {tensor_parallel_size} visible CUDA GPUs")
    revision = model["revision"]
    engine = LLM(model=model["repository"], revision=revision, tokenizer_revision=revision,
                 dtype="bfloat16", tensor_parallel_size=tensor_parallel_size,
                 max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                 seed=seed)
    settings = {"n": model["samples"], "temperature": 0.6, "top_p": 0.95,
                "max_tokens": max_tokens, "seed": seed}
    if model_key.startswith("qwen"):
        settings["top_k"] = 20
    chat_kwargs = {"enable_thinking": True} if model_key.startswith("qwen") else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            messages = [[{"role": "system", "content": SYSTEM_SOLVE},
                         {"role": "user", "content": item["problem"]}] for item in batch]
            responses = engine.chat(messages, SamplingParams(**settings),
                                    chat_template_kwargs=chat_kwargs)
            if len(responses) != len(batch):
                raise RuntimeError("vLLM response count mismatch")
            for item, response in zip(batch, responses):
                if len(response.outputs) != model["samples"]:
                    raise RuntimeError("vLLM sample count mismatch")
                for sample_index, candidate in enumerate(response.outputs):
                    key = (item["item_id"], sample_index)
                    if key in completed:
                        continue
                    row = {
                        "item_id": item["item_id"], "base_id": item["base_id"],
                        "condition": item["condition"], "sample_index": sample_index,
                        "model_key": model_key, "model": model["repository"],
                        "revision": revision, "benchmark_sha256": item_hash,
                        "model_manifest_sha256": manifest_hash, "precision": "bf16",
                        "mode": "thinking", "vllm_version": vllm.__version__,
                        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                        "sampling": settings, "max_tokens": max_tokens,
                        "max_model_len": max_model_len, "tensor_parallel_size": tensor_parallel_size,
                        "chat_template_kwargs": chat_kwargs,
                        "system_prompt": SYSTEM_SOLVE, "user_prompt": item["problem"],
                        "prompt_tokens": len(response.prompt_token_ids),
                        "generation_tokens": len(candidate.token_ids),
                        "raw_response": candidate.text, "finish_reason": candidate.finish_reason,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed.add(key)
            handle.flush()
            print(f"Saved {len(completed)}/{len(expected)} responses", flush=True)


def score_row(item: dict, row: dict) -> dict:
    result = score_answer(item, "solve", row["raw_response"], row["finish_reason"])
    if type(result["parsed"]) is int and result["parsed"] == item["zero_default"] and not result["correct"]:
        result["outcome"] = "zero_default"
    return {**row, "scorer_version": SCORER_VERSION, **result}


def score_file(items_path: Path, raw_path: Path, output_path: Path,
               manifest_path: Path = MANIFEST) -> dict[str, Counter]:
    if output_path.exists():
        raise FileExistsError(output_path)
    items = {item["item_id"]: item for item in load_items(items_path)}
    model_manifest = load_manifest(manifest_path)
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    if not raw:
        raise ValueError("Empty raw result file")
    model_key = raw[0]["model_key"]
    model = model_manifest[model_key]
    expected = {(item_id, sample) for item_id in items for sample in range(model["samples"])}
    seen = {(row["item_id"], row["sample_index"]) for row in raw}
    item_hash = hashlib.sha256(items_path.read_bytes()).hexdigest()
    if (len(raw) != len(expected) or seen != expected or any(
            row["model_key"] != model_key or row["revision"] != model["revision"]
            or row["benchmark_sha256"] != item_hash for row in raw)):
        raise ValueError("Incomplete, unexpected, or mixed raw results")
    counts: dict[str, Counter] = defaultdict(Counter)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in raw:
            scored = score_row(items[row["item_id"]], row)
            counts[row["condition"]][scored["outcome"]] += 1
            handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate")
    create.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("run")
    infer.add_argument("model", choices=load_manifest(MANIFEST))
    infer.add_argument("--items", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--manifest", type=Path, default=MANIFEST)
    infer.add_argument("--batch-size", type=int, default=32)
    infer.add_argument("--max-tokens", type=int, default=16384)
    infer.add_argument("--max-model-len", type=int, default=20480)
    infer.add_argument("--tensor-parallel-size", type=int, default=1)
    infer.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    grade = commands.add_parser("score")
    grade.add_argument("--items", type=Path, required=True)
    grade.add_argument("--raw", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)
    grade.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.command == "generate":
        items = generate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items)} relational items to {args.output}")
    elif args.command == "run":
        run(args.model, args.items, args.output, args.manifest, args.batch_size,
            args.max_tokens, args.max_model_len, args.tensor_parallel_size,
            args.gpu_memory_utilization)
    else:
        for condition, count in score_file(args.items, args.raw, args.output, args.manifest).items():
            total = sum(count.values())
            print(f"{condition}: correct={count['correct']}/{total}, "
                  f"zero_default={count['zero_default']}/{total}, "
                  f"other_error={count['other_error']}, "
                  f"unparseable={count['unparseable']}, truncated={count['truncated']}")


if __name__ == "__main__":
    main()
