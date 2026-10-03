"""Measure zero versus received-number continuation log probabilities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from study.run_case5_confirmatory import load_manifest
from study.run_case5_confirmatory import SYSTEM_SOLVE


def score_continuation(engine, prefix_ids: list[int], candidate_ids: list[int], sampling_params) -> float:
    response = engine.generate([{"prompt_token_ids": prefix_ids + candidate_ids}],
                               sampling_params, use_tqdm=False)[0]
    logprobs = response.prompt_logprobs
    if logprobs is None or len(logprobs) != len(prefix_ids) + len(candidate_ids):
        raise ValueError("Missing prompt log probabilities for probe continuation")
    return sum(logprobs[position][token_id].logprob
               for position, token_id in enumerate(candidate_ids, start=len(prefix_ids)))


def run(model_key: str, probes_path: Path, manifest_path: Path,
        output_path: Path, tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    probes = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines() if line]
    model = load_manifest(manifest_path)[model_key]
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() < tensor_parallel_size:
        raise RuntimeError(f"Need {tensor_parallel_size} visible CUDA GPUs")
    engine = LLM(model=model["repository"], revision=model["revision"],
                 tokenizer_revision=model["revision"], dtype="bfloat16",
                 tensor_parallel_size=tensor_parallel_size, max_model_len=4096,
                 gpu_memory_utilization=gpu_memory_utilization, seed=20261001)
    tokenizer = engine.get_tokenizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for probe in probes:
            # A fixed assistant thought prefix gives comparable continuation
            # measurements in the models' native thinking chat templates.
            messages = [{"role": "system", "content": SYSTEM_SOLVE},
                        {"role": "user", "content": probe["problem"]}]
            kwargs = {"enable_thinking": True} if model_key.startswith("qwen") else {}
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True, **kwargs)
            thought_start = "\n" if rendered.rstrip().endswith("<think>") else "<think>\n"
            prefix = rendered + thought_start + probe["prefix"]
            zero_ids = tokenizer.encode(probe["zero_continuation"], add_special_tokens=False)
            recv_ids = tokenizer.encode(probe["received_continuation"], add_special_tokens=False)
            if not zero_ids or not recv_ids:
                raise ValueError(f"Empty probe candidate: {probe['probe_id']}")
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            params = SamplingParams(max_tokens=1, temperature=0,
                                    prompt_logprobs=0)
            zero = score_continuation(engine, prefix_ids, zero_ids, params)
            received = score_continuation(engine, prefix_ids, recv_ids, params)
            if not math.isfinite(zero) or not math.isfinite(received):
                raise ValueError("Non-finite probe log probability")
            row = {**probe, "model_key": model_key, "model": model["repository"],
                   "revision": model["revision"], "vllm_version": vllm.__version__,
                   "torch_version": torch.__version__, "prompt": prefix,
                   "chat_template_kwargs": kwargs,
                   "zero_token_ids": zero_ids, "received_token_ids": recv_ids,
                   "logprob_zero": zero, "logprob_received": received,
                   "log_odds_zero_vs_received": zero - received,
                   "candidate_mass_zero": math.exp(zero),
                   "candidate_mass_received": math.exp(received)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Saved {len(probes)} probes to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=load_manifest(Path(__file__).with_name("case5_confirmatory_models.json")))
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("case5_confirmatory_models.json"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    run(args.model, args.probes, args.manifest, args.output,
        args.tensor_parallel_size, args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
