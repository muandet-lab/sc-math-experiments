"""Run the unjustified-certainty pilot on one pinned model and mode with vLLM.

Writes one JSONL file with two row types:
  stage='answer'  a task response (T0-T4), one per problem x task x sample
  stage='elicit'  an elicited profile, for arm 'direct' (problem x format x sample) or
                  'two_stage' (after each stage-1 answer, per format)
Invalid profiles are re-requested up to MAX_ATTEMPTS times with a new seed (Yang et al. loop
until valid; we cap it). Every attempt is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from study.shortcut_rep_run import LIMITS, SEED, load_manifest, MANIFEST
from study.uc_elicit import (ARMS, FORMATS, PARSER_VERSION, TASKS, classify_answer,
                             direct_messages, parse_profile, task_messages, two_stage_messages,
                             visible)
from study.uc_items import validate

MAX_ATTEMPTS = 3


def load_problems(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    validate(rows)
    return rows


def plan(problems: list[dict], tasks: list[str], arms: list[str], formats: list[str]) -> dict:
    if any(t not in TASKS for t in tasks) or any(a not in ARMS for a in arms) or any(
            f not in FORMATS for f in formats):
        raise ValueError("Unknown task, arm or format")
    if "two_stage" in arms and "T4" in tasks:
        raise ValueError("Two-stage elicitation after the answerability task is not defined")
    return {"answers": [(p, t) for p in problems for t in tasks],
            "direct": [(p, f) for p in problems for f in formats] if "direct" in arms else [],
            "two_stage": "two_stage" in arms, "formats": formats}


def run(model_key: str, mode: str, problems_path: Path, output_path: Path, tasks: list[str],
        arms: list[str], formats: list[str], batch_size: int = 64,
        tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85,
        manifest_path: Path = MANIFEST) -> None:
    model = load_manifest(manifest_path)[model_key]
    if mode not in model["modes"]:
        raise ValueError(f"{model_key} has no {mode} mode")
    if model.get("sampling_verified") is False:
        raise ValueError(f"Verify {model_key}'s sampling before running it")
    if output_path.exists():
        raise FileExistsError(output_path)
    problems = load_problems(problems_path)
    work = plan(problems, tasks, arms, formats)
    max_tokens, max_model_len = LIMITS[mode]
    samples = model["samples"]
    sampling = model["modes"][mode]["sampling"]
    kwargs = model["modes"][mode]["chat_template_kwargs"]
    system_in_user = bool(model.get("system_in_user"))
    meta = {"model_key": model_key, "mode": mode, "model": model["repository"],
            "revision": model["revision"],
            "problems_sha256": hashlib.sha256(problems_path.read_bytes()).hexdigest(),
            "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "parser_version": PARSER_VERSION, "max_tokens": max_tokens,
            "max_model_len": max_model_len, "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization, "batch_size": batch_size}

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    meta.update({"vllm_version": vllm.__version__, "torch_version": torch.__version__})
    engine = LLM(model=model["repository"], revision=model["revision"],
                 tokenizer_revision=model["revision"], dtype="bfloat16",
                 tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
                 gpu_memory_utilization=gpu_memory_utilization, seed=SEED)

    def generate(conversations: list[list[dict]], n: int, seed: int) -> list:
        outputs = []
        params = SamplingParams(n=n, seed=seed, max_tokens=max_tokens, **sampling)
        for offset in range(0, len(conversations), batch_size):
            batch = conversations[offset:offset + batch_size]
            responses = engine.chat(batch, params, chat_template_kwargs=kwargs)
            if len(responses) != len(batch) or any(len(r.outputs) != n for r in responses):
                raise RuntimeError("vLLM response count mismatch")
            outputs.extend(responses)
            print(f"  generated {len(outputs)}/{len(conversations)} prompts", flush=True)
        return outputs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("x", encoding="utf-8")

    def write(row: dict) -> None:
        handle.write(json.dumps({**meta, **row}, ensure_ascii=False) + "\n")

    def elicit(jobs: list[dict]) -> None:
        """jobs: dicts with 'messages' and identifying fields; retried until valid or capped."""
        pending = jobs
        for attempt in range(MAX_ATTEMPTS):
            if not pending:
                return
            responses = generate([j["messages"] for j in pending], 1, SEED + 1000 * (attempt + 1))
            retry = []
            for job, response in zip(pending, responses):
                out = response.outputs[0]
                parsed = parse_profile(job["format"], out.text, out.finish_reason, mode)
                write({"stage": "elicit", **{k: v for k, v in job.items() if k != "messages"},
                       "attempt": attempt, "messages": job["messages"],
                       "raw_response": out.text, "finish_reason": out.finish_reason,
                       "generation_tokens": len(out.token_ids), **parsed})
                if not parsed["valid"]:
                    retry.append(job)
            handle.flush()
            pending = retry

    print(f"Stage 1: {len(work['answers'])} prompts x {samples} samples", flush=True)
    conversations = [task_messages(p, t, system_in_user) for p, t in work["answers"]]
    two_stage_jobs = []
    for (problem, task), messages, response in zip(
            work["answers"], conversations, generate(conversations, samples, SEED)):
        for sample_index, out in enumerate(response.outputs):
            result = classify_answer(problem, task, out.text, out.finish_reason, mode)
            write({"stage": "answer", "problem_id": problem["problem_id"], "task": task,
                   "sample_index": sample_index, "messages": messages,
                   "raw_response": out.text, "finish_reason": out.finish_reason,
                   "generation_tokens": len(out.token_ids), **result})
            answer = visible(out.text, mode)
            if work["two_stage"] and task != "T4" and answer and out.finish_reason != "length":
                for fmt in work["formats"]:
                    two_stage_jobs.append({
                        "arm": "two_stage", "format": fmt, "problem_id": problem["problem_id"],
                        "task": task, "sample_index": sample_index,
                        "own_answer": result.get("label"),
                        "messages": two_stage_messages(problem, task, answer, fmt, system_in_user)})
    handle.flush()

    direct_jobs = [{"arm": "direct", "format": fmt, "problem_id": problem["problem_id"],
                    "task": None, "sample_index": s, "own_answer": None,
                    "messages": direct_messages(problem, fmt)}
                   for problem, fmt in work["direct"] for s in range(samples)]
    print(f"Elicitation: {len(direct_jobs)} direct, {len(two_stage_jobs)} two-stage", flush=True)
    elicit(direct_jobs + two_stage_jobs)
    handle.close()
    print(f"Done: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=load_manifest())
    parser.add_argument("--mode", required=True, choices=LIMITS)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["T0", "T1"], choices=TASKS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    run(args.model, args.mode, args.problems, args.output, args.tasks, args.arms, args.formats,
        args.batch_size, args.tensor_parallel_size, args.gpu_memory_utilization)


if __name__ == "__main__":
    main()
