"""Exploratory five-base-problem MLX pilot; excluded from the final study."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from study.generate import generate

SEED = 20260929
N_BASE = 8
MAX_STEPS = 3  # Five of the eight generated base problems meet this rule.
SYSTEM = "Answer arithmetic questions using exactly one line: Final answer: <integer>."
MODELS = {
    "qwen": {
        "repo": "mlx-community/Qwen3-0.6B-4bit",
        "revision": "73e3e38d981303bc594367cd910ea6eb48349da8",
        "path": "study/private/models/qwen3_0_6b_4bit",
        "mode": "non-thinking",
        "max_tokens": 2048,
    },
    "olmo": {
        "repo": "mlx-community/Olmo-3-7B-Think-4bit",
        "revision": "65294083d66d52f7bee42c5da768e3ad456332c4",
        "path": "study/private/models/olmo3_7b_think_4bit",
        "mode": "native-thinking",
        "max_tokens": 1024,
    },
}


def extract_answer(raw: str) -> int | None:
    text = raw.replace("**", "").replace("__", "")
    integer = r"(-?\d+)(?!\d|\.\d)"
    matches = re.findall(r"(?is)final\s*answer\s*[:：]?\s*[<\s]*" + integer, text)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"\\boxed\s*\{\s*(-?\d+)\s*\}", text)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"(?is)\banswer\s*[:：]\s*[<\s]*" + integer, text)
    if matches:
        return int(matches[-1])
    return None


def run(model_key: str, output: Path) -> None:
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    spec = MODELS[model_key]
    model, tokenizer = load(spec["path"])
    items = [item for item in generate(N_BASE, SEED) if item["n_steps"] <= MAX_STEPS]
    random.Random(SEED + 1).shuffle(items)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            completed = {json.loads(line)["item_id"] for line in handle if line.strip()}
    with output.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(items, 1):
            if item["item_id"] in completed:
                continue
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": item["problem"]},
            ]
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if model_key == "qwen":
                template_kwargs["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            chunks = []
            last = None
            for response in stream_generate(
                model, tokenizer, prompt, max_tokens=spec["max_tokens"],
                sampler=make_sampler(temp=0.0),
            ):
                chunks.append(response.text)
                last = response
            raw = "".join(chunks)
            parsed = extract_answer(raw)
            row = {
                "item_id": item["item_id"], "base_id": item["base_id"],
                "form": item["form"], "reversal": item["reversal"],
                "operation": item["operation"], "keyword": item["keyword"],
                "problem": item["problem"], "answer": item["answer"],
                "model": spec["repo"], "revision": spec["revision"],
                "precision": "4-bit MLX community conversion", "mode": spec["mode"],
                "mlx_lm_version": mlx_lm.__version__,
                "prompt": prompt, "max_tokens": spec["max_tokens"],
                "decoding": "greedy", "raw_response": raw,
                "parsed_answer": parsed, "correct": parsed == item["answer"],
                "finish_reason": last.finish_reason if last else "empty",
                "generation_tokens": last.generation_tokens if last else 0,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(items)} {item['item_id']} "
                  f"answer={parsed} gold={item['answer']} "
                  f"finish={row['finish_reason']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
