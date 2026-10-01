"""Run one of the five remaining Qwen3 sizes on the matched 20-item pilot.

Thinking is always enabled. This exploratory pilot is excluded from the final
evaluation. Run each size as a separate process to release GPU memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from study.run_gpu_pilot import run as run_gpu_pilot

QWEN_SIZES = {
    "1.7b": "qwen1.7b",
    "4b": "qwen4b",
    "8b": "qwen8b",
    "14b": "qwen14b",
    "32b": "qwen32b",
}


def run(size: str, output_dir: Path) -> Path:
    if size not in QWEN_SIZES:
        raise ValueError(f"Unsupported Qwen3 size: {size}")
    output = output_dir / f"qwen3-{size}-bf16-thinking-pilot.jsonl"
    run_gpu_pilot(QWEN_SIZES[size], "thinking", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=QWEN_SIZES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.size, args.output_dir)


if __name__ == "__main__":
    main()
