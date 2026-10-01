"""Run the three case-5 assumption-instruction variants on existing bases."""

from __future__ import annotations

import argparse
from pathlib import Path

from study.run_case5_diagnostics import run
from study.run_shortcut_cases import MODELS

VARIANTS = ("complete_solve", "missing_initial_solve", "unknown_delta_solve")
BASELINE_REVISIONS = {
    "qwen": "c1899de289a04d12100db370d81485cdf75e47ca",
    "olmo": "d97e442d7cc678210054dbcc9b440894d62c89a4",
}
INSTRUCTION = ("Use only the stated facts. Do not assign values to quantities "
               "whose values are not given. If the requested quantity is not "
               "uniquely determined, state that.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--input", required=True, type=Path,
                        help="Existing nine-variant case-5 diagnostic item file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260929)
    args = parser.parse_args()
    run(args.model, args.input, args.output, args.max_tokens,
        args.max_model_len, args.sampling_seed, VARIANTS, INSTRUCTION,
        BASELINE_REVISIONS[args.model])


if __name__ == "__main__":
    main()
