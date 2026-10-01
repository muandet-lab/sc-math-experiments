"""Run only new instruction-placement cells on the original case-5 bases."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from study.generate_case5_diagnostics import load_items
from study.generate_shortcut_cases import UNDETERMINED
from study.run_case5_assumption_pilot import BASELINE_REVISIONS, INSTRUCTION
from study.run_case5_diagnostics import run
from study.run_shortcut_cases import MODELS

# Existing no-instruction/system cells are saved in the earlier pilots.
NEW_CELLS = (
    ("missing_initial_solve", "user"),
    ("box_solve", "none"), ("box_solve", "system"), ("box_solve", "user"),
    ("tom_reference_solve", "none"),
    ("tom_reference_solve", "system"),
    ("tom_reference_solve", "user"),
    ("missing_given_solve", "system"), ("missing_given_solve", "user"),
    ("complete_solve", "user"),
    ("zero_control_solve", "system"), ("zero_control_solve", "user"),
    ("unknown_delta_solve", "user"),
)


def build_items(source: list[dict]) -> list[dict]:
    by_base: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in source:
        by_base[row["base_id"]][row["variant"]] = row
    rows = []
    for base_id, block in by_base.items():
        required = {"missing_initial_solve", "missing_given_solve",
                    "complete_solve", "zero_control_solve", "unknown_delta_solve"}
        if not required <= block.keys():
            raise ValueError(f"Missing source variants in base {base_id}")
        template = block["missing_initial_solve"]
        name, entity = template["name"], template["entity"]
        receive = f"{name} receives {template['received']} {entity}."
        give = f"{name} gives away {template['given']} {entity}."
        question = f"How many {entity} does {name} have now?"
        novel = {
            "box_solve": {
                **template,
                "variant": "box_solve",
                "problem": (f"{name} has a box of {entity}. "
                            f"{receive} {give} {question}"),
                "answer": UNDETERMINED,
                "symbolic_gold": [1, template["delta"]],
            },
            "tom_reference_solve": {
                **template,
                "variant": "tom_reference_solve",
                "problem": (f"{name} has as many {entity} as Tom. "
                            f"{receive} {give} {question}"),
                "answer": UNDETERMINED,
                "symbolic_gold": [1, template["delta"]],
            },
        }
        for variant, placement in NEW_CELLS:
            item = dict(novel[variant] if variant in novel else block[variant])
            item["item_id"] = f"case5stage1-{base_id:05d}-{variant}-{placement}"
            item["instruction_placement"] = placement
            item["instruction_text"] = INSTRUCTION if placement != "none" else None
            rows.append(item)
    if len(rows) != len(by_base) * len(NEW_CELLS):
        raise ValueError("Incomplete stage-1 grid")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--input", required=True, type=Path,
                        help="Original nine-variant case-5 item file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--sampling-seed", type=int, default=20260929)
    args = parser.parse_args()
    items = build_items(load_items(args.input))
    run(args.model, args.input, args.output, args.max_tokens,
        args.max_model_len, args.sampling_seed,
        revision=BASELINE_REVISIONS[args.model], items_override=items)


if __name__ == "__main__":
    main()
