"""Summarize the incremental case-5 instruction-placement grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study.generate_case5_diagnostics import load_items
from study.run_case5_assumption_pilot import INSTRUCTION, VARIANTS as OLD_SYSTEM
from study.run_case5_diagnostics import SCORER_VERSION, build_messages, grade
from study.run_case5_instruction_stage1 import NEW_CELLS, build_items

CONDITIONS = (
    "missing_initial_solve", "box_solve", "tom_reference_solve",
    "missing_given_solve", "complete_solve", "zero_control_solve",
    "unknown_delta_solve",
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize(baseline_path: Path, system_path: Path,
              stage1_path: Path) -> None:
    baseline, system, stage1 = (load_items(baseline_path), _read(system_path),
                                _read(stage1_path))
    bases = {row["base_id"] for row in baseline}
    if len(system) != len(bases) * len(OLD_SYSTEM):
        raise ValueError("Incomplete earlier system-instruction file")
    if len(stage1) != len(bases) * len(NEW_CELLS):
        raise ValueError("Incomplete stage-1 file")
    expected = build_items(baseline)
    if [(row["item_id"], row["problem"], row["answer"])
            for row in stage1] != [(row["item_id"], row["problem"], row["answer"])
                                  for row in expected]:
        raise ValueError("Stage-1 prompts do not match the baseline bases")
    for row, conversation in zip(stage1, build_messages(expected)):
        if (row.get("system_prompt") != conversation[0]["content"]
                or row.get("user_prompt") != conversation[1]["content"]):
            raise ValueError("Stage-1 instruction placement differs")
    model = baseline[0]["model"]
    settings = baseline[0]["sampling"]
    revision = baseline[0]["revision"]
    for row in baseline + system + stage1:
        if (row["model"], row["revision"], row["sampling"], row["max_model_len"]) != (
                model, revision, settings, baseline[0]["max_model_len"]):
            raise ValueError("Model or sampling settings differ")
        row.update(grade(row, row["raw_response"], row["finish_reason"]))
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in baseline:
        cells.setdefault((row["variant"], "none"), []).append(row)
    for row in system:
        if (row["variant"] not in OLD_SYSTEM
                or row.get("extra_system_instruction") != INSTRUCTION):
            raise ValueError("Unexpected earlier system-instruction row")
        cells.setdefault((row["variant"], "system"), []).append(row)
    for row in stage1:
        placement = row["instruction_placement"]
        if (row["variant"], placement) not in NEW_CELLS:
            raise ValueError("Unexpected new cell")
        if row.get("instruction_text") != (INSTRUCTION if placement != "none" else None):
            raise ValueError("Unexpected instruction text")
        cells.setdefault((row["variant"], placement), []).append(row)
    print(f"{stage1_path.name}: {model}, bases={len(bases)}, "
          f"scorer_version={SCORER_VERSION}")
    print("condition | none correct/total/predicted/other | system | user")
    for variant in CONDITIONS:
        parts = []
        for placement in ("none", "system", "user"):
            group = cells.get((variant, placement))
            if (group is None or len(group) != len(bases)
                    or {row["base_id"] for row in group} != bases):
                raise ValueError(f"Missing cell {variant}/{placement}")
            predicted = sum(row["zero_equivalent_guess"] for row in group)
            correct = sum(row["correct"] for row in group)
            parts.append(f"{correct}/{len(group)}/{predicted}/{len(group)-correct-predicted}")
        print(f"{variant} | " + " | ".join(parts))
    explicit = cells[("explicit_unknown_solve", "none")]
    print(f"explicit_unknown_solve baseline: "
          f"{sum(row['correct'] for row in explicit)}/{len(explicit)} correct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--system", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.baseline, args.system, args.stage1)


if __name__ == "__main__":
    main()
