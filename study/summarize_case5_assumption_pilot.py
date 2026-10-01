"""Compare the assumption-instruction pilot with the matched saved baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from study.generate_case5_diagnostics import load_items
from study.run_case5_assumption_pilot import INSTRUCTION, VARIANTS
from study.run_case5_diagnostics import SCORER_VERSION, grade


def summarize(baseline_path: Path, intervention_path: Path) -> None:
    baseline = [row for row in load_items(baseline_path)
                if row["variant"] in VARIANTS]
    # Intervention files contain only three variants, so read JSONL directly.
    intervention = [json.loads(line) for line in intervention_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(baseline) != len(intervention) or not baseline:
        raise ValueError("Baseline and intervention have different row counts")
    keys = lambda rows: [(row["base_id"], row["variant"]) for row in rows]
    if keys(baseline) != keys(intervention):
        raise ValueError("Baseline and intervention are not matched in order")
    configs = ("model", "revision", "mode", "precision", "max_model_len")
    for old, new in zip(baseline, intervention):
        if any(old[key] != new[key] for key in configs):
            raise ValueError("Model configurations differ")
        if old["sampling"] != new["sampling"]:
            raise ValueError("Sampling settings differ")
        if new.get("extra_system_instruction") != INSTRUCTION:
            raise ValueError("Missing or different assumption instruction")
        if old["problem"] != new["problem"] or old["answer"] != new["answer"]:
            raise ValueError("Baseline and intervention prompts differ")
        old.update(grade(old, old["raw_response"], old["finish_reason"]))
        new.update(grade(new, new["raw_response"], new["finish_reason"]))
    print(f"{intervention_path.name}: {intervention[0]['model']}, "
          f"scorer_version={SCORER_VERSION}, matched_bases={len(intervention)//3}")
    for variant in VARIANTS:
        old = [row for row in baseline if row["variant"] == variant]
        new = [row for row in intervention if row["variant"] == variant]
        old_counts = Counter(row["classification"] for row in old)
        new_counts = Counter(row["classification"] for row in new)
        print(f"{variant}: correct {sum(row['correct'] for row in old)}/{len(old)} -> "
              f"{sum(row['correct'] for row in new)}/{len(new)}; "
              f"unsupported_numeric {old_counts['unsupported_numeric']} -> "
              f"{new_counts['unsupported_numeric']}; "
              f"truncations {old_counts['truncation']} -> {new_counts['truncation']}; "
              f"other_errors {old_counts['other_error']} -> {new_counts['other_error']}")
    old_missing = [row for row in baseline if row["variant"] == "missing_initial_solve"]
    new_missing = [row for row in intervention if row["variant"] == "missing_initial_solve"]
    repaired = sum(not old["correct"] and new["correct"]
                   for old, new in zip(old_missing, new_missing))
    print(f"omitted-start errors corrected on matched bases: "
          f"{repaired}/{len(old_missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--intervention", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.baseline, args.intervention)


if __name__ == "__main__":
    main()
