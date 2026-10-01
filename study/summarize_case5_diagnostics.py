"""Summarize the three case-5 diagnostic tests from raw model responses."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from study.generate_case5_diagnostics import VARIANTS, load_items
from study.run_case5_diagnostics import SCORER_VERSION, grade


def summarize(path: Path) -> None:
    rows = load_items(path)
    if len({(row["model"], row["revision"], row["mode"],
             row["sampling"]["max_tokens"], row["max_model_len"])
            for row in rows}) != 1:
        raise ValueError("Mixed model configurations in one diagnostic file")
    changed = 0
    for row in rows:
        result = grade(row, row["raw_response"], row["finish_reason"])
        changed += any(row.get(key) != value for key, value in result.items())
        row.update(result)
    print(f"\n{path.name}: {rows[0]['model']}, {rows[0]['mode']}, "
          f"revision={rows[0]['revision']}")
    print(f"max_tokens={rows[0]['sampling']['max_tokens']} "
          f"max_model_len={rows[0]['max_model_len']} "
          f"diagnostic_scorer_version={SCORER_VERSION} regraded_rows={changed}")
    by_variant = {variant: [row for row in rows if row["variant"] == variant]
                  for variant in VARIANTS}
    for variant, group in by_variant.items():
        counts = Counter(row["classification"] for row in group)
        completed = sum(row["finish_reason"] != "length" for row in group)
        unparsed = sum(row["parsed_answer"] is None and row["finish_reason"] != "length"
                       for row in group)
        print(f"{variant}: correct={sum(row['correct'] for row in group)}/{len(group)} "
              f"completed={completed}/{len(group)} "
              f"truncated={counts['truncation']} unparsed_completed={unparsed} "
              f"unsupported_numeric={counts['unsupported_numeric']} "
              f"zero_equivalent={sum(row['zero_equivalent_guess'] for row in group)} "
              f"conditional={counts['conditional_answer']} "
              f"symbolic_correct={counts['correct_symbolic_underdetermined']} "
              f"other_error={counts['other_error']}")
    by_base = {}
    for row in rows:
        by_base.setdefault(row["base_id"], {})[row["variant"]] = row
    detected_but_guessed = sum(
        block["missing_initial_answerability"]["classification"] == "correct_answerability"
        and block["missing_initial_solve"]["classification"] == "unsupported_numeric"
        for block in by_base.values())
    print(f"missing-initial answerability correct but solve guessed="
          f"{detected_but_guessed}/{len(by_base)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        summarize(path)


if __name__ == "__main__":
    main()
