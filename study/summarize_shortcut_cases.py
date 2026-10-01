"""Summarize matched shortcut case 2–6 GPU outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from study.generate_shortcut_cases import UNDETERMINED
from study.run_shortcut_cases import (PARSER_VERSION, load_items,
                                      parse_final_answer_detailed)


def summarize(path: Path) -> None:
    rows = load_items(path)
    changed = 0
    for row in rows:
        parsed, status = parse_final_answer_detailed(
            row["raw_response"], row["finish_reason"])
        correct = parsed == row["answer"]
        changed += parsed != row["parsed_answer"] or correct != row["correct"]
        row["parsed_answer"], row["correct"], row["parse_status"] = parsed, correct, status
    case = rows[0]["case"]
    if len({(row["model"], row["revision"], row["mode"]) for row in rows}) != 1:
        raise ValueError(f"Mixed model configurations in {path}")
    model, revision, mode = rows[0]["model"], rows[0]["revision"], rows[0]["mode"]
    count = len(rows) // 2
    left, right = rows[::2], rows[1::2]
    left_correct = sum(row["correct"] for row in left)
    right_correct = sum(row["correct"] for row in right)
    print(f"\n{path.name}: case {case}, {model}, {mode}, revision={revision}")
    print(f"recomputed with parser_version={PARSER_VERSION}")
    print(f"{left[0]['variant']}: {left_correct}/{count}; "
          f"{right[0]['variant']}: {right_correct}/{count}; "
          f"gap={(left_correct - right_correct) / count:+.3f}")
    for group in (left, right):
        print(f"{group[0]['variant']} parse_failures="
              f"{sum(row['parsed_answer'] is None for row in group)} "
              f"truncations={sum(row['finish_reason'] == 'length' for row in group)} "
              f"unparsed_completed={sum(row['parsed_answer'] is None and row['finish_reason'] != 'length' for row in group)}")
    print(f"regraded_rows={changed}")
    if case in (2, 3):
        for operation in (("multiply", "divide") if case == 2 else
                          ("add", "subtract")):
            control = [row for row in left if row["operation"] == operation]
            challenge = [row for row in right if row["operation"] == operation]
            if not control:
                continue
            print(f"{operation}: {sum(row['correct'] for row in control)}/{len(control)} "
                  f"vs {sum(row['correct'] for row in challenge)}/{len(challenge)}")
    if case == 5:
        guessed = sum(isinstance(row["parsed_answer"], int) for row in right)
        abstained = sum(row["parsed_answer"] == UNDETERMINED for row in right)
        valid = guessed + abstained
        print(f"missing-premise numerical guesses={guessed}/{count}; "
              f"correct abstentions={abstained}/{count}; "
              f"valid final answers={valid}/{count}")
        unparsed = [f"{row['item_id']} ({row['parse_status']})" for row in rows
                    if row["parsed_answer"] is None and row["finish_reason"] != "length"]
        if unparsed:
            print("completed responses to inspect: " + ", ".join(unparsed))
    if case == 6:
        repeated = sum(changed["parsed_answer"] == original["answer"]
                       for original, changed in zip(left, right))
        print(f"changed-query answers repeating original gold={repeated}/{count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        summarize(path)


if __name__ == "__main__":
    main()
