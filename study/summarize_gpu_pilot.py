"""Summarize the exploratory bf16 GPU pilot JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PILOT_FILES = (
    "qwen3-0.6b-bf16-nonthinking-pilot.jsonl",
    "qwen3-0.6b-bf16-thinking-pilot.jsonl",
    "olmo3-7b-think-bf16-pilot.jsonl",
)


def summarize(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 20 or len({row["item_id"] for row in rows}) != 20:
        raise ValueError(f"Expected 20 unique matched items in {path}")
    if len({row["revision"] for row in rows}) != 1:
        raise ValueError(f"Mixed model revisions in {path}")

    cells = {}
    print(f"\n{path.name}: {rows[0]['model']} {rows[0]['mode']}")
    print(f"revision={rows[0]['revision']} vllm={rows[0]['vllm_version']} "
          f"precision={rows[0]['precision']}")
    for form in ("verbal", "symbolic"):
        for reversal in (False, True):
            subset = [row for row in rows
                      if row["form"] == form and row["reversal"] == reversal]
            if len(subset) != 5:
                raise ValueError(f"Expected five items in {form}/{reversal}: {path}")
            cells[(form, reversal)] = sum(row["correct"] for row in subset) / 5
            print(f"{form:8} {'reversed' if reversal else 'direct':8} "
                  f"{int(5 * cells[(form, reversal)])}/5")
    verbal_gap = cells[("verbal", False)] - cells[("verbal", True)]
    symbolic_gap = cells[("symbolic", False)] - cells[("symbolic", True)]
    def pp(gap: float) -> str:
        value = round(gap * 100)
        return f"{value:+d}pp" if value else "0pp"

    print(f"CATE_verbal={pp(verbal_gap)} "
          f"CATE_symbolic={pp(symbolic_gap)} "
          f"delta_kw={pp(verbal_gap - symbolic_gap)}")
    print(f"correct={sum(row['correct'] for row in rows)}/20 "
          f"parse_failures={sum(row['parsed_answer'] is None for row in rows)} "
          f"truncations={sum(row['finish_reason'] == 'length' for row in rows)}")
    print("errors:", ", ".join(row["item_id"] for row in rows if not row["correct"]) or "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path,
                        default=[Path("study/outputs") / name for name in PILOT_FILES])
    args = parser.parse_args()
    for path in args.paths:
        summarize(path)


if __name__ == "__main__":
    main()
