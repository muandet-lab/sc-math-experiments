"""Summarize unjustified-certainty pilot runs into per-cell CSV tables.

answers.csv: stage-1 outcome rates per model, mode, content, presentation and task, plus the
             first-order entropy of the sampled answers (mean over problems).
elicit.csv:  validity, and mean measures over the final valid profile of each request, per
             model, mode, content, presentation, arm, format and task.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from study.uc_items import GRID
from study.uc_measures import entropy, summarize

ANSWER_OUTCOMES = ("single", "abstain", "expression", "set", "answerability", "unparseable",
                   "truncated")
MEASURES = ("mmi", "mmi_bound", "set_size", "wasserstein_diameter", "contains_true",
            "within_true", "jaccard", "all_ones", "echo")


def _rows(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines()
            if line]


def answers_table(rows: list[dict], problems: dict) -> list[dict]:
    groups, answers = defaultdict(list), defaultdict(Counter)
    for r in rows:
        if r["stage"] != "answer":
            continue
        p = problems[r["problem_id"]]
        key = (r["model_key"], r["mode"], p["content"], p["presentation"], r["task"])
        groups[key].append(r)
        answers[key + (r["problem_id"],)][json.dumps(r.get("parsed"))] += 1
    table = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        out = Counter(r["outcome"] for r in rs)
        multi = problems[rs[0]["problem_id"]]["true_set_size"] > 1
        row = dict(zip(("model", "mode", "content", "presentation", "task"), key), n=n)
        row.update({f"rate_{o}": out[o] / n for o in ANSWER_OUTCOMES})
        row["rate_correct"] = sum(bool(r.get("correct")) for r in rs) / n
        row["rate_zero_default"] = sum(bool(r.get("zero_default")) for r in rs) / n
        row["rate_unjustified_single"] = out["single"] / n if multi else 0.0
        row["entropy_bits"] = mean(entropy(c) for k, c in answers.items() if k[:5] == key)
        table.append(row)
    return table


def elicit_table(rows: list[dict], problems: dict) -> list[dict]:
    final = {}
    for r in rows:
        if r["stage"] != "elicit":
            continue
        request = (r["model_key"], r["mode"], r["problem_id"], r["arm"], r["format"], r["task"],
                   r["sample_index"])
        if request not in final or r["attempt"] > final[request]["attempt"]:
            final[request] = r
    groups = defaultdict(list)
    for (model, mode, pid, arm, fmt, task, _), r in final.items():
        p = problems[pid]
        groups[(model, mode, p["content"], p["presentation"], arm, fmt, task or "-")].append((r, p))
    table = []
    for key, items in sorted(groups.items()):
        valid = [(r, p) for r, p in items if r["valid"]]
        row = dict(zip(("model", "mode", "content", "presentation", "arm", "format", "task"), key),
                   n=len(items), valid_rate=len(valid) / len(items),
                   mean_attempts=mean(r["attempt"] + 1 for r, _ in items))
        scored = [summarize(r["format"], r["profile"], GRID, p["true_set"], r.get("own_answer"))
                  for r, p in valid]
        for m in MEASURES:
            values = [float(s[m]) for s in scored if m in s]
            row[m] = mean(values) if values else ""
        table.append(row)
    return table


def _write(path: Path, table: list[dict]) -> None:
    if not table:
        return
    fields = list(dict.fromkeys(k for row in table for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    problems = {p["problem_id"]: p for p in _rows([args.problems])}
    rows = _rows(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "answers.csv", answers_table(rows, problems))
    _write(args.output_dir / "elicit.csv", elicit_table(rows, problems))
    print(f"Wrote {args.output_dir}/answers.csv and elicit.csv")


if __name__ == "__main__":
    main()
