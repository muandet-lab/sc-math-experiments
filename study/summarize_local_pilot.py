"""Describe matched pilot accuracy; no significance claims from five items."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from study.generate import generate
from study.run_local_pilot import MAX_STEPS, N_BASE, SEED, extract_answer

ROOT = Path("study/outputs")
FILES = {
    "Qwen3-0.6B (4-bit, non-thinking)": ROOT / "pilot_20260929_qwen_2048.jsonl",
    "OLMo 3 7B Think (4-bit)": ROOT / "pilot_20260929_olmo.jsonl",
}


def load_rows(path: Path, ids: set[str]) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["item_id"] in ids:
                row["parsed_answer"] = extract_answer(row["raw_response"])
                row["correct"] = row["parsed_answer"] == row["answer"]
                rows[row["item_id"]] = row
    return rows


def shortcut_answer(item: dict) -> int | None:
    """Final answer if the displayed comparison operator is applied forward."""
    if not item["reversal"]:
        return None
    comparison = item["steps"][item["comparison_step"] - 1]
    source = Fraction(comparison["logical_form"]["input"])
    amount = comparison["logical_form"]["amount"]
    op = item["keyword_operation"]
    value = {"add": source + amount, "subtract": source - amount,
             "multiply": source * amount, "divide": source / amount}[op]
    for step in item["steps"][item["comparison_step"]:]:
        lf = step["logical_form"]
        value += lf["amount"] if lf["operation"] == "add" else -lf["amount"]
    return int(value) if value.denominator == 1 else None


def main() -> None:
    items = [x for x in generate(N_BASE, SEED) if x["n_steps"] <= MAX_STEPS]
    ids = {x["item_id"] for x in items}
    item_lookup = {x["item_id"]: x for x in items}
    bases = sorted({x["base_id"] for x in items})
    print(f"Exploratory pilot: {len(bases)} base problems, {len(items)} matched renderings")
    print(f"Base IDs: {bases}; seed={SEED}; n_steps <= {MAX_STEPS}")
    for model, path in FILES.items():
        rows = load_rows(path, ids)
        print(f"\n{model}: {len(rows)}/{len(items)} responses")
        if len(rows) != len(items):
            print("INCOMPLETE: metrics omitted")
            continue
        acc = {}
        for form in ("verbal", "symbolic"):
            for reversal in (False, True):
                cell = [x for x in rows.values() if x["form"] == form and x["reversal"] == reversal]
                n_correct = sum(x["correct"] for x in cell)
                acc[(form, reversal)] = n_correct / len(cell)
                print(f"  {form} {'reversal' if reversal else 'direct'}: "
                      f"{n_correct}/{len(cell)} = {acc[(form, reversal)]:.1%}")
        verbal = acc[("verbal", False)] - acc[("verbal", True)]
        symbolic = acc[("symbolic", False)] - acc[("symbolic", True)]
        print(f"  CATE_verbal={verbal:+.1%}; CATE_symbolic={symbolic:+.1%}; "
              f"Delta_kw={verbal-symbolic:+.1%}")
        print(f"  parse failures={sum(x['parsed_answer'] is None for x in rows.values())}; "
              f"truncated={sum(x['finish_reason'] == 'length' for x in rows.values())}")
        for form in ("verbal", "symbolic"):
            reverse = [x for x in rows.values() if x["form"] == form and x["reversal"]]
            signatures = sum(x["parsed_answer"] is not None and
                             x["parsed_answer"] == shortcut_answer(item_lookup[x["item_id"]])
                             for x in reverse if not x["correct"])
            print(f"  {form} reversal shortcut-signature errors: {signatures}/"
                  f"{sum(not x['correct'] for x in reverse)} errors")
        for base in bases:
            group = sorted((x for x in rows.values() if x["base_id"] == base),
                           key=lambda x: (x["form"], x["reversal"]))
            scores = ", ".join(f"{x['form'][0]}{'R' if x['reversal'] else 'D'}="
                               f"{x['parsed_answer']}({'✓' if x['correct'] else '×'})"
                               for x in group)
            print(f"  base {base}: gold={group[0]['answer']} {scores}")


if __name__ == "__main__":
    main()
