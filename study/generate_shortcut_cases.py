"""Generate fresh matched diagnostic items for shortcut cases 2–6.

Use a private seed for evaluation items. These templates need no model-based
grammar correction; every answer is computed and checked before writing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from study.generate import ENTITIES, NAMES, _apply, generate_base

CASES = (2, 3, 4, 5, 6)
VARIANTS = {
    2: ("direct", "reversed"),
    3: ("transfer", "comparison"),
    4: ("dependency_order", "reverse_sentence_order"),
    5: ("solvable", "missing_premise"),
    6: ("original_query", "changed_query"),
}
UNDETERMINED = "cannot_be_determined"


def _pair(case: int, base_id: int, variants: tuple[dict, dict]) -> list[dict]:
    rows = []
    for variant_name, variant in zip(VARIANTS[case], variants):
        rows.append({"case": case, "base_id": base_id,
                     "item_id": f"case{case}-{base_id:05d}-{variant_name}",
                     "variant": variant_name, **variant})
    return rows


def _base(rng: random.Random, base_id: int, operation: str) -> dict:
    return next(row for row in generate_base(rng, base_id, operation)
                if row["form"] == "verbal" and not row["reversal"])


def _multiplicative_pair(rng: random.Random, base_id: int) -> list[dict]:
    operation = ("multiply", "divide")[base_id % 2]
    rows = [row for row in generate_base(rng, base_id, operation)
            if row["form"] == "verbal"]
    direct, reversed_row = rows
    shared = {"operation": operation, "steps": direct["steps"],
              "comparison_step": direct["comparison_step"]}
    return _pair(2, base_id, (
        {**shared, "problem": direct["problem"], "answer": direct["answer"]},
        {**shared, "problem": reversed_row["problem"], "answer": reversed_row["answer"]},
    ))


def _transfer_comparison_pair(rng: random.Random, base_id: int) -> list[dict]:
    source, target = rng.sample(NAMES, 2)
    entity = rng.choice(ENTITIES)
    operation = ("add", "subtract")[base_id % 2]
    for _ in range(100):
        start, amount = rng.randint(2, 20), rng.randint(2, 6)
        result = _apply(start, operation, amount)
        if 2 <= result <= 20:
            break
    else:
        raise RuntimeError("Could not draw a bounded transfer")
    verb = "receives" if operation == "add" else "gives away"
    keyword = "more" if operation == "add" else "fewer"
    transfer = (f"{target} has {start} {entity}. "
                f"{target} {verb} {amount} {entity}. "
                f"How many {entity} does {target} have now?")
    comparison = (f"{source} has {start} {entity}. "
                  f"{target} has {amount} {keyword} {entity} than {source}. "
                  f"How many {entity} does {target} have now?")
    shared = {"operation": operation, "steps": [
        {"step": 1, "operation": operation, "input": start,
         "amount": amount, "value": result}]}
    return _pair(3, base_id, (
        {**shared, "problem": transfer, "answer": result},
        {**shared, "problem": comparison, "answer": result},
    ))


def _sentence_order_pair(rng: random.Random, base_id: int) -> list[dict]:
    first, middle, last = rng.sample(NAMES, 3)
    entity = rng.choice(ENTITIES)
    for _ in range(100):
        start = rng.randint(4, 15)
        operations = rng.sample(("add", "subtract"), 2)
        amounts = (rng.randint(2, 5), rng.randint(2, 5))
        middle_value = _apply(start, operations[0], amounts[0])
        end = _apply(middle_value, operations[1], amounts[1])
        if 2 <= middle_value <= 20 and 2 <= end <= 20 and end != start:
            break
    else:
        raise RuntimeError("Could not draw a bounded comparison chain")
    keyword_1 = "more" if operations[0] == "add" else "fewer"
    keyword_2 = "more" if operations[1] == "add" else "fewer"
    fact = f"{first} has {start} {entity}."
    relation_1 = f"{middle} has {amounts[0]} {keyword_1} {entity} than {first}."
    relation_2 = f"{last} has {amounts[1]} {keyword_2} {entity} than {middle}."
    question = f"How many {entity} does {last} have?"
    shared = {"operation": "two_comparisons", "steps": [
        {"step": 1, "operation": operations[0], "input": start,
         "amount": amounts[0], "value": middle_value},
        {"step": 2, "operation": operations[1], "input": middle_value,
         "amount": amounts[1], "value": end}]}
    return _pair(4, base_id, (
        {**shared, "problem": " ".join((fact, relation_1, relation_2, question)),
         "answer": end},
        {**shared, "problem": " ".join((fact, relation_2, relation_1, question)),
         "answer": end},
    ))


def _solvability_pair(rng: random.Random, base_id: int) -> list[dict]:
    row = _base(rng, base_id, ("add", "subtract", "multiply", "divide")[base_id % 4])
    shared = {"operation": row["operation"], "steps": row["steps"]}
    return _pair(5, base_id, (
        {**shared, "problem": row["problem"], "answer": row["answer"]},
        {**shared, "problem": " ".join(row["sentences"][1:]),
         "answer": UNDETERMINED},
    ))


def _query_switch_pair(rng: random.Random, base_id: int) -> list[dict]:
    operation = ("add", "subtract", "multiply", "divide")[base_id % 4]
    for _ in range(100):
        row = _base(rng, base_id, operation)
        source_answer = row["steps"][row["comparison_step"] - 1]["logical_form"]["input"]
        if (1 < row["comparison_step"] < row["n_steps"]
                and source_answer != row["answer"]):
            break
    else:
        raise RuntimeError("Could not draw two distinct query answers")
    changed_question = (f"How many {row['entity']} does "
                        f"{row['source_agent']} have now?")
    shared = {"operation": operation, "steps": row["steps"]}
    return _pair(6, base_id, (
        {**shared, "problem": row["problem"], "answer": row["answer"]},
        {**shared, "problem": " ".join(row["sentences"][:-1] + [changed_question]),
         "answer": source_answer},
    ))


BUILDERS = {
    2: _multiplicative_pair,
    3: _transfer_comparison_pair,
    4: _sentence_order_pair,
    5: _solvability_pair,
    6: _query_switch_pair,
}


def validate_pair(rows: list[dict]) -> None:
    if len(rows) != 2 or rows[0]["case"] != rows[1]["case"]:
        raise ValueError("Expected two variants of one case")
    case = rows[0]["case"]
    if [row["variant"] for row in rows] != list(VARIANTS[case]):
        raise ValueError("Variant labels changed")
    if rows[0]["base_id"] != rows[1]["base_id"]:
        raise ValueError("Base IDs differ")
    left, right = rows
    if case in (2, 3, 4) and left["answer"] != right["answer"]:
        raise ValueError("Matched forms must have the same answer")
    if case == 2:
        if left["operation"] not in ("multiply", "divide"):
            raise ValueError("Multiplicative case has the wrong operation")
        left_sentences = left["problem"].split(". ")
        right_sentences = right["problem"].split(". ")
        comparison = left["comparison_step"]
        if (len(left_sentences) != len(right_sentences)
                or any(a != b for index, (a, b) in enumerate(
                    zip(left_sentences, right_sentences)) if index != comparison)):
            raise ValueError("Multiplicative pair differs outside the relation")
    if case == 3:
        if (left["operation"] not in ("add", "subtract")
                or left["problem"].split(". ")[-1]
                != right["problem"].split(". ")[-1]):
            raise ValueError("Transfer and comparison must ask the same question")
    if case == 4:
        first = left["problem"].split(". ")
        second = right["problem"].split(". ")
        if len(first) != 4 or second != [first[0], first[2], first[1], first[3]]:
            raise ValueError("Sentence-order pair must reorder only the two facts")
    if case == 5:
        if not isinstance(left["answer"], int) or right["answer"] != UNDETERMINED:
            raise ValueError("Solvability labels are invalid")
        if left["problem"].split(". ")[1:] != right["problem"].split(". "):
            raise ValueError("Exactly one premise must be removed")
    if case == 6:
        if left["answer"] == right["answer"]:
            raise ValueError("Changed query must change the answer")
        if left["problem"].rsplit(". ", 1)[0] != right["problem"].rsplit(". ", 1)[0]:
            raise ValueError("Only the question may change")


def generate(case: int, count: int, seed: int) -> list[dict]:
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    rows = []
    for base_id in range(count):
        pair = BUILDERS[case](rng, base_id)
        validate_pair(pair)
        for row in pair:
            row["generator_seed"] = seed
        rows.extend(pair)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=int, choices=CASES)
    parser.add_argument("--count", type=int, required=True, help="Number of base pairs")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = generate(args.case, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} items ({args.count} pairs) to {args.output}")


if __name__ == "__main__":
    main()
