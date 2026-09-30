"""Generate fresh, matched keyword-consistency problems without model calls.

This is deliberately separate from the published `data/` files. The caller owns
the seed and should keep final evaluation seeds private until runs are complete.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NAMES = ("Eli", "Nora", "Maya", "Owen", "Lena", "Iris", "Amir", "Theo")
ENTITIES = ("tokens", "marbles", "stickers", "cards", "shells", "coins")
OPERATIONS = ("add", "subtract", "multiply", "divide")
FORMS = ("verbal", "symbolic")


def _apply(value: int, operation: str, amount: int) -> int:
    if operation == "add":
        return value + amount
    if operation == "subtract":
        return value - amount
    if operation == "multiply":
        return value * amount
    if operation == "divide":
        if value % amount:
            raise ValueError("Noninteger division")
        return value // amount
    raise ValueError(operation)


def _comparison(rng: random.Random, source: int, operation: str) -> tuple[int, int]:
    candidates = []
    for amount in range(2, 10 if operation in ("add", "subtract") else 6):
        try:
            result = _apply(source, operation, amount)
        except ValueError:
            continue
        if 2 <= result <= 20:
            candidates.append((amount, result))
    if not candidates:
        raise ValueError("No valid comparison from this source")
    return rng.choice(candidates)


def _change(rng: random.Random, value: int) -> tuple[str, int, int]:
    candidates = [(op, amount, _apply(value, op, amount))
                  for op in ("add", "subtract") for amount in range(2, 7)
                  if 2 <= _apply(value, op, amount) <= 20]
    return rng.choice(candidates)


def _relation(source_name: str, target_name: str, entity: str,
              operation: str, amount: int, form: str, reversal: bool) -> tuple[str, str]:
    """Return sentence and the operation suggested by its surface wording."""
    if operation in ("add", "subtract"):
        positive = operation == "add"
        keyword = "more" if positive != reversal else "fewer"
        surface_op = "add" if keyword == "more" else "subtract"
        if form == "verbal":
            subject, referent = ((source_name, target_name) if reversal
                                 else (target_name, source_name))
            return f"{subject} has {amount} {keyword} {entity} than {referent}.", surface_op
        subject, referent = ((source_name, target_name) if reversal
                             else (target_name, source_name))
        sign = "+" if keyword == "more" else "−"
        return f"{subject}'s {entity} = {referent}'s {entity} {sign} {amount}.", surface_op

    positive = operation == "multiply"
    times = positive != reversal
    surface_op = "multiply" if times else "divide"
    subject, referent = ((source_name, target_name) if reversal
                         else (target_name, source_name))
    if form == "verbal":
        phrase = f"{amount} times as many" if times else f"1/{amount} as many"
        return f"{subject} has {phrase} {entity} as {referent}.", surface_op
    operator = "×" if times else "÷"
    return f"{subject}'s {entity} = {referent}'s {entity} {operator} {amount}.", surface_op


def generate_base(rng: random.Random, base_id: int, operation: str) -> list[dict]:
    """Create the four versions of one 1–5-step problem."""
    if operation not in OPERATIONS:
        raise ValueError(operation)
    source_name, target_name = rng.sample(NAMES, 2)
    entity = rng.choice(ENTITIES)
    n_steps = rng.randint(1, 5)
    comparison_step = rng.randint(1, n_steps)
    pre_count = comparison_step - 1
    post_count = n_steps - comparison_step

    # Resample the initial value and pre-steps until the chosen comparison is valid.
    for _ in range(100):
        initial = rng.randint(2, 20)
        value = initial
        pre = []
        for _ in range(pre_count):
            op, amount, after = _change(rng, value)
            pre.append((op, amount, value, after))
            value = after
        try:
            amount, target = _comparison(rng, value, operation)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("Could not draw a valid comparison")

    source_at_comparison = value
    post = []
    value = target
    for _ in range(post_count):
        op, change, after = _change(rng, value)
        post.append((op, change, value, after))
        value = after

    prefix = [f"{source_name} has {initial} {entity}."]
    suffix = []
    steps = []
    for i, (op, change, before, after) in enumerate(pre, 1):
        verb = "receives" if op == "add" else "gives away"
        prefix.append(f"{source_name} {verb} {change} {entity}.")
        steps.append({"step": i, "kind": "change", "agent": source_name,
                      "logical_form": {"operation": op, "input": before, "amount": change},
                      "value": after})
    steps.append({"step": comparison_step, "kind": "comparison", "agent": target_name,
                  "logical_form": {"operation": operation, "input": source_at_comparison,
                                   "amount": amount, "source_agent": source_name},
                  "value": target})
    for i, (op, change, before, after) in enumerate(post, comparison_step + 1):
        verb = "receives" if op == "add" else "gives away"
        suffix.append(f"{target_name} {verb} {change} {entity}.")
        steps.append({"step": i, "kind": "change", "agent": target_name,
                      "logical_form": {"operation": op, "input": before, "amount": change},
                      "value": after})
    question = f"How many {entity} does {target_name} have now?"

    rows = []
    for form in FORMS:
        for reversal in (False, True):
            relation, keyword_operation = _relation(source_name, target_name, entity,
                                                     operation, amount, form, reversal)
            sentences = prefix + [relation] + suffix + [question]
            row = {
                "base_id": base_id, "item_id": f"{base_id:05d}-{form}-{'rev' if reversal else 'direct'}",
                "form": form, "reversal": reversal,
                "consistency": "inconsistent" if reversal else "consistent",
                "operation": operation, "keyword_operation": keyword_operation,
                "keyword": None if form == "symbolic" else
                           (("more" if keyword_operation == "add" else "fewer")
                            if operation in ("add", "subtract") else
                            ("times as many" if keyword_operation == "multiply" else "a fraction of")),
                "amount": amount, "source_agent": source_name, "target_agent": target_name,
                "entity": entity, "initial_value": initial, "comparison_step": comparison_step,
                "n_steps": n_steps, "steps": steps, "sentences": sentences,
                "problem": " ".join(sentences), "answer": value,
            }
            validate_item(row)
            rows.append(row)
    return rows


def validate_item(item: dict) -> None:
    """Fail closed if a renderer changes arithmetic or the manipulation."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    steps = item["steps"]
    require(len(steps) == item["n_steps"], "step count changed")
    require([step["step"] for step in steps] == list(range(1, len(steps) + 1)),
            "step numbering changed")
    require(sum(step["kind"] == "comparison" for step in steps) == 1,
            "comparison count changed")
    require(steps[item["comparison_step"] - 1]["kind"] == "comparison",
            "comparison position changed")
    value = item["initial_value"]
    for step in steps:
        lf = step["logical_form"]
        require(lf["input"] == value, "step input changed")
        value = _apply(value, lf["operation"], lf["amount"])
        require(step["value"] == value, "step value changed")
    require(value == item["answer"], "answer changed")
    require(item["problem"] == " ".join(item["sentences"]), "problem text changed")
    require(len(item["sentences"]) == item["n_steps"] + 2, "sentence count changed")
    sentence = item["sentences"][item["comparison_step"]]
    expected, surface_op = _relation(item["source_agent"], item["target_agent"],
                                     item["entity"], item["operation"], item["amount"],
                                     item["form"], item["reversal"])
    require(sentence == expected and item["keyword_operation"] == surface_op,
            "comparison relation changed")
    require((item["consistency"] == "consistent") == (not item["reversal"]),
            "consistency label changed")
    require(sentence.count(str(item["amount"])) == 1, "comparison amount changed")


def generate(count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    operations = [OPERATIONS[i % 4] for i in range(count)]
    rng.shuffle(operations)
    rows = []
    for base_id, operation in enumerate(operations):
        group = generate_base(rng, base_id, operation)
        reference = group[0]
        for row in group[1:]:
            if row["answer"] != reference["answer"] or row["steps"] != reference["steps"]:
                raise ValueError("Matched items disagree on ground truth")
            if any(a != b for i, (a, b) in enumerate(zip(row["sentences"], reference["sentences"]))
                   if i != row["comparison_step"]):
                raise ValueError("Matched items differ outside the comparison sentence")
        rows.extend(group)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60, help="Number of base problems")
    parser.add_argument("--seed", type=int, required=True, help="Private evaluation seed")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    rows = generate(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items in {args.count} matched groups to {args.output}")


if __name__ == "__main__":
    main()
