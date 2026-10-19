"""Build frozen Phase 1 items for replicating documented shortcuts.

Every family writes one JSONL file with a shared schema: the exact system and
user text, the answer type, the gold answer, and the answer predicted by the
documented shortcut ("signature"), where one is defined. See
shortcut_rep_spec.md for the designs and their sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from study.generate import ENTITIES, NAMES, _apply, generate as generate_keyword

FAMILIES = ("keyword", "opedal_anchor", "distractor", "option_position", "missing_start")
SYSTEM_INTEGER = ("Solve the problem. End with one line: Final answer: followed by "
                  "a whole number.")
SYSTEM_LETTER = ("Answer the multiple-choice question. End with one line: Final answer: "
                 "followed by the letter (A, B, C, or D) of the correct option.")
SYSTEM_TEXT = ("Answer the question by choosing one of the listed options. End with one "
               "line: Final answer: followed by the chosen option exactly as written.")
LETTERS = "ABCD"
SYMBOLS = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷"}
KEYWORD_COUNT, OPEDAL_COUNT, DISTRACTOR_COUNT = 300, 150, 300
MCQ_ARITHMETIC_COUNT, MCQ_MMLU_COUNT = 100, 200
REQUIRED = ("family", "item_id", "base_id", "cell", "system", "user", "answer_type", "gold")


def _item(family: str, item_id: str, base_id: int, cell: str, system: str, user: str,
          answer_type: str, gold, signature=None, **meta) -> dict:
    return {"family": family, "item_id": item_id, "base_id": base_id, "cell": cell,
            "system": system, "user": user, "answer_type": answer_type, "gold": gold,
            "signature": signature, **meta}


def _expression(initial: int, steps: list[tuple[str, int]]) -> str:
    expression = str(initial)
    for operation, amount in steps:
        expression = f"({expression} {SYMBOLS[operation]} {amount})"
    return expression[1:-1] if steps else expression


def _chain(value: int, steps: list[tuple[str, int]]) -> int | None:
    for operation, amount in steps:
        if operation == "divide" and value % amount:
            return None
        value = _apply(value, operation, amount)
    return value


def build_keyword(seed: int, count: int = KEYWORD_COUNT) -> list[dict]:
    """Opedal consistency pairs, the symbolic control, and an arithmetic twin."""
    items = []
    rows = generate_keyword(count, seed)
    for base_id in range(count):
        group = [row for row in rows if row["base_id"] == base_id]
        steps = [(s["logical_form"]["operation"], s["logical_form"]["amount"])
                 for s in group[0]["steps"]]
        comparison = group[0]["comparison_step"] - 1
        for row in group:
            cell = {("verbal", False): "verbal_consistent",
                    ("verbal", True): "verbal_inconsistent",
                    ("symbolic", False): "symbolic_direct",
                    ("symbolic", True): "symbolic_reversed"}[(row["form"], row["reversal"])]
            signature = None
            if row["reversal"]:
                cue_steps = list(steps)
                cue_steps[comparison] = (row["keyword_operation"], row["amount"])
                signature = _chain(row["initial_value"], cue_steps)
                if signature == row["answer"]:
                    raise ValueError("Keyword signature equals gold")
            items.append(_item("keyword", f"kw-{base_id:03d}-{cell}", base_id, cell,
                               SYSTEM_INTEGER, row["problem"], "integer", row["answer"],
                               signature, operation=row["operation"], amount=row["amount"],
                               n_steps=row["n_steps"], comparison_step=row["comparison_step"],
                               keyword=row["keyword"]))
        twin = f"Compute {_expression(group[0]['initial_value'], steps)}."
        if _chain(group[0]["initial_value"], steps) != group[0]["answer"]:
            raise ValueError("Arithmetic twin disagrees with the word problem")
        items.append(_item("keyword", f"kw-{base_id:03d}-arithmetic", base_id, "arithmetic",
                           SYSTEM_INTEGER, twin, "integer", group[0]["answer"],
                           operation=group[0]["operation"], amount=group[0]["amount"],
                           n_steps=group[0]["n_steps"],
                           comparison_step=group[0]["comparison_step"], keyword=None))
    return items


def build_opedal_anchor(csv_path: Path, seed: int, count: int = OPEDAL_COUNT) -> list[dict]:
    """A seeded sample of Opedal et al.'s published consistency pairs, as evaluated there."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    chosen = sorted(random.Random(seed).sample(range(len(rows)), count))
    items = []
    for base_id, index in enumerate(chosen):
        row = rows[index]
        for cell in ("consistent", "inconsistent"):
            items.append(_item("opedal_anchor", f"op-{base_id:03d}-{cell}", base_id, cell,
                               SYSTEM_INTEGER, row[f"{cell} paraphrased"].strip(), "integer",
                               int(row["answer"]), source_row=index, source_sha256=source_sha))
    return items


def build_distractor(seed: int, count: int = DISTRACTOR_COUNT) -> list[dict]:
    """GSM-NoOp-style irrelevant clause, without and with a relational keyword."""
    rng = random.Random(seed)
    items = []
    for base_id in range(count):
        for _ in range(1000):
            name, other = rng.sample(NAMES, 2)
            noun = rng.choice(ENTITIES)
            initial = rng.randint(5, 20)
            steps = [(rng.choice(("add", "subtract")), rng.randint(2, 9)) for _ in range(2)]
            values = [initial]
            for operation, amount in steps:
                values.append(_apply(values[-1], operation, amount))
            answer = values[-1]
            used = {initial, answer, *values, *(a for _, a in steps)}
            options = [d for d in range(2, 10) if d not in used
                       and answer - d >= 1 and answer + d not in used and answer - d not in used]
            if min(values) >= 2 and options:
                break
        else:
            raise RuntimeError("Could not draw a distractor base")
        amount = rng.choice(options)
        sentences = [f"{name} has {initial} {noun}."]
        sentences += [f"{name} {'receives' if op == 'add' else 'gives away'} {a} {noun}."
                      for op, a in steps]
        question = f"How many {noun} does {name} have now?"
        clauses = {"none": None,
                   "plain": f"{amount} of {name}'s {noun} are blue.",
                   "keyword_fewer": f"{other} has {amount} fewer {noun} than {name}.",
                   "keyword_more": f"{other} has {amount} more {noun} than {name}."}
        signatures = {"none": None, "plain": answer - amount,
                      "keyword_fewer": answer - amount, "keyword_more": answer + amount}
        for cell, clause in clauses.items():
            text = sentences[:1] + ([clause] if clause else []) + sentences[1:] + [question]
            items.append(_item("distractor", f"ds-{base_id:03d}-{cell}", base_id, cell,
                               SYSTEM_INTEGER, " ".join(text), "integer", answer,
                               signatures[cell], distractor_amount=amount))
        items.append(_item("distractor", f"ds-{base_id:03d}-arithmetic", base_id, "arithmetic",
                           SYSTEM_INTEGER, f"Compute {_expression(initial, steps)}.",
                           "integer", answer, distractor_amount=amount))
    return items


def _rotations(question: str, gold: str, others: list[str]) -> list[list[str]]:
    """Move the gold option through every position; other options keep their order."""
    if len({gold, *others}) != 4:
        raise ValueError(f"Options are not four distinct strings: {question!r}")
    return [others[:position] + [gold] + others[position:] for position in range(4)]


def build_option_position(seed: int, mmlu_rows: list[dict]) -> list[dict]:
    """Zheng et al.'s answer-moving design on fresh arithmetic and an MMLU sample."""
    rng = random.Random(seed)
    items = []
    for base_id in range(MCQ_ARITHMETIC_COUNT):
        name = rng.choice(NAMES)
        noun = rng.choice(ENTITIES)
        start, received, given = rng.randint(5, 20), rng.randint(2, 9), rng.randint(2, 9)
        gold = start + received - given
        offsets = rng.sample([k for k in range(-5, 6) if k and gold + k >= 1], 3)
        others = [str(gold + k) for k in offsets]
        question = (f"{name} has {start} {noun}. {name} receives {received} {noun}. "
                    f"{name} gives away {given} {noun}. How many {noun} does {name} have now?")
        for position, options in enumerate(_rotations(question, str(gold), others)):
            lettered = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))
            items.append(_item("option_position", f"mcq-ar-{base_id:03d}-ids-{position}",
                               base_id, f"arithmetic:ids:{LETTERS[position]}", SYSTEM_LETTER,
                               f"{question}\n\n{lettered}", "letter", LETTERS[position],
                               options=options, gold_index=position, source="arithmetic",
                               arm="ids"))
            bare = "\n".join(f"- {text}" for text in options)
            items.append(_item("option_position", f"mcq-ar-{base_id:03d}-noids-{position}",
                               base_id, f"arithmetic:no_ids:{LETTERS[position]}", SYSTEM_TEXT,
                               f"{question}\n\n{bare}", "option_text", options[position],
                               options=options, gold_index=position, source="arithmetic",
                               arm="no_ids"))
    for offset, row in enumerate(mmlu_rows):
        base_id = MCQ_ARITHMETIC_COUNT + offset
        choices = [str(c).strip() for c in row["choices"]]
        gold = choices[row["answer"]]
        others = [c for i, c in enumerate(choices) if i != row["answer"]]
        for position, options in enumerate(_rotations(row["question"], gold, others)):
            lettered = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))
            items.append(_item("option_position", f"mcq-mmlu-{offset:03d}-ids-{position}",
                               base_id, f"mmlu:ids:{LETTERS[position]}", SYSTEM_LETTER,
                               f"{row['question'].strip()}\n\n{lettered}", "letter",
                               LETTERS[position], options=options, gold_index=position,
                               source="mmlu", arm="ids", subject=row["subject"],
                               source_row=row["source_row"]))
    return items


def sample_mmlu(parquet_path: Path, seed: int, count: int = MCQ_MMLU_COUNT) -> list[dict]:
    """Seeded MMLU test sample with four distinct options."""
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    rng = random.Random(seed)
    rows = []
    for index in rng.sample(range(len(frame)), len(frame)):
        record = frame.iloc[index]
        choices = [str(c).strip() for c in record["choices"]]
        if len(choices) == 4 and len(set(choices)) == 4 and all(choices):
            rows.append({"question": str(record["question"]), "subject": str(record["subject"]),
                         "choices": choices, "answer": int(record["answer"]),
                         "source_row": index})
        if len(rows) == count:
            return rows
    raise ValueError("Too few usable MMLU rows")


def build_missing_start(case5_items_path: Path) -> list[dict]:
    """The frozen Case 5 confirmatory items, solve task, with and without the instruction."""
    from study.generate_shortcut_cases import UNDETERMINED
    from study.run_case5_confirmatory import conversation, load_items

    items = []
    for item in load_items(case5_items_path):
        # The zero-start default is a shortcut only where the start is unknown.
        signature = item.get("zero_default") if item["gold"] == UNDETERMINED else None
        for placement in ("none", "user"):
            system, user = (m["content"] for m in conversation(item, "solve", placement))
            cell = ":".join(str(item[k]) for k in ("kind", "slot", "salience")) + f":{placement}"
            items.append(_item("missing_start", f"ms-{item['item_id']}-{placement}",
                               item["base_id"], cell, system, user, "case5", item["gold"],
                               signature, case5=item,
                               instruction_placement=placement))
    return items


def validate(items: list[dict], family: str) -> None:
    """Fail closed on schema, duplicate, and matched-design violations."""
    if not items or any(item.get("family") != family for item in items):
        raise ValueError(f"Wrong or empty family file: {family}")
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Duplicate item ids")
    for item in items:
        if any(item.get(key) in (None, "") for key in REQUIRED):
            raise ValueError(f"Missing field in {item.get('item_id')}")
        if item["signature"] is not None and item["signature"] == item["gold"]:
            raise ValueError(f"Signature equals gold in {item['item_id']}")
    bases = {}
    for item in items:
        bases.setdefault(item["base_id"], []).append(item)
    for base_id, group in bases.items():
        if family in ("keyword", "distractor", "opedal_anchor"):
            if len({item["gold"] for item in group}) != 1:
                raise ValueError(f"Matched base {base_id} has different gold answers")
        if family == "option_position":
            for arm in {item["arm"] for item in group}:
                block = [item for item in group if item["arm"] == arm]
                if sorted(item["gold_index"] for item in block) != [0, 1, 2, 3]:
                    raise ValueError(f"Base {base_id} lacks a gold position")
                if len({frozenset(item["options"]) for item in block}) != 1:
                    raise ValueError(f"Base {base_id} changes its option set")


def load_family(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    validate(items, items[0]["family"] if items else "")
    return items


def build_all(seed: int, opedal_csv: Path, mmlu_parquet: Path, case5_items: Path) -> dict:
    # Each family uses its own derived seed so adding a family never changes another.
    return {
        "keyword": build_keyword(seed + 1),
        "opedal_anchor": build_opedal_anchor(opedal_csv, seed + 2),
        "distractor": build_distractor(seed + 3),
        "option_position": build_option_position(seed + 4, sample_mmlu(mmlu_parquet, seed + 5)),
        "missing_start": build_missing_start(case5_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opedal-csv", type=Path, required=True)
    parser.add_argument("--mmlu-parquet", type=Path, required=True)
    parser.add_argument("--case5-items", type=Path,
                        default=Path(__file__).with_name("case5_confirmatory_items.jsonl"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Refusing to write into non-empty {args.output_dir}")
    families = build_all(args.seed, args.opedal_csv, args.mmlu_parquet, args.case5_items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "families": {}}
    for family, items in families.items():
        validate(items, family)
        path = args.output_dir / f"{family}.jsonl"
        path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                        encoding="utf-8")
        summary["families"][family] = {"items": len(items),
                                       "bases": len({i["base_id"] for i in items}),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
