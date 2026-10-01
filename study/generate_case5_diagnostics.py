"""Generate matched probes for why missing-premise arithmetic fails."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from study.generate import ENTITIES, NAMES
from study.generate_shortcut_cases import UNDETERMINED

VARIANTS = (
    "complete_solve", "missing_initial_solve", "missing_received_solve",
    "missing_given_solve", "explicit_unknown_solve", "zero_control_solve",
    "unknown_delta_solve", "complete_answerability",
    "missing_initial_answerability",
)
TESTS = {
    "complete_solve": (1, 3),
    "missing_initial_solve": (1, 2, 3),
    "missing_received_solve": (1,),
    "missing_given_solve": (1,),
    "explicit_unknown_solve": (2,),
    "zero_control_solve": (2,),
    "unknown_delta_solve": (2,),
    "complete_answerability": (3,),
    "missing_initial_answerability": (3,),
}


def generate(count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    rows = []
    for base_id in range(count):
        name = rng.choice(NAMES)
        entity = rng.choice(ENTITIES)
        while True:
            start = rng.randint(7, 16)
            received = rng.randint(3, 8)
            given = rng.randint(2, received - 1)
            if len({start, received, given}) == 3 and start + received - given <= 20:
                break
        delta = received - given
        initial = f"{name} starts with {start} {entity}."
        receive = f"{name} receives {received} {entity}."
        give = f"{name} gives away {given} {entity}."
        question = f"How many {entity} does {name} have now?"
        direct = f"Does the information uniquely determine how many {entity} {name} has now? Explain."
        implicit = (receive, give)
        variants = {
            "complete_solve": ((initial, receive, give, question), start + delta,
                               "solve", None),
            "missing_initial_solve": ((*implicit, question), UNDETERMINED,
                                      "solve", (1, delta)),
            "missing_received_solve": ((initial, f"{name} receives some {entity}.", give,
                                        question), UNDETERMINED, "solve",
                                       (1, start - given)),
            "missing_given_solve": ((initial, receive, f"{name} gives away some {entity}.",
                                     question), UNDETERMINED, "solve",
                                    (-1, start + received)),
            "explicit_unknown_solve": ((f"{name} starts with an unspecified number of {entity}.",
                                        receive, give, question), UNDETERMINED,
                                       "solve", (1, delta)),
            "zero_control_solve": ((f"{name} starts with zero {entity}.", receive,
                                    give, question), delta, "solve", None),
            "unknown_delta_solve": ((f"{name} starts with an unspecified number of {entity}.",
                                     receive, give,
                                     f"By how much has {name}'s number of {entity} changed?"),
                                    delta, "solve", None),
            "complete_answerability": ((initial, receive, give, direct), "yes",
                                       "answerability", None),
            "missing_initial_answerability": ((*implicit, direct), "no",
                                              "answerability", None),
        }
        for variant in VARIANTS:
            sentences, answer, task, symbolic = variants[variant]
            rows.append({"case": 5, "base_id": base_id,
                         "item_id": f"case5diag-{base_id:05d}-{variant}",
                         "variant": variant, "tests": TESTS[variant],
                         "task": task, "problem": " ".join(sentences),
                         "answer": answer, "symbolic_gold": symbolic,
                         "start": start, "received": received, "given": given,
                         "delta": delta, "name": name, "entity": entity,
                         "generator_seed": seed})
    validate_items(rows)
    return rows


def validate_items(rows: list[dict]) -> None:
    if not rows or len(rows) % len(VARIANTS):
        raise ValueError("Input must contain complete diagnostic blocks")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate item IDs")
    for offset in range(0, len(rows), len(VARIANTS)):
        block = rows[offset:offset + len(VARIANTS)]
        if tuple(row["variant"] for row in block) != VARIANTS:
            raise ValueError("Diagnostic variants or order changed")
        if len({row["base_id"] for row in block}) != 1:
            raise ValueError("Diagnostic base IDs differ")
        start, received, given = (block[0][key] for key in ("start", "received", "given"))
        if (len({start, received, given}) != 3 or received <= given
                or block[0]["answer"] != start + received - given
                or block[5]["answer"] != received - given
                or block[6]["answer"] != received - given
                or any(row["case"] != 5 for row in block)):
            raise ValueError("Invalid diagnostic arithmetic")
        for row in block:
            if row["task"] != ("answerability" if "answerability" in row["variant"]
                              else "solve") or tuple(row["tests"]) != TESTS[row["variant"]]:
                raise ValueError("Invalid diagnostic task metadata")


def load_items(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    validate_items(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True, help="Number of base problems")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = generate(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} diagnostic prompts ({args.count} bases) to {args.output}")


if __name__ == "__main__":
    main()
