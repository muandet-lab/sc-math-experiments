"""Generate a frozen, fresh Case 5 factorial set and separate prefill probes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from study.generate_shortcut_cases import UNDETERMINED

NAMES = ("Maya", "Eli", "Nora", "Omar", "Lina", "Theo", "Ava", "Noah", "Iris", "Ravi", "Zoe", "Milo")
DOMAINS = (("shells", "receives", "gives away"),
           ("coins", "earns", "spends"),
           ("cards", "finds", "loses"),
           ("marbles", "receives", "gives away"))
SALIENCE = ("implicit", "some", "explicit")
SLOTS = ("initial", "received", "given")
CHAINS = (2, 3)
CONTROLS = ("complete", "zero_start", "net_change")
SEED = 20261001
COUNT = 36


def _unknown(name: str, noun: str, receive_verb: str, give_verb: str,
             slot: str, salience: str) -> str:
    phrase = {"implicit": "", "some": "some ",
              "explicit": "an unspecified number of "}[salience]
    if slot == "initial":
        return ("" if salience == "implicit" else
                f"{name} starts with {phrase}{noun}.")
    verb = receive_verb if slot == "received" else give_verb
    return f"{name} {verb} {phrase}{noun}."


def _item(base: dict, slot: str | None, salience: str | None,
          chain: int, kind: str, number: int) -> dict:
    name, noun = base["name"], base["noun"]
    rv, gv = base["receive_verb"], base["give_verb"]
    start, received, given, extra = (base[k] for k in ("start", "received", "given", "extra"))
    initial = f"{name} starts with {start} {noun}."
    receive = f"{name} {rv} {received} {noun}."
    give = f"{name} {gv} {given} {noun}."
    if kind == "factorial":
        if slot == "initial":
            initial = _unknown(name, noun, rv, gv, slot, salience)
        elif slot == "received":
            receive = _unknown(name, noun, rv, gv, slot, salience)
        else:
            give = _unknown(name, noun, rv, gv, slot, salience)
    elif kind == "zero_start":
        initial = f"{name} starts with zero {noun}."
    elif kind == "net_change":
        initial = f"{name} starts with an unspecified number of {noun}."
    elif kind == "marked_pocket":
        initial = (f"{name} has {start} {noun} in a pocket and some more "
                   f"in a closed box.")
    elif kind == "infeasible_start":
        initial = ""
        given = received + base["infeasible_margin"]
        give = f"{name} {gv} {given} {noun}."
    actions = [initial, receive, give]
    if chain == 3:
        actions.append(f"{name} {rv} {extra} {noun}.")
    question = (f"By how much has {name}'s number of {noun} changed?" if kind == "net_change"
                else f"How many {noun} does {name} have now?")
    problem = " ".join(s for s in (*actions, question) if s)
    offset = received - given + (extra if chain == 3 else 0)
    symbolic = (None if kind in CONTROLS else
                [1, start + offset] if kind == "marked_pocket" else
                [1, offset] if slot in ("initial", None) else
                [1, start - given + (extra if chain == 3 else 0)] if slot == "received" else
                [-1, start + received + (extra if chain == 3 else 0)])
    gold = (UNDETERMINED if kind in ("factorial", "infeasible_start", "marked_pocket") else
            offset if kind in ("zero_start", "net_change") else start + offset)
    zero_default = (start + offset if kind == "marked_pocket" else
                    offset if slot in ("initial", None) or kind in ("zero_start", "net_change") else
                    start - given + (extra if chain == 3 else 0) if slot == "received" else
                    start + received + (extra if chain == 3 else 0))
    return {"case": 5, "item_id": f"c5c-{base['base_id']:02d}-{number:02d}",
            "base_id": base["base_id"], "kind": kind, "slot": slot,
            "salience": salience, "chain_length": chain, "problem": problem,
            "gold": gold, "symbolic_gold": symbolic,
            "zero_default": zero_default, "name": name, "noun": noun,
            "start": start, "received": received, "given": given,
            "extra": extra if chain == 3 else None, "seed": base["seed"]}


def generate(count: int = COUNT, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    if not 30 <= count <= 40:
        raise ValueError("Confirmatory base count must be 30–40")
    rng = random.Random(seed)
    items, probes = [], []
    for base_id in range(count):
        noun, rv, gv = DOMAINS[base_id % len(DOMAINS)]
        name = NAMES[base_id % len(NAMES)]
        while True:
            start, received, given, extra = (rng.randint(8, 19), rng.randint(4, 11),
                                             rng.randint(2, 7), rng.randint(2, 6))
            if (len({start, received, given, extra}) == 4 and received > given
                    and start + received - given + extra <= 30):
                break
        base = {"base_id": base_id, "name": name, "noun": noun,
                "receive_verb": rv, "give_verb": gv, "start": start,
                "received": received, "given": given, "extra": extra,
                "infeasible_margin": rng.randint(2, 5), "seed": seed}
        cells = [("factorial", slot, salience, chain)
                 for slot in SLOTS for salience in SALIENCE for chain in CHAINS]
        cells += [("infeasible_start", None, None, 2),
                  ("marked_pocket", "initial", "pocket_box", 2)]
        cells += [(kind, None, None, 2) for kind in CONTROLS]
        for n, (kind, slot, salience, chain) in enumerate(cells):
            items.append(_item(base, slot, salience, chain, kind, n))
        probes.append({"probe_id": f"c5probe-{base_id:02d}-omitted",
                       "base_id": base_id, "condition": "omitted_initial",
                       "problem": _item(base, "initial", "implicit", 2, "factorial", 0)["problem"],
                       "prefix": f"Before {name} {rv} any {noun}, {name} had",
                       "zero_continuation": " 0", "received_continuation": f" {received}",
                       "received": received, "seed": seed})
        probes.append({"probe_id": f"c5probe-{base_id:02d}-explicit",
                       "base_id": base_id, "condition": "explicit_unknown",
                       "problem": _item(base, "initial", "explicit", 2, "factorial", 0)["problem"],
                       "prefix": f"Before {name} {rv} any {noun}, {name} had",
                       "zero_continuation": " 0", "received_continuation": f" {received}",
                       "received": received, "seed": seed})
    validate(items, probes, count)
    return items, probes


def validate_items(items: list[dict], count: int = COUNT) -> None:
    if len(items) != count * 23:
        raise ValueError("Incomplete Case 5 confirmatory grid")
    if len({x["item_id"] for x in items}) != len(items):
        raise ValueError("Duplicate item ID")
    for base_id in range(count):
        block = [item for item in items if item["base_id"] == base_id]
        if len(block) != 23 or {int(item["item_id"].rsplit("-", 1)[-1]) for item in block} != set(range(23)):
            raise ValueError(f"Incomplete base {base_id}")
    for item in items:
        start, received, given = (item[k] for k in ("start", "received", "given"))
        extra = item["extra"] or 0
        offset = received - given + extra
        if item["kind"] == "factorial":
            expected = ([1, offset] if item["slot"] == "initial" else
                        [1, start - given + extra] if item["slot"] == "received" else
                        [-1, start + received + extra])
            if item["symbolic_gold"] != expected or item["gold"] != UNDETERMINED:
                raise ValueError(f"Incorrect factorial gold: {item['item_id']}")
        elif item["kind"] == "marked_pocket":
            if item["symbolic_gold"] != [1, start + offset] or item["gold"] != UNDETERMINED:
                raise ValueError(f"Incorrect marked gold: {item['item_id']}")
        elif item["kind"] == "infeasible_start":
            if item["symbolic_gold"] != [1, offset] or item["gold"] != UNDETERMINED:
                raise ValueError(f"Incorrect infeasible gold: {item['item_id']}")
        elif item["kind"] == "complete":
            if item["gold"] != start + offset:
                raise ValueError(f"Incorrect complete gold: {item['item_id']}")
        elif item["kind"] in ("zero_start", "net_change"):
            if item["gold"] != offset:
                raise ValueError(f"Incorrect control gold: {item['item_id']}")
        else:
            raise ValueError(f"Unknown item kind: {item['kind']}")
        if item["zero_default"] != (item["symbolic_gold"][1] if item["symbolic_gold"] is not None else offset):
            raise ValueError(f"Incorrect zero-default: {item['item_id']}")
        if item["gold"] == item["zero_default"] and item["kind"] not in ("zero_start", "net_change"):
            raise ValueError(f"Gold equals zero-default for {item['item_id']}")
        if item["kind"] == "factorial" and item["slot"] == "initial" and item["salience"] == "implicit":
            if str(item["start"]) in item["problem"]:
                raise ValueError("Omitted initial amount leaked into prompt")
        if item["kind"] == "infeasible_start" and item["zero_default"] >= 0:
            raise ValueError("Infeasible zero-start control must be negative")


def validate(items: list[dict], probes: list[dict], count: int = COUNT) -> None:
    validate_items(items, count)
    if len(probes) != count * 2:
        raise ValueError("Incomplete prefill probe set")
    if len({p["probe_id"] for p in probes}) != len(probes):
        raise ValueError("Duplicate prefill probe")


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    args = parser.parse_args()
    if args.items.exists() or args.probes.exists():
        raise FileExistsError("Refusing to overwrite generated study files")
    items, probes = generate()
    _write(args.items, items)
    _write(args.probes, probes)
    print(f"Wrote {len(items)} benchmark items and {len(probes)} prefill probes")


if __name__ == "__main__":
    main()
