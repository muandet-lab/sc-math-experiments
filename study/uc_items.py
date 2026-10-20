"""Unjustified-certainty items: content levels x presentations on the Case 5 bases.

Each problem is stored with a formal start set, from which the true answer set is computed
over a fixed numeric grid. Nothing about the answer is hand-written.

Content levels (true answer set for start set S, offset d = received - given):
  determined    S = {s}                       -> {s + d}
  bounded       S = {a, ..., a + 3}           -> four consecutive values
  spread        S = {a, a + 5, a + 10, a + 15} -> four values, same size as bounded, wider range
  one_sided     S = {a, a + 1, ...}           -> {a + d, ...}
  unconstrained S = {0, 1, ...}               -> {d, ...}
with a = s - 2, so bounded, spread and one_sided share their minimum.

Presentations: 'stated' (determined only), 'prominent' (its own sentence), 'passing' (a relative
clause on the first event), and for unconstrained 'omitted', 'passing' and 'explicit'. The
unconstrained 'omitted' and 'explicit' wordings are the Case 5 wordings verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from study.generate_case5_confirmatory import DOMAINS, generate as generate_case5

GRID_MAX = 50
TAIL = f"more than {GRID_MAX}"
GRID = [str(v) for v in range(GRID_MAX + 1)] + [TAIL]
CONTENTS = ("determined", "bounded", "spread", "one_sided", "unconstrained")
PRESENTATIONS = {"determined": ("stated",), "bounded": ("prominent", "passing"),
                 "spread": ("prominent", "passing"), "one_sided": ("prominent", "passing"),
                 "unconstrained": ("omitted", "passing", "explicit")}
SPREAD_STEP = 5


def case5_bases() -> list[dict]:
    """The 36 Case 5 confirmatory bases (development set), recovered from the frozen generator."""
    items, _ = generate_case5()
    bases = {}
    for item in items:
        if item["kind"] == "complete":
            receive_verb, give_verb = DOMAINS[item["base_id"] % len(DOMAINS)][1:]
            bases[item["base_id"]] = {
                "base_id": item["base_id"], "name": item["name"], "noun": item["noun"],
                "receive_verb": receive_verb, "give_verb": give_verb, "start": item["start"],
                "received": item["received"], "given": item["given"]}
    return [bases[k] for k in sorted(bases)]


def start_set(content: str, start: int) -> dict:
    """Formal start set: explicit values, or an open ray {low, low + 1, ...}."""
    low = start - 2
    if content == "determined":
        return {"values": [start]}
    if content == "bounded":
        return {"values": list(range(low, low + 4))}
    if content == "spread":
        return {"values": [low + SPREAD_STEP * k for k in range(4)]}
    if content == "one_sided":
        return {"from": low}
    if content == "unconstrained":
        return {"from": 0}
    raise ValueError(content)


def true_set(starts: dict, received: int, given: int) -> list[str]:
    """Grid labels of every reachable final count; receiving comes first, so all starts are feasible."""
    offset = received - given
    if "values" in starts:
        finals = sorted(s + offset for s in starts["values"])
        return [str(v) if v <= GRID_MAX else TAIL for v in finals]
    return [str(v) for v in range(starts["from"] + offset, GRID_MAX + 1)] + [TAIL]


def _start_phrase(content: str, starts: dict, noun: str) -> str:
    if content == "determined":
        return f"{starts['values'][0]} {noun}"
    if content == "bounded":
        return f"between {starts['values'][0]} and {starts['values'][-1]} {noun}"
    if content == "spread":
        v = starts["values"]
        return f"either {v[0]}, {v[1]}, {v[2]}, or {v[3]} {noun}"
    if content == "one_sided":
        return f"at least {starts['from']} {noun}"
    return f"some {noun}"


def problem(base: dict, content: str, presentation: str) -> dict:
    if presentation not in PRESENTATIONS[content]:
        raise ValueError(f"{content} has no {presentation} presentation")
    name, noun = base["name"], base["noun"]
    received, given = base["received"], base["given"]
    starts = start_set(content, base["start"])
    receive = f"{base['receive_verb']} {received} {noun}."
    give = f"{name} {base['give_verb']} {given} {noun}."
    if presentation == "omitted":
        sentences = [f"{name} {receive}", give]
    elif presentation == "explicit":
        sentences = [f"{name} starts with an unspecified number of {noun}.", f"{name} {receive}", give]
    elif presentation == "passing":
        sentences = [f"{name}, who started out with {_start_phrase(content, starts, noun)}, {receive}",
                     give]
    else:  # stated or prominent
        sentences = [f"{name} starts with {_start_phrase(content, starts, noun)}.",
                     f"{name} {receive}", give]
    facts = " ".join(sentences)
    answers = true_set(starts, received, given)
    return {"problem_id": f"uc-{base['base_id']:02d}-{content}-{presentation}",
            "base_id": base["base_id"], "content": content, "presentation": presentation,
            "facts": facts, "question": f"How many {noun} does {name} have now?",
            "name": name, "noun": noun, "start_set": starts, "received": received,
            "given": given, "true_set": answers, "true_set_size": len(answers),
            "zero_default": received - given}


def build(bases: list[dict], contents=CONTENTS) -> list[dict]:
    rows = [problem(b, c, p) for b in bases for c in contents for p in PRESENTATIONS[c]]
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    ids = [r["problem_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate problem ids")
    for r in rows:
        if not r["true_set"] or any(v not in GRID for v in r["true_set"]):
            raise ValueError(f"Bad true set: {r['problem_id']}")
        if r["content"] == "determined" and r["true_set_size"] != 1:
            raise ValueError(f"Determined problem is not determined: {r['problem_id']}")
        if r["content"] in ("bounded", "spread") and r["true_set_size"] != 4:
            raise ValueError(f"Finite set has the wrong size: {r['problem_id']}")
        if r["content"] in ("one_sided", "unconstrained") and r["true_set"][-1] != GRID[-1]:
            raise ValueError(f"Open set does not reach the tail bin: {r['problem_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bases", type=int, default=36, help="first N Case 5 bases")
    parser.add_argument("--contents", nargs="+", default=list(CONTENTS), choices=CONTENTS)
    args = parser.parse_args()
    rows = build(case5_bases()[:args.bases], tuple(args.contents))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} problems, sha256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
