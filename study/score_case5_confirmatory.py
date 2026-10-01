"""Frozen Case 5 confirmatory scorer, version 1. Do not edit after model runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from study.generate_shortcut_cases import UNDETERMINED

SCORER_VERSION = "case5-confirmatory-v1"
FINAL = re.compile(r"(?i)\bfinal\s+answer(?:\*\*)?\s*[:：]\s*")
ABSTAIN = re.compile(r"(?i)\b(?:cannot|can't|can not)\s+be\s+determined\b|"
                     r"\b(?:not enough|insufficient)\s+information\b|\bunderdetermined\b")
NUMBER = re.compile(r"^[\s*\\()$]*([+-]?\d+)(?:\s+(?:shells|coins|cards|points|tokens|marbles|stickers))?[.\s*$]*$")


def _affine(line: str) -> tuple[int, int] | None:
    line = re.sub(r"^\\boxed\{(.+)\}$", r"\1", line.strip().strip("*$ "))
    line = line.replace("\\(", "").replace("\\)", "")
    line = re.sub(r"\s+(?:shells|coins|cards|points|tokens|marbles|stickers)\.?$", "", line).strip()
    try:
        tree = ast.parse(line, mode="eval").body
    except SyntaxError:
        return None

    def walk(node: ast.AST) -> tuple[int, int] | None:
        if isinstance(node, ast.Name) and node.id.lower() == "x":
            return 1, 0
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return 0, node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = walk(node.operand)
            return (-v[0], -v[1]) if v else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            l, r = walk(node.left), walk(node.right)
            if l is None or r is None:
                return None
            sign = 1 if isinstance(node.op, ast.Add) else -1
            return l[0] + sign * r[0], l[1] + sign * r[1]
        return None

    return walk(tree)


def score(item: dict, task: str, raw: str, finish_reason: str | None) -> dict:
    """Assign mutually exclusive outcome; truncation overrides partial finals."""
    if finish_reason == "length":
        return {"outcome": "truncated", "parsed": None, "correct": False}
    if "</think>" not in raw:
        return {"outcome": "unparseable", "parsed": None, "correct": False}
    content = raw.rsplit("</think>", 1)[-1]
    matches = list(FINAL.finditer(content))
    if matches:
        line = content[matches[-1].end():].split("\n", 1)[0].strip()
    else:
        line = content.strip().splitlines()[-1].strip() if content.strip() else ""
        if not ABSTAIN.fullmatch(line.rstrip(".")):
            return {"outcome": "unparseable", "parsed": None, "correct": False}
    line = line.replace("**", "").strip()
    if not line:
        return {"outcome": "unparseable", "parsed": None, "correct": False}
    if task == "answerability":
        match = re.match(r"(?i)^(yes|no)\b", line)
        if not match or re.search(r"(?i)\b(?:or|but|maybe)\s+(?:yes|no)\b", line):
            return {"outcome": "unparseable", "parsed": None, "correct": False}
        parsed = match.group(1).lower()
        gold = item["gold"] if item["gold"] in ("yes", "no") else (
            "yes" if item["gold"] != UNDETERMINED else "no")
        return {"outcome": "correct" if parsed == gold else "other_error",
                "parsed": parsed, "correct": parsed == gold}
    if re.search(r"(?i)\b(?:if|assuming|supposing|provided)\b", line):
        return {"outcome": "other_error", "parsed": line, "correct": False,
                "conditional": True}
    if ABSTAIN.search(line):
        if re.search(r"(?<![\w.])[-+]?\d+(?!\w|\.\d)", line):
            return {"outcome": "unparseable", "parsed": None, "correct": False}
        correct = item["gold"] == UNDETERMINED
        return {"outcome": "correct" if correct else "other_error",
                "parsed": UNDETERMINED, "correct": correct}
    affine = _affine(line)
    if affine is not None and affine[0] != 0:
        correct = (item["gold"] == UNDETERMINED and item.get("symbolic_gold") is not None
                   and affine == tuple(item["symbolic_gold"]))
        return {"outcome": "correct" if correct else "other_error",
                "parsed": line, "correct": correct}
    line = re.sub(r"^\$?\\boxed\{([+-]?\d+)\}\$?$", r"\1", line)
    match = NUMBER.fullmatch(line)
    if not match:
        return {"outcome": "unparseable", "parsed": None, "correct": False}
    parsed = int(match.group(1))
    correct = parsed == item["gold"]
    zero = (item["gold"] == UNDETERMINED and
            parsed == item["zero_default"])
    return {"outcome": "correct" if correct else "zero_default" if zero else "other_error",
            "parsed": parsed, "correct": correct}


def _items(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["item_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate benchmark items")
    return by_id


def score_file(items_path: Path, raw_path: Path, output_path: Path) -> Counter:
    if output_path.exists():
        raise FileExistsError(output_path)
    items = _items(items_path)
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("Empty raw result file")
    samples = records[0]["sampling"]["n"]
    expected_keys = {(item_id, task, placement, sample)
                     for item_id in items for task in ("solve", "answerability")
                     for placement in ("none", "user") for sample in range(samples)}
    keys = {(r["item_id"], r["task"], r["instruction_placement"], r["sample_index"])
            for r in records}
    if len(records) != len(expected_keys) or keys != expected_keys:
        raise ValueError(f"Incomplete, unexpected, or duplicated raw results: {len(records)}/{len(expected_keys)}")
    counts = Counter()
    with output_path.open("x", encoding="utf-8") as target:
        for record in records:
            item = items[record["item_id"]]
            scored = score(item, record["task"], record["raw_response"], record["finish_reason"])
            row = {**record, "scorer_version": SCORER_VERSION, **scored}
            counts[scored["outcome"]] += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(dict(score_file(args.items, args.raw, args.output)))


if __name__ == "__main__":
    main()
