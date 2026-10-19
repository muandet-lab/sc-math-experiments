"""Frozen Phase 1 shortcut-replication scorer, version 1. Do not edit after model runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from study.score_case5_confirmatory import score as score_case5
from study.shortcut_rep_items import load_family

SCORER_VERSION = "shortcut-rep-v1"
FINAL = re.compile(r"(?i)\bfinal\s+answer(?:\*\*)?\s*[:：]\s*")
INTEGER = re.compile(r"(?:[A-Za-z' ]+=\s*)?([+-]?\d{1,3}(?:,\d{3})+|[+-]?\d+)(?:\s+[A-Za-z]+)?\.?")
LETTER = re.compile(r"(?:option\s+)?\(?([A-D])\)?(?:[.):]?(?:\s.*)?)?", re.I)


def _strip(line: str) -> str:
    line = line.replace("**", "").strip().strip("$ ").strip()
    line = re.sub(r"^\\boxed\{(.*)\}\.?$", r"\1", line).strip()
    return re.sub(r"^\\text\{(.*)\}$", r"\1", line).strip()


def _final_line(raw: str, mode: str) -> str | None:
    """Return the text after the last final-answer marker outside any thinking block."""
    if "</think>" in raw:
        content = raw.rsplit("</think>", 1)[-1]
    elif mode == "thinking":
        return None
    else:
        content = raw
    matches = list(FINAL.finditer(content))
    if not matches:
        return None
    line = _strip(content[matches[-1].end():].split("\n", 1)[0])
    return line or None


def score(item: dict, raw: str, finish_reason: str | None, mode: str) -> dict:
    """Assign one mutually exclusive outcome; truncation overrides partial answers."""
    if finish_reason == "length":
        return {"outcome": "truncated", "parsed": None, "correct": False}
    if item["answer_type"] == "case5":
        # The frozen Case 5 scorer expects a closed thinking block.
        text = raw if "</think>" in raw or mode == "thinking" else "</think>" + raw
        result = score_case5(item["case5"], "solve", text, finish_reason)
        outcome = "signature" if result["outcome"] == "zero_default" else result["outcome"]
        return {**result, "outcome": outcome}
    line = _final_line(raw, mode)
    if line is None:
        return {"outcome": "unparseable", "parsed": None, "correct": False}
    if item["answer_type"] == "letter":
        match = LETTER.fullmatch(line)
        if not match:
            return {"outcome": "unparseable", "parsed": None, "correct": False}
        parsed = match.group(1).upper()
        return {"outcome": "correct" if parsed == item["gold"] else "other_error",
                "parsed": parsed, "chosen_index": "ABCD".index(parsed),
                "correct": parsed == item["gold"]}
    if item["answer_type"] == "option_text":
        normalized = [option.strip().lower() for option in item["options"]]
        text = line.strip().rstrip(".").lstrip("- ").strip().lower()
        if text not in normalized:
            return {"outcome": "unparseable", "parsed": None, "correct": False}
        index = normalized.index(text)
        correct = index == item["gold_index"]
        return {"outcome": "correct" if correct else "other_error", "parsed": line,
                "chosen_index": index, "correct": correct}
    match = INTEGER.fullmatch(line)
    if not match:
        return {"outcome": "unparseable", "parsed": None, "correct": False}
    parsed = int(match.group(1).replace(",", ""))
    if parsed == item["gold"]:
        outcome = "correct"
    elif item["signature"] is not None and parsed == item["signature"]:
        outcome = "signature"
    else:
        outcome = "other_error"
    return {"outcome": outcome, "parsed": parsed, "correct": outcome == "correct"}


def score_file(items_path: Path, raw_path: Path, output_path: Path) -> Counter:
    if output_path.exists():
        raise FileExistsError(output_path)
    items = {item["item_id"]: item for item in load_family(items_path)}
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError("Empty raw result file")
    samples = records[0]["sampling"]["n"]
    expected = {(item_id, n) for item_id in items for n in range(samples)}
    keys = [(r["item_id"], r["sample_index"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Incomplete, unexpected, or duplicated raw results: "
                         f"{len(set(keys) & expected)}/{len(expected)}")
    if len({(r["model_key"], r["mode"], r["revision"], r["benchmark_sha256"])
            for r in records}) != 1:
        raise ValueError("Raw file mixes runs")
    counts = Counter()
    with output_path.open("x", encoding="utf-8") as target:
        for record in records:
            item = items[record["item_id"]]
            scored = score(item, record["raw_response"], record["finish_reason"], record["mode"])
            row = {**record, "family": item["family"], "cell": item["cell"],
                   "scorer_version": SCORER_VERSION, **scored}
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
