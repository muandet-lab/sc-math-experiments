"""Audit frozen scorer against every available Case 5 pilot response."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from study.generate_shortcut_cases import UNDETERMINED
from study.score_case5_confirmatory import SCORER_VERSION, score


def audit(paths: list[Path]) -> dict:
    report = {"scorer_version": SCORER_VERSION, "files": {}, "disagreements": []}
    for path in paths:
        counts = Counter()
        incompatible = Counter()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"raw_response", "finish_reason", "answer"} <= row.keys():
                incompatible["missing required fields"] += 1
                continue
            task = row.get("task", "solve")
            if task not in ("solve", "answerability"):
                incompatible["unknown task"] += 1
                continue
            symbolic = row.get("symbolic_gold")
            item = {"gold": row["answer"], "symbolic_gold": symbolic,
                    "zero_default": symbolic[1] if symbolic is not None else None}
            result = score(item, task, row["raw_response"], row["finish_reason"])
            counts[result["outcome"]] += 1
            if "correct" in row and bool(row["correct"]) != result["correct"]:
                report["disagreements"].append({"file": path.name,
                                                  "line": line_number,
                                                  "item_id": row.get("item_id"),
                                                  "old": bool(row["correct"]),
                                                  "new": result["outcome"],
                                                  "final": row["raw_response"].rsplit("</think>", 1)[-1][-160:]})
        report["files"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "outcomes": dict(counts), "excluded": dict(incompatible)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("study"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for path in args.directory.glob("*case5*.jsonl")
                   if path.name not in ("case5_confirmatory_items.jsonl",
                                        "case5_confirmatory_prefill.jsonl"))
    paths += sorted((args.directory / "outputs").glob("*case5*.jsonl"))
    report = audit(paths)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audited {len(paths)} files; {len(report['disagreements'])} scoring disagreements")


if __name__ == "__main__":
    main()
