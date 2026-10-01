"""Prepare blind trace coding sheets and calculate two-coder agreement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from study.run_case5_confirmatory import load_items, load_manifest

CODES = ("D1", "D2", "N", "C", "O")


def prepare(results: Path, output: Path, manifest: Path, items_path: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    rng = random.Random(20261001)
    items = {r["item_id"]: r for r in load_items(items_path)}
    mapping, entries = [], []
    for model in load_manifest(manifest):
        rows = [json.loads(line) for line in (results / f"{model}.scored.jsonl").read_text(encoding="utf-8").splitlines()]
        # Item number 00 is the omitted-initial, two-action factorial cell.
        failures = [r for r in rows if r["task"] == "solve" and
                    r["outcome"] == "zero_default" and r["item_id"].endswith("-00")]
        failures.sort(key=lambda r: (r["item_id"], r["instruction_placement"], r["sample_index"]))
        chosen = rng.sample(failures, min(20, len(failures)))
        for row in chosen:
            trace = row["raw_response"].split("</think>", 1)[0].replace("<think>", "").strip()
            token = hashlib.sha256(f"{model}:{row['item_id']}:{row['instruction_placement']}:{row['sample_index']}".encode()).hexdigest()[:16]
            mapping.append({"trace_id": token, "model": model, "item_id": row["item_id"],
                            "instruction": row["instruction_placement"],
                            "sample_index": row["sample_index"]})
            entries.append({"trace_id": token, "problem": items[row["item_id"]]["problem"],
                            "trace": trace, "primary_code": "",
                            "instruction_referenced": ""})
    rng.shuffle(entries)
    if not entries:
        raise ValueError("No omitted-start zero-default failures to code")
    output.mkdir(parents=True)
    for coder in ("coder_a", "coder_b"):
        with (output / f"{coder}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    with (output / "mapping.jsonl").open("w", encoding="utf-8") as handle:
        for row in mapping:
            handle.write(json.dumps(row) + "\n")
    print(f"Prepared {len(entries)} traces across {len(load_manifest(manifest))} models")


def _coded(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    codes = {r["trace_id"]: r["primary_code"].strip().upper() for r in rows}
    if len(codes) != len(rows) or any(code not in CODES for code in codes.values()):
        raise ValueError(f"Duplicate IDs or invalid/missing code in {path}")
    return codes


def agreement(a: Path, b: Path, mapping: Path) -> dict:
    left, right = _coded(a), _coded(b)
    if left.keys() != right.keys():
        raise ValueError("Coder sheets cover different traces")
    n = len(left)
    if not n:
        raise ValueError("No coded traces")
    pa = sum(left[k] == right[k] for k in left) / n
    lc, rc = Counter(left.values()), Counter(right.values())
    pe = sum(lc[code] * rc[code] for code in CODES) / n**2
    kappa = (pa - pe) / (1 - pe) if pe != 1 else None
    by_id = {r["trace_id"]: r for r in (json.loads(x) for x in mapping.read_text(encoding="utf-8").splitlines())}
    per_model = {}
    for model in sorted({r["model"] for r in by_id.values()}):
        ids = [k for k in left if by_id[k]["model"] == model]
        consensus = Counter(left[k] for k in ids if left[k] == right[k])
        dominant_default = (consensus["D1"] + consensus["D2"]) / len(ids) if ids else 0
        dominant_nonrepresentation = consensus["N"] / len(ids) if ids else 0
        label = ("default-pattern" if dominant_default >= 0.6 else
                 "nonrepresentation-pattern" if dominant_nonrepresentation >= 0.6 else "mixed-or-unresolved")
        per_model[model] = {"n": len(ids), "agreement": sum(left[k] == right[k] for k in ids),
                            "consensus_counts": dict(consensus), "provisional_label": label}
    return {"n": n, "observed_agreement": pa, "cohen_kappa": kappa,
            "disagreements": [k for k in left if left[k] != right[k]],
            "per_model": per_model,
            "note": "Labels are provisional until all disagreements are adjudicated."}


def finalized(adjudicated: Path, mapping: Path) -> dict:
    codes = _coded(adjudicated)
    by_id = {r["trace_id"]: r for r in (json.loads(x) for x in mapping.read_text(encoding="utf-8").splitlines())}
    if codes.keys() != by_id.keys():
        raise ValueError("Adjudicated sheet does not cover all traces")
    results = {}
    for model in sorted({r["model"] for r in by_id.values()}):
        ids = [k for k in codes if by_id[k]["model"] == model]
        counts = Counter(codes[k] for k in ids)
        default = (counts["D1"] + counts["D2"]) / len(ids) if ids else 0
        missing = counts["N"] / len(ids) if ids else 0
        results[model] = {"n": len(ids), "counts": dict(counts),
                          "label": "default-pattern" if default >= 0.6 else
                                   "nonrepresentation-pattern" if missing >= 0.6 else "mixed"}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--results", type=Path, required=True)
    prep.add_argument("--items", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--models", type=Path, default=Path(__file__).with_name("case5_confirmatory_models.json"))
    agree = sub.add_parser("agreement")
    agree.add_argument("--coder-a", type=Path, required=True)
    agree.add_argument("--coder-b", type=Path, required=True)
    agree.add_argument("--mapping", type=Path, required=True)
    agree.add_argument("--output", type=Path, required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--adjudicated", type=Path, required=True)
    finish.add_argument("--mapping", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.results, args.output, args.models, args.items)
    elif args.action == "agreement":
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(agreement(args.coder_a, args.coder_b, args.mapping), indent=2) + "\n")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(finalized(args.adjudicated, args.mapping), indent=2) + "\n")


if __name__ == "__main__":
    main()
