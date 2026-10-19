"""Reuse Case 5 confirmatory solve responses as Phase 1 missing_start raw rows.

The confirmatory run used the same prompts, revisions, and thinking-mode sampling,
so rerunning those cells would only spend GPU time. Every imported row is checked
against the missing_start item and the Phase 1 manifest before it is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from study.shortcut_rep_items import load_family
from study.shortcut_rep_run import LIMITS, MANIFEST, SEED, load_manifest

KEYS = {"olmo_7b": "olmo_7b_think", "olmo_32b": "olmo_32b_think"}


def import_case5(raw_path: Path, items_path: Path, output_path: Path,
                 manifest_path: Path = MANIFEST) -> int:
    if output_path.exists():
        raise FileExistsError(output_path)
    items = {item["item_id"]: item for item in load_family(items_path)}
    item_hash = hashlib.sha256(items_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = load_manifest(manifest_path)
    rows = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        if source["task"] != "solve":
            continue
        model_key = KEYS.get(source["model_key"], source["model_key"])
        model = manifest[model_key]
        settings = {"n": model["samples"], **model["modes"]["thinking"]["sampling"],
                    "max_tokens": LIMITS["thinking"][0], "seed": SEED}
        item = items[f"ms-{source['item_id']}-{source['instruction_placement']}"]
        if (source["revision"] != model["revision"] or source["sampling"] != settings
                or source["max_model_len"] != LIMITS["thinking"][1]
                or source["system_prompt"] != item["system"]
                or source["user_prompt"] != item["user"]):
            raise ValueError(f"Case 5 row does not match Phase 1 settings: {source['item_id']}")
        rows.append({**{k: v for k, v in source.items()
                        if k not in ("system_prompt", "user_prompt", "task",
                                     "instruction_placement", "benchmark_sha256",
                                     "model_manifest_sha256")},
                     "item_id": item["item_id"], "family": "missing_start", "cell": item["cell"],
                     "model_key": model_key, "mode": "thinking",
                     "benchmark_sha256": item_hash, "model_manifest_sha256": manifest_hash,
                     "messages": [{"role": "system", "content": item["system"]},
                                  {"role": "user", "content": item["user"]}],
                     "imported_from": {"path": raw_path.name, "model_key": source["model_key"],
                                       "benchmark_sha256": source["benchmark_sha256"],
                                       "model_manifest_sha256": source["model_manifest_sha256"]}})
    samples = rows[0]["sampling"]["n"] if rows else 0
    keys = {(row["item_id"], row["sample_index"]) for row in rows}
    if len(keys) != len(rows) or keys != {(i, n) for i in items for n in range(samples)}:
        raise ValueError("Case 5 solve rows do not cover the missing_start items exactly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Imported {import_case5(args.raw, args.items, args.output)} rows")


if __name__ == "__main__":
    main()
