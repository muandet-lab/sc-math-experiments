"""Summarize complete frozen Case 5 runs and make the preregistered figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from study.generate_shortcut_cases import UNDETERMINED
from study.run_case5_confirmatory import PLACEMENTS, TASKS, load_items, load_manifest
from study.score_case5_confirmatory import SCORER_VERSION


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(rows: list[dict], predicate) -> float | None:
    return sum(bool(predicate(r)) for r in rows) / len(rows) if rows else None


def analyze(items_path: Path, models_path: Path, results_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    items = {r["item_id"]: r for r in load_items(items_path)}
    models = load_manifest(models_path)
    metrics, conditions, regression = {}, [], []
    for key, model in models.items():
        path = results_dir / f"{key}.scored.jsonl"
        probe_path = results_dir / f"{key}.prefill.jsonl"
        rows, probes = _read(path), _read(probe_path)
        expected_keys = {(item_id, task, placement, sample)
                         for item_id in items for task in TASKS for placement in PLACEMENTS
                         for sample in range(model["samples"])}
        unique = {(r["item_id"], r["task"], r["instruction_placement"], r["sample_index"])
                  for r in rows}
        if len(rows) != len(expected_keys) or unique != expected_keys or len(probes) != 72:
            raise ValueError(f"Incomplete or duplicated results for {key}")
        if len({p["probe_id"] for p in probes}) != 72:
            raise ValueError(f"Duplicate prefill probes for {key}")
        if any(r["scorer_version"] != SCORER_VERSION or r["revision"] != model["revision"] for r in rows):
            raise ValueError(f"Mixed scorer or checkpoint versions for {key}")
        if any(p["revision"] != model["revision"] for p in probes):
            raise ValueError(f"Mixed prefill checkpoint for {key}")
        for r in rows:
            r.update({k: items[r["item_id"]][k] for k in
                      ("kind", "slot", "salience", "chain_length", "gold", "zero_default")})
        solve = [r for r in rows if r["task"] == "solve"]
        challenge = [r for r in solve if r["kind"] == "factorial"]
        for r in (challenge + [r for r in solve if r["kind"] == "marked_pocket"]):
            regression.append({"model": key, "family": "qwen" if key.startswith("qwen") else "olmo",
                               "log_size": math.log(model["billions"]), "base_id": r["base_id"],
                               "item_id": r["item_id"], "sample_index": r["sample_index"],
                               "kind": r["kind"], "slot": r["slot"], "salience": r["salience"],
                               "chain_length": r["chain_length"],
                               "instruction": r["instruction_placement"],
                               "correct": int(r["correct"]),
                               "zero_default": int(r["outcome"] == "zero_default"),
                               "truncated": int(r["outcome"] == "truncated"),
                               "unparseable": int(r["outcome"] == "unparseable")})
        omitted = [r for r in challenge if r["slot"] == "initial" and r["salience"] == "implicit"]
        marked = [r for r in solve if r["kind"] == "marked_pocket"]
        controls = [r for r in solve if r["kind"] in ("complete", "zero_start", "net_change")]
        check = [r for r in rows if r["task"] == "answerability" and
                 items[r["item_id"]]["kind"] == "factorial" and
                 items[r["item_id"]]["slot"] == "initial" and
                 items[r["item_id"]]["salience"] == "implicit"]
        by_condition = defaultdict(list)
        for r in challenge:
            by_condition[(r["slot"], r["salience"], r["chain_length"],
                          r["instruction_placement"])].append(r)
        for (slot, salience, chain, placement), group in sorted(by_condition.items()):
            conditions.append({"model": key, "billions": model["billions"],
                               "slot": slot, "salience": salience,
                               "chain_length": chain, "instruction": placement,
                               "n": len(group),
                               "zero_default_rate": _mean(group, lambda r: r["outcome"] == "zero_default"),
                               "accuracy": _mean(group, lambda r: r["correct"]),
                               "truncation_rate": _mean(group, lambda r: r["outcome"] == "truncated"),
                               "unparseable_rate": _mean(group, lambda r: r["outcome"] == "unparseable")})

        def accuracy(group, placement):
            return _mean([r for r in group if r["instruction_placement"] == placement],
                         lambda r: r["correct"])

        implicit = [r for r in challenge if r["salience"] == "implicit"]
        explicit = [r for r in challenge if r["salience"] == "explicit"]
        missing_probe = [p for p in probes if p["condition"] == "omitted_initial"]
        explicit_probe = [p for p in probes if p["condition"] == "explicit_unknown"]
        metrics[key] = {
            "billions": model["billions"], "family": "qwen" if key.startswith("qwen") else "olmo",
            "zero_default_rate": _mean(challenge, lambda r: r["outcome"] == "zero_default"),
            "salience_slope_accuracy": accuracy(explicit, "none") - accuracy(implicit, "none"),
            "omitted_instruction_uplift": accuracy(omitted, "user") - accuracy(omitted, "none"),
            "marked_instruction_uplift": accuracy(marked, "user") - accuracy(marked, "none"),
            "instruction_dissociation": (accuracy(marked, "user") - accuracy(marked, "none") -
                                          accuracy(omitted, "user") + accuracy(omitted, "none")),
            "knows_but_defaults_gap": (accuracy(check, "none") - accuracy(omitted, "none")),
            "prefill_zero_vs_received_logodds_omitted": sum(p["log_odds_zero_vs_received"] for p in missing_probe) / len(missing_probe),
            "prefill_zero_vs_received_logodds_explicit": sum(p["log_odds_zero_vs_received"] for p in explicit_probe) / len(explicit_probe),
            "over_abstention_rate": _mean(controls, lambda r: r["parsed"] == UNDETERMINED),
            "truncation_rate": _mean(rows, lambda r: r["outcome"] == "truncated"),
            "unparseable_rate": _mean(rows, lambda r: r["outcome"] == "unparseable")}
    return metrics, conditions, regression


def figures(metrics: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = (
        ("zero_default_rate", "Zero-default error rate"),
        ("salience_slope_accuracy", "Explicit minus implicit accuracy"),
        ("instruction_dissociation", "Marked minus omitted instruction uplift"),
        ("knows_but_defaults_gap", "Answerability minus solving accuracy"),
        ("prefill_zero_vs_received_logodds_omitted", "Prefill log odds: zero vs received"),
        ("over_abstention_rate", "Over-abstention on controls"),
    )
    for measure, ylabel in measures:
        fig, ax = plt.subplots(figsize=(6, 4))
        qwen = sorted(((v["billions"], v[measure], k) for k, v in metrics.items()
                       if v["family"] == "qwen"))
        olmo = sorted(((v["billions"], v[measure], k) for k, v in metrics.items()
                       if v["family"] == "olmo"))
        ax.plot([x[0] for x in qwen], [x[1] for x in qwen], "o-", label="Qwen3")
        ax.scatter([x[0] for x in olmo], [x[1] for x in olmo], marker="s", label="OLMo 3")
        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters, log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{measure}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("case5_confirmatory_models.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, conditions, regression = analyze(args.items, args.models, args.results)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.output / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    with (args.output / "regression.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regression[0]))
        writer.writeheader()
        writer.writerows(regression)
    figures(metrics, args.output)
    print(f"Saved metrics, conditions, and six figures to {args.output}")


if __name__ == "__main__":
    main()
