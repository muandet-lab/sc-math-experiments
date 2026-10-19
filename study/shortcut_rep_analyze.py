"""Compute Phase 1 replication metrics with base-level bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from study.shortcut_rep_items import load_family

REPLICATES, SEED = 2000, 20261014
OUTCOMES = ("correct", "signature", "other_error", "truncated", "unparseable")


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def aggregate(rows: list[dict], items: dict[str, dict]) -> dict:
    """Return counts[base_id][cell] for one model, mode, and family."""
    counts: dict = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        item = items[row["item_id"]]
        c = counts[item["base_id"]][item["cell"]]
        c["n"] += 1
        c[row["outcome"]] += 1
        if item["signature"] is not None:
            c["n_sig"] += 1
            c["correct_sig"] += row["outcome"] == "correct"
            c["errors_sig"] += row["outcome"] in ("signature", "other_error")
        amount = item.get("distractor_amount")
        if amount is not None and isinstance(row.get("parsed"), int):
            c["minus"] += row["parsed"] == item["gold"] - amount
            c["plus"] += row["parsed"] == item["gold"] + amount
        if row.get("chosen_index") is not None:
            c[f"chose_{row['chosen_index']}"] += 1
    return counts


def _pool(counts: dict, bases: list, cell: str, capable: set | None = None) -> Counter:
    total = Counter()
    for base in bases:
        if capable is None or base in capable:
            total.update(counts[base].get(cell, Counter()))
    return total


def _acc(c: Counter) -> float | None:
    return _rate(c["correct"], c["n"])


def keyword_metrics(counts: dict, bases: list) -> dict:
    cells = {name: _pool(counts, bases, name) for name in (
        "verbal_consistent", "verbal_inconsistent", "symbolic_direct",
        "symbolic_reversed", "arithmetic")}
    cate_v = _diff(_acc(cells["verbal_consistent"]), _acc(cells["verbal_inconsistent"]))
    cate_s = _diff(_acc(cells["symbolic_direct"]), _acc(cells["symbolic_reversed"]))
    capable = {b for b in set(bases)
               if (_acc(counts[b].get("arithmetic", Counter())) or 0) >= 0.5}

    def share(restrict):
        vi = _pool(counts, bases, "verbal_inconsistent", restrict)
        sr = _pool(counts, bases, "symbolic_reversed", restrict)
        excess = _diff(_rate(vi["signature"], vi["n_sig"]), _rate(sr["signature"], sr["n_sig"]))
        failure = _rate(vi["n_sig"] - vi["correct_sig"], vi["n_sig"])
        return None if excess is None or not failure else excess / failure

    vi, sr = cells["verbal_inconsistent"], cells["symbolic_reversed"]
    return {**{f"accuracy:{k}": _acc(v) for k, v in cells.items()},
            "cate_verbal": cate_v, "cate_symbolic": cate_s, "delta_kw": _diff(cate_v, cate_s),
            "cae_verbal": _rate(vi["signature"], vi["errors_sig"]),
            "cae_symbolic": _rate(sr["signature"], sr["errors_sig"]),
            "attributable_share": share(None), "attributable_share_capable": share(capable)}


def opedal_metrics(counts: dict, bases: list) -> dict:
    con, inc = (_acc(_pool(counts, bases, c)) for c in ("consistent", "inconsistent"))
    return {"accuracy:consistent": con, "accuracy:inconsistent": inc,
            "cate_verbal": _diff(con, inc)}


def distractor_metrics(counts: dict, bases: list) -> dict:
    cells = {name: _pool(counts, bases, name) for name in (
        "none", "plain", "keyword_fewer", "keyword_more", "arithmetic")}
    fewer, more, plain = cells["keyword_fewer"], cells["keyword_more"], cells["plain"]
    keyword_acc = _rate(fewer["correct"] + more["correct"], fewer["n"] + more["n"])
    direction = None
    if fewer["n"] and more["n"]:
        direction = ((fewer["minus"] / fewer["n"] - more["minus"] / more["n"]) +
                     (more["plus"] / more["n"] - fewer["plus"] / fewer["n"])) / 2
    keyword_sig = _rate(fewer["minus"] + more["plus"], fewer["n"] + more["n"])
    plain_same = _rate(plain["minus"] + plain["plus"], 2 * plain["n"])
    failure = None if keyword_acc is None else 1 - keyword_acc
    excess = _diff(keyword_sig, plain_same)
    return {**{f"accuracy:{k}": _acc(v) for k, v in cells.items()},
            "plain_cost": _diff(_acc(cells["none"]), _acc(plain)),
            "keyword_cost": _diff(_acc(plain), keyword_acc),
            "direction_effect": direction,
            "attributable_share": None if excess is None or not failure else excess / failure}


def option_metrics(counts: dict, bases: list) -> dict:
    out = {}
    cells = sorted({cell for base in set(bases) for cell in counts[base]})
    for prefix in sorted({cell.rsplit(":", 1)[0] for cell in cells}):
        recalls = [_acc(_pool(counts, bases, f"{prefix}:{letter}")) for letter in "ABCD"]
        if any(r is None for r in recalls):
            continue
        mean = sum(recalls) / 4
        out[f"rstd:{prefix}"] = 100 * (sum((r - mean) ** 2 for r in recalls) / 4) ** 0.5
        out[f"accuracy:{prefix}"] = mean
        chosen = Counter()
        for letter in "ABCD":
            chosen.update(_pool(counts, bases, f"{prefix}:{letter}"))
        answered = sum(chosen[f"chose_{i}"] for i in range(4))
        for i, letter in enumerate("ABCD"):
            out[f"choice_share:{prefix}:{letter}"] = _rate(chosen[f"chose_{i}"], answered)
        for position, letter in enumerate("ABCD"):
            out[f"recall:{prefix}:{letter}"] = recalls[position]
    # Token-ID bias: the same arithmetic questions with and without letter IDs.
    out["rstd_ids_minus_no_ids:arithmetic"] = _diff(out.get("rstd:arithmetic:ids"),
                                                    out.get("rstd:arithmetic:no_ids"))
    return out


def missing_metrics(counts: dict, bases: list) -> dict:
    out = {}
    for cell in sorted({cell for base in set(bases) for cell in counts[base]}):
        c = _pool(counts, bases, cell)
        out[f"accuracy:{cell}"] = _acc(c)
        out[f"signature_rate:{cell}"] = _rate(c["signature"], c["n"])
    return out


METRICS = {"keyword": keyword_metrics, "opedal_anchor": opedal_metrics,
           "distractor": distractor_metrics, "option_position": option_metrics,
           "missing_start": missing_metrics}
HEADLINE = {"keyword": ("delta_kw", "cate_verbal", "cae_verbal", "attributable_share"),
            "opedal_anchor": ("cate_verbal",),
            "distractor": ("keyword_cost", "direction_effect", "attributable_share"),
            "option_position": ("rstd:arithmetic:ids", "rstd:arithmetic:no_ids", "rstd:mmlu:ids",
                                "rstd_ids_minus_no_ids:arithmetic"),
            "missing_start": ("signature_rate:factorial:initial:implicit:none",
                              "signature_rate:factorial:initial:implicit:user")}


def _interval(values: list) -> tuple:
    values = sorted(v for v in values if v is not None)
    if len(values) < 0.9 * REPLICATES:
        return None, None
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def bootstrap(fn, counts: dict, bases: list, rng: random.Random) -> list[dict]:
    """One resample list per replicate, shared across modes for paired contrasts."""
    return [fn(counts, [rng.choice(bases) for _ in bases]) for _ in range(REPLICATES)]


def analyze(scored_paths: list[Path], items_dir: Path) -> list[dict]:
    groups = defaultdict(list)
    for path in scored_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                groups[(row["model_key"], row["mode"], row["family"])].append(row)
    item_cache, results, resamples = {}, [], {}
    for (model_key, mode, family), rows in sorted(groups.items()):
        if family not in item_cache:
            item_cache[family] = {i["item_id"]: i for i in load_family(items_dir / f"{family}.jsonl")}
        items = item_cache[family]
        counts = aggregate(rows, items)
        bases = sorted(counts)
        estimate = METRICS[family](counts, bases)
        # The same seed per family gives identical resamples of base ids across modes.
        draws = bootstrap(METRICS[family], counts, bases,
                          random.Random(f"{SEED}:{family}:{len(bases)}"))
        resamples[(model_key, mode, family)] = draws
        for metric, value in estimate.items():
            low, high = _interval([d.get(metric) for d in draws])
            results.append({"model_key": model_key, "mode": mode, "family": family,
                            "metric": metric, "estimate": value, "ci_low": low,
                            "ci_high": high, "n_bases": len(bases), "n_rows": len(rows)})
        totals = Counter(row["outcome"] for row in rows)
        for outcome in OUTCOMES:
            results.append({"model_key": model_key, "mode": mode, "family": family,
                            "metric": f"outcome_share:{outcome}",
                            "estimate": totals[outcome] / len(rows), "ci_low": None,
                            "ci_high": None, "n_bases": len(bases), "n_rows": len(rows)})
    for (model_key, mode, family), draws in resamples.items():
        other = resamples.get((model_key, "non_thinking", family))
        if mode != "thinking" or other is None:
            continue
        point = {r["metric"]: r["estimate"] for r in results
                 if (r["model_key"], r["family"]) == (model_key, family)
                 and r["mode"] == "thinking"}
        point_off = {r["metric"]: r["estimate"] for r in results
                     if (r["model_key"], r["family"]) == (model_key, family)
                     and r["mode"] == "non_thinking"}
        for metric in HEADLINE[family]:
            low, high = _interval([_diff(a.get(metric), b.get(metric))
                                   for a, b in zip(draws, other)])
            results.append({"model_key": model_key, "mode": "thinking_minus_non_thinking",
                            "family": family, "metric": metric,
                            "estimate": _diff(point.get(metric), point_off.get(metric)),
                            "ci_low": low, "ci_high": high,
                            "n_bases": None, "n_rows": None})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored", nargs="+", type=Path)
    parser.add_argument("--items-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    results = analyze(args.scored, args.items_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        if r["metric"] in HEADLINE[r["family"]] and r["estimate"] is not None:
            interval = ("" if r["ci_low"] is None
                        else f" [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
            print(f"{r['model_key']:<20} {r['mode']:<28} {r['family']:<16} "
                  f"{r['metric']:<48} {r['estimate']:.3f}{interval}")


if __name__ == "__main__":
    main()
