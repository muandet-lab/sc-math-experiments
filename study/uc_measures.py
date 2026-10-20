"""Uncertainty measures for elicited beliefs over an ordered numeric grid.

Two representations, following Yang, Muandet et al. (UAI 2026):
  probability intervals  [l(y), u(y)]  (ProbInt), credal set C = {p : l <= p <= u, sum p = 1}
  possibility function   pi(y)          (Pos),     credal set {p : P(A) <= max_{y in A} pi(y)}

MMI_TV is the maximum mean imprecision with total variation (Chau, Caprio & Muandet), i.e.
max over events A of upper(A) - lower(A), which equals the TV diameter of the credal set.
  * Pos: the second-largest normalized possibility (Yang et al., App. B.3, Eq. 8).
  * ProbInt: Yang et al. use the bound 1 - sum l. We also compute it exactly. For reachable
    intervals, upper(A) - lower(A) = min(w_A, W - w_A, 1 - sum l, sum u - 1), where w_A is the
    summed width over A and W the total width; the maximum over A is a partition problem.
    This closed form is our derivation (from the de Campos et al. 1994 envelope formulas);
    test_uc checks it against vertex enumeration.

Wasserstein diameter (W1 on the grid, unit spacing, tail bin one step past the last value):
max over p, q in C of sum_k |F_p(k) - F_q(k)|. For both representations a single member attains
the upper CDF bound at every k and another the lower bound, so the diameter is
sum_k (F_upper(k) - F_lower(k)).
"""

from __future__ import annotations

import math


def reachable(lower: list[float], upper: list[float]) -> tuple[list[float], list[float]]:
    """Tighten intervals so every bound is attained by some member of the credal set."""
    sl, su = sum(lower), sum(upper)
    if sl > 1 + 1e-9 or su < 1 - 1e-9 or any(l > u + 1e-12 for l, u in zip(lower, upper)):
        raise ValueError("Empty credal set")
    new_l = [max(l, 1 - (su - u)) for l, u in zip(lower, upper)]
    new_u = [min(u, 1 - (sl - l)) for l, u in zip(lower, upper)]
    return new_l, new_u


def mmi_bound_intervals(lower: list[float]) -> float:
    return max(0.0, 1 - sum(lower))


def _best_partition(widths: list[float], resolution: float = 1e-5) -> float:
    """max over subsets A of min(w_A, W - w_A), by subset-sum on a fine integer grid."""
    units = [round(w / resolution) for w in widths if w > 0]
    total = sum(units)
    reach = 1
    for u in units:
        reach |= reach << u
    half = total // 2
    mask = reach & ((1 << (half + 1)) - 1)
    best = mask.bit_length() - 1
    return best * resolution


def mmi_intervals(lower: list[float], upper: list[float]) -> float:
    lo, up = reachable(lower, upper)
    caps = min(1 - sum(lo), sum(up) - 1)
    widths = [u - l for l, u in zip(lo, up)]
    if caps <= 0:
        return 0.0
    return max(0.0, min(caps, _best_partition(widths)))


def normalize_possibility(pi: list[float]) -> list[float]:
    top = max(pi)
    if top <= 0:
        raise ValueError("Possibility function is identically zero")
    return [p / top for p in pi]


def mmi_possibility(pi: list[float]) -> float:
    ordered = sorted(normalize_possibility(pi), reverse=True)
    return ordered[1] if len(ordered) > 1 else 0.0


def cdf_bounds_intervals(lower: list[float], upper: list[float]) -> tuple[list[float], list[float]]:
    lo, up = reachable(lower, upper)
    n = len(lo)
    f_up, f_low = [], []
    for k in range(n):
        left_l, left_u = sum(lo[:k + 1]), sum(up[:k + 1])
        right_l, right_u = sum(lo[k + 1:]), sum(up[k + 1:])
        f_up.append(min(left_u, 1 - right_l))
        f_low.append(max(left_l, 1 - right_u))
    return f_low, f_up


def cdf_bounds_possibility(pi: list[float]) -> tuple[list[float], list[float]]:
    p = normalize_possibility(pi)
    f_up = [max(p[:k + 1]) for k in range(len(p))]
    f_low = [1 - (max(p[k + 1:]) if k + 1 < len(p) else 0.0) for k in range(len(p))]
    return f_low, f_up


def wasserstein_diameter(f_low: list[float], f_up: list[float]) -> float:
    # The last CDF value is 1 for every member, so it contributes nothing.
    return sum(max(0.0, u - l) for l, u in zip(f_low[:-1], f_up[:-1]))


def support(scores: list[float], labels: list[str], eps: float = 1e-9) -> list[str]:
    return [y for y, s in zip(labels, scores) if s > eps]


def entropy(counts: dict) -> float:
    """Shannon entropy (bits) of an empirical answer distribution."""
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else float("nan")


def summarize(fmt: str, profile: dict, labels: list[str], true_set: list[str],
              own_answer: str | None = None) -> dict:
    """All per-response measures for one valid elicited profile."""
    true = set(true_set)
    if fmt == "probint":
        lower = [profile[y][0] for y in labels]
        upper = [profile[y][1] for y in labels]
        lo, up = reachable(lower, upper)
        f_low, f_up = cdf_bounds_intervals(lower, upper)
        supp = support(up, labels)
        result = {"mmi": mmi_intervals(lower, upper), "mmi_bound": mmi_bound_intervals(lo),
                  "lower_sum": sum(lower), "upper_sum": sum(upper),
                  "true_mass_lower": sum(l for y, l in zip(labels, lo) if y in true),
                  "true_mass_upper": min(1.0, sum(u for y, u in zip(labels, up) if y in true))}
        top_scores = up
    elif fmt == "pos":
        pi = [profile[y] for y in labels]
        norm = normalize_possibility(pi)
        f_low, f_up = cdf_bounds_possibility(pi)
        supp = support(norm, labels)
        result = {"mmi": mmi_possibility(pi),
                  "necessity_true": 1 - max((p for y, p in zip(labels, norm) if y not in true),
                                            default=0.0),
                  "possibility_true": max((p for y, p in zip(labels, norm) if y in true),
                                          default=0.0)}
        top_scores = norm
    else:
        raise ValueError(fmt)
    supp_set = set(supp)
    top = max(top_scores)
    result.update({
        "set_size": len(supp), "wasserstein_diameter": wasserstein_diameter(f_low, f_up),
        "contains_true": true <= supp_set, "within_true": supp_set <= true,
        "jaccard": len(true & supp_set) / len(true | supp_set),
        "all_ones": all(s >= top - 1e-9 for s in top_scores),
        "echo": (own_answer is not None and own_answer in labels and
                 supp_set == {own_answer})})
    return result
