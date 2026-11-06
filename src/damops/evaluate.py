"""Simulate a policy on a synthetic inflow series."""

from __future__ import annotations

import numpy as np

from .hydrology import Reservoir, step
from .policy import stage_cost


def simulate(res: Reservoir, inflow: np.ndarray, release_fn, s0: float | None = None) -> dict:
    s = res.capacity * 0.6 if s0 is None else s0
    short_months, flood_months, spill_total, cost = 0, 0, 0.0, 0.0
    max_short, n = 0.0, 0
    for y in range(inflow.shape[0]):
        for m in range(12):
            q = inflow[y, m]
            a = release_fn(s, q, m)
            s, deliv, spill = step(res, s, q, a, m)
            short = max(res.demand[m] - deliv, 0.0)
            short_months += short > 1e-6
            max_short = max(max_short, short / res.demand[m])
            flood_months += s > res.flood_limit[m] + 1e-6
            spill_total += spill
            cost += stage_cost(res, deliv, spill, s, m)
            n += 1
    return {"reliability": 1.0 - short_months / n, "max_shortage_fraction": max_short,
            "flood_limit_violations_per_year": flood_months / (n / 12),
            "spill_hm3_per_year": spill_total / (n / 12), "cost_per_month": cost / n}


def simulate_decisions(res: Reservoir, inflow: np.ndarray, release_fn, s0: float,
                       prev_inflow: float, prev_month: int,
                       evaporation: np.ndarray | None = None,
                       observed_release: np.ndarray | None = None) -> dict:
    """Simulate a release rule that may use only the previous month's inflow.

    ``release_fn(storage, prev_inflow, prev_month, month) -> hm3`` does not receive
    the current month's inflow. ``evaporation`` is (years, 12) in hm3, or omitted.
    """
    years, _ = inflow.shape
    evap = np.zeros_like(inflow) if evaporation is None else np.asarray(evaporation, dtype=float)
    s = float(s0)
    short_months, flood_months, spill_total, cost = 0, 0, 0.0, 0.0
    max_short, n = 0.0, 0
    release_sum, mae_sum, clipped = 0.0, 0.0, 0
    for y in range(years):
        for m in range(12):
            requested = float(release_fn(s, prev_inflow, prev_month, m))
            requested = min(max(requested, 0.0), res.max_release)
            s, delivered, spill = step(res, s, float(inflow[y, m]), requested, m, float(evap[y, m]))
            short = max(res.demand[m] - delivered, 0.0)
            short_months += short > 1e-6
            max_short = max(max_short, short / res.demand[m])
            flood_months += s > res.flood_limit[m] + 1e-6
            spill_total += spill
            cost += stage_cost(res, delivered, spill, s, m)
            release_sum += delivered
            if abs(delivered - requested) > 1e-6:
                clipped += 1
            if observed_release is not None:
                mae_sum += abs(requested - float(observed_release[y, m]))
            prev_inflow = float(inflow[y, m])
            prev_month = m
            n += 1
    out = {"reliability": 1.0 - short_months / n, "max_shortage_fraction": max_short,
           "flood_limit_violations_per_year": flood_months / (n / 12),
           "spill_hm3_per_year": spill_total / (n / 12), "cost_per_month": cost / n,
           "mean_release_hm3": release_sum / n,
           "fraction_months_release_clipped": clipped / n}
    if observed_release is not None:
        out["release_mae_vs_observed"] = mae_sum / n
    return out
