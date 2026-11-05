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
