"""Synthetic monthly inflow generator and reservoir mass balance (labelled synthetic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# mean monthly inflow (hm3), October..September (wet winter/spring, dry summer)
MEAN_INFLOW = np.array([20, 45, 80, 110, 140, 150, 120, 80, 40, 20, 12, 10], dtype=float)
LOG_SD, AR1 = 0.45, 0.4


@dataclass(frozen=True)
class Reservoir:
    capacity: float = 600.0          # hm3
    dead: float = 60.0
    flood_limit: np.ndarray = None    # per-month maximum storage during the flood season
    max_release: float = 150.0        # hm3 / month
    demand: np.ndarray = None         # hm3 / month

    def __post_init__(self):
        if self.flood_limit is None:
            fl = np.full(12, self.capacity)
            fl[3:7] = 0.8 * self.capacity            # Jan..Apr
            object.__setattr__(self, "flood_limit", fl)
        if self.demand is None:
            d = np.array([40, 35, 35, 35, 40, 50, 65, 80, 90, 90, 80, 60], dtype=float)
            object.__setattr__(self, "demand", d)


def generate_inflow(years: int, rng: np.random.Generator) -> np.ndarray:
    """(years, 12) lognormal inflow with AR(1) persistence in the log anomaly."""
    out = np.empty((years, 12))
    z = 0.0
    for y in range(years):
        for m in range(12):
            z = AR1 * z + np.sqrt(1 - AR1**2) * rng.normal()
            out[y, m] = MEAN_INFLOW[m] * np.exp(LOG_SD * z - 0.5 * LOG_SD**2)
    return out


def step(res: Reservoir, storage: float, inflow: float, release: float, month: int):
    """Mass balance. Returns (new storage, delivered, spill)."""
    release = min(release, res.max_release, max(storage + inflow - res.dead, 0.0))
    s = storage + inflow - release
    spill = max(s - res.capacity, 0.0)
    return min(s, res.capacity), release, spill
