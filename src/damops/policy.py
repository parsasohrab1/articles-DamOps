"""Release policies: a rule curve and stochastic dynamic programming (SDP)."""

from __future__ import annotations

import numpy as np

from .forecast import InflowForecaster
from .hydrology import Reservoir, step

SHORTAGE_W, SPILL_W, FLOOD_W = 1.0, 0.2, 5.0


def stage_cost(res: Reservoir, delivered: float, spill: float, storage: float, month: int) -> float:
    shortage = max(res.demand[month] - delivered, 0.0) / res.demand[month]
    flood = max(storage - res.flood_limit[month], 0.0) / res.capacity
    return SHORTAGE_W * shortage**2 * 100 + SPILL_W * spill / 10 + FLOOD_W * flood * 100


def rule_curve(res: Reservoir, storage: float, inflow: float, month: int) -> float:
    """Baseline: release the demand, release more above the flood limit, hold back below 25%."""
    target = res.demand[month]
    if storage + inflow > res.flood_limit[month]:
        target = max(target, storage + inflow - res.flood_limit[month])
    if storage < 0.25 * res.capacity:
        target = 0.7 * target
    return target


class SDPPolicy:
    def __init__(self, res: Reservoir, fc: InflowForecaster, n_states: int = 61,
                 n_actions: int = 31, n_scen: int = 9, gamma: float = 0.97):
        self.res, self.fc = res, fc
        self.grid = np.linspace(res.dead, res.capacity, n_states)
        self.actions = np.linspace(0.0, res.max_release, n_actions)
        # inflow scenarios: quantiles of the conditional lognormal (anomaly state ignored: climatological)
        qs = (np.arange(n_scen) + 0.5) / n_scen
        from statistics import NormalDist
        z = np.array([NormalDist().inv_cdf(q) for q in qs])
        self.scen = [np.exp(fc.mu[m] + fc.sd / np.sqrt(1 - fc.phi**2) * z) for m in range(12)]
        self.value = np.zeros((12, n_states))
        self.policy = np.zeros((12, n_states))
        self._solve(gamma)

    def _solve(self, gamma: float) -> None:
        res, grid = self.res, self.grid
        for _ in range(60):
            new = np.zeros_like(self.value)
            for m in range(12):
                nm = (m + 1) % 12
                for i, s in enumerate(grid):
                    best, best_a = np.inf, 0.0
                    for a in self.actions:
                        c = 0.0
                        for q in self.scen[m]:
                            s2, deliv, spill = step(res, s, q, a, m)
                            c += stage_cost(res, deliv, spill, s2, m) + gamma * np.interp(
                                s2, grid, self.value[nm])
                        c /= len(self.scen[m])
                        if c < best:
                            best, best_a = c, a
                    new[m, i], self.policy[m, i] = best, best_a
            if np.max(np.abs(new - self.value)) < 1e-3:
                self.value = new
                break
            self.value = new

    def release(self, storage: float, inflow: float, month: int) -> float:
        return float(np.interp(storage, self.grid, self.policy[month]))
