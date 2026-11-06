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
                 n_actions: int = 31, n_scen: int = 9, gamma: float = 0.97,
                 evaporation: np.ndarray | None = None):
        self.res, self.fc = res, fc
        self.gamma = float(gamma)
        self.evap = (np.zeros(12) if evaporation is None
                     else np.asarray(evaporation, dtype=float).reshape(12))
        self.grid = np.linspace(res.dead, res.capacity, n_states)
        self.actions = np.linspace(0.0, res.max_release, n_actions)
        # inflow scenarios: quantiles of the marginal lognormal (anomaly state ignored)
        qs = (np.arange(n_scen) + 0.5) / n_scen
        from statistics import NormalDist
        z = np.array([NormalDist().inv_cdf(float(q)) for q in qs])
        scale = fc.sd / np.sqrt(1 - fc.phi**2)
        self.scen = [np.exp(fc.mu[m] + scale * z) for m in range(12)]
        self.value = np.zeros((12, n_states))
        self.policy = np.zeros((12, n_states))
        self._solve(self.gamma)

    def _solve(self, gamma: float) -> None:
        res, grid, actions = self.res, self.grid, self.actions
        for _ in range(60):
            new = np.zeros_like(self.value)
            for m in range(12):
                nm = (m + 1) % 12
                scen = np.asarray(self.scen[m], dtype=float)
                storage = grid[:, None, None]
                action = actions[None, :, None]
                inflow = scen[None, None, :]
                ev = float(self.evap[m])
                release = np.minimum(action, res.max_release)
                release = np.minimum(
                    release, np.maximum(storage + inflow - ev - res.dead, 0.0))
                s2 = storage + inflow - ev - release
                spill = np.maximum(s2 - res.capacity, 0.0)
                s2 = np.minimum(np.maximum(s2, res.dead), res.capacity)
                shortage = np.maximum(res.demand[m] - release, 0.0) / res.demand[m]
                flood = np.maximum(s2 - res.flood_limit[m], 0.0) / res.capacity
                cost = (SHORTAGE_W * shortage**2 * 100
                        + SPILL_W * spill / 10
                        + FLOOD_W * flood * 100)
                cont = np.interp(s2.ravel(), grid, self.value[nm]).reshape(s2.shape)
                total = (cost + gamma * cont).mean(axis=2)
                idx = np.argmin(total, axis=1)
                rows = np.arange(grid.size)
                new[m] = total[rows, idx]
                self.policy[m] = actions[idx]
            if np.max(np.abs(new - self.value)) < 1e-3:
                self.value = new
                break
            self.value = new

    def release(self, storage: float, inflow: float, month: int) -> float:
        return float(np.interp(storage, self.grid, self.policy[month]))


class ForecastReleasePolicy:
    """One-month release that recenters inflow scenarios on the log-AR(1) forecast.

    The current month's inflow is not an input. ``prev_inflow`` is last month's
    inflow. The continuation value is the climatological SDP value, so the
    forecast changes this month's scenarios only.
    """

    def __init__(self, sdp: SDPPolicy, n_scen: int | None = None):
        self.sdp = sdp
        n = len(sdp.scen[0]) if n_scen is None else n_scen
        qs = (np.arange(n) + 0.5) / n
        from statistics import NormalDist
        self.z = np.array([NormalDist().inv_cdf(float(q)) for q in qs])

    def scenarios(self, prev_inflow: float, prev_month: int, month: int) -> np.ndarray:
        fc = self.sdp.fc
        if prev_inflow is None or not np.isfinite(prev_inflow) or prev_inflow <= 0.0:
            anom = 0.0
        else:
            anom = fc.anomaly(prev_inflow, prev_month)
        mean = fc.predict_log(anom, month)
        return np.exp(mean + fc.sd * self.z)

    def release(self, storage: float, prev_inflow: float, prev_month: int, month: int) -> float:
        res, sdp = self.sdp.res, self.sdp
        scen = self.scenarios(prev_inflow, prev_month, month)
        ev = float(sdp.evap[month])
        nm = (month + 1) % 12
        best, best_a = np.inf, float(sdp.actions[0])
        for action in sdp.actions:
            total = 0.0
            for inflow in scen:
                s2, delivered, spill = step(res, storage, float(inflow), float(action), month, ev)
                total += stage_cost(res, delivered, spill, s2, month) + sdp.gamma * float(
                    np.interp(s2, sdp.grid, sdp.value[nm]))
            total /= len(scen)
            if total < best:
                best, best_a = total, float(action)
        return best_a
