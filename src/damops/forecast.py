"""One-month-ahead inflow forecast with split-conformal intervals."""

from __future__ import annotations

import numpy as np


class InflowForecaster:
    """log inflow = monthly mean + AR(1) anomaly (fitted by least squares)."""

    def fit(self, inflow: np.ndarray) -> InflowForecaster:
        logq = np.log(np.maximum(inflow, 1e-6))
        self.mu = logq.mean(axis=0)
        anom = (logq - self.mu).ravel()
        den = float(anom[:-1] @ anom[:-1])
        phi = float(anom[:-1] @ anom[1:] / den) if den > 0.0 else 0.0
        self.phi = float(np.clip(phi, -0.99, 0.99))
        resid = anom[1:] - self.phi * anom[:-1]
        self.sd = float(resid.std())
        return self

    def predict_log(self, prev_anomaly: float, month: int) -> float:
        return float(self.mu[month] + self.phi * prev_anomaly)

    def anomaly(self, value: float, month: int) -> float:
        return float(np.log(max(float(value), 1e-6)) - self.mu[month])


def conformal_interval(fc: InflowForecaster, inflow: np.ndarray, alpha: float = 0.1) -> float:
    """Half-width (in log units) from absolute residuals on a calibration series."""
    flat = np.maximum(inflow.ravel(), 1e-6)
    res = []
    for i in range(1, len(flat)):
        m_prev, m = (i - 1) % 12, i % 12
        pred = fc.predict_log(fc.anomaly(flat[i - 1], m_prev), m)
        res.append(abs(np.log(flat[i]) - pred))
    res = np.sort(res)
    k = int(np.ceil((len(res) + 1) * (1 - alpha)))
    return float(res[min(k, len(res)) - 1])


def coverage(fc: InflowForecaster, inflow: np.ndarray, half_width: float) -> float:
    flat = np.maximum(inflow.ravel(), 1e-6)
    hit = 0
    for i in range(1, len(flat)):
        m_prev, m = (i - 1) % 12, i % 12
        pred = fc.predict_log(fc.anomaly(flat[i - 1], m_prev), m)
        hit += abs(np.log(flat[i]) - pred) <= half_width
    return hit / (len(flat) - 1)
