"""Reservoir operation study on synthetic inflow (not field data).

Usage: python research/run_study.py --out research/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from damops.evaluate import simulate
from damops.forecast import InflowForecaster, conformal_interval, coverage
from damops.hydrology import Reservoir, generate_inflow
from damops.policy import SDPPolicy, rule_curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/synthetic_results.json")
    a = ap.parse_args()
    res = Reservoir()
    train = generate_inflow(200, np.random.default_rng(1))
    fc = InflowForecaster().fit(train)
    hw = conformal_interval(fc, generate_inflow(200, np.random.default_rng(2)))
    pol = SDPPolicy(res, fc, n_states=41, n_actions=21, n_scen=7)
    out = {"note": "synthetic inflow; 500-year test series per setting",
           "forecast": {"phi": fc.phi, "sd_log": fc.sd, "conformal_half_width_log": hw}}
    settings = {"same_climate": 1.0, "drought_inflow_x0.7": 0.7, "wet_inflow_x1.3": 1.3}
    for name, f in settings.items():
        test = generate_inflow(500, np.random.default_rng(10)) * f
        out[name] = {
            "sdp": simulate(res, test, pol.release),
            "rule_curve": simulate(res, test, lambda s, q, m: rule_curve(res, s, q, m)),
            "forecast_coverage_target_0.90": coverage(fc, test, hw)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
