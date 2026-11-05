import numpy as np
import pytest

from damops.evaluate import simulate
from damops.forecast import InflowForecaster, conformal_interval, coverage
from damops.hydrology import Reservoir, generate_inflow, step
from damops.policy import SDPPolicy, rule_curve


def test_mass_balance_conserves_water():
    res = Reservoir()
    s, deliv, spill = step(res, 300.0, 100.0, 80.0, 3)
    assert s + deliv + spill == pytest.approx(400.0)


def test_release_never_below_dead_storage():
    res = Reservoir()
    s, deliv, _ = step(res, 70.0, 0.0, 100.0, 0)
    assert s >= res.dead - 1e-9 and deliv == pytest.approx(10.0)


def test_spill_when_full():
    res = Reservoir()
    s, _, spill = step(res, 590.0, 100.0, 0.0, 0)
    assert s == res.capacity and spill == pytest.approx(90.0)


def test_inflow_generator_seeded_and_positive():
    a = generate_inflow(5, np.random.default_rng(1))
    b = generate_inflow(5, np.random.default_rng(1))
    assert (a == b).all() and (a > 0).all() and a.shape == (5, 12)


def test_forecaster_recovers_persistence():
    q = generate_inflow(400, np.random.default_rng(2))
    fc = InflowForecaster().fit(q)
    assert 0.25 < fc.phi < 0.55


def test_conformal_interval_covers_on_new_data():
    fc = InflowForecaster().fit(generate_inflow(200, np.random.default_rng(3)))
    hw = conformal_interval(fc, generate_inflow(200, np.random.default_rng(4)))
    cov = coverage(fc, generate_inflow(200, np.random.default_rng(5)), hw)
    assert 0.85 <= cov <= 0.95


def test_sdp_beats_rule_curve_on_cost():
    res = Reservoir()
    fc = InflowForecaster().fit(generate_inflow(200, np.random.default_rng(6)))
    pol = SDPPolicy(res, fc, n_states=21, n_actions=13, n_scen=5)
    test = generate_inflow(100, np.random.default_rng(7))
    sdp = simulate(res, test, pol.release)
    rule = simulate(res, test, lambda s, q, m: rule_curve(res, s, q, m))
    assert sdp["cost_per_month"] <= rule["cost_per_month"]
