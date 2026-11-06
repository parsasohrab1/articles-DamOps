import numpy as np
import pytest

from damops.evaluate import simulate_decisions
from damops.forecast import InflowForecaster
from damops.hydrology import Reservoir, generate_inflow, step
from damops.policy import ForecastReleasePolicy, SDPPolicy
from damops.resops import (
    MonthlySeries,
    aggregate_csv,
    days_in_month,
    infer_targets,
    longest_contiguous_run,
    rank_reservoirs,
    split_years,
)


def _csv_for_year(year: int, inflow_m3s: float = 10.0, drop_day: tuple[int, int] | None = None) -> str:
    lines = ["date,storage,inflow,outflow,elevation,evaporation"]
    for month in range(1, 13):
        dim = days_in_month(year, month)
        for day in range(1, dim + 1):
            if drop_day == (month, day):
                continue
            lines.append(
                f"{year}-{month:02d}-{day:02d},100,{inflow_m3s},8,10,0.01"
            )
    return "\n".join(lines) + "\n"


def test_days_in_month_leap_year():
    assert days_in_month(2000, 2) == 29
    assert days_in_month(2001, 2) == 28
    assert days_in_month(2001, 1) == 31


def test_complete_year_converts_cumecs_to_hm3():
    series = aggregate_csv(_csv_for_year(2001, inflow_m3s=10.0), dam_id=7)
    assert series.years.tolist() == [2001]
    assert series.inflow[0, 0] == pytest.approx(10.0 * 86400 * 31 / 1e6)
    assert series.outflow[0, 0] == pytest.approx(8.0 * 86400 * 31 / 1e6)
    assert series.storage[0, 0] == pytest.approx(100.0)
    assert series.evaporation[0, 0] == pytest.approx(0.01 * 31)


def test_missing_day_drops_the_year():
    series = aggregate_csv(_csv_for_year(2001, drop_day=(1, 15)), dam_id=7)
    assert series.years.size == 0


def test_rank_keeps_each_runs_own_length():
    def make(dam: int, years: list[int]) -> MonthlySeries:
        count = len(years)
        ones = np.ones((count, 12))
        return MonthlySeries(
            dam, np.array(years), ones, ones, ones * 100.0, np.full((count, 12), np.nan),
        )

    ranked = rank_reservoirs([
        make(2, list(range(1980, 2010))),
        make(1, list(range(1990, 2005))),
        make(3, list(range(2000, 2022))),
    ], min_years=20)
    assert [(series.dam_id, length) for _start, length, series in ranked] == [(2, 30), (3, 22)]
    years = np.array([1990, 1991, 1993, 1994, 1995, 1996])
    start, length = longest_contiguous_run(years)
    assert (start, length) == (2, 4)
    block = np.arange(2000, 2020)
    fit, cal, test = split_years(block)
    assert fit.tolist() == list(range(2000, 2012))
    assert cal.tolist() == list(range(2012, 2015))
    assert test.tolist() == list(range(2015, 2020))


def test_demand_ignores_years_outside_fit_and_calibration():
    years = np.arange(2000, 2020)
    inflow = np.ones((20, 12)) * 10.0
    outflow = np.ones((20, 12)) * 4.0
    outflow[13:] = 80.0
    storage = np.ones((20, 12)) * 100.0
    series = MonthlySeries(
        1, years, inflow, outflow, storage, np.full((20, 12), np.nan),
    )
    spec = infer_targets(series, years[:10], years[10:13])
    assert spec["reservoir"].demand.max() == pytest.approx(4.0)
    assert spec["capacity_source"] == "observed_max_storage"


def test_evaporation_is_removed_before_spill():
    res = Reservoir()
    storage, delivered, spill = step(res, 590.0, 100.0, 0.0, 0, evaporation=30.0)
    assert delivered == 0.0
    assert spill == pytest.approx(60.0)
    assert storage == res.capacity


def test_conditional_scenarios_follow_the_forecast():
    res = Reservoir()
    inflow = generate_inflow(80, np.random.default_rng(8))
    forecaster = InflowForecaster().fit(inflow)
    policy = SDPPolicy(res, forecaster, n_states=15, n_actions=15, n_scen=5)
    forecast = ForecastReleasePolicy(policy)
    wet = forecast.scenarios(400.0, 0, 1).mean()
    dry = forecast.scenarios(1.0, 0, 1).mean()
    assert forecaster.phi > 0.0
    assert wet > dry
    release = forecast.release(300.0, 20.0, 0, 1)
    assert 0.0 <= release <= res.max_release


def test_decision_simulation_does_not_need_the_current_inflow_to_call_back():
    res = Reservoir(capacity=100.0, dead=0.0, flood_limit=np.full(12, 100.0),
                    max_release=50.0, demand=np.full(12, 10.0))
    inflow = np.full((2, 12), 12.0)
    observed = np.full((2, 12), 10.0)
    seen = []

    def release(storage, prev_inflow, prev_month, month):
        seen.append((prev_month, month, prev_inflow))
        del storage
        return 10.0

    out = simulate_decisions(res, inflow, release, s0=50.0, prev_inflow=12.0, prev_month=11,
                             observed_release=observed)
    assert seen[0] == (11, 0, 12.0)
    assert seen[1][1] == 1 and seen[1][2] == pytest.approx(12.0)
    assert out["release_mae_vs_observed"] == pytest.approx(0.0)
    assert out["reliability"] == pytest.approx(1.0)
