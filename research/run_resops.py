"""Compare a forecast release policy with historical releases on ResOpsUS.

The zip is Zenodo record 5367383 (md5 da9292138ba8dd4bd45071f84eb6bf48).
It is downloaded into data/ResOpsUS.zip, which is gitignored. Metrics are written
only from this run.

Usage: python research/run_resops.py --out research/results.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from damops.evaluate import simulate_decisions
from damops.forecast import InflowForecaster, conformal_interval, coverage
from damops.policy import ForecastReleasePolicy, SDPPolicy
from damops.resops import (
    MAX_FLOW_M3S,
    MAX_STORAGE_HM3,
    closure_residual,
    infer_targets,
    iter_series_members,
    load_attributes,
    load_member,
    rank_reservoirs,
    slice_series,
    split_years,
)

ZENODO_RECORD = "5367383"
ZENODO_DOI = "10.5281/zenodo.5367383"
ZENODO_MD5 = "da9292138ba8dd4bd45071f84eb6bf48"
ZENODO_URL = "https://zenodo.org/records/5367383/files/ResOpsUS.zip?download=1"
N_STATES, N_ACTIONS, N_SCEN, GAMMA = 41, 21, 7, 0.97
MARK_START, MARK_END = "<!-- RESOPS:START -->", "<!-- RESOPS:END -->"

STANDING_LIMITATIONS = [
    "The supply target is the calendar-month median release in the fit and calibration years, not a recorded water demand.",
    "The flood limit is the calendar-month 90th percentile of end-of-month storage in the fit and calibration years, so it describes past storage rather than a published flood pool.",
    "Cost weights are the synthetic-study weights (squared relative shortage 1, spill 0.2, flood-limit exceedance 5), not weights supplied by an operator.",
    "The SDP continuation value does not carry the latest inflow anomaly. The forecast only recenters the current month's inflow scenarios.",
    "Both policies are scored inside a single-reservoir mass balance. Historical releases can be clipped when that balance cannot supply them.",
    "TRL 5 is not claimed: this is an offline replay, and the baseline is the observed release series rather than a published operating rule.",
]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_zip(path: Path) -> None:
    if path.exists() and _md5(path) == ZENODO_MD5:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ZENODO_URL}")
    urlretrieve(ZENODO_URL, path)
    got = _md5(path)
    if got != ZENODO_MD5:
        raise SystemExit(f"ResOpsUS checksum mismatch: {got} != {ZENODO_MD5}")


def _table_release(table: np.ndarray):
    flat = np.asarray(table, dtype=float).ravel()
    state = {"i": 0}

    def release(_storage, _prev_inflow, _prev_month, _month) -> float:
        value = float(flat[state["i"]])
        state["i"] += 1
        return value

    return release


def _builtin(value):
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _builtin(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metrics(block: dict) -> dict:
    keys = (
        "reliability", "max_shortage_fraction", "flood_limit_violations_per_year",
        "spill_hm3_per_year", "cost_per_month", "mean_release_hm3",
        "fraction_months_release_clipped", "release_mae_vs_observed",
    )
    return {key: block[key] for key in keys}


def evaluate_one(series) -> dict:
    """Score one contiguous monthly series. The caller has already sliced the run."""
    fit_years, cal_years, test_years = split_years(series.years)
    n_fit, n_cal = len(fit_years), len(cal_years)
    s0 = float(series.storage[n_fit + n_cal - 1, 11])
    prev_inflow = float(series.inflow[n_fit + n_cal - 1, 11])
    spec = infer_targets(series, fit_years, cal_years, initial_storage=s0)
    res = spec["reservoir"]
    fit_in = series.inflow[:n_fit]
    cal_in = series.inflow[n_fit:n_fit + n_cal]
    test_in = series.inflow[n_fit + n_cal:]
    test_out = series.outflow[n_fit + n_cal:]
    forecaster = InflowForecaster().fit(fit_in)
    half_width = conformal_interval(forecaster, cal_in)
    test_coverage = coverage(forecaster, test_in, half_width)
    policy = SDPPolicy(
        res, forecaster, n_states=N_STATES, n_actions=N_ACTIONS, n_scen=N_SCEN,
        gamma=GAMMA, evaporation=spec["evap_mean"],
    )
    forecast = ForecastReleasePolicy(policy)
    evap = np.broadcast_to(spec["evap_mean"], test_in.shape).copy()
    common = {
        "s0": s0,
        "prev_inflow": prev_inflow,
        "prev_month": 11,
        "evaporation": evap,
        "observed_release": test_out,
    }

    def climatological(storage, prev_inflow, prev_month, month):
        del prev_inflow, prev_month
        return policy.release(storage, 0.0, month)

    historical = simulate_decisions(res, test_in, _table_release(test_out), **common)
    forecast_m = simulate_decisions(res, test_in, forecast.release, **common)
    climatology_m = simulate_decisions(res, test_in, climatological, **common)
    # Closure uses the month before the test so the first test month has a previous storage.
    closure_slice = slice(n_fit + n_cal - 1, None)
    closure = closure_residual(
        series.storage[closure_slice], series.inflow[closure_slice],
        series.outflow[closure_slice], series.evaporation[closure_slice],
    )
    conditional = []
    fraction = closure["median_abs_fraction_of_mean_inflow"]
    if fraction is None or fraction > 0.25:
        shown = "undefined" if fraction is None else f"{fraction:.3f}"
        conditional.append(
            "Median absolute mass-balance residual is "
            f"{shown} of mean inflow, so simulated storage is not observed storage."
        )
    if test_coverage < 0.80:
        conditional.append(
            f"Test forecast-interval coverage is {test_coverage:.3f} against a 0.90 target."
        )
    if not spec["evap_observed"]:
        conditional.append("Evaporation is missing and is treated as zero.")
    if abs(forecaster.phi) >= 0.989:
        conditional.append("The AR(1) coefficient sat on the stability clip at ±0.99.")
    if int(np.sum(test_in <= 0.0)):
        conditional.append(
            f"{int(np.sum(test_in <= 0.0))} test months have non-positive inflow and are floored at 1e-6 in the log forecast."
        )
    if historical["fraction_months_release_clipped"] > 0.05:
        conditional.append(
            "Historical releases were clipped by the mass balance in "
            f"{historical['fraction_months_release_clipped']:.3f} of test months."
        )
    mean_observed = float(np.mean(test_out))
    if mean_observed > 0.0 and forecast_m["release_mae_vs_observed"] > mean_observed:
        conditional.append(
            "Mean absolute error of the forecast release versus the observed release is "
            f"{forecast_m['release_mae_vs_observed']:.1f} hm3, larger than the mean observed "
            f"release of {mean_observed:.1f} hm3."
        )
    return {
        "dam_id": series.dam_id,
        "name": series.name,
        "capacity_hm3": res.capacity,
        "capacity_source": spec["capacity_source"],
        "dead_hm3": res.dead,
        "max_release_hm3": res.max_release,
        "fit_years": fit_years.astype(int),
        "calibration_years": cal_years.astype(int),
        "test_years": test_years.astype(int),
        "forecast": {
            "phi": forecaster.phi,
            "sd_log": forecaster.sd,
            "conformal_half_width_log": half_width,
            "coverage_target_0.90": test_coverage,
        },
        "closure_on_test": closure,
        "historical": _metrics(historical),
        "forecast_policy": _metrics(forecast_m),
        "climatological_sdp": _metrics(climatology_m),
        "forecast_cost_minus_historical": (
            forecast_m["cost_per_month"] - historical["cost_per_month"]
        ),
        "evaporation_observed": spec["evap_observed"],
        "conditional_limitations": conditional,
    }


def load_candidates(path: Path, min_years: int):
    with zipfile.ZipFile(path) as zf:
        attributes = load_attributes(zf)
        members = iter_series_members(zf)
        if not members:
            raise SystemExit("no time_series_all/ResOpsUS_<id>.csv members in the zip")
        loaded = []
        unreadable = 0
        for i, (dam_id, name) in enumerate(members, start=1):
            try:
                series = load_member(zf, dam_id, name, attributes)
            except (OSError, UnicodeError, ValueError, KeyError, IndexError, csv.Error) as exc:
                unreadable += 1
                print(f"unreadable {dam_id}: {exc}")
                continue
            if len(series.years):
                loaded.append(series)
            if i % 50 == 0 or i == len(members):
                print(f"scanned {i}/{len(members)} series; {len(loaded)} have a complete year")
    ranked = rank_reservoirs(loaded, min_years)
    return loaded, ranked, unreadable, len(members)


def run(n_reservoirs: int, min_years: int, zip_path: Path) -> dict:
    threshold = min_years
    loaded, ranked, unreadable, n_members = load_candidates(zip_path, threshold)
    if len(ranked) < n_reservoirs:
        for lower in (15, 12):
            if lower >= threshold:
                continue
            ranked = rank_reservoirs(loaded, lower)
            threshold = lower
            if len(ranked) >= n_reservoirs:
                break
    chosen = []
    skipped = []
    for start, length, series in ranked:
        if len(chosen) >= n_reservoirs:
            break
        piece = slice_series(series, start, length)
        try:
            chosen.append(evaluate_one(piece))
            print(f"evaluated dam {series.dam_id} ({series.name}) years={length}")
        except ValueError as exc:
            skipped.append({"dam_id": series.dam_id, "reason": str(exc)})
            print(f"skipped dam {series.dam_id}: {exc}")
    if len(chosen) < n_reservoirs:
        raise SystemExit(
            f"only {len(chosen)} reservoirs could be evaluated; need {n_reservoirs}"
        )
    costs = [row["forecast_cost_minus_historical"] for row in chosen]
    n_lower = int(sum(delta < 0.0 for delta in costs))
    n_below_sdp = int(sum(
        row["forecast_policy"]["cost_per_month"] < row["climatological_sdp"]["cost_per_month"]
        for row in chosen
    ))
    fair = []
    for row in chosen:
        fraction = row["closure_on_test"]["median_abs_fraction_of_mean_inflow"]
        clipped = row["historical"]["fraction_months_release_clipped"]
        if fraction is not None and fraction <= 0.25 and clipped <= 0.05:
            fair.append(row)
    n_fair_lower = int(sum(row["forecast_cost_minus_historical"] < 0.0 for row in fair))
    n_fair_sdp = int(sum(
        row["forecast_policy"]["cost_per_month"] < row["climatological_sdp"]["cost_per_month"]
        for row in fair
    ))
    summary = (
        f"On the inferred cost, the forecast release is lower than the historical release "
        f"on {n_lower} of {len(chosen)} reservoirs, and lower than the climatological SDP "
        f"on {n_below_sdp} of {len(chosen)}. "
        f"Where the test mass balance closes within 0.25 of mean inflow and historical "
        f"releases are clipped in at most 0.05 of months ({len(fair)} of {len(chosen)}), "
        f"the forecast release is lower than historical on {n_fair_lower} of {len(fair)} "
        f"and lower than the climatological SDP on {n_fair_sdp} of {len(fair)}."
    )
    return {
        "dataset": {
            "name": "ResOpsUS",
            "zenodo_record": ZENODO_RECORD,
            "doi": ZENODO_DOI,
            "md5": ZENODO_MD5,
            "citation": (
                "Steyaert, J. C., Condon, L. E., Turner, S. W. D., and Voisin, N. (2022). "
                "ResOpsUS, a dataset of historical reservoir operations in the contiguous "
                "United States. Scientific Data 9, 34. Data: https://doi.org/10.5281/zenodo.5367383"
            ),
        },
        "method": {
            "series_files": n_members,
            "unreadable_files": unreadable,
            "series_with_a_complete_year": len(loaded),
            "min_contiguous_years_requested": min_years,
            "min_contiguous_years_used": threshold,
            "n_reservoirs": len(chosen),
            "selection": (
                "Earliest longest contiguous run of complete calendar years. "
                "Ordered by run length, then by GRanD dam id. "
                "Days outside the physical screens or with a missing storage, inflow, or outflow drop the month."
            ),
            "physical_screens": {
                "max_storage_hm3": MAX_STORAGE_HM3,
                "max_flow_m3s": MAX_FLOW_M3S,
            },
            "split": (
                "Last max(5, round(0.20*N)) years test; preceding max(3, round(0.15*N)) "
                "years calibrate the conformal interval; earlier years fit the forecast and the SDP value."
            ),
            "sdp_grid": {
                "n_states": N_STATES, "n_actions": N_ACTIONS, "n_scen": N_SCEN, "gamma": GAMMA,
            },
            "cost_weights": {"shortage": 1.0, "spill": 0.2, "flood": 5.0},
        },
        "standing_limitations": STANDING_LIMITATIONS,
        "skipped_before_quota": skipped,
        "reservoirs": chosen,
        "summary": {
            "n_forecast_cost_lower_than_historical": n_lower,
            "n_forecast_cost_lower_than_climatological_sdp": n_below_sdp,
            "n_reservoirs_with_closed_balance": len(fair),
            "n_forecast_cost_lower_than_historical_where_balance_closes": n_fair_lower,
            "n_forecast_cost_lower_than_climatological_sdp_where_balance_closes": n_fair_sdp,
            "sentence": summary,
        },
    }


def render_markdown(result: dict) -> str:
    lines = [result["summary"]["sentence"], "",
            "| Dam | Test years | Coverage | Historical cost | Forecast cost | Climatological SDP cost | Historical reliability | Forecast reliability | Forecast release MAE | Closure / inflow |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["reservoirs"]:
        label = str(row["dam_id"]) if not row["name"] else f"{row['dam_id']} {row['name']}"
        label = label.replace("|", "/")
        years = row["test_years"]
        span = f"{years[0]}–{years[-1]}"
        closure = row["closure_on_test"]["median_abs_fraction_of_mean_inflow"]
        closure_txt = "n/a" if closure is None else f"{closure:.3f}"
        lines.append(
            "| {label} | {span} | {cov:.3f} | {hc:.3f} | {fc:.3f} | {sc:.3f} | {hr:.3f} | {fr:.3f} | {mae:.1f} | {cl} |".format(
                label=label,
                span=span,
                cov=row["forecast"]["coverage_target_0.90"],
                hc=row["historical"]["cost_per_month"],
                fc=row["forecast_policy"]["cost_per_month"],
                sc=row["climatological_sdp"]["cost_per_month"],
                hr=row["historical"]["reliability"],
                fr=row["forecast_policy"]["reliability"],
                mae=row["forecast_policy"]["release_mae_vs_observed"],
                cl=closure_txt,
            )
        )
    lines.append("")
    lines.append("Conditional limitations from this run:")
    for row in result["reservoirs"]:
        if not row["conditional_limitations"]:
            lines.append(f"- Dam {row['dam_id']}: none triggered.")
            continue
        for item in row["conditional_limitations"]:
            lines.append(f"- Dam {row['dam_id']}: {item}")
    lines.append("")
    return "\n".join(lines)


def write_marked(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(f"{path} is missing ResOpsUS markers")
    start = text.index(MARK_START)
    end = text.index(MARK_END)
    updated = text[:start] + MARK_START + "\n" + body.rstrip() + "\n" + text[end:]
    path.write_text(updated, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(ROOT / "data" / "ResOpsUS.zip"))
    parser.add_argument("--out", default=str(ROOT / "research" / "results.json"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--min-years", type=int, default=20)
    parser.add_argument("--write-docs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    zip_path = Path(args.zip)
    ensure_zip(zip_path)
    result = run(args.n, args.min_years, zip_path)
    payload = json.dumps(_builtin(result), indent=2)
    Path(args.out).write_text(payload + "\n", encoding="utf-8", newline="\n")
    print(result["summary"]["sentence"])
    if args.write_docs:
        body = render_markdown(result)
        write_marked(ROOT / "README.md", body)
        write_marked(ROOT / "docs" / "TRL.md", body)


if __name__ == "__main__":
    main()
