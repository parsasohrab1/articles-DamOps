"""Load ResOpsUS daily reservoir series and aggregate them to calendar months.

Units, from Steyaert et al. (Scientific Data, 2022) and Zenodo 10.5281/zenodo.5367383:
storage and evaporation are million cubic metres (hm3); inflow and outflow are
cubic metres per second. A monthly volume is the sum of daily rate * 86400 / 1e6.

A month is kept only when every calendar day has a finite storage, inflow and
outflow inside the physical screens below. A year is kept only when all twelve
months are kept. Gaps are not filled.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np

from .hydrology import Reservoir

# Larger than any CONUS dam inflow or storage we would treat as a real observation.
# The ResOpsUS paper flags isolated points that are orders of magnitude too large.
MAX_FLOW_M3S = 200_000.0
MAX_STORAGE_HM3 = 200_000.0
_MISSING = {"", "na", "nan", "null", "none", "-"}
_SERIES_RE = re.compile(r"ResOpsUS_(\d+)\.csv$", re.IGNORECASE)


@dataclass
class MonthlySeries:
    dam_id: int
    years: np.ndarray
    inflow: np.ndarray
    outflow: np.ndarray
    storage: np.ndarray
    evaporation: np.ndarray
    name: str | None = None
    capacity_attribute: float | None = None


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]


def longest_contiguous_run(years: np.ndarray) -> tuple[int, int]:
    """Return (start index, length) of the earliest longest run of successive years."""
    if len(years) == 0:
        return 0, 0
    best_i, best_n, start = 0, 1, 0
    for i in range(1, len(years) + 1):
        if i == len(years) or int(years[i]) != int(years[i - 1]) + 1:
            if i - start > best_n:
                best_i, best_n = start, i - start
            start = i
    return best_i, best_n


def split_years(years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit, conformal-calibration, and test years. Test is the last block."""
    n = len(years)
    n_test = max(5, round(n * 0.20))
    n_cal = max(3, round(n * 0.15))
    if n_test + n_cal >= n:
        raise ValueError(f"need more than {n_test + n_cal} years to split, got {n}")
    n_fit = n - n_test - n_cal
    return years[:n_fit], years[n_fit:n_fit + n_cal], years[n_fit + n_cal:]


def slice_series(series: MonthlySeries, start: int, length: int) -> MonthlySeries:
    sl = slice(start, start + length)
    return MonthlySeries(
        dam_id=series.dam_id,
        years=np.asarray(series.years[sl], dtype=int).copy(),
        inflow=np.asarray(series.inflow[sl], dtype=float).copy(),
        outflow=np.asarray(series.outflow[sl], dtype=float).copy(),
        storage=np.asarray(series.storage[sl], dtype=float).copy(),
        evaporation=np.asarray(series.evaporation[sl], dtype=float).copy(),
        name=series.name,
        capacity_attribute=series.capacity_attribute,
    )


def _parse_float(text: str) -> float | None:
    raw = text.strip()
    if raw.lower() in _MISSING:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def _parse_date(text: str) -> tuple[int, int, int] | None:
    raw = text.strip()[:10]
    if len(raw) < 10 or raw[4] != "-" or raw[7] != "-":
        return None
    try:
        year, month, day = int(raw[0:4]), int(raw[5:7]), int(raw[8:10])
    except ValueError:
        return None
    if not (1800 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year, month, day


def _column(header: list[str], *options: str) -> int | None:
    for option in options:
        if option in header:
            return header.index(option)
    return None


def aggregate_csv(text: str, dam_id: int) -> MonthlySeries:
    """Aggregate one ResOpsUS time-series CSV into complete calendar years."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = [cell.strip().lower().lstrip("\ufeff") for cell in next(reader)]
    except StopIteration:
        return _empty(dam_id)
    i_date = _column(header, "date", "datetime")
    i_storage = _column(header, "storage")
    i_in = _column(header, "inflow")
    i_out = _column(header, "outflow", "release")
    i_evap = _column(header, "evaporation", "evap")
    if None in (i_date, i_storage, i_in, i_out):
        return _empty(dam_id)
    width = max(i for i in (i_date, i_storage, i_in, i_out, i_evap) if i is not None) + 1
    # slot: n_storage, sum_inflow_m3s, n_inflow, sum_outflow_m3s, n_outflow,
    #       end_storage, end_day, sum_evap, n_evap
    bucket: dict[tuple[int, int], list[float]] = {}
    seen: set[tuple[int, int, int]] = set()
    for row in reader:
        if len(row) < width:
            continue
        parsed = _parse_date(row[i_date])
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        year, month, day = parsed
        slot = bucket.get((year, month))
        if slot is None:
            slot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
            bucket[(year, month)] = slot
        storage = _parse_float(row[i_storage])
        inflow = _parse_float(row[i_in])
        outflow = _parse_float(row[i_out])
        if storage is not None and 0.0 <= storage <= MAX_STORAGE_HM3:
            slot[0] += 1.0
            if day >= slot[6]:
                slot[6] = float(day)
                slot[5] = storage
        if inflow is not None and 0.0 <= inflow <= MAX_FLOW_M3S:
            slot[1] += inflow
            slot[2] += 1.0
        if outflow is not None and 0.0 <= outflow <= MAX_FLOW_M3S:
            slot[3] += outflow
            slot[4] += 1.0
        if i_evap is not None:
            evap = _parse_float(row[i_evap])
            if evap is not None and 0.0 <= evap <= MAX_STORAGE_HM3:
                slot[7] += evap
                slot[8] += 1.0

    complete = []
    for year in sorted({year for year, _month in bucket}):
        months_present = all((year, month) in bucket for month in range(1, 13))
        months_full = months_present and all(
            _month_complete(bucket[(year, month)], year, month) for month in range(1, 13))
        if months_full:
            complete.append(year)
    if not complete:
        return _empty(dam_id)
    n = len(complete)
    inflow_m = np.zeros((n, 12))
    outflow_m = np.zeros((n, 12))
    storage_m = np.zeros((n, 12))
    evap_m = np.full((n, 12), np.nan)
    for i, year in enumerate(complete):
        for month in range(1, 13):
            slot = bucket[(year, month)]
            inflow_m[i, month - 1] = slot[1] * 86400.0 / 1e6
            outflow_m[i, month - 1] = slot[3] * 86400.0 / 1e6
            storage_m[i, month - 1] = slot[5]
            dim = days_in_month(year, month)
            if slot[8] == dim:
                evap_m[i, month - 1] = slot[7]
    return MonthlySeries(
        dam_id=dam_id,
        years=np.asarray(complete, dtype=int),
        inflow=inflow_m,
        outflow=outflow_m,
        storage=storage_m,
        evaporation=evap_m,
    )


def _month_complete(slot: list[float], year: int, month: int) -> bool:
    dim = float(days_in_month(year, month))
    return slot[0] == dim and slot[2] == dim and slot[4] == dim and slot[6] >= 1.0


def _empty(dam_id: int) -> MonthlySeries:
    return MonthlySeries(
        dam_id=dam_id,
        years=np.zeros(0, dtype=int),
        inflow=np.zeros((0, 12)),
        outflow=np.zeros((0, 12)),
        storage=np.zeros((0, 12)),
        evaporation=np.zeros((0, 12)),
    )


def rank_reservoirs(series_list: list[MonthlySeries], min_years: int) -> list[tuple[int, int, MonthlySeries]]:
    """Longest contiguous run first, then smaller dam id. Each item is (start, length, series)."""
    ranked = []
    for series in series_list:
        start, length = longest_contiguous_run(series.years)
        if length < min_years:
            continue
        ranked.append((length, series.dam_id, start, series))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(start, run_length, series) for run_length, _dam, start, series in ranked]


def infer_targets(series: MonthlySeries, fit_years: np.ndarray, cal_years: np.ndarray,
                  initial_storage: float | None = None) -> dict:
    """Reservoir bounds and targets. Test years are not used for demand, flood limit, or evaporation.

    The dead-storage floor is the training 5th percentile, lowered to the storage at the
    start of the test if that observed storage is below the percentile. The start storage
    is the last calibration month, so this does not read the test years.
    """
    train_years = np.concatenate([np.asarray(fit_years), np.asarray(cal_years)])
    train_index = np.isin(series.years, train_years)
    if not np.any(train_index):
        raise ValueError("training years are missing from the series")
    train_storage = series.storage[train_index]
    train_out = series.outflow[train_index]
    observed_max = float(np.max(series.storage))
    attr = series.capacity_attribute
    if (attr is not None and np.isfinite(attr) and attr > 0.0
            and 0.5 * observed_max <= attr <= 5.0 * observed_max):
        capacity = max(float(attr), observed_max)
        source = "reservoir_attributes" if attr >= observed_max else "observed_max_storage"
    else:
        capacity = observed_max
        source = "observed_max_storage"
    if initial_storage is None:
        initial_storage = float(np.min(train_storage))
    dead = min(float(np.quantile(train_storage, 0.05)), float(initial_storage))
    dead = min(max(dead, 0.0), 0.9 * capacity)
    if not capacity > dead + 1.0:
        raise ValueError("storage range is too small")
    max_release = float(max(np.max(series.outflow), 1e-3))
    mean_release = float(np.mean(train_out))
    demand = np.maximum(np.median(train_out, axis=0), max(mean_release * 0.05, 1e-3))
    flood = np.quantile(train_storage, 0.90, axis=0).astype(float)
    flood = np.minimum(np.maximum(flood, dead), capacity)
    train_evap = series.evaporation[train_index]
    if np.isfinite(train_evap).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            evap_mean = np.nanmean(train_evap, axis=0)
    else:
        evap_mean = np.zeros(12)
    evap_mean = np.where(np.isfinite(evap_mean), evap_mean, 0.0)
    reservoir = Reservoir(
        capacity=capacity,
        dead=dead,
        flood_limit=flood,
        max_release=max_release,
        demand=demand,
    )
    return {
        "reservoir": reservoir,
        "evap_mean": evap_mean.astype(float),
        "capacity_source": source,
        "evap_observed": bool(np.isfinite(series.evaporation[train_index]).any()),
    }


def closure_residual(storage: np.ndarray, inflow: np.ndarray, outflow: np.ndarray,
                     evaporation: np.ndarray) -> dict:
    """Median |Δstorage − inflow + outflow + evaporation| on a contiguous monthly series.

    ``storage`` is end-of-month. The first month has no previous storage and is omitted.
    Missing evaporation is treated as zero and counted.
    """
    evap = np.where(np.isfinite(evaporation), evaporation, 0.0)
    flat_s, flat_i = storage.ravel(), inflow.ravel()
    flat_o, flat_e = outflow.ravel(), evap.ravel()
    resid = (flat_s[1:] - flat_s[:-1]) - flat_i[1:] + flat_o[1:] + flat_e[1:]
    mean_in = float(np.mean(flat_i[1:])) if len(flat_i) > 1 else 0.0
    med = float(np.median(np.abs(resid))) if len(resid) else 0.0
    return {
        "median_abs_hm3": med,
        "mean_inflow_hm3": mean_in,
        "median_abs_fraction_of_mean_inflow": (med / mean_in) if mean_in > 0.0 else None,
        "n_months": int(resid.size),
        "fraction_months_evap_missing": float(np.mean(~np.isfinite(evaporation))),
    }


def _read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    raw = zf.read(name)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _pick(fields: dict[str, str], options: tuple[str, ...]) -> str | None:
    for option in options:
        if option in fields:
            return fields[option]
    for option in options:
        for key, original in fields.items():
            if option in key:
                return original
    return None


def load_attributes(zf: zipfile.ZipFile) -> dict[int, dict]:
    names = [name for name in zf.namelist()
             if name.replace("\\", "/").lower().endswith("reservoir_attributes.csv")
             and "__MACOSX" not in name.replace("\\", "/")
             and not name.replace("\\", "/").split("/")[-1].startswith("._")]
    if not names:
        return {}
    reader = csv.DictReader(io.StringIO(_read_zip_text(zf, names[0])))
    if not reader.fieldnames:
        return {}
    fields = {field.strip().lower(): field for field in reader.fieldnames if field}
    id_name = _pick(fields, ("dam_id", "grand_id", "grandid"))
    name_name = _pick(fields, ("dam_name", "res_name", "reservoir_name", "name"))
    cap_name = _pick(fields, ("cap_mcm", "capacity_mcm", "max_storage_mcm", "store_mcm", "capacity"))
    if id_name is None:
        return {}
    out: dict[int, dict] = {}
    for row in reader:
        raw_id = (row.get(id_name) or "").strip()
        try:
            dam_id = int(float(raw_id))
        except ValueError:
            continue
        capacity = None
        if cap_name is not None:
            capacity = _parse_float(row.get(cap_name) or "")
        label = (row.get(name_name) or "").strip() if name_name else ""
        out[dam_id] = {"name": label or None, "capacity": capacity}
    return out


def iter_series_members(zf: zipfile.ZipFile) -> list[tuple[int, str]]:
    found = []
    for name in zf.namelist():
        parts = name.replace("\\", "/").split("/")
        if "__MACOSX" in parts or parts[-1].startswith("._"):
            continue
        if "time_series_all" not in parts:
            continue
        match = _SERIES_RE.search(parts[-1])
        if match:
            found.append((int(match.group(1)), name))
    return sorted(found)


def load_member(zf: zipfile.ZipFile, dam_id: int, name: str,
                attributes: dict[int, dict]) -> MonthlySeries:
    series = aggregate_csv(_read_zip_text(zf, name), dam_id)
    meta = attributes.get(dam_id, {})
    series.name = meta.get("name")
    series.capacity_attribute = meta.get("capacity")
    return series
