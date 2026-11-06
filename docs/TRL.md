# Technology readiness (evidence-based)

**Level: TRL 4.** TRL 5 is not claimed.

The synthetic inflow study is still reproducible with `python research/run_study.py`
and is stored in `research/synthetic_results.json`. It is a laboratory generator that
shares its lognormal-AR(1) form with the forecaster. That study does not use a reservoir record.

The ResOpsUS benchmark below is a retrospective simulation on held-out calendar years.
It is evidence about this code on historical inflow and release, not an operational trial.
The baseline is the observed release series. Supply targets and flood limits are inferred
from the fit and calibration years of the same record. Evaporation is the seasonal mean
when ResOpsUS reports it, and zero when it does not. Because those targets are inferred,
agreement or disagreement with historical releases is not a decision to change an operator's rule.

## ResOpsUS result

`research/results.json` is the source of the table. It is produced by
`python research/run_resops.py` from Zenodo record 5367383.

<!-- RESOPS:START -->
On the inferred cost, the forecast release is lower than the historical release on 4 of 4 reservoirs, and lower than the climatological SDP on 1 of 4. Where the test mass balance closes within 0.25 of mean inflow and historical releases are clipped in at most 0.05 of months (1 of 4), the forecast release is lower than historical on 1 of 1 and lower than the climatological SDP on 0 of 1.

| Dam | Test years | Coverage | Historical cost | Forecast cost | Climatological SDP cost | Historical reliability | Forecast reliability | Forecast release MAE | Closure / inflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 789 Sandy Lake Dam and Lock | 1982–2001 | 0.929 | 215.124 | 0.745 | 0.722 | 0.608 | 0.733 | 19.6 | 0.435 |
| 920 Jeffrey Reg Dam | 2004–2018 | 0.749 | 34.110 | 23.284 | 20.810 | 0.461 | 0.444 | 25.9 | 0.035 |
| 1753 Kentucky | 2005–2019 | 0.927 | 21.929 | 21.728 | 21.987 | 0.339 | 0.344 | 2467.4 | 0.119 |
| 1042 Norfork | 2005–2019 | 0.827 | 49.696 | 5.020 | 4.357 | 0.506 | 0.600 | 68.4 | 0.064 |

Conditional limitations from this run:
- Dam 789: Median absolute mass-balance residual is 0.435 of mean inflow, so simulated storage is not observed storage.
- Dam 789: Evaporation is missing and is treated as zero.
- Dam 920: Test forecast-interval coverage is 0.749 against a 0.90 target.
- Dam 920: Evaporation is missing and is treated as zero.
- Dam 920: Historical releases were clipped by the mass balance in 0.522 of test months.
- Dam 1753: Evaporation is missing and is treated as zero.
- Dam 1753: Historical releases were clipped by the mass balance in 0.694 of test months.
- Dam 1042: Evaporation is missing and is treated as zero.
<!-- RESOPS:END -->

## Not shown

No published rule curve, no water-order record, and no run by an operator. The mass balance
omits routing between dams. The continuation value of the SDP is climatological.
