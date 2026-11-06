# articles-DamOps

Reservoir operation (release policy) with a calibrated one-month inflow forecast and
stochastic dynamic programming. The synthetic study remains available
(`python research/run_study.py`, output `research/synthetic_results.json`). The
retrospective benchmark uses ResOpsUS (Zenodo record 5367383).

**Maturity: TRL 4.** TRL 5 is not claimed. The real-data comparison is an offline replay
against observed releases, with an inferred supply target. See [docs/TRL.md](docs/TRL.md).

- `src/damops/hydrology.py` synthetic monthly inflow and mass balance (optional evaporation)
- `src/damops/forecast.py` log-AR(1) forecaster with split-conformal intervals
- `src/damops/policy.py` rule curve, climatological SDP, and a release that recenters inflow scenarios on the one-month forecast
- `src/damops/resops.py` ResOpsUS daily-to-monthly loader
- `src/damops/evaluate.py` simulation metrics
- Docs: [BRS](docs/BRS.md) · [SRS](docs/SRS.md) · [TRL](docs/TRL.md)

```bash
pip install -e ".[dev]"
ruff check src tests research
python -m pytest -q
python research/run_resops.py --out research/results.json
```

`run_resops.py` downloads `data/ResOpsUS.zip` (about 300 MB, gitignored) when the
local file is missing or the Zenodo md5 does not match, then overwrites
`research/results.json` and the marked blocks below and in `docs/TRL.md`. No metric
in those blocks is typed by hand.

## Real-data study

Selection, before any policy score is computed: the earliest longest contiguous run of
complete calendar years in `time_series_all`, ordered by length and then by GRanD dam id.
A month is complete only when every day has storage, inflow and outflow inside the
screens in `src/damops/resops.py`. The last block of years is the test. The forecast is
fit on the earlier years. The historical policy is the observed monthly release, replayed
in the same mass balance as the forecast policy. The forecast policy does not see the
current month's inflow.

Standing limitations, independent of the numbers below:

- The supply target is the training-month median release, not a water order.
- The flood limit is the training-month 90th percentile of storage, not a published flood pool.
- The cost weights are the synthetic-study weights, not an operator's weights.
- The forecast changes only the current month's inflow scenarios.
- A lower cost is not, by itself, a better policy for the owner of the dam.

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
