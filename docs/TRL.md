# Technology readiness (evidence-based)

**Level: TRL 4 on synthetic inflow.** TRL 5 is not claimed.

## Evidence
7 unit tests pass. `research/run_study.py` (training 200 years, test 500 years per setting):

| Test climate | Policy | Full-supply reliability | Flood-limit violations per year | Spill (hm3/yr) | Cost per month |
|---|---|---|---|---|---|
| Same | SDP | 0.70 | 0.19 | 4.1 | 1.90 |
| Same | Rule curve | 0.93 | 0.67 | 9.5 | 3.70 |
| Drought (inflow x0.7) | SDP | 0.31 | 0.02 | 0.25 | 8.35 |
| Drought | Rule curve | 0.51 | 0.03 | 0.10 | 12.21 |
| Wet (inflow x1.3) | SDP | 0.92 | 0.71 | 26.3 | 3.83 |
| Wet | Rule curve | 1.00 | 2.10 | 67.7 | 11.58 |

The SDP has lower cost and fewer flood-limit violations, but a **lower** full-supply reliability, because the cost penalises relative shortage quadratically and so accepts frequent small shortages to avoid floods. Whether that trade is wanted is a decision for the operator and sets the weights.

Forecast interval coverage (target 0.90): 0.90 in the training climate, 0.86 in drought, 0.88 in the wet climate.

## Not shown
Synthetic inflow only; the generator and the forecaster share the same lognormal-AR(1) form. No real hydrometric series, no real reservoir.

## Next steps toward TRL 5
A real monthly inflow and storage series for one dam, the operator's actual rule curve as baseline, and a validation on held-out years.
