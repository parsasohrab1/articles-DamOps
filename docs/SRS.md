# Software Requirements Specification (SRS)

| ID | Requirement | Module | Test |
|----|-------------|--------|------|
| FR-1 | Mass balance with dead storage, outlet limit and spill | `hydrology.py` | `test_mass_balance_conserves_water`, `test_release_never_below_dead_storage`, `test_spill_when_full` |
| FR-2 | Seeded synthetic inflow with seasonality and persistence | `hydrology.py` | `test_inflow_generator_seeded_and_positive` |
| FR-3 | Fit a monthly-mean plus AR(1) log-inflow forecaster | `forecast.py` | `test_forecaster_recovers_persistence` |
| FR-4 | Split-conformal interval for the forecast | `forecast.py` | `test_conformal_interval_covers_on_new_data` |
| FR-5 | Rule-curve baseline and SDP release policy | `policy.py` | `test_sdp_beats_rule_curve_on_cost` |
| FR-6 | Simulate a policy and report reliability, shortage, flood violations, spill, cost | `evaluate.py` | `test_sdp_beats_rule_curve_on_cost` |

Cost weights (`policy.py`): shortage (squared relative shortage) 1.0, spill 0.2, flood-limit exceedance 5.0. Inflow is lognormal with an AR(1) anomaly; the SDP treats inflow as climatological within a month (it does not condition on the last anomaly).
