# Software Requirements Specification (SRS)

| ID | Requirement | Module | Test |
|----|-------------|--------|------|
| FR-1 | Mass balance with dead storage, outlet limit and spill | `hydrology.py` | `test_mass_balance_conserves_water`, `test_release_never_below_dead_storage`, `test_spill_when_full` |
| FR-2 | Seeded synthetic inflow with seasonality and persistence | `hydrology.py` | `test_inflow_generator_seeded_and_positive` |
| FR-3 | Fit a monthly-mean plus AR(1) log-inflow forecaster | `forecast.py` | `test_forecaster_recovers_persistence` |
| FR-4 | Split-conformal interval for the forecast | `forecast.py` | `test_conformal_interval_covers_on_new_data` |
| FR-5 | Rule-curve baseline and SDP release policy | `policy.py` | `test_sdp_beats_rule_curve_on_cost` |
| FR-6 | Simulate a policy and report reliability, shortage, flood violations, spill, cost | `evaluate.py` | `test_sdp_beats_rule_curve_on_cost` |
| FR-7 | Aggregate a ResOpsUS daily CSV to complete calendar months in hm3 | `resops.py` | `test_complete_year_converts_cumecs_to_hm3`, `test_missing_day_drops_the_year` |
| FR-8 | Choose a release from last month's inflow without reading this month's inflow | `policy.py`, `evaluate.py` | `test_conditional_scenarios_follow_the_forecast`, `test_decision_simulation_does_not_need_the_current_inflow_to_call_back` |
| FR-9 | Fit supply targets on fit and calibration years only | `resops.py` | `test_demand_ignores_years_outside_fit_and_calibration` |

Cost weights (`policy.py`): shortage (squared relative shortage) 1.0, spill 0.2, flood-limit exceedance 5.0. The climatological SDP treats inflow as climatological within a month. `ForecastReleasePolicy` recenters the current month's scenarios on the log-AR(1) forecast and keeps that climatological continuation value. The real-data study replays ResOpsUS releases in the same mass balance; its measured outcome is `research/results.json`, not a number in this specification.
