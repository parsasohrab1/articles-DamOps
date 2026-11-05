# articles-DamOps

Reservoir operation (release policy) with calibrated one-month inflow forecasts and stochastic
dynamic programming. Product for the water special issue "Machine Learning Applications in
Hydrology and Water Resource Management" (Water, MDPI).

**Maturity: TRL 4 on synthetic inflow** ([docs/TRL.md](docs/TRL.md)). No real reservoir data yet.

- `src/damops/hydrology.py` synthetic monthly inflow, mass balance
- `src/damops/forecast.py` log-AR(1) forecaster with split-conformal intervals
- `src/damops/policy.py` rule-curve baseline and SDP policy
- `src/damops/evaluate.py` simulation metrics; `research/run_study.py` study; `tests/` 7 tests
- Docs: [BRS](docs/BRS.md) · [SRS](docs/SRS.md) · [TRL](docs/TRL.md)

```bash
pip install -e ".[dev]" && python -m pytest -q && python research/run_study.py
```
