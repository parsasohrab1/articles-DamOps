# Business Requirements Specification (BRS)

**Product:** DamOps (basket id `damops`). Written from the basket description and this code; not reviewed by the owner.

## Business problem
Dam operators must balance drinking and farm water supply against flood control while inflow is uncertain and droughts persist. Fixed rule curves ignore the forecast and its uncertainty.

## Objectives
| ID | Objective |
|----|-----------|
| BO-1 | Forecast next-month inflow with an interval of stated coverage |
| BO-2 | Choose monthly releases that trade supply shortage against spill and flood-limit violations |
| BO-3 | Show how the policy behaves in a drought and in a wet climate, not only in the training climate |
| BO-4 | Make the trade-off explicit: the cost weights are inputs |

## Users
Regional water company, dam operators, water resource planners.

## Scope
In: monthly single-reservoir model, forecast, SDP policy, simulation. Out: multi-reservoir cascades, hydropower, reinforcement learning, real-time control.

## Success measures (on real data)
Supply reliability, maximum shortage, flood-limit violations per year, spill, forecast coverage, against the historical release series. On ResOpsUS the supply target and the flood limit are inferred from fit and calibration years because the archive does not include a water order or a published rule curve. The comparison is recorded in `research/results.json`. A lower inferred cost is not by itself a reason to replace an operator's rule.
