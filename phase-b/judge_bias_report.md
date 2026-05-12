# Bias Observations Report

## Quantified observations

- Position sensitivity before swap mitigation: 0% of comparisons changed winner when order changed.
- Length bias: longer answer B won 100% of cases where it was longer than answer A.
- Tie rate after swap-and-average: 73%.

## Interpretation

- Swap-and-average removes most order artifacts because disagreements collapse to `tie`.
- Length still correlates with wins because answer B was intentionally more complete on reasoning and multi-context items.

## Table

| Metric | Value |
|---|---|
| Position sensitivity | 0.00% |
| Longer-B win rate | 100.00% |
| Final tie rate | 73.33% |
| Human-vs-judge kappa | 1.0000 |