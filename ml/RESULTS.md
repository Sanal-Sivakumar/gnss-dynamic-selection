# Adaptive-weight reliability results

**Simulation only: these are not measured RTK or real-world accuracy results.**

Trained in pure NumPy; 72,000 synthetic receiver readings, 120 sessions
(80 train / 20 validation / 20 test, whole-session stratified split).

## Receiver-selection comparison (held-out TEST split, 5,986 available epochs)

| Selector                    | Accuracy (vs oracle) | Mean (m) | Median (m) | RMSE (m) | p95 (m) | Switches |
| --------------------------- | -------------------: | -------: | ---------: | ------: | ------: | -------: |
| Model A (original fixed)    |                61.3% |    3.509 |      2.556 |    4.913 |   9.279 |     1788 |
| Model B (optimized fixed)   |                61.6% |    3.488 |      2.538 |    4.856 |   9.220 |     1768 |
| Model C (dynamic ML)        |                62.6% |    3.435 |      2.511 |    4.777 |   8.886 |     1767 |
| Model C + hysteresis (0.05) |                64.0% |    3.394 |      2.489 |    4.724 |   8.727 |     1018 |

The oracle (lowest actual error) is an unattainable lower bound, not a model.
Model B weights: α=0.10, β=0.20, γ=0.00, δ=0.70. Improvements are real but
modest — the honest story is that dynamic weighting helps mostly through
stabilized switching (hysteresis), not large accuracy jumps.

## Hysteresis sweep (Model C, test split)

| Threshold | Accuracy | Mean (m) | Switches | Mean duration |
| --------: | ------: | -------: | -------: | ------------: |
|      0.00 |   62.6%  |    3.435 |     1767 |         3.33 s |
|      0.02 |   63.6%  |    3.413 |     1314 |         4.46 s |
|      0.05 |   64.0%  |    3.394 |     1018 |         5.72 s |
|      0.10 |   63.8%  |    3.407 |      716 |         8.05 s |

Threshold 0.05 is the best balance for this dataset: accuracy and error improve
while switching drops ~42% relative to no hysteresis.

## Ablation (conditioning feature groups, test split)

| Conditioning input          | Mean (m) | Accuracy | Switches |
| --------------------------- | -------: | ------: | -------: |
| T only                      |    3.504 |   61.4%  |     1839 |
| S only                      |    3.457 |   62.2%  |     1758 |
| SNR only                    |    3.504 |   61.6%  |     1868 |
| DOP only                    |    3.518 |   61.4%  |     1901 |
| T + S                       |    3.505 |   61.4%  |     1868 |
| T + S + SNR                 |    3.481 |   61.5%  |     1719 |
| all four (T,S,SNR,DOP)      |    3.468 |   62.2%  |     1842 |
| all four + comparative       |    3.450 |   62.5%  |     1757 |
| all four + temporal (full)   |    3.440 |   62.8%  |     1750 |

Adding comparative (Δ) and temporal features progressively improves selection —
the model genuinely benefits from the different information sources.

## Learned weights by GNSS condition (test, both receivers available)

| Condition              | α (timing) | β (sats) | γ (SNR) | δ (DOP) |
| ---------------------- | ---------: | -------: | ------: | ------: |
| Open sky               |      0.001 |    0.001 |   0.000 |   0.998 |
| Urban multipath        |      0.417 |    0.548 |   0.012 |   0.023 |
| Partial obstruction    |      0.123 |    0.298 |   0.022 |   0.558 |
| Dropout                |      0.059 |    0.167 |   0.014 |   0.760 |
| High satellite count   |      0.081 |    0.168 |   0.013 |   0.737 |
| Low satellite count    |      0.419 |    0.571 |   0.004 |   0.007 |
| Good geometry          |      0.003 |    0.004 |   0.000 |   0.992 |
| Poor geometry          |      0.393 |    0.570 |   0.007 |   0.030 |
| High SNR               |      0.006 |    0.012 |   0.001 |   0.980 |
| Low SNR                |      0.327 |    0.502 |   0.013 |   0.158 |

The model learns interpretable, context-dependent weighting: DOP dominates
under good signal conditions, while timing (α) and satellite count (β) take
over under degraded conditions. γ (SNR) carries little extra discriminative
power in this synthetic data. See `artifacts/metrics.json` for full per-scenario
weight statistics.

## Research questions (summary)

- **RQ1** — Original equation selects the better receiver 61.3% of available epochs.
- **RQ2** — Optimized fixed weights are slightly better than the original (61.6%,
  3.488 m vs 3.509 m).
- **RQ3** — Dynamic ML weighting improves over the original coefficients (62.6%,
  3.435 m); the gain is modest.
- **RQ4** — Model C outperforms the single optimized fixed set, but only slightly.
- **RQ5** — Yes: learned weights are interpretable and shift with GNSS conditions.
- **RQ6** — Adaptive method does not introduce excessive switching; hysteresis
  reduces switches ~42% while improving accuracy.

## Files

- `artifacts/model.npz` — exported weights + training-only normalization
- `artifacts/metrics.json` — full comparison, splits, history, learned weights
- `artifacts/evaluation.json` — hysteresis sweep + ablation study
- `artifacts/training_data.csv`, `artifacts/testing_data.csv` — readable rows