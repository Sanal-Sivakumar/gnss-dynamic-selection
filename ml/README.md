# Adaptive-weight GNSS reliability experiment

This module trains an **adaptive weight learner** for the GNSS receiver-selection
equation. The ML component does **not** replace the existing reliability equation
— it learns context-dependent coefficients that plug into it:

    W_i = α(X)·T_i + β(X)·S_i + γ(X)·SNR_i + δ(X)·DOP_i

    Best Receiver = argmax(W_1, W_2)

where `α + β + γ + δ = 1` (softmax-constrained). The deterministic selection
rule remains unchanged; only the weights become condition-dependent.

**All data is SYNTHETIC/generated.** Nothing here implies real measurements or
validated positioning accuracy.

## Reproduce

From the repository root, with Python 3.10+ and NumPy:

```sh
python3 ml/train.py      # trains Model C + baselines A/B → ml/artifacts/
python3 ml/evaluate.py    # hysteresis sweep + ablation study → ml/artifacts/evaluation.json
python3 -m unittest discover -s ml/tests -v
```

Training runs in pure NumPy (~5-10 seconds on CPU) and is device-portable.
No Apple MLX, PyTorch or GPU is required.

## Data and model

120 independent 300-second simulated sessions provide 72,000 receiver readings
across 5 GNSS scenarios (open sky, partial obstruction, urban multipath, dropout,
mixed). 80 sessions train, 20 validate (checkpoint selection + grid search), and
20 test. Splits are whole-session and stratified by scenario; no session is used
across more than one partition.

Per-receiver raw channels (indices of the `x` array):

| Index | Feature        | Description                      | Units   |
|-------|----------------|----------------------------------|---------|
| 0     | satellites     | Visible satellite count          | count   |
| 1     | hdop           | Horizontal dilution of precision | —       |
| 2     | cn0_dbhz       | Signal-quality proxy (C/N0)      | dB-Hz   |
| 3     | age_s          | Measurement age / staleness      | seconds |
| 4     | step_m         | Position change since last sample| metres  |
| 5     | time_err_s     | Synchronization reliability      | seconds |
| 6     | receiver_id    | 0 or 1                           | —       |

`time_err_s` (index 5) is a synthetic stand-in for the live server's
clock-skew reliability check; it degrades under blockage and during stale
intervals. **Channel 6 (receiver_id) is excluded from the ML conditioning
features;** it is only present to identify the receiver in the raw array.

### Neural-network conditioning features X (16-D vector)

The 16-D vector fed to the MLP is built from the raw channels by
`features.build_features`:

| Columns | Group         | Description                                     |
|---------|---------------|-------------------------------------------------|
| 0-3     | R1 raw        | time_err_s, sat, cn0, hdop for receiver 1       |
| 4-7     | R2 raw        | time_err_s, sat, cn0, hdop for receiver 2       |
| 8-11    | Comparative   | Δtime_err, Δsat, Δcn0, Δhdop (R1 − R2)        |
| 12-13   | Satellite rate| Short-term satellite-count change rate per R    |
| 14-15   | C/N0 MA       | 5-sample causal moving average of C/N0 per R   |

### Reliability terms (input to the preserved equation)

The four reliability scores used in `W = αT + βS + γSNR + δDOP` are computed
per receiver via `features.reliability_scores`, using the exact same deterministic
transforms as the live `server.py`:

| Term | Normalization                                    | Range   |
|------|--------------------------------------------------|---------|
| T    | `clamp(1 − time_err_s / 5, 0, 1)`               | [0, 1]  |
| S    | `clamp(sat / 20, 0, 1)`                          | [0, 1]  |
| SNR  | `clamp(cn0 / 50, 0, 1)`                          | [0, 1]  |
| DOP  | `clamp(1 / hdop, 0, 1)`  (0 if hdop invalid)    | [0, 1]  |

## Architecture and training

The MLP is a compact 16 → 24 → 24 → 4 ReLU network with a softmax output.
It is trained with a differentiable soft-selection surrogate:

    P₁ = sigmoid(τ · (W₁ − W₂))
    Loss = E[ P₁·E₁ + (1 − P₁)·E₂ ]

where E₁, E₂ are the **actual** receiver errors and τ anneals from 2.0 → 0.5
during training. Deployment uses the deterministic argmax(W₁, W₂) instead.
Loss, backpropagation, and Adam are implemented in pure NumPy.

**Model A** — original fixed weights (0.25, 0.25, 0.30, 0.20).
**Model B** — a single globally optimized fixed weight set, found by exhaustive
grid search over the 3-simplex (step 0.05, 1,771 candidates), evaluated on
the validation split.
**Model C** — the ML dynamic-weight model described above.

## Evaluation

Evaluation metrics focus on receiver-selection quality, not neural-network loss:
selection accuracy (vs oracle), mean/median/p95/max selected error, RMSE,
switching count and stability, and average selection duration. Hysteresis
implements a configurable deadband (`|W₁ − W₂| < threshold` retains the current
selection) to suppress oscillation.

All results live in `artifacts/metrics.json` (main comparison) and
`artifacts/evaluation.json` (hysteresis sweep + ablation study).

## Limitations

- All data is SYNTHETIC and generated; no real GNSS/RTK validation.
- The timing channel is a synthetic abstraction of clock-synchronization error.
- SNR is a synthetic signal-quality proxy; per-satellite aggregation is not used.
- Weight interpretation is correlational; no causal claims are made.
- The model was trained on both-available receiver epochs only; single-receiver
  epochs are forced selections.
- Not integrated into the live ESP32/Raspberry Pi pipeline in this phase.

## Before hardware use

Integrate inference into `raspberrypi/server.py`, add ESP32 C/N0 transmission
(custom GSV parsing), retrain on real measurements, evaluate held-out real
sessions, and add connection-recovery and fresh-state handling.
