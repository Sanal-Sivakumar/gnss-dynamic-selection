# 🚀 Multi-GNSS Dynamic Receiver Selection System

![Banner](docs/banner.png)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20RaspberryPi-orange)
![GitHub stars](https://img.shields.io/github/stars/Sanal-Sivakumar/gnss-dynamic-selection?style=social)

---

A low-cost, real-time embedded system that evaluates and dynamically selects the most reliable GNSS receiver. The original reliability equation is **preserved unchanged**; its coefficients are now **learned adaptively** from current GNSS conditions by a compact machine-learning model.

---

## 📌 Overview

This project implements a **multi-receiver GNSS reliability assessment system** using an **ESP32** and **Raspberry Pi 4**. It continuously analyzes data from two GNSS receivers and selects the most reliable positioning source in real time using:

- the original **reliability equation** `W = αT + βS + γSNR + δDOP` (argmax selection),
- an **adaptive weight learner** (Model C): a small MLP maps 16 conditioning features to `[α, β, γ, δ] = f(X)` with a softmax constraint (`α + β + γ + δ = 1`),
- a **hysteresis deadband** so near-equal receivers do not cause rapid switching,
- a **fixed-weight fallback** (the original 0.25 / 0.25 / 0.30 / 0.20) if the model is unavailable.

---

## 🧠 System Architecture

![Architecture](docs/architecture.png)

```
GNSS Receivers → ESP32 (UART + C/N0 tracker) → Raspberry Pi DecisionEngine → Web Dashboard
                                                   ▲
                       ml/ training → model.npz (16 features → αβγδ) ────────┘
```

**Live components**

- `esp32/esp32_gnss.ino` — two UART parsing streams, per-satellite C/N0 from GSV sentences, 1 Hz 7-field frames:
  `GPS,time,lat,lon,sat,hdop,snr|GNSS,time,lat,lon,sat,hdop,snr`
- `raspberrypi/gnss_engine.py` — decision engine: parses frames, computes reliability terms, applies adaptive weights with hysteresis + fallback, logs every decision to `raspberrypi/logs/decisions.jsonl`
- `raspberrypi/server.py` — Flask service with `--simulate` mode (no hardware needed) and serial reconnection
- `raspberrypi/scripts/emulate_esp32.py` — frame simulator / virtual serial (pty) source for testing

**Offline ML experiment** (`ml/`, see [ml/README.md](ml/README.md) and [ml/RESULTS.md](ml/RESULTS.md))

- 72,000 synthetic receiver readings from 120 sessions (80 / 20 / 20 stratified whole-session split), pure NumPy, fixed seed
- **Model A** – original fixed weights · **Model B** – grid-search optimized fixed set (α=0.10, β=0.20, γ=0.00, δ=0.70) · **Model C** – learned dynamic weights

### Held-out test results (5,986 available epochs, synthetic study)

| Selector                    | vs oracle | Mean (m) | Median (m) | ρ95 (m) | Switches |
| --------------------------- | --------: | -------: | ---------: | ------: | -------: |
| Model A (original fixed)    |     61.3% |    3.509 |      2.556 |   9.279 |     1788 |
| Model B (optimized fixed)   |     61.6% |    3.488 |      2.538 |   9.220 |     1768 |
| Model C (dynamic ML)        |     62.6% |    3.435 |      2.511 |   8.886 |     1767 |
| Model C + hysteresis (0.05) |     64.0% |    3.394 |      2.489 |   8.727 |     1018 |

> ⚠️ **These are controlled simulation results, not field/RTK accuracy.** Real-data validation with a trusted reference trajectory is the next step.

The learned weights are interpretable and context-dependent: DOP dominates under open-sky conditions (δ ≈ 0.998), while timing and satellite-count terms take over in urban multipath (α + β ≈ 0.97).

---

## ⚙️ Key Features

- 📡 Multi-receiver GNSS data acquisition with per-second C/N0 (GSV) tracking
- 🔄 Real-time receiver selection (1 Hz) with **adaptive learned weights**
- 🧮 Original reliability equation `W = αT + βS + γSNR + δDOP` preserved
- 🛡️ Hysteresis deadband + fixed-weight fallback + JSONL decision logging
- 🌍 Live visualization on web dashboard
- 💻 Lightweight NumPy inference (1,108 parameters) on the Raspberry Pi
- 💰 Low-cost system (< ₹2000 hardware)

---

## 💻 Quick Start (no hardware)

```bash
# 1. Install dependencies (Raspberry Pi side)
pip install -r raspberrypi/requirements.txt

# 2. Generate the ML artifacts (train Model A/B/C, export weights)
python3 ml/train.py      # writes ml/artifacts/model.npz + metrics.json
python3 ml/evaluate.py   # writes evaluation.json (hysteresis sweep, ablation)

# 3. Run the live server on simulated frames
python3 raspberrypi/server.py --simulate
# open http://127.0.0.1:5000

# 4. Run the tests
python3 -m unittest discover -s ml/tests
python3 -m unittest discover -s raspberrypi/tests
```

Use a pty-emulated serial instead of `--simulate`:

```bash
python3 raspberrypi/scripts/emulate_esp32.py --pty --seconds 3600
python3 raspberrypi/server.py --port /dev/ttysNNN
```

---

## 🔌 Hardware Setup

1. Open `esp32_gnss.ino` in Arduino IDE and install the **TinyGPSPlus** library
2. Wire two GNSS modules (e.g. NEO-M8N and Quectel L89) to UART1 (RX1 18 / TX1 5) and UART2 (RX2 26 / TX2 25)
3. Upload the sketch to the ESP32
4. Connect the ESP32 to the Raspberry Pi and run:

```bash
python3 raspberrypi/server.py --port /dev/ttyUSB0 --baud 115200
```

---

## 📊 Dashboard

- Real-time map visualization
- Active receiver + switch reason (initial / held / switched / reconfirmed)
- Model badge: **ML** weights vs **FIXED** fallback
- Live metrics: satellite count, SNR (C/N0), HDOP, dynamic weights `αβγδ`, reliability score

---

## ⚡ Tech Stack

- ESP32 (Arduino / C++)
- Python (Raspberry Pi) with NumPy inference
- Flask (Web server)
- PySerial (UART communication)
- Leaflet.js (Map UI)

---

## 🧪 Applications

- Autonomous navigation
- Precision agriculture
- Fleet tracking
- GNSS reliability research

---

## 🔍 Key Contribution

- Keeps the decision rule transparent (`W = αT + βS + γSNR + δDOP`) while making its **weights learned and context-dependent**
- Live integration: model → engine → dashboard, with hysteresis, fallback and logging
- Works on low-cost embedded hardware; avoids complex sensor fusion

---

## 🚧 Future Work

- Real-data validation: synchronized receiver logs + trusted reference trajectory (RTK / surveyed), retrain by environment
- Per-satellite C/N0 aggregation telemetry
- Higher update rates (5–10 Hz)
- Calibrated uncertainty tracking / optional sensor fusion

---

## 📁 Repository Layout

```
esp32/                  ESP32 firmware (TinyGPSPlus, C/N0 tracker)
raspberrypi/
  gnss_engine.py        live decision engine (adaptive weights, hysteresis, fallback, logging)
  server.py             Flask server (--simulate, serial, /location)
  scripts/emulate_esp32.py   frame simulator + pty virtual serial
  templates/            Leaflet dashboard
  tests/                engine unit tests
ml/                     offline ML experiment (train/evaluate predict metrics features)
  artifacts/            model.npz, metrics.json, evaluation.json, CSVs
tools/                  project report + viva PDF generator
docs/                   banner / architecture images
```

---

## 🙌 Acknowledgment

Developed as part of PNT Lab research work focused on low-cost GNSS reliability systems.

---

## 📎 License

MIT License