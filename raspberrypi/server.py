"""Flask web server + real-time receiver-selection engine.

Pipeline:  ESP32 frame (serial)  ->  gnss_engine.DecisionEngine
           (parse, reliability terms, ML dynamic weights with hysteresis and
           fixed-weight fallback, JSONL decision logging)
        ->  /location (JSON)     ->  Leaflet dashboard.

Run with hardware:     python3 server.py
Run simulated frames:  python3 server.py --simulate
"""
import argparse
import sys
import threading
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from flask import Flask, jsonify, render_template  # noqa: E402

from gnss_engine import (  # noqa: E402
    DecisionEngine, MODEL_PATH, LOG_DIR, parse_frame,
)

app = Flask(__name__)

engine = None


# ---------------- FRAME SOURCES ----------------
def serial_frames(port, baud, reconnect_delay=5.0):
    """Yield decoded serial lines; reopen the device on failure."""
    try:
        import serial
    except ImportError:
        raise SystemExit(
            "pyserial is required for hardware mode. Install it with:\n"
            "    pip3 install pyserial\n"
            "(or run with --simulate for hardware-free testing)") from None
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            print(f"[engine] serial connected on {port} @ {baud}")
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    yield line
        except (serial.SerialException, OSError) as exc:
            print(f"[engine] serial error: {exc}; retrying in {reconnect_delay}s")
            time.sleep(reconnect_delay)
        finally:
            try:
                ser.close()
            except Exception:  # noqa: BLE001
                pass


def simulated_frames():
    """Yield ESP32-format frames at ~1 Hz for hardware-free testing."""
    from scripts.emulate_esp32 import FrameSimulator
    sim = FrameSimulator()
    for frame in sim.iter_frames():
        yield frame


# ---------------- ENGINE LOOP ----------------
def navigation_engine(engine, frame_iter):
    while True:
        try:
            line = frame_iter.__next__().strip()
            if "|" not in line:
                continue
            r0, r1 = parse_frame(line)
            decision = engine.update(r0, r1)
            _debug_print(decision)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            print("[engine] error:", exc, flush=True)


def _debug_print(d):
    if d is None or d["selected"] is None:
        print("\n===== GNSS STATUS =====\nSelected: none\n======================\n")
        return
    i = d["instrument"]
    ml = "ML" if d["ml_used"] else "FIXED"
    print("\n===== GNSS STATUS =====")
    print(f"GPS  -> Sat:{i['sat'][0]} Hdop:{i['hdop'][0]:.2f} Snr:{i['snr'][0]} W:{d['w_scores'][0]:.3f}")
    print(f"GNSS -> Sat:{i['sat'][1]} Hdop:{i['hdop'][1]:.2f} Snr:{i['snr'][1]} W:{d['w_scores'][1]:.3f}")
    print(f"weights[{ml}] a:{d['weights'][0]:.2f} b:{d['weights'][1]:.2f} "
          f"c:{d['weights'][2]:.2f} d:{d['weights'][3]:.2f}")
    print(f"Selected: {d['source']}  reason: {d['reason']}")
    print("======================", flush=True)


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/location")
def location():
    return jsonify(engine.best_location)


# ---------------- MAIN ----------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GNSS dynamic selection server")
    parser.add_argument("--simulate", action="store_true",
                        help="feed simulated ESP32 frames instead of serial")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial device")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=5000)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--hysteresis", type=float, default=0.05)
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    return parser.parse_args(argv)


def main(argv=None):
    global engine
    args = parse_args(argv)
    engine = DecisionEngine(model_path=args.model, hysteresis=args.hysteresis,
                            log_dir=args.log_dir)
    if not engine.model_ready:
        print(f"[engine] WARNING: ML model not loaded ({engine.model_error}); "
              f"using fixed-weight fallback only.")
    else:
        print("[engine] ML dynamic-weight model loaded.")

    frame_iter = simulated_frames() if args.simulate else serial_frames(args.port, args.baud)
    t = threading.Thread(target=navigation_engine, args=(engine, frame_iter), daemon=True)
    t.start()

    if args.simulate:
        print(f"[engine] simulated frame source active (~1 Hz). Dash: http://127.0.0.1:{args.web_port}")

    app.run(host=args.host, port=args.web_port)


if __name__ == "__main__":
    main()