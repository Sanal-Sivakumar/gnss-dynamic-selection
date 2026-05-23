from flask import Flask, jsonify, render_template
import serial
from datetime import datetime
import threading

PORT = "/dev/ttyUSB0"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

app = Flask(__name__)

best_location = {
    "lat": 10.051602,
    "lon": 76.331617,
    "source": "INIT"
}

# ---------------- TIME PARSER ----------------
def parse_time(t):
    try:
        return datetime.strptime(t, "%H:%M:%S")
    except:
        return None

# ---------------- TIME ERROR ----------------
def get_time_error(gps_time):
    if gps_time is None:
        return None
    now = datetime.utcnow()
    gps_time = gps_time.replace(year=now.year, month=now.month, day=now.day)
    return abs((gps_time - now).total_seconds())

# ---------------- RELIABILITY ----------------
def compute_weight(time_err, sat, hdop, snr_avg):

    # --- Satellite Score ---
    sat_score = min(sat / 20, 1)

    # --- Time Score ---
    if time_err is None:
        time_score = 0
    else:
        time_score = max(0, 1 - (time_err / 5))

    # --- Geometry Score (HDOP) ---
    if hdop == 0 or hdop == 99.99:
        geom_score = 0
    else:
        geom_score = min(1 / hdop, 1)

    # --- SNR Score ---
    if snr_avg is None:
        snr_score = 0
    else:
        snr_score = min(snr_avg / 50, 1)

    # --- FINAL WEIGHT ---
    weight = (
        0.25 * time_score +
        0.25 * sat_score +
        0.30 * snr_score +
        0.20 * geom_score
    )

    return weight

# ---------------- MAIN ENGINE ----------------
def navigation_engine():
    global best_location

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()

            if "|" not in line:
                continue

            gps_part, gnss_part = line.split("|")

            gps_data = gps_part.replace("GPS,", "").split(",")
            gnss_data = gnss_part.replace("GNSS,", "").split(",")

            gps_time = parse_time(gps_data[0])
            gnss_time = parse_time(gnss_data[0])

            gps_lat = None
            gps_lon = None
            gps_sat = 0

            if gps_data[1] != "NO_FIX" and gps_data[2] != "NO_FIX":
                gps_lat = float(gps_data[1])
                gps_lon = float(gps_data[2])
                gps_sat = int(gps_data[3])

            gnss_lat = None
            gnss_lon = None
            gnss_sat = 0

            if gnss_data[1] != "NO_FIX" and gnss_data[2] != "NO_FIX":
                gnss_lat = float(gnss_data[1])
                gnss_lon = float(gnss_data[2])
                gnss_sat = int(gnss_data[3])

            # skip if both invalid
            if gps_lat is None and gnss_lat is None:
                continue

            gps_err = get_time_error(gps_time)
            gnss_err = get_time_error(gnss_time)

            gps_weight = compute_weight(gps_err, gps_sat)
            gnss_weight = compute_weight(gnss_err, gnss_sat)

            # --------- SELECTION ----------
            if gps_lat is not None and gps_weight >= gnss_weight:
                best_location["lat"] = gps_lat
                best_location["lon"] = gps_lon
                best_location["source"] = "GPS"

            elif gnss_lat is not None:
                best_location["lat"] = gnss_lat
                best_location["lon"] = gnss_lon
                best_location["source"] = "GNSS"

            # --------- DEBUG OUTPUT ----------
            print("\n===== GNSS STATUS =====")
            print(f"GPS  -> Sat:{gps_sat} Weight:{gps_weight:.2f}")
            print(f"GNSS -> Sat:{gnss_sat} Weight:{gnss_weight:.2f}")
            print("Selected:", best_location["source"])
            print("======================\n")

        except Exception as e:
            print("Error:", e)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/location")
def location():
    return jsonify(best_location)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    t = threading.Thread(target=navigation_engine)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=5000)