"""Hardware-less ESP32 frame source.

Two uses:
  * FrameSimulator -- yields ESP32-format frames at ~1 Hz so the decision
    engine and dashboard can be exercised without the physical receivers
    (used by `python3 server.py --simulate`).
  * --pty          -- additionally publish the frames to a virtual serial
    device (pseudo-terminal) so the unmodified serial code path in server.py
    can be tested end-to-end:

        python3 scripts/emulate_esp32.py --pty          # prints /dev/ttysNNN
        python3 server.py --port /dev/ttysNNN

All coordinates and measurements here are synthetic simulation values.
"""
import argparse
import math
import os
import pty
import sys
import time
from datetime import datetime, timezone

import numpy as np

PHASES = [
    # (sats_m, hdop_m, snr_m, sat_sigma)  open sky / obstructed / urban / dropout
    (18, 1.1, 44.0, 3),
    (9, 3.2, 33.0, 2),
    (6, 5.5, 27.0, 2),
    (4, 7.0, 22.0, 1),
]
USE_NO_FIX = {3: 0.15}   # dropout phase occasionally loses a fix


class FrameSimulator:
    def __init__(self, seed=20260913, origin=(10.051602, 76.331617), step_s=1.0):
        self.seed = seed
        self.origin = origin
        self.step_s = step_s
        self.rng = np.random.default_rng(seed)

    def iter_frames(self):
        deg = 111320.0
        lon_scale = math.cos(math.radians(self.origin[0]))
        heading = [self.rng.uniform(0, 2 * math.pi) for _ in range(2)]
        speed = [self.rng.uniform(0.5, 2.0) for _ in range(2)]
        pos_m = [[0.0, 0.0], [12.0, -6.0]]
        t = 0
        while True:
            phase = (t // 20) % len(PHASES)
            sats_m, hdop_m, snr_m, sat_sigma = PHASES[phase]
            frames = []
            for i in range(2):
                heading[i] += self.rng.normal(0, 0.05)
                pos_m[i][0] += speed[i] * math.cos(heading[i]) * self.step_s
                pos_m[i][1] += speed[i] * math.sin(heading[i]) * self.step_s
                lat = self.origin[0] + pos_m[i][1] / deg
                lon = self.origin[1] + pos_m[i][0] / (deg * lon_scale)
                no_fix = self.rng.random() < USE_NO_FIX.get(phase, 0.0)
                if no_fix:
                    lat, lon = "NO_FIX", "NO_FIX"
                sat = max(0, int(round(self.rng.normal(sats_m + (2 if i == 0 else 0), sat_sigma))))
                hdop = max(0.5, round(self.rng.normal(hdop_m + (0.2 if i == 1 else 0), 0.4), 2))
                snr = round(self.rng.normal(snr_m, 3.0), 0)
                frames.append((round(lat, 6) if not isinstance(lat, str) else lat,
                               round(lon, 6) if not isinstance(lon, str) else lon,
                               sat, hdop, snr))
            now = datetime.now(timezone.utc)
            stamp = now.strftime("%H:%M:%S")
            g = frames[0]
            n = frames[1]
            yield (f"GPS,{stamp},{g[0]},{g[1]},{g[2]},{g[3]:.2f},{g[4]:.0f}|"
                   f"GNSS,{stamp},{n[0]},{n[1]},{n[2]},{n[3]:.2f},{n[4]:.0f}")
            t += 1
            time.sleep(self.step_s)


def pty_publish(frames, throttle=1.0):
    master, slave = pty.openpty()
    device = os.ttyname(slave)
    print(f"Emulated ESP32 publishing on {device}", file=sys.stderr, flush=True)
    if not os.environ.get('GNSS_SIM_DEVICE_ONLY'):
        print(device, flush=True)
    for frame in frames:
        os.write(master, frame.encode() + b"\n")
        time.sleep(throttle)


def main():
    parser = argparse.ArgumentParser(description="Emulated dual-GNSS frame source")
    parser.add_argument("--pty", action="store_true",
                        help="publish frames to a pseudo-terminal serial device")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--seconds", type=int, default=120, help="frames to emit then exit")
    args = parser.parse_args()

    def take(n, it):
        for _ in range(n):
            yield next(it)

    sim = FrameSimulator(seed=args.seed)
    frames = take(args.seconds if args.seconds > 0 else 1 << 30, sim.iter_frames())
    if args.pty:
        pty_publish(frames)
    else:
        for frame in frames:
            print(frame, flush=True)


if __name__ == "__main__":
    main()