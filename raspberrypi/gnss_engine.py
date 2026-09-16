"""Real-time receiver-selection engine (Phase 6 integration).

The original reliability equation is preserved unchanged:

    W_i = a*T_i + b*S_i + c*SNR_i + d*DOP_i      Best Source = argmax(W)

The ML model is an ADAPTIVE WEIGHT LEARNER: it outputs [a,b,c,d] = f(X) given
the current GNSS conditions (see ml/). If inference is unavailable or produces
abnormal output, the engine falls back to the original fixed coefficients
(0.25, 0.25, 0.30, 0.20). A configurable deadband (hysteresis) prevents rapid
oscillation between receivers. Every decision is logged to a JSONL file.

Frame format from the ESP32 (see esp32/esp32_gnss.ino):
    GPS,time,lat,lon,sat,hdop,snr | GNSS,time,lat,lon,sat,hdop,snr
lat/lon are NO_FIX when a receiver has no fix; snr is the 1 s average C/N0.
The engine also accepts the legacy 6-field frame (without SNR); it then uses
the fixed-weight fallback path.
"""
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'ml'))

from features import build_features, reliability_scores  # noqa: E402
from predict import DynamicWeights, FIXED_FALLBACK_WEIGHTS  # noqa: E402

MODEL_PATH = Path(os.environ.get('GNSS_MODEL_PATH',
                  str(REPO_ROOT / 'ml' / 'artifacts' / 'model.npz')))
HYSTERESIS_THRESHOLD = float(os.environ.get('GNSS_HYSTERESIS', '0.05'))
LOG_DIR = Path(os.environ.get('GNSS_LOG_DIR', str(REPO_ROOT / 'raspberrypi' / 'logs')))

NO_FIX = 'NO_FIX'
HDOP_INVALID = 99.99
SOURCES = ['GPS', 'GNSS']


def parse_time(t):
    try:
        return datetime.strptime(t, "%H:%M:%S")
    except Exception:
        return None


def get_time_error(gps_time):
    if gps_time is None:
        return None
    now = datetime.now(timezone.utc)
    gps_time = gps_time.replace(year=now.year, month=now.month, day=now.day,
                                tzinfo=timezone.utc)
    return abs((gps_time - now).total_seconds())


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_frame(line):
    """Split a serial frame into two receiver measurement dictionaries."""
    if '|' not in line:
        return None
    gps_part, gnss_part = line.split('|')

    def parse(part):
        fields = part.strip().replace('GPS,', '').replace('GNSS,', '').split(',')
        rec = {'time': fields[0] if len(fields) > 0 else '00:00:00',
               'lat': fields[1] if len(fields) > 1 else NO_FIX,
               'lon': fields[2] if len(fields) > 2 else NO_FIX,
               'sat': int(fields[3]) if len(fields) > 3 else 0,
               'hdop': float(fields[4]) if len(fields) > 4 else HDOP_INVALID,
               'snr': None}
        if len(fields) > 5 and fields[5] not in ('', NO_FIX):
            try:
                rec['snr'] = float(fields[5])
            except ValueError:
                rec['snr'] = None
        return rec

    return parse(gps_part), parse(gnss_part)


def reliability_W(weights, time_err, sat, hdop, snr):
    """W = a*T + b*S + c*SNR + d*DOP using the normalized reliability terms."""
    t = 0.0 if time_err is None else float(np.clip(1.0 - time_err / 5.0, 0.0, 1.0))
    s = float(np.clip(sat / 20.0, 0.0, 1.0))
    snr_score = 0.0 if snr is None else float(np.clip(snr / 50.0, 0.0, 1.0))
    dop = 0.0 if (hdop is None or hdop <= 0 or hdop >= HDOP_INVALID) else float(np.clip(1.0 / hdop, 0.0, 1.0))
    return float(np.asarray(weights, dtype=float) @ np.array([t, s, snr_score, dop]))


class DecisionEngine:
    """Stateful selector: ML dynamic weights with fixed-weight fallback and
    hysteresis. Thread-safe via a lock (server uses it from a background
    thread while the Flask thread reads the latest decision)."""

    def __init__(self, model_path=MODEL_PATH, hysteresis=HYSTERESIS_THRESHOLD,
                 log_dir=LOG_DIR):
        self.hysteresis = float(hysteresis)
        self.model = None
        self.model_error = None
        try:
            self.model = DynamicWeights(model_path)
        except Exception as exc:  # noqa: BLE001 - model failure must not stop the engine
            self.model_error = str(exc)
        self._lock = threading.Lock()
        self._current = None
        self._history = deque(maxlen=5)
        self._prev_pos = [None, None]
        self._first_seen = [None, None]
        self._last_seen = [None, None]
        self.last_decision = None
        self._log_path = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / 'decisions.jsonl'

    # ---------------- public API ----------------
    @property
    def model_ready(self):
        return self.model is not None

    @property
    def current(self):
        return self._current

    @property
    def best_location(self):
        with self._lock:
            d = self.last_decision
        return {
            'lat': None if d is None or d['selected'] is None else d['lat'],
            'lon': None if d is None or d['selected'] is None else d['lon'],
            'source': 'INIT' if d is None or d['selected'] is None else SOURCES[d['selected']],
            'selected': None if d is None else d['selected'],
            'weights': None if d is None else d['weights'],
            'w_scores': None if d is None else d['w_scores'],
            'ml_used': None if d is None else d['ml_used'],
            'reason': None if d is None else d['reason'],
            'instrument': None if d is None else d['instrument'],
            'hysteresis': self.hysteresis,
        }

    def update(self, r0, r1, now=None):
        """Process one ESP32 frame and return the decision dictionary."""
        now = datetime.now(timezone.utc) if now is None else now
        mono = time.monotonic()
        recs = []
        for i, r in enumerate((r0, r1)):
            recs.append(self._build_receiver(r, i, mono, now))
        with self._lock:
            decision = self._decide(recs, now)
            self.last_decision = decision
        return decision

    # ---------------- internals ----------------
    def _build_receiver(self, r, receiver_id, mono, now):
        t = parse_time(r.get('time'))
        time_err = get_time_error(t)
        lat = None if r.get('lat') in (None, NO_FIX, '') else float(r['lat'])
        lon = None if r.get('lon') in (None, NO_FIX, '') else float(r['lon'])
        hdop = float(r.get('hdop', HDOP_INVALID))
        snr = r.get('snr')  # None until the firmware provides C/N0
        sat = int(r.get('sat', 0))

        age, step = 0.0, 0.0
        if lat is not None and lon is not None:
            if self._first_seen[receiver_id] is None:
                self._first_seen[receiver_id] = mono
            age = max(0.0, mono - (self._last_seen[receiver_id] or mono))
            self._last_seen[receiver_id] = mono
            prev = self._prev_pos[receiver_id]
            if prev is not None:
                step = haversine_m(prev[0], prev[1], lat, lon)
            self._prev_pos[receiver_id] = (lat, lon)

        raw_time_err = 30.0 if time_err is None else time_err
        # ml feature order: [sat, hdop, cn0, age, step, time_err, receiver_id]
        raw = [sat, hdop, (snr if snr is not None else 0.0), age, step, raw_time_err, receiver_id]
        fixed_w = reliability_W(FIXED_FALLBACK_WEIGHTS, time_err, sat, hdop, snr)
        return {'receiver_id': receiver_id, 'lat': lat, 'lon': lon, 'sat': sat,
                'hdop': hdop, 'snr': snr, 'time_err': time_err, 'age': age,
                'step': step, 'raw': raw, 'fixed_w': fixed_w}

    def _available(self, rec):
        return (rec['lat'] is not None and rec['lon'] is not None
                and rec['sat'] >= 4 and 0.0 < rec['hdop'] < HDOP_INVALID
                and rec['age'] <= 2.0)

    def _decide(self, recs, now):
        available = [self._available(r) for r in recs]
        if not any(available):
            self._current = None
            return {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'selected': None, 'source': None,
                    'reason': 'none', 'available': available, 'ml_used': False,
                    'lat': None, 'lon': None, 'weights': None, 'w_scores': None,
                    'fixed_w': None, 'instrument': None, 'features': None}

        ml_ready = (self.model is not None and available[0] and available[1]
                    and recs[0]['snr'] is not None and recs[1]['snr'] is not None
                    and recs[0]['snr'] > 0 and recs[1]['snr'] > 0)

        if ml_ready:
            self._history.append(np.array([recs[0]['raw'], recs[1]['raw']], dtype=float))
            weights, ml_used = self._ml_weights()
        else:
            if available[0] and available[1]:
                self._history.append(np.array([recs[0]['raw'], recs[1]['raw']], dtype=float))
            weights, ml_used = FIXED_FALLBACK_WEIGHTS, False

        terms = reliability_scores(np.array([[recs[0]['raw'], recs[1]['raw']]], dtype=float))[0]
        w_scores = (terms @ np.asarray(weights)).astype(float)
        fixed_W = [recs[0]['fixed_w'], recs[1]['fixed_w']]

        selected, reason = self._select(w_scores, available)

        lat = lon = None
        if selected is not None:
            lat, lon = recs[selected]['lat'], recs[selected]['lon']

        decision = {
            'timestamp': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'selected': selected,
            'source': None if selected is None else SOURCES[selected],
            'reason': reason,
            'available': available,
            'instrument': {
                'sat': [recs[0]['sat'], recs[1]['sat']],
                'snr': [recs[0]['snr'], recs[1]['snr']],
                'hdop': [recs[0]['hdop'], recs[1]['hdop']],
                'time_err': [recs[0]['time_err'], recs[1]['time_err']],
                'age': [recs[0]['age'], recs[1]['age']],
                'step': [recs[0]['step'], recs[1]['step']],
            },
            'lat': lat,
            'lon': lon,
            'fixed_w': fixed_W,
            'w_scores': [float(w_scores[0]), float(w_scores[1])],
            'weights': [float(v) for v in np.asarray(weights).tolist()],
            'features': self._last_features(),
            'ml_used': ml_used,
            'hysteresis': self.hysteresis,
        }
        self._current = selected
        self._log(decision)
        return decision

    def _ml_weights(self):
        try:
            stack = np.stack(list(self._history)) if self._history else None
            if stack is None:
                raise ValueError('no feature history')
            X = build_features(stack, np.zeros(len(stack)))[-1]
            w = self.model.weights(np.atleast_2d(X))[0]
            if not bool(np.isfinite(w).all()) or not bool(np.isclose(w.sum(), 1.0, atol=1e-3)):
                raise ValueError('abnormal predicted weights')
            return np.asarray(w, dtype=float), True
        except Exception:
            return FIXED_FALLBACK_WEIGHTS, False

    def _last_features(self):
        if not self._history:
            return None
        stack = np.stack(list(self._history))
        return build_features(stack, np.zeros(len(stack)))[-1].tolist()

    def _select(self, w_scores, available):
        cand = [i for i in range(2) if available[i]]
        prev = self._current
        if prev is not None and prev in cand and len(cand) == 2:
            if abs(float(w_scores[0] - w_scores[1])) < self.hysteresis:
                return prev, 'held'
        best = int(cand[np.argmax(w_scores[cand])])
        if prev is None:
            return best, 'initial'
        if best != prev:
            return best, 'switched'
        return best, 'reconfirmed'

    def _log(self, decision):
        if self._log_path is None:
            return
        try:
            with self._log_path.open('a') as handle:
                handle.write(json.dumps(decision) + '\n')
        except OSError:
            pass