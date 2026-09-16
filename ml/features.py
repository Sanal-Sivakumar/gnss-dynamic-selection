"""Feature engineering for the adaptive-weight reliability experiment.

Two quantities are distinguished clearly:

1. Reliability terms -- the normalized indicators T, S, SNR, DOP that the
   preserved equation W = a*T + b*S + c*SNR + d*DOP combines. They reuse the
   exact deterministic scoring transforms of the live engine (server.py):
       T   = clamp(1 - time_err_s/5, 0, 1)      (sync reliability)
       S   = clamp(sat/20, 0, 1)                (satellite-count reliability)
       SNR = clamp(cn0_dbhz/50, 0, 1)           (signal-quality reliability)
       DOP = clamp(1/hdop, 0, 1)                (geometry reliability)

2. Conditioning features X -- the raw GNSS indicators (plus comparative and
   temporal context) that the neural network uses to predict [a, b, c, d].
   Everything here is available during real-time inference.

All offline code is SYTHETIC/generated. Nothing here implies real measurements.
"""
import numpy as np

# Raw per-receiver channel order produced by generate.py.
RAW_FEATURES = ['satellites', 'hdop', 'cn0_dbhz', 'age_s', 'step_m', 'time_err_s', 'receiver_id']

# Neural-network conditioning features X (order of the exported 16-D vector).
FEATURE_NAMES = [
    'time_err_s_1', 'sat_1', 'cn0_dbhz_1', 'hdop_1',
    'time_err_s_2', 'sat_2', 'cn0_dbhz_2', 'hdop_2',
    'dt_time_err_s', 'dt_sat', 'dt_cn0_dbhz', 'dt_hdop',
    'sat_rate_1', 'sat_rate_2', 'cn0_ma_1', 'cn0_ma_2',
]

# Feature subsets used by the ablation study. Weights are always the 4-D
# softmax output; only the conditioning input changes between rows.
FEATURE_GROUPS = {
    'T': [0, 4],
    'S': [1, 5],
    'SNR': [2, 6],
    'DOP': [3, 7],
    'delta': [8, 9, 10, 11],
    'temporal': [12, 13, 14, 15],
}


def eligible(x):
    """Per-receiver experimental validity mask from raw channels."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != len(RAW_FEATURES):
        raise ValueError('Expected raw channels: ' + ', '.join(RAW_FEATURES))
    return (np.isfinite(x).all(axis=-1)
            & (x[..., 0] >= 4) & (x[..., 1] > 0) & (x[..., 1] < 99.99)
            & (x[..., 2] > 0) & (x[..., 3] >= 0) & (x[..., 3] <= 2)
            & (x[..., 4] >= 0) & (x[..., 5] >= 0) & (x[..., 5] < 60)
            & np.isin(x[..., 6], [0, 1]))


def reliability_scores(x):
    """Normalized T, S, SNR, DOP terms, shape (..., 2, 4), each in [0, 1].

    Uses the same deterministic scoring transforms as the live server.
    """
    x = np.asarray(x, dtype=float)
    sat, hdop, cn0, time_err = x[..., 0], x[..., 1], x[..., 2], x[..., 5]
    t = np.clip(1.0 - time_err / 5.0, 0.0, 1.0)
    s = np.clip(sat / 20.0, 0.0, 1.0)
    snr = np.clip(cn0 / 50.0, 0.0, 1.0)
    dop = np.where((hdop <= 0) | (hdop >= 99.99), 0.0, np.clip(1.0 / hdop, 0.0, 1.0))
    return np.stack([t, s, snr, dop], axis=-1)


def _session_start_masks(session):
    session = np.asarray(session)
    return np.r_[True, (session[1:] != session[:-1])]


def build_features(x, session):
    """16-D conditioning vector per timestamp, including per-session temporal
    context (satellite-count rate of change and a causal 5-sample C/N0 moving
    average). Temporal context never crosses session boundaries and is causal,
    so it is available during real-time inference.
    """
    x = np.asarray(x, dtype=float)
    r1, r2 = x[:, 0], x[:, 1]
    t1, s1, c1, d1 = r1[:, 5], r1[:, 0], r1[:, 2], r1[:, 1]
    t2, s2, c2, d2 = r2[:, 5], r2[:, 0], r2[:, 2], r2[:, 1]

    raw = np.stack([t1, s1, c1, d1, t2, s2, c2, d2], axis=-1)
    dif = np.stack([t1 - t2, s1 - s2, c1 - c2, d1 - d2], axis=-1)

    # Satellite-count rate of change, clipped, reset at each session start.
    ds1 = np.clip(np.r_[0.0, np.diff(s1)], -8.0, 8.0)
    ds2 = np.clip(np.r_[0.0, np.diff(s2)], -8.0, 8.0)
    ds1[_session_start_masks(session)] = 0.0
    ds2[_session_start_masks(session)] = 0.0

    # Causal trailing 5-sample moving average of C/N0, per session.
    def moving_average(v, w=5):
        n = len(v)
        if n == 0:
            return np.array([], dtype=float)
        idx = np.arange(n)
        counts = np.minimum(idx + 1, w)
        csum = np.cumsum(np.r_[0.0, v])
        starts = np.maximum(0, idx + 1 - counts)
        return (csum[idx + 1] - csum[starts]) / counts

    m1 = moving_average(c1)
    m2 = moving_average(c2)

    return np.column_stack([raw, dif, ds1, ds2, m1, m2]).astype('float32')