"""Synthetic, non-surveyed ENU trajectories and GNSS-like observations.

This simulator produces generated data only. The geographic reference is a
mathematical trajectory, not an RTK/surveyed measurement. Latent blockage and
multipath affect reported metrics and position independently; this is an
illustrative simulator for the adaptive-weight experiment, not a calibrated
physical GNSS error model.

Per-receiver channel layout (indices of the returned x array):
  0  satellites                 (count)
  1  hdop                       (>0, ~0.25-12)
  2  cn0_dbhz                   (signal-quality proxy)
  3  age_s                      (measurement age / staleness)
  4  step_m                     (position change since previous sample)
  5  time_err_s                 (synchronization/time reliability channel)
  6  receiver_id                (0 or 1)

channel 5 (time_err_s) is a synthetic stand-in for the live server's clock-skew
reliability term T. It jitters normally and degrades under blockage/dropouts.
"""
from pathlib import Path
import numpy as np

SCENARIOS = ['open_sky', 'partial_obstruction', 'urban_multipath', 'dropout', 'mixed']


def generate(seed=20260913, sessions=120, seconds=300):
    rng = np.random.default_rng(seed)
    features, errors, fixes, refs, observations, session_ids, scenarios = [], [], [], [], [], [], []
    for session in range(sessions):
        scenario = session % len(SCENARIOS)
        t = np.arange(seconds)
        moving = session % 2 == 1
        speed = rng.uniform(0.5, 8) if moving else 0
        heading = rng.uniform(0, 2*np.pi)
        reference = np.column_stack((speed*t*np.cos(heading), speed*t*np.sin(heading)))
        if moving:
            reference += np.column_stack((10*np.sin(t/25), 10*np.cos(t/25)))
        # Arbitrary mathematical geographic origin, not a surveyed/RTK location.
        origin = np.array([10 + rng.uniform(-0.2, 0.2), 76 + rng.uniform(-0.2, 0.2)])
        scale = np.array([111320., 111320.*np.cos(np.radians(origin[0]))])
        ref_ll = origin + reference[:, ::-1] / scale
        common = np.zeros((seconds, 2))
        for i in range(1, seconds):
            common[i] = .97*common[i-1] + rng.normal(0, .12, 2)
        pair_x, pair_y, pair_fix, pair_obs = [], [], [], []
        severity = [0.03, .45, .8, .35, rng.uniform(.05, .9)][scenario]
        for receiver in range(2):
            quality = rng.uniform(.75, 1.35) * (1.1 if receiver == 0 else .9)
            blocked = np.clip(severity + .2*np.sin(t/rng.uniform(15, 60)+rng.uniform(0, 6))
                              + rng.normal(0, .08, seconds), 0, 1)
            sats = np.clip(np.rint(27 - 20*blocked + rng.normal(0, 3, seconds)), 0, 36)
            hdop = np.clip(.45 + 3*blocked + rng.lognormal(-2, .6, seconds)
                           + rng.normal(0, .15, seconds), .25, 12)
            cn0 = np.clip(47 - 19*blocked + rng.normal(0, 3, seconds), 10, 55)
            # Synthetic synchronization/time quality: small jitter normally,
            # degradation under blockage, large drift during stale intervals.
            time_err = np.clip(.02 + rng.exponential(.05, seconds)
                               + 2.5*blocked*blocked
                               + rng.normal(0, .02, seconds), 0, 30)
            # Hidden multipath is imperfectly explained by reported metrics.
            bias = np.zeros((seconds, 2))
            for i in range(1, seconds):
                bias[i] = .96*bias[i-1] + rng.normal(0, .15 + severity*.65, 2)
            noise = rng.normal(size=(seconds, 2)) * (quality*(.55+blocked*4))[:, None]
            position = reference + common + bias*quality + noise
            spikes = rng.random(seconds) < (.005 + severity*.035)
            position[spikes] += rng.normal(0, 12, (int(spikes.sum()), 2))
            fix = rng.random(seconds) > (.002 + .02*severity)
            age = rng.uniform(.02, .8, seconds)
            if scenario in [3, 4]:
                start = int(rng.integers(40, seconds-30))
                duration = int(rng.integers(5, 25))
                position[start:start+duration] = position[start-1]
                age[start:start+duration] = np.arange(1, duration+1)
                fix[start+duration//2:start+duration] = False
                # Timing reliability also breaks during the stale window.
                time_err[start:start+duration] = np.maximum(
                    time_err[start:start+duration], np.linspace(2.5, 12, duration))
            step = np.r_[0, np.linalg.norm(np.diff(position, axis=0), axis=1)]
            x = np.column_stack((sats, hdop, cn0, age, step, time_err, np.full(seconds, receiver)))
            pair_x.append(x)
            pair_y.append(np.linalg.norm(position-reference, axis=1))
            pair_fix.append(fix)
            pair_obs.append(origin + position[:, ::-1] / scale)
        features.append(np.stack(pair_x, axis=1))
        errors.append(np.stack(pair_y, axis=1))
        fixes.append(np.stack(pair_fix, axis=1))
        observations.append(np.stack(pair_obs, axis=1))
        refs.append(ref_ll)
        session_ids.extend([session]*seconds)
        scenarios.extend([scenario]*seconds)
    return dict(x=np.concatenate(features).astype('float32'),
                error_m=np.concatenate(errors).astype('float32'),
                fix_valid=np.concatenate(fixes), reference_latlon=np.concatenate(refs),
                observed_latlon=np.concatenate(observations),
                session=np.array(session_ids), scenario=np.array(scenarios),
                synthetic=np.array(True))


if __name__ == '__main__':
    out = Path(__file__).parent/'artifacts'
    out.mkdir(exist_ok=True)
    data = generate()
    np.savez_compressed(out/'synthetic_readings.npz', **data)
    print('Saved', data['x'].shape[0]*2, 'SYNTHETIC receiver readings')