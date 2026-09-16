"""Training pipeline for the adaptive-weight reliability system (pure NumPy).

The ML component is an ADAPTIVE WEIGHT LEARNER, not a black-box selector: a
compact MLP maps conditioning features X to softmax coefficients
[a, b, c, d] = f(X), which plug into the preserved original reliability
equation   W_i = a*T_i + b*S_i + c*SNR_i + d*DOP_i   and selection remains
argmax(W).

Training objective (differentiable): the soft-selection expected positioning
error over a temperature-annealed soft distribution,
    P1 = sigmoid(tau * (W1 - W2)),    loss = P1*E1 + (1-P1)*E2,
where E1/E2 are the receivers' ACTUAL errors. The deployed rule is the
deterministic argmax. tau anneals from soft (temp_start) to sharp (temp_end).

Model A (original fixed weights) and Model B (globally optimized fixed weights
via grid search over the simplex) are produced for comparison. Preprocessing
(mean/std) is fitted on training sessions only; the test split is untouched
until final evaluation. All data is synthetic/generated.
"""
import csv
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from generate import generate, SCENARIOS
from features import build_features, reliability_scores, eligible, RAW_FEATURES, FEATURE_NAMES
from predict import select_scores, apply_hysteresis, DynamicWeights, WEIGHT_NAMES
from baselines import MODEL_A_WEIGHTS, grid_search_optimal_weights, reliability as fixed_reliability
from metrics import evaluate_selection, apply_hysteresis_sessions

OUT = Path(__file__).parent / 'artifacts'
OUT.mkdir(exist_ok=True)


def load_config(path=None):
    path = Path(path) if path else Path(__file__).parent / 'config.json'
    cfg = {
        'seed': 20260913, 'sessions': 120, 'seconds': 300,
        'split_per_scenario': [16, 4, 4],
        'fit': {'hidden': [24, 24], 'lr': 0.001, 'batch': 512,
                'max_epochs': 60, 'patience': 12, 'temp_start': 2.0, 'temp_end': 0.5},
        'grid_step': 0.05, 'hysteresis_thresholds': [0.0, 0.02, 0.05, 0.1],
        'ablation_budget_epochs': 30,
    }
    if path.exists():
        loaded = json.loads(path.read_text())
        cfg.update(loaded)
        cfg['fit'].update(loaded.get('fit', {}))
    return cfg


def session_split(seed, sessions, split_per_scenario, n_scenarios=5):
    """Stratified, whole-session split so no train/val/test session overlaps."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for scenario in range(n_scenarios):
        ids = np.arange(scenario, sessions, n_scenarios)
        rng.shuffle(ids)
        offset = 0
        for dest, n in zip(parts, split_per_scenario):
            dest.extend(ids[offset:offset + n].tolist())
            offset += n
    return parts


def init_params(in_dim, hidden, rng):
    dims = [in_dim] + list(hidden) + [4]
    p = {}
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        p['w%d' % (i + 1)] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
        p['b%d' % (i + 1)] = np.zeros(b)
    return p


def logits(p, X):
    z = np.maximum(X @ p['w1'] + p['b1'], 0)
    z = np.maximum(z @ p['w2'] + p['b2'], 0)
    return z @ p['w3'] + p['b3']


def softmax_weights(L):
    e = np.exp(L - L.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def selection_loss(weights, terms, E, tau):
    """Expected error under soft selection P1 = sigmoid(tau*(W1-W2))."""
    W = np.sum(np.asarray(terms) * np.asarray(weights)[..., None, :], axis=-1)
    t = np.clip(tau * (W[:, 0] - W[:, 1]), -40, 40)
    P1 = 1.0 / (1.0 + np.exp(-t))
    return np.mean(P1 * E[:, 0] + (1.0 - P1) * E[:, 1])


def loss_and_grad(p, X, terms, E, tau):
    B = X.shape[0]
    a1 = np.maximum(X @ p['w1'] + p['b1'], 0)
    a2 = np.maximum(a1 @ p['w2'] + p['b2'], 0)
    L = a2 @ p['w3'] + p['b3']
    w = softmax_weights(L)
    W = np.sum(terms * w[:, None, :], axis=-1)
    t = np.clip(tau * (W[:, 0] - W[:, 1]), -40, 40)
    P1 = 1.0 / (1.0 + np.exp(-t))
    P2 = 1.0 - P1
    loss = np.mean(P1 * E[:, 0] + P2 * E[:, 1])
    # dL/dW1 = g, dL/dW2 = -g
    g = tau * P1 * P2 * (E[:, 0] - E[:, 1])
    term_diff = terms[:, 0, :] - terms[:, 1, :]          # (B,4)
    gw = term_diff * g[:, None]                          # dL/dw
    dL = w * (gw - (gw * w).sum(axis=-1, keepdims=True)) # softmax backward

    gw3 = a2.T @ dL
    gb3 = dL.sum(axis=0)
    da2 = (dL @ p['w3'].T) * (a2 > 0)
    gw2 = a1.T @ da2
    gb2 = da2.sum(axis=0)
    da1 = (da2 @ p['w2'].T) * (a1 > 0)
    gw1 = X.T @ da1
    gb1 = da1.sum(axis=0)
    return float(loss), {'w1': gw1, 'b1': gb1, 'w2': gw2, 'b2': gb2, 'w3': gw3, 'b3': gb3}


def fit_dynamic_weights(Xtr, terms_tr, E_tr, Xval, terms_val, E_val, cfg, verbose=True):
    f = cfg['fit']
    mean = Xtr.mean(axis=0)
    std = np.maximum(Xtr.std(axis=0), 1e-6)
    Xs = (Xtr - mean) / std
    Xvs = (Xval - mean) / std
    rng = np.random.default_rng(cfg['seed'] + 1)
    p = init_params(Xs.shape[1], f['hidden'], rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, f['lr']
    best_p, best_epoch, best_loss, patience = None, 0, np.inf, 0
    history, step = [], 0
    n = len(Xs)
    idx = np.arange(n)
    for epoch in range(1, f['max_epochs'] + 1):
        tau = f['temp_start'] + (f['temp_end'] - f['temp_start']) * (epoch / f['max_epochs'])
        rng.shuffle(idx)
        for start in range(0, n, f['batch']):
            b = idx[start:start + f['batch']]
            _, grads = loss_and_grad(p, Xs[b], terms_tr[b], E_tr[b], tau)
            step += 1
            for k in p:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                mhat = m[k] / (1 - beta1 ** step)
                vhat = v[k] / (1 - beta2 ** step)
                p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        vl = float(selection_loss(softmax_weights(logits(p, Xvs)), terms_val, E_val, tau))
        history.append({'epoch': epoch, 'validation_expected_error_m': vl, 'temperature': tau})
        if vl < best_loss - 1e-5:
            best_loss, best_epoch, patience = vl, epoch, 0
            best_p = {k: v.copy() for k, v in p.items()}
        else:
            patience += 1
        if (verbose and (epoch == 1 or epoch % 10 == 0)):
            print(f'Epoch {epoch:3d}: validation expected error={vl:.4f} m', flush=True)
        if patience >= f['patience']:
            break
    return {'p': best_p, 'mean': mean, 'std': std, 'history': history,
            'best_epoch': best_epoch}


def export_csv(path, data, session_ids, scenario_ids, mask, split_name):
    fields = ['session', 'scenario', 'receiver'] + RAW_FEATURES + \
             ['error_m', 'fix_valid', 'reference_lat', 'reference_lon',
              'observed_lat', 'observed_lon']
    indices = np.flatnonzero(mask)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i in indices:
            for receiver in range(2):
                writer.writerow([int(session_ids[i]), SCENARIOS[int(scenario_ids[i])],
                                 receiver, *data['x'][i, receiver].tolist(),
                                 float(data['error_m'][i, receiver]), int(data['fix_valid'][i, receiver]),
                                 *data['reference_latlon'][i].tolist(),
                                 *data['observed_latlon'][i, receiver].tolist()])
    print(f'Exported {split_name}: {path.name}')


def condition_buckets(x, y, weights, available):
    """Mean learned weights per GNSS-condition regime (test, both available)."""
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    both = np.asarray(available).all(axis=-1)
    r1, r2 = x[:, 0], x[:, 1]
    sat = np.minimum(r1[:, 0], r2[:, 0])
    hdop = np.maximum(r1[:, 1], r2[:, 1])
    cn0 = np.minimum(r1[:, 2], r2[:, 2])
    t_err = np.maximum(r1[:, 5], r2[:, 5])
    buckets = {
        'high_satellite_count': sat >= 12,
        'low_satellite_count': sat < 8,
        'good_geometry': hdop < 1.5,
        'poor_geometry': hdop > 3.0,
        'high_snr': cn0 >= 42,
        'low_snr': cn0 < 34,
        'stable_timing': t_err < 1.0,
        'unstable_timing': t_err > 3.0,
    }
    out = {}
    for name, mask in buckets.items():
        sel = both & mask
        if sel.any():
            out[name] = {'n': int(sel.sum()),
                         'weights': {k: float(weights[sel, j].mean())
                                     for j, k in enumerate(WEIGHT_NAMES)}}
    return out


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg['seed'])
    data = generate(cfg['seed'], cfg['sessions'], cfg['seconds'])
    np.savez_compressed(OUT / 'synthetic_readings.npz', **data)

    x, y = data['x'], data['error_m']
    available = eligible(x) & data['fix_valid']
    F = build_features(x, data['session'])
    terms = reliability_scores(x)

    parts = session_split(cfg['seed'], cfg['sessions'], cfg['split_per_scenario'])
    train_msk = np.isin(data['session'], parts[0])
    val_msk = np.isin(data['session'], parts[1])
    test_msk = np.isin(data['session'], parts[2])

    both = available.all(axis=-1)
    btr, bva = both & train_msk, both & val_msk

    fit_cfg = {**cfg, 'seed': cfg['seed'] + 2}   # distinct seed stream per phase
    model = fit_dynamic_weights(F[btr], terms[btr], y[btr], F[bva], terms[bva], y[bva], fit_cfg)
    np.savez(OUT / 'model.npz', **model['p'], mean=model['mean'], std=model['std'])
    print('Best validation epoch:', model['best_epoch'])

    bestB, bestB_val_err, n_candidates = grid_search_optimal_weights(
        terms[bva], y[bva], available[bva], cfg['grid_step'])
    print('Model B (grid) validation mean error:', round(bestB_val_err, 4),
          'candidates:', n_candidates)

    wA = fixed_reliability(MODEL_A_WEIGHTS, terms)
    wB = fixed_reliability(bestB, terms)
    dm = DynamicWeights(OUT / 'model.npz')
    wC = dm.reliability(dm.weights(F), terms)

    test_rows = np.flatnonzero(test_msk)
    test_session = data['session'][test_rows]
    test_y = y[test_rows]
    test_avail = available[test_rows]

    def run(name, W, threshold=None):
        if threshold is None:
            dec = select_scores(W, available)[test_rows]
        else:
            dec = apply_hysteresis_sessions(W, available, data['session'], threshold)[test_rows]
        return evaluate_selection(dec, test_y, test_avail, test_session)

    results = {
        'original_fixed_weights_model_A': run('A', wA),
        'optimized_fixed_weights_model_B': run('B', wB),
        'dynamic_ml_weights_model_C': run('C', wC),
    }
    hyst0 = cfg['hysteresis_thresholds'][0]
    hyst = cfg['hysteresis_thresholds'][2]
    results['dynamic_ml_no_hysteresis'] = results['dynamic_ml_weights_model_C']
    results['dynamic_ml_with_hysteresis'] = run('C+h', wC, threshold=hyst)

    weightsC = dm.weights(F[test_msk])
    w_stats = {
        'overall': {'n': int(both[test_msk].sum()),
                    'weights': {k: float(weightsC[both[test_msk], j].mean())
                                for j, k in enumerate(WEIGHT_NAMES)}},
        'by_scenario': {SCENARIOS[s]: {'n': int((data['scenario'][test_msk] == s)[both[test_msk]].sum()),
                                       'weights': {k: float(
                                           weightsC[(data['scenario'][test_msk] == s) & both[test_msk], j].mean())
                                           for j, k in enumerate(WEIGHT_NAMES)}}
                        for s in range(5)},
        'by_condition': condition_buckets(x[test_msk], y[test_msk], weightsC, available[test_msk]),
    }

    per_scenario = {}
    for s, name in enumerate(SCENARIOS):
        rows = np.flatnonzero(test_msk & (data['scenario'] == s))
        per_scenario[name] = {
            'model_A': evaluate_selection(select_scores(wA, available)[rows], y[rows],
                                          available[rows], data['session'][rows]),
            'model_B': evaluate_selection(select_scores(wB, available)[rows], y[rows],
                                          available[rows], data['session'][rows]),
            'model_C': evaluate_selection(select_scores(wC, available)[rows], y[rows],
                                          available[rows], data['session'][rows]),
        }

    export_csv(OUT / 'training_data.csv', data, data['session'], data['scenario'],
               train_msk, 'training')
    export_csv(OUT / 'testing_data.csv', data, data['session'], data['scenario'],
               test_msk, 'testing')

    report = {
        'synthetic_only': True,
        'seed': cfg['seed'],
        'backend': 'numpy',
        'runtime': {'python': 'pys' if False else None},
        'numpy_version': importlib.metadata.version('numpy'),
        'model': {'type': 'dynamic_weight_learner',
                  'reliability_equation': 'W = a*T + b*S + c*SNR + d*DOP',
                  'output': WEIGHT_NAMES,
                  'constraint': 'softmax, sum=1'},
        'feature_names': FEATURE_NAMES,
        'raw_channels': RAW_FEATURES,
        'receiver_readings': int(x.shape[0] * 2),
        'split_sessions': dict(zip(['train', 'validation', 'test'], parts)),
        'decision_relevant_training_epochs_both_available': int(btr.sum()),
        'normalization_fitted_on': 'training sessions, both-available epochs',
        'best_epoch': model['best_epoch'],
        'temperature_schedule': [hist['temperature'] for hist in model['history']],
        'training_history': model['history'],
        'grid_search': {'step': cfg['grid_step'], 'candidates': n_candidates,
                        'best_validation_mean_error_m': bestB_val_err},
        'model_B_weights': {'alpha': float(bestB[0]), 'beta': float(bestB[1]),
                            'gamma': float(bestB[2]), 'delta': float(bestB[3])},
        'model_A_weights': {'alpha': float(MODEL_A_WEIGHTS[0]), 'beta': float(MODEL_A_WEIGHTS[1]),
                            'gamma': float(MODEL_A_WEIGHTS[2]), 'delta': float(MODEL_A_WEIGHTS[3])},
        'test': results,
        'hysteresis_threshold_used': hyst,
        'test_by_scenario': per_scenario,
        'learned_weights': w_stats,
        'hysteresis_thresholds_available': cfg['hysteresis_thresholds'],
        'limitations': [
            'All data is SYNTHETIC/generated; no real GNSS or RTK validation.',
            'Synthetic timing channel models synchronization quality; not a calibrated clock-error model.',
            'Trained only on timestamps where both receivers are available; single-receiver epochs are forced selections.',
            'Softmax coefficient constraint (sum=1, nonnegative) is a modeling choice, not an empirical law.',
            'C/N0 is a synthetic signal-quality proxy; per-satellite aggregation is not implemented.',
            'Weight interpretation is correlational; no causal claims are made.',
            'The neural network is separate from the neural-network-trained weight outputs; outputs a,b,c,d are a learned mapping, not the network parameters.',
            'Not integrated into the live ESP32/Raspberry Pi engine in this phase.',
        ],
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'model_A': results['original_fixed_weights_model_A'],
                      'model_B': results['optimized_fixed_weights_model_B'],
                      'model_C': results['dynamic_ml_weights_model_C'],
                      'model_C+hyst': results['dynamic_ml_with_hysteresis']}, indent=2))


if __name__ == '__main__':
    main()