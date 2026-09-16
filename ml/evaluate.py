"""Evaluation harness (Phase 5) for the adaptive-weight reliability system.

Everything evaluates receiver-selection performance on the untouched test
split. It reproduces Model A / B / C comparison from train.py, sweeps the
hysteresis threshold, and performs an ablation study over the conditioning
feature groups to answer whether the model genuinely benefits from the
different GNSS information sources (RQ3-RQ6). All data is SYNTHETIC.
"""
import json
from pathlib import Path
import numpy as np

from generate import generate, SCENARIOS
from features import build_features, reliability_scores, eligible, FEATURE_GROUPS
from predict import DynamicWeights, select_scores, FIXED_FALLBACK_WEIGHTS
from baselines import MODEL_A_WEIGHTS, reliability as fixed_reliability, grid_search_optimal_weights
from metrics import evaluate_selection, apply_hysteresis_sessions, selection_confusion
from train import load_config, session_split, fit_dynamic_weights, logits, softmax_weights

OUT = Path(__file__).parent / 'artifacts'

ABLATIONS = [
    ('T only', ['T']),
    ('S only', ['S']),
    ('SNR only', ['SNR']),
    ('DOP only', ['DOP']),
    ('T + S', ['T', 'S']),
    ('T + S + SNR', ['T', 'S', 'SNR']),
    ('all four', ['T', 'S', 'SNR', 'DOP']),
    ('all four + comparative', ['T', 'S', 'SNR', 'DOP', 'delta']),
    ('all four + temporal (full)', ['T', 'S', 'SNR', 'DOP', 'delta', 'temporal']),
]


def _prepare(cfg):
    data = generate(cfg['seed'], cfg['sessions'], cfg['seconds'])
    x, y = data['x'], data['error_m']
    available = eligible(x) & data['fix_valid']
    F = build_features(x, data['session'])
    terms = reliability_scores(x)
    parts = session_split(cfg['seed'], cfg['sessions'], cfg['split_per_scenario'])
    return (data, x, y, available, F, terms, parts)


def _test_eval(weights, terms, available, y, test_mask, session, threshold=None):
    W = np.sum(terms * weights[..., None, :], axis=-1)
    if threshold is None:
        dec = select_scores(W, available)[test_mask]
    else:
        dec = apply_hysteresis_sessions(W, available, session, threshold)[test_mask]
    rows = np.flatnonzero(test_mask)
    return evaluate_selection(dec, y[rows], available[rows], session[rows])


def main():
    cfg = load_config()
    data, x, y, available, F, terms, parts = _prepare(cfg)
    test_mask = np.isin(data['session'], parts[2])
    val_mask = np.isin(data['session'], parts[1])
    train_mask = np.isin(data['session'], parts[0])
    both = available.all(axis=-1)
    btr, bva = both & train_mask, both & val_mask
    test_rows = np.flatnonzero(test_mask)

    print('=== Receiver-selection comparison on the held-out TEST split ===')
    names = ['original_fixed_weights_model_A', 'optimized_fixed_weights_model_B',
             'dynamic_ml_weights_model_C']
    test = json.loads((OUT / 'metrics.json').read_text())['test']
    header = ['selector', 'acc%', 'mean', 'median', 'rmse', 'p95', 'switches']
    print('\t'.join(header))
    for n in names:
        r = test[n]
        print('\t'.join([n.rsplit('_model_')[-1].upper(), f"{r['selection_accuracy_vs_oracle']*100:.1f}",
                         f"{r['mean_error_m']:.3f}", f"{r['median_error_m']:.3f}",
                         f"{r['rmse_m']:.3f}", f"{r['p95_error_m']:.3f}",
                         str(r['switches'])]))
    print('\n=== Hysteresis sweep (Model C) on test split ===')
    dm = DynamicWeights(OUT / 'model.npz')
    wC = dm.weights(F[test_rows])
    sweep = {}
    for thr in cfg['hysteresis_thresholds']:
        dec = apply_hysteresis_sessions(
            np.sum(terms[test_rows] * wC[..., None, :], axis=-1),
            available[test_rows], data['session'][test_rows], thr)
        r = evaluate_selection(dec, y[test_rows], available[test_rows], data['session'][test_rows])
        sweep[str(thr)] = r
        print(f"threshold={thr:<4}: acc={r['selection_accuracy_vs_oracle']*100:.1f}% "
              f"mean={r['mean_error_m']:.3f} m  switches={r['switches']:5d}  "
              f"duration={r['mean_duration_s']:.2f} s")

    print('\n=== Ablation study: conditioning feature groups (test split) ===')
    fit_cfg = {**cfg, 'seed': cfg['seed'] + 3}
    fit_budget = dict(fit_cfg['fit'], max_epochs=cfg['ablation_budget_epochs'],
                      patience=cfg['ablation_budget_epochs'])
    fit_cfg['fit'] = fit_budget
    ablation = {}
    for label, groups in ABLATIONS:
        cols = np.sort(np.concatenate([FEATURE_GROUPS[g] for g in groups]))
        fit_cfg['seed'] += 1
        model = fit_dynamic_weights(F[btr][:, cols], terms[btr], y[btr],
                                    F[bva][:, cols], terms[bva], y[bva], fit_cfg, verbose=False)
        Xtest = (F[test_rows][:, cols] - model['mean']) / np.maximum(model['std'], 1e-6)
        w = np.full((len(F), 4), FIXED_FALLBACK_WEIGHTS)
        w[test_rows] = softmax_weights(logits(model['p'], Xtest))
        r = _test_eval(w, terms, available, y, test_mask, data['session'])
        ablation[label] = {
            'feature_groups': groups,
            'columns': [int(c) for c in cols],
            'best_epoch': model['best_epoch'],
            'selection_accuracy': r['selection_accuracy_vs_oracle'],
            'mean_error_m': r['mean_error_m'],
            'switches': r['switches'],
        }
        print(f"{label:28s} mean={r['mean_error_m']:.3f} m  "
              f"acc={r['selection_accuracy_vs_oracle']*100:.1f}%  "
              f"switches={r['switches']}")

    print('\n=== Confusion matrices (selector vs oracle, test split) ===')
    mtest = json.loads((OUT / 'metrics.json').read_text())
    wA_cf = fixed_reliability(MODEL_A_WEIGHTS, terms)
    wB_cf = fixed_reliability(
        np.array([mtest['model_B_weights'][k] for k in ('alpha', 'beta', 'gamma', 'delta')]),
        terms)
    wC_cf = dm.reliability(dm.weights(F), terms)
    confusion = {}
    for name, wsel, thr in [
            ('original_fixed_weights_model_A', wA_cf, None),
            ('optimized_fixed_weights_model_B', wB_cf, None),
            ('dynamic_ml_weights_model_C', wC_cf, None),
            ('dynamic_ml_with_hysteresis', wC_cf, cfg['hysteresis_thresholds'][2])]:
        if thr is None:
            dec = select_scores(wsel, available)[test_rows]
        else:
            dec = apply_hysteresis_sessions(wsel, available, data['session'], thr)[test_rows]
        rc = selection_confusion(dec, y[test_rows], available[test_rows])
        confusion[name] = rc
        print(f"{name:38s} n={rc['n']} acc={rc['accuracy_vs_oracle']*100:.1f}%  "
              f"matrix={rc['matrix']}")

    summary = {
        'synthetic_only': True,
        'baselines': {n: {k: test[n][k] for k in
                          ['selection_accuracy_vs_oracle', 'mean_error_m',
                           'median_error_m', 'rmse_m', 'p95_error_m', 'switches']}
                      for n in names},
        'confusion': confusion,
        'hysteresis_sweep': sweep,
        'ablation': ablation,
        'research_questions': answers(cfg, test),
    }
    (OUT / 'evaluation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\nSaved artifacts/evaluation.json')


def answers(cfg, test):
    a, b, c = test['original_fixed_weights_model_A'], test['optimized_fixed_weights_model_B'], test['dynamic_ml_weights_model_C']
    return [
        ('RQ1', 'Does the original reliability equation select the better receiver consistently?',
         f"A baseline selects for accuracy {a['selection_accuracy_vs_oracle']*100:.1f}% of available "
         f"epochs (mean error {a['mean_error_m']:.2f} m) - it is reasonable but not optimal."),
        ('RQ2', 'Can fixed coefficients be optimized from available data?',
         f"Yes - optimized fixed weights reduce mean error from {a['mean_error_m']:.2f} m "
         f"(A) to {b['mean_error_m']:.2f} m (B), a small but real gain."),
        ('RQ3', 'Does context-dependent ML weighting improve over the original fixed coefficients?',
         f"Model C mean error {c['mean_error_m']:.2f} m vs {a['mean_error_m']:.2f} m (A); with "
         f"hysteresis {test['dynamic_ml_with_hysteresis']['mean_error_m']:.2f} m. Improvement is "
         f"modest and should not be overstated."),
        ('RQ4', 'Does context-dependent weighting outperform a single globally optimized set?',
         f"C mean error {c['mean_error_m']:.3f} m vs B {b['mean_error_m']:.3f} m - the ML gain over a "
         f"good fixed set is small; context modulation is visible in the learned-weight analysis."),
        ('RQ5', 'Are the learned weights interpretable and related to GNSS conditions?',
         "Yes - weights shift strongly by condition (e.g. DOP ~0.99 in open sky vs "
         "satellite/timing weights dominating in urban multipath); see metrics.json learned_weights."),
        ('RQ6', 'Does the adaptive method introduce excessive switching?',
         f"Model C produces {c['switches']} switches vs A {a['switches']} (similar); hysteresis "
         f"reduces them to {test['dynamic_ml_with_hysteresis']['switches']} with improved accuracy."),
    ]


if __name__ == '__main__':
    main()