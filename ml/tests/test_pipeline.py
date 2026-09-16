import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate import generate
from features import build_features, reliability_scores, eligible, FEATURE_NAMES
from predict import DynamicWeights, select_scores, apply_hysteresis, FIXED_FALLBACK_WEIGHTS, WEIGHT_NAMES
from baselines import MODEL_A_WEIGHTS, grid_search_optimal_weights, reliability as fixed_reliability
from metrics import evaluate_selection, apply_hysteresis_sessions

ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts'


class PipelineTests(unittest.TestCase):
    def test_deterministic_generation_and_labels(self):
        a, b = generate(sessions=5, seconds=100), generate(sessions=5, seconds=100)
        np.testing.assert_array_equal(a['x'], b['x'])
        self.assertEqual(a['x'].shape[2], 7)
        self.assertEqual(a['x'].shape[1], 2)
        # Independent geographic reconstruction within local tangent-plane tolerance.
        delta = a['observed_latlon'] - a['reference_latlon'][:, None, :]
        north = delta[:, :, 0] * 111320
        east = delta[:, :, 1] * 111320 * np.cos(np.radians(a['reference_latlon'][:, None, 0]))
        np.testing.assert_allclose(np.hypot(east, north), a['error_m'], rtol=0.001, atol=.002)

    def test_validity_rules(self):
        good = np.array([15, .8, 40, .5, 2, .2, 0])
        self.assertTrue(eligible(good))
        self.assertEqual(eligible(good[None]).shape, (1,))
        for index, value in [(0, -1), (1, 0), (2, -2), (3, -1), (3, 3),
                             (4, -1), (5, -1), (5, 60), (6, 3), (1, np.nan)]:
            bad = good.copy()
            bad[index] = value
            self.assertFalse(eligible(bad), f'expected invalid for index {index}={value}')

    def test_feature_and_reliability_domains(self):
        a = generate(sessions=4, seconds=90)
        F = build_features(a['x'], a['session'])
        self.assertEqual(F.shape, (360, 16))
        self.assertEqual(len(FEATURE_NAMES), 16)
        np.testing.assert_array_equal(build_features(a['x'], a['session']), F)
        terms = reliability_scores(a['x'])
        self.assertEqual(terms.shape, (360, 2, 4))
        self.assertTrue((terms >= 0).all() and (terms <= 1).all())

    def test_dynamic_weights_softmax_constraint(self):
        model = DynamicWeights(ARTIFACTS / 'model.npz')
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 16))
        w = model.weights(X)
        self.assertEqual(w.shape, (50, 4))
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-6)
        self.assertTrue((w >= 0).all() and (w <= 1).all())
        self.assertEqual(sorted(WEIGHT_NAMES),
                         ['alpha', 'beta', 'delta', 'gamma'])

    def test_selection_availability_and_fallback(self):
        W = np.array([[0.5, 0.3], [0.2, 0.9]])
        avail = np.array([[True, True], [True, False]])
        self.assertListEqual(select_scores(W, avail).tolist(), [0, 0])
        self.assertListEqual(select_scores(W, [[False, False], [True, True]]).tolist(), [-1, 1])
        # Original coefficients remain the documented fallback.
        self.assertListEqual(FIXED_FALLBACK_WEIGHTS.tolist(), [0.25, 0.25, 0.30, 0.20])

    def test_hysteresis_deadband(self):
        W = np.array([[0.50, 0.49], [0.51, 0.50], [0.10, 0.11], [0.55, 0.45]])
        avail = np.ones((4, 2), dtype=bool)
        self.assertListEqual(apply_hysteresis(W, avail, 0.0).tolist(), [0, 0, 1, 0])
        # Deadband keeps the previous selection within |W1-W2| < 0.10.
        self.assertListEqual(apply_hysteresis(W, avail, 0.10).tolist(), [0, 0, 0, 0])
        # Session boundaries reset hysteresis state.
        session = np.array([0, 0, 1, 1])
        W2 = np.array([[0.49, 0.51], [0.10, 0.11], [0.49, 0.51], [0.49, 0.51]])
        self.assertListEqual(apply_hysteresis_sessions(W2, avail, session, 0.05).tolist(),
                             [1, 1, 1, 1])

    def test_grid_search_simplex_and_candidates(self):
        data = generate(sessions=4, seconds=90)
        terms, y = reliability_scores(data['x']), data['error_m']
        avail = np.ones((data['x'].shape[0], 2), dtype=bool)
        w, err, count = grid_search_optimal_weights(terms, y, avail, step=0.05)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-9)
        self.assertEqual(count, 1771)
        self.assertLess(err, np.inf)
        self.assertEqual(MODEL_A_WEIGHTS.sum(), 1.0)

    def test_split_isolation(self):
        import json
        d = json.loads((ARTIFACTS / 'metrics.json').read_text())['split_sessions']
        train, val, test = map(set, [d['train'], d['validation'], d['test']])
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train | val | test), 120)

    def test_evaluation_metrics_shape(self):
        data = generate(sessions=4, seconds=90)
        y, avail = data['error_m'], np.ones((360, 2), dtype=bool)
        W = fixed_reliability(MODEL_A_WEIGHTS, reliability_scores(data['x']))
        dec = select_scores(W, avail)
        m = evaluate_selection(dec, y, avail, data['session'])
        for key in ('selection_accuracy_vs_oracle', 'mean_error_m', 'median_error_m',
                    'rmse_m', 'p95_error_m', 'max_error_m', 'switches'):
            self.assertIn(key, m)
        self.assertGreaterEqual(m['mean_error_m'], 0)


if __name__ == '__main__':
    unittest.main()