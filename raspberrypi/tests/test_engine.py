import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from gnss_engine import (  # noqa: E402
    DecisionEngine, parse_frame, reliability_W,
)
from predict import FIXED_FALLBACK_WEIGHTS  # noqa: E402

MODEL = Path(__file__).resolve().parents[2] / 'ml' / 'artifacts' / 'model.npz'
SOURCES = ['GPS', 'GNSS']


def frame(r0, r1, stamp=None):
    stamp = stamp or datetime.now(timezone.utc).strftime("%H:%M:%S")

    def part(tag, lat, lon, sat, hdop, snr):
        return f"{tag},{stamp},{lat},{lon},{sat},{hdop:.2f},{snr:.0f}"

    return part('GPS', *r0) + '|' + part('GNSS', *r1)


class EngineTests(unittest.TestCase):

    def test_parse_frame_new_and_legacy_format(self):
        r0, r1 = ('10.0', '76.0', 15, 1.2, 42.0), ('10.1', '76.1', 18, 0.9, 44.0)
        a, b = parse_frame(frame(r0, r1))
        self.assertEqual(a['sat'], 15)
        self.assertEqual(a['hdop'], 1.2)
        self.assertEqual(a['snr'], 42.0)
        self.assertEqual(b['sat'], 18)
        # legacy 6-field frame (no SNR) yields snr=None, backward compatible
        legacy = 'GPS,12:00:00,10.0,76.0,15,1.20|GNSS,12:00:00,10.1,76.1,18,0.90'
        a, b = parse_frame(legacy)
        self.assertIsNone(a['snr'])
        self.assertEqual(b['sat'], 18)
        # NO_FIX handling
        a, b = parse_frame(frame(('NO_FIX', 'NO_FIX', 0, 99.99, 0), r1))
        self.assertEqual(a['lat'], 'NO_FIX')
        self.assertIsNone(parse_frame('garbage'))

    def test_reliability_W_matches_fixed_weights_equation(self):
        # W = 0.25*T + 0.25*S + 0.30*SNR + 0.20*DOP, exactly per server scoring.
        w = reliability_W(FIXED_FALLBACK_WEIGHTS, time_err=0.0, sat=20, hdop=1.0, snr=50.0)
        self.assertAlmostEqual(w, 0.25 + 0.25 + 0.30 + 0.20, places=6)
        # invalid timing / geometry / no SNR contribute 0 to their terms
        w = reliability_W(FIXED_FALLBACK_WEIGHTS, time_err=None, sat=0, hdop=99.99, snr=None)
        self.assertAlmostEqual(w, 0.0, places=6)

    def test_no_fix_both_no_selection(self):
        engine = DecisionEngine(model_path=str(MODEL), log_dir=None)
        bad = ('NO_FIX', 'NO_FIX', 0, 99.99, 0.0)
        d = engine.update(*parse_frame(frame(bad, bad)))
        self.assertIsNone(d['selected'])
        self.assertEqual(engine.best_location['source'], 'INIT')
        self.assertTrue(engine.best_location['lat'] is None)

    def test_ml_used_when_both_receivers_valid_with_snr(self):
        engine = DecisionEngine(model_path=str(MODEL), log_dir=None)
        self.assertTrue(engine.model_ready)
        r0 = ('10.05', '76.33', 14, 1.5, 40.0)
        r1 = ('10.06', '76.34', 16, 1.2, 44.0)
        for _ in range(3):
            d = engine.update(*parse_frame(frame(r0, r1)))
            self.assertTrue(d['ml_used'])
            self.assertEqual(len(d['weights']), 4)
            np.testing.assert_allclose(sum(d['weights']), 1.0, atol=1e-3)
            self.assertIsNotNone(d['selected'])
        self.assertIn(engine.best_location['source'], SOURCES)

    def test_fallback_when_model_unavailable(self):
        engine = DecisionEngine(model_path='/nonexistent/model.npz', log_dir=None)
        self.assertFalse(engine.model_ready)
        r0 = ('10.05', '76.33', 14, 1.5, 40.0)
        r1 = ('10.06', '76.34', 16, 1.2, 44.0)
        d = engine.update(*parse_frame(frame(r0, r1)))
        self.assertFalse(d['ml_used'])
        np.testing.assert_allclose(d['weights'], FIXED_FALLBACK_WEIGHTS, atol=1e-9)

    def test_hysteresis_holds_selection_in_deadband(self):
        engine = DecisionEngine(model_path='/nonexistent/model.npz',
                                hysteresis=0.10, log_dir=None)
        a = ('10.05', '76.33', 10, 2.0, 40.0)   # W == W of b
        b = ('10.06', '76.34', 10, 2.0, 40.0)
        d1 = engine.update(*parse_frame(frame(a, b)))
        self.assertEqual(d1['reason'], 'initial')
        first = d1['selected']
        for _ in range(3):
            d = engine.update(*parse_frame(frame(a, b)))
            self.assertEqual(d['reason'], 'held')
            self.assertEqual(d['selected'], first)

    def test_switch_outside_deadband(self):
        engine = DecisionEngine(model_path='/nonexistent/model.npz',
                                hysteresis=0.02, log_dir=None)
        poor = ('10.05', '76.33', 6, 6.0, 20.0)
        good = ('10.06', '76.34', 18, 1.0, 48.0)
        d = engine.update(*parse_frame(frame(poor, good)))
        self.assertEqual(d['selected'], 1)      # receiver 1 is clearly better
        self.assertIn(d['reason'], ('initial', 'reconfirmed'))
        # receiver 0 now far better than receiver 1 -> switch outside deadband
        better = ('10.05', '76.33', 20, 0.7, 50.0)
        weaker = ('10.06', '76.34', 14, 1.6, 40.0)
        d = engine.update(*parse_frame(frame(better, weaker)))
        self.assertEqual(d['selected'], 0)
        self.assertEqual(d['reason'], 'switched')

    def test_decision_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DecisionEngine(model_path=str(MODEL), log_dir=tmp)
            r0 = ('10.05', '76.33', 14, 1.5, 40.0)
            r1 = ('10.06', '76.34', 16, 1.2, 44.0)
            engine.update(*parse_frame(frame(r0, r1)))
            log = Path(tmp) / 'decisions.jsonl'
            self.assertTrue(log.exists())
            rows = [json.loads(line) for line in log.read_text().strip().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            for key in ('timestamp', 'selected', 'weights', 'w_scores',
                        'features', 'ml_used', 'reason', 'instrument'):
                self.assertIn(key, row)
            self.assertIsNotNone(row['features'])
            self.assertEqual(len(row['features']), 16)
            self.assertEqual(len(row['instrument']['sat']), 2)


if __name__ == '__main__':
    unittest.main()