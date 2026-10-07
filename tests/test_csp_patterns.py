"""Check that visualization preserves forward-pattern orientation and polarity."""
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.plot_csp_patterns import normalized_patterns, condition_pattern_rms


class PatternTests(unittest.TestCase):
    def test_uses_pattern_rows_not_filters_and_preserves_original(self):
        original = np.array([[1., -4., 2.], [-6., 3., 0.], [3., 1., 2.]])
        csp = SimpleNamespace(patterns_=original.copy(), filters_=np.ones((3, 3)))
        raw, scaled = normalized_patterns(csp, 2)
        np.testing.assert_array_equal(raw, original[:2])
        np.testing.assert_allclose(scaled, [[.25, -1., .5], [-1., .5, 0.]])
        np.testing.assert_array_equal(csp.patterns_, original)
        self.assertFalse(np.shares_memory(raw, csp.patterns_))

    def test_class_rms_uses_shared_patterns_and_retains_amplitude_difference(self):
        # Guitar has twice piano's amplitude; oscillations average to zero,
        # so a trial/time mean waveform would incorrectly erase this signal.
        X = np.array([[[1., -1., 1., -1.], [2., -2., 2., -2.]],
                      [[2., -2., 2., -2.], [4., -4., 4., -4.]]]) * 1e-6
        y = np.array([0, 1])
        model = SimpleNamespace(filters_=np.eye(2), patterns_=np.eye(2))
        rms = condition_pattern_rms(model, X, y, 2)
        np.testing.assert_allclose(rms, [[[1., 0.], [0., 2.]], [[2., 0.], [0., 4.]]])
        # CSP's arbitrary sign and reciprocal filter/pattern scaling cancel.
        changed = SimpleNamespace(filters_=np.eye(2) * -3, patterns_=np.eye(2) / -3)
        np.testing.assert_allclose(condition_pattern_rms(changed, X, y, 2), rms)
        np.testing.assert_allclose(condition_pattern_rms(model, X, 1 - y, 2), rms[::-1])

    def test_rejects_unusable_patterns(self):
        for values in ([[0., 0.]], [[np.nan, 1.]]):
            with self.assertRaises(ValueError):
                normalized_patterns(SimpleNamespace(patterns_=np.array(values)), 1)


if __name__ == "__main__":
    unittest.main()
