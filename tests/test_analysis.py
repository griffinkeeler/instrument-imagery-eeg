"""Synthetic checks for trigger decoding, alignment, isolation, and CV."""
import unittest
from unittest.mock import patch

import mne
from mne.decoding import CSP
import numpy as np
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from scripts.run_lda import build_epochs, decode_triggers, evaluate, make_pipeline


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        mne.set_log_level("ERROR")

    def test_trigger_runs_not_samples(self):
        codes = np.array([0, 1, 1, 0, 2, 2, 0, 4, 4, 8, 8, 0])
        digital = np.stack([(codes & bit != 0) * 5. for bit in (1, 2, 4, 8)])
        reconstructed, events = decode_triggers(digital)
        np.testing.assert_array_equal(reconstructed, codes)
        self.assertEqual(events, [(1, 1, 2), (4, 2, 2), (7, 4, 2), (9, 8, 2)])

    def recording(self):
        rng = np.random.default_rng(12)
        eeg = rng.normal(scale=1e-5, size=(9, 5000))
        eog = rng.normal(scale=1e-5, size=5000)
        events = [(100, 1, 10), (300, 4, 10), (1500, 2, 10), (1700, 8, 10)]
        return eeg, eog, events

    def test_alignment_practice_and_filter_isolation(self):
        eeg, eog, events = self.recording()
        X, y, rows = build_epochs(eeg, eog, 100, events, [], exclude_first=1)
        self.assertEqual(X.shape, (1, 9, 600))
        self.assertEqual(y.tolist(), [1])
        self.assertEqual(rows[0]["reason"], "explicit_exclusion")
        self.assertEqual(rows[1]["onset_sample"], 1700)
        # Changing all data outside this trial must not change its features.
        changed = eeg.copy()
        changed[:, :1700] = 999
        changed[:, 2700:] = -999
        other, _, _ = build_epochs(changed, eog, 100, events, [], exclude_first=1)
        np.testing.assert_array_equal(X, other)

    def test_boundary_and_incomplete_trial_rejection(self):
        eeg, eog, events = self.recording()
        _, y, rows = build_epochs(eeg, eog, 100, events, [900])
        self.assertEqual(y.tolist(), [1])
        self.assertEqual(rows[0]["reason"], "acquisition_boundary")
        _, y, rows = build_epochs(eeg[:, :2400], eog[:2400], 100, events, [])
        self.assertEqual(y.tolist(), [0])
        self.assertEqual(rows[1]["reason"], "incomplete_imagery_trial")

    def test_cue_validation_and_artifact_rejection(self):
        eeg, eog, events = self.recording()
        with self.assertRaisesRegex(ValueError, "matching preceding cue"):
            build_epochs(eeg, eog, 100, [(100, 2, 10), (300, 4, 10)], [])
        eog[600] = .01
        _, y, rows = build_epochs(eeg, eog, 100, events, [], reject_eog_uv=1000)
        self.assertEqual(y.tolist(), [1])
        self.assertEqual(rows[0]["reason"], "eog_amplitude")

    def test_csp_fit_receives_only_training_trials(self):
        rng = np.random.default_rng(51)
        X = rng.normal(size=(20, 9, 100))
        y = np.tile([0, 1], 10)
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        expected = list(splitter.split(X, y))
        original_fit = CSP.fit
        calls = []

        def audited_fit(csp, fit_X, fit_y, **kwargs):
            train, test = expected[len(calls)]
            np.testing.assert_array_equal(fit_X, X[train])
            np.testing.assert_array_equal(fit_y, y[train])
            self.assertEqual(np.intersect1d(train, test).size, 0)
            calls.append(csp)
            return original_fit(csp, fit_X, fit_y, **kwargs)

        with patch.object(CSP, "fit", audited_fit):
            report, _, _ = evaluate(make_pipeline(), X, y, splitter)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len({id(csp) for csp in calls}), 5)
        self.assertTrue(report["fold_audit"]["train_test_disjoint"])
        self.assertEqual(report["fold_audit"]["test_appearances_per_trial"], [1] * 20)

    def test_permutations_refit_csp_on_each_shuffled_training_fold(self):
        X = np.random.default_rng(51).normal(size=(20, 9, 100))
        y = np.tile([0, 1], 10)
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        rng = np.random.RandomState(42)
        label_sets = [y] + [y[rng.permutation(len(y))] for _ in range(3)]
        expected = [(labels, train) for labels in label_sets
                    for train, _ in splitter.split(X, labels)]
        original_fit = CSP.fit
        calls = []

        def audited_fit(csp, fit_X, fit_y, **kwargs):
            labels, train = expected[len(calls)]
            np.testing.assert_array_equal(fit_X, X[train])
            np.testing.assert_array_equal(fit_y, labels[train])
            calls.append(len(fit_X))
            return original_fit(csp, fit_X, fit_y, **kwargs)

        with patch.object(CSP, "fit", audited_fit):
            _, null, _ = permutation_test_score(make_pipeline(), X, y, cv=splitter,
                scoring="balanced_accuracy", n_permutations=3, random_state=42)
        self.assertEqual(calls, [16] * 20)
        self.assertEqual(len(null), 3)

    def test_csp_lda_recovers_synthetic_variance_signal(self):
        rng = np.random.default_rng(4)
        y = np.tile([0, 1], 20)
        X = rng.normal(size=(40, 9, 200))
        X[y == 0, 0] *= 4
        X[y == 1, 1] *= 4
        model = make_pipeline(4)
        report, predictions, folds = evaluate(model, X, y, StratifiedKFold(4, shuffle=True, random_state=1))
        self.assertGreater(report["balanced_accuracy"], .9)
        self.assertEqual(len(predictions), 40)
        self.assertTrue(np.all(folds > 0))
        self.assertFalse(hasattr(model.named_steps["csp"], "filters_"))


if __name__ == "__main__":
    unittest.main()
