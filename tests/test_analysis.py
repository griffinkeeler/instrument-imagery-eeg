"""Synthetic checks for trigger decoding, alignment, isolation, and CV."""
import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import mne
from mne.decoding import CSP
import numpy as np
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from scripts.run_lda import build_epochs, decode_triggers, evaluate, make_pipeline, main, parse_args


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

    def test_initial_active_trigger_is_not_an_observed_onset(self):
        codes = np.array([4, 4, 4, 1, 1, 0, 4, 4, 0])
        digital = np.stack([(codes & bit != 0) * 5. for bit in (1, 2, 4, 8)])
        reconstructed, events = decode_triggers(digital)
        np.testing.assert_array_equal(reconstructed, codes)
        self.assertEqual(events, [(3, 1, 2), (6, 4, 2)])

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

    def test_cross_session_fits_only_training_recording_and_saves_test_predictions(self):
        rng = np.random.default_rng(24)
        training = rng.normal(size=(20, 9, 200))
        testing = rng.normal(size=(6, 9, 200))
        train_y, test_y = np.tile([0, 1], 10), np.tile([0, 1], 3)
        for X, y in ((training, train_y), (testing, test_y)):
            X[y == 0, 0] *= 5
            X[y == 1, 1] *= 5
        loads, fits = [], []
        original_fit = CSP.fit

        def loader(args):
            loads.append((args.input.name, args.exclude_first, args.exclude_trials))
            X, y = (training, train_y) if args.input.name == "train.acq" else (testing, test_y)
            rows = [dict(trial=i + 1, included=True, reason="", epoch_index=i) for i in range(len(y))]
            return X, y, rows, 100., [(0, 1, 1), (100, 4, 1)], []

        def fit(csp, X, y, **kwargs):
            np.testing.assert_array_equal(X, training)
            np.testing.assert_array_equal(y, train_y)
            fits.append(len(X))
            return original_fit(csp, X, y, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.acq", root / "test.acq"
            train.touch()
            test.touch()
            with patch("scripts.run_lda.load_recording_epochs", side_effect=loader), patch.object(CSP, "fit", fit):
                report = main(["--cross-session", str(train), str(test), "--output", str(root),
                               "--test-exclude-first", "0", "--test-exclude-trials"])
            self.assertEqual(fits, [20])
            self.assertEqual(loads, [("train.acq", 4, []), ("test.acq", 0, [])])
            self.assertGreater(report["cross_session"]["balanced_accuracy"], .9)
            saved = json.loads((root / "summary.json").read_text())
            self.assertEqual(saved["cross_session"]["n_test"], 6)
            with (root / "predictions.csv").open() as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual(len(predictions), 6)
            self.assertTrue(all(row["session"] == "test" for row in predictions))
            with self.assertRaisesRegex(ValueError, "must be different"):
                main(["--cross-session", str(train), str(train)])
            with self.assertRaisesRegex(ValueError, "single-recording CV"):
                main(["--cross-session", str(train), str(test), "--permutations", "1"])
            with patch("scripts.run_lda.load_recording_epochs", side_effect=[
                (training, train_y, [], 100., [], []), (testing, test_y, [], 200., [], [])
            ]):
                with self.assertRaisesRegex(ValueError, "matching sample rates"):
                    main(["--cross-session", str(train), str(test)])

    def test_cross_session_requires_exactly_two_files(self):
        with self.assertRaises(SystemExit):
            parse_args(["--cross-session", "train.acq"])
        with self.assertRaises(SystemExit):
            parse_args(["--input", "one.acq", "--cross-session", "train.acq", "test.acq"])


if __name__ == "__main__":
    unittest.main()
