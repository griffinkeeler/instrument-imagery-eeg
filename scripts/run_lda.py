"""Offline, trial-level piano/guitar imagery decoding from BIOPAC recordings.

Run --help for options. See ANALYSIS.md for assumptions and interpretation.
Importing this module does not read data or fit a model.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import bioread
import mne
import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TRIGGER_NAMES = {1: "piano_cue", 2: "guitar_cue", 4: "piano_imagery", 8: "guitar_imagery"}
EEG_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "Cz", "Fz", "Pz"]


def decode_triggers(digital, threshold=2.5):
    """Rows are D0..D3; return one code per sample and nonzero run onsets."""
    digital = np.asarray(digital)
    if digital.ndim != 2 or digital.shape[0] != 4 or not digital.shape[1]:
        raise ValueError("Expected four nonempty digital channels (D0..D3).")
    if not np.isfinite(digital).all():
        raise ValueError("Digital channels contain nonfinite samples.")
    codes = np.sum((digital > threshold) * np.array([1, 2, 4, 8])[:, None], axis=0)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    stops = np.r_[starts[1:], len(codes)]
    events = [(int(s), int(codes[s]), int(e - s))
              for s, e in zip(starts, stops) if codes[s] != 0]
    return codes, events


def to_volts(channel):
    scale = {"v": 1.0, "volts": 1.0, "mv": 1e-3, "uv": 1e-6}
    unit = channel.units.strip().lower().replace("µ", "u").replace("μ", "u")
    if unit not in scale:
        raise ValueError(f"Unsupported units {channel.units!r} on {channel.name}.")
    return np.asarray(channel.data, dtype=float) * scale[unit]


def build_epochs(eeg, eog, fs, events, boundaries, *, tmin=2., tmax=8.,
                 duration=10., low=8., high=30., exclude_first=0,
                 exclude_trials=(), reject_eeg_uv=None, reject_eog_uv=None):
    """Filter each full imagery trial independently, then crop [tmin, tmax).

    One sample array per trial prevents filtering across trials or acquisition
    boundaries. Exclusions use one-based imagery trial numbers before rejection.
    QC amplitudes are measured on the unfiltered analysis window in microvolts.
    """
    if not (0 <= tmin < tmax <= duration and 0 < low < high < fs / 2):
        raise ValueError("Require 0 <= tmin < tmax <= duration and 0 < low < high < Nyquist.")
    if exclude_first < 0:
        raise ValueError("exclude_first must be nonnegative.")
    n_full, a, b = (round(x * fs) for x in (duration, tmin, tmax))
    if b <= a:
        raise ValueError("Analysis window is shorter than one sample.")
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    X, y, rows = [], [], []
    cue = None
    trial = 0
    for event_idx, (onset, code, width) in enumerate(events):
        if code not in TRIGGER_NAMES:
            raise ValueError(f"Unexpected trigger {code} at sample {onset}.")
        if code in (1, 2):
            cue = (onset, code)
            continue
        trial += 1
        expected = 1 if code == 4 else 2
        if cue is None or cue[1] != expected:
            raise ValueError(f"Imagery trial {trial} has no matching preceding cue.")
        row = dict(trial=trial, onset_sample=onset, onset_seconds=onset / fs,
                   trigger=code, label=0 if code == 4 else 1,
                   condition="piano" if code == 4 else "guitar",
                   cue_to_imagery_seconds=(onset - cue[0]) / fs,
                   pulse_seconds=width / fs, eeg_max_ptp_uv="", eog_ptp_uv="",
                   included=False, reason="", epoch_index="")
        cue = None
        stop = onset + n_full
        reason = ""
        if trial <= exclude_first or trial in exclude_trials:
            reason = "explicit_exclusion"
        elif stop > eeg.shape[1]:
            reason = "incomplete_imagery_trial"
        elif any(onset < boundary < stop for boundary in boundaries):
            reason = "acquisition_boundary"
        elif event_idx + 1 < len(events) and events[event_idx + 1][0] < stop:
            reason = "next_event_before_imagery_end"
        else:
            full = eeg[:, onset:stop]
            eye = eog[onset:stop]
            if not np.isfinite(full).all() or not np.isfinite(eye).all():
                reason = "nonfinite_data"
            else:
                ptp = np.ptp(full[:, a:b], axis=1) * 1e6
                eye_ptp = float(np.ptp(eye[a:b]) * 1e6)
                row.update(eeg_max_ptp_uv=float(ptp.max()), eog_ptp_uv=eye_ptp)
                if np.any(ptp == 0):
                    reason = "flat_eeg_channel"
                elif reject_eeg_uv is not None and ptp.max() > reject_eeg_uv:
                    reason = "eeg_amplitude"
                elif reject_eog_uv is not None and eye_ptp > reject_eog_uv:
                    reason = "eog_amplitude"
                else:
                    filtered = sosfiltfilt(sos, full, axis=-1)[:, a:b]
                    row.update(included=True, epoch_index=len(X))
                    X.append(filtered)
                    y.append(row["label"])
        row["reason"] = reason
        rows.append(row)
    if not X:
        raise ValueError("No usable imagery trials remain.")
    return np.stack(X), np.asarray(y), rows


def make_pipeline(n_components=4):
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True,
                    norm_trace=False, cov_est="epoch")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def evaluate(model, X, y, splitter):
    """Fit a fresh complete pipeline in each fold; predict each trial once."""
    predictions = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)
    folds = []
    test_counts = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(X, y), 1):
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError(f"Fold {fold} lacks one class; change fold count or trial selection.")
        if np.intersect1d(train, test).size:
            raise ValueError(f"Fold {fold} contains overlapping training/test trials.")
        test_counts[test] += 1
        fitted = clone(model).fit(X[train], y[train])
        predictions[test] = fitted.predict(X[test])
        fold_ids[test] = fold
        folds.append(dict(fold=fold, n_train=len(train), n_test=len(test),
                          train_epoch_indices=train.tolist(), test_epoch_indices=test.tolist(),
                          accuracy=float(accuracy_score(y[test], predictions[test])),
                          balanced_accuracy=float(balanced_accuracy_score(y[test], predictions[test]))))
    if not np.all(test_counts == 1):
        raise ValueError("Each trial must occur in exactly one held-out fold.")
    recalls = recall_score(y, predictions, labels=[0, 1], average=None)
    return dict(accuracy=float(accuracy_score(y, predictions)),
                per_class_recall={"piano": float(recalls[0]), "guitar": float(recalls[1])},
                fold_audit={"train_test_disjoint": True, "test_appearances_per_trial": test_counts.tolist()},
                balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
                mean_fold_balanced_accuracy=float(np.mean([f["balanced_accuracy"] for f in folds])),
                confusion_matrix=confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
                folds=folds), predictions, fold_ids


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/Instrument_10_6_26.acq")
    p.add_argument("--output", type=Path, default=ROOT / "results/csp_lda")
    p.add_argument("--tmin", type=float, default=2.)
    p.add_argument("--tmax", type=float, default=8., help="Exclusive end, seconds after imagery onset")
    p.add_argument("--duration", type=float, default=10.)
    p.add_argument("--low", type=float, default=8.)
    p.add_argument("--high", type=float, default=30.)
    p.add_argument("--components", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--chronological-folds", type=int, default=4,
                   help="Contiguous held-out trial chunks; 0 disables this sensitivity check")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permutations", type=int, default=0,
                   help="Label permutations refitting CSP+LDA; e.g. 999 for inference, 0 skips")
    p.add_argument("--exclude-first", type=int, default=4,
                   help="Number of initial imagery trials to exclude as practice")
    p.add_argument("--exclude-trials", type=int, nargs="*", default=[],
                   help="Additional one-based imagery trial numbers to exclude")
    p.add_argument("--reject-eeg-uv", type=float, default=None)
    p.add_argument("--reject-eog-uv", type=float, default=None)
    return p.parse_args(argv)


def load_recording_epochs(args):
    """Shared recording validation and preprocessing for decoding and maps."""
    for threshold in (args.reject_eeg_uv, args.reject_eog_uv):
        if threshold is not None and threshold <= 0:
            raise ValueError("Artifact thresholds must be positive microvolt values.")
    data = bioread.read_file(str(args.input))
    if len(data.channels) != 14:
        raise ValueError("Expected 9 EEG, 1 EOG, and 4 digital channels; inspect channel mapping.")
    fs = float(data.channels[0].samples_per_second)
    n_samples = len(data.channels[0].data)
    for i, ch in enumerate(data.channels):
        if ch.samples_per_second != fs or len(ch.data) != n_samples:
            raise ValueError("All channels must have the same sample rate and length.")
        expected = EEG_NAMES[i] if i < 9 else "EOG" if i == 9 else "Digital"
        if ch.name.split()[0] != expected:
            raise ValueError(f"Channel {i}: expected {expected}, found {ch.name!r}.")
    eeg = np.stack([to_volts(ch) for ch in data.channels[:9]])
    eog = to_volts(data.channels[9])
    _, events = decode_triggers([ch.data for ch in data.channels[10:14]])
    markers = [dict(sample=int(m.sample_index), text=m.text)
               for m in data.event_markers or [] if m.sample_index is not None]
    boundaries = [m["sample"] for m in markers if m["text"].lower().startswith("segment")]
    n_imagery = sum(code in (4, 8) for _, code, _ in events)
    if args.exclude_first > n_imagery or any(t < 1 or t > n_imagery for t in args.exclude_trials):
        raise ValueError("Requested trial exclusions are outside the recording.")
    X, y, rows = build_epochs(eeg, eog, fs, events, boundaries,
        tmin=args.tmin, tmax=args.tmax, duration=args.duration, low=args.low, high=args.high,
        exclude_first=args.exclude_first, exclude_trials=args.exclude_trials,
        reject_eeg_uv=args.reject_eeg_uv, reject_eog_uv=args.reject_eog_uv)
    return X, y, rows, fs, events, markers


def main(argv=None):
    args = parse_args(argv)
    mne.set_log_level("ERROR")
    if not 1 <= args.components <= 9 or args.folds < 2 or args.permutations < 0:
        raise ValueError("Require 1..9 components, >=2 folds, and >=0 permutations.")
    if args.chronological_folds not in (0,) and args.chronological_folds < 2:
        raise ValueError("chronological-folds must be 0 or >=2.")
    X, y, rows, fs, events, markers = load_recording_epochs(args)
    if len(np.unique(y)) != 2 or min(Counter(y).values()) < args.folds:
        raise ValueError("Too few trials per class for the requested stratified folds.")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "trials.csv", rows)
    write_csv(args.output / "events.csv", [dict(sample=s, seconds=s / fs, code=c,
        name=TRIGGER_NAMES[c], pulse_seconds=w / fs) for s, c, w in events])
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    model = make_pipeline(args.components)
    primary, pred, fold_ids = evaluate(model, X, y, splitter)
    included = [r for r in rows if r["included"]]
    predictions = [dict(trial=r["trial"], true_label=int(y[i]), predicted_label=int(pred[i]),
                        fold=int(fold_ids[i])) for i, r in enumerate(included)]
    report = dict(input=str(args.input.resolve()), settings={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, sample_rate=fs, epoch_shape=list(X.shape),
                  channels=EEG_NAMES, labels={0: "piano", 1: "guitar"},
                  trigger_counts=dict(Counter(c for _, c, _ in events)), markers=markers,
                  included_counts={"piano": int(sum(y == 0)), "guitar": int(sum(y == 1))},
                  excluded_counts=dict(Counter(r["reason"] for r in rows if not r["included"])),
                  stratified_cv=primary, permutation_test=None,
                  versions={p: version(p) for p in ("numpy", "scipy", "mne", "scikit-learn", "bioread")})
    if args.chronological_folds:
        chronological, cpred, cfold = evaluate(model, X, y, KFold(args.chronological_folds))
        report["chronological_cv"] = chronological
        print(f"Chronological CV balanced accuracy: {chronological['balanced_accuracy']:.3f}", flush=True)
        for i, row in enumerate(predictions):
            row.update(chronological_predicted_label=int(cpred[i]), chronological_fold=int(cfold[i]))
    write_csv(args.output / "predictions.csv", predictions)
    print(f"Epochs: {X.shape}; piano={sum(y == 0)}, guitar={sum(y == 1)}", flush=True)
    print(f"Stratified CV balanced accuracy: {primary['balanced_accuracy']:.3f}", flush=True)
    if args.permutations:
        print(f"Running {args.permutations} permutations (full pipeline refitted each time)...", flush=True)
        score, null_scores, pvalue = permutation_test_score(model, X, y, cv=splitter,
            scoring="balanced_accuracy", n_permutations=args.permutations,
            random_state=args.seed, n_jobs=1)
        report["permutation_test"] = dict(score=float(score), p_value=float(pvalue),
            permutations=args.permutations, null_scores=null_scores.tolist(),
            null_mean=float(np.mean(null_scores)), null_std=float(np.std(null_scores)),
            null_percentiles={str(q): float(np.percentile(null_scores, q)) for q in [2.5, 50, 95, 97.5, 99]},
            null_max=float(np.max(null_scores)),
            null_at_least_observed=int(np.sum(null_scores >= score)),
            assumption="Trial labels are exchangeable; global permutations, not block-restricted.")
        print(f"Permutation p-value: {pvalue:.4f}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved summary, events, trial QC, and out-of-fold predictions to {args.output}")
    return report


if __name__ == "__main__":
    main()
