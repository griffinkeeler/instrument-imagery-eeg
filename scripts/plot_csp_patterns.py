"""Plot descriptive CSP scalp patterns using the decoding pipeline's epochs.

Fits CSP to all retained trials for visualization only; does not estimate
accuracy or replace cross-validation. Run --help for preprocessing options.
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a display or blocking PyCharm.
import matplotlib.pyplot as plt
import mne
import numpy as np

if __package__:
    from .run_lda import EEG_NAMES, ROOT, load_recording_epochs, make_pipeline, write_csv
else:
    from run_lda import EEG_NAMES, ROOT, load_recording_epochs, make_pipeline, write_csv


def normalized_patterns(csp, n_components):
    """Return patterns (components × channels), normalized within each map.

    Use forward patterns_, never decoding filters_. Preserve arbitrary CSP sign
    and the fitted estimator's original values. Values are not importance scores.
    """
    raw = np.asarray(csp.patterns_[:n_components], dtype=float).copy()
    if raw.ndim != 2 or raw.shape[0] != n_components:
        raise ValueError("Requested components exceed the fitted CSP patterns.")
    scale = np.max(np.abs(raw), axis=1, keepdims=True)
    if not np.isfinite(raw).all() or np.any(scale == 0):
        raise ValueError("Cannot display nonfinite or zero CSP patterns.")
    return raw, raw / scale


def plot_patterns(patterns, info, subtitle):
    """Use a symmetric within-component scale and show every recorded sensor."""
    count = len(patterns)
    ncols = min(2 if count <= 4 else 3, count)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.9 * nrows + 1.2),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= count:
            ax.set_visible(False)
            continue
        im, _ = mne.viz.plot_topomap(patterns[i], info, axes=ax,
            names=info.ch_names, sensors=True, cmap="RdBu_r", vlim=(-1, 1),
            contours=0, image_interp="linear", extrapolate="local", show=False)
        ax.set_title(f"CSP component {i + 1}", fontsize=13, pad=15)
    fig.subplots_adjust(left=.07, right=.84, top=.83, bottom=.13, hspace=.25, wspace=.25)
    bar = fig.colorbar(im, cax=fig.add_axes([.89, .26, .025, .43]))
    bar.set_label("Relative pattern amplitude\n(each map normalized separately)")
    fig.suptitle("CSP spatial patterns", fontsize=20, y=.975)
    fig.text(.5, .924, subtitle, ha="center", fontsize=10)
    fig.text(.5, .055,
             "Descriptive fit to all retained trials • Standard 10–20 sensor positions\n"
             "Polarity is arbitrary; colors do not identify piano/guitar. Patterns are not electrode importance.",
             ha="center", va="center", fontsize=9, linespacing=1.6)
    return fig


def condition_pattern_rms(csp, X, y, n_components):
    """Scalp RMS of each individual CSP component, separately by condition.

    s_k = W_k X; component k's sensor reconstruction is A_k * s_k.
    Return sqrt(mean((A_k*s_k)^2)) in microvolts, without averaging away
    oscillations. Uses raw patterns (not normalized maps or log-power features).
    Axes: condition (piano, guitar), component, channel.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 3 or len(y) != len(X) or set(y) != {0, 1}:
        raise ValueError("Expected trial epochs with both piano (0) and guitar (1) labels.")
    raw, _ = normalized_patterns(csp, n_components)
    sources = np.einsum("kc,nct->nkt", csp.filters_[:n_components], X)
    power = np.stack([np.mean(sources[y == label] ** 2, axis=(0, 2)) for label in (0, 1)])
    rms_uv = np.abs(raw)[None, :, :] * np.sqrt(power)[:, :, None] * 1e6
    if not np.isfinite(rms_uv).all() or not np.any(rms_uv > 0):
        raise ValueError("Nonfinite or all-zero component RMS amplitudes.")
    return rms_uv


def plot_condition_comparison(rms_uv, info, subtitle):
    """One component per row, piano/guitar columns; one scale across all maps."""
    count = rms_uv.shape[1]
    fig, axes = plt.subplots(count, 2, figsize=(9, 3.3 * count + 2), squeeze=False)
    maximum = float(rms_uv.max())
    for component in range(count):
        for label, name in enumerate(("Piano", "Guitar")):
            ax = axes[component, label]
            im, _ = mne.viz.plot_topomap(rms_uv[label, component], info, axes=ax,
                names=info.ch_names, sensors=True, cmap="viridis", vlim=(0, maximum),
                contours=0, image_interp="linear", extrapolate="local", show=False)
            ax.set_title(f"{name} · CSP {component + 1}", fontsize=12, pad=10)
    # Reserve fixed-size title/footer bands for any requested component count.
    height = fig.get_figheight()
    fig.subplots_adjust(left=.06, right=.82, top=1 - 1.2 / height,
                        bottom=.85 / height, hspace=.27, wspace=.22)
    bar = fig.colorbar(im, cax=fig.add_axes([.88, .33, .025, .34]))
    bar.set_label("Component scalp RMS (µV)\nSame scale for both conditions and all components")
    fig.suptitle("Piano vs guitar: CSP component amplitude", fontsize=17, y=1 - .2 / height)
    fig.text(.46, 1 - .62 / height, subtitle, ha="center", fontsize=10)
    fig.text(.46, .32 / height,
        "Shared CSP patterns, scaled by each condition’s component RMS.\n"
        "Descriptive full-data fit; individual component reconstructions, not total EEG or electrode importance.",
        ha="center", va="center", fontsize=9, linespacing=1.6)
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/Instrument_10_6_26.acq")
    parser.add_argument("--output", type=Path, default=ROOT / "results/csp_patterns")
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--tmin", type=float, default=2.)
    parser.add_argument("--tmax", type=float, default=8., help="Exclusive end after imagery onset (s)")
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--low", type=float, default=8.)
    parser.add_argument("--high", type=float, default=30.)
    parser.add_argument("--exclude-first", type=int, default=4)
    parser.add_argument("--exclude-trials", type=int, nargs="*", default=[])
    parser.add_argument("--reject-eeg-uv", type=float, default=None)
    parser.add_argument("--reject-eog-uv", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mne.set_log_level("ERROR")
    if not 1 <= args.components <= len(EEG_NAMES):
        raise ValueError("Require 1..9 CSP components.")
    X, y, rows, fs, _, _ = load_recording_epochs(args)
    if set(y) != {0, 1} or min(np.bincount(y)) < 2:
        raise ValueError("At least two retained trials per class are required.")
    # Reuse exactly the decoder's CSP configuration. This separate fit is never
    # passed back into the cross-validation evaluation or used to report accuracy.
    csp = make_pipeline(args.components).named_steps["csp"].fit(X, y)
    raw, normalized = normalized_patterns(csp, args.components)
    info = mne.create_info(EEG_NAMES, sfreq=fs, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    subtitle = (f"{sum(y == 0)} piano + {sum(y == 1)} guitar trials | "
                f"{args.low:g}–{args.high:g} Hz | {args.tmin:g}–{args.tmax:g} s after imagery onset")
    fig = plot_patterns(normalized, info, subtitle)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"csp_patterns.{extension}", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    rms_uv = condition_pattern_rms(csp, X, y, args.components)
    comparison = plot_condition_comparison(rms_uv, info, subtitle)
    comparison.savefig(args.output / "csp_patterns_piano_vs_guitar.png", dpi=200,
                       facecolor="white", bbox_inches="tight")
    plt.close(comparison)
    write_csv(args.output / "condition_pattern_rms.csv", [
        dict(condition=name, component=i + 1, electrode=channel,
             component_scalp_rms_uv=float(rms_uv[label, i, j]))
        for label, name in enumerate(("piano", "guitar"))
        for i in range(args.components) for j, channel in enumerate(EEG_NAMES)])
    write_csv(args.output / "patterns.csv", [
        dict(component=i + 1, electrode=name, raw_pattern=float(raw[i, j]),
             normalized_pattern=float(normalized[i, j]))
        for i in range(args.components) for j, name in enumerate(EEG_NAMES)])
    write_csv(args.output / "trials.csv", rows)
    metadata = dict(
        fit_scope="all retained trials; descriptive visualization only, not cross-validation",
        settings={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        csp_parameters=csp.get_params(), epoch_shape=list(X.shape), sample_rate=fs,
        class_counts={"piano": int(sum(y == 0)), "guitar": int(sum(y == 1))},
        included_trial_numbers=[r["trial"] for r in rows if r["included"]],
        channel_order=EEG_NAMES, montage="standard_1020; approximate, not digitized participant locations",
        normalization="Each component divided by its maximum absolute electrode pattern value; sign unchanged",
        condition_comparison=dict(
            file="csp_patterns_piano_vs_guitar.png",
            quantity="Individual component scalp RMS in microvolts, not total EEG RMS",
            formula="abs(pattern[k, channel]) * sqrt(mean((filter[k] @ X_condition)**2)) * 1e6",
            averaging="Equal-weight mean over retained trials and time samples within each class",
            color_scale="0 to maximum RMS across both conditions, all components and channels",
            fit="Same joint CSP fit for both classes; no separate class-specific fit"),
        interpretation="Spatial patterns, not electrode importance, activation, or source localization. Polarity arbitrary.",
        versions={p: version(p) for p in ("numpy", "scipy", "mne", "scikit-learn", "bioread", "matplotlib")})
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.components} CSP pattern maps from {len(y)} trials to {args.output}")
    return metadata


if __name__ == "__main__":
    main()
