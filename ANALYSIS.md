# Piano versus guitar imagery: CSP + LDA

Run from the project root with the existing analysis environment:

```sh
/opt/anaconda3/envs/instrument-imagery/bin/python scripts/run_lda.py
```

The default recording is `data/Instrument_10_6_26.acq`. All paths are resolved
relative to the project, so running the script from PyCharm also works. The
script does not run on import. Dependencies are in `requirements-analysis.txt`.

## Protocol and fixed baseline

- Channels 0–8: F3, F4, C3, C4, P3, P4, Cz, Fz, Pz. Channel 9 is EOG,
  excluded from classification. The script validates channel names, sample
  rates, lengths, and units. Recorded mV are converted to volts.
- Digital channels 10–13 use a >2.5 V threshold and weights 1, 2, 4, 8.
  Each nonzero run is one event. Codes 1/2 are piano/guitar cues; 4/8 are
  piano/guitar imagery onsets. Labels come directly from 4/8 and the preceding
  cue is checked for agreement.
- The first four imagery trials are practice and are excluded by default,
  as confirmed for this recording. This leaves 40 trials per class. Use
  `--exclude-first 0` only for a recording without initial practice trials.
- Imagery lasts 10 seconds. The analysis uses **[2, 8) seconds** after imagery
  onset: 3,000 samples per trial at 500 Hz, one observation per whole trial.
  This follows the shared discussion's recommendation to omit onset activity.
- Each full 10-second imagery trial is filtered independently with a fourth-order
  Butterworth 8–30 Hz bandpass applied forward/backward (zero phase), then cropped.
  The order specifies the Butterworth design before forward/backward application.
  Filtering the full trial supplies two seconds of within-trial context at each
  end of the default analysis window, without mixing training/test trials or
  crossing recording segments. This is an offline, noncausal analysis.
- Four CSP components with Ledoit–Wolf covariance regularization and log power
  features feed LDA (`solver='lsqr', shrinkage='auto'`). CSP and LDA are refitted
  together inside every training fold. No component, band, or window search is
  performed. The acquired EEG reference is preserved; there is no added
  rereferencing, baseline subtraction, ICA, or EOG regression.

These acquisition settings follow the user's mapping for this file, rather than
older trigger codes or durations in the experiment script/documentation.

## Validation and interpretation

The primary estimate uses shuffled, stratified five-fold cross-validation with
seed 42. Every trial is held out once. Summary accuracy and balanced accuracy are
computed from all held-out predictions; mean fold balanced accuracy is also
reported and is the statistic used in the permutation test. Chance balanced
accuracy is 50%. The confusion matrix uses true classes as rows, predictions as
columns, in piano/guitar order.

A second check holds out four contiguous chronological chunks, training on the
other chunks. These are **not verified experimental blocks** and this is not
future-only evaluation. A drop relative to shuffled CV can indicate sensitivity
to recording drift. For a multi-session or multi-participant study, use actual
session/participant groups rather than this single-recording evaluation.

For a label-permutation test, run:

```sh
/opt/anaconda3/envs/instrument-imagery/bin/python scripts/run_lda.py --permutations 999
```

The default skips permutations for a quick run. Each permutation refits the
entire CSP+LDA pipeline, using the same stratified CV rule. The one-sided p-value
is `(1 + count(null >= observed)) / (1 + permutations)`. Global shuffling assumes
exchangeable trial labels: it does not preserve block balance or serial
randomization constraints. Treat this as exploratory unless that assumption is
appropriate for the actual trial randomization. Repeatedly changing analysis
settings based on the reported scores invalidates a confirmatory interpretation;
use nested validation or independent data to evaluate tuned choices.

Above-chance decoding alone does not establish that the signal comes from motor
imagery rather than eye movements, muscle activity, or other condition-correlated
artifacts. No artifact amplitude cutoff is imposed without reviewing the data.
`trials.csv` reports the maximum EEG-channel and EOG peak-to-peak amplitudes in
microvolts over the **unfiltered** analysis window. Optional, prespecified QC
thresholds are `--reject-eeg-uv` and `--reject-eog-uv`; explicit manual exclusions
use `--exclude-trials 7 12` (one-based original imagery trial numbers, including
practice). Review raw traces and EOG before interpreting the pilot scientifically.

Incomplete trials, segment-crossing trials, windows with nonfinite samples,
flat EEG channels, and trials interrupted by another event before the nominal
imagery end are excluded and logged. There are no imagery-offset triggers, so
10-second duration is supplied by the protocol, not independently verified.

## Outputs and options

The default output directory is `results/csp_lda/` (ignored by Git):

- `summary.json`: settings, package versions, counts, markers, CV metrics,
  confusion matrices, per-class recall, training/test epoch indices and fold audits,
  and optional permutation scores/p-value and null-distribution summaries.
- `events.csv`: all cue/imagery event onsets and pulse widths.
- `trials.csv`: every imagery trial's inclusion status, reason, and amplitude QC.
- `predictions.csv`: true labels, held-out predictions, and fold assignments.

Rerunning replaces those files. Use `--output results/another_run` to preserve a
separate analysis. Use `--help` for window, frequency, component, fold, and trial
exclusion options. No full-data fitted model is exported: outputs describe
held-out evaluation rather than training accuracy.

Checks:

```sh
/opt/anaconda3/envs/instrument-imagery/bin/python -m unittest discover -s tests -v
```

References: [shared discussion](https://chatgpt.com/share/6ac57578-e9fc-83e9-993d-e0e5dc9186b3),
[MNE CSP motor imagery example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html),
[scikit-learn permutation test](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html).

The validation tests instrument `CSP.fit` to verify its exact input trials and
labels for both ordinary cross-validation and shuffled-label permutations.
The fold audit checks disjoint training/test sets and exactly one held-out
prediction per trial. `trials.csv` maps zero-based epoch indices back to original
one-based imagery trial numbers.

## CSP pattern maps

```sh
/opt/anaconda3/envs/instrument-imagery/bin/python scripts/plot_csp_patterns.py
```

This script uses the same recording loader, trial exclusions, filter, epoch
window, and CSP configuration as `run_lda.py`. Defaults are four CSP components,
8–30 Hz, 2–8 s after imagery onset, and exclusion of the first four practice
trials. It supports `--input`, `--output`, `--components`, `--low`, `--high`,
`--tmin`, `--tmax`, `--duration`, `--exclude-first`, `--exclude-trials`,
`--reject-eeg-uv`, and `--reject-eog-uv`. If you change decoding preprocessing,
pass the same changes when generating the maps.

Outputs go to `results/csp_patterns/`:

- `csp_patterns.png` and `csp_patterns.svg`: labeled scalp maps.
- `patterns.csv`: raw MNE forward-pattern values and normalized values for each
  component/electrode (component numbers are one-based).
- `trials.csv`: trial selection and quality checks.
- `summary.json`: settings, CSP parameters, versions, and retained trial IDs.

The script fits CSP on **all retained trials for descriptive visualization**.
It neither computes accuracy nor passes this fitted CSP into cross-validation.
The components are the first four returned by the same CSP configuration used
in decoding; they are not ranked by LDA importance. Forward `patterns_` are
plotted, rather than decoding `filters_`. Each map is divided by its own maximum
absolute sensor value to show spatial shape; amplitudes are not comparable
between maps. Sign is unchanged and arbitrary, so red/blue does not identify
piano/guitar or activation/deactivation.

Locations use MNE's standard 10–20 montage, not digitized participant positions.
Linear interpolation is limited to the local sensor region; blank peripheral
areas do not imply zero activity. With nine electrodes these are coarse scalp
patterns, not precise brain sources or a quantitative electrode-importance
ranking. No additional artifact rejection is performed by default.

### Piano/guitar side-by-side comparison

The same command additionally saves `csp_patterns_piano_vs_guitar.png`, with
piano on the left, guitar on the right, and one shared CSP component per row.
`condition_pattern_rms.csv` contains the plotted values in microvolts.

CSP jointly learns one set of patterns from both classes, so a component's
spatial shape is shared. To compare its strength, the script projects each
trial through the fitted spatial filters and computes, separately by condition:

`component_scalp_RMS[k, channel] = abs(pattern[k, channel]) * sqrt(mean((filter[k] @ trial)**2))`

The mean is over samples and retained trials of that condition; results are
converted from volts to microvolts. This is the RMS of each component's own
sensor-space reconstruction. It preserves oscillatory amplitude without
cancellation from averaging signed EEG waveforms. The figure uses one common
zero-to-maximum color scale across both conditions and all components, without
separately normalizing the two conditions. Polarity is removed by RMS.

These are amplitude-scaled shared CSP patterns, not independently fitted piano
and guitar CSP patterns. They are not total EEG power maps: contributions from
multiple components and their cross-terms are not combined. As with the original
maps, this is a descriptive fit to all retained trials, not held-out inference
or an electrode-importance ranking. No statistical significance is assigned to
visual class differences.
