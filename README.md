# Classification of Imagined Instrument from EEG 
The goal of this project is to test whether EEG can identify what musical instrument a person imagines themselves
playing above chance. 

## Experiment

Run [`scripts/imagined_instrument.py`](scripts/imagined_instrument.py) in PsychoPy Coder. It defaults to dummy triggers; configure the actual interface before collection. See the [setup, trigger testing, output, and EEGLAB guide](docs/RUN_EXPERIMENT.md).

Headless experiment checks: `python3 -m unittest discover -s tests -p test_experiment.py -v`. Analysis checks require the analysis environment (see below).

## Offline CSP + LDA analysis

Run `scripts/run_lda.py` in the `instrument-imagery` Python environment to
classify piano versus guitar imagery. The default excludes the first four
practice trials and uses 2–8 seconds of each 10-second imagery period.
See [analysis instructions and assumptions](ANALYSIS.md) for validation,
artifact checks, permutation testing, and output files.

Run `scripts/plot_csp_patterns.py` in the same environment to save four labeled
CSP scalp pattern maps and their numerical values. See the CSP pattern maps
section in [ANALYSIS.md](ANALYSIS.md) for interpretation and options.
