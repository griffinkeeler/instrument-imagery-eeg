# Classification of Imagined Instrument from EEG 
The goal of this project is to test whether EEG can identify what musical instrument a person imagines themselves
playing above chance. 

## Experiment

Run [`imagined_instrument.py`](imagined_instrument.py) in PsychoPy Coder. It defaults to dummy triggers; configure the actual interface before collection. See the [setup, trigger testing, output, and EEGLAB guide](docs/RUN_EXPERIMENT.md).

Headless checks: `python3 -m unittest discover -s tests -v`.
