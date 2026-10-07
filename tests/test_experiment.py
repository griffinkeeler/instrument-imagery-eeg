"""Headless behavioral checks; these cannot validate physical display/TTL timing."""
import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('experiment', Path(__file__).resolve().parents[1] / 'scripts' / 'imagined_instrument.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Clock:
    t = 0.0
    def getTime(self):
        return self.t


class Window:
    def __init__(self, clock):
        self.clock, self.callbacks = clock, []
    def timeOnFlip(self, obj, key):
        self.callbacks.append(lambda: obj.update({key: self.clock.t}))
    def callOnFlip(self, fn):
        self.callbacks.append(fn)
    def flip(self):
        self.clock.t += 1 / 60
        callbacks, self.callbacks = self.callbacks, []
        for fn in callbacks:
            fn()


class Stimulus:
    def __init__(self, *args, **kwargs):
        pass
    def draw(self):
        pass


class Logger:
    def __init__(self):
        self.rows = []
    def check(self):
        pass
    def put(self, kind, row, console=None):
        self.rows.append((kind, dict(row)))


class Device(m.TriggerDevice):
    def __init__(self):
        self.codes = []
    def write(self, code):
        self.codes.append(code)


class ExperimentTests(unittest.TestCase):
    def test_balance_runs_and_reproduction(self):
        for seed in range(200):
            def generate():
                rng = random.Random(seed)
                history = []
                for _ in range(4):
                    block = m.balanced_sequence(rng, 10, history)
                    self.assertEqual(block.count('PIANO'), 10)
                    self.assertEqual(block.count('GUITAR'), 10)
                    history.extend(block)
                return history
            history = generate()
            self.assertEqual(history, generate())
            self.assertFalse(any(len(set(history[i:i+4])) == 1 for i in range(77)))

    def setup_experiment(self):
        event = types.SimpleNamespace(getKeys=lambda **kw: [], clearEvents=lambda **kw: None)
        visual = types.SimpleNamespace(TextStim=Stimulus, Circle=Stimulus)
        sys.modules['psychopy'] = types.SimpleNamespace(event=event, visual=visual)
        clock, logger, device = Clock(), Logger(), Device()
        context = dict(participant_id='test', session='1', block='', trial_in_block='',
                       overall_experimental_trial='', practice='', condition='', run_type='experiment')
        trigger = m.TriggerController(device, clock, logger, context, True)
        exp = m.Experiment(Window(clock), clock, logger, trigger, context, 60, 1)
        return exp, logger, device

    def test_full_trial_timing_markers_and_resets(self):
        for condition, cue, imagery in [('PIANO', 11, 21), ('GUITAR', 12, 22)]:
            exp, logger, device = self.setup_experiment()
            exp.run_trial(condition, 1, 1, 1, False, 1)
            self.assertEqual(device.codes, [3, 0, cue, 0, 20, 0, imagery, 0, 30, 0])
            row = [row for kind, row in logger.rows if kind == 'trials'][-1]
            self.assertEqual(row['status'], 'completed')
            for stage, duration in m.DURATIONS.items():
                self.assertAlmostEqual(row[stage + '_duration'], duration, places=8)
            event = next(row for kind, row in logger.rows if kind == 'events' and row['event_code'] == imagery)
            self.assertEqual(event['stimulus_onset'], row['imagery_onset'])
            self.assertEqual(event['psychopy_timestamp'], row['imagery_onset'])
            self.assertEqual(row['imagery_offset'], row['rest_onset'])

    def test_abort_preserves_partial_trial(self):
        exp, logger, device = self.setup_experiment()
        exp.event.getKeys = lambda **kw: ['escape'] if exp.clock.t > 3.1 else []
        with self.assertRaises(m.AbortExperiment):
            exp.run_trial('PIANO', 0, 1, '', True, 1)
        exp.trigger.reset(force=True)
        row = [row for kind, row in logger.rows if kind == 'trials'][-1]
        self.assertEqual(row['status'], 'aborted')
        self.assertTrue(row['practice'])
        self.assertEqual(row['overall_experimental_trial'], '')
        self.assertIn('imagery_onset', row)
        self.assertNotIn('imagery_offset', row)
        self.assertEqual(device.codes[-1], 0)

    def test_incremental_writer(self):
        import tempfile
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            logger = m.IncrementalCSV(Path(tmp), 'test')
            logger.put('events', {'event_code': 21, 'event_name': 'PIANO_IMAGERY_ONSET'})
            logger.close()
            with (Path(tmp) / 'test_events.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]['event_code'], '21')


if __name__ == '__main__':
    unittest.main()
