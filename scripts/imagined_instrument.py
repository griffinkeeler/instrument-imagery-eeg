"""Kinesthetic piano/guitar imagery EEG experiment; run in PsychoPy Coder.

All times use logging.defaultClock, set to the session clock before opening the
window. Flip times estimate display onset, NOT photon onset or BIOPAC arrival.
Use recorded hardware codes 21/22 for EEG epoching; validate latency physically.
"""
from __future__ import annotations

import csv
import json
import queue
import random
import re
import secrets
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------- Experimenter configuration ----------------
TRIGGER_MODE = "dummy"              # "dummy", "parallel", or "serial"
PARALLEL_ADDRESS = None              # Supply verified address/device node.
SERIAL_PORT = None                   # Supply verified port name.
BAUD_RATE = None                     # Supply adapter's required baud rate.
SERIAL_PROTOCOL_CONFIRMED = False    # True ONLY for the raw-byte protocol below.
TTL_PULSE_SECONDS = 0.010
# Resets are serviced after display flips, without sleeping in callbacks.
# At 60 Hz a 10 ms target normally becomes ~16.7 ms; long frames extend pulses.
# This is NOT a precision 10 ms pulse generator. If that width is required, use
# an adapter with hardware-timed pulses and replace the device implementation.
FULLSCREEN = True
WINDOW_SIZE = (1280, 800)
SCREEN = 0
MONITOR = "testMonitor"               # Replace with your calibrated profile.
FONT = "Arial"
DRY_RUN = False                      # Forces dummy mode; 2 practice + 4 study trials.
DRY_RUN_SHORTEN = True               # Only applied when DRY_RUN is True.
TRIGGER_TEST_ONLY = False            # Sends table once, then exits; no study trials.
RANDOM_SEED = None                   # None generates and records a fresh seed.
FALLBACK_REFRESH_HZ = 60.0
LONG_FRAME_FACTOR = 1.5
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data"
DURATIONS = dict(baseline=1.0, cue=0.75, preparation=1.25, imagery=4.0, rest=2.0)
CODES = {0: "IDLE_RESET", 1: "SESSION_START", 2: "BLOCK_START",
         3: "BASELINE_ONSET", 11: "PIANO_CUE_ONSET", 12: "GUITAR_CUE_ONSET",
         20: "PREPARATION_ONSET", 21: "PIANO_IMAGERY_ONSET",
         22: "GUITAR_IMAGERY_ONSET", 30: "REST_ONSET", 40: "BLOCK_END",
         255: "SESSION_END"}
CONTEXT_FIELDS = ["participant_id", "session", "block", "trial_in_block",
                  "overall_experimental_trial", "practice", "condition", "run_type"]
EVENT_FIELDS = CONTEXT_FIELDS + ["event_name", "event_code", "psychopy_timestamp",
    "stimulus_onset", "frame_number", "pulse_width_seconds", "detail"]
TRIAL_FIELDS = CONTEXT_FIELDS + ["order_index", "random_seed", "status"] + [
    f"{stage}_{suffix}" for stage in DURATIONS for suffix in
    ("onset", "offset", "duration", "nominal_duration", "planned_frames")]
FRAME_FIELDS = CONTEXT_FIELDS + ["frame_number", "flip_time", "interval_seconds", "long_frame"]


class AbortExperiment(Exception):
    """Raised by ESC; handled by the single shutdown path."""


def balanced_sequence(rng: random.Random, per_condition: int, previous=()) -> list[str]:
    """Rejection sampling preserves balance; also enforce the block boundary run.

    Each block gets a fresh shuffle, conditional on the last three study trials.
    Practice is randomized separately and does not constrain experimental order.
    """
    for _ in range(100000):
        sequence = ["PIANO"] * per_condition + ["GUITAR"] * per_condition
        rng.shuffle(sequence)
        combined = list(previous[-3:]) + sequence
        if not any(len(set(combined[i:i + 4])) == 1 for i in range(len(combined) - 3)):
            return sequence
    raise RuntimeError("Could not generate a constrained sequence")


class IncrementalCSV:
    """One disk writer keeps disk flushes and dummy printing off the flip path.

    Every row is flushed as received. A power loss can lose the small in-flight
    queue/OS cache; this is incremental saving, not a power-loss guarantee.
    Writer failures propagate to the presentation loop instead of going unnoticed.
    """
    def __init__(self, directory: Path, stem: str):
        self.files, self.writers = {}, {}
        self.pending = queue.Queue()
        self.error = None
        for name, fields in (("trials", TRIAL_FIELDS), ("events", EVENT_FIELDS),
                             ("frames", FRAME_FIELDS)):
            filename = f"{stem}.csv" if name == "trials" else f"{stem}_{name}.csv"
            handle = (directory / filename).open("x", newline="", encoding="utf-8")
            self.files[name] = handle
            self.writers[name] = csv.DictWriter(handle, fieldnames=fields)
            self.writers[name].writeheader()
            handle.flush()
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def _write(self):
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    break
                kind, row, console = item
                self.writers[kind].writerow(row)
                self.files[kind].flush()
                if console:
                    print(console, flush=True)
        except BaseException as error:
            self.error = error

    def check(self):
        if self.error:
            raise RuntimeError("Incremental data saving failed") from self.error

    def put(self, kind, row, console=None):
        self.check()
        self.pending.put((kind, dict(row), console))

    def close(self):
        self.pending.put(None)
        self.worker.join()
        for handle in self.files.values():
            handle.close()
        self.check()


class TriggerDevice:
    """Abstract byte-output interface; substitute adapter-specific protocol here."""
    def write(self, code: int):
        raise NotImplementedError

    def close(self):
        pass


class DummyDevice(TriggerDevice):
    def write(self, code: int):
        pass                            # Controller logs every write, including zero.


class ParallelDevice(TriggerDevice):
    def __init__(self):
        if PARALLEL_ADDRESS is None:
            raise ValueError("Supply PARALLEL_ADDRESS for your verified interface")
        from psychopy import parallel
        self.port = parallel.ParallelPort(address=PARALLEL_ADDRESS)

    def write(self, code: int):
        self.port.setData(code)

    def close(self):
        # Most ParallelPort backends have no public close(); release if provided.
        close = getattr(self.port, "close", None)
        if callable(close):
            close()
        self.port = None


class SerialDevice(TriggerDevice):
    def __init__(self):
        if not SERIAL_PORT or not BAUD_RATE or not SERIAL_PROTOCOL_CONFIRMED:
            raise ValueError("Supply serial settings and confirm adapter protocol")
        import serial
        self.port = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE,
                                  timeout=0, write_timeout=0)

    def write(self, code: int):
        # EXPLICIT PROTOCOL: one raw unsigned byte (21 -> b'\x15', NOT ASCII '21').
        # Requires an adapter which latches that byte onto eight TTL data lines
        # and clears them on b'\x00'. A generic serial cable is NOT such an adapter.
        # Supply your adapter manual/command framing if it uses another protocol.
        if self.port.write(bytes([code])) != 1:
            raise IOError("Serial trigger byte was not accepted")
        # Do not flush() here: it can wait for transmission. USB/serial buffering
        # means the timestamp measures submission, not TTL arrival at BIOPAC.

    def close(self):
        self.port.close()


class TriggerController:
    def __init__(self, device, clock, logger, context, dummy):
        self.device, self.clock, self.logger = device, clock, logger
        self.context, self.dummy = dict(context), dummy
        self.active_since = None

    def send_trigger(self, code, context, onset="", frame="", detail=""):
        if self.active_since is not None:
            raise RuntimeError("Overlapping triggers; increase event separation")
        self.context = dict(context)
        self.device.write(code)
        timestamp = self.clock.getTime()
        if code:
            self.active_since = timestamp
        self._log(code, timestamp, onset, frame, "", detail)

    def _log(self, code, timestamp, onset, frame, width, detail):
        row = dict(self.context, event_name=CODES[code], event_code=code,
                   psychopy_timestamp=timestamp, stimulus_onset=onset,
                   frame_number=frame, pulse_width_seconds=width, detail=detail)
        console = f"{timestamp:.6f}  {code:3d}  {CODES[code]} {detail}" if self.dummy else None
        self.logger.put("events", row, console)

    def reset(self, frame="", force=False):
        if force or self.active_since is not None:
            start = self.active_since
            self.device.write(0)         # Hardware reset happens BEFORE any logging.
            timestamp = self.clock.getTime()
            self.active_since = None
            self._log(0, timestamp, "", frame,
                      timestamp - start if start is not None else "", "pulse reset")

    def service(self, frame):
        if self.active_since is not None and self.clock.getTime() - self.active_since >= TTL_PULSE_SECONDS:
            self.reset(frame)


INSTRUCTIONS = (
    "In each trial, you will see either PIANO or GUITAR.\n\n"
    "When the instrument name appears, prepare to imagine yourself playing that "
    "instrument, but do not begin yet. A fixation cross (+) follows.\n\n"
    "When the cross changes to an open circle, begin imagining yourself physically "
    "playing C–D–E–F–G–A–B–C on the cued instrument.\n\n"
    "Focus on the movements and physical sensations of playing. Keep your actual "
    "hands, fingers, face, and body as still as possible.\n\n"
    "Continue imagining until the circle disappears.\n\n"
    "First you will complete practice trials. Press SPACE to begin practice."
)


class Experiment:
    def __init__(self, win, clock, logger, trigger, context, refresh_hz, seed):
        from psychopy import event, visual
        self.event, self.win, self.clock = event, win, clock
        self.logger, self.trigger, self.context = logger, trigger, context
        self.refresh_hz, self.seed, self.frame = refresh_hz, seed, 0
        self.last_flip = None
        self.intervals = []
        self.fixation = visual.TextStim(win, text="+", height=0.07, color="white", font=FONT, autoLog=False)
        self.cues = {name: visual.TextStim(win, text=name, height=0.07,
                     color="white", font=FONT, pos=(0, 0), autoLog=False) for name in ("PIANO", "GUITAR")}
        # Geometric open circle avoids font-dependent missing-glyph substitutions.
        # Exactly this same stimulus instance is used for both imagery conditions.
        self.circle = visual.Circle(win, radius=0.027, edges=128, lineColor="white",
                                    fillColor=None, lineWidth=2, pos=(0, 0), autoLog=False)
        self.message = visual.TextStim(win, text="", height=0.032, wrapWidth=1.3,
                                       color="white", font=FONT, autoLog=False)
        self.durations = {k: (v * 0.1 if DRY_RUN and DRY_RUN_SHORTEN else v)
                          for k, v in DURATIONS.items()}

    def check_escape(self):
        if "escape" in self.event.getKeys(keyList=["escape"]):
            raise AbortExperiment()
        self.logger.check()

    def flip(self, stimulus=None, code=None, context=None, record=False):
        self.check_escape()
        context = self.context if context is None else context
        if stimulus is not None:
            stimulus.draw()
        self.frame += 1
        timing = {}
        self.win.timeOnFlip(timing, "onset")
        if code is not None:
            # Callback executes on the SAME flip that first reveals the cue/circle.
            # timeOnFlip is queued first so the callback can log its flip timestamp.
            self.win.callOnFlip(lambda: self.trigger.send_trigger(
                code, context, timing["onset"], self.frame))
        self.win.flip()
        self.trigger.service(self.frame)
        onset = timing["onset"]
        if record and self.last_flip is not None:
            interval = onset - self.last_flip
            self.intervals.append(interval)
            self.logger.put("frames", dict(context, frame_number=self.frame,
                flip_time=onset, interval_seconds=interval,
                long_frame=interval > LONG_FRAME_FACTOR / self.refresh_hz))
        self.last_flip = onset if record else None
        return onset

    def screen(self, text, code=None, context=None):
        self.win.recordFrameIntervals = False
        self.message.text = text
        self.event.clearEvents(eventType="keyboard")
        self.flip(self.message, code, context)
        while True:
            self.flip(self.message)
            if "space" in self.event.getKeys(keyList=["space"]):
                # Do not allow a rapid SPACE response to overlap marker pulses.
                self.drain_pulse(self.message)
                return

    def drain_pulse(self, stimulus=None):
        while self.trigger.active_since is not None:
            self.flip(stimulus)

    def software_event(self, name, context, onset="", detail=""):
        self.logger.put("events", dict(context, event_name=name, event_code="",
            psychopy_timestamp=self.clock.getTime(), stimulus_onset=onset,
            frame_number=self.frame, detail=detail))

    def run_trial(self, condition, block, trial, overall, practice, order):
        context = dict(self.context, condition=condition, block=block,
            trial_in_block=trial, overall_experimental_trial=overall, practice=practice)
        row = dict(context, order_index=order, random_seed=self.seed, status="started")
        self.logger.put("trials", row)
        stages = [("baseline", self.fixation, 3), ("cue", self.cues[condition],
            11 if condition == "PIANO" else 12), ("preparation", self.fixation, 20),
            ("imagery", self.circle, 21 if condition == "PIANO" else 22), ("rest", None, 30)]
        previous_stage = None
        self.last_flip = None
        self.win.recordFrameIntervals = True
        try:
            for stage, stimulus, code in stages:
                frames = max(1, round(self.durations[stage] * self.refresh_hz))
                onset = self.flip(stimulus, code, context, record=True)
                row[f"{stage}_onset"] = onset
                row[f"{stage}_nominal_duration"] = self.durations[stage]
                row[f"{stage}_planned_frames"] = frames
                if previous_stage:
                    row[f"{previous_stage}_offset"] = onset
                    row[f"{previous_stage}_duration"] = onset - row[f"{previous_stage}_onset"]
                if stage == "rest":
                    self.software_event("IMAGERY_OFFSET", context, onset)
                row["status"] = f"{stage}_in_progress"
                self.logger.put("trials", row)
                # Stage's first flip above counts as frame one. The next stage's
                # first flip ends it. Quantize to nearest refresh; never add frames
                # for pulse width. Nominal imagery stays 4 s in real sessions.
                for _ in range(frames - 1):
                    self.flip(stimulus, context=context, record=True)
                previous_stage = stage
            offset = self.flip(context=context, record=True)
            row["rest_offset"] = offset
            row["rest_duration"] = offset - row["rest_onset"]
            row["status"] = "completed"
            self.software_event("TRIAL_COMPLETE", context, offset)
        except BaseException as error:
            row["status"] = "aborted" if isinstance(error, AbortExperiment) else "error"
            raise
        finally:
            self.win.recordFrameIntervals = False
            # Append-only checkpoints: use the LAST row per practice/block/trial
            # for trial-level analysis. Partial trials retain observed onsets only.
            self.logger.put("trials", row)

    def trigger_test(self):
        """Verify every nonzero code, with >=1 second of idle between pulses."""
        self.screen("Trigger test. Start AcqKnowledge recording.\n\nPress SPACE to send each code once.")
        self.trigger.reset(force=True)
        for code in CODES:
            if code == 0:
                continue
            self.message.text = f"Trigger test: {code} = {CODES[code]}"
            self.flip(self.message, code)
            self.drain_pulse(self.message)
            idle_start = self.clock.getTime()
            while self.clock.getTime() - idle_start < 1.0:
                self.flip(self.message)
        self.screen("Trigger test complete.\n\nPress SPACE to close.")


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    from psychopy import core, event, gui, logging, visual, __version__
    for code, name in CODES.items():
        print(f"{code:3d} = {name}")
    if TTL_PULSE_SECONDS <= 0 or TTL_PULSE_SECONDS >= min(DURATIONS.values()) * (0.1 if DRY_RUN and DRY_RUN_SHORTEN else 1):
        raise ValueError("Pulse duration must be positive and shorter than every stage")
    info = {"Participant ID": "", "Session": ["1", "2", "3"]}
    while True:
        dialog = gui.DlgFromDict(info, title="Imagined instrument EEG", sortKeys=False)
        if not dialog.OK:
            return
        participant = str(info["Participant ID"]).strip()
        if re.fullmatch(r"[A-Za-z0-9_-]+", participant) and str(info["Session"]) in ("1", "2", "3"):
            break
        error_dialog = gui.Dlg(title="Invalid ID/session")
        error_dialog.addText("Use a nonempty ID with letters, digits, hyphens or underscores; session 1–3.")
        error_dialog.show()
    run_type = "trigger-test" if TRIGGER_TEST_ONLY else "dry-run" if DRY_RUN else "experiment"
    stem = f"sub-{participant}_session-{info['Session']}_imagined-instrument"
    if run_type != "experiment":
        stem += f"_{run_type}"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    directory = OUTPUT_DIRECTORY / stem
    # Atomic reservation protects ALL outputs from overwrite, including crashes.
    # To repeat a session deliberately, archive its existing directory first.
    try:
        directory.mkdir()
    except FileExistsError:
        notice = gui.Dlg(title="Existing data protected")
        notice.addText(f"Data already exist:\n{directory}\nArchive them before repeating this session.")
        notice.show()
        return
    seed = RANDOM_SEED if RANDOM_SEED is not None else secrets.randbits(32)
    rng = random.Random(seed)
    practice = balanced_sequence(rng, 1 if DRY_RUN else 2)
    blocks, history = [], []
    for _ in range(1 if DRY_RUN else 4):
        sequence = balanced_sequence(rng, 2 if DRY_RUN else 10, history)
        blocks.append(sequence)
        history.extend(sequence)
    context = dict(participant_id=participant, session=str(info["Session"]),
        block="", trial_in_block="", overall_experimental_trial="", practice="",
        condition="", run_type=run_type)
    mode = "dummy" if DRY_RUN else TRIGGER_MODE
    metadata = dict(context, status="initializing", random_seed=seed,
        started_utc=datetime.now(timezone.utc).isoformat(), psychopy_version=__version__,
        trigger_mode=mode, pulse_target_seconds=TTL_PULSE_SECONDS,
        serial_encoding="raw unsigned byte; zero clears adapter output",
        parallel_address=PARALLEL_ADDRESS, serial_port=SERIAL_PORT, baud_rate=BAUD_RATE,
        fullscreen=FULLSCREEN, window_size=WINDOW_SIZE, screen=SCREEN, monitor=MONITOR,
        dry_run=DRY_RUN, nominal_durations=DURATIONS, practice_order=practice,
        experimental_blocks=blocks, timestamp_reference="logging.defaultClock = session Clock",
        trial_csv_semantics="append-only snapshots; use last row per practice/block/trial")
    metadata_path = directory / f"{stem}_session.json"
    save_json(metadata_path, metadata)
    logger = win = device = trigger = experiment = None
    cleanup_errors = []
    try:
        logger = IncrementalCSV(directory, stem)
        clock = core.Clock()
        logging.setDefaultClock(clock)
        device = {"dummy": DummyDevice, "parallel": ParallelDevice, "serial": SerialDevice}[mode]()
        trigger = TriggerController(device, clock, logger, context, mode == "dummy")
        trigger.reset(force=True)
        win = visual.Window(size=WINDOW_SIZE, fullscr=FULLSCREEN, screen=SCREEN,
            monitor=MONITOR, units="height", color=(0, 0, 0), waitBlanking=True,
            checkTiming=False, allowGUI=False, autoLog=False)
        win.mouseVisible = False
        # Measure refresh while polling ESC on every frame (including startup).
        samples = []
        last = None
        for i in range(150):
            if "escape" in event.getKeys(keyList=["escape"]):
                raise AbortExperiment()
            now = win.flip()
            if last is not None and i > 30:
                samples.append(now - last)
            last = now
        import statistics
        median = statistics.median(samples) if samples else None
        stable = bool(median and max(samples) < median * LONG_FRAME_FACTOR)
        hz = 1 / median if median and median > 0 else FALLBACK_REFRESH_HZ
        metadata.update(measured_refresh_hz=1 / median if median else None,
                        refresh_estimate_stable=stable, refresh_hz_used=hz,
                        calibration_intervals_seconds=samples)
        win.refreshThreshold = LONG_FRAME_FACTOR / hz
        experiment = Experiment(win, clock, logger, trigger, context, hz, seed)
        metadata.update(status="running", effective_durations=experiment.durations)
        save_json(metadata_path, metadata)
        print(f"Refresh estimate: {hz:.3f} Hz; stable: {stable}; seed: {seed}")
        if not stable:
            experiment.screen("Refresh measurement was unstable.\n\nResearcher: inspect the display setup.\nESC to stop, or SPACE to continue with this estimate.")
        if TRIGGER_TEST_ONLY:
            experiment.trigger_test()
        else:
            experiment.screen(INSTRUCTIONS, 1)
            for trial, condition in enumerate(practice, 1):
                experiment.run_trial(condition, 0, trial, "", True, trial)
            experiment.screen("Practice complete.\n\nPlease ask the researcher any questions you have.\n\nPress SPACE when the researcher tells you to begin.")
            overall = 0
            for block, sequence in enumerate(blocks, 1):
                block_context = dict(context, block=block, practice=False)
                experiment.flip(code=2, context=block_context)
                experiment.drain_pulse()
                for trial, condition in enumerate(sequence, 1):
                    overall += 1
                    experiment.run_trial(condition, block, trial, overall, False, overall)
                text = ("Block complete.\n\nYou may take a short break.\n\nPress SPACE when you are ready to continue."
                        if block < len(blocks) else "All blocks complete.\n\nPress SPACE to finish.")
                experiment.screen(text, 40, block_context)
            experiment.flip(code=255)
            experiment.drain_pulse()
        metadata["status"] = "completed"
    except AbortExperiment:
        metadata["status"] = "aborted"
    except BaseException:
        metadata["status"] = "error"
        metadata["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        # Each cleanup is isolated so a disk failure cannot skip the hardware reset.
        # No software can guarantee reset after power loss/OS kill/disconnected
        # hardware; hardware watchdogs or self-clearing adapters cover those cases.
        if device is not None:
            try:
                if trigger is not None:
                    trigger.reset(force=True)
                else:
                    device.write(0)
            except BaseException as error:
                cleanup_errors.append(f"Reset failed: {error}")
            try:
                device.close()
            except BaseException as error:
                cleanup_errors.append(f"Device close failed: {error}")
        if win is not None:
            try:
                win.close()
            except BaseException as error:
                cleanup_errors.append(f"Window close failed: {error}")
        if experiment is not None:
            intervals = experiment.intervals
            summary = dict(frame_count=len(intervals),
                long_frames=sum(v > LONG_FRAME_FACTOR / experiment.refresh_hz for v in intervals),
                long_frame_threshold_seconds=LONG_FRAME_FACTOR / experiment.refresh_hz,
                maximum_interval_seconds=max(intervals, default=0),
                mean_interval_seconds=sum(intervals) / len(intervals) if intervals else None)
            metadata["timing_quality"] = summary
            print("Timing quality:", json.dumps(summary, indent=2))
            try:
                experiment.software_event("SESSION_STATUS", context, detail=metadata["status"])
            except BaseException as error:
                cleanup_errors.append(f"Final event save failed: {error}")
        if logger is not None:
            try:
                logger.close()
            except BaseException as error:
                cleanup_errors.append(f"Data save failed: {error}")
        metadata["cleanup_errors"] = cleanup_errors
        if cleanup_errors and metadata["status"] == "completed":
            metadata["status"] = "error"
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(metadata_path, metadata)
        print(f"Session {metadata['status']}; outputs: {directory}")
        for error in cleanup_errors:
            print(error)


if __name__ == "__main__":
    main()
