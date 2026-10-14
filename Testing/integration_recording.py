"""Integration test of GTReview's microphone and video recording, inside Slicer.

Run it by hand (a display is needed; ``xvfb-run -a`` provides one)::

    Slicer --no-splash \
        --additional-module-path <repo>/GTReview \
        --python-script <repo>/Testing/integration_recording.py

No microphone is opened: a stand-in (:class:`FakeMicrophone`) delivers a hum at
the wall clock's pace from a Qt timer, the way Qt's own audio backends deliver
on the main thread.  It can hold audio through a freeze and deliver it late (as
PulseAudio does) or drop it (as CoreAudio and ALSA do).  The video is the real
thing, rendered from the slice views and encoded by ffmpeg, which the test
needs (as the video does), with ffprobe next to it.

On two synthetic cases (numpy + SimpleITK, never patient data) it

1. loads the batch and checks the Recording section is ready: both switches
   enabled, a microphone list, the views to choose from;
2. records the microphone and the view area together while lesions are
   selected, re-clicked, deleted and brought back by Undo (which renumbers the
   selected one), while the main thread freezes twice -- once with the audio
   held, once with it dropped -- and across a switch to the next case, where
   the video is pointed at one slice view;
3. checks what the first case left: ``GTReview-recordings/microphone/<case>.wav``
   with the lesions spoken in and a cue point per lesion change, and
   ``video/<case>.mp4`` with that same sound, a chapter per lesion stretch and
   the red-dot banner on frames that change as the reader moves; each with its
   event log, whose seed voxels lie inside the lesion every event names (also
   after the renumbering); audio and video as long as the wall clock ran,
   through both freezes; and nothing else in the folders;
4. checks the second case's takes and the view switch in the video's log;
5. stops the microphone before the video: the video keeps its sound;
6. without ffmpeg the video does not start, and ffmpeg is offered;
7. a microphone that fails to open: the microphone switch goes back off and
   the video records without sound, saying so; one unplugged mid-recording
   stops the microphone, keeps what it had and leaves the video running;
8. a microphone that delivers only zeros is reported;
9. high-DPI grabs, a lesion selected before Record (announced at the start),
   a disk that fails mid-recording, the frames owed for a freeze;
10. one recording per case: recording a recorded case asks first; "keep it"
    leaves it alone, "replace" replaces it -- with the newest take even while
    the one before is still being saved; moving on to a recorded case while
    recording asks too, and "stop recording" stops; a reload of the case goes
    on in the same take;
11. Open folder starts the file manager without Slicer's library paths;
12. a take is saved within seconds while Slicer idles in its event loop.

Not named ``test_*`` so the unit-test discovery leaves it alone.  Set
GTREVIEW_KEEP_RECORDINGS=1 to keep the temporary batch and its recordings.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest

import numpy as np
import SimpleITK as sitk

import qt
import slicer

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import GTReview as gtreview  # noqa: E402
from GTReviewLib import recordlog, recording, wavfile  # noqa: E402


class _Tee(object):
    """Everything printed, also into GTREVIEW_TEST_LOG: Slicer on Windows is a
    GUI program, and what it prints reaches no console."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if os.environ.get("GTREVIEW_TEST_LOG"):
    sys.stdout = _Tee(sys.stdout, open(os.environ["GTREVIEW_TEST_LOG"], "w", encoding="utf-8", buffering=1))
    # and GTReview's own warnings with it, which otherwise only reach Slicer's log
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.WARNING)
    _handler.setFormatter(logging.Formatter("  [log] %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_handler)

MICROPHONE, VIDEO = recording.MICROPHONE, recording.VIDEO
SIZE = (40, 40, 12)
SPACING = (1.0, 1.0, 3.0)
ORIGIN = (-20.0, -20.0, -18.0)
DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#: three lesions of three sizes, so they are numbered A=1, B=2, C=3
BLOCK_A = (slice(4, 10), slice(4, 10), slice(3, 6))     # 108 voxels, label 2
BLOCK_B = (slice(26, 31), slice(4, 9), slice(3, 6))     # 75 voxels, label 1
BLOCK_C = (slice(26, 30), slice(28, 32), slice(6, 9))   # 48 voxels, label 3
CASE_ONE, CASE_TWO = "RC_001", "RC_002"
SEQUENCES = ("t1", "t1c", "t2", "flair")
#: the fake microphone's hum
HUM_HZ, HUM_LEVEL = 220.0, 3000.0


class Checks(object):
    def __init__(self):
        self.passed, self.failed = [], []

    def check(self, ok, description, detail=""):
        if ok:
            self.passed.append(description)
            print("  [ OK ] {}".format(description))
            sys.stdout.flush()
            return True
        self.failed.append((description, detail))
        print("  [FAIL] {}{}".format(description, ("  --  " + str(detail)) if detail else ""))
        sys.stdout.flush()
        raise AssertionError(description)

    def step(self, message):
        print("\n== {}".format(message))
        sys.stdout.flush()


CHECKS = Checks()


def pump(seconds=0.3):
    deadline = time.time() + seconds
    while time.time() < deadline:
        slicer.app.processEvents()
        time.sleep(0.005)


def writeVolume(path, array_ijk):
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_ijk.transpose(2, 1, 0)))
    image.SetOrigin(ORIGIN)
    image.SetSpacing(SPACING)
    image.SetDirection(DIRECTION)
    sitk.WriteImage(image, path, True)


def buildBatch(root):
    mask = np.zeros(SIZE, dtype=np.uint8)
    mask[BLOCK_A], mask[BLOCK_B], mask[BLOCK_C] = 2, 1, 3
    one = os.path.join(root, CASE_ONE)
    os.makedirs(one)
    image = np.full(SIZE, 20, dtype=np.int16)
    image[mask > 0] = 200
    writeVolume(os.path.join(one, CASE_ONE + "_t1c.nii.gz"), image)
    writeVolume(os.path.join(one, CASE_ONE + "_seg.nii.gz"), mask)
    two = os.path.join(root, CASE_TWO)
    os.makedirs(two)
    second = np.zeros(SIZE, dtype=np.uint8)
    second[BLOCK_A], second[BLOCK_C] = 1, 2
    for n, key in enumerate(SEQUENCES, start=1):
        image = np.full(SIZE, 20 * n, dtype=np.int16)
        image[second > 0] = 200
        writeVolume(os.path.join(two, "{}_{}.nii.gz".format(CASE_TWO, key)), image)
    writeVolume(os.path.join(two, CASE_TWO + "_seg.nii.gz"), second)
    return mask


def blockOf(seed):
    i, j, k = seed
    for name, block in (("A", BLOCK_A), ("B", BLOCK_B), ("C", BLOCK_C)):
        if all(s.start <= v < s.stop for s, v in zip(block, (i, j, k))):
            return name
    return None


class FakeMicrophone(object):
    """Stands in for recording.MicrophoneInput: a 220 Hz hum at the wall
    clock's pace, delivered from a Qt timer on the main thread."""

    def __init__(self, onData, onLevel=None, rate=16000, silent=False):
        self._onData, self._onLevel = onData, onLevel
        self.rate, self.channels, self.label = rate, 1, "Fake microphone"
        self.started = None
        self.lastData = None
        self.heardSomething = False
        self.silent = silent
        self.unplugged = False
        self.dropNextFreeze = False
        self.failOnStart = False
        self.dropped = 0.0  # seconds thrown away, as a device that cannot hold them would
        self._produced = 0
        self._active = False
        self._timer = qt.QTimer()
        self._timer.setInterval(40)
        self._timer.connect("timeout()", self.drain)

    @property
    def active(self):
        return self._active

    def start(self, name, realm, label="", hold=True):
        if self.failOnStart:
            raise RuntimeError("the fake microphone refuses to open")
        self.started = time.monotonic()
        self.lastData = self.started
        self._produced = 0
        self._active = True
        self._timer.start()

    def failed(self):
        return False

    def gone(self):
        return False

    def drain(self):
        if not self._active or self.unplugged:
            return
        self.lastData = time.monotonic()
        owed = int((time.monotonic() - self.started) * self.rate) - self._produced
        if owed <= 0:
            return
        if self.dropNextFreeze and owed > self.rate:
            # a freeze longer than a second: everything but the last 50 ms is gone
            kept = int(0.05 * self.rate)
            self.dropped += (owed - kept) / float(self.rate)
            self._produced += owed - kept
            owed = kept
            self.dropNextFreeze = False
        t = np.arange(self._produced, self._produced + owed) / float(self.rate)
        level = 0.0 if self.silent else HUM_LEVEL
        samples = (np.sin(2 * np.pi * HUM_HZ * t) * level).astype("<i2")
        self._produced += owed
        if np.any(samples):
            self.heardSomething = True
        if self._onLevel is not None:
            self._onLevel(-20.0)
        self._onData(samples.tobytes())

    def stop(self):
        self.drain()
        self._active = False
        self._timer.stop()
        if self._onLevel is not None:
            self._onLevel(None)


def ffprobe(path, entries, section="stream"):
    out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries",
                          "{}={}".format(section, entries), "-of", "json", path],
                         capture_output=True, text=True, env=recording._environment())
    return json.loads(out.stdout or "{}")


def chapters(path):
    out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_chapters", "-of", "json", path],
                         capture_output=True, text=True, env=recording._environment())
    return json.loads(out.stdout or "{}").get("chapters", [])


def frameAt(video, seconds, png):
    ffmpeg = recording.findFfmpeg()[0]
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", str(seconds), "-i", video, "-frames:v", "1", png],
                   env=recording._environment(), check=True)
    return qt.QImage(png)


def soundOf(video, wav):
    """The video's sound track, decoded: ``(rate, samples)``."""
    subprocess.run([recording.findFfmpeg()[0], "-v", "error", "-y", "-i", video, "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", wav], env=recording._environment(), check=True)
    return wavfile.read_wav(wav)


def humResidual(samples, rate, seconds, length=0.8):
    """RMS of what is left of the audio from *seconds* on once the best-fitting
    hum is taken out: about nothing where only the fake microphone is heard,
    a lot where a voice was mixed in."""
    start = max(0, int(seconds * rate))
    x = samples[start:start + int(length * rate), 0].astype(np.float64)
    if x.size < rate // 10:
        return 0.0
    t = (np.arange(x.size) + start) / float(rate)
    basis = np.stack([np.sin(2 * np.pi * HUM_HZ * t), np.cos(2 * np.pi * HUM_HZ * t), np.ones_like(t)], axis=1)
    coefficients = np.linalg.lstsq(basis, x, rcond=None)[0]
    return float(np.sqrt(np.mean((x - basis @ coefficients) ** 2)))


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) if samples.size else 0.0


def waitForFiles(widget, timeout=120.0):
    deadline = time.time() + timeout
    while widget.recorder.busyFinishing() and time.time() < deadline:
        pump(0.2)
    pump(0.1)  # the recorder reports what was saved from the event loop
    return widget.recorder.busyFinishing() == 0


class RecordingIntegrationTest(unittest.TestCase):
    root = None
    widget = None
    mask = None

    @classmethod
    def setUpClass(cls):
        CHECKS.step("Building two synthetic cases and opening the module")
        cls.root = tempfile.mkdtemp(prefix="gtreview_recording_")
        cls.mask = buildBatch(cls.root)
        mainWindow = slicer.util.mainWindow()
        if mainWindow is not None:
            mainWindow.resize(1400, 1000)
        pump()
        slicer.util.selectModule("GTReview")
        pump()
        cls.widget = slicer.modules.gtreview.widgetRepresentation().self()
        # what macOS makes of this Slicer (test_18); the stand-in microphone
        # needs no permission, so the other steps run as if it had one
        cls.macProblem = recording.macMicrophoneProblem()
        recording._MAC_MICROPHONE["problem"] = None
        cls.warnings = []
        cls._warningDisplay = slicer.util.warningDisplay
        cls._confirm = slicer.util.confirmYesNoDisplay
        cls._error = slicer.util.errorDisplay
        slicer.util.warningDisplay = lambda text, *a, **k: cls.warnings.append(str(text))
        slicer.util.errorDisplay = lambda text, *a, **k: cls.warnings.append(str(text))
        slicer.util.confirmYesNoDisplay = lambda *a, **k: True
        # "already has a recording -- replace it?": answered from cls.answers
        # (replace when empty), every question kept in cls.asked
        cls.asked, cls.answers = [], []

        def confirmReplace(caseId, kinds, switching):
            cls.asked.append((caseId, tuple(kinds), switching))
            return cls.answers.pop(0) if cls.answers else True

        cls.widget._confirmReplaceRecording = confirmReplace

    @classmethod
    def tearDownClass(cls):
        slicer.util.warningDisplay = cls._warningDisplay
        slicer.util.errorDisplay = cls._error
        slicer.util.confirmYesNoDisplay = cls._confirm
        try:
            if cls.widget is not None and cls.widget.recorder is not None:
                cls.widget.recorder.shutdown(30)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        if os.environ.get("GTREVIEW_KEEP_RECORDINGS"):
            print("  kept: {}".format(cls.root))
        elif cls.root:
            shutil.rmtree(cls.root, ignore_errors=True)

    # ---------------------------------------------------------------- helpers
    def fake(self, **kwargs):
        widget = self.widget
        mic = FakeMicrophone(widget.recorder._onAudio, widget._onRecorderLevel, **kwargs)
        widget.recorder.mic = mic
        return mic

    def exported(self, kind, case):
        return recording.recordingPath(os.path.join(self.root, recording.RECORDINGS_FOLDER), kind, case)

    def log(self, kind, case):
        with open(os.path.splitext(self.exported(kind, case))[0] + ".json", encoding="utf-8") as handle:
            return json.load(handle)

    def seconds(self, kind, case):
        if kind == MICROPHONE:
            rate, samples = wavfile.read_wav(self.exported(kind, case))
            return samples.shape[0] / float(rate)
        return float(ffprobe(self.exported(kind, case), "duration", section="format")["format"]["duration"])

    def leftovers(self):
        """Anything in microphone/ and video/ but the cases' recordings and logs."""
        out = []
        for kind in (MICROPHONE, VIDEO):
            folder = os.path.join(self.root, recording.RECORDINGS_FOLDER, kind)
            for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
                stem, extension = os.path.splitext(name)
                if extension not in (recording.MEDIA_SUFFIX[kind], ".json") or "." in stem:
                    out.append("{}/{}".format(kind, name))
        return out

    def selectLesion(self, number, settle=0.05):
        table = self.widget.lesionTable
        for row in range(table.rowCount):
            if table.item(row, 0).data(qt.Qt.UserRole) == number:
                table.selectRow(row)
                pump(settle)
                return
        raise AssertionError("no lesion #{} in the table".format(number))

    # ------------------------------------------------------------------ steps
    def test_01_ready_to_record(self):
        CHECKS.step("The Recording section is ready once a batch is loaded")
        widget = self.widget
        CHECKS.check(widget.recorder is not None, "the recorder was built")
        CHECKS.check(not widget.recordAudioButton.enabled, "nothing can record before a case is open")
        widget.datasetPathEdit.currentPath = self.root
        widget.onLoadDataset()
        pump(0.5)
        CHECKS.check(widget.currentCase().case_id == CASE_ONE, "the first case is open")
        CHECKS.check(widget.recordAudioButton.enabled and widget.recordVideoButton.enabled,
                     "both switches are enabled")
        CHECKS.check("microphone and /video" in widget.recordingStatusLabel.text,
                     "the panel says where recordings go", widget.recordingStatusLabel.text)
        CHECKS.check(widget.microphoneComboBox.count >= 1
                     and widget.microphoneComboBox.itemText(0) == "System default",
                     "the microphone list starts with the system default",
                     str([widget.microphoneComboBox.itemText(i) for i in range(widget.microphoneComboBox.count)]))
        labels = [widget.videoSourceComboBox.itemText(i) for i in range(widget.videoSourceComboBox.count)]
        CHECKS.check(labels and labels[0] == "All views", "the video can show all views", str(labels))
        devices = recording.inputDevices()
        CHECKS.check(all(not d["name"].endswith(".monitor") for d in devices),
                     "no speaker monitor is offered as a microphone", str([d["name"] for d in devices]))
        CHECKS.check(len(widget.lesionList) == 3, "three lesions", str(len(widget.lesionList)))

    def test_02_record_a_review(self):
        CHECKS.step("Record microphone and video through a review with two freezes and a case switch")
        widget = self.widget
        mic = self.fake()
        widget.videoSourceComboBox.currentIndex = 0
        widget.recordAudioButton.click()
        widget.recordVideoButton.click()
        pump(0.3)
        recorder = widget.recorder
        CHECKS.check(recorder.audioOn and recorder.videoOn, "both are recording", str(self.warnings))
        CHECKS.check(not self.asked, "a case without recordings is not asked about", str(self.asked))
        CHECKS.check(widget.recordAudioButton.text == "Stop microphone", "the button says how to stop")
        CHECKS.check(widget._recIndicator is not None and widget._recIndicator.visible,
                     "a REC sign is in the status bar")
        micTake, videoTake = recorder.takes.get(MICROPHONE), recorder.takes.get(VIDEO)
        CHECKS.check(micTake is not None and videoTake is not None, "a take of each kind is open")
        CHECKS.check(os.path.isfile(micTake.working(".recording.wav")),
                     "the microphone is written as it comes in")
        CHECKS.check(videoTake.wav is not None and os.path.isfile(videoTake.working(".recording.wav")),
                     "the video records the microphone too")
        CHECKS.check(not widget.microphoneComboBox.enabled and not widget.testMicrophoneButton.enabled,
                     "the microphone cannot be changed or tested while it records")
        pump(1.2)
        self.selectLesion(1)                      # A
        pump(1.5)
        for number in (2, 3, 2):                  # a burst: only the last is announced
            self.selectLesion(number, settle=0.0)  # (no redraws between: slow machines)
        pump(1.5)
        table = widget.lesionTable
        row = [r for r in range(table.rowCount) if table.item(r, 0).data(qt.Qt.UserRole) == 2][0]
        widget.onLesionClicked(table.item(row, 0))  # the same lesion again
        pump(0.5)

        # a freeze the sound system bridges: the audio arrives late, nothing lost
        time.sleep(2.0)
        pump(1.0)

        self.selectLesion(1)                      # A, then delete it: B and C move up
        pump(0.6)
        widget.onDeleteLesion()
        pump(1.0)
        CHECKS.check(len(widget.lesionList) == 2, "lesion A deleted")
        self.selectLesion(1)                      # B, now number 1
        pump(1.2)
        widget.onUndo()                           # A is back: B is number 2 again
        pump(1.5)
        CHECKS.check(widget.selectedLesion() is not None and widget.selectedLesion().index == 2,
                     "after Undo the selected lesion is number 2 again")

        # a freeze the sound system does not bridge: the audio is lost
        mic.dropNextFreeze = True
        time.sleep(2.5)
        pump(3.0)
        self.__class__.dropped = mic.dropped

        # the delete and the undo left the case "unsaved": no Save / Discard prompt
        widget._confirmDiscardIfNeeded = lambda *args, **kwargs: True
        widget.onNextCase()
        pump(0.8)
        CHECKS.check(widget.currentCase().case_id == CASE_TWO, "the second case is open")
        CHECKS.check(set(recorder.takes) == {MICROPHONE, VIDEO}
                     and all(take.name == CASE_TWO for take in recorder.takes.values()),
                     "the recording carried on into takes of the second case")
        CHECKS.check(not self.asked, "it had no recordings, so nothing was asked")
        CHECKS.check(widget.layoutComboBox.itemData(widget.layoutComboBox.currentIndex)
                     == gtreview.SEQUENCES_LAYOUT_ID, "the second case shows its sequences side by side")
        index = widget.videoSourceComboBox.findData("t1c")
        CHECKS.check(index > 0, "the t1c view can be picked for the video")
        widget.videoSourceComboBox.currentIndex = index
        widget.onVideoSourceChanged(index)
        pump(0.5)
        self.selectLesion(1)
        pump(1.5)
        widget.recordAudioButton.click()
        widget.recordVideoButton.click()
        pump(0.2)
        CHECKS.check(not recorder.recording and not recorder.capturing, "both stopped, the microphone closed")
        CHECKS.check(widget._recIndicator is None, "the REC sign is gone")
        CHECKS.check(waitForFiles(widget), "the files were saved")
        status = widget.recordingStatusLabel.text.replace("\u2060", "")  # the word joiners
        CHECKS.check("Saved" in status and "microphone/{}.wav".format(CASE_TWO) in status
                     and "video/{}.mp4".format(CASE_TWO) in status, "the panel says what was saved", status)

    def test_03_first_case_files(self):
        CHECKS.step("What the first case left behind")
        folder = os.path.join(self.root, recording.RECORDINGS_FOLDER)
        CHECKS.check(sorted(os.listdir(folder)) == [MICROPHONE, VIDEO],
                     "GTReview-recordings in the batch folder holds microphone/ and video/",
                     str(os.listdir(folder)))
        micPath, videoPath = self.exported(MICROPHONE, CASE_ONE), self.exported(VIDEO, CASE_ONE)
        for path in (micPath, videoPath):
            for name in (path, os.path.splitext(path)[0] + ".json"):
                CHECKS.check(os.path.isfile(name), "{} written".format(os.path.relpath(name, folder)))
        CHECKS.check(not self.leftovers(), "nothing else is left in the folders", str(self.leftovers()))

        log = self.log(MICROPHONE, CASE_ONE)
        events = log["events"]
        kinds = [e["type"] for e in events]
        for kind in ("recording_started", "case_state", "audio_started",
                     recordlog.LESION_SELECTED, recordlog.LESION_REVISITED, "lesion_deleted",
                     recordlog.LESION_CLEARED, recordlog.LESION_RENUMBERED, "undo", "audio_gap",
                     "recording_ended"):
            CHECKS.check(kind in kinds, "the log has a {} event".format(kind), str(sorted(set(kinds))))
        CHECKS.check(log["case_id"] == CASE_ONE and log["format"] == recordlog.FORMAT_NAME
                     and log["kind"] == MICROPHONE and log["file"] == CASE_ONE + ".wav",
                     "the log names its case, its kind and its file")
        selected = [e for e in events if e["type"] in recordlog.LESION_STATE_EVENTS and "seed_ijk" in e]
        CHECKS.check(all(blockOf(e["seed_ijk"]) is not None for e in selected),
                     "every logged seed voxel lies inside a lesion of the original mask",
                     str([e["seed_ijk"] for e in selected]))
        first = [e for e in selected if e["type"] == recordlog.LESION_SELECTED][0]
        CHECKS.check(first["number"] == 1 and blockOf(first["seed_ijk"]) == "A",
                     "lesion #1 is block A")
        deleted = [e for e in events if e["type"] == "lesion_deleted"][0]
        afterDelete = [e for e in selected if e["t"] > deleted["t"] and e["type"] == recordlog.LESION_SELECTED][0]
        CHECKS.check(afterDelete["number"] == 1 and blockOf(afterDelete["seed_ijk"]) == "B",
                     "after the delete, lesion #1 is block B: same number, another lesion")
        renumbered = [e for e in events if e["type"] == recordlog.LESION_RENUMBERED][0]
        CHECKS.check(renumbered["number"] == 2 and renumbered["previous_number"] == 1
                     and blockOf(renumbered["seed_ijk"]) == "B",
                     "Undo renumbered the selected block B from #1 to #2, and the log says so",
                     str(renumbered))
        states = [e["type"] for e in events if e["type"] in recordlog.LESION_STATE_EVENTS]
        CHECKS.check(states[-1] == recordlog.LESION_RENUMBERED,
                     "it ends with the case's last lesion state, not the next case's loading", str(states[-3:]))

        rate, narrated = wavfile.read_wav(micPath)
        audio = log["audio"]
        wall = log["duration"] - audio["offset"]
        CHECKS.check(abs(narrated.shape[0] / float(rate) - wall) < 0.4,
                     "the recording lasts as long as the clock ran, through both freezes",
                     "{:.2f} s of audio for {:.2f} s".format(narrated.shape[0] / float(rate), wall))
        # (on a slow machine the freeze outlasts its 2.5 s of sleep)
        CHECKS.check(len(audio["gaps"]) == 1 and abs(audio["gaps"][0]["seconds"] - self.dropped) < 0.35,
                     "only the dropped freeze was padded, by what was dropped",
                     "{} for {:.2f} s dropped".format(audio["gaps"], self.dropped))
        schedule = recordlog.announcement_schedule(events, audio["offset"])
        CHECKS.check(audio["announcements"] == len(schedule) >= 4,
                     "an announcement per lesion settled on", "{} / {}".format(audio["announcements"], len(schedule)))
        CHECKS.check(not any("lesion three" == text for _t, _c, text in schedule),
                     "the burst was not announced lesion by lesion", str([s[2] for s in schedule]))
        for at, _clips, text in schedule:
            residual = humResidual(narrated, rate, at)
            CHECKS.check(residual > 500, "'{}' is spoken in at {:.1f} s".format(text, at), "{:.0f}".format(residual))
        quiet = humResidual(narrated, rate, 0.2, 0.4)
        CHECKS.check(quiet < 50, "away from the announcements it is the microphone alone", "{:.1f}".format(quiet))
        cues = wavfile.read_cue_points(micPath)
        CHECKS.check(cues and len(cues) == len(recordlog.cue_marks(events, audio["offset"])),
                     "a cue point per lesion change", str(cues[:3]))

        videoLog = self.log(VIDEO, CASE_ONE)
        meta = videoLog["video"]
        CHECKS.check(videoLog["kind"] == VIDEO and videoLog["file"] == CASE_ONE + ".mp4" and meta["sound"] is True,
                     "the video's log says it has sound", str(meta))
        CHECKS.check("video_started" in [e["type"] for e in videoLog["events"]]
                     and recordlog.LESION_RENUMBERED in [e["type"] for e in videoLog["events"]],
                     "and has the lesion events too")
        CHECKS.check(abs(meta["duration"] - (videoLog["duration"] - meta["offset"])) < 0.5,
                     "the video lasts as long as the clock ran, freezes included",
                     "{:.2f} s of video for {:.2f} s".format(meta["duration"], videoLog["duration"] - meta["offset"]))
        streams = ffprobe(videoPath, "codec_type,codec_name")["streams"]
        # (a third, "data" stream is the QuickTime chapter track)
        CHECKS.check({"audio", "video"} <= {s["codec_type"] for s in streams},
                     "the MP4 has a sound track", str(streams))
        marks = chapters(videoPath)
        CHECKS.check(len(marks) >= 5 and any(m["tags"]["title"].startswith("Lesion #1") for m in marks),
                     "the MP4 has a chapter per lesion stretch", str([m["tags"]["title"] for m in marks]))
        soundRate, sound = soundOf(videoPath, os.path.join(self.root, "sound.wav"))
        CHECKS.check(abs(sound.shape[0] / float(soundRate) - meta["duration"]) < 0.5,
                     "the sound runs through the whole video",
                     "{:.2f} s for {:.2f} s".format(sound.shape[0] / float(soundRate), meta["duration"]))
        CHECKS.check(rms(sound[: int(1.0 * soundRate)]) > 0.5 * HUM_LEVEL / np.sqrt(2),
                     "it is the microphone (its hum)", "{:.0f}".format(rms(sound[: soundRate])))
        heard = 0
        shift = videoLog["audio"]["offset"] - meta["offset"]
        for at, _clips, text in recordlog.announcement_schedule(videoLog["events"], videoLog["audio"]["offset"]):
            if 0.0 <= at + shift < meta["duration"] - 1.0:
                residual = humResidual(sound, soundRate, at + shift)
                CHECKS.check(residual > 500, "'{}' can be heard in the video at {:.1f} s".format(text, at + shift),
                             "{:.0f}".format(residual))
                heard += 1
        CHECKS.check(heard >= 4, "every announcement is in the video's sound", str(heard))
        png = os.path.join(self.root, "frame.png")
        early = frameAt(videoPath, 0.5, png)
        CHECKS.check(not early.isNull() and early.width() == meta["size"][0], "a frame decodes")
        dot = qt.QColor(early.pixel(8 + recording.BANNER_HEIGHT // 4, recording.BANNER_HEIGHT // 2))
        CHECKS.check(dot.red() > 150 and dot.green() < 90, "the banner has its red dot",
                     str((dot.red(), dot.green(), dot.blue())))
        late = frameAt(videoPath, meta["duration"] - 1.0, os.path.join(self.root, "frame2.png"))
        early = early.convertToFormat(qt.QImage.Format_RGB32)
        late = late.convertToFormat(qt.QImage.Format_RGB32)
        CHECKS.check(bytes(early.constBits()) != bytes(late.constBits()),
                     "frames change as the reader moves")

    def test_04_second_case_files(self):
        CHECKS.step("The second case's takes")
        log = self.log(MICROPHONE, CASE_TWO)
        CHECKS.check(log["case_id"] == CASE_TWO, "they are the second case's")
        CHECKS.check(-0.3 < log["audio"]["offset"] <= 0.0,
                     "its audio starts with the take, give or take what was still in flight",
                     str(log["audio"]["offset"]))
        CHECKS.check(any(e["type"] == recordlog.LESION_SELECTED for e in log["events"]),
                     "its lesion selection is logged")
        videoLog = self.log(VIDEO, CASE_TWO)
        sources = [e.get("source") for e in videoLog["events"] if e["type"] == "video_source"]
        CHECKS.check(sources == ["t1c"], "the switch to the t1c view is in the video's log", str(sources))
        CHECKS.check(not any(e["type"] == "video_source" for e in log["events"]),
                     "and not in the microphone's")

    def test_05_microphone_stopped_before_the_video(self):
        CHECKS.step("The microphone switch goes off first: the video keeps its sound")
        widget = self.widget
        self.fake()
        self.asked[:] = []
        # the clicks are timed, not assumed: a slow machine can stall for
        # seconds between two of them
        widget.recordAudioButton.click()
        audioOn = time.monotonic()
        widget.recordVideoButton.click()
        videoOn = time.monotonic()
        pump(1.5)
        audioOff = time.monotonic()
        widget.recordAudioButton.click()
        pump(0.3)
        CHECKS.check(widget.recorder.capturing and widget.recorder.videoOn,
                     "the microphone stays open for the video")
        CHECKS.check(list(widget.recorder.takes) == [VIDEO], "but its own take is closed",
                     str(list(widget.recorder.takes)))
        pump(2.2)
        widget.recordVideoButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        CHECKS.check(self.asked == [(CASE_TWO, (MICROPHONE,), False), (CASE_TWO, (VIDEO,), False)],
                     "each switch asked before replacing what the case had", str(self.asked))
        microphone = self.seconds(MICROPHONE, CASE_TWO)
        CHECKS.check(abs(microphone - (audioOff - audioOn)) < 0.4,
                     "the microphone recording ends where its switch went off",
                     "{:.2f} s, the switch on for {:.2f} s".format(microphone, audioOff - audioOn))
        videoLog = self.log(VIDEO, CASE_TWO)
        meta = videoLog["video"]
        container = self.seconds(VIDEO, CASE_TWO)
        CHECKS.check(abs(container - meta["duration"]) < 0.4
                     and abs(meta["duration"] - (videoLog["duration"] - meta["offset"])) < 0.5
                     and videoLog["duration"] > (audioOff - videoOn) + 2.0,
                     "the MP4 runs as long as the video, which ran on after the microphone",
                     "{:.2f} s of MP4, {:.2f} s of frames from {:.2f} s into a {:.2f} s take; "
                     "the microphone went off {:.2f} s in".format(
                         container, meta["duration"], meta["offset"], videoLog["duration"],
                         audioOff - videoOn))
        soundRate, sound = soundOf(self.exported(VIDEO, CASE_TWO), os.path.join(self.root, "sound5.wav"))
        tail = sound[-int(1.0 * soundRate):]
        CHECKS.check(abs(sound.shape[0] / float(soundRate) - container) < 0.4
                     and rms(tail) > 0.5 * HUM_LEVEL / np.sqrt(2),
                     "and has sound to its end", "{:.2f} s, last second rms {:.0f}".format(
                         sound.shape[0] / float(soundRate), rms(tail)))

    def test_06_without_ffmpeg(self):
        CHECKS.step("No ffmpeg: no video, and ffmpeg is offered")
        widget = self.widget
        original = recording.findFfmpeg
        offered = []
        recording.findFfmpeg = lambda refresh=False: (None, False)
        slicer.util.confirmYesNoDisplay = lambda *a, **k: offered.append(a) and False
        try:
            widget._updateRecordingControls()
            CHECKS.check(widget.getFfmpegButton.visible, "the panel offers to get ffmpeg")
            CHECKS.check("needs ffmpeg" in widget.recordingStatusLabel.text, "and says the video needs it",
                         widget.recordingStatusLabel.text)
            widget.recordVideoButton.click()
            pump(0.3)
            CHECKS.check(len(offered) == 1, "Record video offers to download it")
            CHECKS.check(not widget.recorder.videoOn and not widget.recordVideoButton.checked
                         and not widget.recorder.takes and not widget.recorder.capturing,
                         "declined: nothing starts")
        finally:
            recording.findFfmpeg = original
            slicer.util.confirmYesNoDisplay = lambda *a, **k: True
            widget._updateRecordingControls()

    def test_07_a_microphone_that_will_not_open(self):
        CHECKS.step("A microphone that fails to open")
        widget = self.widget
        mic = self.fake()
        mic.failOnStart = True
        folder = os.path.join(self.root, recording.RECORDINGS_FOLDER, MICROPHONE)
        before = {name: os.stat(os.path.join(folder, name)).st_mtime_ns for name in os.listdir(folder)}
        self.warnings[:] = []
        widget.recordAudioButton.click()
        pump(0.2)
        CHECKS.check(not widget.recorder.audioOn and not widget.recordAudioButton.checked,
                     "the microphone switch went back off")
        CHECKS.check(not widget.recorder.takes, "no take is left open")
        CHECKS.check(any("could not be started" in w for w in self.warnings), "the reader is told",
                     str(self.warnings))
        after = {name: os.stat(os.path.join(folder, name)).st_mtime_ns for name in os.listdir(folder)}
        CHECKS.check(after == before, "the case's recording is untouched")

        self.warnings[:] = []
        widget.recordVideoButton.click()
        pump(1.5)
        take = widget.recorder.takes.get(VIDEO)
        CHECKS.check(widget.recorder.videoOn and take is not None and take.wav is None,
                     "the video records, without sound")
        CHECKS.check(any("without sound" in w for w in self.warnings), "and says so", str(self.warnings))
        widget.recordVideoButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        videoLog = self.log(VIDEO, CASE_TWO)
        streams = ffprobe(self.exported(VIDEO, CASE_TWO), "codec_type")["streams"]
        CHECKS.check(videoLog["video"]["sound"] is False and "audio" not in {s["codec_type"] for s in streams},
                     "its log and its file agree it has no sound", str(streams))

    def test_08_a_microphone_unplugged_while_recording(self):
        CHECKS.step("A microphone unplugged mid-recording is noticed and its audio kept")
        widget = self.widget
        mic = self.fake()
        self.warnings[:] = []
        widget.recordAudioButton.click()
        widget.recordVideoButton.click()
        pump(1.5)
        mic.unplugged = True                      # Qt keeps saying "active"
        pump(recording.QUIET_SECONDS + 1.5)
        CHECKS.check(not widget.recorder.audioOn and not widget.recordAudioButton.checked,
                     "the microphone recording stopped by itself")
        CHECKS.check(widget.recorder.videoOn, "the video goes on")
        CHECKS.check(any("stopped delivering audio" in w and "without sound" in w for w in self.warnings),
                     "the reader is told, of both", str(self.warnings))
        widget.recordVideoButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        microphone = self.seconds(MICROPHONE, CASE_TWO)
        CHECKS.check(1.2 < microphone < 2.5, "what the microphone recorded until then is kept",
                     "{:.2f} s".format(microphone))
        videoLog = self.log(VIDEO, CASE_TWO)
        CHECKS.check(videoLog["video"]["sound"] is True
                     and videoLog["audio"]["duration"] < videoLog["video"]["duration"] - 2.0,
                     "the video has the sound up to the unplugging", str(videoLog.get("audio")))

    def test_09_a_microphone_that_gives_silence(self):
        CHECKS.step("A microphone that delivers only zeros is reported")
        widget = self.widget
        self.fake(silent=True)
        self.warnings[:] = []
        widget.recordAudioButton.click()
        pump(recording.SILENCE_WARNING_SECONDS + 1.5)
        widget.recordAudioButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        CHECKS.check(any("pure silence" in w for w in self.warnings), "the reader is warned",
                     str(self.warnings))

    def test_10_high_dpi_grabs_fill_the_frame(self):
        CHECKS.step("A grab from a high-DPI screen fills the frame")
        image = qt.QImage(400, 300, qt.QImage.Format_RGB32)
        image.fill(qt.QColor(255, 0, 0))
        image.setDevicePixelRatio(2.0)            # what a Retina screen hands back
        canvas = recording.composeFrame(image, (400, 300 + recording.BANNER_HEIGHT), "REC", "case", "lesion")
        corner = qt.QColor(canvas.pixel(395, 300 + recording.BANNER_HEIGHT - 5))
        CHECKS.check(corner.red() > 200 and corner.green() < 50,
                     "the far corner of the picture is the picture, not black",
                     str((corner.red(), corner.green(), corner.blue())))

    def test_11_a_lesion_selected_before_recording_is_announced(self):
        CHECKS.step("Record pressed with a lesion selected: it is announced at the start")
        widget = self.widget
        self.selectLesion(1)
        pump(0.3)
        self.fake()
        widget.recordAudioButton.click()
        pump(2.5)
        widget.recordAudioButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        log = self.log(MICROPHONE, CASE_TWO)
        CHECKS.check(log["audio"]["announcements"] == 1, "one announcement", str(log["audio"]))
        path = self.exported(MICROPHONE, CASE_TWO)
        cues = wavfile.read_cue_points(path)
        CHECKS.check(cues and cues[0][0] < 0.05 * 16000 and cues[0][1].startswith("Lesion #1"),
                     "and a cue point at the very start", str(cues))
        rate, narrated = wavfile.read_wav(path)
        CHECKS.check(humResidual(narrated, rate, 0.0) > 500, "spoken into the first second")

    def test_12_a_disk_that_fails_mid_recording(self):
        CHECKS.step("A recording file cannot be written: the recording stops, what it had is kept")
        widget = self.widget
        self.fake()
        self.warnings[:] = []
        widget.recordAudioButton.click()
        widget.recordVideoButton.click()
        pump(1.2)
        take = widget.recorder.takes[MICROPHONE]

        def fullDisk(data):
            raise OSError(28, "No space left on device")

        take.wav.write = fullDisk
        pump(1.0)
        CHECKS.check(not widget.recorder.recording and not widget.recordAudioButton.checked
                     and not widget.recordVideoButton.checked and widget._recIndicator is None,
                     "both stopped (their files share the disk)")
        CHECKS.check(any("could not be written" in w for w in self.warnings), "the reader is told",
                     str(self.warnings))
        CHECKS.check(waitForFiles(widget), "saved")
        log = self.log(MICROPHONE, CASE_TWO)
        CHECKS.check("audio_write_failed" in [e["type"] for e in log["events"]],
                     "the failure is in the event log")
        CHECKS.check(1.0 < self.seconds(MICROPHONE, CASE_TWO) < 1.8 and self.seconds(VIDEO, CASE_TWO) > 0.5,
                     "what reached the disk and the video are kept",
                     "{:.2f} s / {:.2f} s".format(self.seconds(MICROPHONE, CASE_TWO), self.seconds(VIDEO, CASE_TWO)))

    def test_13_frames_owed_for_a_freeze_show_the_old_picture(self):
        CHECKS.step("After a freeze, the frames it owes repeat the picture from before it")
        widget = self.widget
        self.fake()
        widget.recordVideoButton.click()
        pump(1.0)
        writer = widget.recorder.takes[VIDEO].writer
        pushes = []
        original = writer.push

        def spy(frame, repeat=1):
            pushes.append((frame, repeat))
            original(frame, repeat)

        writer.push = spy
        pump(0.5)
        before = pushes[-1][0]
        time.sleep(1.5)                           # Slicer frozen
        pump(0.3)
        widget.recordVideoButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        burst = [i for i, (_frame, repeat) in enumerate(pushes) if repeat >= 10]
        CHECKS.check(len(burst) == 1, "one burst of repeats after the freeze", str([r for _f, r in pushes]))
        i = burst[0]
        CHECKS.check(pushes[i][0] is before, "the burst is the picture from before the freeze")
        CHECKS.check(i + 1 < len(pushes) and pushes[i + 1][1] == 1 and pushes[i + 1][0] is not before,
                     "followed by one new picture")

    def test_14_one_recording_per_case(self):
        CHECKS.step("One recording per case: asked before replacing, the newest take wins")
        widget = self.widget
        path = self.exported(MICROPHONE, CASE_TWO)
        stamp = os.stat(path).st_mtime_ns
        self.fake()
        self.asked[:], self.answers[:] = [], [False]
        widget.recordAudioButton.click()
        pump(0.3)
        CHECKS.check(self.asked == [(CASE_TWO, (MICROPHONE,), False)], "recording a recorded case asks first",
                     str(self.asked))
        CHECKS.check(not widget.recorder.audioOn and not widget.recordAudioButton.checked
                     and not widget.recorder.capturing, "\"Keep it\": nothing starts")
        CHECKS.check(os.stat(path).st_mtime_ns == stamp, "and the recording stays as it was")

        # a case without a recording; its first take is held back from being
        # saved while the second one starts
        os.remove(path)
        os.remove(os.path.splitext(path)[0] + ".json")
        gate = threading.Event()
        original = recording.finishTake

        def heldBack(take):
            gate.wait(60)
            return original(take)

        recording.finishTake = heldBack
        try:
            self.asked[:], self.answers[:] = [], [True]
            widget.recordAudioButton.click()
            pump(1.0)
            widget.recordAudioButton.click()      # first take: about a second
            CHECKS.check(not self.asked, "a case without a recording is not asked about")
            widget.recordAudioButton.click()
            CHECKS.check(self.asked == [(CASE_TWO, (MICROPHONE,), False)],
                         "a take still being saved counts as the case's recording", str(self.asked))
            pump(2.5)
            widget.recordAudioButton.click()      # second take: two and a half
            gate.set()
            CHECKS.check(waitForFiles(widget), "both saved")
        finally:
            recording.finishTake = original
            gate.set()
        seconds = self.seconds(MICROPHONE, CASE_TWO)
        CHECKS.check(2.2 < seconds < 3.2, "the newest take is the case's recording", "{:.2f} s".format(seconds))
        CHECKS.check(not self.leftovers(), "and nothing else is left", str(self.leftovers()))
        folder = os.path.join(self.root, recording.RECORDINGS_FOLDER)
        names = sorted(os.listdir(os.path.join(folder, MICROPHONE))) + sorted(os.listdir(os.path.join(folder, VIDEO)))
        CHECKS.check(names == sorted(c + e for c in (CASE_ONE, CASE_TWO) for e in (".json", ".wav"))
                     + sorted(c + e for c in (CASE_ONE, CASE_TWO) for e in (".json", ".mp4")),
                     "one recording of each kind per case", str(names))

    def test_15_moving_on_to_a_recorded_case(self):
        CHECKS.step("Moving on to a recorded case while recording asks; a reload goes on")
        widget = self.widget
        widget._confirmDiscardIfNeeded = lambda *args, **kwargs: True
        caseOne = self.exported(MICROPHONE, CASE_ONE)
        stamp = os.stat(caseOne).st_mtime_ns
        self.fake()
        started = time.monotonic()
        widget.recordAudioButton.click()          # CASE_TWO's recording is replaced
        pump(1.0)
        self.asked[:], self.answers[:] = [], [False]
        widget.onPreviousCase()                   # the take ends once the case is up
        switched = time.monotonic() - started
        pump(0.8)
        CHECKS.check(widget.currentCase().case_id == CASE_ONE, "the first case is open")
        CHECKS.check(self.asked == [(CASE_ONE, (MICROPHONE,), True)], "it has a recording, so it was asked",
                     str(self.asked))
        CHECKS.check(not widget.recorder.recording and not widget.recordAudioButton.checked,
                     "\"Stop recording\" stopped it")
        CHECKS.check(waitForFiles(widget), "saved")
        CHECKS.check(os.stat(caseOne).st_mtime_ns == stamp, "the recorded case keeps its recording")
        seconds = self.seconds(MICROPHONE, CASE_TWO)
        CHECKS.check(0.8 < seconds < switched + 0.3, "the case before was saved, up to the switch",
                     "{:.2f} s of {:.2f} s".format(seconds, switched))

        self.answers[:] = [True, True]
        started = time.monotonic()
        widget.recordAudioButton.click()          # replaces CASE_ONE's
        pump(0.8)
        take = widget.recorder.takes[MICROPHONE]
        widget.onReset()                          # the same case again, from disk
        pump(0.8)
        CHECKS.check(widget.recorder.takes.get(MICROPHONE) is take, "a reload of the case goes on in the same take")
        widget.onNextCase()                       # replaces CASE_TWO's
        switched = time.monotonic() - started
        pump(0.8)
        CHECKS.check(widget.recorder.audioOn and widget.recorder.takes[MICROPHONE].name == CASE_TWO,
                     "\"Replace\" records the next case", str(self.asked))
        pump(1.0)
        widget.recordAudioButton.click()
        CHECKS.check(waitForFiles(widget), "saved")
        log = self.log(MICROPHONE, CASE_ONE)
        CHECKS.check("case_reloaded" in [e["type"] for e in log["events"]], "the reload is in the log")
        seconds = self.seconds(MICROPHONE, CASE_ONE)
        CHECKS.check(1.2 < seconds < switched + 0.3, "the first case got the new take, up to the switch",
                     "{:.2f} s of {:.2f} s".format(seconds, switched))

    def test_16_open_folder(self):
        CHECKS.step("Open folder shows the batch's recordings, without Slicer's library paths")
        widget = self.widget
        calls = []

        class Started(object):
            def __init__(self, args, **kwargs):
                calls.append((list(args), kwargs))

            def wait(self, timeout=None):
                return 0

        original, originalUrl = recording.subprocess.Popen, recording._openUrl
        opened = []
        recording.subprocess.Popen = Started
        recording._openUrl = lambda path: opened.append(path) or True
        try:
            widget.onOpenRecordingsFolder()
        finally:
            recording.subprocess.Popen, recording._openUrl = original, originalUrl
        folder = os.path.join(self.root, recording.RECORDINGS_FOLDER)
        if not sys.platform.startswith("linux"):
            CHECKS.check([os.path.normcase(p) for p in opened] == [os.path.normcase(folder)],
                         "Explorer or the Finder opens the recordings folder", str(opened))
            return
        CHECKS.check(len(calls) == 1 and os.path.basename(calls[0][0][0]) == "xdg-open"
                     and calls[0][0][1] == folder, "xdg-open opens GTReview-recordings", str(calls))
        env = calls[0][1].get("env") or {}
        home = os.path.realpath(slicer.app.slicerHome)
        paths = [os.path.realpath(p) for key in ("LD_LIBRARY_PATH", "QT_PLUGIN_PATH")
                 for p in env.get(key, "").split(os.pathsep) if p]
        CHECKS.check(env and not any(p.startswith(home) for p in paths) and "PYTHONHOME" not in env,
                     "with the environment Slicer was started in",
                     str({k: env.get(k) for k in ("LD_LIBRARY_PATH", "PYTHONHOME", "QT_PLUGIN_PATH")}))

    def test_17_saved_while_slicer_idles(self):
        CHECKS.step("A take is saved within seconds while Slicer idles in its event loop")
        widget = self.widget
        self.fake()
        widget.recordAudioButton.click()
        pump(3.0)
        recording._VOICE_CACHE.clear()            # the clips are loaded again: some work
        widget.recordAudioButton.click()
        # wait in Qt's loop, as Slicer does when nothing happens -- not in
        # pump(), whose sleeps hand the GIL to the saving thread
        loop = qt.QEventLoop()
        started = time.monotonic()
        poll = qt.QTimer()
        poll.setInterval(500)
        poll.connect("timeout()", lambda: loop.quit() if not widget.recorder.busyFinishing()
                     or time.monotonic() - started > 90 else None)
        poll.start()
        loop.exec_()
        poll.stop()
        took = time.monotonic() - started
        CHECKS.check(widget.recorder.busyFinishing() == 0 and took < 6.0, "saved in {:.1f} s".format(took))
        CHECKS.check(not self.leftovers(), "nothing else is left in the folders", str(self.leftovers()))

    def test_18_a_slicer_that_macos_keeps_from_the_microphone(self):
        CHECKS.step("A Slicer macOS keeps from the microphone: an explanation instead of silence")
        widget = self.widget
        if sys.platform == "darwin":
            signing = recording.macSigningProblem()
            if os.environ.get("GTREVIEW_EXPECT_MAC_MICROPHONE", "blocked") == "blocked":
                CHECKS.check(bool(signing), "the official Slicer.app lacks what macOS asks for the microphone",
                             str(signing))
            else:
                CHECKS.check(signing is None, "this Slicer.app carries what macOS asks for the microphone",
                             str(signing))
            shell = recording._startedFromAShell()
            CHECKS.check(self.macProblem == (None if shell else signing),
                         "started from {}, it is {}".format("a shell" if shell else "launchd",
                                                            "let try" if shell else "judged by its own app"),
                         str(self.macProblem))
        else:
            CHECKS.check(self.macProblem is None, "only macOS keeps Slicer from the microphone")
        shown = []
        widget._macMicrophoneHelp = lambda: shown.append(True)
        recording._MAC_MICROPHONE["problem"] = "this Slicer's signature does not carry the microphone entitlement"
        self.fake()
        self.warnings[:] = []
        try:
            widget.recordAudioButton.click()
            pump(0.3)
            CHECKS.check(shown and not widget.recorder.audioOn and not widget.recordAudioButton.checked
                         and not widget.recorder.capturing, "Record microphone explains, and records nothing")
            widget.testMicrophoneButton.click()
            pump(0.3)
            CHECKS.check(len(shown) == 2 and not widget.recorder.testing, "and so does Test")
            widget.recordVideoButton.click()
            pump(recording.SILENCE_WARNING_SECONDS + 1.5)
            take = widget.recorder.takes.get(VIDEO)
            CHECKS.check(widget.recorder.videoOn and take is not None and take.wav is None
                         and not widget.recorder.capturing,
                         "Record video records the views without sound, the microphone left closed")
            CHECKS.check(not self.warnings, "and nobody is warned about silence", str(self.warnings))
            widget.recordVideoButton.click()
            CHECKS.check(waitForFiles(widget), "saved")
            CHECKS.check(self.log(VIDEO, CASE_ONE if widget.currentCase().case_id == CASE_ONE else CASE_TWO)
                         ["video"]["sound"] is False, "its log says it has no sound")
        finally:
            recording._MAC_MICROPHONE["problem"] = None
            del widget._macMicrophoneHelp


def main():
    print("=" * 72)
    print("GTReview recording integration test")
    print("  Slicer  : {} ({}, Python {})".format(slicer.app.applicationVersion, platform.platform(),
                                                platform.python_version()))
    print("  machine : {}".format(platform.machine()))
    ffmpeg = recording.findFfmpeg()[0]
    print("  ffmpeg  : {}".format(ffmpeg or "none"))
    try:
        devices = ["{} [{}]".format(d["label"], d["realm"]) for d in recording.inputDevices()]
    except Exception as error:  # noqa: BLE001
        devices = ["listing failed: {}".format(error)]
    print("  inputs  : {}".format(", ".join(devices) or "none"))
    if sys.platform == "darwin":
        print("  mac mic : {}".format(recording.macMicrophoneProblem() or "Slicer may use the microphone"))
        print("  app     : {}".format(recording.macAppBundle()))
    print("=" * 72)
    if not ffmpeg or not shutil.which("ffprobe"):
        print("RESULT: FAIL -- the test needs ffmpeg and ffprobe")
        sys.stdout.flush()
        slicer.util.exit(1)
        return
    suite = unittest.TestLoader().loadTestsFromTestCase(RecordingIntegrationTest)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=0, failfast=True).run(suite)
    print("\n" + "=" * 72)
    print("SUMMARY: {} checks passed, {} failed, {} steps errored".format(
        len(CHECKS.passed), len(CHECKS.failed), len(result.errors)))
    for description, detail in CHECKS.failed:
        print("  FAILED: {}{}".format(description, ("  --  " + str(detail)) if detail else ""))
    for test, message in result.errors:
        print("  ERROR : {}\n{}".format(test, message))
    ok = result.wasSuccessful() and not CHECKS.failed and bool(CHECKS.passed)
    print("RESULT: {}".format("PASS" if ok else "FAIL"))
    print("=" * 72)
    sys.stdout.flush()
    slicer.util.exit(0 if ok else 1)


main()
