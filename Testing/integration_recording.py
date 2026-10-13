"""Integration test of GTReview's microphone and video recording, inside Slicer.

Run it by hand (a display is needed; ``xvfb-run -a`` provides one)::

    Slicer --no-splash \
        --additional-module-path <repo>/GTReview \
        --python-script <repo>/Testing/integration_recording.py

No microphone is opened: a stand-in (:class:`FakeMicrophone`) delivers a hum at
the wall clock's pace from a Qt timer, the way Qt's own audio backends deliver
on the main thread.  It can hold audio through a freeze and deliver it late (as
PulseAudio does) or drop it (as CoreAudio and ALSA do).  The video is the real
thing, rendered from the slice views, encoded by ffmpeg when one is found and
by VTK's Theora writer when not.

On two synthetic cases (numpy + SimpleITK, never patient data) it

1. loads the batch and checks the Recording section is ready: both switches
   enabled, a microphone list, the views to choose from;
2. records the microphone and the view area together while lesions are
   selected, re-clicked, deleted and brought back by Undo (which renumbers the
   selected one), while the main thread freezes twice -- once with the audio
   held, once with it dropped -- and across a switch to the next case, where
   the video is pointed at one slice view;
3. checks what the first case left: the microphone WAV with a cue point per
   lesion change, the narrated WAV with an announcement per lesion settled on,
   subtitles, an Audacity label track, the event log with a seed voxel inside
   the lesion each event names (also after the renumbering), audio and video
   that last as long as the wall clock did through both freezes, and an MP4
   with the narrated audio, a chapter per lesion stretch and the red-dot
   banner on frames that change as the reader moves;
4. checks the second case's segment and the source switch in its log;
5. records without ffmpeg and gets a Theora .ogv;
6. starts a microphone that fails to open and checks nothing is left behind,
   and unplugs one mid-recording and checks the recording stops with a warning
   and keeps what it had;
7. records a microphone that delivers only zeros and checks the reader is
   warned.

Not named ``test_*`` so the unit-test discovery leaves it alone.  Set
GTREVIEW_KEEP_RECORDINGS=1 to keep the temporary batch and its recordings.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
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
            self._produced += owed - kept
            owed = kept
            self.dropNextFreeze = False
        t = np.arange(self._produced, self._produced + owed) / float(self.rate)
        level = 0.0 if self.silent else 3000.0
        samples = (np.sin(2 * np.pi * 220.0 * t) * level).astype("<i2")
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
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    out = subprocess.run([exe, "-v", "error", "-show_entries", "{}={}".format(section, entries),
                          "-of", "json", path], capture_output=True, text=True,
                         env=recording._environment())
    return json.loads(out.stdout or "{}")


def chapters(path):
    exe = shutil.which("ffprobe")
    out = subprocess.run([exe, "-v", "error", "-show_chapters", "-of", "json", path],
                         capture_output=True, text=True, env=recording._environment())
    return json.loads(out.stdout or "{}").get("chapters", [])


def frameAt(video, seconds, png):
    ffmpeg = recording.findFfmpeg()[0]
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", str(seconds), "-i", video, "-frames:v", "1", png],
                   env=recording._environment(), check=True)
    return qt.QImage(png)


def waitForFiles(widget, timeout=120.0):
    deadline = time.time() + timeout
    while widget.recorder.busyFinishing() and time.time() < deadline:
        pump(0.2)
    return widget.recorder.busyFinishing() == 0


class RecordingIntegrationTest(unittest.TestCase):
    root = None
    widget = None
    mask = None
    started = None
    caseOneStem = None

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
        cls.warnings = []
        cls._warningDisplay = slicer.util.warningDisplay
        cls._confirm = slicer.util.confirmYesNoDisplay
        cls._error = slicer.util.errorDisplay
        slicer.util.warningDisplay = lambda text, *a, **k: cls.warnings.append(str(text))
        slicer.util.errorDisplay = lambda text, *a, **k: cls.warnings.append(str(text))
        slicer.util.confirmYesNoDisplay = lambda *a, **k: True

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

    def fake(self, **kwargs):
        widget = self.widget
        mic = FakeMicrophone(widget.recorder._onAudio, widget._onRecorderLevel, **kwargs)
        widget.recorder.mic = mic
        return mic

    def selectLesion(self, number):
        table = self.widget.lesionTable
        for row in range(table.rowCount):
            if table.item(row, 0).data(qt.Qt.UserRole) == number:
                table.selectRow(row)
                pump(0.05)
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
        CHECKS.check(recorder.audioOn and recorder.videoOn, "both are recording")
        CHECKS.check(widget.recordAudioButton.text == "Stop microphone", "the button says how to stop")
        CHECKS.check(widget._recIndicator is not None and widget._recIndicator.visible,
                     "a REC sign is in the status bar")
        segment = recorder.segment
        self.__class__.caseOneStem = segment.path("")
        CHECKS.check(os.path.isfile(segment.path("_mic.wav")), "the microphone file is being written")
        self.__class__.started = time.monotonic()
        pump(1.2)
        self.selectLesion(1)                      # A
        pump(1.5)
        for number in (2, 3, 2):                  # a burst: only the last is announced
            self.selectLesion(number)
            pump(0.1)
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
        self.__class__.caseOneEnd = time.monotonic()

        # the delete and the undo left the case "unsaved": no Save / Discard prompt
        widget._confirmDiscardIfNeeded = lambda *args, **kwargs: True
        widget.onNextCase()
        pump(0.8)
        CHECKS.check(widget.currentCase().case_id == CASE_TWO, "the second case is open")
        CHECKS.check(recorder.segment is not None and CASE_TWO in recorder.segment.stem,
                     "the recording carried on into the second case's own files")
        CHECKS.check(widget.layoutComboBox.itemData(widget.layoutComboBox.currentIndex)
                     == gtreview.SEQUENCES_LAYOUT_ID, "the second case shows its sequences side by side")
        index = widget.videoSourceComboBox.findData("t1c")
        CHECKS.check(index > 0, "the t1c view can be picked for the video")
        widget.videoSourceComboBox.currentIndex = index
        widget.onVideoSourceChanged(index)
        pump(0.5)
        self.selectLesion(1)
        pump(1.5)
        self.__class__.caseTwoStem = recorder.segment.path("")
        widget.recordAudioButton.click()
        widget.recordVideoButton.click()
        pump(0.2)
        CHECKS.check(not recorder.recording, "both stopped")
        CHECKS.check(widget._recIndicator is None, "the REC sign is gone")
        CHECKS.check(waitForFiles(widget), "the files were finished")

    def test_03_first_case_files(self):
        CHECKS.step("What the first case left behind")
        stem = self.caseOneStem
        folder = os.path.dirname(stem)
        CHECKS.check(folder == os.path.join(self.root, recording.RECORDINGS_FOLDER, CASE_ONE),
                     "recordings sit in GTReview-recordings/<case> in the batch folder", folder)
        ffmpeg = recording.findFfmpeg()[0]
        video = stem + (".mp4" if ffmpeg else ".ogv")
        for suffix in ("_mic.wav", ".wav", ".srt", "_labels.txt", ".json"):
            CHECKS.check(os.path.isfile(stem + suffix), "{} written".format(os.path.basename(stem + suffix)))
        CHECKS.check(os.path.isfile(video), "{} written".format(os.path.basename(video)))
        CHECKS.check(not os.path.exists(stem + "_video.mp4"), "the unmuxed video was cleaned up")

        with open(stem + ".json", encoding="utf-8") as handle:
            log = json.load(handle)
        events = log["events"]
        kinds = [e["type"] for e in events]
        for kind in ("recording_started", "case_state", "audio_started", "video_started",
                     recordlog.LESION_SELECTED, recordlog.LESION_REVISITED, "lesion_deleted",
                     recordlog.LESION_CLEARED, recordlog.LESION_RENUMBERED, "undo", "audio_gap",
                     "recording_ended"):
            CHECKS.check(kind in kinds, "the log has a {} event".format(kind), str(sorted(set(kinds))))
        CHECKS.check(log["case_id"] == CASE_ONE and log["format"] == recordlog.FORMAT_NAME,
                     "the log names its case and format")
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

        rate, mic = wavfile.read_wav(stem + "_mic.wav")
        _rate, narrated = wavfile.read_wav(stem + ".wav")
        audio = log["audio"]
        wall = log["duration"] - audio["offset"]
        CHECKS.check(abs(mic.shape[0] / float(rate) - wall) < 0.4,
                     "the microphone track lasts as long as the clock ran, through both freezes",
                     "{:.2f} s of audio for {:.2f} s".format(mic.shape[0] / float(rate), wall))
        CHECKS.check(len(audio["gaps"]) == 1 and 1.5 < audio["gaps"][0]["seconds"] < 2.6,
                     "only the dropped freeze was padded with silence", str(audio["gaps"]))
        CHECKS.check(narrated.shape == mic.shape, "the narrated copy is as long as the microphone")
        schedule = recordlog.announcement_schedule(events, audio["offset"])
        CHECKS.check(audio["announcements"] == len(schedule) >= 4,
                     "an announcement per lesion settled on", "{} / {}".format(audio["announcements"], len(schedule)))
        CHECKS.check(not any("lesion three" == text for _t, _c, text in schedule),
                     "the burst was not announced lesion by lesion", str([s[2] for s in schedule]))
        for at, _clips, text in schedule:
            start = int(at * rate)
            window = slice(start, start + int(0.8 * rate))
            CHECKS.check(not np.array_equal(narrated[window], mic[window]),
                         "'{}' is spoken in at {:.1f} s".format(text, at))
        quiet = slice(int(0.2 * rate), int(0.6 * rate))
        CHECKS.check(np.array_equal(narrated[quiet], mic[quiet]),
                     "away from the announcements the narrated copy is the microphone")
        cues = wavfile.read_cue_points(stem + "_mic.wav")
        CHECKS.check(len(cues) == len(recordlog.cue_marks(events, audio["offset"])) and cues,
                     "a cue point per lesion change", str(cues[:3]))
        with open(stem + ".srt", encoding="utf-8") as handle:
            srt = handle.read()
        CHECKS.check("Lesion #2 of 3" in srt and "No lesion selected" in srt and " --> " in srt,
                     "the subtitles name the lesions")
        with open(stem + "_labels.txt", encoding="utf-8") as handle:
            labels = [line.split("\t") for line in handle.read().splitlines()]
        CHECKS.check(labels and all(len(row) == 3 for row in labels), "the Audacity label track parses")

        videoLog = log["video"]
        CHECKS.check(abs(videoLog["duration"] - (log["duration"] - videoLog["offset"])) < 0.5,
                     "the video lasts as long as the clock ran, freezes included",
                     "{:.2f} s of video for {:.2f} s".format(videoLog["duration"], log["duration"] - videoLog["offset"]))
        if ffmpeg and shutil.which("ffprobe"):
            streams = ffprobe(video, "codec_type,codec_name")["streams"]
            # (a third, "data" stream is the QuickTime chapter track)
            CHECKS.check({"audio", "video"} <= {s["codec_type"] for s in streams},
                         "the MP4 carries the narrated audio", str(streams))
            marks = chapters(video)
            CHECKS.check(len(marks) >= 5 and any(m["tags"]["title"].startswith("Lesion #1") for m in marks),
                         "the MP4 has a chapter per lesion stretch", str([m["tags"]["title"] for m in marks]))
            png = os.path.join(self.root, "frame.png")
            early = frameAt(video, 0.5, png)
            CHECKS.check(not early.isNull() and early.width() == videoLog["size"][0], "a frame decodes")
            dot = qt.QColor(early.pixel(8 + recording.BANNER_HEIGHT // 4, recording.BANNER_HEIGHT // 2))
            CHECKS.check(dot.red() > 150 and dot.green() < 90, "the banner has its red dot",
                         str((dot.red(), dot.green(), dot.blue())))
            late = frameAt(video, videoLog["duration"] - 1.0, os.path.join(self.root, "frame2.png"))
            early = early.convertToFormat(qt.QImage.Format_RGB32)
            late = late.convertToFormat(qt.QImage.Format_RGB32)
            CHECKS.check(bytes(early.constBits()) != bytes(late.constBits()),
                         "frames change as the reader moves")

    def test_04_second_case_files(self):
        CHECKS.step("The second case's segment")
        stem = self.caseTwoStem
        with open(stem + ".json", encoding="utf-8") as handle:
            log = json.load(handle)
        CHECKS.check(log["case_id"] == CASE_TWO, "it is the second case's")
        sources = [e.get("source") for e in log["events"] if e["type"] == "video_source"]
        CHECKS.check(sources == ["t1c"], "the switch to the t1c view is logged", str(sources))
        CHECKS.check(log["audio"]["offset"] == 0.0, "its audio starts with the segment")
        CHECKS.check(any(e["type"] == recordlog.LESION_SELECTED for e in log["events"]),
                     "its lesion selection is logged")

    def test_04b_microphone_stopped_before_the_video(self):
        CHECKS.step("The microphone stops first: the MP4 keeps the whole video")
        widget = self.widget
        if not recording.findFfmpeg()[0] or not shutil.which("ffprobe"):
            print("  (no ffmpeg: skipped)")
            return
        self.fake()
        widget.recordAudioButton.click()
        widget.recordVideoButton.click()
        pump(1.5)
        stem = widget.recorder.segment.path("")
        widget.recordAudioButton.click()
        pump(2.5)
        widget.recordVideoButton.click()
        CHECKS.check(waitForFiles(widget), "finished")
        with open(stem + ".json", encoding="utf-8") as handle:
            log = json.load(handle)
        container = float(ffprobe(stem + ".mp4", "duration", section="format")["format"]["duration"])
        CHECKS.check(abs(container - log["video"]["duration"]) < 0.4,
                     "the MP4 runs as long as the video",
                     "{:.2f} s for {:.2f} s of video".format(container, log["video"]["duration"]))
        CHECKS.check(container > log["audio"]["duration"] + 1.5,
                     "and longer than the microphone, which stopped first",
                     "{:.2f} s vs {:.2f} s of audio".format(container, log["audio"]["duration"]))

    def test_05_without_ffmpeg(self):
        CHECKS.step("No ffmpeg: VTK's Theora writer")
        widget = self.widget
        original = recording.findFfmpeg
        recording.findFfmpeg = lambda refresh=False: (None, False)
        try:
            widget._updateRecordingControls()
            CHECKS.check(widget.getFfmpegButton.visible, "the panel offers to get ffmpeg")
            widget.recordVideoButton.click()
            pump(2.5)
            stem = widget.recorder.segment.path("")
            widget.recordVideoButton.click()
            CHECKS.check(waitForFiles(widget), "finished")
        finally:
            recording.findFfmpeg = original
            widget._updateRecordingControls()
        CHECKS.check(os.path.isfile(stem + ".ogv"), "an .ogv was written")
        with open(stem + ".json", encoding="utf-8") as handle:
            log = json.load(handle)
        CHECKS.check(log["video"]["encoder"] == "ogv" and log["video"]["fps"] == recording.THEORA_FPS,
                     "at the Theora frame rate", str(log["video"]))
        if shutil.which("ffprobe"):
            streams = ffprobe(stem + ".ogv", "codec_name")["streams"]
            CHECKS.check(streams and streams[0]["codec_name"] == "theora", "it is Theora", str(streams))

    def test_06_a_microphone_that_will_not_open(self):
        CHECKS.step("A microphone that fails to open leaves nothing behind")
        widget = self.widget
        mic = self.fake()
        mic.failOnStart = True
        folder = os.path.join(self.root, recording.RECORDINGS_FOLDER, CASE_TWO)
        before = sorted(os.listdir(folder))
        self.warnings[:] = []
        widget.recordAudioButton.click()
        pump(0.2)
        CHECKS.check(not widget.recorder.audioOn and not widget.recordAudioButton.checked,
                     "the switch went back off")
        CHECKS.check(widget.recorder.segment is None, "no segment is left open")
        CHECKS.check(any("could not be started" in w for w in self.warnings), "the reader is told",
                     str(self.warnings))
        CHECKS.check(sorted(os.listdir(folder)) == before, "no file was written")

    def test_06b_a_microphone_unplugged_while_recording(self):
        CHECKS.step("A microphone unplugged mid-recording is noticed and its audio kept")
        widget = self.widget
        mic = self.fake()
        self.warnings[:] = []
        widget.recordAudioButton.click()
        pump(1.5)
        stem = widget.recorder.segment.path("")
        mic.unplugged = True                      # Qt keeps saying "active"
        pump(recording.QUIET_SECONDS + 1.5)
        CHECKS.check(not widget.recorder.audioOn and not widget.recordAudioButton.checked,
                     "the microphone recording stopped by itself")
        CHECKS.check(any("stopped delivering audio" in w for w in self.warnings),
                     "the reader is told", str(self.warnings))
        CHECKS.check(waitForFiles(widget), "finished")
        rate, kept = wavfile.read_wav(stem + "_mic.wav")
        CHECKS.check(1.2 < kept.shape[0] / float(rate) < 2.5, "what it recorded until then is kept",
                     "{:.2f} s".format(kept.shape[0] / float(rate)))

    def test_07_a_microphone_that_gives_silence(self):
        CHECKS.step("A microphone that delivers only zeros is reported")
        widget = self.widget
        self.fake(silent=True)
        self.warnings[:] = []
        widget.recordAudioButton.click()
        pump(recording.SILENCE_WARNING_SECONDS + 1.5)
        widget.recordAudioButton.click()
        CHECKS.check(waitForFiles(widget), "finished")
        CHECKS.check(any("pure silence" in w for w in self.warnings), "the reader is warned",
                     str(self.warnings))


def main():
    print("=" * 72)
    print("GTReview recording integration test")
    print("  Slicer  : {}".format(slicer.app.applicationVersion))
    print("  ffmpeg  : {}".format(recording.findFfmpeg()[0] or "none (Theora)"))
    print("=" * 72)
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
