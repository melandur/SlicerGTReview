"""Record the reader's voice and the axial views during a review -- the Slicer half.

Two switches in the panel, and one recording per case for each, in the batch
folder's ``GTReview-recordings``:

``microphone/<case_id>.wav``
    the microphone, narrated: a chime and "lesion three" spoken in wherever the
    reader moved to another lesion, the reader's voice turned down under it,
    and a WAV cue point at every lesion change
``video/<case_id>.mp4``
    the view area, or one slice view, under a banner naming the case and the
    active lesion, with the narrated microphone as its sound and one chapter
    per lesion -- recording the video records the microphone with it
``<case_id>.json``, next to each
    every event with the lesion's seed voxel, which survives renumbering --
    the record a later re-annotation can follow exactly

Recording a case again replaces its recording of that kind, once the panel has
asked (``confirmReplace``); the old one stays until the new one is saved.
While a take runs, its files are ``<case_id>_<YYYYMMDD-HHMMSS>.recording.*``
in the same folder, written as they come in, so a crash leaves them playable.

Capture never leaves Slicer's process.  The video is rendered from Slicer's own
views (no OS screen-recording permission, any monitor, a covered window); the
microphone goes through Qt Multimedia, which Slicer ships.  Qt moves audio on
the main thread, so while Slicer is busy the sound system holds the samples;
what it cannot hold is padded with silence (see :class:`recordlog.AudioClock`)
so the rest stays in step.

Encoding happens off the main thread: frames are piped to an ffmpeg process by
a writer thread, and a finished take is turned into its file by a thread of
its own, one take after the other.  Nothing in those threads touches Qt, VTK or
the scene.  Slicer's main thread keeps Python's GIL while it idles in Qt, so
while a take is saved a timer hands it over every few milliseconds, as Slicer's
own SimpleFilters module does (measured: without it, the saving thread ran at
3 % of its speed, and a half-minute recording took two minutes to appear).
"""

import datetime
import getpass
import logging
import os
import plistlib
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import ctk
import qt
import slicer

import numpy as np

from . import maskio, recordlog, wavfile

#: the spoken clips (whole phrases and single words)
VOICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "GTReviewVoice"
)
#: run once per Slicer.app so that macOS lets it use the microphone
MAC_MICROPHONE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "GTReviewMacMicrophone.sh"
)
#: the folder of the batch directory recordings go into
RECORDINGS_FOLDER = "GTReview-recordings"
#: the two kinds of recording: the folder each has in RECORDINGS_FOLDER, and
#: the file every case gets in it
MICROPHONE = "microphone"
VIDEO = "video"
MEDIA_SUFFIX = {MICROPHONE: ".wav", VIDEO: ".mp4"}
#: the files of a take while it runs (``.recording``) and while it is saved
WORKING_SUFFIXES = (
    ".recording.wav", ".recording.mp4", ".narrated.wav",
    ".saving.wav", ".saving.mp4", ".saving.json", "_chapters.txt",
)
#: what the microphone is asked for; the device may offer something close
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
#: how much audio WinMM is asked to hold while Slicer is busy (Windows only).
#: Qt splits it into five blocks and hands over only full ones, so it is also
#: what a stop can lose at the very end (a fifth of it) -- five seconds bridge
#: a slow case load and keep that to a second.  PulseAudio / PipeWire hold
#: minutes on their own, and Qt passes this size to them as the size of every
#: delivery: asked there, start() blocked for ten seconds and audio came in
#: ten-second lumps (measured).  ALSA ignores it and CoreAudio caps it near a
#: second, so it is not asked for there either.
AUDIO_BUFFER_SECONDS = 5
#: a microphone that has delivered nothing for this long while Slicer kept
#: running has gone (unplugged); Qt does not say so itself
QUIET_SECONDS = 4.0
#: a microphone that delivered nothing but exact zeros for this long is muted
#: by the system (macOS: Slicer has no microphone permission)
SILENCE_WARNING_SECONDS = 4.0
#: video frame rate and largest frame
VIDEO_FPS = 10
VIDEO_MAX_SIZE = (1920, 1080)
#: the strip above the views naming the case and the lesion
BANNER_HEIGHT = 26
#: Slicer's setting for ffmpeg, shared with the Screen Capture module
FFMPEG_SETTING = "General/ffmpegPath"
#: frames waiting for the encoder; past this the newest repeat the last one
FRAME_QUEUE = 30
#: finishing a take (encoder flush, mux) may take this long at exit
FINISH_TIMEOUT_S = 120
#: while a take is saved, the main thread sleeps this long this often, which
#: lets the saving thread have the GIL (measured: a thread that took 0.3 s
#: alone took 20 s while Slicer idled, 0.5 s with this)
GIL_VALVE_MS = 20
GIL_VALVE_SLEEP_S = 0.001

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
#: queued in place of a frame: show the last frame again
_REPEAT_LAST = object()


_STARTUP_ENVIRONMENT = None


def _startupEnvironment():
    """The environment Slicer was started with, before its launcher put its own
    library and Python paths in: what outside programs need on Linux, where
    Slicer's copies of system libraries break them (a file manager started
    with Slicer's paths died on Slicer's libgssapi_krb5, measured).

    Read from Slicer once, on the main thread (the Recorder asks at once), and
    kept: the finishing thread runs ffmpeg too, and must not call into Qt."""
    global _STARTUP_ENVIRONMENT
    if _STARTUP_ENVIRONMENT is None:
        try:
            env = dict(slicer.util.startupEnvironment())
        except Exception:  # noqa: BLE001 - very old Slicer
            env = dict(os.environ)
        _STARTUP_ENVIRONMENT = env
    return dict(_STARTUP_ENVIRONMENT)


def _environment():
    """The startup environment in the C locale: for programs whose output is
    read (ffmpeg, pactl)."""
    env = _startupEnvironment()
    env["LC_ALL"] = "C"
    return env


def _run(command, timeout=10.0):
    """``(returncode, stdout, stderr)`` of a short outside command, or None."""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, env=_environment(),
            creationflags=_NO_WINDOW,
        )
    except Exception:  # noqa: BLE001 - missing, not executable, hung
        logging.debug("GTReview: running %s failed", command[0], exc_info=True)
        return None
    return done.returncode, done.stdout or "", done.stderr or ""


# --------------------------------------------------------------------------- #
# where recordings go
# --------------------------------------------------------------------------- #
def _homeFolder(batchDir):
    return os.path.join(
        os.path.expanduser("~"), RECORDINGS_FOLDER,
        recordlog.safe_name(os.path.basename(os.path.normpath(batchDir))),
    )


def recordingsFolder(batchDir):
    """``<batch>/GTReview-recordings``; ``~/GTReview-recordings/<batch>`` when the
    batch folder cannot be written (a read-only share)."""
    preferred = os.path.join(batchDir, RECORDINGS_FOLDER)
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=preferred, prefix=".gtreview-", delete=True)
        probe.close()
        return preferred
    except OSError:
        fallback = _homeFolder(batchDir)
        os.makedirs(fallback, exist_ok=True)
        return fallback


def recordingsFolders(batchDir):
    """Where the recordings of *batchDir* can be, the usual place first;
    nothing is created."""
    return [os.path.join(batchDir, RECORDINGS_FOLDER), _homeFolder(batchDir)]


def recordingPath(root, kind, caseId):
    """The recording of *kind* a case keeps: ``<root>/<kind>/<case_id>.wav|.mp4``."""
    return os.path.join(root, kind, recordlog.safe_name(caseId) + MEDIA_SUFFIX[kind])


def openFolder(path):
    """Show *path* in the system's file manager; False when nothing opened it.

    On Linux not through Qt: QDesktopServices starts ``xdg-open`` with Slicer's
    library path, and the file manager that starts then loads Slicer's copies
    of system libraries -- one died on the spot, so the button did nothing.
    With Slicer's startup environment it runs as if started from the desktop.
    """
    path = os.path.abspath(path)
    if sys.platform.startswith("linux"):
        env = _startupEnvironment()
        opener = shutil.which("xdg-open", path=env.get("PATH"))
        if opener:
            try:
                process = subprocess.Popen(
                    [opener, path], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True,
                )
            except OSError:
                logging.warning("GTReview: xdg-open could not be started", exc_info=True)
            else:
                # reaped in the background: xdg-open returns once the file
                # manager is up, and must not stay a zombie meanwhile
                threading.Thread(target=process.wait, name="GTReview xdg-open", daemon=True).start()
                return True
    return bool(_openUrl(path))


def _openUrl(path):
    """Explorer on Windows, the Finder on macOS (a seam the tests replace)."""
    return qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path))


# --------------------------------------------------------------------------- #
# macOS: whether Slicer may use the microphone at all
# --------------------------------------------------------------------------- #
_MAC_MICROPHONE = {}


def macAppBundle():
    """The Slicer.app this Slicer runs from; None on other systems and when
    it runs from a build tree."""
    if sys.platform != "darwin":
        return None
    path = os.path.realpath(slicer.app.applicationFilePath())
    while os.path.dirname(path) != path:
        if path.endswith(".app"):
            return path
        path = os.path.dirname(path)
    return None


def macMicrophoneProblem():
    """Why macOS will not let this Slicer use the microphone, or None.

    macOS hands an app the microphone only if its Info.plist says what for
    (NSMicrophoneUsageDescription) and -- signed with the hardened runtime,
    as the official Slicer builds are -- if its signature carries the
    audio-input entitlement.  Otherwise the microphone delivers silence and
    nobody is asked.  Slicer 5.12 has neither; GTReviewMacMicrophone.sh adds
    both.  A Slicer started from a terminal is judged by the terminal instead
    (measured: the official Slicer heard a tone that way), so it is let try.
    Found out once a session; None on other systems."""
    if "problem" not in _MAC_MICROPHONE:
        problem = macSigningProblem()
        if problem and _startedFromAShell():
            problem = None
        _MAC_MICROPHONE["problem"] = problem
    return _MAC_MICROPHONE["problem"]


def _startedFromAShell():
    """Whether Slicer was started from a shell rather than by the Finder, the
    Dock or ``open``, which start it from launchd (process 1)."""
    launcher = os.getppid()  # the launcher in Contents/MacOS, or a shell
    if launcher <= 1:
        return False
    result = _run(["ps", "-o", "ppid=", "-p", str(launcher)])
    try:
        return int(result[1].strip()) > 1
    except Exception:  # noqa: BLE001 - not known: assume the usual start
        return False


def macSigningProblem():
    """What in this Slicer.app keeps macOS from handing it the microphone
    (its Info.plist, its signature), or None; None on other systems."""
    bundle = macAppBundle()
    if bundle is None:
        return None
    try:
        with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as handle:
            usage = plistlib.load(handle).get("NSMicrophoneUsageDescription")
    except Exception:  # noqa: BLE001 - unreadable: better say nothing than something wrong
        logging.debug("GTReview: reading %s's Info.plist failed", bundle, exc_info=True)
        return None
    if not usage:
        return "this Slicer does not tell macOS what it would use the microphone for"
    # the process that opens the microphone, and the app macOS asks for
    for code in (os.path.realpath(slicer.app.applicationFilePath()), bundle):
        signature = _run(["codesign", "-dv", code])
        if signature is None or "runtime" not in signature[2]:
            continue  # no hardened runtime: no entitlement needed
        entitlements = _run(["codesign", "-d", "--entitlements", ":-", code])
        try:
            granted = plistlib.loads((entitlements[1] if entitlements else "").encode("utf-8")).get(
                "com.apple.security.device.audio-input")
        except Exception:  # noqa: BLE001 - no entitlements at all
            granted = False
        if not granted:
            return "this Slicer's signature does not carry the microphone entitlement"
    return None


def macEnableMicrophoneCommand():
    """The Terminal command that lets this Slicer use the microphone: the
    script with this Slicer.app, behind sudo when the user cannot change it."""
    bundle = macAppBundle() or "/Applications/Slicer.app"
    writable = os.access(os.path.join(bundle, "Contents", "Info.plist"), os.W_OK)
    return "{}bash {} {}".format("" if writable else "sudo ", shlex.quote(MAC_MICROPHONE_SCRIPT),
                                 shlex.quote(bundle))


# --------------------------------------------------------------------------- #
# microphones
# --------------------------------------------------------------------------- #
def _pulseDescriptions():
    """Readable names of PulseAudio / PipeWire sources, by source name; {} when
    there is no ``pactl``.  Qt 5 only knows the raw names."""
    if not sys.platform.startswith("linux"):
        return {}
    executable = shutil.which("pactl", path=_environment().get("PATH"))
    if not executable:
        return {}
    result = _run([executable, "list", "sources"], timeout=3.0)
    if result is None or result[0] != 0:
        return {}
    descriptions, name = {}, None
    for line in result[1].splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Description:") and name:
            descriptions[name] = line.split(":", 1)[1].strip()
            name = None
    return descriptions


def inputDevices():
    """The microphones to offer: ``[{"name", "realm", "label"}]``."""
    devices = qt.QAudioDeviceInfo.availableDevices(qt.QAudio.AudioInput)
    pairs = [(info.deviceName(), info.realm()) for info in devices]
    default = qt.QAudioDeviceInfo.defaultInputDevice()
    defaultName = None if default.isNull() else default.deviceName()
    return recordlog.input_device_choices(pairs, _pulseDescriptions(), defaultName)


def deviceInfo(name, realm):
    """The QAudioDeviceInfo for a device chosen earlier; '' is the system default.
    None when the device is not there any more."""
    if not name:
        default = qt.QAudioDeviceInfo.defaultInputDevice()
        return None if default.isNull() else default
    for info in qt.QAudioDeviceInfo.availableDevices(qt.QAudio.AudioInput):
        if info.deviceName() == name and (not realm or info.realm() == realm):
            return info
    return None


class MicrophoneInput:
    """One microphone through Qt.  Audio arrives on the main thread, in
    *onData(bytes)*; *onLevel(dBFS)* follows the loudness."""

    def __init__(self, onData, onLevel=None):
        self._onData = onData
        self._onLevel = onLevel
        self._audio = None
        self._io = None
        self.rate = self.channels = None
        self.label = ""
        self.started = None
        self.lastData = None
        self.heardSomething = False
        self._info = None
        self._named = False

    @property
    def active(self):
        return self._audio is not None

    def start(self, name, realm, label="", hold=True):
        """Open the device.  *hold* asks the sound system to keep audio for
        AUDIO_BUFFER_SECONDS while Slicer is busy -- on Windows, where that
        works; off for a level test, which wants small deliveries."""
        info = deviceInfo(name, realm)
        if info is None:
            raise RuntimeError("The microphone {} is not connected.".format(label or name or "(default)"))
        wanted = qt.QAudioFormat()
        wanted.setSampleRate(AUDIO_RATE)
        wanted.setChannelCount(AUDIO_CHANNELS)
        wanted.setSampleSize(16)
        wanted.setCodec("audio/pcm")
        wanted.setByteOrder(qt.QAudioFormat.LittleEndian)
        wanted.setSampleType(qt.QAudioFormat.SignedInt)
        fmt = wanted if info.isFormatSupported(wanted) else info.nearestFormat(wanted)
        if (fmt.sampleSize() != 16 or fmt.sampleType() != qt.QAudioFormat.SignedInt
                or fmt.byteOrder() != qt.QAudioFormat.LittleEndian or fmt.codec() != "audio/pcm"
                or fmt.channelCount() < 1):
            raise RuntimeError("{} offers no 16-bit audio.".format(label or info.deviceName()))
        audio = qt.QAudioInput(info, fmt)
        if hold and sys.platform == "win32":
            audio.setBufferSize(int(fmt.sampleRate() * fmt.channelCount() * 2 * AUDIO_BUFFER_SECONDS))
        io = audio.start()
        if io is None or audio.error() != qt.QAudio.NoError:
            audio.stop()
            raise RuntimeError("{} could not be opened (Qt audio error {}).".format(
                label or info.deviceName(), int(audio.error())))
        io.connect("readyRead()", self._onReadyRead)
        self._audio, self._io = audio, io
        self.rate, self.channels = int(fmt.sampleRate()), int(fmt.channelCount())
        self.label = label or info.deviceName()
        self.started = time.monotonic()
        self.lastData = self.started
        self.heardSomething = False
        self._info = info
        # only a device whose format reads valid while it is there can be
        # told gone by the format failing to read
        self._named = bool(name) and bool(info.preferredFormat().isValid())

    def failed(self):
        """A Qt error on an open device: unplugged, or the system took it away."""
        if self._audio is None:
            return False
        return self._audio.error() not in (qt.QAudio.NoError, qt.QAudio.UnderrunError)

    def gone(self):
        """The microphone picked by name has disappeared (unplugged).

        PulseAudio and PipeWire then move the recording to the default input
        and Qt carries on as if nothing happened -- a different microphone,
        unannounced.  The device info Qt keeps for a vanished source has no
        valid format any more (measured; costs microseconds, where listing the
        devices again takes ~50 ms).  A recording of the system default is
        meant to follow the default, so it is never "gone"."""
        if self._audio is None or not self._named:
            return False
        try:
            return not self._info.preferredFormat().isValid()
        except Exception:  # noqa: BLE001 - the plugin dropped the device altogether
            return True

    def _onReadyRead(self):
        self.drain()

    def drain(self):
        """Pass on whatever audio is waiting.  Called before a take ends, so
        audio captured before it does not land after it."""
        if self._io is None:
            return
        try:
            data = self._io.readAll().data()
        except Exception:  # noqa: BLE001 - device gone between signal and read
            return
        if not data:
            return
        self.lastData = time.monotonic()
        samples = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2")
        if samples.size:
            if not self.heardSomething and np.any(samples):
                self.heardSomething = True
            if self._onLevel is not None:
                rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
                self._onLevel(20.0 * np.log10(max(rms, 1.0) / 32768.0))
        self._onData(data)

    def stop(self):
        """Close the device, whatever the last delivery's receiver does."""
        if self._audio is None:
            return
        try:
            self.drain()
        finally:
            try:
                self._io.disconnect("readyRead()", self._onReadyRead)
            except Exception:  # noqa: BLE001 - already gone with the device
                pass
            self._audio.stop()
            self._audio = self._io = None
            if self._onLevel is not None:
                self._onLevel(None)


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #
_FFMPEG = {}


def _ffmpegCandidates():
    settings = slicer.app.userSettings()
    if settings.contains(FFMPEG_SETTING):
        try:
            yield slicer.app.toSlicerHomeAbsolutePath(str(settings.value(FFMPEG_SETTING)))
        except Exception:  # noqa: BLE001
            pass
    try:
        import imageio_ffmpeg

        yield imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - not installed, or its binary missing
        pass
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = shutil.which(name, path=_environment().get("PATH"))
    if found:
        yield found
    for path in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        yield path


def findFfmpeg(refresh=False):
    """``(path, h264)`` of a working ffmpeg, ``(None, False)`` without one.

    Looked for where Slicer's Screen Capture module keeps it, then in the
    ``imageio-ffmpeg`` package, then on the PATH.  *h264* says whether it has
    libx264; without, the video is MPEG-4 part 2.  The video needs it: it is
    what puts the sound into the video.
    """
    if _FFMPEG and not refresh:
        return _FFMPEG["path"], _FFMPEG["h264"]
    _FFMPEG.update(path=None, h264=False)
    for candidate in _ffmpegCandidates():
        if not candidate or not os.path.isfile(candidate):
            continue
        result = _run([candidate, "-hide_banner", "-encoders"], timeout=15.0)
        if result is None or result[0] != 0:
            continue
        _FFMPEG.update(path=candidate, h264=" libx264 " in result[1])
        break
    return _FFMPEG["path"], _FFMPEG["h264"]


def installFfmpeg():
    """Install ``imageio-ffmpeg`` (a self-contained ffmpeg) from PyPI; the path or None."""
    slicer.util.pip_install("imageio-ffmpeg")
    return findFfmpeg(refresh=True)[0]


class FfmpegWriter:
    """Frames into an ffmpeg process, written by a thread of its own.

    The video is fragmented MP4, so a file cut short by a crash still plays.
    """

    def __init__(self, ffmpeg, h264, path, width, height, fps):
        self.path = path
        codec = (["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"] if h264
                 else ["-c:v", "mpeg4", "-q:v", "4"])
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgra", "-s", "{}x{}".format(width, height),
            "-r", str(fps), "-i", "-", "-an",
        ] + codec + [
            "-g", str(int(fps) * 2), "-pix_fmt", "yuv420p",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof", path,
        ]
        self._errors = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._errors,
            env=_environment(), creationflags=_NO_WINDOW,
        )
        self._queue = queue.Queue(maxsize=FRAME_QUEUE)
        self._carry = 0
        self.frames = 0
        self.error = None
        self._thread = threading.Thread(target=self._write, name="GTReview video writer", daemon=True)
        self._thread.start()

    def push(self, frame, repeat=1):
        """Queue *frame* (bytes) to be shown *repeat* times; never blocks.  When
        the encoder falls behind, a frame is dropped and the next one repeats
        for it, so the video still runs as long as the clock did."""
        repeat = int(repeat) + self._carry
        try:
            self._queue.put_nowait((frame, repeat))
            self._carry = 0
        except queue.Full:
            self._carry = repeat

    def _write(self):
        stream = self._process.stdin
        last = None
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, repeat = item
            if frame is _REPEAT_LAST:
                frame = last
            if frame is None or self.error is not None:
                continue
            last = frame
            try:
                for _ in range(repeat):
                    stream.write(frame)
                self.frames += repeat
            except (BrokenPipeError, OSError, ValueError) as error:
                self.error = "ffmpeg stopped taking frames ({})".format(error)

    @property
    def broken(self):
        return self.error is not None or self._process.poll() is not None

    def close(self, timeout=FINISH_TIMEOUT_S):
        """Flush and wait for ffmpeg; True when it finished cleanly.  Runs in the
        finishing thread, never on the main thread."""
        if self._carry:
            # frames owed for pictures the full queue turned away
            self._queue.put((_REPEAT_LAST, self._carry))
            self._carry = 0
        self._queue.put(None)
        self._thread.join(timeout)
        try:
            self._process.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            code = self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            code = -1
        if code != 0 and self.error is None:
            self._errors.seek(0)
            message = self._errors.read().decode("utf-8", "replace").strip()
            self.error = "ffmpeg exited with {}: {}".format(code, message[-400:])
        self._errors.close()
        return self.error is None


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def viewSources():
    """``[(label, name)]`` of what the video can show: the whole view area
    (name '') and each slice view of the current layout."""
    sources = [("All views", "")]
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return sources
    for name in layoutManager.sliceViewNames():
        widget = layoutManager.sliceWidget(name)
        if widget is not None and widget.isVisible():
            sources.append(("{} view".format(name), name))
    return sources


def grabViews(source=""):
    """A QImage of slice view *source* with its controller bar, or of the whole
    view area; None when there is nothing to grab.

    Rendered from Slicer's own widgets, not read off the screen: no
    screen-recording permission, whatever covers the window."""
    layoutManager = slicer.app.layoutManager()
    mainWindow = slicer.util.mainWindow()
    if layoutManager is None or mainWindow is None:
        return None
    if source:
        widget = layoutManager.sliceWidget(source)
        if widget is not None and widget.isVisible():
            image = ctk.ctkWidgetsUtils.grabWidget(widget)
            return image if image.width() > 1 and image.height() > 1 else None
    viewport = layoutManager.viewport()
    topLeft = viewport.mapTo(mainWindow, viewport.rect.topLeft())
    bottomRight = viewport.mapTo(mainWindow, viewport.rect.bottomRight())
    size = bottomRight - topLeft
    if size.x() < 2 or size.y() < 2:
        return None
    return ctk.ctkWidgetsUtils.grabWidget(
        mainWindow, qt.QRect(topLeft.x(), topLeft.y(), size.x(), size.y()))


def frameSize(image, maximum):
    """``(width, height)`` of the video for a first picture *image*: the picture
    fitted into *maximum* under the banner, both even."""
    width, height = image.width(), image.height()
    scale = min(1.0, float(maximum[0]) / width, float(maximum[1] - BANNER_HEIGHT) / height)
    width = max(2, int(width * scale) // 2 * 2)
    height = max(2, (int(height * scale) + BANNER_HEIGHT) // 2 * 2)
    return width, height


def composeFrame(image, size, clock, case, lesion):
    """The video frame: *image* fitted under a banner, on a canvas of *size*."""
    width, height = size
    canvas = qt.QImage(width, height, qt.QImage.Format_RGB32)
    canvas.fill(qt.QColor(0, 0, 0))
    painter = qt.QPainter(canvas)
    try:
        area = height - BANNER_HEIGHT
        if image is not None:
            # A grab on a high-DPI screen (Retina, Windows at 150 %) carries its
            # device pixel ratio, and QPainter draws such an image at its
            # logical size -- the views would fill a quarter of a Retina frame.
            # Its device pixels are the picture.
            image.setDevicePixelRatio(1.0)
            if image.width() != width or image.height() != area:
                image = image.scaled(width, area, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
            painter.drawImage((width - image.width()) // 2,
                              BANNER_HEIGHT + (area - image.height()) // 2, image)
        painter.fillRect(0, 0, width, BANNER_HEIGHT, qt.QColor(24, 24, 24))
        painter.setRenderHint(qt.QPainter.Antialiasing, True)
        painter.setPen(qt.Qt.NoPen)
        painter.setBrush(qt.QColor(220, 40, 40))
        dot = BANNER_HEIGHT // 2
        painter.drawEllipse(8, (BANNER_HEIGHT - dot) // 2, dot, dot)
        font = qt.QFont()
        font.setPixelSize(max(10, BANNER_HEIGHT - 11))
        font.setBold(True)
        painter.setFont(font)
        metrics = qt.QFontMetrics(font)
        baseline = (BANNER_HEIGHT + metrics.ascent() - metrics.descent()) // 2
        x = 8 + dot + 8
        for text, color in ((clock, qt.QColor(235, 235, 235)), (case, qt.QColor(160, 200, 255)),
                            (lesion, qt.QColor(255, 214, 74))):
            room = width - x - 8
            if room <= 10 or not text:
                continue
            shown = metrics.elidedText(text, qt.Qt.ElideRight, room)
            painter.setPen(color)
            painter.drawText(x, baseline, shown)
            x += metrics.horizontalAdvance(shown) + 18
    finally:
        painter.end()
    return canvas


# --------------------------------------------------------------------------- #
# takes
# --------------------------------------------------------------------------- #
class Take:
    """One recording of one case, of one kind (MICROPHONE or VIDEO): from the
    switch, or the case switch, that started it to the one that ended it."""

    def __init__(self, kind, folder, name, stem, meta, start):
        self.kind = kind
        self.folder = folder
        self.name = name
        self.stem = stem
        self.start = start
        self.log = recordlog.EventLog(meta)
        self.wav = None
        self.audioOffset = None
        self.audioDevice = ""
        self.gaps = []
        self.writer = None
        self.h264 = False
        self.videoOffset = None
        self.videoSize = None
        self.videoFps = None
        self.videoSource = ""
        self.pacer = None
        self.lastFrame = None
        self.ffmpeg = None
        self.end = None
        self.problem = None
        #: whether the saved video carries the microphone
        self.sound = False

    def working(self, suffix):
        """A file of this take while it runs or is saved."""
        return os.path.join(self.folder, self.stem + suffix)

    def export(self, suffix):
        """The case's recording of this kind, which this take replaces."""
        return os.path.join(self.folder, self.name + suffix)

    @property
    def target(self):
        return self.export(MEDIA_SUFFIX[self.kind])

    def at(self, now):
        return max(0.0, now - self.start)


_VOICE_CACHE = {}
_VOICE_LOCK = threading.Lock()


def _voice(rate, names):
    """The clips *names* at *rate* that the voice has, each loaded once: a
    recording speaks a few of the 532, and all of them at a microphone's rate
    would hold some 50 MB."""
    with _VOICE_LOCK:
        missing = [name for name in names if (rate, name) not in _VOICE_CACHE]
        for name, clip in wavfile.load_voice(VOICE_DIR, rate, missing).items():
            _VOICE_CACHE[(rate, name)] = clip
        return {name: _VOICE_CACHE[(rate, name)] for name in names if (rate, name) in _VOICE_CACHE}


def finishTake(take):
    """Turn a stopped take into the case's recording of its kind, replacing the
    one it had.  Runs in a thread: no Qt, VTK or scene access.

    Returns ``(paths written, problem)``, *problem* being a sentence for the
    reader when the take could not be saved as meant.  A case's recording is
    replaced only by one that was made: a take that caught nothing leaves it
    alone.  The take's own files go once it is saved, and stay whenever
    something failed, so nothing recorded is ever lost."""
    meta = take.log.meta
    events = list(take.log.events)
    end = float(take.end or 0.0)
    meta["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["duration"] = end
    working = [take.working(suffix) for suffix in WORKING_SUFFIXES]
    keep = set()
    produced = None
    try:
        if take.kind == MICROPHONE:
            produced = _finishMicrophone(take, events, meta)
        else:
            produced = _finishVideo(take, events, end, meta, keep)
    except Exception as error:  # noqa: BLE001 - reported; the take's files stay
        logging.exception("GTReview: saving %s failed", take.stem)
        keep.update(working)
        take.problem = "The {} recording of {} could not be saved ({}); what was recorded is kept as {}*.".format(
            take.kind, take.name, error, take.stem)
    written = []
    if produced is not None:
        written.append(produced)
        meta["file"] = os.path.basename(produced)
        try:
            saving = take.working(".saving.json")
            take.log.write_json(saving)
            maskio.replace_file(saving, take.export(".json"))
            written.append(take.export(".json"))
        except Exception:  # noqa: BLE001 - the recording itself is saved
            logging.exception("GTReview: writing the event log of %s failed", take.stem)
    for path in working:
        if path in keep or not os.path.exists(path):
            continue
        try:
            maskio.remove_file(path)
        except OSError as error:
            logging.warning("GTReview: %s left behind: %s", path, error)
    logging.info("GTReview: %s recording of %s saved: %s", take.kind, take.name,
                 ", ".join(os.path.basename(p) for p in written) or "nothing (nothing was recorded)")
    return written, take.problem


def _narrateInto(take, events, source, target, audio):
    """*source* with the announcements mixed in, as *target*."""
    rate, mic = wavfile.read_wav(source)
    audio.update(rate=rate, channels=int(mic.shape[1]))
    placements = []
    for at, names, _text in recordlog.announcement_schedule(events, float(audio["offset"])):
        clip = wavfile.phrase(names, _voice(rate, names), rate)
        if clip is None:
            # a voice without this phrase (one recorded by hand) says it word by word
            words = recordlog.words_for(names)
            clip = wavfile.phrase(words, _voice(rate, words), rate)
        if clip is not None:
            placements.append((at, clip))
    wavfile.write_wav(target, rate, wavfile.narrate(mic, rate, placements))
    audio["announcements"] = len(placements)


def _audioMeta(take, meta):
    meta["audio"] = {
        "device": take.audioDevice, "offset": float(take.audioOffset or 0.0),
        "duration": take.wav.seconds, "gaps": take.gaps,
    }
    return meta["audio"]


def _finishMicrophone(take, events, meta):
    """``microphone/<case>.wav``: the narrated microphone with its cue points;
    None when the microphone delivered nothing."""
    if take.wav is None or take.wav.frames == 0 or not os.path.isfile(take.wav.path):
        return None
    audio = _audioMeta(take, meta)
    saving = take.working(".saving.wav")
    try:
        _narrateInto(take, events, take.wav.path, saving, audio)
    except Exception as error:  # noqa: BLE001 - the reader's voice rather than nothing
        logging.exception("GTReview: narrating %s failed", take.stem)
        audio["narration_error"] = str(error)
        maskio.replace_file(take.wav.path, saving)
        take.problem = ("The lesion numbers could not be spoken into the recording of {} ({}); "
                        "it is saved without them.".format(take.name, error))
    try:
        wavfile.append_cue_points(saving, recordlog.cue_marks(events, audio["offset"]))
    except Exception as error:  # noqa: BLE001 - the recording itself is whole
        logging.warning("GTReview: no cue points in %s: %s", saving, error)
        audio["cue_error"] = str(error)
    maskio.replace_file(saving, take.target)
    return take.target


def _finishVideo(take, events, end, meta, keep):
    """``video/<case>.mp4`` with the narrated microphone and the chapters; None
    when not a single frame was recorded."""
    if take.writer is None:
        return None
    ok = take.writer.close()  # ffmpeg flushes here
    frames = take.writer.frames
    offset = float(take.videoOffset or 0.0)
    duration = frames / float(take.videoFps or VIDEO_FPS)
    video = meta["video"] = {
        "source": take.videoSource or "all views", "codec": "H.264" if take.h264 else "MPEG-4",
        "fps": take.videoFps, "size": list(take.videoSize or ()), "frames": frames,
        "offset": offset, "duration": duration,
    }
    if not ok:
        video["error"] = take.writer.error
        logging.warning("GTReview: the video of %s: %s", take.stem, take.writer.error)
    if frames == 0 or not os.path.isfile(take.writer.path):
        return None
    sound = None
    if take.wav is not None and take.wav.frames > 0 and os.path.isfile(take.wav.path):
        audio = _audioMeta(take, meta)
        try:
            _narrateInto(take, events, take.wav.path, take.working(".narrated.wav"), audio)
            sound = take.working(".narrated.wav")
        except Exception as error:  # noqa: BLE001 - the reader's voice rather than nothing
            logging.exception("GTReview: narrating %s failed", take.stem)
            audio["narration_error"] = str(error)
            sound = take.wav.path
            take.problem = ("The lesion numbers could not be spoken into the video of {} ({}); "
                            "its sound is the microphone alone.".format(take.name, error))
    saving = take.working(".saving.mp4")
    intervals = recordlog.lesion_intervals(events, end)
    if not _mux(take, sound, intervals, offset, duration, saving):
        if not ok:
            # neither the encoder nor ffmpeg's second pass made a video of it:
            # the case keeps the one it had, and this take's files stay
            keep.update(take.working(s) for s in WORKING_SUFFIXES)
            take.problem = ("The video of {} could not be saved ({}); what was recorded is kept "
                            "as {}*.".format(take.name, video.get("error"), take.stem))
            return None
        # the picture as encoded, without sound or chapters, is still the
        # case's video; the sound stays next to it
        video["mux_error"] = True
        if sound is not None:
            keep.add(sound)
        maskio.replace_file(take.writer.path, saving)
        take.problem = ("The sound could not be put into the video of {}; the video is saved "
                        "without it{}.".format(take.name, ", and the sound is kept as " + os.path.basename(sound)
                                               if sound else ""))
    video["sound"] = take.sound and "mux_error" not in video
    maskio.replace_file(saving, take.target)
    return take.target


def _mux(take, audioPath, intervals, videoOffset, duration, final):
    """The finished MP4: video, the narrated audio lined up with it, and one
    chapter per lesion.  False when ffmpeg refused."""
    chapters = take.working("_chapters.txt")
    try:
        with open(chapters, "w", encoding="utf-8") as handle:
            handle.write(recordlog.ffmetadata(intervals, videoOffset, duration,
                                              title=take.log.meta.get("case_id")))
        command = [take.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", take.writer.path]
        audio = audioPath is not None and os.path.isfile(audioPath) and take.wav is not None
        if audio:
            audioOffset = float(take.audioOffset or 0.0)
            overlap = (min(videoOffset + duration, audioOffset + take.wav.seconds)
                       - max(videoOffset, audioOffset))
            # a microphone that stopped before the first frame shares nothing with it
            audio = overlap > 0.1
        take.sound = bool(audio)
        if audio:
            # the audio clipped to the video's stretch of the clock; never the
            # other way round: a microphone lost early leaves the video whole
            shift = videoOffset - audioOffset
            command += ["-ss", "{:.3f}".format(shift)] if shift >= 0 else ["-itsoffset", "{:.3f}".format(-shift)]
            command += ["-t", "{:.3f}".format(duration + min(0.0, shift)), "-i", audioPath]
        command += ["-f", "ffmetadata", "-i", chapters, "-map", "0:v"]
        if audio:
            command += ["-map", "1:a", "-c:a", "aac", "-b:a", "96k"]
        meta = "2" if audio else "1"
        command += ["-map_metadata", meta, "-map_chapters", meta, "-c:v", "copy",
                    "-movflags", "+faststart", "-f", "mp4", final]
        result = _run(command, timeout=FINISH_TIMEOUT_S)
        if result is None or result[0] != 0:
            logging.warning("GTReview: muxing %s failed: %s", take.stem,
                            (result[2] if result else "ffmpeg did not run")[-400:])
            if os.path.exists(final):
                maskio.remove_file(final)
            return False
        return True
    finally:
        try:
            if os.path.exists(chapters):
                maskio.remove_file(chapters)
        except OSError as error:
            logging.warning("GTReview: %s left behind: %s", chapters, error)


# --------------------------------------------------------------------------- #
# the recorder
# --------------------------------------------------------------------------- #
class Recorder:
    """The microphone and video switches: one take per case for each.

    The video records the microphone too, as its sound.  The panel tells the
    recorder what happens (:meth:`note`), when the case changes
    (:meth:`caseChanged`) and what the banner should say (:meth:`setBanner`).
    *context()* returns the meta of the case on screen, or None without one.

    Callbacks: *onChange()* after any switch and when a take has been saved;
    *onLevel(dBFS or None)*; *onTick(seconds)* twice a second while
    recording; *onWarning(text)*; *onTake(take)* right after a take opens (to
    log where the case stands, into that take); and *confirmReplace(case_id,
    kinds, switching)*, asked before takes would replace recordings a case
    already has -- *switching* when the reader moved on to that case while
    recording.  Without it, recordings are replaced without asking.
    """

    def __init__(self, context, onChange=None, onLevel=None, onTick=None, onWarning=None,
                 onTake=None, confirmReplace=None):
        self._context = context
        self._onChange = onChange or (lambda: None)
        self._onTick = onTick or (lambda seconds: None)
        self._onWarning = onWarning or (lambda text: None)
        self._onTake = onTake or (lambda take: None)
        self._confirmReplace = confirmReplace or (lambda caseId, kinds, switching: True)
        self.mic = MicrophoneInput(self._onAudio, onLevel)
        self.audioOn = False
        self.videoOn = False
        self.testing = False
        self.microphone = ("", "", "")
        self.videoSource = ""
        self.takes = {}
        self.banner = ("", "No lesion selected")
        self.lastFolder = None
        self.lastSaved = []
        self._bytes = 0
        self._clock = None
        self._silenceWarned = False
        self._lastTick = None
        self._quiet = 0.0
        self._micLost = None
        self._finishing = []
        self._pending = {}
        self._pendingLock = threading.Lock()
        self._results = queue.SimpleQueue()
        _environment()  # read from Slicer here, on the main thread, once
        self._frameTimer = qt.QTimer()
        self._frameTimer.connect("timeout()", self._onFrameTick)
        self._clockTimer = qt.QTimer()
        self._clockTimer.setInterval(500)
        self._clockTimer.connect("timeout()", self._onClockTick)
        self._testTimer = qt.QTimer()
        self._testTimer.setSingleShot(True)
        self._testTimer.connect("timeout()", self.stopTest)
        self._valveTimer = qt.QTimer()
        self._valveTimer.setInterval(GIL_VALVE_MS)
        self._valveTimer.connect("timeout()", self._onValve)

    # ---------------------------------------------------------------- state
    @property
    def recording(self):
        return self.audioOn or self.videoOn

    @property
    def capturing(self):
        """The microphone is open for a recording, not for a level test."""
        return self.mic.active and not self.testing

    def elapsed(self):
        """Seconds since the oldest open take began."""
        now = time.monotonic()
        return max([take.at(now) for take in self.takes.values()] or [0.0])

    def busyFinishing(self):
        self._finishing = [t for t in self._finishing if t.is_alive()]
        return len(self._finishing)

    def note(self, kind, take=None, **fields):
        """Log an event into every open take, or into *take* only; ignored
        when nothing records."""
        now = time.monotonic()
        event = None
        for one in ([take] if take is not None else list(self.takes.values())):
            event = one.log.add(one.at(now), kind, **fields)
        return event

    def setBanner(self, case, lesion):
        self.banner = (str(case or ""), str(lesion or ""))

    def hasRecording(self, context, kind):
        """Whether the case of *context* has a recording of *kind*, saved or
        still being saved."""
        for root in recordingsFolders(context["batch_dir"]):
            path = recordingPath(root, kind, context["case_id"])
            if os.path.exists(path):
                return True
            with self._pendingLock:
                if self._pending.get(path):
                    return True
        return False

    # ---------------------------------------------------------- microphone
    def startAudio(self, name, realm, label=""):
        """Turn the microphone switch on: a take of the case on screen.  False
        when the reader chose to keep the recording that case has."""
        if self.audioOn:
            return True
        self.stopTest()
        context = self._caseContext()
        if not self._mayReplace(context, [MICROPHONE], switching=False):
            return False
        self._openMicrophone(name, realm, label)
        try:
            self._openTake(MICROPHONE, context)
        except Exception:
            self._closeMicrophone()
            raise
        self.audioOn = True
        self.lastSaved = []
        self._started()
        return True

    def stopAudio(self):
        """Turn the microphone switch off and save its take; always completes,
        so a failing disk or device can never leave it recording."""
        if not self.audioOn:
            return
        self.audioOn = False
        try:
            self._closeTakes([MICROPHONE], "audio_stopped")
        finally:
            self._closeMicrophone()
            self._stoppedOne()

    def _openMicrophone(self, name, realm, label):
        """Open the microphone for recording, unless it is open already."""
        if self.mic.active:
            return
        self.mic.start(name, realm, label)
        self.microphone = (name, realm, label)
        self._bytes = 0
        self._clock = recordlog.AudioClock(self.mic.rate, time.monotonic())
        self._silenceWarned = False
        self._lastTick = None
        self._quiet = 0.0
        self._micLost = None

    def _closeMicrophone(self):
        """Close the microphone once neither switch needs it."""
        if self.recording or self.testing or not self.mic.active:
            return
        try:
            self.mic.stop()
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: stopping the microphone failed")
        self._clock = None

    @staticmethod
    def _closeWav(take):
        if take is None or take.wav is None or take.wav.closed:
            return
        try:
            take.wav.close()
        except OSError:
            logging.exception("GTReview: closing %s failed", take.wav.path)

    def _onAudio(self, data):
        if self.testing or self._micLost is not None:
            return
        if self.mic.gone():
            # Checked on every delivery, not only on the clock: what arrives
            # now may already be another microphone's (see MicrophoneInput.gone)
            self._microphoneLost(
                "The microphone {} was disconnected.  (Left running, the system would have "
                "carried on from another microphone.)".format(self.mic.label))
            return
        self._bytes += len(data)
        takes = [take for take in self.takes.values() if take.wav is not None and not take.wav.closed]
        for take in takes:
            try:
                take.wav.write(data)
            except OSError as error:
                self._audioWriteFailed(take, error)
                return
        if self._clock is None:
            return
        # Judged when audio arrives, not on a timer: WinMM delivers in blocks
        # of seconds, and between two blocks the file is always "behind".
        now = time.monotonic()
        block = 2 * max(1, int(self.mic.channels or 1))
        missing = self._clock.observe(now, self._bytes // block)
        if not missing:
            return
        seconds = missing / float(self.mic.rate)
        self._bytes += missing * block
        for take in takes:
            try:
                take.wav.write_silence(missing)
            except OSError as error:
                self._audioWriteFailed(take, error)
                return
            take.gaps.append({"t": take.at(now), "seconds": seconds})
            take.log.add(take.at(now), "audio_gap", seconds=seconds)
        logging.warning("GTReview: %.1f s of microphone audio was lost; padded with silence", seconds)

    def _microphoneLost(self, text):
        """The microphone stopped (unplugged, or the system took it): its take
        is saved with what it has, and the video carries on without sound.
        Done from the event loop, not from inside a delivery; nothing the
        microphone delivers from now on is kept."""
        if self._micLost is not None:
            return
        self._micLost = text

        def stop():
            audio, video = self.audioOn, self.videoOn
            self._dropMicrophone()
            parts = [text]
            if audio:
                parts.append("The microphone recording was stopped; what it recorded until then is kept.")
            if video:
                parts.append("The video carries on without sound.")
            self._onWarning("  ".join(parts))

        qt.QTimer.singleShot(0, stop)

    def _dropMicrophone(self):
        if self.audioOn:
            self.audioOn = False
            try:
                self._closeTakes([MICROPHONE], "audio_stopped")
            except Exception:  # noqa: BLE001
                logging.exception("GTReview: saving the microphone recording failed")
        video = self.takes.get(VIDEO)
        if video is not None and video.wav is not None and not video.wav.closed:
            video.log.add(video.at(time.monotonic()), "audio_stopped")
            self._closeWav(video)
        try:
            self.mic.stop()
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: stopping the microphone failed")
        self._clock = None
        self._stoppedOne()

    def _audioWriteFailed(self, take, error):
        """A recording file cannot be written (a full disk, a share gone).
        Close it as far as it goes and stop everything from the event loop --
        the other files are on the same disk."""
        if self._micLost is not None:
            return
        self._micLost = str(error)
        logging.error("GTReview: writing %s failed: %s", take.wav.path, error)
        self._closeWav(take)
        take.log.add(take.at(time.monotonic()), "audio_write_failed", detail=str(error))

        def stop():
            self.stopAll()
            self._onWarning("The recording could not be written ({}), so it was stopped; "
                            "what reached the disk is kept.".format(error))

        qt.QTimer.singleShot(0, stop)

    def startTest(self, name, realm, label="", seconds=10):
        """Open the microphone for *seconds* to watch its level; nothing is saved."""
        if self.mic.active or self.testing:
            return
        self.mic.start(name, realm, label, hold=False)
        self.testing = True
        self._testTimer.start(int(seconds * 1000))
        self._onChange()

    def stopTest(self):
        if not self.testing:
            return
        self._testTimer.stop()
        self.mic.stop()
        self.testing = False
        self._onChange()

    # ---------------------------------------------------------------- video
    def startVideo(self, source="", name="", realm="", label="", withSound=True):
        """Turn the video switch on: a take of the case on screen, with
        microphone *name* as its sound unless not *withSound* (a Mac whose
        Slicer may not use the microphone).  False when the reader chose to
        keep the video that case has.  Needs ffmpeg."""
        if self.videoOn:
            return True
        if not findFfmpeg()[0]:
            raise RuntimeError("Recording video needs ffmpeg: get it with \"Get ffmpeg...\".")
        self.stopTest()
        context = self._caseContext()
        if not self._mayReplace(context, [VIDEO], switching=False):
            return False
        silent = None
        if withSound:
            try:
                self._openMicrophone(name, realm, label)
            except Exception as error:  # noqa: BLE001 - the video goes ahead without sound
                silent = str(error)
        try:
            self._openTake(VIDEO, context, source)
        except Exception:
            self._closeMicrophone()
            raise
        self.videoSource = source or ""
        self.videoOn = True
        self.lastSaved = []
        self._frameTimer.setInterval(int(1000 / VIDEO_FPS))
        self._frameTimer.start()
        self._started()
        if silent is not None:
            self._onWarning("The video records without sound: {}".format(silent))
        return True

    def stopVideo(self):
        """Turn the video switch off and save its take."""
        if not self.videoOn:
            return
        self._frameTimer.stop()
        self.videoOn = False
        try:
            self._closeTakes([VIDEO], "video_stopped")
        finally:
            self._closeMicrophone()
            self._stoppedOne()

    def setVideoSource(self, source):
        """Change what the video shows; takes effect from the next frame."""
        self.videoSource = source or ""
        take = self.takes.get(VIDEO)
        if take is not None and self.videoOn:
            take.videoSource = self.videoSource
            self.note("video_source", take=take, source=self.videoSource or "all views")

    def _openVideo(self, take, image, now):
        ffmpeg, h264 = findFfmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg is not there any more")
        take.videoSize = frameSize(image, VIDEO_MAX_SIZE)
        take.videoFps = VIDEO_FPS
        take.writer = FfmpegWriter(ffmpeg, h264, take.working(".recording.mp4"),
                                   take.videoSize[0], take.videoSize[1], VIDEO_FPS)
        take.h264 = bool(h264)
        take.ffmpeg = ffmpeg
        take.videoOffset = take.at(now)
        take.pacer = recordlog.FramePacer(take.videoFps, now)

    def _onFrameTick(self):
        take = self.takes.get(VIDEO)
        if not self.videoOn or take is None:
            return
        now = time.monotonic()
        try:
            if take.writer is None:
                image = grabViews(self.videoSource)
                if image is None:
                    return
                self._openVideo(take, image, now)
            due = take.pacer.due(now)
            if due <= 0:
                return
            if take.writer.broken:
                raise RuntimeError(take.writer.error or "the video encoder stopped")
            # In the middle of a load, a recount or a save Slicer shows the wait
            # cursor and the views may be half updated: repeat the last picture.
            busy = qt.QApplication.overrideCursor() is not None and take.lastFrame is not None
            image = None if busy else grabViews(self.videoSource)
            if image is None and take.lastFrame is None:
                return
            if image is not None:
                if due > 1 and take.lastFrame is not None:
                    # Slicer was frozen: what was on screen then is the old
                    # picture, and only the newest frame shows the new one
                    take.writer.push(take.lastFrame, due - 1)
                    take.pacer.mark(due - 1)
                    due = 1
                canvas = composeFrame(image, take.videoSize,
                                      "REC " + recordlog.format_clock(take.at(now)), *self.banner)
                take.lastFrame = bytes(canvas.constBits())
            take.writer.push(take.lastFrame, due)
            take.pacer.mark(due)
        except Exception as error:  # noqa: BLE001 - never let a frame take Slicer down
            logging.exception("GTReview: video frame failed")
            self.stopVideo()
            self._onWarning("Video recording stopped: {}".format(error))

    # ---------------------------------------------------------------- takes
    def _caseContext(self):
        context = self._context()
        if not context:
            raise RuntimeError("Load a case first: recordings are kept per case.")
        return context

    def _mayReplace(self, context, kinds, switching):
        """True unless the case has recordings of *kinds* and the reader
        chose to keep them."""
        existing = [kind for kind in kinds if self.hasRecording(context, kind)]
        if not existing:
            return True
        return bool(self._confirmReplace(context["case_id"], existing, switching))

    def _openTake(self, kind, context, source=""):
        """A new take of *kind* for the case of *context*, recording at once."""
        root = recordingsFolder(context["batch_dir"])
        folder = os.path.join(root, kind)
        os.makedirs(folder, exist_ok=True)
        now = datetime.datetime.now()
        stem = recordlog.unique_stem(folder, recordlog.take_stem(context["case_id"], now), WORKING_SUFFIXES)
        meta = dict(context)
        meta.update(kind=kind, started=now.isoformat(timespec="seconds"), reader=_reader())
        start = time.monotonic()
        take = Take(kind, folder, recordlog.safe_name(context["case_id"]), stem, meta, start)
        if self.mic.active and not self.testing and self._micLost is None:
            take.wav = wavfile.WavWriter(take.working(".recording.wav"), self.mic.rate, self.mic.channels)
            # audio captured before this moment that the system has not handed
            # over yet -- WinMM keeps a block of a second -- starts the file,
            # so it starts that much before the take
            pending = max(0.0, self._clock.shortfall(start, self._bytes // (2 * self.mic.channels))) \
                if self._clock is not None else 0.0
            take.audioOffset = -pending
            take.audioDevice = self.mic.label
        if kind == VIDEO:
            take.videoSource = source or ""
        self.takes[kind] = take
        self.lastFolder = root
        take.log.add(0.0, "recording_started", media=kind)
        if take.wav is not None:
            take.log.add(0.0, "audio_started", device=self.mic.label, rate=self.mic.rate,
                         channels=self.mic.channels)
        if kind == VIDEO:
            take.log.add(0.0, "video_started", source=take.videoSource or "all views",
                         sound=take.wav is not None)
        self._onTake(take)
        return take

    def _closeTakes(self, kinds, reason=None):
        """End and save the open takes of *kinds*.  Each stands alone: a
        microphone or a disk that fails here never leaves one half closed."""
        if self.mic.active and not self.testing and self._micLost is None:
            try:
                self.mic.drain()
            except Exception:  # noqa: BLE001
                logging.exception("GTReview: reading the last of the microphone failed")
        now = time.monotonic()
        for kind in kinds:
            take = self.takes.pop(kind, None)
            if take is None:
                continue
            try:
                if reason:
                    take.log.add(take.at(now), reason)
                self._closeWav(take)
                take.end = take.at(now)
                take.log.add(take.end, "recording_ended")
            finally:
                self._finish(take)

    def caseChanged(self):
        """The case on screen changed: the takes of the one before are saved
        and new ones start for this case -- after asking, when it has
        recordings they would replace.  A reload of the same case (Reset,
        another mask) carries on in the same takes."""
        if not self.takes:
            return
        context = self._context()
        if context is None:
            self.stopAll()
            self._onWarning("Recording stopped: no case is open.")
            return
        meta = next(iter(self.takes.values())).log.meta
        if (meta.get("case_dir"), meta.get("batch_dir")) == (context.get("case_dir"), context.get("batch_dir")):
            self.note("case_reloaded", mask_file=context.get("mask_file"))
            return
        kinds = list(self.takes)
        self._closeTakes(kinds)
        if not self._mayReplace(context, kinds, switching=True):
            self.stopAll()
            return
        try:
            for kind in kinds:
                self._openTake(kind, context, self.videoSource)
        except Exception as error:  # noqa: BLE001
            self.stopAll()
            self._onWarning("Recording stopped: {}".format(error))
            return
        self.lastSaved = []
        self._onChange()

    # ------------------------------------------------------------- saving
    def _finish(self, take):
        """Save *take* in a thread, after the takes before it, so a case's
        newest take is always the one that ends up its recording."""
        with self._pendingLock:
            self._pending[take.target] = self._pending.get(take.target, 0) + 1
        previous = self._finishing[-1] if self._finishing else None
        thread = threading.Thread(target=self._finishQuietly, args=(take, previous),
                                  name="GTReview saving {}".format(take.stem), daemon=True)
        thread.start()
        self._finishing.append(thread)
        if not self._valveTimer.isActive():
            self._valveTimer.start()

    def _finishQuietly(self, take, previous):
        written, problem = [], None
        try:
            if previous is not None:
                previous.join()
            written, problem = finishTake(take)
        except Exception as error:  # noqa: BLE001 - logged; the take's files stay
            logging.exception("GTReview: saving %s failed", take.stem)
            problem = "The {} recording of {} could not be saved ({}).".format(take.kind, take.name, error)
        finally:
            with self._pendingLock:
                left = self._pending.get(take.target, 1) - 1
                if left > 0:
                    self._pending[take.target] = left
                else:
                    self._pending.pop(take.target, None)
            self._results.put((written, problem))

    def _onValve(self):
        """While a take is saved: hand the GIL to the saving thread (see
        GIL_VALVE_MS), and tell the panel what was saved."""
        try:
            busy = self.busyFinishing()
            self._report()
            if busy:
                time.sleep(GIL_VALVE_SLEEP_S)
            else:
                self._valveTimer.stop()
        except Exception:  # noqa: BLE001 - a timer callback must never raise
            logging.exception("GTReview: the saving timer failed")

    def _report(self):
        changed = False
        while True:
            try:
                written, problem = self._results.get_nowait()
            except queue.Empty:
                break
            changed = True
            # everything saved since the switches were last turned on
            self.lastSaved += [path for path in written if path not in self.lastSaved]
            if problem:
                self._onWarning(problem)
        if changed:
            self._onChange()

    # ---------------------------------------------------------- lifecycle
    def _started(self):
        if not self._clockTimer.isActive():
            self._clockTimer.start()
        self._onChange()

    def _stoppedOne(self):
        if not self.recording:
            self._clockTimer.stop()
        self._onChange()

    def stopAll(self):
        """Stop everything and save the takes.  Never raises: it runs as a
        scene closes and as Slicer exits."""
        try:
            self.stopTest()
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: stopping the microphone test failed")
        self._frameTimer.stop()
        self.audioOn = self.videoOn = False
        try:
            self._closeTakes(list(self.takes))
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: saving the recording failed")
            self.takes = {}
        if self.mic.active:
            try:
                self.mic.stop()
            except Exception:  # noqa: BLE001
                logging.exception("GTReview: stopping the microphone failed")
        self._clock = None
        self._clockTimer.stop()
        self._onChange()

    def shutdown(self, wait=FINISH_TIMEOUT_S):
        """Stop everything and wait (up to *wait* s) for the files to be written."""
        self.stopAll()
        deadline = time.monotonic() + float(wait)
        for thread in list(self._finishing):
            thread.join(max(0.0, deadline - time.monotonic()))
        self._valveTimer.stop()
        return self.busyFinishing() == 0

    # ---------------------------------------------------------------- clock
    def _onClockTick(self):
        try:
            self._tick()
        except Exception:  # noqa: BLE001 - a timer callback must never raise
            logging.exception("GTReview: the recording clock failed")

    def _tick(self):
        now = time.monotonic()
        # Ticks more than a second apart mean Slicer itself was busy, and a
        # microphone owes nothing for that: audio held meanwhile arrives late.
        running = self._lastTick is not None and now - self._lastTick < 1.0
        self._lastTick = now
        if self.capturing and self._micLost is None:
            self.mic.drain()
            self._quiet = self._quiet + 0.5 if running and now - (self.mic.lastData or now) > 0.5 else 0.0
            if self.mic.gone():
                self._microphoneLost(  # in case nothing was delivered since
                    "The microphone {} was disconnected.".format(self.mic.label))
            elif self.mic.failed() or self._quiet >= QUIET_SECONDS:
                self._microphoneLost("The microphone stopped delivering audio (unplugged?).")
            elif (not self._silenceWarned and not self.mic.heardSomething
                    and now - (self.mic.started or now) > SILENCE_WARNING_SECONDS):
                self._silenceWarned = True
                hint = ("Check that Slicer may use it, under System Settings > Privacy & Security > "
                        "Microphone; Slicer as downloaded first needs GTReviewMacMicrophone.sh, see "
                        "\"Enable the microphone\" in GTReview's README." if sys.platform == "darwin" else
                        "Check that the microphone is not muted and that the system lets "
                        "desktop apps use it (Windows: Settings > Privacy > Microphone).")
                self._onWarning("The microphone records pure silence. " + hint)
        if self.recording:
            self._onTick(self.elapsed())


def _reader():
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name (a service account)
        return ""
