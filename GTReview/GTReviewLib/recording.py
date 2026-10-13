"""Record the reader's voice and the axial views during a review -- the Slicer half.

Two independent switches in the panel: the microphone and the video.  While
either is on, everything goes into one *segment* per case, in
``<batch>/GTReview-recordings/<case_id>/``, under a stem
``<case_id>_<YYYYMMDD-HHMMSS>`` shared by every file of the segment:

``<stem>_mic.wav``
    the microphone as captured, 16-bit PCM, written as it arrives
``<stem>.wav``
    the narrated copy: a chime and "lesion three" spoken in wherever the
    reader moved to another lesion, the reader's voice turned down under it
``<stem>.srt``, ``<stem>_labels.txt``
    the active lesion as subtitles (VLC, mpv) and as an Audacity label track
``<stem>.mp4`` (``<stem>.ogv`` without ffmpeg)
    the view area, or one slice view, under a banner naming the case and the
    active lesion; with the narrated audio and one chapter per lesion when
    ffmpeg muxes it
``<stem>.json``
    every event with the lesion's seed voxel, which survives renumbering --
    the record a later re-annotation can follow exactly

Capture never leaves Slicer's process.  The video is rendered from Slicer's own
views (no OS screen-recording permission, any monitor, a covered window); the
microphone goes through Qt Multimedia, which Slicer ships.  Qt moves audio on
the main thread, so while Slicer is busy the sound system holds the samples;
what it cannot hold is padded with silence (see :class:`recordlog.AudioClock`)
so the rest stays in step.

Encoding happens off the main thread: frames are piped to an ffmpeg process by
a writer thread, and a finished segment is turned into its files by a thread
of its own.  Nothing in those threads touches Qt, VTK or the scene.
"""

import datetime
import getpass
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import ctk
import qt
import slicer
import vtk

import numpy as np

from . import maskio, recordlog, wavfile

#: the spoken clips (whole phrases and single words)
VOICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "GTReviewVoice"
)
#: the folder of the batch directory recordings go into
RECORDINGS_FOLDER = "GTReview-recordings"
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
#: video frame rate and largest frame, with ffmpeg
VIDEO_FPS = 10
VIDEO_MAX_SIZE = (1920, 1080)
#: without ffmpeg: VTK's Theora writer encodes on the main thread, so less
THEORA_FPS = 5
THEORA_MAX_SIZE = (960, 540)
#: the strip above the views naming the case and the lesion
BANNER_HEIGHT = 26
#: Slicer's setting for ffmpeg, shared with the Screen Capture module
FFMPEG_SETTING = "General/ffmpegPath"
#: frames waiting for the encoder; past this the newest repeat the last one
FRAME_QUEUE = 30
#: finishing a segment (encoder flush, mux) may take this long at exit
FINISH_TIMEOUT_S = 120

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
#: queued in place of a frame: show the last frame again
_REPEAT_LAST = object()


_ENVIRONMENT = None


def _environment():
    """The environment for outside programs: Slicer's own library paths would
    break a system ffmpeg or pactl on Linux.

    Read from Slicer once, on the main thread (the Recorder asks at once), and
    kept: the finishing thread runs ffmpeg too, and must not call into Qt."""
    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        try:
            env = dict(slicer.util.startupEnvironment())
        except Exception:  # noqa: BLE001 - very old Slicer
            env = dict(os.environ)
        env["LC_ALL"] = "C"
        _ENVIRONMENT = env
    return dict(_ENVIRONMENT)


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
        fallback = os.path.join(
            os.path.expanduser("~"), RECORDINGS_FOLDER, recordlog.safe_name(os.path.basename(batchDir))
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


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
        """Pass on whatever audio is waiting.  Called before a segment switch,
        so audio captured before it does not land after it."""
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
    libx264; without, the video is MPEG-4 part 2.
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


class TheoraWriter:
    """VTK's own Ogg/Theora encoder, for when there is no ffmpeg.  It runs on
    the main thread, so the video is smaller and slower (THEORA_*)."""

    def __init__(self, path, width, height, fps):
        self.path = path
        self.frames = 0
        self.error = None
        self._image = vtk.vtkImageData()
        self._writer = vtk.vtkOggTheoraWriter()
        self._writer.SetInputData(self._image)
        self._writer.SetFileName(path)
        self._writer.SetRate(int(fps))
        self._writer.SetQuality(1)
        self._started = False

    broken = property(lambda self: self.error is not None)

    def push(self, frame, repeat=1):
        try:
            ctk.ctkVTKWidgetsUtils.qImageToVTKImageData(frame, self._image)
            self._image.Modified()
            if not self._started:
                self._writer.Start()
                self._started = True
            for _ in range(int(repeat)):
                self._writer.Write()
            self.frames += int(repeat)
        except Exception as error:  # noqa: BLE001
            self.error = "Theora encoding failed ({})".format(error)

    def close(self, timeout=None):
        """Finish the file; once.  The recorder closes it on the main thread
        (VTK) when it stops, so the finishing thread finds it done."""
        del timeout
        if self._started:
            self._started = False
            self._writer.End()
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
# segments
# --------------------------------------------------------------------------- #
class Segment:
    """What is recorded of one case between two switches: the files, the
    writers while it runs, and the event log."""

    def __init__(self, folder, stem, meta, start):
        self.folder = folder
        self.stem = stem
        self.start = start
        self.log = recordlog.EventLog(meta)
        self.wav = None
        self.audioOffset = None
        self.audioDevice = ""
        self.gaps = []
        self.writer = None
        self.videoKind = None
        self.videoOffset = None
        self.videoSize = None
        self.videoFps = None
        self.videoSource = ""
        self.pacer = None
        self.lastFrame = None
        self.hadAudio = False
        self.hadVideo = False
        self.ffmpeg = None
        self.end = None

    def path(self, suffix):
        return os.path.join(self.folder, self.stem + suffix)

    def at(self, now):
        return max(0.0, now - self.start)


_VOICE_CACHE = {}
_VOICE_LOCK = threading.Lock()


def _voice(rate):
    with _VOICE_LOCK:
        if rate not in _VOICE_CACHE:
            _VOICE_CACHE[rate] = wavfile.load_voice(VOICE_DIR, rate, recordlog.VOICE_CLIPS)
        return _VOICE_CACHE[rate]


def finishSegment(segment):
    """Turn a stopped segment into its files.  Runs in a thread: no Qt, VTK or
    scene access.  Returns the paths written.

    Each step stands on its own, so one that fails -- a file a virus scanner
    holds, a full disk -- costs only its own file, and the event log, the
    record a re-annotation follows, is written whatever happened."""
    written = []
    log = segment.log
    events = list(log.events)
    end = float(segment.end or 0.0)
    intervals = recordlog.lesion_intervals(events, end)
    meta = log.meta
    meta["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["duration"] = end
    try:
        audio = _finishAudio(segment, events, end, meta, written)
        video = _finishVideo(segment, audio, intervals, meta, written)
        _writeSubtitles(segment, intervals, video, meta, written)
    except Exception:  # noqa: BLE001 - the event log below still gets written
        logging.exception("GTReview: finishing recording %s failed", segment.stem)
    finally:
        meta["files"] = sorted(os.path.basename(p) for p in written)
        try:
            log.write_json(segment.path(".json"))
            written.append(segment.path(".json"))
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: writing the event log of %s failed", segment.stem)
    logging.info("GTReview: recording %s saved: %s", segment.stem, ", ".join(meta["files"]))
    return written


def _finishAudio(segment, events, end, meta, written):
    """Cue points, the narrated copy and the label track; the WAV the video
    should carry (the narrated one, else the microphone itself), or None."""
    if segment.wav is None or not os.path.isfile(segment.wav.path):
        return None
    micPath = segment.wav.path
    offset = float(segment.audioOffset or 0.0)
    written.append(micPath)
    audio = meta["audio"] = {
        "device": segment.audioDevice, "offset": offset, "duration": segment.wav.seconds,
        "gaps": segment.gaps, "microphone_file": os.path.basename(micPath),
    }
    marks = recordlog.cue_marks(events, offset)
    try:
        wavfile.append_cue_points(micPath, marks)
    except Exception as error:  # noqa: BLE001 - the recording itself is intact
        logging.warning("GTReview: no cue points in %s: %s", micPath, error)
        audio["cue_error"] = str(error)
    try:
        with open(segment.path("_labels.txt"), "w", encoding="utf-8") as handle:
            handle.write(recordlog.audacity_labels(events, end, offset))
        written.append(segment.path("_labels.txt"))
    except Exception as error:  # noqa: BLE001
        logging.warning("GTReview: no label track for %s: %s", segment.stem, error)
    try:
        rate, mic = wavfile.read_wav(micPath)
        audio.update(rate=rate, channels=int(mic.shape[1]))
        clips = _voice(rate)
        placements = []
        for at, names, _text in recordlog.announcement_schedule(events, offset):
            clip = wavfile.phrase(names, clips, rate)
            if clip is not None:
                placements.append((at, clip))
        narrated = segment.path(".wav")
        wavfile.write_wav(narrated, rate, wavfile.narrate(mic, rate, placements))
        written.append(narrated)
        audio.update(narrated_file=os.path.basename(narrated), announcements=len(placements))
    except Exception as error:  # noqa: BLE001 - the video still gets the reader's voice
        logging.exception("GTReview: narrating %s failed", segment.stem)
        audio["narration_error"] = str(error)
        return micPath
    try:
        wavfile.append_cue_points(narrated, marks)
    except Exception as error:  # noqa: BLE001
        logging.warning("GTReview: no cue points in %s: %s", narrated, error)
    return narrated


def _finishVideo(segment, audioPath, intervals, meta, written):
    """The finished video file; ``(offset, duration)`` of it, or None."""
    if segment.writer is None:
        return None
    ok = segment.writer.close()  # ffmpeg flushes here; Theora was closed on the main thread
    frames = segment.writer.frames
    offset = float(segment.videoOffset or 0.0)
    duration = frames / float(segment.videoFps or 1)
    video = meta["video"] = {
        "source": segment.videoSource or "all views", "encoder": segment.videoKind,
        "fps": segment.videoFps, "size": list(segment.videoSize or ()), "frames": frames,
        "offset": offset, "duration": duration,
    }
    if not ok:
        video["error"] = segment.writer.error
        logging.warning("GTReview: the video of %s: %s", segment.stem, segment.writer.error)
    produced = segment.writer.path
    if segment.videoKind == "mp4":
        final = segment.path(".mp4")
        try:
            if ok and _mux(segment, audioPath, intervals, offset, duration, final):
                maskio.remove_file(segment.writer.path)
            elif os.path.isfile(segment.writer.path):
                maskio.replace_file(segment.writer.path, final)  # unmuxed, but kept
        except Exception as error:  # noqa: BLE001 - whichever file is there is listed
            logging.warning("GTReview: tidying the video of %s: %s", segment.stem, error)
            video["file_error"] = str(error)
        produced = final if os.path.isfile(final) else segment.writer.path
    if not os.path.isfile(produced):
        return None
    video["file"] = os.path.basename(produced)
    written.append(produced)
    return offset, duration


def _writeSubtitles(segment, intervals, video, meta, written):
    """``<stem>.srt``, which players load for ``<stem>.mp4`` / ``.ogv`` and for
    ``<stem>.wav`` alike: on the video's clock when there is a video.  When the
    microphone started at another moment than the video, the audio's own copy
    is ``<stem>_audio.srt``."""
    audio = meta.get("audio")
    files = []
    if video is not None:
        files.append((".srt", video[0], video[1]))
    if audio is not None:
        if video is None:
            files.append((".srt", audio["offset"], audio["duration"]))
        elif abs(audio["offset"] - video[0]) > 0.25:
            files.append(("_audio.srt", audio["offset"], audio["duration"]))
    for suffix, offset, duration in files:
        try:
            with open(segment.path(suffix), "w", encoding="utf-8") as handle:
                handle.write(recordlog.srt(intervals, offset, duration))
            written.append(segment.path(suffix))
        except Exception as error:  # noqa: BLE001
            logging.warning("GTReview: no subtitles for %s: %s", segment.stem, error)


def _mux(segment, audioPath, intervals, videoOffset, duration, final):
    """The finished MP4: video, the narrated audio lined up with it, and one
    chapter per lesion.  False when ffmpeg refused."""
    chapters = segment.path("_chapters.txt")
    try:
        with open(chapters, "w", encoding="utf-8") as handle:
            handle.write(recordlog.ffmetadata(intervals, videoOffset, duration,
                                              title=segment.log.meta.get("case_id")))
        command = [segment.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", segment.writer.path]
        audio = audioPath is not None and os.path.isfile(audioPath) and segment.wav is not None
        if audio:
            audioOffset = float(segment.audioOffset or 0.0)
            overlap = (min(videoOffset + duration, audioOffset + segment.wav.seconds)
                       - max(videoOffset, audioOffset))
            # a microphone started only after the video stopped shares nothing with it
            audio = overlap > 0.1
        if audio:
            # the audio clipped to the video's stretch of the clock; never the
            # other way round: a microphone stopped early leaves the video whole
            shift = videoOffset - audioOffset
            command += ["-ss", "{:.3f}".format(shift)] if shift >= 0 else ["-itsoffset", "{:.3f}".format(-shift)]
            command += ["-t", "{:.3f}".format(duration + min(0.0, shift)), "-i", audioPath]
        command += ["-f", "ffmetadata", "-i", chapters, "-map", "0:v"]
        if audio:
            command += ["-map", "1:a", "-c:a", "aac", "-b:a", "96k"]
        meta = "2" if audio else "1"
        command += ["-map_metadata", meta, "-map_chapters", meta, "-c:v", "copy",
                    "-movflags", "+faststart", final]
        result = _run(command, timeout=FINISH_TIMEOUT_S)
        if result is None or result[0] != 0:
            logging.warning("GTReview: muxing %s failed: %s", segment.stem,
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
    """Microphone and video switches, one segment per case.

    The panel tells it what happens (:meth:`note`), when the case changes
    (:meth:`caseChanged`) and what the banner should say (:meth:`setBanner`).
    *context()* returns the meta of the case on screen, or None without one.
    Callbacks: *onChange()* after any switch, *onLevel(dBFS or None)*,
    *onTick(seconds)* twice a second while recording, *onWarning(text)*, and
    *onSegment()* right after a segment opens (to log the current lesion).
    """

    def __init__(self, context, onChange=None, onLevel=None, onTick=None, onWarning=None, onSegment=None):
        self._context = context
        self._onChange = onChange or (lambda: None)
        self._onTick = onTick or (lambda seconds: None)
        self._onWarning = onWarning or (lambda text: None)
        self._onSegment = onSegment or (lambda: None)
        self.mic = MicrophoneInput(self._onAudio, onLevel)
        self.audioOn = False
        self.videoOn = False
        self.testing = False
        self.microphone = ("", "", "")
        self.videoSource = ""
        self.segment = None
        self.banner = ("", "No lesion selected")
        self._frames = 0
        self._clock = None
        self._silenceWarned = False
        self._lastTick = None
        self._quiet = 0.0
        self._writeError = None
        self._finishing = []
        self.lastFolder = None
        _environment()  # read from Slicer here, on the main thread, once
        self._frameTimer = qt.QTimer()
        self._frameTimer.connect("timeout()", self._onFrameTick)
        self._clockTimer = qt.QTimer()
        self._clockTimer.setInterval(500)
        self._clockTimer.connect("timeout()", self._onClockTick)
        self._testTimer = qt.QTimer()
        self._testTimer.setSingleShot(True)
        self._testTimer.connect("timeout()", self.stopTest)

    # ---------------------------------------------------------------- state
    @property
    def recording(self):
        return self.audioOn or self.videoOn

    def elapsed(self):
        return self.segment.at(time.monotonic()) if self.segment is not None else 0.0

    def busyFinishing(self):
        self._finishing = [t for t in self._finishing if t.is_alive()]
        return len(self._finishing)

    def note(self, kind, **fields):
        """Log an event in the open segment; ignored when nothing records."""
        if self.segment is None:
            return None
        return self.segment.log.add(self.segment.at(time.monotonic()), kind, **fields)

    def setBanner(self, case, lesion):
        self.banner = (str(case or ""), str(lesion or ""))

    # ---------------------------------------------------------- microphone
    def startAudio(self, name, realm, label=""):
        if self.audioOn:
            return
        self.stopTest()
        segment = self._ensureSegment(rollIf="hadAudio")
        try:
            self.mic.start(name, realm, label)
        except Exception:
            self._dropIfIdle()
            raise
        try:
            segment.wav = wavfile.WavWriter(segment.path("_mic.wav"), self.mic.rate, self.mic.channels)
        except Exception:
            # an unwritable folder: close the microphone again, leave no segment
            self.mic.stop()
            self._dropIfIdle()
            raise
        self.microphone = (name, realm, label)
        self._writeError = None
        segment.audioOffset = segment.at(time.monotonic())
        segment.audioDevice = self.mic.label
        segment.hadAudio = True
        self._frames = 0
        self._clock = recordlog.AudioClock(self.mic.rate, time.monotonic())
        self._silenceWarned = False
        self._lastTick = None
        self._quiet = 0.0
        self.audioOn = True
        self.note("audio_started", device=self.mic.label, rate=self.mic.rate, channels=self.mic.channels)
        self._started()

    def stopAudio(self):
        """Stop the microphone; always completes, so a failing disk or device
        can never leave it recording."""
        if not self.audioOn:
            return
        try:
            self.mic.stop()
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: stopping the microphone failed")
        self.audioOn = False
        self.note("audio_stopped")
        self._closeWav(self.segment)
        self._stoppedOne()

    @staticmethod
    def _closeWav(segment):
        if segment is None or segment.wav is None or segment.wav.closed:
            return
        try:
            segment.wav.close()
        except OSError:
            logging.exception("GTReview: closing %s failed", segment.wav.path)

    def _onAudio(self, data):
        if self.testing or not self.audioOn or self._writeError is not None:
            return
        segment = self.segment
        if segment is None or segment.wav is None or segment.wav.closed:
            return
        if self.mic.gone():
            # Checked on every delivery, not only on the clock: what arrives
            # now may already be another microphone's (see MicrophoneInput.gone)
            self._microphoneGone()
            return
        try:
            self._frames += segment.wav.write(data)
        except OSError as error:
            self._audioWriteFailed(segment, error)
            return
        # Judged when audio arrives, not on a timer: WinMM delivers in blocks
        # of seconds, and between two blocks the file is always "behind".
        now = time.monotonic()
        missing = self._clock.observe(now, self._frames) if self._clock is not None else 0
        if missing:
            try:
                self._frames += segment.wav.write_silence(missing)
            except OSError as error:
                self._audioWriteFailed(segment, error)
                return
            seconds = missing / float(self.mic.rate)
            segment.gaps.append({"t": segment.at(now), "seconds": seconds})
            self.note("audio_gap", seconds=seconds)
            logging.warning("GTReview: %.1f s of microphone audio was lost; padded with silence", seconds)

    def _microphoneGone(self):
        """Stop the microphone that was unplugged, from the event loop; nothing
        it delivers from now on is kept."""
        if self._writeError is not None:
            return
        self._writeError = "gone"
        label = self.mic.label

        def stop():
            self.stopAudio()
            self._onWarning("The microphone {} was disconnected, so its recording was stopped; "
                            "what it recorded until then is kept.  (Left running, the system "
                            "would have carried on from another microphone.)".format(label))

        qt.QTimer.singleShot(0, stop)

    def _audioWriteFailed(self, segment, error):
        """The microphone file cannot be written (a full disk, a share gone).
        Close it as far as it goes and stop the microphone from the event loop,
        not from inside the delivery that failed."""
        self._writeError = error
        logging.error("GTReview: writing %s failed: %s", segment.wav.path, error)
        self._closeWav(segment)
        segment.log.add(segment.at(time.monotonic()), "audio_write_failed", detail=str(error))

        def stop():
            self.stopAudio()
            self._onWarning("The microphone recording could not be written ({}), so it was "
                            "stopped; what reached the disk is kept.".format(error))

        qt.QTimer.singleShot(0, stop)

    def startTest(self, name, realm, label="", seconds=10):
        """Open the microphone for *seconds* to watch its level; nothing is saved."""
        if self.audioOn or self.testing:
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
    def startVideo(self, source=""):
        if self.videoOn:
            return
        ffmpeg, _h264 = findFfmpeg()
        self._ensureSegment(rollIf="hadVideo")
        self.videoSource = source or ""
        self.segment.videoSource = self.videoSource
        self.segment.hadVideo = True
        self.videoOn = True
        fps = VIDEO_FPS if ffmpeg else THEORA_FPS
        self._frameTimer.setInterval(int(1000 / fps))
        self._frameTimer.start()
        self.note("video_started", source=self.videoSource or "all views",
                  encoder="ffmpeg" if ffmpeg else "theora")
        self._started()

    def stopVideo(self):
        if not self.videoOn:
            return
        self._frameTimer.stop()
        self.note("video_stopped")
        self.videoOn = False
        segment = self.segment
        if segment is not None and segment.writer is not None and segment.videoKind == "ogv":
            segment.writer.close()  # VTK: on the main thread
        self._stoppedOne()

    def setVideoSource(self, source):
        """Change what the video shows; takes effect from the next frame."""
        self.videoSource = source or ""
        if self.segment is not None and self.videoOn:
            self.segment.videoSource = self.videoSource
            self.note("video_source", source=self.videoSource or "all views")

    def _openVideo(self, segment, image, now):
        ffmpeg, h264 = findFfmpeg()
        if ffmpeg:
            segment.videoSize = frameSize(image, VIDEO_MAX_SIZE)
            segment.videoFps = VIDEO_FPS
            segment.writer = FfmpegWriter(ffmpeg, h264, segment.path("_video.mp4"),
                                          segment.videoSize[0], segment.videoSize[1], VIDEO_FPS)
            segment.videoKind = "mp4"
            segment.ffmpeg = ffmpeg
        else:
            segment.videoSize = frameSize(image, THEORA_MAX_SIZE)
            segment.videoFps = THEORA_FPS
            segment.writer = TheoraWriter(segment.path(".ogv"), segment.videoSize[0],
                                          segment.videoSize[1], THEORA_FPS)
            segment.videoKind = "ogv"
        segment.videoOffset = segment.at(now)
        segment.pacer = recordlog.FramePacer(segment.videoFps, now)

    def _onFrameTick(self):
        segment = self.segment
        if not self.videoOn or segment is None:
            return
        now = time.monotonic()
        try:
            if segment.writer is None:
                image = grabViews(self.videoSource)
                if image is None:
                    return
                self._openVideo(segment, image, now)
            due = segment.pacer.due(now)
            if due <= 0:
                return
            if segment.writer.broken:
                raise RuntimeError(segment.writer.error or "the video encoder stopped")
            # In the middle of a load, a recount or a save Slicer shows the wait
            # cursor and the views may be half updated: repeat the last picture.
            busy = qt.QApplication.overrideCursor() is not None and segment.lastFrame is not None
            image = None if busy else grabViews(self.videoSource)
            if image is None and segment.lastFrame is None:
                return
            if image is not None:
                if due > 1 and segment.lastFrame is not None:
                    # Slicer was frozen: what was on screen then is the old
                    # picture, and only the newest frame shows the new one
                    segment.writer.push(segment.lastFrame, due - 1)
                    segment.pacer.mark(due - 1)
                    due = 1
                canvas = composeFrame(image, segment.videoSize,
                                      "REC " + recordlog.format_clock(segment.at(now)), *self.banner)
                segment.lastFrame = bytes(canvas.constBits()) if segment.videoKind == "mp4" else canvas
            segment.writer.push(segment.lastFrame, due)
            segment.pacer.mark(due)
        except Exception as error:  # noqa: BLE001 - never let a frame take Slicer down
            logging.exception("GTReview: video frame failed")
            self.stopVideo()
            self._onWarning("Video recording stopped: {}".format(error))

    # ------------------------------------------------------------- segments
    def _ensureSegment(self, rollIf):
        """The open segment, or a new one for the case on screen.  A segment
        that already had this kind of recording is closed and a new one opened,
        so restarting a stream never needs a second file in one segment."""
        if self.segment is not None and getattr(self.segment, rollIf):
            self._roll()
        if self.segment is None:
            self.segment = self._openSegment()
        return self.segment

    def _openSegment(self):
        context = self._context()
        if not context:
            raise RuntimeError("Load a case first: recordings are kept per case.")
        folder = os.path.join(recordingsFolder(context["batch_dir"]), recordlog.safe_name(context["case_id"]))
        os.makedirs(folder, exist_ok=True)
        now = datetime.datetime.now()
        stem = recordlog.unique_stem(
            folder, recordlog.segment_stem(context["case_id"], now),
            (".json", ".wav", "_mic.wav", ".mp4", "_video.mp4", ".ogv"),
        )
        meta = dict(context)
        meta["started"] = now.isoformat(timespec="seconds")
        meta["reader"] = _reader()
        segment = Segment(folder, stem, meta, time.monotonic())
        self.lastFolder = folder
        self.segment = segment
        segment.log.add(0.0, "recording_started")
        self._onSegment()
        return segment

    def _roll(self):
        """Close the open segment and open the next one for the same case,
        carrying on whatever is recording."""
        self._closeSegment(reopen=True)

    def _dropIfIdle(self):
        """Forget a segment opened for a stream that then failed to start, when
        nothing else records into it: it holds no media, so no files."""
        segment = self.segment
        if segment is None or self.recording:
            return
        self.segment = None
        try:
            os.rmdir(segment.folder)  # only if this left it empty
        except OSError:
            pass

    def caseChanged(self):
        """The case on screen changed: its predecessor's segment is finished and,
        while anything records, a new one opens for this case."""
        if self.segment is None:
            return
        context = self._context()
        if context is None:
            self.stopAll()
            self._onWarning("Recording stopped: no case is open.")
            return
        self._closeSegment(reopen=self.recording)

    def _closeSegment(self, reopen):
        """Finish the open segment and, with *reopen*, start the next one for
        the case on screen.  Each step stands alone: a microphone or a disk
        that fails here never leaves a segment half closed."""
        segment = self.segment
        if segment is None:
            return
        now = time.monotonic()
        if self.audioOn:
            try:
                self.mic.drain()
            except Exception:  # noqa: BLE001
                logging.exception("GTReview: reading the last of the microphone failed")
        # audio captured before this moment that the system has not handed
        # over yet -- WinMM keeps a block of a second -- lands in the next file
        pending = max(0.0, self._clock.shortfall(now, self._frames)) if self._clock else 0.0
        self._closeWav(segment)
        if segment.writer is not None and segment.videoKind == "ogv":
            try:
                segment.writer.close()
            except Exception:  # noqa: BLE001
                logging.exception("GTReview: finishing the Theora video failed")
        segment.end = segment.at(now)
        segment.log.add(segment.end, "recording_ended")
        self.segment = None
        self._finish(segment)
        if not reopen:
            return
        try:
            fresh = self._openSegment()
            if self.audioOn and self._writeError is None:
                fresh.wav = wavfile.WavWriter(fresh.path("_mic.wav"), self.mic.rate, self.mic.channels)
        except Exception as error:  # noqa: BLE001
            self.stopAll()
            self._onWarning("Recording stopped: {}".format(error))
            return
        if fresh.wav is not None:
            # the file starts with that pending audio: before the segment does
            fresh.audioOffset = (now - pending) - fresh.start
            fresh.audioDevice = self.mic.label
            fresh.hadAudio = True
            fresh.log.add(0.0, "audio_started", device=self.mic.label, rate=self.mic.rate,
                          channels=self.mic.channels)
        if self.videoOn:
            fresh.videoSource = self.videoSource
            fresh.hadVideo = True
            fresh.log.add(0.0, "video_started", source=self.videoSource or "all views")
        self._onChange()

    def _finish(self, segment):
        thread = threading.Thread(target=self._finishQuietly, args=(segment,),
                                  name="GTReview finish {}".format(segment.stem), daemon=True)
        thread.start()
        self._finishing.append(thread)

    @staticmethod
    def _finishQuietly(segment):
        try:
            finishSegment(segment)
        except Exception:  # noqa: BLE001 - logged; the raw files stay
            logging.exception("GTReview: finishing recording %s failed", segment.stem)

    def _started(self):
        if not self._clockTimer.isActive():
            self._clockTimer.start()
        self._onChange()

    def _stoppedOne(self):
        if not self.recording:
            self._closeSegment(reopen=False)
            self._clockTimer.stop()
        self._onChange()

    def stopAll(self):
        """Stop everything and finish the files.  Never raises: it runs as a
        scene closes and as Slicer exits."""
        try:
            self.stopTest()
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: stopping the microphone test failed")
        if self.audioOn:
            try:
                self.mic.stop()
            except Exception:  # noqa: BLE001
                logging.exception("GTReview: stopping the microphone failed")
            self.audioOn = False
        if self.videoOn:
            self._frameTimer.stop()
            self.videoOn = False
        try:
            self._closeSegment(reopen=False)
        except Exception:  # noqa: BLE001
            logging.exception("GTReview: finishing the recording failed")
            self.segment = None
        self._clockTimer.stop()
        self._onChange()

    def shutdown(self, wait=FINISH_TIMEOUT_S):
        """Stop everything and wait (up to *wait* s) for the files to be written."""
        self.stopAll()
        deadline = time.monotonic() + float(wait)
        for thread in list(self._finishing):
            thread.join(max(0.0, deadline - time.monotonic()))
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
        if self.audioOn:
            self.mic.drain()
            self._quiet = self._quiet + 0.5 if running and now - (self.mic.lastData or now) > 0.5 else 0.0
            if self.mic.gone():
                self._microphoneGone()  # in case nothing was delivered since
            elif self.mic.failed() or self._quiet >= QUIET_SECONDS:
                self.stopAudio()
                self._onWarning("The microphone stopped delivering audio (unplugged?), so its "
                                "recording was stopped; what it recorded until then is kept.")
            elif (not self._silenceWarned and not self.mic.heardSomething
                    and now - (self.mic.started or now) > SILENCE_WARNING_SECONDS):
                self._silenceWarned = True
                hint = ("macOS keeps the microphone from Slicer unless Slicer may use it, "
                        "and stock Slicer may not." if sys.platform == "darwin" else
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
