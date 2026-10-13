"""What a microphone really hands Slicer (CI only).

Run inside Slicer with GTReview on the module path.  Opens the virtual
microphone whose name holds PROBE_DEVICE (BlackHole on macOS, VB-CABLE on
Windows), which the workflow plays a 440 Hz tone into, through GTReview's own
MicrophoneInput for a few seconds, and writes how long and how loud it was,
and its pitch, to PROBE_REPORT as JSON.  Silence is what macOS hands an app
it keeps from the microphone.  When GTReview cannot open it, every matching
device of every Qt back end is tried in its own format as well, and Qt's
warnings are kept.
"""
import json
import os
import time
import traceback

import numpy as np
import qt
import slicer

SECONDS = float(os.environ.get("PROBE_SECONDS", "4"))
report = {}
code = 1


def pump(seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        slicer.app.processEvents()
        time.sleep(0.01)


def loudness(data, rate, channels):
    samples = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2").astype(np.float64)
    channels = max(1, int(channels))
    samples = samples[: samples.size - samples.size % channels].reshape(-1, channels)[:, 0]
    out = {"seconds": round(samples.size / float(rate), 2),
           "rms": round(float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0, 1)}
    if samples.size > 1000:
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
        out["dominant_hz"] = round(float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(np.argmax(spectrum))]), 1)
    out["heard"] = out["rms"] > 100.0
    return out


def qtWarnings(last=15):
    """The newest entries of Slicer's error log: where Qt says why."""
    model = slicer.app.errorLogModel()
    rows = model.rowCount()
    return [" | ".join(str(model.index(row, column).data()) for column in range(model.columnCount()))
            for row in range(max(0, rows - last), rows)]


def raw(info):
    """Open *info* in its own preferred format with plain Qt, for comparison."""
    fmt = info.preferredFormat()
    entry = {"name": info.deviceName(), "realm": info.realm(),
             "format": [fmt.sampleRate(), fmt.channelCount(), fmt.sampleSize(), fmt.codec(), int(fmt.sampleType())]}
    if fmt.sampleSize() != 16 or fmt.sampleType() != qt.QAudioFormat.SignedInt:
        fmt.setSampleSize(16)
        fmt.setSampleType(qt.QAudioFormat.SignedInt)
        entry["format_used"] = [fmt.sampleRate(), fmt.channelCount(), 16]
    audio = qt.QAudioInput(info, fmt)
    io = audio.start()
    entry["open_error"] = int(audio.error())
    if io is not None and audio.error() == qt.QAudio.NoError:
        chunks = []
        deadline = time.time() + SECONDS
        while time.time() < deadline:
            slicer.app.processEvents()
            chunks.append(io.readAll().data())
            time.sleep(0.01)
        entry.update(loudness(b"".join(chunks), fmt.sampleRate(), fmt.channelCount()))
    audio.stop()
    return entry


try:
    from GTReviewLib import recording

    report["problem"] = recording.macMicrophoneProblem()
    report["signing"] = recording.macSigningProblem()
    devices = recording.inputDevices()
    report["inputs"] = ["{} [{}]".format(d["label"], d["realm"]) for d in devices]
    wanted = os.environ.get("PROBE_DEVICE", "blackhole").lower()
    target = next((d for d in devices if wanted in (d["name"] + d["label"]).lower()), None)
    if target is None:
        raise RuntimeError("Qt lists no input named like {!r}".format(wanted))
    chunks = []
    microphone = recording.MicrophoneInput(lambda data: chunks.append(bytes(data)))
    try:
        microphone.start(target["name"], target["realm"], target["label"], hold=False)
    except Exception as error:  # noqa: BLE001 - reported, then the raw tries below
        report["gtreview_error"] = str(error)
        report["qt_warnings"] = qtWarnings()
        report["raw"] = [raw(info) for info in qt.QAudioDeviceInfo.availableDevices(qt.QAudio.AudioInput)
                         if wanted in info.deviceName().lower()]
    else:
        report.update(device=microphone.label, rate=microphone.rate, channels=microphone.channels)
        pump(SECONDS)
        microphone.stop()
        report.update(loudness(b"".join(chunks), microphone.rate, microphone.channels))
    code = 0
except Exception:  # noqa: BLE001 - reported
    report["error"] = traceback.format_exc()
finally:
    with open(os.environ["PROBE_REPORT"], "w") as handle:
        json.dump(report, handle, indent=1)
    slicer.util.exit(code)
