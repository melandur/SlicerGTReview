"""What a microphone really hands Slicer (CI only).

Run inside Slicer with GTReview on the module path.  Opens the virtual
microphone whose name holds PROBE_DEVICE (BlackHole on macOS, VB-CABLE on
Windows), which the workflow plays a 440 Hz tone into, through GTReview's own
MicrophoneInput for a few seconds, and writes how long and how loud it was,
and its pitch, to PROBE_REPORT as JSON.  Silence is what macOS hands an app
it keeps from the microphone.
"""
import json
import os
import time
import traceback

import numpy as np
import slicer

report = {}
code = 1
try:
    from GTReviewLib import recording

    report["problem"] = recording.macMicrophoneProblem()
    devices = recording.inputDevices()
    report["inputs"] = ["{} [{}]".format(d["label"], d["realm"]) for d in devices]
    wanted = os.environ.get("PROBE_DEVICE", "blackhole").lower()
    target = next((d for d in devices if wanted in (d["name"] + d["label"]).lower()), None)
    if target is None:
        raise RuntimeError("Qt lists no input named like {!r}".format(wanted))
    chunks = []
    microphone = recording.MicrophoneInput(lambda data: chunks.append(bytes(data)))
    microphone.start(target["name"], target["realm"], target["label"], hold=False)
    report.update(device=microphone.label, rate=microphone.rate, channels=microphone.channels)
    deadline = time.time() + float(os.environ.get("PROBE_SECONDS", "4"))
    while time.time() < deadline:
        slicer.app.processEvents()
        time.sleep(0.01)
    microphone.stop()
    samples = np.frombuffer(b"".join(chunks), dtype="<i2").astype(np.float64)
    channels = max(1, int(microphone.channels))
    samples = samples[: samples.size - samples.size % channels].reshape(-1, channels)[:, 0]
    report["seconds"] = round(samples.size / float(microphone.rate), 2)
    report["rms"] = round(float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0, 1)
    if samples.size > 1000:
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
        report["dominant_hz"] = round(float(np.fft.rfftfreq(samples.size, 1.0 / microphone.rate)[int(np.argmax(spectrum))]), 1)
    report["heard"] = report["rms"] > 100.0
    code = 0
except Exception:  # noqa: BLE001 - reported
    report["error"] = traceback.format_exc()
finally:
    with open(os.environ["PROBE_REPORT"], "w") as handle:
        json.dump(report, handle, indent=1)
    slicer.util.exit(code)
