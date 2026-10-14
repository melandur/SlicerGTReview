"""WAV files for GTReview's microphone recordings -- pure python (stdlib + numpy).

The microphone track is written as it is captured by :class:`WavWriter`, which
rewrites the two sizes in the header every couple of seconds: if Slicer dies
mid-recording, the file on disk is still a WAV that plays up to the last few
seconds.  Once a take is over, :func:`narrate` builds the recording meant for
listening, with a short chime and a spoken "lesion three" mixed in wherever
the reader moved to another lesion, and :func:`append_cue_points` adds a
marker per lesion change (ffmpeg, and players built on it such as mpv, show
them as chapters).

The voice comes from clips in ``Resources/GTReviewVoice`` -- whole phrases
("lesion 7") and single words -- strung together by :func:`phrase`; the
reader's voice is turned down while an announcement plays, never cut.  The
clips are 8-bit G.711 mu-law, which keeps a hundred phrases near a megabyte;
recordings are 16-bit PCM.

Nothing here imports ``slicer`` or ``qt``.
"""

import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: recordings are 16-bit PCM
SAMPLE_WIDTH = 2
#: WAVE format tags read here
_PCM, _MULAW, _EXTENSIBLE = 1, 7, 0xFFFE
_MULAW_BIAS, _MULAW_CLIP = 0x84, 32635


def encode_mulaw(samples: np.ndarray) -> np.ndarray:
    """G.711 mu-law bytes (uint8) of int16 *samples*."""
    x = np.asarray(samples).astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(x), _MULAW_CLIP) + _MULAW_BIAS
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def decode_mulaw(codes: np.ndarray) -> np.ndarray:
    """int16 samples of G.711 mu-law bytes."""
    u = ~np.asarray(codes).astype(np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    magnitude = ((((u & 0x0F) << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return np.where(u & 0x80, -magnitude, magnitude).astype(np.int16)


def _header(rate: int, channels: int, data_bytes: int) -> bytes:
    block = channels * SAMPLE_WIDTH
    return b"".join((
        b"RIFF", struct.pack("<I", 36 + data_bytes), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * block, block, 8 * SAMPLE_WIDTH),
        b"data", struct.pack("<I", data_bytes),
    ))


class WavWriter:
    """16-bit PCM WAV, appended to as audio arrives.

    The header's sizes are brought up to date every *patch_seconds* of audio
    and on :meth:`flush` / :meth:`close`.  Bytes that do not make a whole
    frame are held back until the rest of the frame arrives.
    """

    def __init__(self, path: str, rate: int, channels: int = 1, patch_seconds: float = 2.0):
        self.path = str(path)
        self.rate = int(rate)
        self.channels = int(channels)
        self.block = self.channels * SAMPLE_WIDTH
        self._patch_every = max(1, int(float(patch_seconds) * self.rate)) * self.block
        self._handle = open(self.path, "wb")
        self._handle.write(_header(self.rate, self.channels, 0))
        self._data_bytes = 0
        self._since_patch = 0
        self._partial = b""

    @property
    def frames(self) -> int:
        return self._data_bytes // self.block

    @property
    def seconds(self) -> float:
        return self.frames / float(self.rate)

    @property
    def closed(self) -> bool:
        return self._handle is None

    def write(self, data: bytes) -> int:
        """Append raw little-endian PCM; returns the whole frames written."""
        if self._handle is None:
            raise ValueError("write to a closed WavWriter")
        data = self._partial + bytes(data)
        whole = len(data) - len(data) % self.block
        self._partial = data[whole:]
        if whole:
            self._handle.write(data[:whole])
            self._data_bytes += whole
            self._since_patch += whole
            if self._since_patch >= self._patch_every:
                self.flush()
        return whole // self.block

    def write_silence(self, frames: int) -> int:
        frames = max(0, int(frames))
        if frames:
            self.write(b"\x00" * (frames * self.block))
        return frames

    def flush(self) -> None:
        if self._handle is None:
            return
        position = self._handle.tell()
        self._handle.seek(0)
        self._handle.write(_header(self.rate, self.channels, self._data_bytes))
        self._handle.seek(position)
        self._handle.flush()
        self._since_patch = 0

    def close(self) -> None:
        """Bring the header up to date and close; closed even when that fails
        (a full disk, a share gone), so a stop never has to close it twice."""
        if self._handle is None:
            return
        handle = self._handle
        try:
            self.flush()
        finally:
            self._handle = None
            handle.close()


def _chunks(blob: bytes):
    """``(id, offset, size)`` of every chunk after the RIFF header."""
    position = 12
    while position + 8 <= len(blob):
        chunk_id = blob[position:position + 4]
        size = struct.unpack("<I", blob[position + 4:position + 8])[0]
        yield chunk_id, position + 8, size
        position += 8 + size + (size & 1)


def read_wav(path: str) -> Tuple[int, np.ndarray]:
    """``(rate, samples)`` of a 16-bit PCM or 8-bit mu-law WAV.

    Samples are int16 ``(frames, channels)``.  A data chunk that claims more
    than the file holds -- a recording whose header was last patched before a
    crash -- is read up to what is there.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("{} is not a WAV file".format(path))
    rate = channels = width = None
    mulaw = False
    for chunk_id, offset, size in _chunks(blob):
        if chunk_id == b"fmt ":
            fmt, channels, rate, _byte_rate, _block, bits = struct.unpack(
                "<HHIIHH", blob[offset:offset + 16])
            if fmt in (_PCM, _EXTENSIBLE) and bits == 16:
                width = 2
            elif fmt == _MULAW and bits == 8:
                width, mulaw = 1, True
            else:
                raise ValueError("{}: only 16-bit PCM and 8-bit mu-law are supported".format(path))
        elif chunk_id == b"data":
            if rate is None:
                raise ValueError("{}: data before fmt".format(path))
            data = blob[offset:offset + size]
            usable = len(data) - len(data) % (channels * width)
            if mulaw:
                samples = decode_mulaw(np.frombuffer(data[:usable], dtype=np.uint8))
            else:
                samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.int16)
            return int(rate), samples.reshape(-1, channels)
    raise ValueError("{}: no audio data".format(path))


def write_wav(path: str, rate: int, samples: np.ndarray, mulaw: bool = False) -> None:
    """Write int16 *samples* (``(frames,)`` or ``(frames, channels)``) in one go.

    With *mulaw* the file is 8-bit G.711 mu-law (format 7, with the ``fact``
    chunk a non-PCM WAV needs), half the size of 16-bit PCM.
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    rate, channels = int(rate), samples.shape[1]
    with open(path, "wb") as handle:
        if not mulaw:
            data = np.ascontiguousarray(samples.astype("<i2", copy=False)).tobytes()
            handle.write(_header(rate, channels, len(data)))
            handle.write(data)
            return
        data = encode_mulaw(samples).tobytes()
        pad = b"\x00" if len(data) & 1 else b""
        body = b"".join((
            b"WAVE",
            b"fmt ", struct.pack("<IHHIIHHH", 18, _MULAW, channels, rate, rate * channels, channels, 8, 0),
            b"fact", struct.pack("<II", 4, samples.shape[0]),
            b"data", struct.pack("<I", len(data)), data, pad,
        ))
        handle.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def append_cue_points(path: str, marks: Sequence[Tuple[float, str]]) -> int:
    """Append a ``cue `` chunk and a ``LIST/adtl`` of labels to a closed WAV.

    *marks* are ``(seconds, text)``; points past the end of the audio are
    dropped.  Returns the number of points written.

    The chunks are appended in place and the RIFF size patched last: the file
    is the only copy of a recording, and rewriting it whole would leave it
    truncated if the write failed half way.
    """
    rate = channels = None
    data_frames = 0
    with open(path, "rb") as handle:
        if handle.read(12)[8:12] != b"WAVE":
            raise ValueError("{} is not a WAV file".format(path))
        length = handle.seek(0, os.SEEK_END)
        position = 12
        while position + 8 <= length:
            handle.seek(position)
            head = handle.read(16)
            chunk_id, size = head[:4], struct.unpack("<I", head[4:8])[0]
            if chunk_id == b"fmt ":
                _fmt, channels, rate = struct.unpack("<HHI", head[8:16])
            elif chunk_id == b"data" and channels:
                data_frames = min(size, length - position - 8) // (channels * SAMPLE_WIDTH)
            position += 8 + size + (size & 1)
    if rate is None:
        raise ValueError("{} has no fmt chunk".format(path))
    points = []
    for seconds, text in marks:
        frame = int(round(float(seconds) * rate))
        if 0 <= frame <= data_frames:
            points.append((frame, str(text)))
    if not points:
        return 0
    cue = struct.pack("<I", len(points))
    labels = b""
    for number, (frame, text) in enumerate(points, 1):
        cue += struct.pack("<II4sIII", number, frame, b"data", 0, 0, frame)
        body = struct.pack("<I", number) + text.encode("utf-8") + b"\x00"
        labels += b"labl" + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")
    extra = b"cue " + struct.pack("<I", len(cue)) + cue
    extra += b"LIST" + struct.pack("<I", 4 + len(labels)) + b"adtl" + labels
    if length & 1:
        extra = b"\x00" + extra  # chunks start on an even offset
    with open(path, "r+b") as handle:
        handle.seek(length)
        handle.write(extra)
        handle.flush()
        handle.seek(4)
        handle.write(struct.pack("<I", length + len(extra) - 8))
    return len(points)


def read_cue_points(path: str) -> List[Tuple[int, str]]:
    """``(frame, text)`` of the cue points :func:`append_cue_points` wrote."""
    with open(path, "rb") as handle:
        blob = handle.read()
    frames: Dict[int, int] = {}
    texts: Dict[int, str] = {}
    for chunk_id, offset, size in _chunks(blob):
        if chunk_id == b"cue ":
            count = struct.unpack("<I", blob[offset:offset + 4])[0]
            for index in range(count):
                start = offset + 4 + 24 * index
                number, _pos, _chunk, _cs, _bs, frame = struct.unpack("<II4sIII", blob[start:start + 24])
                frames[number] = frame
        elif chunk_id == b"LIST" and blob[offset:offset + 4] == b"adtl":
            inner = offset + 4
            while inner + 8 <= offset + size:
                sub_id = blob[inner:inner + 4]
                sub_size = struct.unpack("<I", blob[inner + 4:inner + 8])[0]
                if sub_id == b"labl":
                    number = struct.unpack("<I", blob[inner + 8:inner + 12])[0]
                    texts[number] = blob[inner + 12:inner + 8 + sub_size].split(b"\x00")[0].decode("utf-8")
                inner += 8 + sub_size + (sub_size & 1)
    return [(frames[n], texts.get(n, "")) for n in sorted(frames)]


# --------------------------------------------------------------------------- #
# announcements
# --------------------------------------------------------------------------- #
def resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited resampling of a short mono clip (FFT based)."""
    signal = np.asarray(signal, dtype=np.float32)
    if int(source_rate) == int(target_rate) or signal.size == 0:
        return signal.copy()
    length = max(1, int(round(signal.size * float(target_rate) / float(source_rate))))
    spectrum = np.fft.rfft(signal)
    # truncating the spectrum band-limits a downsample; zero-padding it
    # interpolates an upsample; irfft divides by the new length, hence the scale
    resized = np.zeros(length // 2 + 1, dtype=spectrum.dtype)
    keep = min(resized.size, spectrum.size)
    resized[:keep] = spectrum[:keep]
    out = np.fft.irfft(resized, n=length) * (float(length) / signal.size)
    return out.astype(np.float32)


def load_clip(path: str, rate: int) -> np.ndarray:
    """A voice clip as float32 mono in [-1, 1] at *rate*."""
    clip_rate, samples = read_wav(path)
    mono = samples.astype(np.float32).mean(axis=1) / 32768.0
    return resample(mono, clip_rate, rate)


def load_voice(folder: str, rate: int, names: Iterable[str]) -> Dict[str, np.ndarray]:
    """Clips *names* found in *folder* (``<name>.wav``); missing or unreadable
    ones are left out, so a partly re-recorded voice still works."""
    clips = {}
    for name in names:
        path = os.path.join(folder, "{}.wav".format(name))
        if os.path.isfile(path):
            try:
                clips[name] = load_clip(path, rate)
            except ValueError:
                continue
    return clips


def chime(rate: int, level: float = 0.18) -> np.ndarray:
    """Two soft rising notes, ~0.2 s: "listen, the lesion changed"."""
    pieces = []
    for frequency in (784.0, 1175.0):
        n = int(0.085 * rate)
        t = np.arange(n, dtype=np.float32) / rate
        note = np.sin(2 * np.pi * frequency * t) * np.hanning(n).astype(np.float32)
        pieces.append(note * level)
        pieces.append(np.zeros(int(0.015 * rate), dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def phrase(names: Sequence[str], clips: Dict[str, np.ndarray], rate: int,
           gap: float = 0.05, with_chime: bool = True) -> Optional[np.ndarray]:
    """The announcement made of clips *names*, or None when one is missing."""
    if any(name not in clips for name in names):
        return None
    pause = np.zeros(int(gap * rate), dtype=np.float32)
    parts: List[np.ndarray] = [chime(rate), np.zeros(int(0.06 * rate), dtype=np.float32)] if with_chime else []
    for index, name in enumerate(names):
        if index:
            parts.append(pause)
        parts.append(clips[name])
    return np.concatenate(parts).astype(np.float32)


def narrate(mic: np.ndarray, rate: int, placements: Sequence[Tuple[float, np.ndarray]],
            duck: float = 0.35, fade: float = 0.04, voice_level: float = 0.8) -> np.ndarray:
    """*mic* (int16, ``(frames, channels)``) with announcements mixed in.

    *placements* are ``(seconds, clip)``.  An announcement starts no earlier
    than the end of the one before, so a quick succession never overlaps; one
    held back until the next is due is left out, so long phrases ("lesion
    three hundred seventy-seven") never leave the voice running behind the
    reader; one that would start past the end of the recording is dropped.
    The reader's voice is scaled by *duck* under an announcement, with *fade*
    second ramps, so both stay audible.
    """
    mic = np.asarray(mic)
    if mic.ndim == 1:
        mic = mic[:, None]
    # Only the stretches under an announcement are touched, in floating point;
    # an hour of microphone is never copied to float as a whole.
    out = mic.astype(np.int16, copy=True)
    frames = out.shape[0]
    ramp = max(1, int(fade * rate))
    free_from = 0
    placements = sorted(placements, key=lambda p: p[0])
    for index, (seconds, clip) in enumerate(placements):
        start = max(int(round(float(seconds) * rate)), free_from)
        if start >= frames or clip is None or clip.size == 0:
            continue
        if index + 1 < len(placements) and start >= int(round(float(placements[index + 1][0]) * rate)):
            continue  # the reader has moved on: the next one is due already
        stop = min(frames, start + clip.size)
        lo, hi = max(0, start - ramp), min(frames, stop + ramp)
        envelope = np.ones(hi - lo, dtype=np.float32)
        envelope[start - lo:stop - lo] = duck
        if start > lo:
            envelope[:start - lo] = np.linspace(1.0, duck, start - lo, endpoint=False)
        if hi > stop:
            envelope[stop - lo:] = np.linspace(duck, 1.0, hi - stop)
        stretch = out[lo:hi].astype(np.float32) * envelope[:, None]
        stretch[start - lo:stop - lo] += (clip[:stop - start] * voice_level * 32767.0)[:, None]
        out[lo:hi] = np.clip(np.round(stretch), -32768, 32767).astype(np.int16)
        free_from = stop + int(0.15 * rate)
    return out


# --------------------------------------------------------------------------- #
# the microphone test
# --------------------------------------------------------------------------- #
#: a voice should come in about here (dBFS of its loud stretches) ...
VOICE_TARGET_DB = -18.0
#: ... and below this it is too quiet to stand out against the room
VOICE_QUIET_DB = -30.0
#: a loudest moment below this means nothing was heard
SILENT_PEAK_DB = -60.0
#: a sample this loud is clipped; more than CLIPPED_SHARE of them is too loud
CLIP_LEVEL, CLIPPED_SHARE = 32000, 0.001
#: what a suggested boost leaves between the loudest moment and full scale
BOOST_HEADROOM_DB = 3.0


def _dbfs(value: float) -> float:
    return 20.0 * float(np.log10(max(float(value), 1.0) / 32768.0))


def level_report(samples: np.ndarray, rate: int, boost_steps: Sequence[int] = (6, 12, 18, 24)) -> Dict:
    """How loud a microphone test came in, and what would help.

    *samples* are int16, ``(frames,)`` or ``(frames, channels)``.  The voice
    is measured over its loud stretches -- the 90th percentile of 50 ms
    windows -- so the pauses between words do not count against it.

    Returns ``{"peak_db", "voice_db", "clipped", "verdict", "boost_db"}``:
    *verdict* is ``"silent"`` (nothing heard), ``"loud"`` (clipped),
    ``"quiet"`` or ``"good"``; for ``"quiet"``, *boost_db* is the largest of
    *boost_steps* that brings the voice no higher than VOICE_TARGET_DB and
    leaves BOOST_HEADROOM_DB above the loudest moment -- 0 when even the
    smallest would clip it.
    """
    x = np.asarray(samples).astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    window = max(1, int(rate * 0.05))
    stretches = x.size // window
    if stretches:
        rms = np.sqrt(np.mean(np.square(x[: stretches * window].reshape(stretches, window)), axis=1))
        voice = float(np.percentile(rms, 90))
    else:
        voice = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    clipped = float(np.mean(np.abs(x) >= CLIP_LEVEL)) if x.size else 0.0
    report = {"peak_db": round(_dbfs(peak), 1), "voice_db": round(_dbfs(voice), 1),
              "clipped": round(clipped, 4), "verdict": "good", "boost_db": 0}
    if report["peak_db"] < SILENT_PEAK_DB:
        report["verdict"] = "silent"
    elif clipped > CLIPPED_SHARE:
        report["verdict"] = "loud"
    elif report["voice_db"] < VOICE_QUIET_DB:
        report["verdict"] = "quiet"
        room = min(VOICE_TARGET_DB - report["voice_db"], -BOOST_HEADROOM_DB - report["peak_db"])
        report["boost_db"] = max([step for step in boost_steps if step <= room] or [0])
    return report


def convert_pcm(data: bytes, rate: int, channels: int, to_rate: int, to_channels: int) -> bytes:
    """16-bit PCM *data* as *to_rate* and *to_channels*, for an output that
    plays nothing else: mixed down, resampled, copied to every channel."""
    channels = max(1, int(channels))
    frames = len(data) // (2 * channels)
    x = np.frombuffer(data[: frames * 2 * channels], dtype="<i2").reshape(frames, channels)
    mono = x.astype(np.float32).mean(axis=1)
    if int(to_rate) != int(rate):
        mono = resample(mono, rate, to_rate)
    out = np.repeat(mono[:, None], max(1, int(to_channels)), axis=1)
    return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()
