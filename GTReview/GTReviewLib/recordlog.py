"""What a GTReview recording says happened, and when -- pure python.

A recording of a review is only useful later if whoever listens to it, or
re-annotates from it, can tell which lesion the reader was on at every moment.
The lesion number in the table is no help on its own: it is a rank by size,
handed out again on every recount, so "lesion 4" at minute three and "lesion 4"
at minute ten can be two different lesions, and neither has to match the
numbering of the saved mask.  Every lesion event therefore carries a seed
voxel inside the lesion as well: the panel already uses that voxel to keep the
selection and the Done ticks across renumbering, and because the reviewed mask
keeps the source voxel grid, the lesion it points at can be found again in the
original mask and in the reviewed one.

This module turns the events into what goes next to the media files: the JSON
event log, subtitles (SRT), an Audacity label track, ffmpeg chapter metadata,
WAV cue points and the schedule of spoken announcements ("lesion three"),
plus the small clocks that keep audio and video in real time.

Nothing here imports ``slicer``, ``qt`` or ``numpy``.
"""

import datetime
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: written into every event log, so a reader can tell the layout apart later
FORMAT_NAME = "GTReview recording"
FORMAT_VERSION = 1

# --------------------------------------------------------------------------- #
# event types
# --------------------------------------------------------------------------- #
#: another lesion became the active one
LESION_SELECTED = "lesion_selected"
#: a lesion painted in "New lesion" mode was numbered and selected
LESION_ADOPTED = "lesion_adopted"
#: the active lesion is the same one, but a recount gave it another number
LESION_RENUMBERED = "lesion_renumbered"
#: no lesion is active any more
LESION_CLEARED = "lesion_cleared"
#: the reader clicked the active lesion again (jump and flash once more)
LESION_REVISITED = "lesion_revisited"

#: the events that change what the listener should be told
LESION_STATE_EVENTS = (LESION_SELECTED, LESION_ADOPTED, LESION_RENUMBERED, LESION_CLEARED)
#: the events that are spoken into the narrated audio
ANNOUNCED_EVENTS = (LESION_SELECTED, LESION_ADOPTED, LESION_RENUMBERED)

# --------------------------------------------------------------------------- #
# spoken numbers
# --------------------------------------------------------------------------- #
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

#: single words: the "new" / "now" in front of a phrase, and the parts of
#: numbers too large for a phrase of their own
VOICE_WORDS = ("lesion", "new", "now") + _ONES + tuple(t for t in _TENS if t) + ("hundred",)
#: "lesion 1" to "lesion 99" are spoken in one breath, from one clip each:
#: words strung together one by one were misheard far more often (a speech
#: recogniser got 4 of 7 numbers right from word clips, every one from phrases)
PHRASE_NUMBERS = range(1, 100)


def phrase_clip(number: int) -> str:
    """Clip name of the whole phrase "lesion <number>"."""
    return "lesion-{}".format(int(number))


#: every clip in Resources/GTReviewVoice, as ``<name>.wav``
VOICE_CLIPS = VOICE_WORDS + tuple(phrase_clip(n) for n in PHRASE_NUMBERS)


def number_words(number: int) -> List[str]:
    """English words for a non-negative integer, one clip per word.

    ``23 -> ["twenty", "three"]``, ``105 -> ["one", "hundred", "five"]``.  From
    1000 on the digits are read one by one, which is unambiguous and needs no
    more clips.
    """
    number = int(number)
    if number < 0:
        raise ValueError("no words for a negative number: {}".format(number))
    if number < 20:
        return [_ONES[number]]
    if number < 100:
        tens, ones = divmod(number, 10)
        return [_TENS[tens]] + ([_ONES[ones]] if ones else [])
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return [_ONES[hundreds], "hundred"] + (number_words(rest) if rest else [])
    return [_ONES[int(digit)] for digit in str(number)]


def _lead(kind: str) -> List[str]:
    if kind == LESION_ADOPTED:
        return ["new"]
    if kind == LESION_RENUMBERED:
        return ["now"]
    return []


def announcement_words(kind: str, number: int) -> List[str]:
    """The words spoken for a lesion event: ``lesion three``, ``new lesion
    three``, ``now lesion two`` (the active lesion was renumbered)."""
    return _lead(kind) + ["lesion"] + number_words(number)


def announcement_clips(kind: str, number: int) -> List[str]:
    """The clips that speak :func:`announcement_words`, in order."""
    number = int(number)
    if number in PHRASE_NUMBERS:
        return _lead(kind) + [phrase_clip(number)]
    return _lead(kind) + ["lesion"] + number_words(number)


# --------------------------------------------------------------------------- #
# what changed in the selection
# --------------------------------------------------------------------------- #
def selection_change(
    previous: Optional[Dict],
    current: Optional[Dict],
    previous_seed_owner: Optional[int],
    adopted: bool = False,
) -> Optional[str]:
    """The lesion event, if any, between two snapshots of the selection.

    *previous* and *current* are ``{"seed": (i, j, k), "number": n}`` or None
    for no selection.  *previous_seed_owner* is the number of the lesion that
    owns the previous seed voxel NOW (None when no lesion does): when it is the
    lesion selected now, the selection never moved and only its number may
    have.  *adopted* marks a selection made by "New lesion" adopting what was
    just painted.
    """
    if previous is None and current is None:
        return None
    if current is None:
        return LESION_CLEARED
    if previous is None:
        return LESION_ADOPTED if adopted else LESION_SELECTED
    same = previous_seed_owner is not None and int(previous_seed_owner) == int(current["number"])
    if not same:
        return LESION_ADOPTED if adopted else LESION_SELECTED
    if int(previous["number"]) != int(current["number"]):
        return LESION_RENUMBERED
    return None


# --------------------------------------------------------------------------- #
# the event log
# --------------------------------------------------------------------------- #
def _rounded(value):
    """Floats rounded for the JSON file; containers walked; the rest as is."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


class EventLog:
    """Timestamped events of one recording segment (one case).

    ``t`` is seconds since the segment started; :attr:`meta` says where each
    media file starts on that clock (``audio.offset``, ``video.offset``).
    """

    def __init__(self, meta: Optional[Dict] = None):
        self.meta: Dict = dict(meta or {})
        self.events: List[Dict] = []

    def add(self, t: float, kind: str, **fields) -> Dict:
        event = {"t": max(0.0, float(t)), "type": str(kind)}
        event.update(fields)
        self.events.append(event)
        return event

    def to_dict(self) -> Dict:
        document = {"format": FORMAT_NAME, "version": FORMAT_VERSION}
        document.update(_rounded(self.meta))
        document["events"] = [_rounded(event) for event in self.events]
        return document

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1, ensure_ascii=False)
            handle.write("\n")


def lesion_text(event: Dict) -> str:
    """One line for a lesion state: subtitles, labels, chapters, banner."""
    kind = event.get("type")
    if kind == LESION_CLEARED or event.get("number") is None:
        return "No lesion selected"
    text = "Lesion #{}".format(event["number"])
    if event.get("count"):
        text += " of {}".format(event["count"])
    if event.get("label_name"):
        text += " - {}".format(event["label_name"])
    if event.get("voxels") is not None:
        text += " ({} voxels)".format(event["voxels"])
    if kind == LESION_ADOPTED:
        text += " - new"
    elif kind == LESION_RENUMBERED and event.get("previous_number") is not None:
        text += " - was #{}".format(event["previous_number"])
    return text


def event_text(event: Dict) -> str:
    """A short description of any event, for the Audacity label track."""
    kind = event.get("type")
    if kind in LESION_STATE_EVENTS:
        return lesion_text(event)
    if kind == LESION_REVISITED:
        return "Lesion #{} again".format(event.get("number"))
    if kind == "lesion_deleted":
        return "Lesion #{} deleted".format(event.get("number"))
    if kind == "audio_gap":
        return "Audio lost: {:.1f} s of silence inserted".format(float(event.get("seconds", 0.0)))
    words = str(kind).replace("_", " ")
    detail = event.get("detail")
    return "{}: {}".format(words, detail) if detail else words.capitalize()


def lesion_intervals(events: Sequence[Dict], end: float) -> List[Tuple[float, float, str]]:
    """``(start, end, text)`` for each stretch with one lesion state, on the
    segment clock, from 0 to *end*.  The stretch before the first lesion event
    reads "No lesion selected"."""
    marks = [(0.0, "No lesion selected")]
    for event in events:
        if event.get("type") in LESION_STATE_EVENTS:
            marks.append((float(event["t"]), lesion_text(event)))
    out = []
    for index, (start, text) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else float(end)
        if stop > start:
            out.append((start, stop, text))
    return out


def _shifted(intervals, offset, duration=None):
    """Intervals moved onto a media file's clock and clipped to it."""
    out = []
    for start, stop, text in intervals:
        start, stop = start - offset, stop - offset
        if duration is not None:
            stop = min(stop, duration)
        start = max(0.0, start)
        if stop > start:
            out.append((start, stop, text))
    return out


def format_srt_time(seconds: float) -> str:
    millis = int(round(max(0.0, seconds) * 1000.0))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, secs, millis)


def format_clock(seconds: float) -> str:
    """``H:MM:SS`` under ten hours, for the banner and the panel."""
    seconds = int(max(0.0, seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def srt(intervals: Sequence[Tuple[float, float, str]], offset: float = 0.0,
        duration: Optional[float] = None) -> str:
    """SubRip subtitles for a media file starting *offset* s into the segment."""
    blocks = []
    for number, (start, stop, text) in enumerate(_shifted(intervals, offset, duration), 1):
        blocks.append("{}\n{} --> {}\n{}\n".format(
            number, format_srt_time(start), format_srt_time(stop), text))
    return "\n".join(blocks)


def audacity_labels(events: Sequence[Dict], end: float, offset: float = 0.0) -> str:
    """Audacity label track: one region per lesion state, a point per other event.

    Tab-separated ``start end text`` lines in seconds; File > Import > Labels.
    """
    lines = []
    for start, stop, text in _shifted(lesion_intervals(events, end), offset):
        lines.append((start, stop, text))
    for event in events:
        if event.get("type") in LESION_STATE_EVENTS:
            continue
        at = float(event["t"]) - offset
        if at >= 0:
            lines.append((at, at, event_text(event)))
    lines.sort(key=lambda line: (line[0], line[1]))
    return "".join("{:.6f}\t{:.6f}\t{}\n".format(a, b, text) for a, b, text in lines)


def ffmetadata(intervals: Sequence[Tuple[float, float, str]], offset: float,
               duration: float, title: Optional[str] = None) -> str:
    """ffmpeg FFMETADATA1 with one chapter per lesion stretch (``-map_chapters``)."""

    def escape(text):
        return re.sub(r"([=;#\\\n])", r"\\\1", str(text))

    lines = [";FFMETADATA1"]
    if title:
        lines.append("title=" + escape(title))
    for start, stop, text in _shifted(intervals, offset, duration):
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            "START={}".format(int(round(start * 1000))),
            "END={}".format(int(round(stop * 1000))),
            "title=" + escape(text),
        ]
    return "\n".join(lines) + "\n"


def cue_marks(events: Sequence[Dict], offset: float = 0.0) -> List[Tuple[float, str]]:
    """``(seconds, text)`` WAV cue points: one per lesion state change."""
    out = []
    for event in events:
        if event.get("type") in LESION_STATE_EVENTS:
            at = float(event["t"]) - offset
            if at >= 0:
                out.append((at, lesion_text(event)))
    return out


def announcement_schedule(events: Sequence[Dict], offset: float = 0.0,
                          settle: float = 0.6) -> List[Tuple[float, List[str], str]]:
    """``(seconds, clips, text)`` to speak into an audio file starting at *offset*.

    A reader paging through the list with the arrow keys produces a burst of
    selections; announcing each one would stack the voice up seconds behind
    the screen.  An announcement whose next announceable event follows within
    *settle* seconds is skipped -- only where the reader stops is spoken.
    """
    spoken = [e for e in events if e.get("type") in ANNOUNCED_EVENTS and e.get("number") is not None]
    out = []
    for index, event in enumerate(spoken):
        if index + 1 < len(spoken) and float(spoken[index + 1]["t"]) - float(event["t"]) < settle:
            continue
        at = float(event["t"]) - offset
        if at < 0:
            continue
        number = int(event["number"])
        out.append((at, announcement_clips(event["type"], number),
                    " ".join(announcement_words(event["type"], number))))
    return out


# --------------------------------------------------------------------------- #
# file names
# --------------------------------------------------------------------------- #
def safe_name(text: str) -> str:
    """*text* with anything a file system may refuse replaced by ``_``."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return cleaned or "case"


def segment_stem(case_id: str, when: datetime.datetime) -> str:
    """``<case_id>_<YYYYMMDD-HHMMSS>``, the stem every file of a segment shares."""
    return "{}_{}".format(safe_name(case_id), when.strftime("%Y%m%d-%H%M%S"))


def unique_stem(folder: str, stem: str, suffixes: Iterable[str]) -> str:
    """*stem*, or *stem* with ``-2``, ``-3``... when a file of it already exists."""
    suffixes = list(suffixes)
    candidate, counter = stem, 1
    while any(os.path.exists(os.path.join(folder, candidate + s)) for s in suffixes):
        counter += 1
        candidate = "{}-{}".format(stem, counter)
    return candidate


# --------------------------------------------------------------------------- #
# microphones
# --------------------------------------------------------------------------- #
def friendly_device_name(name: str) -> str:
    """A readable label for a raw PulseAudio / PipeWire source name.

    ``alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source`` reads
    ``Generic USB Audio - Mic``.  Names that are not of that shape are kept.
    """
    match = re.match(r"^alsa_input\.(?:[a-z]+-)?(.+?)\.([^.]+)$", name)
    if not match:
        return name
    device, profile = match.group(1), match.group(2)
    device = re.sub(r"(_[0-9A-Fa-f]{6,})", "", device)          # serial numbers
    device = re.sub(r"-\d{2}$", "", device)                        # interface index
    device = re.sub(r"_+", " ", device).strip(" -")
    profile = re.sub(r"^HiFi__|__(source|sink)$", "", profile)
    profile = re.sub(r"_+", " ", profile).strip()
    return "{} - {}".format(device, profile) if profile else device


def input_device_choices(
    devices: Sequence[Tuple[str, str]],
    descriptions: Optional[Dict[str, str]] = None,
    default_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    """The microphones worth offering, from ``(name, realm)`` pairs as Qt lists them.

    * ``.monitor`` sources record what the speakers play, not a microphone.
    * On Linux, Qt lists every card again under ALSA aliases (realm ``alsa``)
      next to the sound server's own sources; with a sound server, the aliases
      go.  Without one, only the plain aliases stay.
    * On Windows, Qt 5 lists every device a second time under realm
      ``wasapi``; those go while the default (WinMM) list has entries.
    * Exact duplicates go.

    Returns ``[{"name", "realm", "label"}]``, the system default first, then
    by label.  *descriptions* maps raw names to readable ones (PulseAudio's
    ``pactl``); *default_name* marks the system default.
    """
    descriptions = descriptions or {}
    pairs = [(str(n), str(r)) for n, r in devices if str(n).strip()]
    pairs = [(n, r) for n, r in pairs if not n.endswith(".monitor")]
    if any(r != "alsa" for _, r in pairs):
        pairs = [(n, r) for n, r in pairs if r != "alsa"]
    else:
        pairs = [(n, r) for n, r in pairs if not re.match(r"^(dmix|dsnoop|hw|plughw)(:|$)", n)]
    if any(r != "wasapi" for _, r in pairs):
        pairs = [(n, r) for n, r in pairs if r != "wasapi"]
    seen, choices = set(), []
    for name, realm in pairs:
        if (name, realm) in seen:
            continue
        seen.add((name, realm))
        label = descriptions.get(name) or friendly_device_name(name)
        choices.append({"name": name, "realm": realm, "label": label})
    choices.sort(key=lambda c: (c["name"] != default_name, c["label"].lower()))
    for choice in choices:
        if choice["name"] == default_name:
            choice["label"] += "  (system default)"
    return choices


# --------------------------------------------------------------------------- #
# clocks
# --------------------------------------------------------------------------- #
class FramePacer:
    """How many video frames are due so the video keeps real time.

    Frame *n* (from 0) shows the moment ``start + n / fps``.  When the main
    thread was busy and missed ticks, the next tick owes several frames; they
    repeat the last picture, so the video runs as long as the wall clock did
    and stays in step with the audio and the event log.
    """

    def __init__(self, fps: float, start: float):
        if fps <= 0:
            raise ValueError("fps must be positive, got {}".format(fps))
        self.fps = float(fps)
        self.start = float(start)
        self.emitted = 0

    def due(self, now: float) -> int:
        target = int((float(now) - self.start) * self.fps) + 1
        return max(0, target - self.emitted)

    def mark(self, count: int) -> None:
        self.emitted += int(count)


class AudioClock:
    """Notices audio that the device dropped, so the file can be padded.

    Qt moves captured audio on Slicer's main thread; while that thread is
    busy the sound system holds the samples, and how much it holds depends
    on the platform (minutes with PulseAudio / PipeWire, a fraction of a
    second with plain ALSA, CoreAudio, or WinMM at its default buffer).  Held
    samples arrive late in a burst; dropped ones never arrive.

    The two look the same while it happens, so this only acts on a shortfall
    that stays put: when the file has been short of the wall clock by more
    than *threshold* seconds for *settle* seconds, by an amount that moved by
    less than *tolerance* all along, the missing time is gone for good and
    :meth:`observe` returns the frames of silence that put the rest of the
    recording back in step.  A catch-up burst shrinks the shortfall, which
    starts the wait over, so it never qualifies.
    """

    def __init__(self, rate: int, start: float, threshold: float = 0.75,
                 settle: float = 1.5, tolerance: float = 0.1, margin: float = 0.1):
        self.rate = int(rate)
        self.start = float(start)
        self.threshold = float(threshold)
        self.settle = float(settle)
        self.tolerance = float(tolerance)
        self.margin = float(margin)
        self._since: Optional[float] = None
        self._reference = 0.0

    def shortfall(self, now: float, frames: int) -> float:
        """Seconds the file is behind the wall clock."""
        return (float(now) - self.start) - float(frames) / self.rate

    def observe(self, now: float, frames: int) -> int:
        now = float(now)
        short = self.shortfall(now, frames)
        if short <= self.threshold:
            self._since = None
            return 0
        if self._since is None or abs(short - self._reference) >= self.tolerance:
            self._since, self._reference = now, short
            return 0
        if now - self._since < self.settle:
            return 0
        self._since = None
        return int(max(0.0, short - self.margin) * self.rate)
