#!/usr/bin/env python3
"""Make the spoken clips GTReview mixes into narrated recordings.

GTReview's narrated recordings announce every switch to another lesion
("lesion seven", "new lesion seven", "now lesion two").  The announcement is
built at run time from the clips in ``GTReview/Resources/GTReviewVoice``:

* ``lesion-1.wav`` ... ``lesion-99.wav``: the whole phrase, spoken in one
  breath.  Strung together word by word, numbers were misheard far more often.
* ``<word>.wav`` for every word in ``GTReviewLib.recordlog.VOICE_WORDS``: the
  "new" / "now" in front of a phrase, and the words of numbers from 100 up.

The clips that ship were synthesised with Piper TTS and its LJSpeech voice
(``en_US-ljspeech-high``), trained on public-domain recordings.  Piper itself
is GPL; only its output is kept, never the program or the model.  They are
8-bit mu-law at 11025 Hz, about a megabyte in all.

Run this with a Python that has ``piper-tts`` installed -- not inside Slicer:

    python -m venv /tmp/piper && /tmp/piper/bin/pip install piper-tts faster-whisper
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json
    /tmp/piper/bin/python Packaging/make_voice_clips.py --voice en_US-ljspeech-high.onnx --verify small.en

Piper's synthesis is random, and now and then a take comes out mumbled.
``--verify`` (needs ``faster-whisper``) has a speech recogniser listen to every
phrase as GTReview will play it and synthesises again any whose number it
does not hear.

To use a human voice instead, record WAVs under the same names (mono, 16-bit
PCM or 8-bit mu-law, any sample rate).  A phrase clip that is missing is
spoken word by word, so recording the words alone is enough to start with.
"""

import argparse
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "GTReview"))

from GTReviewLib import recordlog, wavfile  # noqa: E402

#: sample rate of the clips; GTReview resamples to the microphone's rate
RATE = 11025
#: peak level of every clip, so the words sound equally loud
PEAK = 0.7
#: kept around the voice when the silence is trimmed
MARGIN_S = 0.02
#: takes tried per phrase under --verify before giving up
ATTEMPTS = 8

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def numbers_heard(text):
    """Every number in a transcript, written in digits or in words."""
    text = text.lower().replace("-", " ")
    found = [int(d) for d in re.findall(r"\d+", text)]
    total, have = 0, False
    for word in re.findall(r"[a-z]+", text):
        if word in _ONES or word in _TENS:
            total += _ONES.get(word, 0) + _TENS.get(word, 0)
            have = True
        elif word == "hundred" and have:
            total *= 100
        elif have:
            found.append(total)
            total, have = 0, False
    if have:
        found.append(total)
    return found


def trim(signal, rate):
    """Cut the silence Piper leaves before and after the voice."""
    loud = np.flatnonzero(np.abs(signal) > 0.02 * np.abs(signal).max())
    if loud.size == 0:
        return signal
    margin = int(MARGIN_S * rate)
    return signal[max(0, loud[0] - margin):loud[-1] + margin + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--voice", required=True, help="Piper .onnx voice (its .onnx.json next to it)")
    parser.add_argument("--out", default=os.path.join(REPO, "GTReview", "Resources", "GTReviewVoice"))
    parser.add_argument("--verify", metavar="WHISPER_MODEL",
                        help="re-synthesise phrases a faster-whisper model mishears, e.g. small.en")
    args = parser.parse_args()

    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(args.voice)
    config = SynthesisConfig(length_scale=1.0, noise_scale=0.333, noise_w_scale=0.6)
    listener = None
    if args.verify:
        from faster_whisper import WhisperModel

        listener = WhisperModel(args.verify, device="cpu", compute_type="int8")

    def synthesise(text):
        chunks = list(voice.synthesize(text, syn_config=config))
        signal = np.concatenate([c.audio_float_array for c in chunks])
        signal = trim(wavfile.resample(signal, chunks[0].sample_rate, RATE), RATE)
        signal = signal * (PEAK / max(1e-6, float(np.abs(signal).max())))
        return np.round(signal * 32767).astype(np.int16)

    def heard(samples):
        # through mu-law and back, the way GTReview will play it
        played = wavfile.decode_mulaw(wavfile.encode_mulaw(samples)).astype(np.float32) / 32768.0
        played = wavfile.resample(played, RATE, 16000)
        pad = np.zeros(8000, dtype=np.float32)
        segments, _ = listener.transcribe(np.concatenate([pad, played, pad]), language="en", beam_size=5)
        return " ".join(s.text for s in segments).strip()

    os.makedirs(args.out, exist_ok=True)
    texts = [(word, word + ".", None) for word in recordlog.VOICE_WORDS]
    texts += [(recordlog.phrase_clip(n), "Lesion {}.".format(n), n) for n in recordlog.PHRASE_NUMBERS]
    total, unverified = 0, []
    for name, text, number in texts:
        samples = synthesise(text)
        note = ""
        if listener is not None and number is not None:
            for attempt in range(1, ATTEMPTS + 1):
                transcript = heard(samples)
                if number in numbers_heard(transcript):
                    note = "heard {!r}".format(transcript) + (" (take {})".format(attempt) if attempt > 1 else "")
                    break
                samples = synthesise(text)
            else:
                unverified.append(name)
                note = "NOT HEARD after {} takes".format(ATTEMPTS)
        path = os.path.join(args.out, "{}.wav".format(name))
        wavfile.write_wav(path, RATE, samples, mulaw=True)
        total += os.path.getsize(path)
        print("{:10s} {:.2f} s  {!r}  {}".format(name, samples.size / float(RATE), text, note))
    print("{} clips, {:.0f} KB".format(len(texts), total / 1024.0))
    if unverified:
        print("never heard right: " + ", ".join(unverified))
        sys.exit(1)


if __name__ == "__main__":
    main()
