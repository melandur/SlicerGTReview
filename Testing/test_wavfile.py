"""Unit tests for GTReviewLib.wavfile -- plain unittest, no Slicer needed.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_wavfile.py' -v
"""

import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from GTReviewLib import recordlog, wavfile  # noqa: E402

VOICE_DIR = os.path.join(_MODULE_DIR, "Resources", "GTReviewVoice")


def sine(frequency, seconds, rate, level=0.5):
    t = np.arange(int(seconds * rate)) / float(rate)
    return (np.sin(2 * np.pi * frequency * t) * level).astype(np.float32)


def dominant(signal, rate):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    return np.fft.rfftfreq(signal.size, 1.0 / rate)[int(np.argmax(spectrum))]


class WavWriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gtreview_wav_")
        self.path = os.path.join(self.tmp, "mic.wav")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        samples = (np.arange(1000) - 500).astype(np.int16)
        writer = wavfile.WavWriter(self.path, 16000)
        writer.write(samples.tobytes())
        writer.close()
        rate, back = wavfile.read_wav(self.path)
        self.assertEqual(rate, 16000)
        np.testing.assert_array_equal(back[:, 0], samples)
        with wave.open(self.path, "rb") as handle:  # the stdlib agrees
            self.assertEqual(handle.getnframes(), 1000)

    def test_half_frames_wait_for_the_rest(self):
        writer = wavfile.WavWriter(self.path, 8000, channels=2)
        data = struct.pack("<4h", 1, 2, 3, 4)
        self.assertEqual(writer.write(data[:3]), 0)
        self.assertEqual(writer.write(data[3:]), 2)
        self.assertEqual(writer.frames, 2)
        writer.close()
        _rate, back = wavfile.read_wav(self.path)
        np.testing.assert_array_equal(back, [[1, 2], [3, 4]])

    def test_the_header_is_kept_current_while_recording(self):
        writer = wavfile.WavWriter(self.path, 1000, patch_seconds=1.0)
        writer.write(np.zeros(2500, dtype=np.int16).tobytes())
        # still open, as if Slicer had died here: the file already plays
        with wave.open(self.path, "rb") as handle:
            self.assertGreaterEqual(handle.getnframes(), 1000)
        writer.close()
        with wave.open(self.path, "rb") as handle:
            self.assertEqual(handle.getnframes(), 2500)

    def test_silence_and_seconds(self):
        writer = wavfile.WavWriter(self.path, 16000)
        writer.write_silence(8000)
        self.assertAlmostEqual(writer.seconds, 0.5)
        writer.close()
        self.assertTrue(writer.closed)
        with self.assertRaises(ValueError):
            writer.write(b"\x00\x00")

    def test_a_header_claiming_more_than_the_file_holds(self):
        writer = wavfile.WavWriter(self.path, 1000)
        writer.write(np.ones(100, dtype=np.int16).tobytes())
        writer.close()
        with open(self.path, "r+b") as handle:  # a size patched before a crash
            handle.seek(40)
            handle.write(struct.pack("<I", 10 ** 6))
        rate, back = wavfile.read_wav(self.path)
        self.assertEqual(back.shape, (100, 1))

    def test_closed_even_when_the_last_update_fails(self):
        writer = wavfile.WavWriter(self.path, 1000)
        writer.write(np.ones(10, dtype=np.int16).tobytes())
        handle = writer._handle

        def full_disk():
            raise OSError(28, "No space left on device")

        writer.flush = full_disk
        with self.assertRaises(OSError):
            writer.close()
        self.assertTrue(writer.closed)
        self.assertTrue(handle.closed)
        writer.close()  # a second close is harmless

    def test_not_a_wav(self):
        with open(self.path, "wb") as handle:
            handle.write(b"hello world, not audio")
        with self.assertRaises(ValueError):
            wavfile.read_wav(self.path)


class CuePointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gtreview_cue_")
        self.path = os.path.join(self.tmp, "mic.wav")
        wavfile.write_wav(self.path, 16000, np.zeros(16000 * 4, dtype=np.int16))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        written = wavfile.append_cue_points(
            self.path, [(0.5, "Lesion #1"), (2.25, "Lesion #2 – Ödem"), (9.0, "past the end")])
        self.assertEqual(written, 2)
        self.assertEqual(wavfile.read_cue_points(self.path),
                         [(8000, "Lesion #1"), (36000, "Lesion #2 – Ödem")])

    def test_the_audio_is_untouched_and_the_file_stays_valid(self):
        wavfile.append_cue_points(self.path, [(1.0, "odd")])
        rate, back = wavfile.read_wav(self.path)
        self.assertEqual((rate, back.shape), (16000, (64000, 1)))
        with wave.open(self.path, "rb") as handle:
            self.assertEqual(handle.getnframes(), 64000)
        with open(self.path, "rb") as handle:
            blob = handle.read()
        self.assertEqual(struct.unpack("<I", blob[4:8])[0], len(blob) - 8)

    def test_appended_in_place_without_rewriting_the_audio(self):
        with open(self.path, "rb") as handle:
            before = handle.read()
        wavfile.append_cue_points(self.path, [(1.0, "one"), (2.0, "two")])
        with open(self.path, "rb") as handle:
            after = handle.read()
        # everything but the RIFF size is the same bytes, then the new chunks
        self.assertEqual(after[:4], before[:4])
        self.assertEqual(after[8:len(before)], before[8:])
        self.assertEqual(struct.unpack("<I", after[4:8])[0], len(after) - 8)

    def test_an_odd_length_file_gets_its_pad_byte(self):
        odd = os.path.join(self.tmp, "odd.wav")
        with open(self.path, "rb") as handle:
            blob = handle.read()
        with open(odd, "wb") as handle:  # a stray odd-sized chunk after the data
            handle.write(blob[:4] + struct.pack("<I", len(blob) - 8 + 11) + blob[8:]
                         + b"junk" + struct.pack("<I", 3) + b"abc")
        wavfile.append_cue_points(odd, [(0.5, "half")])
        self.assertEqual(wavfile.read_cue_points(odd), [(8000, "half")])
        with wave.open(odd, "rb") as handle:
            self.assertEqual(handle.getnframes(), 64000)

    def test_no_points_leaves_the_file_alone(self):
        before = os.path.getsize(self.path)
        self.assertEqual(wavfile.append_cue_points(self.path, []), 0)
        self.assertEqual(os.path.getsize(self.path), before)


class MulawTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gtreview_mulaw_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_codec_round_trip_is_close(self):
        samples = np.linspace(-32768, 32767, 5000).astype(np.int16)
        back = wavfile.decode_mulaw(wavfile.encode_mulaw(samples)).astype(np.int32)
        error = np.abs(back - samples.astype(np.int32))
        # mu-law keeps the error proportional to the level: under 4 %, worst
        # just above each segment boundary
        self.assertTrue(np.all(error <= np.maximum(16, 0.04 * np.abs(samples.astype(np.int32)))))

    def test_known_g711_codes(self):
        np.testing.assert_array_equal(wavfile.encode_mulaw(np.array([0, -1, 32767, -32768])),
                                      [0xFF, 0x7F, 0x80, 0x00])
        np.testing.assert_array_equal(wavfile.decode_mulaw(np.array([0xFF, 0x80, 0x00])),
                                      [0, 32124, -32124])

    def test_mulaw_file_round_trip(self):
        path = os.path.join(self.tmp, "clip.wav")
        samples = (sine(440.0, 0.5, 11025) * 32767).astype(np.int16)
        wavfile.write_wav(path, 11025, samples, mulaw=True)
        self.assertLess(os.path.getsize(path), samples.size + 100)  # one byte a sample
        rate, back = wavfile.read_wav(path)
        self.assertEqual(rate, 11025)
        self.assertEqual(back.shape, (samples.size, 1))
        self.assertAlmostEqual(dominant(back[:, 0].astype(np.float32), rate), 440.0, delta=3.0)


class ResampleTest(unittest.TestCase):
    def test_length_and_pitch_are_kept(self):
        clip = sine(440.0, 1.0, 22050)
        for target in (16000, 44100, 48000):
            out = wavfile.resample(clip, 22050, target)
            self.assertEqual(out.size, target)
            self.assertAlmostEqual(dominant(out, target), 440.0, delta=2.0)
            self.assertAlmostEqual(float(np.abs(out).max()), 0.5, delta=0.05)

    def test_same_rate_is_a_copy(self):
        clip = sine(300.0, 0.1, 8000)
        out = wavfile.resample(clip, 8000, 8000)
        np.testing.assert_array_equal(out, clip)
        self.assertIsNot(out, clip)


class AnnouncementTest(unittest.TestCase):
    RATE = 16000

    def clips(self):
        return {name: sine(200.0 + 10 * (i % 50), 0.3, self.RATE, 0.4)
                for i, name in enumerate(recordlog.VOICE_CLIPS)}

    def test_phrase_strings_the_words_together(self):
        clips = self.clips()
        out = wavfile.phrase(["lesion", "three"], clips, self.RATE, with_chime=False)
        self.assertEqual(out.size, int(0.3 * self.RATE) * 2 + int(0.05 * self.RATE))
        with_chime = wavfile.phrase(["lesion", "three"], clips, self.RATE)
        self.assertGreater(with_chime.size, out.size)

    def test_a_missing_word_gives_no_phrase(self):
        self.assertIsNone(wavfile.phrase(["lesion", "three"], {"lesion": sine(1, 0.1, 100)}, 100))

    def test_narrate_places_ducks_and_keeps_the_rest(self):
        rate = self.RATE
        mic = (sine(1000.0, 4.0, rate, 0.25) * 32767).astype(np.int16)
        clip = sine(300.0, 0.5, rate, 0.5)
        out = wavfile.narrate(mic, rate, [(1.0, clip)])
        self.assertEqual(out.shape, (mic.size, 1))
        # before the announcement (and its fade) nothing changed
        np.testing.assert_array_equal(out[: int(0.9 * rate), 0], mic[: int(0.9 * rate)])
        # during it the 300 Hz voice dominates and the reader is still there
        middle = out[int(1.1 * rate): int(1.4 * rate), 0].astype(np.float32)
        self.assertAlmostEqual(dominant(middle, rate), 300.0, delta=5.0)
        spectrum = np.abs(np.fft.rfft(middle))
        freqs = np.fft.rfftfreq(middle.size, 1.0 / rate)
        self.assertGreater(spectrum[np.argmin(np.abs(freqs - 1000.0))], 0)
        # afterwards it is the microphone again
        np.testing.assert_array_equal(out[int(1.7 * rate):, 0], mic[int(1.7 * rate):])

    def test_announcements_never_overlap(self):
        rate = self.RATE
        mic = np.zeros(rate * 4, dtype=np.int16)
        clip = np.full(rate // 2, 0.5, dtype=np.float32)
        out = wavfile.narrate(mic, rate, [(1.0, clip), (1.1, clip), (9.0, clip)])
        loud = np.flatnonzero(out[:, 0] != 0)
        # two clips back to back with a short pause, the third is past the end
        self.assertEqual(loud.size, 2 * (rate // 2))
        self.assertEqual(loud[0], rate)
        self.assertGreaterEqual(loud[-1], rate + rate // 2 + int(0.15 * rate))

    def test_a_held_back_announcement_gives_way_to_the_next(self):
        rate = self.RATE
        mic = np.zeros(rate * 10, dtype=np.int16)
        long_phrase = np.full(int(1.5 * rate), 0.5, dtype=np.float32)
        # a lesion a second, every announcement a second and a half long
        out = wavfile.narrate(mic, rate, [(t, long_phrase) for t in (1.0, 2.0, 3.0, 4.0)])
        loud = out[:, 0] != 0
        starts = (np.flatnonzero(loud[1:] & ~loud[:-1]) + 1) / float(rate)
        # 1 s on time, 2 s after the first, 3 s left out (it could start only
        # after 4 s was due), 4 s after the second: the voice never runs behind
        np.testing.assert_allclose(starts, [1.0, 2.65, 4.3], atol=1.0 / rate)

    def test_loud_input_is_clipped_not_wrapped(self):
        mic = np.full(1000, 32000, dtype=np.int16)
        out = wavfile.narrate(mic, 1000, [(0.1, np.ones(300, dtype=np.float32))], duck=1.0)
        self.assertEqual(int(out.max()), 32767)
        self.assertGreater(int(out.min()), 0)

    def test_stereo_microphones_keep_both_channels(self):
        mic = np.zeros((2000, 2), dtype=np.int16)
        out = wavfile.narrate(mic, 1000, [(0.5, np.ones(100, dtype=np.float32) * 0.1)])
        self.assertEqual(out.shape, (2000, 2))
        np.testing.assert_array_equal(out[:, 0], out[:, 1])


class ShippedVoiceTest(unittest.TestCase):
    """The clips in Resources/GTReviewVoice: every one there, short, not silent."""

    def test_every_clip_is_there(self):
        clips = wavfile.load_voice(VOICE_DIR, 16000, recordlog.VOICE_CLIPS)
        missing = sorted(set(recordlog.VOICE_CLIPS) - set(clips))
        self.assertEqual(missing, [])
        for name, clip in clips.items():
            # "lesion three hundred seventy-seven" is the longest kind
            self.assertLess(clip.size, 16000 * 3.0, name)
            self.assertGreater(float(np.abs(clip).max()), 0.3, name)

    def test_nothing_else_is_there(self):
        names = {os.path.splitext(n)[0] for n in os.listdir(VOICE_DIR) if n.endswith(".wav")}
        self.assertEqual(sorted(names - set(recordlog.VOICE_CLIPS)), [])

    def test_clips_resample_to_the_microphone_rate(self):
        path = os.path.join(VOICE_DIR, "lesion-7.wav")
        clip = wavfile.load_clip(path, 48000)
        rate, raw = wavfile.read_wav(path)
        self.assertAlmostEqual(clip.size / 48000.0, raw.shape[0] / float(rate), places=2)

    def test_an_announcement_can_be_built_for_any_number(self):
        clips = wavfile.load_voice(VOICE_DIR, 16000, recordlog.VOICE_CLIPS)
        for number in (1, 42, 99, 100, 377, 500, 501, 4321):
            names = recordlog.announcement_clips(recordlog.LESION_SELECTED, number)
            self.assertIsNotNone(wavfile.phrase(names, clips, 16000), number)

    def test_a_voice_without_the_phrase_speaks_the_words(self):
        # a voice recorded by hand may have the single words only
        words = wavfile.load_voice(VOICE_DIR, 16000, recordlog.VOICE_WORDS)
        names = recordlog.announcement_clips(recordlog.LESION_ADOPTED, 123)
        self.assertIsNone(wavfile.phrase(names, words, 16000))
        self.assertIsNotNone(wavfile.phrase(recordlog.words_for(names), words, 16000))



class MicrophoneTestReportTest(unittest.TestCase):
    """level_report: how loud a microphone test came in, and the boost offered."""

    RATE = 16000

    def voice(self, peak_dbfs, seconds=3.0, pauses=True):
        """A stand-in voice: a 200 Hz tone in quarter-second bursts, every
        other one silent, peaking at *peak_dbfs* (overdriven above 0)."""
        t = np.arange(int(seconds * self.RATE)) / float(self.RATE)
        x = np.sin(2 * np.pi * 200.0 * t) * 32767.0 * 10 ** (peak_dbfs / 20.0)
        if pauses:
            x *= np.sin(2 * np.pi * 2.0 * t) > 0
        return np.clip(np.round(x), -32768, 32767).astype(np.int16)

    def test_silence_is_nothing_heard(self):
        self.assertEqual(wavfile.level_report(np.zeros(self.RATE, np.int16), self.RATE)["verdict"], "silent")
        self.assertEqual(wavfile.level_report(np.zeros(0, np.int16), self.RATE)["verdict"], "silent")

    def test_a_voice_at_a_good_level(self):
        report = wavfile.level_report(self.voice(-12.0), self.RATE)
        self.assertEqual((report["verdict"], report["boost_db"]), ("good", 0))
        self.assertAlmostEqual(report["peak_db"], -12.0, delta=0.2)
        # a tone's loud stretches are 3 dB under its peak; the pauses do not count
        self.assertAlmostEqual(report["voice_db"], -15.0, delta=0.5)

    def test_a_quiet_voice_is_offered_the_boost_that_brings_it_up(self):
        report = wavfile.level_report(self.voice(-40.0), self.RATE)   # voice at -43: 25 dB short
        self.assertEqual((report["verdict"], report["boost_db"]), ("quiet", 24))
        report = wavfile.level_report(self.voice(-30.0), self.RATE)   # voice at -33: 15 dB short
        self.assertEqual((report["verdict"], report["boost_db"]), ("quiet", 12))

    def test_the_boost_offered_leaves_the_loudest_moment_room(self):
        x = self.voice(-40.0)
        x[100] = int(32767 * 10 ** (-8 / 20.0))                       # one click at -8 dBFS
        report = wavfile.level_report(x, self.RATE)
        self.assertEqual((report["verdict"], report["boost_db"]), ("quiet", 0))

    def test_an_overdriven_voice_clips(self):
        self.assertEqual(wavfile.level_report(self.voice(6.0), self.RATE)["verdict"], "loud")

    def test_two_channels_are_judged_as_heard(self):
        x = self.voice(-12.0)
        self.assertEqual(wavfile.level_report(np.stack([x, x], axis=1), self.RATE)["verdict"], "good")

    def test_converted_for_an_output_that_plays_48_khz_stereo(self):
        x = self.voice(-12.0, seconds=1.0, pauses=False)
        out = np.frombuffer(wavfile.convert_pcm(x.tobytes(), self.RATE, 1, 48000, 2), dtype="<i2")
        out = out.reshape(-1, 2)
        self.assertEqual(out.shape, (48000, 2))
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        self.assertAlmostEqual(float(np.max(np.abs(out[300:-300]))), float(np.max(np.abs(x))), delta=300)

if __name__ == "__main__":
    unittest.main()
