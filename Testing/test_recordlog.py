"""Unit tests for GTReviewLib.recordlog -- plain unittest, no Slicer needed.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_recordlog.py' -v
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import unittest

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from GTReviewLib import recordlog  # noqa: E402
from GTReviewLib.recordlog import (  # noqa: E402
    LESION_ADOPTED,
    LESION_CLEARED,
    LESION_RENUMBERED,
    LESION_SELECTED,
    AudioClock,
    EventLog,
    FramePacer,
)


def lesionEvent(t, kind=LESION_SELECTED, number=3, **extra):
    event = {"t": t, "type": kind, "number": number, "count": 6, "label": 2,
             "label_name": "Enhancing Tumor", "voxels": 120, "seed_ijk": [1, 2, 3]}
    event.update(extra)
    if kind == LESION_CLEARED:
        event = {"t": t, "type": kind}
    return event


class NumberWordsTest(unittest.TestCase):
    def test_small_numbers_are_one_word(self):
        self.assertEqual(recordlog.number_words(0), ["zero"])
        self.assertEqual(recordlog.number_words(7), ["seven"])
        self.assertEqual(recordlog.number_words(13), ["thirteen"])

    def test_tens(self):
        self.assertEqual(recordlog.number_words(20), ["twenty"])
        self.assertEqual(recordlog.number_words(21), ["twenty", "one"])
        self.assertEqual(recordlog.number_words(99), ["ninety", "nine"])

    def test_hundreds(self):
        self.assertEqual(recordlog.number_words(100), ["one", "hundred"])
        self.assertEqual(recordlog.number_words(105), ["one", "hundred", "five"])
        self.assertEqual(recordlog.number_words(342), ["three", "hundred", "forty", "two"])

    def test_thousands_are_read_digit_by_digit(self):
        self.assertEqual(recordlog.number_words(1024), ["one", "zero", "two", "four"])

    def test_negative_numbers_are_refused(self):
        with self.assertRaises(ValueError):
            recordlog.number_words(-1)

    def test_every_announcement_has_its_clips(self):
        clips = set(recordlog.VOICE_CLIPS)
        for number in range(0, 1500):
            for kind in (LESION_SELECTED, LESION_ADOPTED, LESION_RENUMBERED):
                self.assertLessEqual(set(recordlog.announcement_clips(kind, number)), clips)
                self.assertLessEqual(set(recordlog.announcement_words(kind, number)),
                                     set(recordlog.VOICE_WORDS))

    def test_up_to_ninety_nine_is_one_phrase(self):
        self.assertEqual(recordlog.announcement_clips(LESION_SELECTED, 7), ["lesion-7"])
        self.assertEqual(recordlog.announcement_clips(LESION_ADOPTED, 99), ["new", "lesion-99"])
        self.assertEqual(recordlog.announcement_clips(LESION_RENUMBERED, 2), ["now", "lesion-2"])
        self.assertEqual(recordlog.announcement_clips(LESION_SELECTED, 0), ["lesion", "zero"])
        self.assertEqual(recordlog.announcement_clips(LESION_SELECTED, 105),
                         ["lesion", "one", "hundred", "five"])

    def test_announcements_name_the_kind_of_change(self):
        self.assertEqual(recordlog.announcement_words(LESION_SELECTED, 3), ["lesion", "three"])
        self.assertEqual(recordlog.announcement_words(LESION_ADOPTED, 3), ["new", "lesion", "three"])
        self.assertEqual(recordlog.announcement_words(LESION_RENUMBERED, 2), ["now", "lesion", "two"])


class SelectionChangeTest(unittest.TestCase):
    def test_nothing_to_nothing_is_no_event(self):
        self.assertIsNone(recordlog.selection_change(None, None, None))

    def test_a_first_selection(self):
        current = {"seed": (1, 2, 3), "number": 4}
        self.assertEqual(recordlog.selection_change(None, current, None), LESION_SELECTED)
        self.assertEqual(recordlog.selection_change(None, current, None, adopted=True), LESION_ADOPTED)

    def test_dropping_the_selection(self):
        previous = {"seed": (1, 2, 3), "number": 4}
        self.assertEqual(recordlog.selection_change(previous, None, 4), LESION_CLEARED)

    def test_same_lesion_same_number_is_no_event(self):
        state = {"seed": (1, 2, 3), "number": 4}
        self.assertIsNone(recordlog.selection_change(state, dict(state), 4))

    def test_same_lesion_new_number_is_a_renumbering(self):
        previous = {"seed": (1, 2, 3), "number": 4}
        current = {"seed": (1, 2, 4), "number": 3}
        # the old seed now lies in lesion 3, which is the one selected
        self.assertEqual(recordlog.selection_change(previous, current, 3), LESION_RENUMBERED)

    def test_another_lesion(self):
        previous = {"seed": (1, 2, 3), "number": 4}
        current = {"seed": (9, 9, 9), "number": 1}
        self.assertEqual(recordlog.selection_change(previous, current, 4), LESION_SELECTED)

    def test_another_lesion_after_the_old_one_was_erased(self):
        previous = {"seed": (1, 2, 3), "number": 4}
        current = {"seed": (9, 9, 9), "number": 4}
        # the same NUMBER, but the old seed belongs to no lesion any more
        self.assertEqual(recordlog.selection_change(previous, current, None), LESION_SELECTED)

    def test_adoption_of_a_new_lesion(self):
        previous = {"seed": (1, 2, 3), "number": 4}
        current = {"seed": (9, 9, 9), "number": 7}
        self.assertEqual(recordlog.selection_change(previous, current, 4, adopted=True),
                         LESION_ADOPTED)


class EventLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gtreview_recordlog_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_events_are_timestamped_and_never_negative(self):
        log = EventLog({"case_id": "C1"})
        log.add(-0.5, "case_loaded")
        log.add(1.23456789, LESION_SELECTED, number=2)
        self.assertEqual(log.events[0]["t"], 0.0)
        self.assertEqual(log.events[1]["number"], 2)

    def test_json_round_trip_with_rounded_floats(self):
        log = EventLog({"case_id": "C1", "audio": {"rate": 16000, "offset": 0.123456789}})
        log.add(1.23456789, LESION_SELECTED, number=2, seed_ras=[1.123456789, 2.0, 3.0])
        path = os.path.join(self.tmp, "events.json")
        log.write_json(path)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["format"], recordlog.FORMAT_NAME)
        self.assertEqual(document["version"], recordlog.FORMAT_VERSION)
        self.assertEqual(document["case_id"], "C1")
        self.assertEqual(document["audio"]["offset"], 0.1235)
        self.assertEqual(document["events"][0]["t"], 1.2346)
        self.assertEqual(document["events"][0]["seed_ras"], [1.1235, 2.0, 3.0])

    def test_non_ascii_case_ids_survive(self):
        log = EventLog({"case_id": "Zürich-α"})
        path = os.path.join(self.tmp, "events.json")
        log.write_json(path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["case_id"], "Zürich-α")


class SidecarTest(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"t": 0.5, "type": "case_loaded"},
            lesionEvent(2.0, number=1),
            lesionEvent(5.0, kind=LESION_RENUMBERED, number=2, previous_number=1),
            {"t": 6.0, "type": "lesion_deleted", "number": 3},
            lesionEvent(8.0, kind=LESION_CLEARED),
        ]

    def test_lesion_text(self):
        self.assertEqual(recordlog.lesion_text(lesionEvent(0, number=3)),
                         "Lesion #3 of 6 - Enhancing Tumor (120 voxels)")
        self.assertIn("was #1", recordlog.lesion_text(
            lesionEvent(0, kind=LESION_RENUMBERED, number=2, previous_number=1)))
        self.assertIn("new", recordlog.lesion_text(lesionEvent(0, kind=LESION_ADOPTED)))
        self.assertEqual(recordlog.lesion_text(lesionEvent(0, kind=LESION_CLEARED)),
                         "No lesion selected")

    def test_intervals_cover_the_segment_without_gaps(self):
        intervals = recordlog.lesion_intervals(self.events, 10.0)
        self.assertEqual([(a, b) for a, b, _ in intervals],
                         [(0.0, 2.0), (2.0, 5.0), (5.0, 8.0), (8.0, 10.0)])
        self.assertEqual(intervals[0][2], "No lesion selected")
        self.assertTrue(intervals[1][2].startswith("Lesion #1"))

    def test_an_event_at_zero_replaces_the_opening_stretch(self):
        intervals = recordlog.lesion_intervals([lesionEvent(0.0, number=4)], 3.0)
        self.assertEqual(len(intervals), 1)
        self.assertTrue(intervals[0][2].startswith("Lesion #4"))

    def test_srt_times_and_offset(self):
        intervals = recordlog.lesion_intervals(self.events, 10.0)
        text = recordlog.srt(intervals, offset=3.0)
        blocks = text.strip().split("\n\n")
        # the stretch from 0 to 2 s lies before a file that starts at 3 s
        self.assertEqual(len(blocks), 3)
        self.assertIn("00:00:00,000 --> 00:00:02,000", blocks[0])
        self.assertTrue(blocks[0].startswith("1\n"))
        self.assertIn("00:00:05,000 --> 00:00:07,000", blocks[2])

    def test_srt_clipped_to_the_media_duration(self):
        intervals = recordlog.lesion_intervals(self.events, 10.0)
        text = recordlog.srt(intervals, offset=0.0, duration=6.0)
        self.assertNotIn("No lesion selected\n", text.split("\n\n")[-1] + "\n")
        self.assertIn("00:00:05,000 --> 00:00:06,000", text)

    def test_srt_time_format(self):
        self.assertEqual(recordlog.format_srt_time(3725.5), "01:02:05,500")
        self.assertEqual(recordlog.format_srt_time(-1), "00:00:00,000")
        self.assertEqual(recordlog.format_clock(3725.9), "1:02:05")

    def test_audacity_labels_regions_and_points(self):
        lines = recordlog.audacity_labels(self.events, 10.0).strip().split("\n")
        rows = [line.split("\t") for line in lines]
        self.assertTrue(all(len(row) == 3 for row in rows))
        starts = [float(row[0]) for row in rows]
        self.assertEqual(starts, sorted(starts))
        self.assertIn(["6.000000", "6.000000", "Lesion #3 deleted"], rows)
        self.assertIn("Lesion #1", rows[2][2])  # the region from 2 s to 5 s

    def test_ffmetadata_chapters_in_milliseconds(self):
        intervals = recordlog.lesion_intervals(self.events, 10.0)
        text = recordlog.ffmetadata(intervals, offset=1.0, duration=9.0, title="Case a=b")
        self.assertTrue(text.startswith(";FFMETADATA1\n"))
        self.assertIn("title=Case a\\=b", text)
        self.assertEqual(text.count("[CHAPTER]"), 4)
        self.assertIn("START=1000\nEND=4000", text)

    def test_cue_marks_only_lesion_changes(self):
        marks = recordlog.cue_marks(self.events, offset=1.0)
        self.assertEqual([round(t, 3) for t, _ in marks], [1.0, 4.0, 7.0])

    def test_announcement_schedule_and_offset(self):
        schedule = recordlog.announcement_schedule(self.events, offset=1.0)
        self.assertEqual([(round(t, 3), c, text) for t, c, text in schedule],
                         [(1.0, ["lesion-1"], "lesion one"),
                          (4.0, ["now", "lesion-2"], "now lesion two")])

    def test_a_burst_of_selections_announces_only_where_the_reader_stops(self):
        burst = [lesionEvent(1.0 + 0.2 * i, number=i + 1) for i in range(5)]
        burst.append(lesionEvent(4.0, number=9))
        schedule = recordlog.announcement_schedule(burst)
        self.assertEqual([c for _, c, _ in schedule], [["lesion-5"], ["lesion-9"]])

    def test_a_lesion_selected_before_the_file_starts_is_announced_at_its_start(self):
        # Record pressed with lesion 3 selected: logged a moment before the
        # microphone opened
        events = [lesionEvent(0.0001, number=3), lesionEvent(5.0, number=4)]
        schedule = recordlog.announcement_schedule(events, offset=0.0005)
        self.assertEqual([(round(t, 3), c) for t, c, _ in schedule],
                         [(0.0, ["lesion-3"]), (5.0, ["lesion-4"])])
        marks = recordlog.cue_marks(events, offset=0.0005)
        self.assertEqual([round(t, 3) for t, _ in marks], [0.0, 5.0])

    def test_only_the_state_at_the_start_is_carried(self):
        events = [lesionEvent(1.0, number=2), lesionEvent(1.5, kind=LESION_RENUMBERED, number=1,
                                                         previous_number=2)]
        schedule = recordlog.announcement_schedule(events, offset=3.0)
        # said plainly at the start: "lesion one", not "now lesion one"
        self.assertEqual([(t, c) for t, c, _ in schedule], [(0.0, ["lesion-1"])])
        self.assertNotIn("was #", recordlog.cue_marks(events, offset=3.0)[0][1])

    def test_nothing_is_carried_after_the_selection_was_dropped(self):
        events = [lesionEvent(1.0), lesionEvent(2.0, kind=LESION_CLEARED)]
        self.assertEqual(recordlog.announcement_schedule(events, offset=3.0), [])
        self.assertEqual(recordlog.cue_marks(events, offset=3.0), [])


class FileNameTest(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(recordlog.safe_name("YG_78CQZ7VA3H2G_27"), "YG_78CQZ7VA3H2G_27")
        self.assertEqual(recordlog.safe_name("P39 2023/11:09"), "P39_2023_11_09")
        self.assertEqual(recordlog.safe_name("..."), "case")

    def test_segment_stem(self):
        when = datetime.datetime(2026, 10, 13, 9, 5, 7)
        self.assertEqual(recordlog.segment_stem("1-007-03", when), "1-007-03_20261013-090507")

    def test_unique_stem(self):
        tmp = tempfile.mkdtemp(prefix="gtreview_stem_")
        try:
            self.assertEqual(recordlog.unique_stem(tmp, "a", [".wav"]), "a")
            open(os.path.join(tmp, "a.json"), "w").close()
            self.assertEqual(recordlog.unique_stem(tmp, "a", [".wav", ".json"]), "a-2")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class DeviceChoiceTest(unittest.TestCase):
    #: what Qt 5.15 listed on a Linux workstation with PipeWire (realm, name)
    LINUX = [
        ("default", "alsa"), ("pipewire", "alsa"), ("sysdefault", "alsa"),
        ("sysdefault:CARD=AIMicro", "alsa"), ("front:CARD=AIMicro,DEV=0", "alsa"),
        ("alsa_input.usb-R__DE_R__DE_AI-Micro_BBBFCE8C-01.analog-stereo", "default"),
        ("alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source", "default"),
        ("alsa_input.usb-Generic_USB_Audio-00.HiFi__Line__source", "default"),
        ("alsa_output.usb-Generic_USB_Audio-00.HiFi__Speaker__sink.monitor", "default"),
    ]

    def test_linux_keeps_the_sound_server_sources_and_drops_monitors(self):
        choices = recordlog.input_device_choices(
            self.LINUX, default_name="alsa_input.usb-R__DE_R__DE_AI-Micro_BBBFCE8C-01.analog-stereo")
        names = [c["name"] for c in choices]
        self.assertEqual(len(names), 3)
        self.assertTrue(all(n.startswith("alsa_input.") for n in names))
        self.assertTrue(choices[0]["name"].startswith("alsa_input.usb-R__DE"))
        self.assertTrue(choices[0]["label"].endswith("(system default)"))

    def test_descriptions_win_over_raw_names(self):
        name = "alsa_input.usb-R__DE_R__DE_AI-Micro_BBBFCE8C-01.analog-stereo"
        choices = recordlog.input_device_choices(
            self.LINUX, descriptions={name: "RØDE AI-Micro Analog Stereo"})
        self.assertIn("RØDE AI-Micro Analog Stereo", [c["label"] for c in choices])

    def test_friendly_names(self):
        self.assertEqual(recordlog.friendly_device_name(
            "alsa_input.usb-Generic_USB_Audio-00.HiFi__Mic__source"), "Generic USB Audio - Mic")
        self.assertEqual(recordlog.friendly_device_name(
            "alsa_input.usb-R__DE_R__DE_AI-Micro_BBBFCE8C-01.analog-stereo"),
            "R DE R DE AI-Micro - analog-stereo")
        self.assertEqual(recordlog.friendly_device_name("Microphone (Realtek(R) Audio)"),
                         "Microphone (Realtek(R) Audio)")

    def test_alsa_only_keeps_the_plain_aliases(self):
        choices = recordlog.input_device_choices(
            [("default", "alsa"), ("hw:CARD=PCH,DEV=0", "alsa"), ("plughw:CARD=PCH,DEV=0", "alsa"),
             ("sysdefault:CARD=PCH", "alsa")])
        self.assertEqual(sorted(c["name"] for c in choices), ["default", "sysdefault:CARD=PCH"])

    def test_windows_drops_the_wasapi_duplicates(self):
        choices = recordlog.input_device_choices(
            [("Microphone (Realtek(R) Audio)", "default"), ("Headset (Jabra)", "default"),
             ("Microphone (Realtek(R) Audio)", "wasapi"), ("Headset (Jabra)", "wasapi")])
        self.assertEqual(len(choices), 2)
        self.assertTrue(all(c["realm"] == "default" for c in choices))

    def test_duplicates_and_blank_names_go(self):
        choices = recordlog.input_device_choices([("Mic", "default"), ("Mic", "default"), ("", "default")])
        self.assertEqual(len(choices), 1)


class FramePacerTest(unittest.TestCase):
    def test_one_frame_per_tick_in_real_time(self):
        pacer = FramePacer(10, start=0.0)
        self.assertEqual(pacer.due(0.0), 1)
        pacer.mark(1)
        self.assertEqual(pacer.due(0.05), 0)
        self.assertEqual(pacer.due(0.1), 1)
        pacer.mark(1)
        self.assertEqual(pacer.due(0.19), 0)

    def test_a_stall_is_filled_with_repeats(self):
        pacer = FramePacer(10, start=0.0)
        pacer.mark(pacer.due(0.0))
        # the main thread froze for two seconds
        self.assertEqual(pacer.due(2.0), 20)
        pacer.mark(20)
        self.assertEqual(pacer.emitted, 21)
        self.assertEqual(pacer.due(2.05), 0)

    def test_fps_must_be_positive(self):
        with self.assertRaises(ValueError):
            FramePacer(0, 0.0)


class AudioClockTest(unittest.TestCase):
    RATE = 16000

    def test_a_steady_stream_is_never_padded(self):
        clock = AudioClock(self.RATE, start=0.0)
        for step in range(1, 60):
            now = step * 0.5
            # the file lags the clock by a period of buffering, nothing more
            self.assertEqual(clock.observe(now, int((now - 0.05) * self.RATE)), 0)

    def test_held_audio_arriving_late_is_not_padded(self):
        clock = AudioClock(self.RATE, start=0.0)
        for now in (0.5, 1.0, 1.5, 2.0):
            clock.observe(now, int(now * self.RATE))
        # main thread frozen from 2.0 to 6.0; the server held everything
        frames = int(2.0 * self.RATE)
        for now, caught_up in ((6.0, 4.0), (6.5, 6.4), (7.0, 6.95), (7.5, 7.45), (8.0, 7.95)):
            frames = int(caught_up * self.RATE)
            self.assertEqual(clock.observe(now, frames), 0)

    def test_dropped_audio_is_padded_once_the_shortfall_settles(self):
        clock = AudioClock(self.RATE, start=0.0)
        lost = 3.0
        padded = 0
        for step in range(1, 30):
            now = step * 0.5
            frames = int((now - (lost if now > 4.0 else 0.0)) * self.RATE) + padded
            padded += clock.observe(now, frames)
        self.assertAlmostEqual(padded / self.RATE, lost - 0.1, delta=0.05)

    def test_blocks_of_seconds_are_not_mistaken_for_loss(self):
        # WinMM with a ten-second buffer delivers two seconds at a time; the
        # clock is consulted as each block arrives
        clock = AudioClock(self.RATE, start=0.0)
        for block in range(1, 15):
            now = block * 2.0 + 0.03
            self.assertEqual(clock.observe(now, int(block * 2.0 * self.RATE)), 0)

    def test_loss_between_blocks_is_padded(self):
        clock = AudioClock(self.RATE, start=0.0)
        padded = 0
        for block in range(1, 15):
            now = block * 2.0 + 0.03
            lost = 4.0 if block > 3 else 0.0
            padded += clock.observe(now, int((block * 2.0 - lost) * self.RATE) + padded)
        # the 4 s lost plus the 0.03 s each block arrives late, less the margin
        self.assertAlmostEqual(padded / self.RATE, 4.0 + 0.03 - 0.1, delta=0.02)

    def test_a_drifting_shortfall_is_not_padded_while_it_moves(self):
        clock = AudioClock(self.RATE, start=0.0)
        for step in range(1, 12):
            now = step * 0.5
            # behind by more each time: still catching up, or still losing
            shortfall = 1.0 + 0.3 * step
            self.assertEqual(clock.observe(now, int((now - shortfall) * self.RATE)), 0)


if __name__ == "__main__":
    unittest.main()
