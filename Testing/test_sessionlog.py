"""Unit tests for GTReviewLib.sessionlog -- plain unittest, no Slicer needed.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_sessionlog.py' -v
"""

import faulthandler
import gc
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
import weakref
from unittest import mock

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from GTReviewLib import sessionlog  # noqa: E402
from GTReviewLib.sessionlog import (  # noqa: E402
    LOG_FILE_NAME,
    MAX_BYTES,
    WATCHDOG_REARM_MS,
    WATCHDOG_TIMEOUT_S,
    SessionLog,
    SourceFilter,
    installed_build,
    installed_revision,
    keep_slicer_entry,
    rotate_if_large,
    session_header,
)

#: how the file stamps a line, the same for records and for everything else
_STAMP = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"


class _Held:
    """Stands for a mask array or the widget: something a log must not keep alive."""


def _fail_holding(held):
    raise RuntimeError("failed while holding a {}".format(type(held).__name__))


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _Fixture(unittest.TestCase):
    """A private logger and a fake GTReview source folder per test.

    The exception hook and faulthandler are process-wide, so each test gets
    them back as it found them even when it fails half way.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source_dir = os.path.join(self.tmp.name, "src", "GTReview")
        os.makedirs(self.source_dir)
        self.inside = os.path.join(self.source_dir, "GTReviewLib", "dataset.py")
        self.outside = os.path.join(self.tmp.name, "src", "OtherModule", "other.py")
        self.logger = logging.getLogger("test_sessionlog.{}".format(id(self)))
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.addCleanup(self._restoreProcessState, sys.excepthook, faulthandler.is_enabled())
        self.addCleanup(self._dropHandlers)

    @staticmethod
    def _restoreProcessState(excepthook, fault_enabled):
        sys.excepthook = excepthook
        faulthandler.cancel_dump_traceback_later()
        if faulthandler.is_enabled() and not fault_enabled:
            faulthandler.disable()
        elif fault_enabled and not faulthandler.is_enabled():
            faulthandler.enable(file=sys.__stderr__)

    def raised_in(self, path, message="boom"):
        """``sys.exc_info()`` of an exception raised by code that claims to live in *path*."""
        namespace = {}
        exec(compile("def fail(message):\n    raise RuntimeError(message)\n", path, "exec"), namespace)
        try:
            namespace["fail"](message)
        except RuntimeError:
            return sys.exc_info()
        raise AssertionError("fail() did not raise")

    def _dropHandlers(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass

    def emit(self, message, pathname=None, level=logging.INFO, args=(), exc_info=None):
        record = self.logger.makeRecord(
            self.logger.name, level, pathname or self.inside, 42, message, args, exc_info
        )
        self.logger.handle(record)

    def folder(self, name="batch_01"):
        path = os.path.join(self.tmp.name, name)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def read(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()


class SourceFilterTest(_Fixture):
    def _record(self, message, pathname):
        return self.logger.makeRecord(self.logger.name, logging.INFO, pathname, 1, message, (), None)

    def test_records_from_the_module_folder_pass(self):
        self.assertTrue(SourceFilter(self.source_dir).filter(self._record("x", self.inside)))

    def test_records_from_elsewhere_are_dropped(self):
        self.assertFalse(SourceFilter(self.source_dir).filter(self._record("x", self.outside)))

    def test_a_sibling_folder_sharing_the_prefix_is_not_inside(self):
        sibling = self.source_dir + "Extras" + os.sep + "x.py"
        self.assertFalse(SourceFilter(self.source_dir).filter(self._record("x", sibling)))

    def test_messages_that_name_gtreview_pass_from_anywhere(self):
        self.assertTrue(
            SourceFilter(self.source_dir).filter(self._record("GTReview: 3 lesions", self.outside))
        )

    def test_a_record_whose_message_cannot_format_is_dropped_not_raised(self):
        record = self.logger.makeRecord(self.logger.name, logging.INFO, self.outside, 1, "%d", ("x",), None)
        self.assertFalse(SourceFilter(self.source_dir).filter(record))


class SessionLogTest(_Fixture):
    def test_records_before_a_folder_is_loaded_are_written_after_the_header(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        self.emit("early one")
        self.emit("early two")
        path = log.attach(self.folder(), details=[("batch folder", "B")])
        self.emit("after attach")
        text = self.read(path)
        header_end = text.index("=" * 72, text.index("GTReview session started"))
        self.assertLess(header_end, text.index("early one"))
        self.assertLess(text.index("early one"), text.index("early two"))
        self.assertLess(text.index("early two"), text.index("after attach"))

    def test_the_file_lands_in_the_loaded_folder(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        folder = self.folder()
        self.assertEqual(log.attach(folder), os.path.join(folder, LOG_FILE_NAME))
        self.assertTrue(os.path.isfile(os.path.join(folder, LOG_FILE_NAME)))

    def test_the_header_names_the_session_and_its_details(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder(), details=[("slicer", "5.12.3 (34627)"), ("gtreview", "abc123")])
        text = self.read(path)
        for expected in ("GTReview session started", "platform", "python", "5.12.3 (34627)", "abc123"):
            self.assertIn(expected, text)

    def test_other_modules_are_not_logged(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        self.emit("from another module", pathname=self.outside)
        self.emit("from GTReview")
        text = self.read(path)
        self.assertNotIn("from another module", text)
        self.assertIn("from GTReview", text)

    def test_debug_records_are_kept(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        self.emit("a debug detail", level=logging.DEBUG)
        self.assertIn("DEBUG", self.read(path))
        self.assertIn("a debug detail", self.read(path))

    def test_attaching_the_same_folder_again_writes_no_second_header(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        folder = self.folder()
        log.attach(folder)
        log.attach(os.path.join(folder, ".", ""))
        self.assertEqual(self.read(os.path.join(folder, LOG_FILE_NAME)).count("GTReview session started"), 1)

    def test_switching_folders_moves_the_log(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        first = log.attach(self.folder("batch_01"))
        self.emit("in batch one")
        second = log.attach(self.folder("batch_02"))
        self.emit("in batch two")
        self.assertIn("in batch one", self.read(first))
        self.assertNotIn("in batch two", self.read(first))
        self.assertIn("in batch two", self.read(second))
        self.assertNotIn("in batch one", self.read(second))

    def test_a_folder_that_cannot_be_written_returns_none_and_keeps_buffering(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        missing = os.path.join(self.tmp.name, "no", "such", "folder")
        self.emit("kept for later")
        self.assertIsNone(log.attach(missing))
        self.assertIsNone(log.path)
        path = log.attach(self.folder())
        self.assertIn("kept for later", self.read(path))

    def test_a_large_log_is_rotated_when_a_session_starts(self):
        folder = self.folder()
        target = os.path.join(folder, LOG_FILE_NAME)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("x" * 100)
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=100)
        log.attach(folder)
        self.assertEqual(self.read(target + ".1"), "x" * 100)
        self.assertIn("GTReview session started", self.read(target))
        self.assertNotIn("xxxx", self.read(target))

    def test_a_small_log_is_appended_to(self):
        folder = self.folder()
        target = os.path.join(folder, LOG_FILE_NAME)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("previous session\n")
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=10 ** 6)
        log.attach(folder)
        self.assertTrue(self.read(target).startswith("previous session\n"))
        self.assertFalse(os.path.exists(target + ".1"))

    def test_non_ascii_messages_round_trip_as_utf8(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        self.emit("Läsion Ödem 病変")
        self.assertIn("Läsion Ödem 病変", self.read(path))

    def test_exceptions_carry_their_traceback(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        try:
            raise ValueError("broken mask")
        except ValueError:
            self.emit("saving failed", level=logging.ERROR, exc_info=sys.exc_info())
        text = self.read(path)
        self.assertIn("Traceback", text)
        self.assertIn("ValueError: broken mask", text)

    def test_the_buffer_is_bounded(self):
        log = SessionLog(self.source_dir, logger=self.logger, buffered_records=3)
        for n in range(10):
            self.emit("record {}".format(n))
        text = self.read(log.attach(self.folder()))
        self.assertNotIn("record 6", text)
        self.assertIn("record 7", text)
        self.assertIn("record 9", text)

    def test_a_buffered_exception_lets_go_of_the_frames_it_passed_through(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        held = _Held()
        alive = weakref.ref(held)
        try:
            _fail_holding(held)
        except RuntimeError:
            self.emit("saving failed", level=logging.ERROR, exc_info=sys.exc_info())
        del held
        # a whole session whose folder could not be written buffers too
        self.assertIsNone(log.attach(os.path.join(self.tmp.name, "no", "such", "folder")))
        gc.collect()
        self.assertIsNone(alive(), "the buffered record still holds the raising frame's locals")
        text = self.read(log.attach(self.folder()))
        self.assertIn("saving failed", text)
        self.assertIn("Traceback (most recent call last):", text)
        self.assertIn("in _fail_holding", text)
        self.assertIn("RuntimeError: failed while holding a _Held", text)

    def test_a_buffered_record_lets_go_of_its_arguments(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        held = _Held()
        alive = weakref.ref(held)
        self.emit("GTReview: saving %s", args=(held,))
        del held
        gc.collect()
        self.assertIsNone(alive())
        self.assertRegex(self.read(log.attach(self.folder())), r"GTReview: saving <.*_Held object at ")

    def test_other_handlers_still_get_the_record_as_it_was_logged(self):
        SessionLog(self.source_dir, logger=self.logger)
        after = _Capture()
        self.logger.addHandler(after)
        try:
            raise ValueError("broken mask")
        except ValueError:
            self.emit("saving %s failed", args=("case_001",), level=logging.ERROR, exc_info=sys.exc_info())
        (record,) = after.records
        self.assertEqual((record.msg, record.args), ("saving %s failed", ("case_001",)))
        self.assertIs(record.exc_info[0], ValueError)

    def test_detach_stops_writing(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.detach()
        self.emit("after detach")
        self.assertNotIn("after detach", self.read(path))

    def test_close_takes_the_handler_off_the_logger(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.close()
        self.emit("after close")
        self.assertNotIn("after close", self.read(path))
        self.assertFalse(any(getattr(h, "is_gtreview_session_log", False) for h in self.logger.handlers))

    def test_a_reloaded_module_replaces_the_previous_handler(self):
        first = SessionLog(self.source_dir, logger=self.logger)
        first_path = first.attach(self.folder("batch_01"))
        second = SessionLog(self.source_dir, logger=self.logger)
        second_path = second.attach(self.folder("batch_02"))
        self.emit("once")
        marked = [h for h in self.logger.handlers if getattr(h, "is_gtreview_session_log", False)]
        self.assertEqual(len(marked), 1)
        self.assertNotIn("once", self.read(first_path))
        self.assertEqual(self.read(second_path).count("once"), 1)


class HelpersTest(_Fixture):
    def test_rotate_ignores_a_missing_file(self):
        self.assertFalse(rotate_if_large(os.path.join(self.tmp.name, "absent.log"), 1))

    def test_rotate_replaces_an_older_backup(self):
        target = os.path.join(self.folder(), LOG_FILE_NAME)
        for name, text in ((target + ".1", "old backup"), (target, "y" * 20)):
            with open(name, "w", encoding="utf-8") as handle:
                handle.write(text)
        self.assertTrue(rotate_if_large(target, 10))
        self.assertEqual(self.read(target + ".1"), "y" * 20)

    def test_session_header_lists_details_in_order(self):
        lines = session_header([("first", 1), ("second", 2)])
        joined = "\n".join(lines)
        self.assertLess(joined.index("first"), joined.index("second"))

    def test_installed_revision_reads_the_description_file(self):
        root = os.path.join(self.tmp.name, "Extensions-34627", "GTReview")
        scripted = os.path.join(root, "lib", "Slicer-5.12", "qt-scripted-modules")
        share = os.path.join(root, "share", "Slicer-5.12")
        os.makedirs(scripted)
        os.makedirs(share)
        with open(os.path.join(share, "GTReview.s4ext"), "w", encoding="utf-8") as handle:
            handle.write("# generated\nscm git\nscmrevision 33cec61\ncategory Segmentation\n")
        self.assertEqual(installed_revision(os.path.join(scripted, "GTReview.py")), "33cec61")

    def test_installed_revision_is_none_in_a_source_tree(self):
        self.assertIsNone(installed_revision(os.path.join(self.source_dir, "GTReview.py")))

    def installed(self, description=None, root=None):
        """The module file of an extension installed under *root*, described by *description*.

        *description* is the .s4ext text, bytes written as they are, or None
        for no file at all.
        """
        root = root or os.path.join(self.tmp.name, "Extensions-34627", "GTReview")
        scripted = os.path.join(root, "lib", "Slicer-5.12", "qt-scripted-modules")
        share = os.path.join(root, "share", "Slicer-5.12")
        os.makedirs(scripted, exist_ok=True)
        os.makedirs(share, exist_ok=True)
        if isinstance(description, str):
            description = description.encode("utf-8")
        if description is not None:
            with open(os.path.join(share, "GTReview.s4ext"), "wb") as handle:
                handle.write(description)
        return os.path.join(scripted, "GTReview.py")

    #: what Packaging/make_package.sh writes, cut short
    PACKAGED = (
        "# Generated by Packaging/make_package.sh from the top-level CMakeLists.txt.\n"
        "# gtreview-version v0.2.0-14-g64ae4e5-dirty\n"
        "scm git\n"
        "scmrevision 64ae4e5d0c1b2a3f\n"
        "category Segmentation\n"
    )

    def test_installed_build_names_the_version_and_the_revision(self):
        mac_root = os.path.join(self.tmp.name, "Slicer.app", "Contents", "Extensions-34627", "GTReview")
        for root in (None, mac_root):
            with self.subTest(root=root):
                self.assertEqual(
                    installed_build(self.installed(self.PACKAGED, root)),
                    "v0.2.0-14-g64ae4e5-dirty (64ae4e5d0c1b2a3f)",
                )

    def test_installed_revision_ignores_the_version_comment(self):
        self.assertEqual(installed_revision(self.installed(self.PACKAGED)), "64ae4e5d0c1b2a3f")

    def test_installed_build_without_the_version_comment_is_the_revision(self):
        module = self.installed("# generated\n# scmrevision 1111111\nscm git\nscmrevision 33cec61\n")
        self.assertEqual(installed_build(module), "33cec61")

    def test_installed_build_with_only_the_version_comment_is_the_version(self):
        self.assertEqual(installed_build(self.installed("#gtreview-version  v0.2.0 \nscm git\n")), "v0.2.0")

    def test_installed_build_that_cannot_be_told_names_the_extension_folder(self):
        root = os.path.join(self.tmp.name, "Extensions-34627", "GTReview")
        expected = "unknown build at {}".format(root)
        with self.subTest("no description file"):
            self.assertEqual(installed_build(self.installed(None)), expected)
        with self.subTest("a description naming neither"):
            self.assertEqual(installed_build(self.installed("scm git\ncategory Segmentation\n")), expected)
        with self.subTest("a description that is not UTF-8"):
            self.assertEqual(installed_build(self.installed(b"scmrevision \xff\xfe\n")), expected)
        with self.subTest("a description that is a folder"):
            module = self.installed(None)
            os.remove(os.path.join(root, "share", "Slicer-5.12", "GTReview.s4ext"))
            os.makedirs(os.path.join(root, "share", "Slicer-5.12", "GTReview.s4ext"))
            self.assertEqual(installed_build(module), expected)

    def test_installed_build_is_none_in_a_source_tree(self):
        self.assertIsNone(installed_build(os.path.join(self.source_dir, "GTReview.py")))
        # even with a description where an installed one would be
        repo = os.path.join(self.tmp.name, "SlicerGTReview")
        share = os.path.join(repo, "share", "GTReview")
        os.makedirs(share)
        with open(os.path.join(share, "GTReview.s4ext"), "w", encoding="utf-8") as handle:
            handle.write(self.PACKAGED)
        self.assertIsNone(installed_build(os.path.join(repo, "GTReview", "GTReview.py")))

    def test_crash_reports_default_to_off_on_windows_only(self):
        self.assertEqual(sessionlog.CRASH_REPORTS_ENABLED, os.name != "nt")

    def test_session_header_writes_a_plain_string_detail_as_it_is(self):
        self.assertIn("  Crash reports: on", session_header([("first", 1), "Crash reports: on"]))

    def test_the_watchdog_is_rearmed_well_inside_its_timeout(self):
        self.assertEqual(WATCHDOG_TIMEOUT_S, 15.0)
        self.assertEqual(WATCHDOG_REARM_MS, 5000)
        # two missed re-arms in a row still do not report a hang
        self.assertLess(2 * WATCHDOG_REARM_MS, WATCHDOG_TIMEOUT_S * 1000)


class KeepSlicerEntryTest(unittest.TestCase):
    LEVELS = {
        "None": False,
        "Unknown": False,
        "Status": False,
        "Trace": False,
        "Debug": False,
        "Info": False,
        "": False,
        "Warning": True,
        "Error": True,
        "Critical": True,
        "Fatal": True,
    }
    ORIGINS = {"VTK": True, "Qt": True, "ctkErrorLogModel": True, "": True, "Python": False, "Stream": False}

    def test_every_level_and_origin(self):
        for level, level_kept in self.LEVELS.items():
            for origin, origin_kept in self.ORIGINS.items():
                for spelled in {level, level.upper(), level.lower()}:
                    with self.subTest(level=spelled, origin=origin):
                        self.assertEqual(keep_slicer_entry(spelled, origin), level_kept and origin_kept)

    def test_python_and_stream_are_skipped_however_they_are_spelled(self):
        for origin in ("python", "PYTHON", " Python ", "stream", "STREAM"):
            with self.subTest(origin=origin):
                self.assertFalse(keep_slicer_entry("Error", origin))

    def test_missing_or_unprintable_values_are_not_kept_and_do_not_raise(self):
        class Unprintable:
            def __str__(self):
                raise RuntimeError("no text")

        self.assertFalse(keep_slicer_entry(None, None))
        self.assertFalse(keep_slicer_entry(Unprintable(), "VTK"))
        self.assertFalse(keep_slicer_entry("Error", Unprintable()))
        self.assertTrue(keep_slicer_entry(" warning ", None))


class RecordExternalTest(_Fixture):
    def lines(self, path):
        return self.read(path).splitlines()

    def test_the_line_names_time_level_origin_and_message(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.record_external("Warning", "VTK", "vtkSegmentationHistory: no state to restore")
        self.assertRegex(
            self.read(path),
            re.compile("^" + _STAMP + r" WARNING \[VTK\] vtkSegmentationHistory: no state to restore$", re.M),
        )

    def test_it_goes_to_the_file_only_never_through_logging(self):
        root = logging.getLogger()
        root_capture, private_capture = _Capture(), _Capture()
        self.addCleanup(root.setLevel, root.level)
        self.addCleanup(root.removeHandler, root_capture)
        root.setLevel(logging.DEBUG)
        root.addHandler(root_capture)
        self.logger.addHandler(private_capture)
        log = SessionLog(self.source_dir, logger=self.logger)
        log.record_external("Error", "Qt", "buffered entry")
        path = log.attach(self.folder())
        log.record_external("Error", "Qt", "attached entry")
        log.record_external("Error", "Qt", "attached entry")
        log.detach()
        self.assertEqual(root_capture.records, [])
        self.assertEqual(private_capture.records, [])
        text = self.read(path)
        self.assertIn("buffered entry", text)
        self.assertIn("attached entry", text)
        self.assertIn("repeated 1 more times", text)

    def test_entries_before_attach_are_buffered_in_order_with_records(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        self.emit("record one")
        log.record_external("Error", "Qt", "qt entry")
        self.emit("record two")
        text = self.read(log.attach(self.folder()))
        header_end = text.index("=" * 72, text.index("GTReview session started"))
        self.assertLess(header_end, text.index("record one"))
        self.assertLess(text.index("record one"), text.index("qt entry"))
        self.assertLess(text.index("qt entry"), text.index("record two"))

    def test_entries_share_the_bounded_buffer(self):
        log = SessionLog(self.source_dir, logger=self.logger, buffered_records=3)
        for n in range(5):
            log.record_external("Warning", "VTK", "entry {}".format(n))
        text = self.read(log.attach(self.folder()))
        self.assertNotIn("entry 1", text)
        self.assertIn("entry 2", text)
        self.assertIn("entry 4", text)

    def test_consecutive_duplicates_are_counted_until_a_different_entry_arrives(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        for _ in range(4):
            log.record_external("Warning", "VTK", "same warning")
        text = self.read(path)
        self.assertEqual(text.count("same warning"), 1)
        self.assertNotIn("repeated", text)
        log.record_external("Error", "Qt", "another entry")
        lines = self.lines(path)
        first = next(i for i, line in enumerate(lines) if line.endswith("same warning"))
        self.assertRegex(lines[first + 1], "^" + _STAMP + r" WARNING \[VTK\] \.\.\. repeated 3 more times$")
        self.assertTrue(lines[first + 2].endswith("ERROR [Qt] another entry"))

    def test_the_count_is_written_on_detach(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        for _ in range(3):
            log.record_external("Warning", "VTK", "same warning")
        log.detach()
        self.assertIn("... repeated 2 more times", self.read(path))

    def test_the_count_is_written_on_close(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        for _ in range(3):
            log.record_external("Warning", "VTK", "same warning")
        log.close()
        self.assertIn("... repeated 2 more times", self.read(path))
        log.record_external("Warning", "VTK", "after close")
        self.assertNotIn("after close", self.read(path))

    def test_duplicates_before_attach_are_counted_too(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        for _ in range(3):
            log.record_external("Warning", "VTK", "early warning")
        text = self.read(log.attach(self.folder()))
        self.assertEqual(text.count("early warning"), 1)
        self.assertLess(text.index("early warning"), text.index("... repeated 2 more times"))

    def test_another_level_or_origin_is_not_a_repeat(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.record_external("Warning", "VTK", "message")
        log.record_external("Error", "VTK", "message")
        log.record_external("Error", "Qt", "message")
        text = self.read(path)
        self.assertEqual(text.count("] message"), 3)
        self.assertNotIn("repeated", text)

    def test_a_record_between_duplicates_ends_the_count_where_it_happened(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.record_external("Warning", "VTK", "same warning")
        log.record_external("Warning", "VTK", "same warning")
        self.emit("GTReview: case loaded")
        log.record_external("Warning", "VTK", "same warning")
        text = self.read(path)
        self.assertLess(text.index("repeated 1 more times"), text.index("GTReview: case loaded"))
        self.assertEqual(text.count("same warning"), 2)
        self.assertLess(text.index("GTReview: case loaded"), text.rindex("same warning"))

    def test_multi_line_messages_keep_their_lines_indented(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.record_external(
            "Warning",
            "VTK",
            "Generic Warning: In vtkThing.cxx, line 12\r\nvtkThing (0x1): nothing to undo\n\n",
        )
        log.record_external("Error", "Qt", "next entry")
        lines = self.lines(path)
        first = next(i for i, line in enumerate(lines) if "Generic Warning" in line)
        self.assertRegex(
            lines[first], "^" + _STAMP + r" WARNING \[VTK\] Generic Warning: In vtkThing.cxx, line 12$"
        )
        self.assertEqual(lines[first + 1], "    vtkThing (0x1): nothing to undo")
        self.assertTrue(lines[first + 2].endswith("ERROR [Qt] next entry"))

    def test_it_never_raises(self):
        class Unprintable:
            def __str__(self):
                raise RuntimeError("no text")

        log = SessionLog(self.source_dir, logger=self.logger)
        log.record_external(Unprintable(), "VTK", "x")
        log.record_external(None, None, None)
        log.attach(self.folder())
        log._file_handler.stream.close()
        log.record_external("Warning", "VTK", "into a closed file")
        log.record_external("Error", "VTK", "into a closed file")


class ExcepthookTest(_Fixture):
    def setUp(self):
        super().setUp()
        self.previous_calls = []
        sys.excepthook = self.previous = lambda *args: self.previous_calls.append(args)

    def test_an_exception_through_gtreview_is_written_and_passed_on(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.install_excepthook()
        self.assertIsNot(sys.excepthook, self.previous)
        info = self.raised_in(self.inside, "in a Qt slot")
        sys.excepthook(*info)
        self.assertEqual(self.previous_calls, [info])
        text = self.read(path)
        self.assertRegex(text, re.compile("^" + _STAMP + " ERROR Uncaught exception$", re.M))
        self.assertIn("Traceback (most recent call last):", text)
        self.assertIn(self.inside, text)
        self.assertIn("RuntimeError: in a Qt slot", text)

    def test_an_exception_elsewhere_is_only_passed_on(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.install_excepthook()
        info = self.raised_in(self.outside)
        sys.excepthook(*info)
        self.assertEqual(self.previous_calls, [info])
        self.assertNotIn("Uncaught exception", self.read(path))

    def test_an_exception_before_attach_is_buffered(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        log.install_excepthook()
        sys.excepthook(*self.raised_in(self.inside, "during setup"))
        text = self.read(log.attach(self.folder()))
        self.assertLess(text.index("GTReview session started"), text.index("Uncaught exception"))
        self.assertIn("RuntimeError: during setup", text)

    def test_uninstall_restores_the_previous_hook(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        log.install_excepthook()
        log.uninstall_excepthook()
        self.assertIs(sys.excepthook, self.previous)
        log.uninstall_excepthook()
        self.assertIs(sys.excepthook, self.previous)

    def test_uninstall_leaves_a_hook_installed_on_top_in_place(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.install_excepthook()
        ours = sys.excepthook
        top_calls = []

        def top(*args):
            top_calls.append(args)
            ours(*args)

        sys.excepthook = top
        log.uninstall_excepthook()
        self.assertIs(sys.excepthook, top)
        info = self.raised_in(self.inside)
        sys.excepthook(*info)
        self.assertEqual(top_calls, [info])
        self.assertEqual(self.previous_calls, [info])
        self.assertNotIn("Uncaught exception", self.read(path))

    def test_installing_twice_reports_once_and_uninstalls_in_one_step(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.install_excepthook()
        log.install_excepthook()
        info = self.raised_in(self.inside)
        sys.excepthook(*info)
        self.assertEqual(self.read(path).count("Uncaught exception"), 1)
        self.assertEqual(self.previous_calls, [info])
        log.uninstall_excepthook()
        self.assertIs(sys.excepthook, self.previous)

    def test_installing_again_above_another_hook_reports_once(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.install_excepthook()
        ours = sys.excepthook
        sys.excepthook = lambda *args: ours(*args)
        log.install_excepthook()
        info = self.raised_in(self.inside)
        sys.excepthook(*info)
        self.assertEqual(self.read(path).count("Uncaught exception"), 1)
        self.assertEqual(self.previous_calls, [info])

    def test_close_uninstalls(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        log.install_excepthook()
        log.close()
        self.assertIs(sys.excepthook, self.previous)

    def test_a_reloaded_module_reports_once(self):
        first = SessionLog(self.source_dir, logger=self.logger)
        first.install_excepthook()
        second = SessionLog(self.source_dir, logger=self.logger)
        self.assertIs(sys.excepthook, self.previous)
        second.install_excepthook()
        path = second.attach(self.folder())
        info = self.raised_in(self.inside)
        sys.excepthook(*info)
        self.assertEqual(self.read(path).count("Uncaught exception"), 1)
        self.assertEqual(self.previous_calls, [info])

    def test_the_hook_never_raises(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        log.attach(self.folder())
        log.install_excepthook()
        not_a_traceback = object()
        sys.excepthook(RuntimeError, RuntimeError("odd"), not_a_traceback)
        log._file_handler.stream.close()
        info = self.raised_in(self.inside)
        sys.excepthook(*info)
        self.assertEqual(len(self.previous_calls), 2)
        self.assertIs(self.previous_calls[0][2], not_a_traceback)


class CrashReportTest(_Fixture):
    def setUp(self):
        super().setUp()
        # the fixture enables it again afterwards if the runner had it on
        faulthandler.disable()
        # faulthandler itself works on every platform; the tests below that
        # expect crash reports want them whatever the default is here
        self.enabledFlag(True)

    def enabledFlag(self, value):
        patch = mock.patch.object(sessionlog, "CRASH_REPORTS_ENABLED", value)
        patch.start()
        self.addCleanup(patch.stop)

    def test_where_crash_reports_are_off_attach_leaves_faulthandler_alone(self):
        self.enabledFlag(False)
        log = SessionLog(self.source_dir, logger=self.logger)
        with mock.patch.object(sessionlog.faulthandler, "enable", wraps=faulthandler.enable) as enable:
            path = log.attach(self.folder(), ["Crash reports: on", ("cases found", 3)])
        enable.assert_not_called()
        self.assertFalse(faulthandler.is_enabled())
        lines = self.read(path).splitlines()
        self.assertIn("  Crash reports: off (Windows)", lines)
        self.assertEqual(sum("Crash reports" in line for line in lines), 1)
        log.detach()
        self.assertFalse(faulthandler.is_enabled())

    def test_where_crash_reports_are_off_the_watchdog_still_reports(self):
        self.enabledFlag(False)
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        self.assertTrue(log.arm_watchdog(0.2))
        time.sleep(0.6)
        log.disarm_watchdog()
        self.assertIn("Timeout (0:00:00.200000)!", self.read(path))

    def test_where_crash_reports_are_off_a_faulthandler_enabled_before_stays_as_it_was(self):
        self.enabledFlag(False)
        prior = open(os.path.join(self.tmp.name, "stderr.txt"), "w", encoding="utf-8")
        self.addCleanup(prior.close)
        faulthandler.enable(file=prior)
        self.addCleanup(faulthandler.disable)
        log = SessionLog(self.source_dir, logger=self.logger)
        with mock.patch.object(sessionlog.faulthandler, "disable", wraps=faulthandler.disable) as disable:
            log.attach(self.folder())
            log.close()
        disable.assert_not_called()
        self.assertTrue(faulthandler.is_enabled())

    def test_the_flag_is_read_at_every_attach(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        with mock.patch.object(sessionlog, "CRASH_REPORTS_ENABLED", False):
            first = log.attach(self.folder("batch_01"))
            self.assertFalse(faulthandler.is_enabled())
        with mock.patch.object(sessionlog, "CRASH_REPORTS_ENABLED", True):
            second = log.attach(self.folder("batch_02"))
            self.assertTrue(faulthandler.is_enabled())
        self.assertIn("Crash reports: off (Windows)", self.read(first))
        self.assertIn("  Crash reports: on", self.read(second).splitlines())
        log.detach()
        self.assertFalse(faulthandler.is_enabled())

    def test_attach_turns_crash_reports_on_and_detach_off(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        self.assertTrue(faulthandler.is_enabled())
        self.assertIn("  Crash reports: on", self.read(path).splitlines())
        log.detach()
        self.assertFalse(faulthandler.is_enabled())

    def test_close_turns_them_off(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        log.attach(self.folder())
        log.close()
        self.assertFalse(faulthandler.is_enabled())

    def test_a_crash_reports_detail_from_the_caller_is_not_doubled(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder(), [("Crash reports", "on"), "Crash reports: on", ("cases found", 3)])
        text = self.read(path)
        self.assertEqual(text.count("Crash reports"), 1)
        self.assertIn("Crash reports: on", text)
        self.assertIn("cases found", text)

    def test_a_refused_faulthandler_says_off_and_still_logs(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        with mock.patch.object(sessionlog.faulthandler, "enable", side_effect=RuntimeError("refused")):
            path = log.attach(self.folder())
        self.assertIsNotNone(path)
        self.assertIn("  Crash reports: off", self.read(path).splitlines())
        log.detach()

    def test_a_faulthandler_enabled_before_is_enabled_again(self):
        prior = open(os.path.join(self.tmp.name, "stderr.txt"), "w", encoding="utf-8")
        self.addCleanup(prior.close)
        faulthandler.enable(file=prior)
        # closes the stand-in stderr only after faulthandler lets go of it
        self.addCleanup(faulthandler.disable)
        with mock.patch.object(sys, "stderr", prior):
            log = SessionLog(self.source_dir, logger=self.logger)
            log.attach(self.folder("batch_01"))
            log.attach(self.folder("batch_02"))
            log.detach()
            self.assertTrue(faulthandler.is_enabled())
            log.attach(self.folder("batch_01"))
            log.close()
            self.assertTrue(faulthandler.is_enabled())

    # A real crash is the only way to see where faulthandler writes.  The child
    # installs a Python SIGSEGV handler first, so faulthandler writes its
    # report, hands the signal on, and the child leaves with code 7 instead of
    # dying and leaving a core dump behind.
    CHILD = (
        "import faulthandler, os, signal, sys, time\n"
        "module_dir, source_dir, folder, mode = sys.argv[1:5]\n"
        "sys.path.insert(0, module_dir)\n"
        "from GTReviewLib.sessionlog import SessionLog\n"
        "signal.signal(signal.SIGSEGV, lambda signum, frame: os._exit(7))\n"
        "if 'prior' in mode:\n"
        "    faulthandler.enable()\n"
        "log = SessionLog(source_dir)\n"
        "log.attach(folder)\n"
        "if 'detach' in mode:\n"
        "    log.detach()\n"
        "faulthandler._sigsegv()\n"
        "for _ in range(500):\n"
        "    time.sleep(0.01)\n"
        "os._exit(3)\n"
    )

    def crash_child(self, mode):
        if os.name == "nt" or not hasattr(faulthandler, "_sigsegv"):
            self.skipTest("needs POSIX signal chaining and faulthandler._sigsegv")
        script = os.path.join(self.tmp.name, "crash_child.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(self.CHILD)
        folder = self.folder()
        stderr_path = os.path.join(self.tmp.name, "child_stderr.txt")
        with open(stderr_path, "wb") as stderr:
            result = subprocess.run(
                [sys.executable, script, _MODULE_DIR, self.source_dir, folder, mode],
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                timeout=60,
            )
        child_stderr = self.read(stderr_path)
        self.assertEqual(result.returncode, 7, child_stderr)
        return self.read(os.path.join(folder, LOG_FILE_NAME)), child_stderr

    def test_a_crash_while_attached_writes_the_stacks_into_the_log(self):
        log_text, child_stderr = self.crash_child("attached")
        self.assertIn("Crash reports: on", log_text)
        self.assertIn("Fatal Python error: Segmentation fault", log_text)
        self.assertRegex(log_text, r'File ".*crash_child\.py", line \d+ in <module>')
        self.assertNotIn("Fatal Python error", child_stderr)

    def test_a_crash_after_detach_is_not_in_the_log(self):
        log_text, child_stderr = self.crash_child("detach")
        self.assertNotIn("Fatal Python error", log_text)
        self.assertNotIn("Fatal Python error", child_stderr)

    def test_a_crash_after_detach_goes_where_it_went_before(self):
        log_text, child_stderr = self.crash_child("prior detach")
        self.assertNotIn("Fatal Python error", log_text)
        self.assertIn("Fatal Python error: Segmentation fault", child_stderr)


class WatchdogTest(_Fixture):
    def attached(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        return log, log.attach(self.folder())

    def test_a_thread_sleeping_past_the_timeout_is_reported(self):
        log, path = self.attached()
        self.assertTrue(log.arm_watchdog(0.2))
        time.sleep(0.6)
        text = self.read(path)
        log.disarm_watchdog()
        self.assertIn("Timeout (0:00:00.200000)!", text)
        self.assertRegex(
            text,
            r'File ".*test_sessionlog\.py", line \d+ in '
            r"test_a_thread_sleeping_past_the_timeout_is_reported",
        )

    def test_rearming_in_time_writes_nothing(self):
        log, path = self.attached()
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            log.arm_watchdog(0.5)
            time.sleep(0.05)
        log.disarm_watchdog()
        self.assertNotIn("Timeout", self.read(path))

    def test_a_disarmed_watchdog_writes_nothing(self):
        log, path = self.attached()
        log.arm_watchdog(0.2)
        log.disarm_watchdog()
        time.sleep(0.5)
        self.assertNotIn("Timeout", self.read(path))

    def test_without_a_file_it_is_not_armed(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        self.assertFalse(log.arm_watchdog(0.2))
        path = log.attach(self.folder())
        time.sleep(0.5)
        self.assertNotIn("Timeout", self.read(path))

    @staticmethod
    def spy_on_cancel():
        return mock.patch.object(
            sessionlog.faulthandler,
            "cancel_dump_traceback_later",
            wraps=faulthandler.cancel_dump_traceback_later,
        )

    def assertNoStrayDump(self, path):
        # The closed log's descriptor number is usually the next one handed
        # out; a watchdog still armed would write its dump into this file.
        reused = os.path.join(self.tmp.name, "opened_after.txt")
        with open(reused, "w", encoding="utf-8"):
            time.sleep(0.5)
        self.assertNotIn("Timeout", self.read(reused))
        self.assertNotIn("Timeout", self.read(path))

    def test_detach_disarms_and_a_detached_log_cannot_be_armed(self):
        log, path = self.attached()
        log.arm_watchdog(0.2)
        with self.spy_on_cancel() as cancel:
            log.detach()
        cancel.assert_called()
        self.assertFalse(log.arm_watchdog(0.2))
        self.assertNoStrayDump(path)

    def test_close_disarms(self):
        log, path = self.attached()
        log.arm_watchdog(0.2)
        with self.spy_on_cancel() as cancel:
            log.close()
        cancel.assert_called()
        self.assertNoStrayDump(path)


class SizeLimitTest(_Fixture):
    """Slicer's entries stop at ``max_bytes``; GTReview's own lines never do."""

    LIMIT = 8000
    NOTICE = "further Slicer warnings are not copied into it this session"

    def entry(self, n):
        return "vtkThing {:03d}: {}".format(n, "x" * 80)

    def flood(self, log, count=200):
        for n in range(count):
            log.record_external("Warning", "VTK", self.entry(n))

    def test_slicer_entries_stop_with_one_line_that_says_where_the_rest_are(self):
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=self.LIMIT)
        path = log.attach(self.folder())
        self.flood(log)
        text = self.read(path)
        lines = text.splitlines()
        notices = [i for i, line in enumerate(lines) if self.NOTICE in line]
        self.assertEqual(len(notices), 1, text)
        self.assertRegex(
            lines[notices[0]],
            "^" + _STAMP + r" WARNING \[GTReview\] GTReview\.log has reached 7\.8 KB: "
            + re.escape(self.NOTICE) + r"; Slicer's own log \(\"Slicer log\" in the header\) has them$",
        )
        written = [n for n in range(200) if self.entry(n) in text]
        self.assertTrue(written)
        # written up to the limit, none after
        self.assertEqual(written, list(range(len(written))))
        self.assertLess(len(written), 200)
        self.assertLessEqual(len(text.encode("utf-8")) - len(lines[notices[0]]) - 1, self.LIMIT)
        self.assertTrue(lines[notices[0] - 1].endswith(self.entry(len(written) - 1)))

    def test_gtreview_lines_are_still_written_after_the_stop(self):
        sys.excepthook = lambda *args: None
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=self.LIMIT)
        path = log.attach(self.folder())
        log.install_excepthook()
        self.flood(log)
        self.emit("GTReview: case saved " + "y" * 9000)
        sys.excepthook(*self.raised_in(self.inside, "after the stop"))
        log.record_external("Error", "Qt", "not copied either")
        text = self.read(path)
        self.assertIn("GTReview: case saved yyy", text)
        self.assertIn("RuntimeError: after the stop", text)
        self.assertNotIn("not copied either", text)
        self.assertEqual(text.count(self.NOTICE), 1)

    def test_repeats_after_the_stop_are_not_counted_or_written(self):
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=self.LIMIT)
        path = log.attach(self.folder())
        self.flood(log)
        for _ in range(5):
            log.record_external("Warning", "VTK", "same warning")
        self.emit("GTReview: in between")
        log.detach()
        text = self.read(path)
        self.assertNotIn("same warning", text)
        self.assertNotIn("repeated", text)

    def test_the_limit_counts_what_the_file_held_before_the_session(self):
        folder = self.folder()
        with open(os.path.join(folder, LOG_FILE_NAME), "w", encoding="utf-8") as handle:
            handle.write("p" * (self.LIMIT - 1))
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=self.LIMIT)
        path = log.attach(folder)
        self.assertFalse(os.path.exists(path + ".1"), "a file under the limit is not rotated")
        log.record_external("Warning", "VTK", "first entry")
        self.emit("GTReview: still written")
        text = self.read(path)
        self.assertNotIn("first entry", text)
        self.assertIn(self.NOTICE, text)
        self.assertIn("GTReview: still written", text)

    def test_entries_buffered_before_attach_count_against_the_limit(self):
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=self.LIMIT)
        self.flood(log)
        self.emit("GTReview: logged after the flood")
        text = self.read(log.attach(self.folder()))
        self.assertIn(self.entry(0), text)
        self.assertNotIn(self.entry(199), text)
        self.assertEqual(text.count(self.NOTICE), 1)
        self.assertLess(text.index(self.NOTICE), text.index("GTReview: logged after the flood"))

    def test_the_next_session_copies_slicer_entries_again(self):
        log = SessionLog(self.source_dir, logger=self.logger, max_bytes=self.LIMIT)
        first = log.attach(self.folder("batch_01"))
        self.flood(log)
        log.record_external("Warning", "VTK", "in the first batch")
        second = log.attach(self.folder("batch_02"))
        log.record_external("Warning", "VTK", "in the second batch")
        self.assertNotIn("in the first batch", self.read(first))
        self.assertIn("in the second batch", self.read(second))
        self.assertNotIn(self.NOTICE, self.read(second))

    def test_the_count_follows_the_file(self):
        log = SessionLog(self.source_dir, logger=self.logger)
        sys.excepthook = lambda *args: None
        folder = self.folder()
        with open(os.path.join(folder, LOG_FILE_NAME), "w", encoding="utf-8") as handle:
            handle.write("previous session\n")
        self.emit("buffered Läsion")
        path = log.attach(folder)
        log.install_excepthook()
        self.emit("Ödem 病変\nsecond line")
        log.record_external("Warning", "VTK", "multi\nline 病変")
        log.record_external("Warning", "VTK", "multi\nline 病変")
        log.record_external("Error", "Qt", "next")
        sys.excepthook(*self.raised_in(self.inside))
        log.record_external("Error", "Qt", "open run")
        log.record_external("Error", "Qt", "open run")
        self.assertEqual(log._bytes, os.path.getsize(path))
        log.detach()  # writes the open run's count
        self.assertEqual(log._bytes, 0)
        size = os.path.getsize(path)
        log.attach(folder)
        self.assertGreater(log._bytes, size)
        self.assertEqual(log._bytes, os.path.getsize(path))

    def test_the_default_limit_is_5_mb(self):
        self.assertEqual(MAX_BYTES, 5 * 1024 * 1024)
        folder = self.folder()
        with open(os.path.join(folder, LOG_FILE_NAME), "w", encoding="utf-8") as handle:
            handle.write("p" * (MAX_BYTES - 1))
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(folder)
        log.record_external("Warning", "VTK", "one too many")
        with open(path, encoding="utf-8") as handle:
            handle.seek(MAX_BYTES - 1)
            tail = handle.read()
        self.assertIn("GTReview.log has reached 5 MB: " + self.NOTICE, tail)


class FlushTest(_Fixture):
    def test_every_kind_of_line_is_in_the_file_while_still_attached(self):
        sys.excepthook = lambda *args: None
        patch = mock.patch.object(sessionlog, "CRASH_REPORTS_ENABLED", True)
        patch.start()
        self.addCleanup(patch.stop)
        log = SessionLog(self.source_dir, logger=self.logger)
        path = log.attach(self.folder())
        log.install_excepthook()
        self.emit("a record")
        log.record_external("Warning", "VTK", "an external entry")
        log.record_external("Warning", "VTK", "an external entry")
        log.record_external("Error", "Qt", "a different entry")
        sys.excepthook(*self.raised_in(self.inside, "an uncaught one"))
        text = self.read(path)
        for expected in (
            "Crash reports: on",
            "a record",
            "an external entry",
            "repeated 1 more times",
            "a different entry",
            "RuntimeError: an uncaught one",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
