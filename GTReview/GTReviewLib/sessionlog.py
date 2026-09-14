"""Keep GTReview's log in a file next to the batch being reviewed -- pure python.

When a reviewer runs into a problem, the evidence that matters is what
GTReview did and logged in that session: which case, which edits, which
exception.  Slicer's own log has all of it, but mixed with every other module,
rotated away after a few sessions and kept in a temporary folder nobody finds.
:class:`SessionLog` copies GTReview's records into ``GTReview.log`` in the
folder the reviewer loads, so the log travels with the data and can be sent as
it is.

Slicer runs its root logger at DEBUG with one handler of its own, and GTReview
logs through the root logger, so no call site changes: a handler added to the
root logger picks GTReview's records out by the source file they were logged
from.  Records logged before a folder is chosen wait in a bounded buffer,
already formatted, and are written once one is.

Some evidence never passes through Python's logging: VTK and Qt warnings
(handed in by the panel through :meth:`SessionLog.record_external`), uncaught
exceptions in Qt slots and VTK observers (:meth:`SessionLog.install_excepthook`),
and hard crashes or a frozen main thread, which only :mod:`faulthandler` can
report because it writes from C without needing the interpreter to run.  Crash
reports stay off on Windows (see :data:`CRASH_REPORTS_ENABLED`); the report of
a frozen main thread works everywhere.

The file moves to ``GTReview.log.1`` only when a session starts, because
Windows refuses to rename a file that is open.  Within a session, Slicer's
warnings stop being copied once they would take the file past its size limit,
so a warning flood cannot grow it without bound; GTReview's own lines keep
coming.

Nothing here imports ``slicer``.
"""

from __future__ import annotations

import collections
import datetime
import faulthandler
import logging
import os
import platform
import re
import sys
import threading
import time
import traceback
from typing import List, Optional, Sequence, Tuple, Union

#: name of the log file written into the loaded folder
LOG_FILE_NAME = "GTReview.log"
#: a log at least this large is moved to ``GTReview.log.1`` when a session starts;
#: within a session, Slicer's warnings are no longer copied past this size
MAX_BYTES = 5 * 1024 * 1024
#: records kept from before a folder is loaded
BUFFERED_RECORDS = 2000
#: one line per record: when, how bad, where from, what
FORMAT = "%(asctime)s %(levelname)-8s %(module)s:%(lineno)d  %(message)s"
#: a main thread that has not re-armed the watchdog for this long is reported as hung
WATCHDOG_TIMEOUT_S = 15.0
#: how often the panel re-arms the watchdog; well inside the timeout, so only a
#: blocked event loop lets it expire
WATCHDOG_REARM_MS = 5000
#: whether :meth:`SessionLog.attach` has faulthandler write crash reports into
#: the file; read at every attach.  Off on Windows: faulthandler.enable() there
#: installs a vectored exception handler that also reports first-chance
#: exceptions the program goes on to handle -- the native file dialog raises
#: 0x8001010d -- as "Windows fatal exception" with every thread's stack, while
#: Slicer keeps running.  The watchdog (dump_traceback_later) is not affected.
CRASH_REPORTS_ENABLED = os.name != "nt"
#: the comment Packaging/make_package.sh writes into the .s4ext ahead of the
#: build's ``git describe --dirty``: scmrevision alone names the commit and
#: hides uncommitted changes
BUILD_VERSION_COMMENT = "gtreview-version"

#: marks the handler, so a reloaded module can find the previous instance's
_HANDLER_MARK = "is_gtreview_session_log"
#: Slicer error log levels worth keeping from other sources
_KEPT_LEVELS = frozenset(("warning", "error", "critical", "fatal"))
#: Slicer error log origins that are never copied: Python records from GTReview
#: are already written by the logging handler (and other modules' are not
#: ours), and "Stream" is anything printed to the console.
_SKIPPED_ORIGINS = frozenset(("python", "stream"))
#: continuation lines of a multi-line entry
_INDENT = "    "
#: a header detail is either (key, value) or a line written as it is
Detail = Union[Tuple[str, object], str]


def _normalized(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _asctime(created: float) -> str:
    """*created* the way :data:`FORMAT` prints ``%(asctime)s``, so every line
    of the file carries the same kind of time stamp."""
    stamp = time.strftime(logging.Formatter.default_time_format, time.localtime(created))
    return logging.Formatter.default_msec_format % (stamp, int((created - int(created)) * 1000))


def _file_bytes(text: str) -> int:
    """How many bytes *text* takes in the file: UTF-8, and a text stream turns
    each newline into ``os.linesep``."""
    size = len(text.encode("utf-8", "replace"))
    if os.linesep != "\n":
        size += text.count("\n") * (len(os.linesep) - 1)
    return size


def _size_text(size: int) -> str:
    megabytes = size / (1024 * 1024)
    if megabytes >= 1:
        return "{:g} MB".format(round(megabytes, 1))
    if size >= 1024:
        return "{:g} KB".format(round(size / 1024, 1))
    return "{} bytes".format(size)


def keep_slicer_entry(level_text: object, origin: object) -> bool:
    """Whether an entry of Slicer's error log belongs in ``GTReview.log``.

    Warnings and worse from VTK, Qt and the like are kept: they are often the
    only trace of why a segmentation or a view misbehaved.  Python entries are
    skipped because GTReview's own records reach the file through logging
    already, and console output ("Stream") is not a diagnosis.
    """
    try:
        if str(origin or "").strip().lower() in _SKIPPED_ORIGINS:
            return False
        return str(level_text or "").strip().lower() in _KEPT_LEVELS
    except Exception:  # noqa: BLE001 - an unreadable entry is simply not kept
        return False


class SourceFilter(logging.Filter):
    """Pass records logged from a file under *source_dir*, or whose message
    starts with *prefix*."""

    def __init__(self, source_dir: str, prefix: str = "GTReview"):
        super().__init__()
        root = _normalized(source_dir)
        self._root = root if root.endswith(os.sep) else root + os.sep
        self._prefix = prefix

    def covers(self, path: str) -> bool:
        """Whether *path* is a file under the source folder."""
        return bool(path) and _normalized(path).startswith(self._root)

    def filter(self, record: logging.LogRecord) -> bool:
        pathname = getattr(record, "pathname", "") or ""
        if self.covers(pathname):
            return True
        if not self._prefix:
            return False
        try:
            return str(record.getMessage()).startswith(self._prefix)
        except Exception:  # noqa: BLE001 - a malformed record is simply not ours
            return False


def rotate_if_large(path: str, max_bytes: int = MAX_BYTES) -> bool:
    """Move *path* to ``path + ".1"`` once it has reached *max_bytes*.

    Only done when a session starts, never while the file is open: Windows
    refuses to rename a file another handle holds.  Returns whether it moved.
    """
    try:
        if os.path.getsize(path) < max_bytes:
            return False
    except OSError:
        return False
    try:
        os.replace(path, path + ".1")
        return True
    except OSError:
        return False


def session_header(details: Sequence[Detail] = ()) -> List[str]:
    """Lines that open a session in the log: when, where it runs, then *details*.

    A detail is a ``(key, value)`` pair, written as an aligned column, or a
    plain string, written as it is.
    """
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "",
        "=" * 72,
        "GTReview session started {}".format(now),
        "  {:<16} {}".format("platform", platform.platform()),
        "  {:<16} {}".format("python", sys.version.split()[0]),
    ]
    for detail in details:
        if isinstance(detail, str):
            lines.append("  " + detail)
        else:
            key, value = detail
            lines.append("  {:<16} {}".format(key, value))
    lines.append("=" * 72)
    return lines


def _names_crash_reports(detail: Detail) -> bool:
    try:
        text = detail if isinstance(detail, str) else detail[0]
        return str(text).strip().lower().startswith("crash report")
    except Exception:  # noqa: BLE001 - a malformed detail is left for the header to show
        return False


def _extension_layout(module_file: str, extension_name: str) -> Tuple[str, str, bool]:
    """``(extension root, description file, installed)`` for *module_file*.

    Installed, the module sits at ``<ext>/lib/Slicer-X.Y/qt-scripted-modules/<name>.py``
    and its description at ``<ext>/share/Slicer-X.Y/<name>.s4ext``; *installed*
    says whether the folders around the module have those names.
    """
    scripted_dir = os.path.dirname(os.path.abspath(module_file))
    lib_slicer_dir = os.path.dirname(scripted_dir)
    lib_dir = os.path.dirname(lib_slicer_dir)
    extension_root = os.path.dirname(lib_dir)
    slicer_dir_name = os.path.basename(lib_slicer_dir)
    description = os.path.join(extension_root, "share", slicer_dir_name, extension_name + ".s4ext")
    installed = (
        os.path.basename(scripted_dir).lower() == "qt-scripted-modules"
        and re.match(r"slicer-\d+\.\d+$", slicer_dir_name.lower()) is not None
        and os.path.basename(lib_dir).lower() == "lib"
    )
    return extension_root, description, installed


def _read_description(path: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """``(scmrevision, build version)`` from an .s4ext file, or None when it cannot be read.

    Keys and values are split at the first run of blanks, as Slicer parses the
    file; the build version is the ``# gtreview-version`` comment.
    """
    revision = version = None
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text.startswith("#"):
                    parts = text[1:].strip().split(None, 1)
                    if len(parts) == 2 and parts[0] == BUILD_VERSION_COMMENT and version is None:
                        version = parts[1].strip()
                    continue
                parts = text.split(None, 1)
                if len(parts) == 2 and parts[0] == "scmrevision" and revision is None:
                    revision = parts[1].strip()
    except (OSError, ValueError):  # ValueError: not UTF-8
        return None
    return revision, version


def installed_revision(module_file: str, extension_name: str = "GTReview") -> Optional[str]:
    """The ``scmrevision`` of an installed extension, or None when run from a source tree.

    Installed, the module sits at ``<ext>/lib/Slicer-X.Y/qt-scripted-modules/<name>.py``
    and its description at ``<ext>/share/Slicer-X.Y/<name>.s4ext``.
    """
    _, description, _ = _extension_layout(module_file, extension_name)
    fields = _read_description(description)
    return fields[0] if fields is not None else None


def installed_build(module_file: str, extension_name: str = "GTReview") -> Optional[str]:
    """Which GTReview build runs, for the header of a session; None in a source tree.

    ``"<version> (<scmrevision>)"`` when the installed .s4ext carries the
    ``# gtreview-version`` comment Packaging/make_package.sh writes -- the
    version says ``-dirty`` for a package built from uncommitted changes, which
    the revision cannot -- else the scmrevision alone.  An installed layout
    whose description cannot be read, or names neither, is
    ``"unknown build at <extension folder>"``.
    """
    extension_root, description, installed = _extension_layout(module_file, extension_name)
    if not installed:
        return None
    revision, version = _read_description(description) or (None, None)
    if version and revision:
        return "{} ({})".format(version, revision)
    if version or revision:
        return version or revision
    return "unknown build at {}".format(extension_root)


class _SessionHandler(logging.Handler):
    def __init__(self, owner: "SessionLog"):
        super().__init__(logging.DEBUG)
        self.owner = owner
        setattr(self, _HANDLER_MARK, True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.owner._emit(record)
        except Exception:  # noqa: BLE001 - logging must never raise into GTReview
            self.handleError(record)

    def close(self) -> None:
        # logging.shutdown() at exit, or anyone else closing this handler,
        # must close the log file too, or it stays open until the process dies
        try:
            self.owner.detach()
        finally:
            super().close()


class _LogFileHandler(logging.FileHandler):
    def __init__(self, owner: "SessionLog", path: str):
        super().__init__(path, mode="a", encoding="utf-8")
        self.owner = owner

    def close(self) -> None:
        # faulthandler writes from C straight to this file's descriptor.  It
        # has to let go before the descriptor closes -- also when
        # logging.shutdown() closes this handler before the session handler --
        # or a crash or hang report could land in whichever file reuses the
        # number.  A handler of an earlier folder, closed again at exit, must
        # not switch off the reports of the file open now.
        try:
            if self.owner._file_handler is self:
                self.owner._stop_fault_reports()
        except Exception:  # noqa: BLE001
            pass
        super().close()


class _Run:
    """Consecutive identical external entries: the first is written, the rest counted."""

    __slots__ = ("key", "repeats", "last_created")

    def __init__(self, key: Tuple[str, str, str], created: float):
        self.key = key
        self.repeats = 0
        self.last_created = created


class SessionLog:
    """GTReview's records, buffered until :meth:`attach` names a folder, then
    appended to ``<folder>/GTReview.log``.

    While attached, a hard crash writes every thread's stack into the file
    (except on Windows, see :data:`CRASH_REPORTS_ENABLED`), and
    :meth:`arm_watchdog` can have a hung main thread do the same.  Every line
    is flushed as it is written so it survives the process dying right after.
    """

    def __init__(
        self,
        source_dir: str,
        logger: Optional[logging.Logger] = None,
        file_name: str = LOG_FILE_NAME,
        max_bytes: int = MAX_BYTES,
        buffered_records: int = BUFFERED_RECORDS,
    ):
        self.file_name = file_name
        self.max_bytes = int(max_bytes)
        self.path: Optional[str] = None
        self._logger = logger if logger is not None else logging.getLogger()
        self._formatter = logging.Formatter(FORMAT)
        self._source = SourceFilter(source_dir)
        # Everything waits here as (external, finished text), records and
        # lines that bypass logging alike, so they come out in the order they
        # arrived.  Never a LogRecord: its exc_info would keep the traceback
        # alive, and with it every frame's locals -- mask arrays, the widget --
        # for as long as no folder is loaded.
        self._pending: collections.deque = collections.deque(maxlen=int(buffered_records))
        self._file_handler: Optional[_LogFileHandler] = None
        # bytes in the attached file, and whether Slicer's entries have hit max_bytes
        self._bytes = 0
        self._external_stopped = False
        # Records may be logged from worker threads while the main thread
        # writes an external entry; one re-entrant lock keeps lines whole and
        # in order.
        self._lock = threading.RLock()
        self._run: Optional[_Run] = None
        self._closed = False
        self._previous_excepthook = None
        self._excepthook = None
        self._crash_reports = False
        self._crash_reports_were_enabled = False
        self._watchdog_armed = False
        # A module reload (Slicer's developer "Reload") builds a new instance
        # without the old widget's cleanup having run; drop its handler so
        # records are not written twice.
        for handler in list(self._logger.handlers):
            if getattr(handler, _HANDLER_MARK, False):
                try:
                    handler.owner.close()
                except Exception:  # noqa: BLE001
                    self._logger.removeHandler(handler)
        self._handler = _SessionHandler(self)
        self._handler.addFilter(self._source)
        self._logger.addHandler(self._handler)

    # ----------------------------------------------------------------- file

    def attach(self, folder: str, details: Sequence[Detail] = ()) -> Optional[str]:
        """Start writing to ``<folder>/GTReview.log``; returns its path, or None.

        None means the folder cannot be written (read-only share, no such
        folder): records keep buffering and nothing raises, because a log must
        never stop a reviewer from loading cases.

        The header ends with a ``Crash reports: on`` line (``off`` when
        faulthandler refused the file, ``off (Windows)`` where
        :data:`CRASH_REPORTS_ENABLED` keeps them off); a crash-reports entry in
        *details* is replaced by it, so the header never claims a state that
        does not hold.
        """
        target = os.path.join(folder, self.file_name)
        with self._lock:
            if self.path is not None and _normalized(self.path) == _normalized(target):
                return self.path
            self.detach()
            rotate_if_large(target, self.max_bytes)
            try:
                handler = _LogFileHandler(self, target)
            except (OSError, ValueError):
                return None
            try:
                size = os.path.getsize(target)
            except OSError:
                size = 0
            enabled = CRASH_REPORTS_ENABLED
            crash_reports = enabled and self._start_crash_reports(handler.stream)
            lines = [detail for detail in details if not _names_crash_reports(detail)]
            lines.append(
                "Crash reports: {}".format("on" if crash_reports else "off" if enabled else "off (Windows)")
            )
            header = "\n".join(session_header(lines)) + "\n"
            try:
                handler.stream.write(header)
                handler.flush()
            except Exception:  # noqa: BLE001 - an unwritable file is a folder we cannot log to
                self._stop_fault_reports()
                try:
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass
                return None
            self._file_handler = handler
            self.path = target
            self._bytes = size + _file_bytes(header)
            self._external_stopped = False
            while self._pending:
                external, text = self._pending.popleft()
                self._write_line(text, external)
            return target

    def detach(self) -> None:
        """Stop writing to the current file; later records buffer again."""
        with self._lock:
            try:
                self._end_run()
            except Exception:  # noqa: BLE001
                pass
            handler, self._file_handler = self._file_handler, None
            self.path = None
            self._bytes = 0
            self._external_stopped = False
            self._stop_fault_reports()
            if handler is not None:
                try:
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        """Detach and take the handler and the exception hook off for good."""
        self._closed = True
        self.uninstall_excepthook()
        self.detach()
        self._logger.removeHandler(self._handler)
        self._pending.clear()

    def _emit(self, record: logging.LogRecord) -> None:
        # Formatted now, attached or not, so the buffer holds text only: the
        # traceback, the arguments and whatever else the record refers to are
        # let go as soon as logging is done with it.  The time stamp comes
        # from record.created, so a line written later still says when.
        text = self._formatter.format(record)
        with self._lock:
            # A record between two identical external entries splits them, so
            # the count is written before the record, where it happened.
            self._end_run()
            self._write_line(text)

    def _write_line(self, text: str, external: bool = False) -> bool:
        """*text* into the file, flushed, or into the buffer until one is attached.

        An *external* line -- Slicer's entries and their repeat counts -- that
        would take the file past ``max_bytes`` is not written, nor is any
        external line after it until the next :meth:`attach`; one line says so
        instead.  GTReview's own lines are always written.  Returns whether
        *text* was written or buffered.
        """
        handler = self._file_handler
        if handler is None:
            self._pending.append((external, text))
            return True
        stream = handler.stream
        if stream is None:
            return False
        line = text + "\n"
        if external and (self._external_stopped or self._bytes + _file_bytes(line) > self.max_bytes):
            if not self._external_stopped:
                self._external_stopped = True
                self._write(stream, self._stop_notice() + "\n")
            return False
        self._write(stream, line)
        return True

    def _write(self, stream, line: str) -> None:
        stream.write(line)
        stream.flush()
        self._bytes += _file_bytes(line)

    def _stop_notice(self) -> str:
        return (
            "{} WARNING [GTReview] {} has reached {}: further Slicer warnings are not copied "
            "into it this session; Slicer's own log (\"Slicer log\" in the header) has them"
        ).format(_asctime(time.time()), self.file_name, _size_text(self.max_bytes))

    # ------------------------------------------------------ external entries

    def record_external(self, level_text: object, origin: object, message: object) -> None:
        """Write one entry of Slicer's error log (VTK, Qt, ...) into the file.

        It goes to the file only, never through :mod:`logging`: Slicer's own
        handler would put it back into the error log it came from and hand it
        to the panel again.  A burst of the same warning -- VTK repeats one per
        render -- is written once, followed by how many more times it came.

        Once these entries would take the file past ``max_bytes``, they stop
        being written for the rest of the session (see :meth:`_write_line`).
        """
        try:
            created = time.time()
            level = str(level_text if level_text is not None else "")
            where = str(origin if origin is not None else "")
            text = str(message if message is not None else "")
            key = (level, where, text)
            with self._lock:
                if self._closed or (self._external_stopped and self._file_handler is not None):
                    return
                if self._run is not None and self._run.key == key:
                    self._run.repeats += 1
                    self._run.last_created = created
                    return
                self._end_run()
                written = self._write_line(self._external_line(created, level, where, text), external=True)
                self._run = _Run(key, created) if written else None
        except Exception:  # noqa: BLE001 - the log must never raise into GTReview
            pass

    @staticmethod
    def _external_line(created: float, level: str, origin: str, message: str) -> str:
        lines = [line.rstrip() for line in message.splitlines()]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        first = "{} {} [{}] {}".format(
            _asctime(created), level.strip().upper(), origin.strip(), lines[0] if lines else ""
        )
        return "\n".join([first.rstrip()] + [_INDENT + line for line in lines[1:]])

    def _end_run(self) -> None:
        run, self._run = self._run, None
        if run is not None and run.repeats:
            level, origin, _ = run.key
            self._write_line(
                "{} {} [{}] ... repeated {} more times".format(
                    _asctime(run.last_created), level.strip().upper(), origin.strip(), run.repeats
                ),
                external=True,
            )

    # -------------------------------------------------- uncaught exceptions

    def install_excepthook(self) -> None:
        """Copy uncaught exceptions that pass through GTReview's code into the file.

        Slicer reports exceptions escaping a Qt slot or a VTK observer through
        ``sys.excepthook`` and carries on, so without this they would reach
        only the Python console.  The hook it replaces still runs afterwards.
        """
        if self._excepthook is not None and sys.excepthook is self._excepthook:
            return
        self.uninstall_excepthook()
        previous = sys.excepthook

        def hook(kind, value, tb, _owner=self, _previous=previous):
            # Only the hook installed last writes: an earlier one left inside
            # another hook's chain would otherwise report the exception twice.
            if _owner._excepthook is hook:
                _owner._report_uncaught(kind, value, tb)
            if _previous is not None:
                _previous(kind, value, tb)

        self._previous_excepthook = previous
        self._excepthook = hook
        sys.excepthook = hook

    def uninstall_excepthook(self) -> None:
        """Put the previous hook back, if nobody has installed one on top since.

        A hook installed on top keeps calling ours, so ours stays in its chain
        but stops writing and only passes the exception on.
        """
        hook, previous = self._excepthook, self._previous_excepthook
        self._excepthook = None
        self._previous_excepthook = None
        if hook is not None and sys.excepthook is hook:
            sys.excepthook = previous

    def _report_uncaught(self, kind, value, tb) -> None:
        try:
            if self._closed:
                return
            if not any(self._source.covers(frame.f_code.co_filename) for frame, _ in traceback.walk_tb(tb)):
                return
            formatted = "".join(traceback.format_exception(kind, value, tb)).rstrip("\n")
            with self._lock:
                self._end_run()
                self._write_line("{} ERROR Uncaught exception\n{}".format(_asctime(time.time()), formatted))
        except Exception:  # noqa: BLE001 - the hook must never add an error of its own
            pass

    # ----------------------------------------------- crashes and hangs

    def _start_crash_reports(self, stream) -> bool:
        try:
            self._crash_reports_were_enabled = faulthandler.is_enabled()
            faulthandler.enable(file=stream, all_threads=True)
        except Exception:  # noqa: BLE001 - no crash reports is better than no log
            self._crash_reports = False
            return False
        self._crash_reports = True
        return True

    def _stop_crash_reports(self) -> None:
        if not self._crash_reports:
            return
        self._crash_reports = False
        try:
            faulthandler.disable()
        except Exception:  # noqa: BLE001
            pass
        if not self._crash_reports_were_enabled:
            return
        # faulthandler cannot say which file it wrote to before; it is
        # enabled on stderr in practice (python -X faulthandler or
        # PYTHONFAULTHANDLER).  Slicer replaces sys.stderr with a console
        # stream that has no descriptor, hence the fallbacks.
        for target in (sys.stderr, sys.__stderr__, 2):
            if target is None:
                continue
            try:
                faulthandler.enable(file=target, all_threads=True)
                return
            except Exception:  # noqa: BLE001
                continue

    def _stop_fault_reports(self) -> None:
        self.disarm_watchdog()
        self._stop_crash_reports()

    def arm_watchdog(self, timeout_s: float = WATCHDOG_TIMEOUT_S) -> bool:
        """Write every thread's stack into the file if not re-armed within *timeout_s*.

        The panel calls this from a Qt timer, so it is only late when the main
        thread is stuck; the stacks then show where.  It repeats every
        *timeout_s* while the thread stays stuck.  Does nothing (and returns
        False) when no file is attached.
        """
        handler = self._file_handler
        if handler is None or handler.stream is None:
            return False
        try:
            if self._watchdog_armed:
                faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(float(timeout_s), repeat=True, file=handler.stream)
        except Exception:  # noqa: BLE001 - no watchdog is better than an error per tick
            self._watchdog_armed = False
            return False
        self._watchdog_armed = True
        return True

    def disarm_watchdog(self) -> None:
        """Cancel the watchdog armed by :meth:`arm_watchdog`, if any."""
        if not self._watchdog_armed:
            return
        self._watchdog_armed = False
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:  # noqa: BLE001
            pass
