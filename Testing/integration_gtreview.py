"""Integration test: drive the real GTReview widget inside a running Slicer.

Run it by hand::

    Slicer --no-splash \
        --additional-module-path <repo>/GTReview \
        --python-script <repo>/Testing/integration_gtreview.py

``smoke_headless.py`` exercises :class:`GTReviewLogic` against the real data
tree with no GUI.  This file covers the half that only exists once Slicer has a
main window: the widget, the segment editor, the slice views and their mouse
interactors.  On a synthetic case (numpy + SimpleITK, never patient data) it

1. loads a batch directory through the Dataset section, and checks GTReview.log
   appears in it: a header naming the Slicer version and the GTReview build,
   a line for the case loaded, a Slicer VTK warning written once (and a burst
   of one warning collapsed into a "repeated" line), each warning written
   exactly once with the Error Log window's Warning filter toggled between
   them, different warnings Slicer grouped into one row each written, an
   exception escaping a Qt callback in GTReview code with its traceback, and
   the watchdog's stack while the main thread is blocked -- and that the log
   never passes for a case, an image or a mask,
2. selects a lesion in the table and checks the brush unlocks and the views
   jump onto it,
3. paints a stroke with the Paint effect by sending real mouse events to the
   Red slice view's interactor,
4. undoes the WHOLE stroke with a single Undo press and redoes it with a single
   Redo press (a stroke is one gesture, not one undo state per brush stamp),
   then squeezes the undo memory budget: the history keeps no more states than
   the budget allows, drops the oldest first, still redoes cleanly, never
   lowers the cap while redo states exist, and Undo presses run out cleanly;
   then queues a burst of Undo clicks at the end of the history with the brush
   locked and checks the panel stays responsive (one failed step per press, no
   nested presses, one lesion recount after the burst, no key autorepeat),
5. drags the Sphere threshold effect with "2D: this slice only" off and then
   on, and checks the 2D result never leaves the slice it was drawn on,
6. deletes a lesion with its row's trash button, confirmation stubbed,
7. saves, then removes the review with the Delete review button, confirmation
   stubbed, and checks the file is gone and the case reopened from its
   original mask; then makes the save offered as the scene closes fail and
   checks the reviewer is told which case lost its edits,
8. opens a second, multi-sequence case (t1, t1c, t2, flair) and checks the
   Sequences (axial) layout is chosen on its own, shows each sequence in its
   own axial view, links the views so a scroll in one moves the others, and
   gives way to a layout the reviewer picked; also that the lesion list bridges
   a one-voxel gap (dilation before connected components) and that label 3
   (Edema) is present, paintable and exported as 3,
9. checks the shortcut labels: those built at setup are Qt's native spelling
   of their keys, those built on demand carry a marker put in place of
   shortcutText (on Linux a native label and a hand-written one look alike):
   the footer, the Save & next case, Undo, Redo and Delete review tooltips,
   a row's delete button and the delete-lesion and delete-label prompts;
   and Esc stays the word Esc; also that the Mac delete key (Backspace) is
   bound only when the platform flag says macOS -- and then deletes the
   lesion but leaves a focused text box alone,
10. makes case discovery raise PermissionError and checks the panel says
    Slicer was denied access (with the macOS privacy hint only on a Mac)
    instead of reporting 0 cases, keeps the open case and starts no
    GTReview.log,
11. checks the batch-directory history lists one directory once however it
    was spelled, Windows spellings included,
12. flips the application palette between light and dark and checks the
    section tints and the drawn icons follow it,
13. starts a scene close with unsaved edits, the prompt stubbed, and checks
    Discard saves nothing while Save goes through _saveBeforeSceneClose
    before the case is torn down,
14. loads a batch, then a folder with no cases, and checks no GTReview.log is
    written there and the dropped batch's log is closed,
15. runs the panel's cleanup and checks it takes GTReview's handler off the
    root logger, puts sys.excepthook back and stops the watchdog timer.

The file is deliberately NOT named ``test_*``: the unit suite
(``PythonSlicer -m unittest discover -s Testing``) must not pick it up, because
everything here needs a running Slicer, and that suite is restricted to the
standard library plus numpy.

Prints one PASS/FAIL line per check plus a summary, and exits non-zero through
``slicer.util.exit`` when anything failed.
"""

import logging
import ntpath
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

import ctk
import qt
import slicer
import vtk

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTING_DIR)
_MODULE_DIR = os.path.join(_REPO_ROOT, "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import GTReview as gtreview  # noqa: E402  (path set up above)
from GTReviewLib import maskio  # noqa: E402


# --------------------------------------------------------------------------- #
# the synthetic case
# --------------------------------------------------------------------------- #
CASE_ID = "IT_001"

#: [i, j, k] voxel counts, and mm per voxel.  k is deliberately the coarse axis
#: (3 mm): a 4 mm sphere then reaches the neighbouring slices, which is what
#: makes the "2D: this slice only" check mean something.
SIZE = (32, 32, 12)
SPACING = (1.0, 1.0, 3.0)
ORIGIN = (-16.0, -16.0, -18.0)
DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

#: two blocks, one per review label, far enough apart to stay two components
LESION_A = (slice(4, 9), slice(4, 9), slice(3, 6))    # label 1, 75 voxels
LESION_B = (slice(22, 26), slice(22, 26), slice(6, 9))  # label 2, 48 voxels
LESION_A_VOXELS = 5 * 5 * 3
LESION_B_VOXELS = 4 * 4 * 3

BACKGROUND_INTENSITY = 20
LESION_INTENSITY = 200

#: a bright, UNMASKED cylinder for the Sphere threshold to grow into: bright in
#: every slice, so a ball spills onto the neighbours and a 2D disc cannot
BRIGHT_CENTRE_IJ = (16, 16)
BRIGHT_RADIUS_VOXELS = 4
BRIGHT_K = slice(2, 10)

#: an empty spot to paint on, and the far end of the stroke
PAINT_START_IJK = (16, 6, 4)
PAINT_END_IJK = (22, 6, 4)
#: the Sphere threshold seed (centre of the bright cylinder) and its drag end,
#: 4 mm away -- one voxel further than the 3 mm slice gap
SPHERE_SEED_IJK = (16, 16, 5)
SPHERE_EDGE_IJK = (20, 16, 5)


def buildSyntheticCase(root):
    """Write ``<root>/IT_001/`` with one image and one mask; return the dir.

    Written with SimpleITK directly rather than through ``maskio.write_mask``
    so the fixture does not depend on the code under test.  SimpleITK's array
    order is ``[k, j, i]``; everything else in this file is ``[i, j, k]``.
    """
    caseDir = os.path.join(root, CASE_ID)
    os.makedirs(caseDir)

    mask = np.zeros(SIZE, dtype=np.uint8)
    mask[LESION_A] = 1
    mask[LESION_B] = 2

    image = np.full(SIZE, BACKGROUND_INTENSITY, dtype=np.int16)
    image[mask > 0] = LESION_INTENSITY
    i = np.arange(SIZE[0])[:, None]
    j = np.arange(SIZE[1])[None, :]
    disc = ((i - BRIGHT_CENTRE_IJ[0]) ** 2 + (j - BRIGHT_CENTRE_IJ[1]) ** 2
            <= BRIGHT_RADIUS_VOXELS ** 2)
    bright = np.zeros(SIZE, dtype=bool)
    bright[:, :, BRIGHT_K] = disc[:, :, None]
    image[bright] = LESION_INTENSITY

    _writeVolume(os.path.join(caseDir, CASE_ID + "_seg.nii.gz"), mask)
    _writeVolume(os.path.join(caseDir, CASE_ID + "_t1c.nii.gz"), image)
    return caseDir, mask, image


MULTI_CASE_ID = "IT_002"
MULTI_SEQUENCES = ("flair", "t2", "t1c", "t1")  # deliberately out of display order
#: two label-1 blocks one voxel apart along i: one lesion after dilation
GAP_A = (slice(4, 8), slice(4, 8), slice(3, 6))     # 48 voxels
GAP_B = (slice(9, 12), slice(4, 8), slice(3, 6))    # 36 voxels, gap at i == 8
GAP_VOXELS = 4 * 4 * 3 + 3 * 4 * 3
#: and a far-away label-2 block that must stay its own lesion
FAR_C = (slice(22, 26), slice(22, 26), slice(6, 9))  # 48 voxels
FAR_C_VOXELS = 4 * 4 * 3


def buildMultiSequenceCase(root):
    """Write ``<root>/IT_002/`` with four sequences and one mask; return the dir."""
    caseDir = os.path.join(root, MULTI_CASE_ID)
    os.makedirs(caseDir)
    mask = np.zeros(SIZE, dtype=np.uint8)
    mask[GAP_A] = 1
    mask[GAP_B] = 1
    mask[FAR_C] = 2
    for n, key in enumerate(MULTI_SEQUENCES, start=1):
        image = np.full(SIZE, BACKGROUND_INTENSITY * n, dtype=np.int16)
        image[mask > 0] = LESION_INTENSITY
        _writeVolume(os.path.join(caseDir, "{}_{}.nii.gz".format(MULTI_CASE_ID, key)), image)
    _writeVolume(os.path.join(caseDir, MULTI_CASE_ID + "_seg.nii.gz"), mask)
    return caseDir, mask


def _writeVolume(path, array_ijk):
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_ijk.transpose(2, 1, 0)))
    image.SetOrigin(ORIGIN)
    image.SetSpacing(SPACING)
    image.SetDirection(DIRECTION)
    sitk.WriteImage(image, path, True)


# --------------------------------------------------------------------------- #
# check harness
# --------------------------------------------------------------------------- #
class Checks(object):
    """One printed line per check; a failure also fails the unittest step."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def check(self, ok, description, detail=""):
        if ok:
            self.passed.append(description)
            print("  [ OK ] {}".format(description))
            sys.stdout.flush()
            return True
        self.failed.append((description, detail))
        print("  [FAIL] {}{}".format(description, ("  --  " + detail) if detail else ""))
        sys.stdout.flush()
        raise AssertionError(description + (("  --  " + detail) if detail else ""))

    def step(self, message):
        print("\n== {}".format(message))
        sys.stdout.flush()


CHECKS = Checks()


class ConfirmStub(object):
    """Answer every confirmation dialog without showing one, and record it."""

    def __init__(self, answer=True):
        self.answer = answer
        self.prompts = []
        self._original = None

    def __enter__(self):
        self._original = slicer.util.confirmYesNoDisplay

        def stub(text, *args, **kwargs):
            del args, kwargs
            self.prompts.append(str(text))
            return self.answer

        slicer.util.confirmYesNoDisplay = stub
        return self

    def __exit__(self, excType, excValue, excTraceback):
        slicer.util.confirmYesNoDisplay = self._original
        return False


# --------------------------------------------------------------------------- #
# driving the views
# --------------------------------------------------------------------------- #
def pump(seconds=0.3):
    """Let Qt and VTK catch up: deliver queued events, render, run timers."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        slicer.app.processEvents()


def sliceWidgetNamed(name="Red"):
    layoutManager = slicer.app.layoutManager()
    widget = layoutManager.sliceWidget(name) if layoutManager else None
    if widget is None:
        raise RuntimeError("the {} slice view is not in the current layout".format(name))
    return widget


def eventPositionFor(sliceWidget, ras):
    """Interactor event position of a RAS point (view pixels, origin bottom left).

    This is the inverse of the transform the segment editor effects use to turn
    a click into a voxel, so a position computed here lands on the voxel it was
    computed from -- which the Sphere threshold step asserts explicitly.
    """
    rasToXy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(sliceWidget.mrmlSliceNode().GetXYToRAS(), rasToXy)
    xy = rasToXy.MultiplyPoint([float(ras[0]), float(ras[1]), float(ras[2]), 1.0])
    return int(round(xy[0])), int(round(xy[1]))


def centreOn(sliceWidget, ras):
    """Put *ras* on the visible slice and at the centre of the view."""
    slicer.modules.markups.logic().JumpSlicesToLocation(ras[0], ras[1], ras[2], True)
    pump(0.1)
    return eventPositionFor(sliceWidget, ras)


def dragBetween(sliceWidget, startRas, endRas, steps=12):
    """Press at *startRas*, drag to *endRas*, release.  Returns both positions."""
    start = centreOn(sliceWidget, startRas)
    end = eventPositionFor(sliceWidget, endRas)
    slicer.util.clickAndDrag(sliceWidget, start=start, end=end, steps=steps)
    pump()
    return start, end


def distanceToSlicePlane(sliceWidget, ras):
    """How far *ras* is from the plane the slice view currently shows, in mm."""
    sliceToRas = sliceWidget.mrmlSliceNode().GetSliceToRAS()
    normal = [sliceToRas.GetElement(row, 2) for row in range(3)]
    origin = [sliceToRas.GetElement(row, 3) for row in range(3)]
    return abs(sum(n * (p - o) for n, p, o in zip(normal, ras, origin)))


def rasToIjk(volumeNode, ras):
    """RAS -> that volume node's own IJK index (which is not the mask's)."""
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(matrix)
    ijk = matrix.MultiplyPoint([float(ras[0]), float(ras[1]), float(ras[2]), 1.0])
    return tuple(int(round(v)) for v in ijk[:3])


def planesOf(mask_ijk):
    """The k planes a boolean ``[i, j, k]`` mask touches."""
    return sorted(set(int(k) for k in np.argwhere(mask_ijk)[:, 2]))


def backgroundNameOf(sliceWidget):
    node = slicer.mrmlScene.GetNodeByID(
        sliceWidget.mrmlSliceCompositeNode().GetBackgroundVolumeID() or ""
    )
    return node.GetName() if node is not None else None


def scrollLikeTheMouse(sliceWidget, deltaMm):
    """Move the slice offset the way the interactor does, so linking broadcasts."""
    logic = sliceWidget.sliceLogic()
    logic.StartSliceOffsetInteraction()
    logic.SetSliceOffset(logic.GetSliceOffset() + deltaMm)
    logic.EndSliceOffsetInteraction()
    pump(0.1)


def moduleGlobals(widget):
    """The globals the widget's own methods look names up in.

    ``gtreview`` above is whatever ``import GTReview`` returned here, which is
    only the module Slicer loaded the widget from if both ended up under the
    same sys.modules entry.  A flag patched through the methods' own globals
    reaches the widget either way.
    """
    return type(widget).installShortcuts.__globals__


def nativeText(keys):
    return qt.QKeySequence(keys).toString(qt.QKeySequence.NativeText)


def plainToolTip(widget):
    """A widget's tooltip as the panel wrote it.

    Slicer's tooltip trapper (ctkToolTipTrapper, with word wrap on) turns a
    plain tooltip into rich text the moment it is set: "<p>...</p>", a newline
    as <br>, and "<", ">" and "&" escaped.  Read back through a text document,
    the tooltip is again exactly the text the panel passed.
    """
    document = qt.QTextDocument()
    document.setHtml(str(widget.toolTip))
    return str(document.toPlainText())


def readText(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


#: every entry of GTReview.log starts with a logging time stamp
LOG_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} ")


def logEntries(text):
    """GTReview.log as entries: a time-stamped line with the lines under it."""
    entries = []
    for line in text.splitlines():
        if LOG_STAMP.match(line) or not entries:
            entries.append(line)
        else:
            entries[-1] += "\n" + line
    return entries


def lastSessionHeader(text):
    """The header of the newest session in GTReview.log, up to its closing rule."""
    start = text.rfind("GTReview session started")
    if start < 0:
        return ""
    end = text.find("=" * 72, start)
    return text[start:] if end < 0 else text[start:end]


def rowOfLesion(widget, lesionIndex):
    for row in range(widget.lesionTable.rowCount):
        item = widget.lesionTable.item(row, widget.LESION_COLUMN_NUMBER)
        if item is not None and item.data(qt.Qt.UserRole) == lesionIndex:
            return row
    return -1


# --------------------------------------------------------------------------- #
# the test
# --------------------------------------------------------------------------- #
class GTReviewIntegrationTest(unittest.TestCase):
    """One review session, step by step.

    The steps share the loaded case, so they run in name order (``test_01`` ...
    ``test_08``) and each one starts where the previous one left off.
    """

    tempRoot = None
    widget = None
    sourceMask = None
    beforeStroke = None
    afterStroke = None
    savedHistory = None
    multiRoot = None

    @classmethod
    def setUpClass(cls):
        CHECKS.step("Building a synthetic case and opening the module")
        cls.tempRoot = tempfile.mkdtemp(prefix="gtreview_integration_")
        caseDir, mask, _image = buildSyntheticCase(cls.tempRoot)
        cls.sourceMask = mask
        print("  case dir: {}".format(caseDir))

        # the batch directory is remembered in the user's settings; this one is
        # a temp path that will not exist after the run, so put the real list
        # back when the test is done
        cls.savedHistory = slicer.app.userSettings().value(gtreview.DATASET_HISTORY_KEY)

        mainWindow = slicer.util.mainWindow()
        if mainWindow is not None:
            mainWindow.resize(1400, 1000)
        pump()
        slicer.util.selectModule("GTReview")
        pump()
        cls.widget = slicer.modules.gtreview.widgetRepresentation().self()

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.widget is not None and cls.widget.logic is not None:
                cls.widget.logic.unloadCase()
        except Exception:  # noqa: BLE001 - teardown must not hide a real failure
            traceback.print_exc()
        settings = slicer.app.userSettings()
        if cls.savedHistory is None:
            settings.remove(gtreview.DATASET_HISTORY_KEY)
        else:
            settings.setValue(gtreview.DATASET_HISTORY_KEY, cls.savedHistory)
        settings.sync()
        if cls.tempRoot and os.path.isdir(cls.tempRoot):
            shutil.rmtree(cls.tempRoot, ignore_errors=True)

    # ------------------------------------------------------------------ steps
    def test_01_load_the_dataset(self):
        CHECKS.step("Loading the batch directory through the Dataset section")
        widget = self.widget
        widget.datasetPathEdit.currentPath = self.tempRoot
        widget.onLoadDataset()
        pump()

        CHECKS.check(len(widget.cases) == 1, "one case discovered",
                     "{} cases".format(len(widget.cases)))
        case = widget.currentCase()
        CHECKS.check(case is not None and case.case_id == CASE_ID,
                     "the case was selected and loaded")
        CHECKS.check(widget.logic.segmentationNode is not None,
                     "a segmentation node was built for it")
        exported = widget.logic.exportLabelmapArrayIJK()
        CHECKS.check(np.array_equal(exported.astype(np.uint8), self.sourceMask),
                     "the segmentation round-trips the mask on disk voxel-exactly")
        CHECKS.check(len(widget.lesionList) == 2,
                     "both lesions are in the lesion list",
                     "{} lesions".format(len(widget.lesionList)))
        CHECKS.check(widget.lesionTable.rowCount == 2,
                     "the lesion table shows one row per lesion")

        # a single big slice view: the drags below aim at individual voxels, so
        # the view wants as many pixels per millimetre as it can get
        index = widget.layoutComboBox.findText("1x1 Red (axial)")
        widget.layoutComboBox.currentIndex = index
        widget.onLayoutChanged()
        pump()
        red = sliceWidgetNamed("Red")
        CHECKS.check(red.mrmlSliceNode() is not None, "the Red slice view is up")

    def test_01b_session_log_in_the_batch_folder(self):
        CHECKS.step("GTReview.log: written into the batch folder that was loaded")
        widget = self.widget
        namespace = moduleGlobals(widget)
        logModule = namespace["sessionlog"]
        logPath = os.path.join(self.tempRoot, logModule.LOG_FILE_NAME)
        sessionLog = widget.sessionLog
        CHECKS.check(os.path.isfile(logPath), "loading the batch created GTReview.log in it",
                     logPath)
        CHECKS.check(sessionLog is not None and sessionLog.path is not None
                     and os.path.samefile(sessionLog.path, logPath),
                     "and the session log is writing there",
                     str(sessionLog.path if sessionLog is not None else None))

        # ---- the header and the event lines ---------------------------------
        text = readText(logPath)
        header = lastSessionHeader(text)
        CHECKS.check(slicer.app.applicationVersion in header,
                     "the header names the Slicer version", header.replace("\n", " | "))
        build = logModule.installed_build(namespace["__file__"]) or "source tree {}".format(
            os.path.dirname(os.path.abspath(namespace["__file__"])))
        CHECKS.check("GTReview build" in header and build in header,
                     "the header names the GTReview build", "expected {}".format(build))
        crashReports = ("Crash reports: on" if getattr(logModule, "CRASH_REPORTS_ENABLED", True)
                        else "Crash reports: off (Windows)")
        CHECKS.check(crashReports in header, "and says whether crash stacks go into the file",
                     "expected {}".format(crashReports))
        session = text[text.rfind("GTReview session started"):]
        CHECKS.check("GTReview: dataset loaded from" in session,
                     "a line records the dataset that was loaded")
        caseLines = [entry for entry in logEntries(session)
                     if "GTReview: case {} loaded".format(CASE_ID) in entry]
        CHECKS.check(bool(caseLines) and CASE_ID + "_seg.nii.gz" in caseLines[-1]
                     and "2 lesions" in caseLines[-1],
                     "a line records the case loaded, its mask file and its lesion count",
                     " | ".join(caseLines))

        # ---- the log is never data ------------------------------------------
        datasetModule = namespace["dataset"]

        def filesOf(cases):
            return [path for case in cases
                    for path in [case.directory] + list(case.images.values())
                    + list(case.masks.values())]

        found = datasetModule.discover_cases(self.tempRoot)
        CHECKS.check(len(found) == 1 and found[0].case_id == CASE_ID,
                     "with GTReview.log beside the case folder, discovery finds the one case",
                     str([case.case_id for case in found]))
        CHECKS.check(not any(os.path.basename(path).startswith(logModule.LOG_FILE_NAME)
                             for path in filesOf(found)),
                     "and the log is not a case, an image or a mask", str(filesOf(found)))
        # a batch that is a single case folder has the log among the case's files
        caseDir = os.path.join(self.tempRoot, CASE_ID)
        stray = os.path.join(caseDir, logModule.LOG_FILE_NAME)
        with open(stray, "w", encoding="utf-8") as handle:
            handle.write("a log among the case's files\n")
        try:
            single = datasetModule.discover_cases(caseDir)
        finally:
            os.remove(stray)
        CHECKS.check(len(single) == 1 and not any(
            os.path.basename(path).startswith(logModule.LOG_FILE_NAME)
            for path in filesOf(single)[1:]),
            "in a case folder loaded as the batch, the log is not an image or a mask",
            str(filesOf(single)))
        CHECKS.check(not datasetModule.is_nifti(logModule.LOG_FILE_NAME)
                     and not datasetModule.is_nifti(logModule.LOG_FILE_NAME + ".1"),
                     "neither the log nor its rotated copy passes for a NIfTI file")

        # ---- Slicer's own warnings ------------------------------------------
        CHECKS.check(widget._errorLogModel is not None,
                     "precondition: the panel follows Slicer's error log")

        def historyWarnings():
            return [entry for entry in logEntries(readText(logPath))
                    if "[VTK]" in entry.split("\n", 1)[0] and "vtkSegmentationHistory" in entry]

        # Two history objects alive at once: VTK prints the object's address,
        # so their warnings differ, while one object repeats itself exactly.
        first = slicer.vtkSegmentationHistory()
        burst = slicer.vtkSegmentationHistory()
        before = len(historyWarnings())
        first.RestorePreviousState()  # no segmentation, nothing to restore: VTK warns
        pump(0.3)
        warnings = historyWarnings()
        CHECKS.check(len(warnings) == before + 1,
                     "a VTK warning raised in Slicer is written once, as a [VTK] entry",
                     "{} new entries".format(len(warnings) - before))
        for _ in range(5):
            burst.RestorePreviousState()
        pump(0.3)
        afterBurst = historyWarnings()
        CHECKS.check(len(afterBurst) == len(warnings) + 1,
                     "five identical warnings in a row are written once",
                     "{} new entries".format(len(afterBurst) - len(warnings)))
        first.RestorePreviousState()  # a different warning ends the run
        pump(0.3)
        entries = logEntries(readText(logPath))
        burstAt = entries.index(afterBurst[-1]) if afterBurst and afterBurst[-1] in entries else -1
        following = entries[burstAt + 1].split("\n", 1)[0] if 0 <= burstAt < len(entries) - 1 else ""
        CHECKS.check(following.endswith("[VTK] ... repeated 4 more times"),
                     "and the next, different, one writes how many more times it came, "
                     "right under it", following)

        # ---- the Error Log window's level filter ----------------------------
        # errorLogModel() is a filter proxy following the Error Log window's
        # level checkboxes: unticking Warning takes every warning row out of it,
        # ticking it again puts them back.  Neither may cost a warning or write
        # one twice.  vtkOutputWindow raises a VTK warning with a text of ours.
        outputWindow = vtk.vtkOutputWindow.GetInstance()

        def vtkWarnings(marker):
            return [entry for entry in logEntries(readText(logPath))
                    if entry.split("\n", 1)[0].endswith("[VTK] " + marker)]

        errorLog = slicer.app.errorLogModel()
        levelFilter = ctk.ctkErrorLogWidget()  # the Error Log window's checkboxes
        levelFilter.setErrorLogModel(errorLog)
        filtered = ["GTReview integration: a warning {}".format(when) for when in (
            "before Warning is unticked", "while Warning is unticked",
            "after Warning is ticked again")]
        try:
            outputWindow.DisplayWarningText(filtered[0])
            pump(0.3)
            levelFilter.setWarningEntriesVisible(False)
            pump(0.1)
            shownRows, allRows = int(errorLog.rowCount()), int(errorLog.sourceModel.rowCount())
            outputWindow.DisplayWarningText(filtered[1])
            pump(0.3)
            levelFilter.setWarningEntriesVisible(True)
            pump(0.1)
            outputWindow.DisplayWarningText(filtered[2])
            pump(0.3)
        finally:
            levelFilter.setWarningEntriesVisible(True)
            levelFilter.deleteLater()
        CHECKS.check(shownRows < allRows,
                     "precondition: unticking Warning in the Error Log window hides rows of "
                     "errorLogModel()", "{} of {} rows shown".format(shownRows, allRows))
        for marker in filtered:
            found = vtkWarnings(marker)
            CHECKS.check(len(found) == 1, "written exactly once: " + marker.split(": ", 1)[1],
                         "{} entries".format(len(found)))

        # ---- entries Slicer groups into one row -----------------------------
        # An entry from the same thread, level and origin as the row before it,
        # within about a second, adds no row -- whatever its text -- and is
        # still signalled.  Sent back to back, with no check printing between
        # them, these land in one row or two.
        grouped = ["GTReview integration: grouped warning {}".format(name) for name in "ABC"]
        same = "GTReview integration: one warning again and again"
        closing = "GTReview integration: a different warning ends the run"
        sameTimes = 8
        for marker in grouped:
            outputWindow.DisplayWarningText(marker)
        for _ in range(sameTimes):
            outputWindow.DisplayWarningText(same)
        outputWindow.DisplayWarningText(closing)
        pump(0.3)
        firstLines = [entry.split("\n", 1)[0] for entry in logEntries(readText(logPath))]

        def positions(marker):
            return [i for i, line in enumerate(firstLines) if line.endswith("[VTK] " + marker)]

        found = [positions(marker) for marker in grouped]
        CHECKS.check(all(len(where) == 1 for where in found) and found == sorted(found),
                     "different warnings Slicer grouped into one row are each written once, "
                     "in order", str(found))
        sameAt, closingAt = positions(same), positions(closing)
        CHECKS.check(len(sameAt) == 1,
                     "{} identical warnings in a row are written once".format(sameTimes),
                     "{} entries".format(len(sameAt)))
        between = (firstLines[sameAt[0] + 1:closingAt[0]]
                   if len(sameAt) == 1 and len(closingAt) == 1 else [])
        CHECKS.check(bool(between)
                     and between[0].endswith("[VTK] ... repeated {} more times".format(sameTimes - 1))
                     and sum(1 for line in between if "... repeated" in line) == 1,
                     "followed by exactly one line saying how many more times it came",
                     " | ".join(between))

        # ---- an exception escaping a Qt callback in GTReview's code ---------
        def uncaughtReports():
            return [entry for entry in logEntries(readText(logPath))
                    if "Uncaught exception" in entry.split("\n", 1)[0]]

        before = len(uncaughtReports())
        # _elide compares the length of its text with the limit: a limit that is
        # not a number raises inside GTReview.py, called from a Qt timer.
        qt.QTimer.singleShot(0, lambda: widget._elide("a path to elide", limit="not a number"))
        pump(0.3)
        uncaught = uncaughtReports()
        CHECKS.check(len(uncaught) == before + 1,
                     "an uncaught exception in a Qt callback is written to GTReview.log",
                     "{} new entries".format(len(uncaught) - before))
        report = uncaught[-1] if uncaught else ""
        CHECKS.check("Traceback (most recent call last)" in report and "in _elide" in report
                     and "TypeError" in report,
                     "with its traceback through GTReview's code",
                     report.replace("\n", " | ")[-400:])

        # ---- a blocked main thread ------------------------------------------
        timer = widget._watchdogTimer
        CHECKS.check(timer is not None and timer.isActive()
                     and int(timer.interval) == logModule.WATCHDOG_REARM_MS,
                     "the panel keeps re-arming the hang watchdog from a timer")
        reportsBefore = readText(logPath).count("Timeout (")
        try:
            sessionLog.arm_watchdog(0.5)
            time.sleep(1.2)  # no events run, so nothing re-arms it
        finally:
            sessionLog.arm_watchdog()  # back to the panel's own timeout
        text = readText(logPath)
        CHECKS.check(text.count("Timeout (") > reportsBefore,
                     "blocking the main thread past the timeout writes a Timeout report")
        stack = text[text.rfind("Timeout ("):]
        CHECKS.check("most recent call first" in stack
                     and "test_01b_session_log_in_the_batch_folder" in stack,
                     "with the stack of the thread that was stuck",
                     stack[:400].replace("\n", " | "))

    def test_02_select_a_lesion(self):
        CHECKS.step("Selecting the largest lesion in the table")
        widget = self.widget
        widget.lesionTable.selectRow(0)
        pump()

        lesion = widget.selectedLesion()
        CHECKS.check(lesion is not None, "clicking a row selects a lesion")
        CHECKS.check(lesion.voxel_count == LESION_A_VOXELS,
                     "the table is sorted largest first, so row 0 is the big lesion",
                     "{} voxels".format(lesion.voxel_count))
        CHECKS.check(int(lesion.label) == 1, "it carries its original label value")
        CHECKS.check(widget._editingAllowed(), "a selected lesion unlocks the brush")
        CHECKS.check(
            widget.logic.labelValueForSegmentId(widget.editor.currentSegmentID()) == 1,
            "the editor's current segment followed the lesion's label",
        )

        ras = widget.logic.centroidToRAS(lesion.centroid_ijk)
        offset = distanceToSlicePlane(sliceWidgetNamed("Red"), ras)
        CHECKS.check(offset <= SPACING[2] / 2.0 + 1e-3,
                     "the slice views jumped onto the lesion",
                     "{:.3f} mm off the visible plane".format(offset))

    def test_03_paint_a_stroke(self):
        CHECKS.step("Painting a stroke with the Paint effect (real mouse events)")
        widget = self.widget
        type(self).beforeStroke = widget.logic.exportLabelmapArrayIJK()

        widget.onActivateEffect("Paint")
        pump()
        effect = widget.editor.activeEffect()
        CHECKS.check(effect is not None and effect.name == "Paint",
                     "the Paint effect is active")
        CHECKS.check(abs(effect.doubleParameter("BrushAbsoluteDiameter") - widget.BRUSH_MM) < 1e-6,
                     "the brush starts at {:g} mm".format(widget.BRUSH_MM),
                     "{} mm".format(effect.doubleParameter("BrushAbsoluteDiameter")))

        red = sliceWidgetNamed("Red")
        startRas = widget.logic.centroidToRAS(PAINT_START_IJK)
        endRas = widget.logic.centroidToRAS(PAINT_END_IJK)
        start, end = dragBetween(red, startRas, endRas)
        print("  dragged {} -> {} in the Red view".format(start, end))

        after = widget.logic.exportLabelmapArrayIJK()
        type(self).afterStroke = after
        added = (after != 0) & (self.beforeStroke == 0)
        CHECKS.check(bool(added.any()), "the stroke painted voxels",
                     "{} voxels".format(int(added.sum())))
        painted = set(int(v) for v in np.unique(after[added]))
        CHECKS.check(painted == {1}, "it painted the selected lesion's label",
                     "values {}".format(sorted(painted)))
        untouched = self.beforeStroke != 0
        CHECKS.check(np.array_equal(after[untouched], self.beforeStroke[untouched]),
                     "nothing that was already labelled changed")
        CHECKS.check(planesOf(added) == [PAINT_START_IJK[2]],
                     "the stroke stayed on the slice it was drawn on",
                     "planes {}".format(planesOf(added)))

        # everything painted is within a brush radius of the line that was
        # dragged; a stroke landing anywhere else means the click mapped to the
        # wrong voxel, which is exactly what this step is here to catch
        indices = np.argwhere(added)
        margin = widget.BRUSH_MM  # mm == voxels along i and j here
        CHECKS.check(
            indices[:, 0].min() >= PAINT_START_IJK[0] - margin
            and indices[:, 0].max() <= PAINT_END_IJK[0] + margin
            and abs(indices[:, 1] - PAINT_START_IJK[1]).max() <= margin,
            "the painted voxels sit where the mouse went",
            "i {}..{}, j {}..{}".format(
                indices[:, 0].min(), indices[:, 0].max(),
                indices[:, 1].min(), indices[:, 1].max()),
        )
        CHECKS.check(len(widget._strokeStarts) == 1,
                     "the mask at mouse-down was fingerprinted, once for the stroke",
                     "{} marks".format(len(widget._strokeStarts)))
        # The effect aborts the release event it handles, so this only holds
        # while GTReview's release observer runs ahead of the segment editor.
        CHECKS.check(not widget._strokeInProgress,
                     "the mouse release ended the stroke")
        CHECKS.check(not widget._undoPeaks.editInProgress,
                     "and closed its edit for the undo memory budget")

    def test_03b_live_fill_toggle(self):
        CHECKS.step("Live fill: immediate brush by default, delayed when unticked")
        widget = self.widget
        effect = widget.editor.activeEffect()
        CHECKS.check(effect is not None and effect.name == "Paint", "Paint is still active")
        CHECKS.check(widget.liveFillCheckBox.checked, "Live fill is ticked by default")
        CHECKS.check(not effect.delayedPaint, "so the brush commits stamp by stamp")
        widget.liveFillCheckBox.checked = False
        pump(0.1)
        CHECKS.check(effect.delayedPaint, "unticking switches the effect to delayed paint")
        widget.editor.setActiveEffectByName("Erase")
        pump(0.1)
        CHECKS.check(widget.editor.activeEffect().delayedPaint,
                     "and a newly activated brush follows the box")
        eraseMm = widget.editor.activeEffect().doubleParameter("BrushAbsoluteDiameter")
        CHECKS.check(abs(eraseMm - widget.BRUSH_MM) < 1e-6,
                     "Erase shares the {:g} mm brush".format(widget.BRUSH_MM),
                     "{} mm".format(eraseMm))
        widget.liveFillCheckBox.checked = True
        pump(0.1)
        CHECKS.check(not widget.editor.activeEffect().delayedPaint,
                     "ticking it again restores immediate paint")
        widget.editor.setActiveEffectByName("Paint")
        pump(0.1)

    def test_04_undo_the_whole_stroke(self):
        CHECKS.step("Undoing the stroke with one Undo press")
        widget = self.widget

        # A stroke is one state per brush stamp in the editor's own history, so
        # a raw undo() steps back a fraction of it.  Stepping over the whole
        # stroke is what the Undo button adds; check the raw behaviour first,
        # otherwise the test below could pass on a stroke that happened to be a
        # single state.
        widget.editor.undo()
        pump(0.1)
        partial = widget.logic.exportLabelmapArrayIJK()
        print("  one raw undo(): {} voxels still painted, {} of them gone".format(
            int(((partial != 0) & (self.beforeStroke == 0)).sum()),
            int(((self.afterStroke != 0) & (partial == 0)).sum())))
        CHECKS.check(not np.array_equal(partial, self.beforeStroke),
                     "one raw editor undo does NOT undo the stroke")
        widget.editor.redo()
        pump(0.1)
        CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), self.afterStroke),
                     "a raw redo puts that state back, so the press below starts "
                     "from the finished stroke")

        widget.onUndo()
        pump()
        exported = widget.logic.exportLabelmapArrayIJK()
        CHECKS.check(np.array_equal(exported, self.beforeStroke),
                     "one Undo press removed the whole stroke, not one brush stamp",
                     "{} voxels still differ".format(
                         int((exported != self.beforeStroke).sum())))
        CHECKS.check(not widget._strokeStarts, "the stroke mark was consumed")

    def test_05_redo_the_whole_stroke(self):
        CHECKS.step("Redoing the stroke with one Redo press")
        widget = self.widget
        widget.onRedo()
        pump()
        exported = widget.logic.exportLabelmapArrayIJK()
        CHECKS.check(np.array_equal(exported, self.afterStroke),
                     "one Redo press put the whole stroke back",
                     "{} voxels differ".format(
                         int((exported != self.afterStroke).sum())))

        # leave the mask as it was loaded, so the steps below start from a
        # known case
        widget.onUndo()
        pump()
        CHECKS.check(
            np.array_equal(widget.logic.exportLabelmapArrayIJK(), self.beforeStroke),
            "and Undo takes it away again",
        )

    def test_05b_undo_memory_budget(self):
        CHECKS.step("Undo memory budget: the oldest states go first, never mid-undo")
        widget = self.widget
        editor = widget.editor
        budget = gtreview.undobudget
        loaded = widget.logic.exportLabelmapArrayIJK()
        undoButton = slicer.util.findChild(editor, "UndoButton")
        redoButton = slicer.util.findChild(editor, "RedoButton")
        red = sliceWidgetNamed("Red")

        def stroke(rowJ):
            """A Live fill stroke along row j of the paint slice: a dozen states."""
            startRas = widget.logic.centroidToRAS((PAINT_START_IJK[0], rowJ, PAINT_START_IJK[2]))
            endRas = widget.logic.centroidToRAS((PAINT_END_IJK[0], rowJ, PAINT_END_IJK[2]))
            dragBetween(red, startRas, endRas)
            return widget.logic.exportLabelmapArrayIJK()

        def expectedCap():
            return budget.states_for_budget(
                widget._undoPeaks.estimate(widget._historyStateBytes()),
                int(float(widget.UNDO_MEMORY_BUDGET_MB) * budget.MIB),
                widget.MIN_UNDO_STATES,
                widget.MAX_UNDO_STATES,
            )

        try:
            CHECKS.check(not editor.readOnly,
                         "the brush is unlocked, so the editor's history buttons are exact")
            for rowJ in (PAINT_START_IJK[1], PAINT_START_IJK[1] + 3, PAINT_START_IJK[1] + 6):
                stroke(rowJ)
            stateBytes = widget._historyStateBytes()
            CHECKS.check(stateBytes is not None and stateBytes > 0,
                         "one undo state is priced from the mask", "{} bytes".format(stateBytes))
            CHECKS.check(editor.maximumNumberOfUndoStates == widget.MAX_UNDO_STATES,
                         "a small mask keeps the full undo depth under the default budget",
                         "cap {}".format(editor.maximumNumberOfUndoStates))

            # a budget worth a dozen states of this mask
            widget.UNDO_MEMORY_BUDGET_MB = (
                (widget.MIN_UNDO_STATES + 8) * widget._undoPeaks.estimate(stateBytes)
                / float(budget.MIB)
            )
            afterD = stroke(PAINT_START_IJK[1] + 9)
            cap = editor.maximumNumberOfUndoStates
            CHECKS.check(cap == expectedCap(),
                         "the stroke priced the history and set the cap from the budget",
                         "cap {}, expected {}".format(cap, expectedCap()))
            CHECKS.check(widget.MIN_UNDO_STATES < cap < widget.MAX_UNDO_STATES,
                         "a tight budget lowers the depth", "cap {}".format(cap))

            steps = 0
            while undoButton.enabled and steps < widget.MAX_UNDO_STATES + 5:
                editor.undo()
                steps += 1
            pump(0.1)
            CHECKS.check(1 <= steps <= cap,
                         "the history holds no more states than the cap",
                         "{} undo steps, cap {}".format(steps, cap))
            CHECKS.check(not np.array_equal(widget.logic.exportLabelmapArrayIJK(), loaded),
                         "the oldest states were the ones dropped: the loaded mask is out of reach")
            redone = 0
            while redoButton.enabled and redone < widget.MAX_UNDO_STATES + 5:
                editor.redo()
                redone += 1
            pump(0.1)
            CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), afterD),
                         "every kept state redoes back to the last stroke: the trim left "
                         "the history consistent",
                         "{} redo steps".format(redone))

            widget.onUndo()
            pump()
            CHECKS.check(redoButton.enabled, "one Undo press leaves states to redo")
            capBefore = editor.maximumNumberOfUndoStates
            CHECKS.check(capBefore > widget.MIN_UNDO_STATES,
                         "precondition: the cap has room to drop", "cap {}".format(capBefore))
            widget.UNDO_MEMORY_BUDGET_MB = 1e-6  # far below a single state
            widget._enforceUndoBudget()
            CHECKS.check(editor.maximumNumberOfUndoStates == capBefore,
                         "a lower cap waits while redo states exist (trimming then would "
                         "underflow Slicer's history position)",
                         "cap {}".format(editor.maximumNumberOfUndoStates))
            widget.onRedo()
            pump()
            CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), afterD),
                         "and Redo still restores the stroke afterwards")

            stroke(PAINT_START_IJK[1] + 12)
            CHECKS.check(editor.maximumNumberOfUndoStates == widget.MIN_UNDO_STATES,
                         "the next edit clears redo, so the deferred cap lands on the floor",
                         "cap {}".format(editor.maximumNumberOfUndoStates))

            started = time.time()
            presses = 0
            while undoButton.enabled and presses < widget.MIN_UNDO_STATES + 5:
                widget.onUndo()
                pump(0.05)
                presses += 1
            elapsed = time.time() - started
            CHECKS.check(not undoButton.enabled,
                         "Undo presses run out at the oldest kept state",
                         "{} presses".format(presses))
            CHECKS.check(elapsed < 15.0,
                         "without walking past the start of the history on every press",
                         "{:.1f} s".format(elapsed))
        finally:
            if "UNDO_MEMORY_BUDGET_MB" in vars(widget):
                del widget.UNDO_MEMORY_BUDGET_MB
            widget.loadCurrentCase()
            pump()
            widget.lesionTable.selectRow(0)
            pump()
        CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), loaded),
                     "reloading the case puts the loaded mask back for the steps below")
        CHECKS.check(editor.maximumNumberOfUndoStates == widget.MAX_UNDO_STATES,
                     "and a fresh case starts from the full depth again",
                     "cap {}".format(editor.maximumNumberOfUndoStates))

    def test_05c_rapid_undo_presses(self):
        CHECKS.step("Clicking Undo fast at the end of the history does not freeze the panel")
        widget = self.widget
        editor = widget.editor
        loaded = widget.logic.exportLabelmapArrayIJK()
        red = sliceWidgetNamed("Red")
        clicks = 25
        stats = {"presses": 0, "depth": 0, "maxDepth": 0, "tries": 0, "failed": 0,
                 "refreshes": 0, "refreshing": 0, "pressesInsideRecount": 0}
        gap = {"last": time.time(), "max": 0.0}
        timer = qt.QTimer()
        originalStepHistory = widget._stepHistory
        originalRefresh = widget.refreshLesions
        # the test's own count of segmentation events: a raw undo that raises
        # none found nothing to undo (and logged a warning for it)
        events = {"n": 0}
        observed = []

        def countEvent(caller, event):
            events["n"] += 1

        def stepHistory(step, *args):
            def countedStep():
                stats["tries"] += 1
                before = events["n"]
                step()
                if events["n"] == before:
                    stats["failed"] += 1

            stats["presses"] += 1
            if stats["refreshing"]:
                # the recount's busy cursor pumps events: a queued click ran here
                stats["pressesInsideRecount"] += 1
            stats["depth"] += 1
            stats["maxDepth"] = max(stats["maxDepth"], stats["depth"])
            try:
                return originalStepHistory(countedStep, *args)
            finally:
                stats["depth"] -= 1

        def refreshLesions():
            stats["refreshes"] += 1
            stats["refreshing"] += 1
            try:
                return originalRefresh()
            finally:
                stats["refreshing"] -= 1

        def tick():
            now = time.time()
            gap["max"] = max(gap["max"], now - gap["last"])
            gap["last"] = now

        def postClick():
            button = widget.undoButton
            centre = qt.QPointF(button.width / 2.0, button.height / 2.0)
            qt.QApplication.postEvent(button, qt.QMouseEvent(
                qt.QEvent.MouseButtonPress, centre, qt.Qt.LeftButton, qt.Qt.LeftButton,
                qt.Qt.NoModifier))
            qt.QApplication.postEvent(button, qt.QMouseEvent(
                qt.QEvent.MouseButtonRelease, centre, qt.Qt.LeftButton, qt.Qt.NoButton,
                qt.Qt.NoModifier))

        try:
            for shortcut in widget._shortcuts:
                keys = shortcut.key.toString()
                if keys in ("Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z"):
                    CHECKS.check(not shortcut.autoRepeat,
                                 "{} does not autorepeat while held".format(keys))

            widget.onActivateEffect("Paint")
            pump()
            widget.UNDO_MEMORY_BUDGET_MB = 1e-6  # the history floors at a few states
            for offset in (0, 3, 6, 9):
                startRas = widget.logic.centroidToRAS(
                    (PAINT_START_IJK[0], PAINT_START_IJK[1] + offset, PAINT_START_IJK[2]))
                endRas = widget.logic.centroidToRAS(
                    (PAINT_END_IJK[0], PAINT_START_IJK[1] + offset, PAINT_END_IJK[2]))
                dragBetween(red, startRas, endRas)
            CHECKS.check(editor.maximumNumberOfUndoStates == widget.MIN_UNDO_STATES,
                         "precondition: the history is down to its floor",
                         "cap {}".format(editor.maximumNumberOfUndoStates))
            CHECKS.check(len(widget._strokeStarts) >= 3,
                         "precondition: stroke marks outlive the states they point at",
                         "{} marks".format(len(widget._strokeStarts)))

            widget.lesionTable.clearSelection()
            pump(0.2)
            CHECKS.check(editor.readOnly,
                         "precondition: no lesion selected, so the brush is locked and the "
                         "editor's own Undo button cannot tell whether a step exists")

            segmentation = widget.logic.segmentationNode.GetSegmentation()
            for eventName in ("SourceRepresentationModified", "RepresentationModified",
                              "SegmentModified", "SegmentAdded", "SegmentRemoved"):
                eventId = getattr(slicer.vtkSegmentation, eventName, None)
                if eventId is not None:
                    observed.append((segmentation, segmentation.AddObserver(eventId, countEvent)))
            widget._stepHistory = stepHistory
            widget.refreshLesions = refreshLesions
            timer.setInterval(20)
            timer.connect("timeout()", tick)
            gap["last"] = time.time()
            timer.start()
            started = time.time()
            for _ in range(clicks):
                postClick()
            pump(0.3)
            burstSeconds = time.time() - started
            refreshesDuringBurst = stats["refreshes"]
            pump(1.2)  # past the lesion refresh debounce
            timer.stop()
            print("  {} clicks: {} presses handled, {} raw steps, {} found nothing, "
                  "longest event-loop gap {:.3f} s".format(
                      clicks, stats["presses"], stats["tries"], stats["failed"], gap["max"]))

            CHECKS.check(stats["presses"] == clicks,
                         "every click was handled, none dropped",
                         "{} of {}".format(stats["presses"], clicks))
            CHECKS.check(stats["maxDepth"] == 1 and stats["pressesInsideRecount"] == 0,
                         "no press ran nested inside another press or its lesion recount",
                         "depth {}, {} presses inside a recount".format(
                             stats["maxDepth"], stats["pressesInsideRecount"]))
            CHECKS.check(stats["failed"] <= clicks,
                         "a press stops at the first step that finds nothing to undo",
                         "{} empty steps for {} presses".format(stats["failed"], clicks))
            CHECKS.check(stats["tries"] - stats["failed"] <= widget.MIN_UNDO_STATES,
                         "only the states actually kept were stepped through",
                         "{} real steps".format(stats["tries"] - stats["failed"]))
            CHECKS.check(refreshesDuringBurst == 0,
                         "no lesion recount inside the burst",
                         "{} recounts".format(refreshesDuringBurst))
            if widget.autoRefreshCheckBox.checked:
                CHECKS.check(stats["refreshes"] == 1,
                             "one debounced recount once the clicks stop",
                             "{} recounts".format(stats["refreshes"]))
            CHECKS.check(gap["max"] < 1.0,
                         "the panel never stopped responding",
                         "longest gap {:.3f} s, burst handled in {:.3f} s".format(
                             gap["max"], burstSeconds))
        finally:
            timer.stop()
            for observedObject, tag in observed:
                observedObject.RemoveObserver(tag)
            for name in ("_stepHistory", "refreshLesions", "UNDO_MEMORY_BUDGET_MB"):
                if name in vars(widget):
                    delattr(widget, name)
            widget.loadCurrentCase()
            pump()
            widget.lesionTable.selectRow(0)
            pump()
        CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), loaded),
                     "reloading the case puts the loaded mask back for the steps below")

    def test_06_sphere_threshold_2d(self):
        CHECKS.step("Sphere threshold: a ball, then a disc with 2D ticked")
        widget = self.widget
        lesion = widget.selectedLesion()
        combo = widget.activeLabelComboBox
        combo.currentIndex = combo.findData(3)
        widget.onActiveLabelChanged(combo.currentIndex)
        widget.onActivateEffect(gtreview.SPHERE_THRESHOLD_EFFECT)  # what the 3 key calls
        pump()
        segmentLabel = widget.logic.labelValueForSegmentId(widget.editor.currentSegmentID())
        CHECKS.check(
            combo.itemData(combo.currentIndex) == 3 and segmentLabel == 3,
            "choosing a tool by its key keeps the label picked in Active label",
            "box {}, editor segment {}, selected lesion's label {}".format(
                combo.itemData(combo.currentIndex), segmentLabel,
                None if lesion is None else int(lesion.label)),
        )
        if lesion is not None:
            widget._selectSegmentForLabel(int(lesion.label))
            pump(0.1)
        widget.onActivateEffect(gtreview.SPHERE_THRESHOLD_EFFECT)
        pump()
        effect = widget.editor.activeEffect()
        CHECKS.check(
            effect is not None and effect.name == gtreview.SPHERE_THRESHOLD_EFFECT,
            "the Sphere threshold effect is active",
        )
        scripted = effect.self()
        red = sliceWidgetNamed("Red")
        before = widget.logic.exportLabelmapArrayIJK()
        seedRas = widget.logic.centroidToRAS(SPHERE_SEED_IJK)
        edgeRas = widget.logic.centroidToRAS(SPHERE_EDGE_IJK)

        scripted.twoDCheckBox.checked = False
        pump(0.1)
        CHECKS.check(not scripted.twoDimensional(), "2D is off for the first drag")
        dragBetween(red, seedRas, edgeRas, steps=8)

        sourceNode = widget.editor.sourceVolumeNode()
        expectedSeed = rasToIjk(sourceNode, seedRas)
        CHECKS.check(tuple(scripted.seedIjk) == expectedSeed,
                     "the click landed on the voxel it was aimed at",
                     "{} != {}".format(tuple(scripted.seedIjk), expectedSeed))

        ball = (widget.logic.exportLabelmapArrayIJK() != 0) & (before == 0)
        CHECKS.check(bool(ball.any()), "the drag grew the lesion from the seed",
                     "{} voxels".format(int(ball.sum())))
        CHECKS.check(len(planesOf(ball)) >= 2,
                     "with 2D off the ball reaches the neighbouring slices",
                     "planes {}".format(planesOf(ball)))

        widget.onUndo()
        pump()
        CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), before),
                     "one Undo press removes the whole ball")

        scripted.twoDCheckBox.checked = True
        pump(0.1)
        CHECKS.check(scripted.twoDimensional(), "2D is on for the second drag")
        dragBetween(red, seedRas, edgeRas, steps=8)

        disc = (widget.logic.exportLabelmapArrayIJK() != 0) & (before == 0)
        CHECKS.check(bool(disc.any()), "the 2D drag added voxels",
                     "{} voxels".format(int(disc.sum())))
        CHECKS.check(planesOf(disc) == [SPHERE_SEED_IJK[2]],
                     "2D kept every voxel on the slice that was drawn on",
                     "planes {}".format(planesOf(disc)))
        CHECKS.check(int(disc.sum()) < int(ball.sum()),
                     "the disc is a subset of the ball, not a differently-sized guess",
                     "{} vs {} voxels".format(int(disc.sum()), int(ball.sum())))

        widget.onUndo()
        pump()
        CHECKS.check(np.array_equal(widget.logic.exportLabelmapArrayIJK(), before),
                     "one Undo press removes the disc")
        widget.onStopEditing()
        pump(0.1)

    def test_07_delete_a_lesion_from_its_row(self):
        CHECKS.step("Deleting a lesion with the trash button on its row")
        widget = self.widget
        widget.refreshLesions()
        pump()
        target = next(l for l in widget.lesionList if int(l.label) == 2)
        CHECKS.check(target.voxel_count == LESION_B_VOXELS,
                     "the lesion to delete is the one that was written to disk",
                     "{} voxels".format(target.voxel_count))
        row = rowOfLesion(widget, target.index)
        CHECKS.check(row >= 0, "the lesion has a row in the table")
        button = widget.lesionTable.cellWidget(row, widget.LESION_DELETE_COLUMN)
        CHECKS.check(button is not None, "the row carries a delete button")

        with ConfirmStub(True) as confirm:
            button.click()
            pump()
        CHECKS.check(len(confirm.prompts) == 1,
                     "the delete asked for confirmation exactly once",
                     "{} prompts".format(len(confirm.prompts)))
        CHECKS.check(str(target.voxel_count) in confirm.prompts[0],
                     "the prompt spells out what is about to go",
                     confirm.prompts[0].replace("\n", " ") if confirm.prompts else "")
        CHECKS.check("({})".format(nativeText("Ctrl+Z")) in confirm.prompts[0],
                     "and names the Undo key the way this platform spells it",
                     confirm.prompts[0].replace("\n", " "))

        exported = widget.logic.exportLabelmapArrayIJK()
        CHECKS.check(int((exported == 2).sum()) == 0,
                     "every voxel of the deleted lesion is gone",
                     "{} left".format(int((exported == 2).sum())))
        CHECKS.check(int((exported == 1).sum()) == LESION_A_VOXELS,
                     "the other lesion is untouched")
        CHECKS.check(len(widget.lesionList) == 1,
                     "the table lists the one remaining lesion",
                     "{} rows".format(len(widget.lesionList)))
        CHECKS.check(widget.unsavedChanges, "the case is marked as edited")

    def test_08_delete_the_review(self):
        CHECKS.step("Saving a review, then removing it with Delete review")
        widget = self.widget
        case = widget.currentCase()
        CHECKS.check(not widget.deleteReviewButton.enabled,
                     "Delete review is disabled while there is no saved review")

        # saving refuses until every listed lesion is ticked Done
        for row in range(widget.lesionTable.rowCount):
            widget.lesionTable.item(row, widget.LESION_COLUMN_DONE).setCheckState(
                qt.Qt.Checked
            )
        pump(0.1)
        widget.onSave()
        pump()
        CHECKS.check(os.path.isfile(case.reviewed_path), "the review was written",
                     case.reviewed_path)
        CHECKS.check(widget.deleteReviewButton.enabled,
                     "Delete review is enabled once the file exists")
        saved, _geometry = maskio.read_mask(case.reviewed_path)
        CHECKS.check(int((saved == 2).sum()) == 0,
                     "the saved file carries the deletion")

        widgetMaskio = moduleGlobals(widget)["maskio"]
        originalRemove = widgetMaskio.remove_file
        removed = []

        def recordingRemove(path):
            removed.append(path)
            return originalRemove(path)

        widgetMaskio.remove_file = recordingRemove
        try:
            with ConfirmStub(True) as confirm:
                widget.deleteReviewButton.click()
                pump()
        finally:
            widgetMaskio.remove_file = originalRemove
        CHECKS.check(removed == [case.reviewed_path],
                     "the file is removed through maskio.remove_file, which waits out "
                     "a file another program still holds on Windows",
                     str(removed))
        CHECKS.check(bool(confirm.prompts), "Delete review asked first")
        CHECKS.check(case.reviewed_path in confirm.prompts[0],
                     "the prompt names the file it is about to erase",
                     confirm.prompts[0].replace("\n", " "))
        CHECKS.check(not os.path.isfile(case.reviewed_path),
                     "the reviewed file is gone from disk")
        CHECKS.check(not widget.deleteReviewButton.enabled,
                     "the button disables itself again")
        CHECKS.check(widget.logic.maskPath == case.masks["seg"],
                     "the case reopened from its ORIGINAL mask",
                     str(widget.logic.maskPath))
        exported = widget.logic.exportLabelmapArrayIJK()
        CHECKS.check(np.array_equal(exported.astype(np.uint8), self.sourceMask),
                     "the deleted lesion is back, the mask is the one on disk")
        CHECKS.check(len(widget.lesionList) == 2,
                     "both lesions are listed again",
                     "{} lesions".format(len(widget.lesionList)))


    def test_08b_failed_save_before_scene_close(self):
        CHECKS.step("A save asked for as the scene closes, failing, is shown to the reviewer")
        widget = self.widget
        case = widget.currentCase()
        CHECKS.check(case is not None and widget.logic.case is not None,
                     "precondition: a case is open")
        logPath = os.path.join(self.tempRoot, moduleGlobals(widget)["sessionlog"].LOG_FILE_NAME)
        saves = []
        shown = []

        def failingSave():
            saves.append(True)
            raise OSError(28, "No space left on device", case.reviewed_path)

        def recordError(text, *args, **kwargs):
            del args, kwargs
            shown.append(str(text))

        originalErrorDisplay = slicer.util.errorDisplay
        widget.saveCurrentCase = failingSave
        slicer.util.errorDisplay = recordError
        try:
            result = widget._saveBeforeSceneClose()
        finally:
            del widget.saveCurrentCase
            slicer.util.errorDisplay = originalErrorDisplay
        CHECKS.check(saves == [True], "the save was attempted")
        CHECKS.check(result is None,
                     "the failure goes no further, so the scene close carries on")
        CHECKS.check(len(shown) == 1, "the reviewer is told, once", " | ".join(shown))
        message = shown[0] if shown else ""
        CHECKS.check("Saving case {} failed".format(case.case_id) in message,
                     "the message names the case", message.replace("\n", " "))
        CHECKS.check("unsaved edits are lost" in message,
                     "and says its unsaved edits are lost", message.replace("\n", " "))
        CHECKS.check("No space left on device" in message, "and why the save failed")
        logged = [entry for entry in logEntries(readText(logPath))
                  if "GTReview: saving before scene close failed" in entry]
        CHECKS.check(bool(logged) and "Traceback" in logged[-1] and "OSError" in logged[-1],
                     "GTReview.log keeps the failure with its traceback",
                     logged[-1].replace("\n", " | ")[:300] if logged else "no entry")

    def test_09_multi_sequence_case(self):
        CHECKS.step("Opening a case with four sequences")
        widget = self.widget
        self.__class__.multiRoot = tempfile.mkdtemp(prefix="gtreview_integration_multi_")
        _caseDir, multiMask = buildMultiSequenceCase(self.multiRoot)

        # test_01 picked a layout by hand; a fresh reviewer has not
        widget._layoutChosenByUser = False
        widget.datasetPathEdit.currentPath = self.multiRoot
        with ConfirmStub(True):
            widget.onLoadDataset()
        pump()
        case = widget.currentCase()
        CHECKS.check(case is not None and case.case_id == MULTI_CASE_ID,
                     "the multi-sequence case was loaded")
        CHECKS.check(sorted(widget.logic.volumeNodes) == sorted(MULTI_SEQUENCES),
                     "every sequence became its own volume node",
                     str(sorted(widget.logic.volumeNodes)))

        # ---- layout chosen on its own, one axial view per sequence ----------
        layoutManager = slicer.app.layoutManager()
        CHECKS.check(layoutManager.layout == gtreview.SEQUENCES_LAYOUT_ID,
                     "the Sequences (axial) layout was chosen automatically",
                     "layout id {}".format(layoutManager.layout))
        CHECKS.check(widget.layoutComboBox.currentText == "Sequences (axial)",
                     "and the Layout box says so")
        order = widget._sequenceKeys()
        CHECKS.check(order == ["t1", "t1c", "t2", "flair"],
                     "the views are ordered t1, t1c, t2, flair", str(order))
        for key in order:
            view = sliceWidgetNamed(key)
            CHECKS.check(view.mrmlSliceNode().GetOrientation() == "Axial",
                         "the {} view is axial".format(key),
                         view.mrmlSliceNode().GetOrientation())
            CHECKS.check(backgroundNameOf(view) == "{}_{}".format(MULTI_CASE_ID, key),
                         "the {} view shows the {} volume".format(key, key),
                         str(backgroundNameOf(view)))
            CHECKS.check(not view.mrmlSliceCompositeNode().GetForegroundVolumeID(),
                         "the {} view has no foreground blend".format(key))
        visible = [k for k in order if sliceWidgetNamed(k).visible]
        CHECKS.check(visible == order, "all four views are on screen", str(visible))
        CHECKS.check(not layoutManager.sliceWidget("Red").visible,
                     "the Red view is not part of it")

        # ---- linked interaction ---------------------------------------------
        for key in order:
            composite = sliceWidgetNamed(key).mrmlSliceCompositeNode()
            CHECKS.check(composite.GetLinkedControl() and composite.GetHotLinkedControl(),
                         "the {} view is hot-linked".format(key))
        offsets = [sliceWidgetNamed(k).mrmlSliceNode().GetSliceOffset() for k in order]
        CHECKS.check(max(offsets) - min(offsets) < 1e-6,
                     "the views start on the same slice", str(offsets))
        before = offsets[0]
        scrollLikeTheMouse(sliceWidgetNamed("t2"), 2 * SPACING[2])
        after = [sliceWidgetNamed(k).mrmlSliceNode().GetSliceOffset() for k in order]
        CHECKS.check(all(abs(o - (before + 2 * SPACING[2])) < 1e-6 for o in after),
                     "scrolling the t2 view moved all four views two slices",
                     str(after))

        # ---- dilation before connected components ---------------------------
        CHECKS.check(len(widget.lesionList) == 2,
                     "the two blocks one voxel apart are ONE lesion, the far one another",
                     "{} lesions".format(len(widget.lesionList)))
        biggest = widget.lesionList[0]
        CHECKS.check(biggest.voxel_count == GAP_VOXELS,
                     "the bridged lesion counts only real voxels, not the grown ones",
                     "{} voxels".format(biggest.voxel_count))
        CHECKS.check(widget.lesionList[1].voxel_count == FAR_C_VOXELS,
                     "the far lesion is untouched")
        CHECKS.check(np.array_equal(widget.componentMap != 0, multiMask != 0),
                     "the component map covers exactly the mask voxels")
        _map, undilated = widget.logic.computeLesions(dilate=0)
        CHECKS.check(len(undilated) == 3, "without dilation the same mask has three lesions",
                     "{} lesions".format(len(undilated)))

        # ---- label 3 (Edema) ------------------------------------------------
        CHECKS.check(sorted(widget.logic.labelValues()) == [1, 2, 3],
                     "labels 1, 2 and 3 are all present on a mask that has no 3",
                     str(sorted(widget.logic.labelValues())))
        CHECKS.check(widget.activeLabelComboBox.findData(3) >= 0,
                     "Active label offers 3 - Edema")
        CHECKS.check(widget.paintOverComboBox.findData(3) >= 0,
                     "Paint over offers Only 3 - Edema")
        CHECKS.check(gtreview.nameForLabelValue(3) == "3 - Edema",
                     "and it is called Edema", gtreview.nameForLabelValue(3))
        widget.lesionTable.selectRow(1)  # the far, label-2 lesion
        pump(0.1)
        lesion = widget.selectedLesion()
        CHECKS.check(lesion is not None and int(lesion.label) == 2, "the far lesion is selected")
        widget._markEdit()
        widget.logic.changeLesionLabel(
            gtreview.lesions.lesion_mask(widget.componentMap, lesion.index), 3
        )
        widget.unsavedChanges = True
        widget.refreshLesions()
        pump(0.1)
        exported = widget.logic.exportLabelmapArrayIJK()
        CHECKS.check(int((exported == 3).sum()) == FAR_C_VOXELS,
                     "relabelling a lesion to 3 exports those voxels as 3",
                     "{} voxels".format(int((exported == 3).sum())))
        CHECKS.check(int((exported == 2).sum()) == 0, "and nothing is left as 2")
        CHECKS.check(any(int(l.label) == 3 for l in widget.lesionList),
                     "the lesion list reports the label-3 lesion")
        widget._selectSegmentForLabel(3)
        pump(0.1)
        CHECKS.check(
            widget.logic.labelValueForSegmentId(widget.editor.currentSegmentID()) == 3,
            "the brush can be set to label 3",
        )

        # ---- a layout picked by hand sticks across reloads ------------------
        index = widget.layoutComboBox.findText("Four-Up")
        widget.layoutComboBox.currentIndex = index
        widget.onLayoutChanged()
        pump()
        CHECKS.check(layoutManager.layout == gtreview.layoutId("SlicerLayoutFourUpView", 3),
                     "choosing Four-Up by hand applies it")
        with ConfirmStub(True):
            widget.setCurrentCaseIndex(widget.currentCaseIndex, force=True)
        pump()
        CHECKS.check(layoutManager.layout == gtreview.layoutId("SlicerLayoutFourUpView", 3),
                     "reloading the case keeps the layout the reviewer chose",
                     "layout id {}".format(layoutManager.layout))
        red = sliceWidgetNamed("Red")
        CHECKS.check(backgroundNameOf(red) == "{}_{}".format(
            MULTI_CASE_ID, widget.backgroundComboBox.currentText),
            "in Four-Up every view shows the Image box's choice",
            str(backgroundNameOf(red)))
        widget.unsavedChanges = False
        if self.multiRoot and os.path.isdir(self.multiRoot):
            shutil.rmtree(self.multiRoot, ignore_errors=True)

    def test_10_shortcut_labels_and_the_mac_delete_key(self):
        CHECKS.step("Shortcut labels in the platform's spelling, and the Mac delete key")
        widget = self.widget
        namespace = moduleGlobals(widget)

        # ---- labels built at setup: Qt's NativeText of the keys --------------
        CHECKS.check(namespace["shortcutText"]("Ctrl+Z") == nativeText("Ctrl+Z"),
                     "shortcutText is QKeySequence's NativeText")
        CHECKS.check(plainToolTip(widget.undoButton) == widget._undoToolTip()
                     == "Undo the last edit ({})".format(nativeText("Ctrl+Z")),
                     "the Undo tooltip names the native Undo key", widget.undoButton.toolTip)
        CHECKS.check(plainToolTip(widget.redoButton) == widget._redoToolTip()
                     == "Redo ({} or {})".format(nativeText("Ctrl+Y"), nativeText("Ctrl+Shift+Z")),
                     "the Redo tooltip names both native Redo keys", widget.redoButton.toolTip)
        CHECKS.check(nativeText("Ctrl+Z") in widget.deleteReviewButton.toolTip,
                     "the Delete review tooltip names the native Undo key")
        CHECKS.check("Esc cancels" in widget.newLesionButton.toolTip,
                     "the New lesion tooltip says Esc as the word, never the Mac symbol",
                     widget.newLesionButton.toolTip)
        footer = " ".join(str(label.text)
                          for label in widget.shortcutsFrame.findChildren(qt.QLabel))
        CHECKS.check(all(text in footer for text in (
            nativeText("Ctrl+Z"), nativeText("Ctrl+S"), "Esc",
            namespace["modifierGestureText"]("Ctrl", "wheel"))),
            "the footer on screen shows the native spellings, and Esc", footer)

        # ---- labels built on demand ask shortcutText -------------------------
        # On Linux NativeText spells a key exactly as a hand-written label
        # would, so comparing with it cannot catch a label that never asked Qt.
        # A marker standing in for shortcutText can: whatever is built while it
        # stands in must carry it.
        originalShortcutText = namespace["shortcutText"]
        widget.lesionTable.selectRow(0)
        pump(0.1)
        CHECKS.check(widget.selectedLesion() is not None, "precondition: a lesion is selected")
        try:
            namespace["shortcutText"] = lambda keys: "<" + keys + ">"
            CHECKS.check("<Ctrl+S> saves without moving on" in widget._saveAndNextToolTip(),
                         "the Save & next case tooltip asks shortcutText for the Save key",
                         widget._saveAndNextToolTip())
            CHECKS.check(widget._undoToolTip() == "Undo the last edit (<Ctrl+Z>)",
                         "the Undo tooltip asks shortcutText for the Undo key",
                         widget._undoToolTip())
            CHECKS.check(widget._redoToolTip() == "Redo (<Ctrl+Y> or <Ctrl+Shift+Z>)",
                         "the Redo tooltip asks shortcutText for both Redo keys",
                         widget._redoToolTip())
            CHECKS.check("-- <Ctrl+Z> does not reach it." in widget._deleteReviewToolTip(),
                         "the Delete review tooltip asks shortcutText for the Undo key",
                         widget._deleteReviewToolTip())
            keyRows = dict((what, keys) for keys, what in widget._shortcutKeyRows())
            CHECKS.check(keyRows["undo / redo"] == "<Ctrl+Z> / <Ctrl+Y>",
                         "footer: undo / redo", keyRows["undo / redo"])
            CHECKS.check(keyRows["save"] == "<Ctrl+S>", "footer: save", keyRows["save"])
            CHECKS.check(keyRows["delete the selected lesion"]
                         == ("<Backspace>" if namespace["IS_MAC"] else "<Del>"),
                         "footer: the delete key as this keyboard labels it",
                         keyRows["delete the selected lesion"])
            CHECKS.check(keyRows["stop editing"] == "Esc",
                         "footer: stop editing is the word Esc, not a spelling Qt picks",
                         keyRows["stop editing"])
            viewRows = dict((what, keys) for keys, what in widget._shortcutViewRows())
            for modifier, gesture, what in (("Ctrl", "wheel", "zoom too"),
                                            ("Shift", "drag", "move the image too")):
                CHECKS.check(viewRows[what].startswith("<{}+".format(modifier))
                             and viewRows[what].endswith(gesture),
                             "footer: {} is the {} modifier, as shortcutText spells it, "
                             "plus {}".format(what, modifier, gesture), viewRows[what])
            with ConfirmStub(False) as confirm:
                widget.onDeleteLesion()
                pump(0.1)
            CHECKS.check(len(confirm.prompts) == 1 and "(<Ctrl+Z>)" in confirm.prompts[0],
                         "the delete-lesion prompt asks shortcutText for the Undo key",
                         confirm.prompts[0].replace("\n", " ") if confirm.prompts else "no prompt")
            with ConfirmStub(False) as confirm:
                widget.onDeleteLabel()
                pump(0.1)
            CHECKS.check(len(confirm.prompts) == 1
                         and "This can be undone (<Ctrl+Z>)." in confirm.prompts[0],
                         "the delete-label prompt asks shortcutText for the Undo key",
                         confirm.prompts[0].replace("\n", " ") if confirm.prompts else "no prompt")
            widget._installLesionDeleteButtons()
            rowButton = widget.lesionTable.cellWidget(0, widget.LESION_DELETE_COLUMN)
            CHECKS.check(rowButton is not None and "<Ctrl+Z>" in plainToolTip(rowButton),
                         "a row's delete button asks shortcutText for the Undo key",
                         rowButton.toolTip if rowButton is not None else "no button")
        finally:
            namespace["shortcutText"] = originalShortcutText
            # the row buttons built under the marker must not stay on screen
            widget._installLesionDeleteButtons()
        rowButton = widget.lesionTable.cellWidget(0, widget.LESION_DELETE_COLUMN)
        CHECKS.check(rowButton is not None and nativeText("Ctrl+Z") in rowButton.toolTip,
                     "with shortcutText back, the row buttons name the native Undo key again")

        # ---- the Mac delete key is Backspace, and only a Mac binds it -------
        originalMac = namespace["IS_MAC"]
        originalFocus = namespace["focusedTextInput"]
        deletes = []

        def backspaceRows():
            return [position for position, shortcut in enumerate(widget._shortcuts)
                    if shortcut.key.toString() == "Backspace"]

        try:
            namespace["IS_MAC"] = False
            widget.installShortcuts()
            CHECKS.check(not backspaceRows(), "off a Mac, Backspace is not bound")
            CHECKS.check(any(s.key.toString() == "Del" for s in widget._shortcuts),
                         "Delete is bound")
            countOffMac = len(widget._shortcuts)

            namespace["IS_MAC"] = True
            CHECKS.check(dict((what, keys) for keys, what in widget._shortcutKeyRows())
                         ["delete the selected lesion"] == nativeText("Backspace"),
                         "on a Mac the footer names the delete key, which sends Backspace")
            widget.onDeleteLesion = lambda: deletes.append(True)
            widget.installShortcuts()
            rows = backspaceRows()
            CHECKS.check(len(rows) == 1 and len(widget._shortcuts) == countOffMac + 1,
                         "on a Mac, reinstalling the shortcuts adds one Backspace binding",
                         "{} Backspace, {} shortcuts".format(len(rows), len(widget._shortcuts)))
            CHECKS.check(any(s.key.toString() == "Del" for s in widget._shortcuts),
                         "and Delete stays bound next to it")
            CHECKS.check(widget._shortcuts[rows[0]].context == qt.Qt.ApplicationShortcut,
                         "Backspace works wherever the focus is in the main window")
            handler = widget._shortcutHandlers[rows[0]]
            namespace["focusedTextInput"] = lambda: None
            handler()
            CHECKS.check(len(deletes) == 1, "Backspace deletes the selected lesion")
            namespace["focusedTextInput"] = lambda: widget.datasetPathEdit
            handler()
            CHECKS.check(len(deletes) == 1,
                         "but a focused text box keeps Backspace for itself")
        finally:
            namespace["IS_MAC"] = originalMac
            namespace["focusedTextInput"] = originalFocus
            if "onDeleteLesion" in vars(widget):
                del widget.onDeleteLesion
            widget.installShortcuts()
        CHECKS.check(len(backspaceRows()) == (1 if originalMac else 0),
                     "the platform's own shortcuts are back")

    def test_11_denied_batch_directory(self):
        CHECKS.step("A batch directory Slicer may not read is reported as such")
        widget = self.widget
        namespace = moduleGlobals(widget)
        datasetModule = namespace["dataset"]
        originalDiscover = datasetModule.discover_cases
        originalMac = namespace["IS_MAC"]
        originalPath = str(widget.datasetPathEdit.currentPath)
        casesBefore = list(widget.cases)
        caseBefore = widget.currentCase()
        historyBefore = widget._datasetHistory()
        deniedRoot = os.path.join(tempfile.gettempdir(), "gtreview_integration_denied")
        logName = namespace["sessionlog"].LOG_FILE_NAME
        logBefore = widget.sessionLog.path
        asked = []
        shown = []

        def deny(root):
            asked.append(root)
            raise PermissionError(13, "Permission denied", root)

        def recordError(text, *args, **kwargs):
            del args, kwargs
            shown.append(str(text))

        originalErrorDisplay = slicer.util.errorDisplay
        widget.unsavedChanges = False
        try:
            datasetModule.discover_cases = deny
            slicer.util.errorDisplay = recordError
            widget.datasetPathEdit.currentPath = deniedRoot
            typed = str(widget.datasetPathEdit.currentPath)

            namespace["IS_MAC"] = False
            widget.onLoadDataset()
            pump(0.1)
            CHECKS.check(asked == [typed], "discovery was asked for the typed directory",
                         str(asked))
            CHECKS.check(len(shown) == 1, "the denial is reported once, without a crash",
                         " | ".join(shown))
            message = shown[0] if shown else ""
            CHECKS.check(message.startswith("Slicer was denied access to {}".format(typed)),
                         "it says Slicer was denied access to the directory",
                         message.replace("\n", " "))
            CHECKS.check("0 cases" not in message
                         and "0 cases" not in str(widget.caseStatusLabel.text),
                         "rather than reporting 0 cases found")
            CHECKS.check("Privacy & Security" not in message,
                         "off a Mac there is no macOS privacy hint")
            CHECKS.check(len(widget.cases) == len(casesBefore)
                         and all(a is b for a, b in zip(widget.cases, casesBefore))
                         and widget.currentCase() is caseBefore,
                         "the dataset and the case already open stay as they were")
            CHECKS.check(widget._datasetHistory() == historyBefore,
                         "and the unreadable directory is not remembered")
            CHECKS.check(widget.sessionLog.path == logBefore
                         and not os.path.exists(os.path.join(deniedRoot, logName)),
                         "nor given a GTReview.log", str(widget.sessionLog.path))

            del shown[:]
            namespace["IS_MAC"] = True
            widget.onLoadDataset()
            pump(0.1)
            message = shown[0] if shown else ""
            CHECKS.check(len(shown) == 1 and message.startswith("Slicer was denied access to"),
                         "on a Mac the denial is reported the same way",
                         message.replace("\n", " "))
            CHECKS.check("System Settings > Privacy & Security > Files and Folders" in message,
                         "with the macOS privacy setting to allow it in",
                         message.replace("\n", " "))
        finally:
            datasetModule.discover_cases = originalDiscover
            slicer.util.errorDisplay = originalErrorDisplay
            namespace["IS_MAC"] = originalMac
            widget.datasetPathEdit.currentPath = originalPath

    def test_12_history_lists_a_directory_once(self):
        CHECKS.step("The batch-directory history lists one directory once")
        widget = self.widget
        settings = slicer.app.userSettings()
        historyRoot = tempfile.mkdtemp(prefix="gtreview_integration_history_")
        originalPath = str(widget.datasetPathEdit.currentPath)
        try:
            batch = os.path.join(historyRoot, "batch_01")
            other = os.path.join(historyRoot, "batch_02")
            settings.setValue(gtreview.DATASET_HISTORY_KEY, [
                batch + os.sep, other, os.path.join(historyRoot, ".", "batch_01"),
            ])
            widget.datasetPathEdit.currentPath = batch
            current = str(widget.datasetPathEdit.currentPath)
            widget._rememberDatasetPath()
            history = widget._datasetHistory()
            CHECKS.check(history == [current, other],
                         "a trailing separator or a ./ does not make a second entry",
                         str(history))

            # Windows spellings of one folder, judged the way Windows judges
            # them: ntpath stands in for os.path only while the history is built
            settings.setValue(gtreview.DATASET_HISTORY_KEY, [
                "C:/Data/Batch_01", "D:\\other", "c:\\data\\batch_01\\",
            ])
            widget.datasetPathEdit.currentPath = "C:\\Data\\Batch_01"
            current = str(widget.datasetPathEdit.currentPath)
            with mock.patch.object(os.path, "normcase", ntpath.normcase), \
                    mock.patch.object(os.path, "normpath", ntpath.normpath):
                widget._rememberDatasetPath()
            history = widget._datasetHistory()
            CHECKS.check(history == [current, "D:\\other"],
                         "C:/Data/Batch_01 and c:\\data\\batch_01\\ are the directory just "
                         "loaded, so neither is listed again",
                         str(history))
        finally:
            widget.datasetPathEdit.currentPath = originalPath
            shutil.rmtree(historyRoot, ignore_errors=True)
        # tearDownClass puts the reviewer's own history back

    def test_13_theme_switch_retints_the_panel(self):
        CHECKS.step("Switching between light and dark re-tints the sections and icons")
        widget = self.widget
        app = slicer.app
        original = qt.QPalette(app.palette())
        section = next(s for s in widget._sections
                       if s.objectName == "GTReviewSectionDataset")
        accent = widget.ACCENT_DATASET
        startedDark = widget._isDarkTheme()
        undoIcon = widget.undoButton.icon.cacheKey()
        flipped = qt.QPalette(original)
        window, text = ("#efefef", "#101010") if startedDark else ("#262626", "#f0f0f0")
        flipped.setColor(qt.QPalette.Window, qt.QColor(window))
        flipped.setColor(qt.QPalette.ButtonText, qt.QColor(text))
        try:
            app.setPalette(flipped)
            pump(0.3)
            CHECKS.check(widget._isDarkTheme() != startedDark,
                         "precondition: the application palette went from {} to {}".format(
                             "dark" if startedDark else "light",
                             "light" if startedDark else "dark"))
            expectedFill = widget._sectionColors(accent)[0]
            fill = section.palette.color(qt.QPalette.Window).name()
            CHECKS.check(fill == expectedFill,
                         "the palette change re-ran the section tinting for the new theme",
                         "{} vs {}".format(fill, expectedFill))
            CHECKS.check(section.autoFillBackground, "and the fill is still painted")
            CHECKS.check(widget.undoButton.icon.cacheKey() != undoIcon,
                         "the drawn Undo icon was redrawn")
        finally:
            app.setPalette(original)
            pump(0.3)
        expectedFill = widget._sectionColors(accent)[0]
        CHECKS.check(section.palette.color(qt.QPalette.Window).name() == expectedFill,
                     "switching back tints the sections back")

    def test_13b_scene_close_saves_through_the_shared_save(self):
        CHECKS.step("A scene closing with unsaved edits offers the save and saves through "
                    "_saveBeforeSceneClose")
        widget = self.widget
        case = widget.logic.case
        CHECKS.check(case is not None, "precondition: a case is open")
        asked = []
        saves = []

        def answering(choice):
            def ask(caseId):
                asked.append(caseId)
                return choice
            return ask

        def recordSave():
            # what the save found: the edits still unsaved, the case still open
            saves.append((widget.unsavedChanges, widget.logic.case is not None))

        widget._saveBeforeSceneClose = recordSave
        try:
            widget.unsavedChanges = True
            widget._askSaveBeforeSceneClose = answering(False)
            widget._offerSaveBeforeSceneClose()
            CHECKS.check(asked == [case.case_id] and not saves,
                         "answered Discard, the prompt names the case and nothing is saved",
                         "asked {}, saves {}".format(asked, saves))
            del asked[:]
            widget._askSaveBeforeSceneClose = answering(True)
            widget.onSceneStartClose()
            CHECKS.check(asked == [case.case_id],
                         "a scene starting to close with unsaved edits asks, naming the case",
                         str(asked))
            CHECKS.check(saves == [(True, True)],
                         "answered Save, onSceneStartClose saves through _saveBeforeSceneClose "
                         "while the edits and the case are still there", str(saves))
            CHECKS.check(not widget.unsavedChanges and widget.logic.case is None,
                         "and tears the case down after that")
        finally:
            for name in ("_saveBeforeSceneClose", "_askSaveBeforeSceneClose"):
                if name in vars(widget):
                    delattr(widget, name)
            widget.unsavedChanges = False
            # what the end of a real scene close does; the next step loads a batch
            widget.onSceneEndClose()
            pump(0.1)

    def test_14_folder_without_cases_closes_the_session_log(self):
        CHECKS.step("A folder with no cases gets no GTReview.log, and the dropped batch's "
                    "log is closed")
        widget = self.widget
        logName = moduleGlobals(widget)["sessionlog"].LOG_FILE_NAME
        emptyRoot = tempfile.mkdtemp(prefix="gtreview_integration_empty_")
        originalPath = str(widget.datasetPathEdit.currentPath)
        batchLog = os.path.join(self.tempRoot, logName)
        try:
            # a batch first, so there is a log to close: the batch of the steps
            # before this one has been deleted along with its log
            widget.unsavedChanges = False
            widget.datasetPathEdit.currentPath = self.tempRoot
            widget.onLoadDataset()
            pump(0.1)
            CHECKS.check(len(widget.cases) == 1 and widget.sessionLog.path is not None
                         and os.path.samefile(widget.sessionLog.path, batchLog),
                         "precondition: a batch is loaded and logging into its folder",
                         str(widget.sessionLog.path))

            # one level above a batch: a folder of batch folders and a stray file
            os.makedirs(os.path.join(emptyRoot, "batch_01", "IT_009"))
            with open(os.path.join(emptyRoot, "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("not a case\n")
            widget.unsavedChanges = False
            widget.datasetPathEdit.currentPath = emptyRoot
            typed = str(widget.datasetPathEdit.currentPath)
            widget.onLoadDataset()
            pump(0.1)
            CHECKS.check(not widget.cases and widget.logic.case is None,
                         "precondition: the folder holds no cases, so the batch is dropped",
                         "{} cases".format(len(widget.cases)))
            CHECKS.check(not os.path.exists(os.path.join(emptyRoot, logName)),
                         "loading it wrote no GTReview.log there", str(os.listdir(emptyRoot)))
            CHECKS.check(widget.sessionLog.path is None,
                         "the dropped batch's GTReview.log is closed, so on Windows its folder "
                         "can be moved, renamed or deleted", str(widget.sessionLog.path))
            closing = [entry for entry in logEntries(readText(batchLog))
                       if "no batch is loaded" in entry]
            CHECKS.check(bool(closing) and "0 cases found in {}".format(typed) in closing[-1],
                         "its last GTReview line says why it was closed",
                         closing[-1] if closing else "no such line")
            size = os.path.getsize(batchLog)
            marker = "GTReview: integration line logged after the batch was dropped"
            logging.info(marker)
            pump(0.1)
            CHECKS.check(os.path.getsize(batchLog) == size and marker not in readText(batchLog),
                         "a GTReview line logged afterwards is not appended to it")
        finally:
            widget.datasetPathEdit.currentPath = originalPath
            shutil.rmtree(emptyRoot, ignore_errors=True)

    def test_15_cleanup_takes_the_session_log_down(self):
        CHECKS.step("The panel's cleanup takes GTReview.log's handler and exception hook down")
        widget = self.widget
        sessionLog = widget.sessionLog
        mark = getattr(moduleGlobals(widget)["sessionlog"], "_HANDLER_MARK",
                       "is_gtreview_session_log")

        def gtreviewHandlers():
            return [handler for handler in logging.getLogger().handlers
                    if getattr(handler, mark, False)]

        # the hook the session log chained onto when the panel was set up
        previousHook = getattr(sessionLog, "_previous_excepthook", None)
        CHECKS.check(len(gtreviewHandlers()) == 1,
                     "precondition: one GTReview handler on the root logger",
                     "{} handlers".format(len(gtreviewHandlers())))
        CHECKS.check(previousHook is not None and sys.excepthook is not previousHook,
                     "precondition: GTReview's exception hook is installed")
        widget.cleanup()  # Slicer calls it again at exit; it must bear that
        pump(0.1)
        CHECKS.check(not gtreviewHandlers(), "no GTReview handler is left on the root logger",
                     "{} handlers".format(len(gtreviewHandlers())))
        CHECKS.check(sys.excepthook is previousHook, "sys.excepthook is the one from before",
                     repr(sys.excepthook))
        CHECKS.check(widget._watchdogTimer is None or not widget._watchdogTimer.isActive(),
                     "the watchdog timer has stopped")
        CHECKS.check(widget._errorLogModel is None, "Slicer's error log is no longer followed")
        CHECKS.check(sessionLog.path is None, "and GTReview.log is closed")


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def main():
    print("=" * 72)
    print("GTReview integration test (real widget, real slice views)")
    print("  Slicer  : {}".format(slicer.app.applicationVersion))
    print("  module  : {}".format(gtreview.__file__))
    print("=" * 72)
    sys.stdout.flush()

    suite = unittest.TestLoader().loadTestsFromTestCase(GTReviewIntegrationTest)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=0).run(suite)

    print("\n" + "=" * 72)
    print("SUMMARY: {} checks passed, {} failed, {} steps errored".format(
        len(CHECKS.passed), len(CHECKS.failed), len(result.errors)))
    for description, detail in CHECKS.failed:
        print("  FAILED: {}{}".format(description, ("  --  " + detail) if detail else ""))
    for test, message in result.errors:
        print("  ERROR : {}\n{}".format(test, message))
    ok = result.wasSuccessful() and not CHECKS.failed and bool(CHECKS.passed)
    print("RESULT: {}".format("PASS" if ok else "FAIL"))
    print("=" * 72)
    sys.stdout.flush()
    slicer.util.exit(0 if ok else 1)


main()
