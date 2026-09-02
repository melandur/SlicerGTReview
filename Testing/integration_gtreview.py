"""Integration test: drive the real GTReview widget inside a running Slicer.

Run it by hand::

    Slicer --no-splash \
        --additional-module-path <repo>/GTReview \
        --python-script <repo>/Testing/integration_gtreview.py

``smoke_headless.py`` exercises :class:`GTReviewLogic` against the real data
tree with no GUI.  This file covers the half that only exists once Slicer has a
main window: the widget, the segment editor, the slice views and their mouse
interactors.  On a synthetic case (numpy + SimpleITK, never patient data) it

1. loads a batch directory through the Dataset section,
2. selects a lesion in the table and checks the brush unlocks and the views
   jump onto it,
3. paints a stroke with the Paint effect by sending real mouse events to the
   Red slice view's interactor,
4. undoes the WHOLE stroke with a single Undo press and redoes it with a single
   Redo press (a stroke is one gesture, not one undo state per brush stamp),
5. drags the Sphere threshold effect with "2D: this slice only" off and then
   on, and checks the 2D result never leaves the slice it was drawn on,
6. deletes a lesion with its row's trash button, confirmation stubbed,
7. saves, then removes the review with the Delete review button, confirmation
   stubbed, and checks the file is gone and the case reopened from its
   original mask,
8. opens a second, multi-sequence case (t1, t1c, t2, flair) and checks the
   Sequences (axial) layout is chosen on its own, shows each sequence in its
   own axial view, links the views so a scroll in one moves the others, and
   gives way to a layout the reviewer picked; also that the lesion list bridges
   a one-voxel gap (dilation before connected components) and that label 3
   (Edema) is present, paintable and exported as 3.

The file is deliberately NOT named ``test_*``: the unit suite
(``PythonSlicer -m unittest discover -s Testing``) must not pick it up, because
everything here needs a running Slicer, and that suite is restricted to the
standard library plus numpy.

Prints one PASS/FAIL line per check plus a summary, and exits non-zero through
``slicer.util.exit`` when anything failed.
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np
import SimpleITK as sitk

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

    def test_06_sphere_threshold_2d(self):
        CHECKS.step("Sphere threshold: a ball, then a disc with 2D ticked")
        widget = self.widget
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

        with ConfirmStub(True) as confirm:
            widget.deleteReviewButton.click()
            pump()
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
