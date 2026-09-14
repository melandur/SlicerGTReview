"""Measure what one undo state costs in memory, on a real case.

Run it with::

    Slicer --no-main-window --no-splash \
        --python-script <repo>/Testing/measure_undo_memory.py

Each undo state Slicer keeps is a copy of the segmentation's shared labelmap,
so the cost per state depends on the labelmap's extent, not on how many voxels
a stroke touched.  The extent starts as the bounding box of the lesions and
grows as painting reaches its edge, up to the whole reference volume.  This
script measures both ends:

1. the labelmap as loaded (lesion bounding box),
2. the labelmap grown to the full volume (a voxel painted in two opposite
   corners).

For each it takes ``STATES`` undo states, each after a one-voxel change so the
state is not a no-op, and reports the resident-memory growth per state next to
the labelmap's own byte size.  The case is copied to a temp directory first;
the data tree is never written to.

Environment: ``GTREVIEW_SMOKE_DATA_ROOT`` (batch directory), ``GTREVIEW_CASE_ID``
(optional, pick a specific case), ``GTREVIEW_UNDO_STATES`` (default 40).
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import slicer
import vtk

DATA_ROOT = os.environ.get(
    "GTREVIEW_SMOKE_DATA_ROOT",
    "/home/melandur/Neosoma Inc. Dropbox/Neosoma Inc. R&D AI/01_Annotation/METS"
    "/04_Groundtruthed/01_Yale/batch_01",
)
ONLY_CASE_ID = os.environ.get("GTREVIEW_CASE_ID", "")
STATES = int(os.environ.get("GTREVIEW_UNDO_STATES", "40"))

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
# An installed GTReview extension is imported by Slicer at startup; drop it so
# the import below measures the source tree, not whatever package is installed.
for _name in [m for m in sys.modules if m == "GTReview" or m.startswith("GTReviewLib")]:
    del sys.modules[_name]

import GTReview as gtreview  # noqa: E402
from GTReviewLib import dataset  # noqa: E402


def rssMiB():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def sharedLabelmap(segmentationNode, segmentId):
    name = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    return segmentationNode.GetSegmentation().GetSegment(segmentId).GetRepresentation(name)


def labelmapBytes(labelmap):
    dims = labelmap.GetDimensions()
    voxels = dims[0] * dims[1] * dims[2]
    return voxels * labelmap.GetScalarSize() * labelmap.GetNumberOfScalarComponents(), dims


def paintVoxel(logic, editor, segmentId, ijk, value):
    """Set one voxel through the segment's labelmap, as one undo state.

    Uses the same modifier path the effects use, so the labelmap extent grows
    the way it does under a brush when *ijk* lies outside it.
    """
    node = logic.segmentationNode
    ref = logic.referenceVolumeNode
    stamp = slicer.vtkOrientedImageData()
    stamp.SetExtent(ijk[0], ijk[0], ijk[1], ijk[1], ijk[2], ijk[2])
    stamp.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    stamp.GetPointData().GetScalars().Fill(1)
    ijkToRas = vtk.vtkMatrix4x4()
    ref.GetIJKToRASMatrix(ijkToRas)
    stamp.SetGeometryFromImageToWorldMatrix(ijkToRas)
    editor.saveStateForUndo()
    mode = (
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_MERGE_MASK
        if value
        else slicer.vtkSlicerSegmentationsModuleLogic.MODE_MERGE_MIN
    )
    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        stamp, node, segmentId, mode, stamp.GetExtent(), False, []
    )


def measure(label, logic, editor, segmentId, ijkA, ijkB):
    labelmap = sharedLabelmap(logic.segmentationNode, segmentId)
    nbytes, dims = labelmapBytes(labelmap)
    print("\n== {}".format(label))
    print("  labelmap extent {} = {:.2f} MiB per copy".format(dims, nbytes / 2.0**20))
    slicer.app.processEvents()
    before = rssMiB()
    samples = []
    for i in range(STATES):
        paintVoxel(logic, editor, segmentId, ijkA if i % 2 == 0 else ijkB, 1 if i % 4 < 2 else 0)
        samples.append(rssMiB())
    after = samples[-1]
    perState = (after - before) / float(STATES)
    steps = np.diff([before] + samples)
    print("  RSS before {:.1f} MiB, after {} states {:.1f} MiB".format(before, STATES, after))
    print("  per state: mean {:.2f} MiB, median step {:.2f} MiB, max step {:.2f} MiB".format(
        perState, float(np.median(steps)), float(steps.max())))
    return nbytes / 2.0**20, perState


def main():
    print("=" * 72)
    print("GTReview undo-state memory measurement  (Slicer {})".format(
        slicer.app.applicationVersion))
    print("  module {}".format(gtreview.__file__))
    cases = dataset.discover_cases(DATA_ROOT)
    if ONLY_CASE_ID:
        cases = [c for c in cases if c.case_id == ONLY_CASE_ID]
    if not cases:
        print("no cases under {}".format(DATA_ROOT))
        slicer.util.exit(1)
    source = cases[0]
    tempRoot = tempfile.mkdtemp(prefix="gtreview_undomem_")
    try:
        caseDir = os.path.join(tempRoot, source.case_id)
        shutil.copytree(source.directory, caseDir)
        case = dataset.parse_case_files(caseDir)
        print("  case {}".format(case.case_id))

        logic = gtreview.GTReviewLogic()
        logic.loadCase(case)
        import qSlicerSegmentationsModuleWidgetsPythonQt as segmentationWidgets

        editor = segmentationWidgets.qMRMLSegmentEditorWidget()
        editor.setMaximumNumberOfUndoStates(max(STATES, 10) * 4)
        editor.setUndoEnabled(True)
        editorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        editor.setMRMLSegmentEditorNode(editorNode)
        editor.setMRMLScene(slicer.mrmlScene)
        editor.setSegmentationNode(logic.segmentationNode)
        editor.setSourceVolumeNode(logic.referenceVolumeNode)
        logic.editorWidget = editor

        ref = logic.referenceVolumeNode
        refDims = ref.GetImageData().GetDimensions()
        refBytes = refDims[0] * refDims[1] * refDims[2]
        print("  reference volume {} = {:.2f} MiB as uint8".format(refDims, refBytes / 2.0**20))
        segmentId = logic.segmentIds()[0]

        # 1. natural extent: edit two voxels inside the current labelmap
        labelmap = sharedLabelmap(logic.segmentationNode, segmentId)
        ext = labelmap.GetExtent()
        inA = (ext[0], ext[2], ext[4])
        inB = (ext[1], ext[3], ext[5])
        naturalMiB, naturalPerState = measure(
            "labelmap at its loaded extent (lesion bounding box)",
            logic, editor, segmentId, inA, inB)

        # 1b. the module's own operations: do any of them grow the extent?
        from GTReviewLib import lesions as lesionsLib

        def extentMiB():
            nbytes, dims = labelmapBytes(sharedLabelmap(logic.segmentationNode, segmentId))
            return "{} = {:.2f} MiB".format(dims, nbytes / 2.0**20)

        print("\n== extent after the module's own operations")
        print("  loaded                      : {}".format(extentMiB()))
        componentMap, lesionList = logic.computeLesions()
        if len(lesionList) >= 2:
            mask = lesionsLib.lesion_mask(componentMap, lesionList[1].index).copy()
            logic.changeLesionLabel(mask, 2 if int(lesionList[1].label) == 1 else 1)
            print("  after relabel of a lesion   : {}".format(extentMiB()))
            mask = lesionsLib.lesion_mask(componentMap, lesionList[0].index).copy()
            logic.deleteLesionVoxels(mask)
            print("  after delete of a lesion    : {}".format(extentMiB()))
        # a brush stamp the way Paint applies one: a full-geometry modifier
        # with one voxel set, 10 voxels outside the current box
        effect = editor.effectByName("Paint")
        modifier = gtreview.modifierImageFromMaskIJK(
            ref, np.zeros(refDims, dtype=np.uint8))
        outside = tuple(min(e + 10, d - 1) for e, d in zip((ext[1], ext[3], ext[5]), refDims))
        arr = np.zeros(refDims, dtype=np.uint8)
        arr[outside] = 1
        modifier = gtreview.modifierImageFromMaskIJK(ref, arr)
        editor.saveStateForUndo()
        effect.modifySegmentByLabelmap(
            logic.segmentationNode, segmentId, modifier,
            slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd, True)
        print("  after a brush stamp 10 voxels outside the box: {}".format(extentMiB()))

        # 2. grow to the full volume: two opposite corners of the reference
        cornerA = (0, 0, 0)
        cornerB = (refDims[0] - 1, refDims[1] - 1, refDims[2] - 1)
        paintVoxel(logic, editor, segmentId, cornerA, 1)
        paintVoxel(logic, editor, segmentId, cornerB, 1)
        fullMiB, fullPerState = measure(
            "labelmap grown to the full reference volume",
            logic, editor, segmentId, cornerA, cornerB)

        widget = gtreview.GTReviewWidget
        undobudget = gtreview.undobudget
        budgetBytes = int(widget.UNDO_MEMORY_BUDGET_MB * undobudget.MIB)
        print("\n== what the undo budget keeps ({} MiB, {}..{} states)".format(
            widget.UNDO_MEMORY_BUDGET_MB, widget.MIN_UNDO_STATES, widget.MAX_UNDO_STATES))
        for label, perCopy in (("lesion-box extent", naturalMiB), ("full-volume extent", fullMiB)):
            states = undobudget.states_for_budget(
                int(perCopy * undobudget.MIB), budgetBytes,
                widget.MIN_UNDO_STATES, widget.MAX_UNDO_STATES)
            print("  {:<19}: {} states, {:.0f} MiB of history".format(
                label, states, states * perCopy))
        print("=" * 72)
        sys.stdout.flush()
    finally:
        shutil.rmtree(tempRoot, ignore_errors=True)
    slicer.util.exit(0)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:  # noqa: BLE001 - never leave a headless Slicer hanging
        traceback.print_exc()
        sys.stdout.flush()
        slicer.util.exit(1)
