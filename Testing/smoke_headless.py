"""End-to-end headless smoke test for the GTReview extension.

Run it with::

    Slicer --no-main-window --no-splash \
        --python-script <repo>/Testing/smoke_headless.py

It implements the SPEC's "Definition of done" for the Slicer side of the tool:

1. discover the real cases in a batch directory,
2. pick a case with at least two lesions and at least two label values,
3. **copy that case into a temp directory** — the source data tree is opened
   read-only and is verified to be byte-identical afterwards,
4. load the copy, build the segmentation from the mask,
5. list the lesions,
6. delete the largest lesion (through the segment editor's undo stack),
7. change another lesion's label,
8. save ``<case_id>_reviewed_seg.nii.gz`` into the temp directory,
9. re-read it and assert
   * the geometry is identical to the source mask's -- size/origin/spacing
     bit-exact, and the direction cosines identical to a plain SimpleITK copy
     of the source (NIfTI's float32 qform is not bit-reversible for oblique
     volumes, so that copy is the format's floor, not ours),
   * the deleted lesion's voxels are gone,
   * the relabelled lesion carries the new value,
   * every stored label value is one of the ORIGINAL integers (no ordinals),
   * every other voxel is untouched.

Prints a PASS/FAIL summary and calls ``slicer.util.exit(0)`` / ``exit(1)``.
"""

import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import slicer

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = os.environ.get(
    "GTREVIEW_SMOKE_DATA_ROOT",
    "/home/melandur/Neosoma Inc. Dropbox/Neosoma Inc. R&D AI/01_Annotation/METS"
    "/04_Groundtruthed/01_Yale/batch_01",
)

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTING_DIR)
_MODULE_DIR = os.path.join(_REPO_ROOT, "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import GTReview as gtreview  # noqa: E402  (path set up above)
from GTReviewLib import dataset, lesions, maskio  # noqa: E402


# --------------------------------------------------------------------------- #
# tiny check harness (unittest is awkward inside a --python-script run)
# --------------------------------------------------------------------------- #
class Checks(object):
    def __init__(self):
        self.passed = []
        self.failed = []

    def check(self, ok, description, detail=""):
        if ok:
            self.passed.append(description)
            print("  [ OK ] {}".format(description))
        else:
            self.failed.append((description, detail))
            print("  [FAIL] {}{}".format(description, ("  --  " + detail) if detail else ""))
        return bool(ok)

    def step(self, message):
        print("\n== {}".format(message))


CHECKS = Checks()


def snapshot(directory):
    """{relative path: (size, mtime_ns, sha1)} for every file under *directory*."""
    import hashlib

    out = {}
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            try:
                stat = os.stat(path)
                digest = hashlib.sha1()
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                out[os.path.relpath(path, directory)] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    digest.hexdigest(),
                )
            except OSError as exc:  # pragma: no cover
                out[os.path.relpath(path, directory)] = ("unreadable", str(exc), "")
    return out


def makeSegmentEditorWidget(logic):
    """A real ``qMRMLSegmentEditorWidget``, wired to *logic*'s nodes."""
    import qSlicerSegmentationsModuleWidgetsPythonQt as segmentationWidgets

    editor = segmentationWidgets.qMRMLSegmentEditorWidget()
    editor.setMaximumNumberOfUndoStates(20)
    editor.setUndoEnabled(True)
    editorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    editor.setMRMLSegmentEditorNode(editorNode)
    editor.setMRMLScene(slicer.mrmlScene)
    editor.setSegmentationNode(logic.segmentationNode)
    editor.setSourceVolumeNode(logic.referenceVolumeNode)
    segmentIds = logic.segmentIds()
    if segmentIds:
        editor.setCurrentSegmentID(segmentIds[0])
    return editor


#: optional: force one particular case id (used to re-run the smoke test on a
#: harder case without editing this file)
ONLY_CASE_ID = os.environ.get("GTREVIEW_SMOKE_CASE_ID", "")


def pickCase(cases):
    """First case whose default mask has >=2 lesions and >=2 label values."""
    fallback = None
    if ONLY_CASE_ID:
        cases = [c for c in cases if c.case_id == ONLY_CASE_ID]
    for case in cases:
        path = case.default_mask_path()
        if not path or not os.path.isfile(path):
            continue
        try:
            array, geometry = maskio.read_mask(path)
        except Exception:  # noqa: BLE001
            continue
        _cmap, found = lesions.find_lesions(array, geometry.spacing)
        if len(found) < 2:
            continue
        labelValues = sorted(int(v) for v in np.unique(array) if v != 0)
        if len(labelValues) >= 2:
            return case, path, array, geometry, found, labelValues
        if fallback is None:
            fallback = (case, path, array, geometry, found, labelValues)
    return fallback


# --------------------------------------------------------------------------- #
# the test
# --------------------------------------------------------------------------- #
def run():
    tempRoot = None
    try:
        CHECKS.step("Discovering cases in the real data tree (read-only)")
        if not os.path.isdir(DATA_ROOT):
            CHECKS.check(False, "data root exists", DATA_ROOT)
            return
        cases = dataset.discover_cases(DATA_ROOT)
        print("  {} cases discovered under {}".format(len(cases), DATA_ROOT))
        if not CHECKS.check(bool(cases), "discover_cases found at least one case"):
            return

        picked = pickCase(cases)
        if not CHECKS.check(picked is not None, "found a case with >= 2 lesions"):
            return
        sourceCase, sourceMaskPath, sourceArray, sourceGeometry, sourceLesions, sourceLabels = picked
        print("  using case {} ({} lesions, labels {})".format(
            sourceCase.case_id, len(sourceLesions), sourceLabels))
        print("  mask: {}".format(sourceMaskPath))

        CHECKS.step("Copying the case into a temp directory (the data tree stays untouched)")
        dataSnapshotBefore = snapshot(sourceCase.directory)
        tempRoot = tempfile.mkdtemp(prefix="gtreview_smoke_")
        tempCaseDir = os.path.join(tempRoot, sourceCase.case_id)
        shutil.copytree(sourceCase.directory, tempCaseDir)
        print("  temp case dir: {}".format(tempCaseDir))

        case = dataset.parse_case_files(tempCaseDir)
        CHECKS.check(case.case_id == sourceCase.case_id, "case id survives the copy")
        CHECKS.check(
            os.path.abspath(case.reviewed_path).startswith(os.path.abspath(tempRoot)),
            "the reviewed-mask destination is inside the temp dir",
            case.reviewed_path,
        )
        CHECKS.check(not case.is_reviewed, "the copy has no reviewed mask yet")

        CHECKS.step("Loading the case and building the segmentation from the mask")
        logic = gtreview.GTReviewLogic()
        logic.loadCase(case)
        CHECKS.check(logic.segmentationNode is not None, "a vtkMRMLSegmentationNode was built")
        CHECKS.check(
            bool(logic.volumeNodes),
            "the image sequences were loaded ({})".format(sorted(logic.volumeNodes)),
        )
        CHECKS.check(
            logic.maskGeometry.is_compatible(sourceGeometry),
            "the loaded mask geometry matches the source mask",
        )
        CHECKS.check(
            set(sourceLabels) <= set(logic.labelValues()),
            "a segment for every original label value",
            "{} not within {}".format(sourceLabels, sorted(logic.labelValues())),
        )
        CHECKS.check(
            {1, 2} <= set(logic.labelValues()),
            "both review labels (1, 2) are always present so either can be painted",
            sorted(logic.labelValues()),
        )

        exported = logic.exportLabelmapArrayIJK()
        CHECKS.check(
            np.array_equal(exported.astype(np.int64), sourceArray.astype(np.int64)),
            "the segmentation round-trips the source mask voxel-exactly",
        )

        logic.editorWidget = makeSegmentEditorWidget(logic)
        CHECKS.check(logic.editorWidget is not None, "the segment editor widget is available")

        CHECKS.step("Listing lesions")
        componentMap, lesionList = logic.computeLesions()
        for lesion in lesionList[:10]:
            print("  #{:<3} label={} voxels={:<7} volume={:.1f} mm3 centre={}".format(
                lesion.index, lesion.label, lesion.voxel_count,
                lesion.volume_mm3, lesion.centroid_ijk))
        CHECKS.check(
            len(lesionList) == len(sourceLesions),
            "the lesion list matches the one computed straight from the file",
            "{} != {}".format(len(lesionList), len(sourceLesions)),
        )
        if not CHECKS.check(len(lesionList) >= 2, "at least two lesions to work with"):
            return
        CHECKS.check(
            all(sourceArray[l.centroid_ijk] != 0 for l in lesionList),
            "every lesion centre is a voxel inside its own lesion",
        )
        CHECKS.check(
            lesionList[0].voxel_count >= lesionList[-1].voxel_count,
            "the lesion list is sorted by voxel count descending",
        )

        largest = lesionList[0]
        largestMask = lesions.lesion_mask(componentMap, largest.index).copy()

        other = lesionList[1]
        otherMask = lesions.lesion_mask(componentMap, other.index).copy()
        # relabel to the OTHER review label -- the tool never introduces a third
        newLabel = 2 if int(other.label) == 1 else 1
        expectedLabels = sorted(set(sourceLabels) | {newLabel})

        CHECKS.step("Deleting the largest lesion (#{}, {} voxels, label {})".format(
            largest.index, largest.voxel_count, largest.label))
        logic.deleteLesionVoxels(largestMask)
        afterDelete = logic.exportLabelmapArrayIJK()
        CHECKS.check(
            int(afterDelete[largestMask].max(initial=0)) == 0,
            "in-memory: the deleted lesion is empty",
        )

        CHECKS.step("Undoing / redoing the delete")
        logic.editorWidget.undo()
        afterUndo = logic.exportLabelmapArrayIJK()
        CHECKS.check(
            np.array_equal(afterUndo.astype(np.int64), sourceArray.astype(np.int64)),
            "one undo puts the deleted lesion back (SPEC: edits go through the "
            "segment editor's undo stack)",
        )
        logic.editorWidget.redo()
        afterRedo = logic.exportLabelmapArrayIJK()
        CHECKS.check(
            np.array_equal(afterRedo.astype(np.int64), afterDelete.astype(np.int64)),
            "one redo removes it again",
        )

        CHECKS.step("Relabelling lesion #{} ({} voxels) from label {} to {}".format(
            other.index, other.voxel_count, other.label, newLabel))
        logic.changeLesionLabel(otherMask, newLabel)
        afterRelabel = logic.exportLabelmapArrayIJK()
        CHECKS.check(
            set(np.unique(afterRelabel[otherMask]).tolist()) == {newLabel},
            "in-memory: the relabelled lesion carries the new value",
        )

        CHECKS.step("Undoing / redoing the relabel")
        logic.editorWidget.undo()
        CHECKS.check(
            np.array_equal(
                logic.exportLabelmapArrayIJK()[otherMask].astype(np.int64),
                afterDelete[otherMask].astype(np.int64),
            ),
            "a relabel is a single undo step",
        )
        logic.editorWidget.redo()
        CHECKS.check(
            np.array_equal(
                logic.exportLabelmapArrayIJK().astype(np.int64),
                afterRelabel.astype(np.int64),
            ),
            "redo restores the relabelled mask",
        )

        CHECKS.step("Saving the reviewed mask into the temp directory")
        writtenPath = logic.saveReviewedMask()
        print("  wrote {}".format(writtenPath))
        CHECKS.check(
            os.path.abspath(writtenPath).startswith(os.path.abspath(tempRoot)),
            "the mask was written inside the temp dir",
            writtenPath,
        )
        CHECKS.check(
            os.path.basename(writtenPath) == case.case_id + "_reviewed_seg.nii.gz",
            "the output file follows the SPEC's naming contract",
            os.path.basename(writtenPath),
        )
        CHECKS.check(os.path.isfile(writtenPath), "the output file exists")

        CHECKS.step("Re-reading the saved mask")
        savedArray, savedGeometry = maskio.read_mask(writtenPath)

        CHECKS.check(
            tuple(savedGeometry.size) == tuple(sourceGeometry.size),
            "geometry: size identical to the source mask",
            "{} != {}".format(tuple(savedGeometry.size), tuple(sourceGeometry.size)),
        )
        for name in ("origin", "spacing"):
            saved = tuple(getattr(savedGeometry, name))
            source = tuple(getattr(sourceGeometry, name))
            CHECKS.check(
                saved == source,
                "geometry: {} bit-identical to the source mask".format(name),
                "{} != {}".format(saved, source),
            )

        # Direction cosines are the one field NIfTI cannot round-trip
        # bit-exactly: ITK re-derives them from the float32 qform quaternion, so
        # even ``sitk.ReadImage(x); sitk.WriteImage(x)`` moves them by ~1e-8 on
        # the obliquely-acquired Yale volumes.  Asserting ``==`` here would be
        # asserting something the file format cannot deliver.  Instead pin the
        # two things that ARE in our control:
        #   1. the saved direction equals a plain SimpleITK copy of the source
        #      mask -- i.e. GTReview adds no geometry error of its own, and in
        #      particular never round-trips geometry through Slicer's RAS
        #      conversion (which would show up at ~1e-3, not ~1e-8);
        #   2. the residual is negligible (maskio.DEFAULT_TOL = 1e-4).
        controlPath = os.path.join(tempRoot, "control_sitk_copy.nii.gz")
        try:
            import SimpleITK as sitk

            sitk.WriteImage(sitk.ReadImage(sourceMaskPath), controlPath, True)
            controlDirection = tuple(maskio.read_geometry(controlPath).direction)
        except Exception:  # noqa: BLE001 - fall back to the source direction
            traceback.print_exc()
            controlDirection = tuple(sourceGeometry.direction)

        savedDirection = tuple(savedGeometry.direction)
        sourceDirection = tuple(sourceGeometry.direction)
        residual = max(abs(a - b) for a, b in zip(savedDirection, sourceDirection))
        print("  direction residual vs source: {:.3e}  (NIfTI float32 qform floor)".format(
            residual))
        CHECKS.check(
            savedDirection == controlDirection,
            "geometry: direction identical to a plain SimpleITK copy of the source "
            "(GTReview adds no geometry error)",
            "{} != {}".format(savedDirection, controlDirection),
        )
        CHECKS.check(
            residual <= maskio.DEFAULT_TOL,
            "geometry: direction within {:g} of the source mask (residual {:.3e})".format(
                maskio.DEFAULT_TOL, residual),
        )
        CHECKS.check(
            savedGeometry.is_compatible(sourceGeometry),
            "geometry: the saved mask is compatible with the source mask",
            str(savedGeometry.mismatch_reason(sourceGeometry)),
        )

        CHECKS.check(
            savedArray.shape == sourceArray.shape,
            "the saved array has the source shape",
            "{} != {}".format(savedArray.shape, sourceArray.shape),
        )
        CHECKS.check(
            int(savedArray[largestMask].max(initial=0)) == 0
            and int((savedArray[largestMask] != 0).sum()) == 0,
            "the deleted lesion's {} voxels are gone from the saved file".format(
                int(largestMask.sum())),
        )
        savedRelabelValues = set(np.unique(savedArray[otherMask]).tolist())
        CHECKS.check(
            savedRelabelValues == {newLabel},
            "the relabelled lesion has value {} in the saved file".format(newLabel),
            "found {}".format(sorted(savedRelabelValues)),
        )

        storedLabels = sorted(int(v) for v in np.unique(savedArray) if v != 0)
        CHECKS.check(
            set(storedLabels).issubset(set(expectedLabels)),
            "the saved label values are the ORIGINAL integers (not segment ordinals)",
            "stored {} vs allowed {}".format(storedLabels, expectedLabels),
        )

        expectedArray = sourceArray.astype(np.int64).copy()
        expectedArray[largestMask] = 0
        expectedArray[otherMask] = newLabel
        CHECKS.check(
            np.array_equal(savedArray.astype(np.int64), expectedArray),
            "every other voxel of the mask is byte-for-byte unchanged",
            "{} differing voxels".format(
                int((savedArray.astype(np.int64) != expectedArray).sum())),
        )

        CHECKS.step("Verifying the source data directory was not touched")
        dataSnapshotAfter = snapshot(sourceCase.directory)
        CHECKS.check(
            dataSnapshotBefore == dataSnapshotAfter,
            "the source case directory is byte-identical (checksums + mtimes)",
            "changed: {}".format(sorted(
                set(dataSnapshotBefore.items()) ^ set(dataSnapshotAfter.items()))),
        )
        CHECKS.check(
            not os.path.exists(sourceCase.reviewed_path),
            "no reviewed mask was created in the real data tree",
            sourceCase.reviewed_path,
        )

        logic.unloadCase()

    except Exception:  # noqa: BLE001 - a crash is a failed smoke test, not a traceback dump
        traceback.print_exc()
        CHECKS.check(False, "the smoke test ran without raising", "see the traceback above")
    finally:
        if tempRoot and os.path.isdir(tempRoot):
            shutil.rmtree(tempRoot, ignore_errors=True)


def main():
    print("=" * 72)
    print("GTReview headless smoke test")
    print("  Slicer  : {}".format(slicer.app.applicationVersion))
    print("  module  : {}".format(gtreview.__file__))
    print("  data    : {}".format(DATA_ROOT))
    print("=" * 72)

    run()

    print("\n" + "=" * 72)
    print("SUMMARY: {} passed, {} failed".format(len(CHECKS.passed), len(CHECKS.failed)))
    for description, detail in CHECKS.failed:
        print("  FAILED: {}{}".format(description, ("  --  " + detail) if detail else ""))
    ok = not CHECKS.failed and bool(CHECKS.passed)
    print("RESULT: {}".format("PASS" if ok else "FAIL"))
    print("=" * 72)
    sys.stdout.flush()
    slicer.util.exit(0 if ok else 1)


main()
