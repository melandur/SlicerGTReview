"""GTReview — review and correct ground-truth segmentation masks in 3D Slicer.

Scripted loadable module (category *Segmentation*).  The GUI is built
programmatically with ``qt`` / ``ctk``; there is no ``.ui`` file.

Design invariants (see SPEC.md, they are binding):

* The mask lives in a ``vtkMRMLSegmentationNode``, one segment per label value.
  Every edit goes through the embedded ``qMRMLSegmentEditorWidget`` so that a
  single coherent undo/redo stack covers painting *and* the custom operations
  ("delete lesion", "change label").
* Array index order is ``[i, j, k]`` everywhere outside Slicer's own API.
  Slicer hands out ``[k, j, i]`` arrays, so this file transposes at that exact
  boundary (:func:`arrayFromVolumeIJK`, :func:`_orientedImageToArrayIJK`,
  :func:`_fillOrientedImageFromIJK`) and nowhere else.
* Only ``<case_dir>/<case_id>_reviewed_seg.nii.gz`` is ever written.
"""

import functools
import logging
import math
import os
import sys
import time

import ctk
import qt
import vtk
import slicer

import numpy as np
from vtk.util import numpy_support as vtk_np

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

try:  # Slicer >= 5.4
    from slicer.i18n import tr as _
    from slicer.i18n import translate
except ImportError:  # pragma: no cover - very old Slicer

    def _(text):
        return text

    def translate(context, text):  # noqa: D103
        return text


# --------------------------------------------------------------------------- #
# defensive import shim: works from the source tree and from an install tree
# --------------------------------------------------------------------------- #
# The directory holding THIS file goes to the front, not merely onto the path:
# a Slicer that also has GTReview installed as an extension already has that
# copy's directory on sys.path, and "GTReviewLib" would resolve there while
# GTReview.py itself is being loaded from the source tree -- two halves of two
# versions, which fails outright the moment one of them grows a new module.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.dirname(_THIS_DIR), _THIS_DIR):
    if not os.path.isdir(os.path.join(_candidate, "GTReviewLib")):
        continue
    while _candidate in sys.path:
        sys.path.remove(_candidate)
    sys.path.insert(0, _candidate)
del _candidate
# a GTReviewLib already imported from somewhere else must not stay cached
_name = _file = None
for _name in [_n for _n in list(sys.modules)
              if _n == "GTReviewLib" or _n.startswith("GTReviewLib.")]:
    _file = getattr(sys.modules[_name], "__file__", "") or ""
    if not os.path.abspath(_file).startswith(_THIS_DIR + os.sep):
        del sys.modules[_name]
del _name, _file

try:
    from GTReviewLib import dataset, lesions, maskio
except ImportError as exc:  # pragma: no cover - broken install
    raise ImportError(
        "GTReview could not import GTReviewLib from {} — is the extension "
        "installed completely? ({})".format(_THIS_DIR, exc)
    )


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
SEGMENT_LABEL_TAG = "GTReviewLabelValue"

#: where the batch-directory history lives, and how many entries are kept
DATASET_HISTORY_KEY = "GTReview/DatasetDirectoryHistory"
DATASET_HISTORY_LIMIT = 10

#: palette used for segments (label value -> RGB).  Label 1 is red and label 2
#: is green by request; the rest are a qualitative fallback for extra labels.
_SEGMENT_COLORS = (
    (0.90, 0.20, 0.20),   # 1 - red
    (0.25, 0.75, 0.30),   # 2 - green
    (0.25, 0.60, 0.90),   # 3 - blue
    (0.95, 0.75, 0.20),   # 4 - amber
    (0.70, 0.40, 0.85),   # 5 - purple
    (0.30, 0.85, 0.85),   # 6 - cyan
    (0.95, 0.55, 0.25),   # 7 - orange
)


def colorForLabelValue(value):
    """RGB for a label value; wraps around past the end of the palette."""
    return _SEGMENT_COLORS[(int(value) - 1) % len(_SEGMENT_COLORS)]


#: human-readable name per label value, for display only.  The integer value is
#: what lands in the saved NIfTI; these names never leave the GUI.
LABEL_NAMES = {
    1: "Necrosis and Cavity",
    2: "Enhancing Tumor",
}


SPHERE_THRESHOLD_EFFECT = "Sphere threshold"


def registerSphereThresholdEffect():
    """Register GTReviewLib's Sphere threshold effect with the editor factory (once)."""
    factory = slicer.qSlicerSegmentEditorEffectFactory.instance()
    try:
        if any(effect.name == SPHERE_THRESHOLD_EFFECT for effect in factory.registeredEffects()):
            return True
    except Exception:  # noqa: BLE001 - API differences
        logging.debug("GTReview: listing registered effects failed", exc_info=True)
    try:
        import qSlicerSegmentationsEditorEffectsPythonQt as effects

        source = os.path.join(
            os.path.dirname(os.path.abspath(lesions.__file__)),
            "SegmentEditorSphereThresholdEffect.py",
        )
        instance = effects.qSlicerSegmentEditorScriptedEffect(None)
        instance.setPythonSource(source.replace("\\", "/"))
        instance.self().register()
        return True
    except Exception:  # noqa: BLE001 - the brush still works without it
        logging.exception("GTReview: registering the Sphere threshold effect failed")
        return False


def nameForLabelValue(value):
    """Display name for a label value, e.g. ``1 - Necrosis and Cavity``.

    The value is kept as a prefix so a segment still maps obviously onto the
    Label column of the lesion table and onto the integers in the saved file.
    """
    value = int(value)
    name = LABEL_NAMES.get(value)
    return "{} - {}".format(value, name) if name else str(value)

LAYOUT_CHOICES = (
    ("Four-Up", "SlicerLayoutFourUpView", 3),
    ("1x1 Red (axial)", "SlicerLayoutOneUpRedSliceView", 6),
    ("1x1 Yellow (sagittal)", "SlicerLayoutOneUpYellowSliceView", 7),
    ("1x1 Green (coronal)", "SlicerLayoutOneUpGreenSliceView", 8),
    ("2x2 slices", "SlicerLayoutTwoOverTwoView", 27),
    ("Conventional", "SlicerLayoutConventionalView", 2),
)


def layoutId(attributeName, fallback):
    """Layout id by name, tolerant of Slicer renaming a constant."""
    return int(getattr(slicer.vtkMRMLLayoutNode, attributeName, fallback))


def arrayFromVolumeIJK(volumeNode):
    """``slicer.util.arrayFromVolume`` ([k,j,i]) transposed to ``[i,j,k]``."""
    return np.ascontiguousarray(slicer.util.arrayFromVolume(volumeNode).transpose(2, 1, 0))


def _orientedImageToArrayIJK(image):
    """Copy a ``vtkOrientedImageData`` into a C-contiguous ``[i,j,k]`` array."""
    dims = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        return np.zeros(dims, dtype=np.uint8)
    flat = vtk_np.vtk_to_numpy(scalars)
    array_kji = flat.reshape(dims[2], dims[1], dims[0])
    return np.ascontiguousarray(array_kji.transpose(2, 1, 0))


def _fillOrientedImageFromIJK(image, array_ijk):
    """Write an ``[i,j,k]`` array into an existing ``vtkOrientedImageData``."""
    dims = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    flat = vtk_np.vtk_to_numpy(scalars)
    view = flat.reshape(dims[2], dims[1], dims[0])
    source = np.asarray(array_ijk)
    if tuple(source.shape) != (dims[0], dims[1], dims[2]):
        raise ValueError(
            "modifier shape {} does not match the reference geometry {} "
            "(expected an [i, j, k] array)".format(tuple(source.shape), tuple(dims))
        )
    view[:] = source.transpose(2, 1, 0).astype(view.dtype, copy=False)
    scalars.Modified()
    image.Modified()
    return image


def ijkToRASMatrixFromGeometry(geometry):
    """4x4 IJK->RAS matrix for a :class:`maskio.MaskGeometry` (which is LPS).

    ``physical_LPS = origin + direction @ (spacing * ijk)`` and RAS flips the
    first two axes.  Building the reference volume node from this matrix makes
    Slicer's IJK indices identical to maskio's ``[i, j, k]`` indices, which is
    what keeps a saved mask voxel-exact.  (Slicer's own NIfTI reader is free to
    permute/flip the storage axes and compensate in the matrix — it does flip
    K on the real Yale data — so a volume loaded with ``loadLabelVolume`` must
    NEVER be used as the array-order reference.)
    """
    direction = np.asarray(geometry.direction_matrix(), dtype=float)
    spacing = np.asarray(geometry.spacing, dtype=float)
    origin = np.asarray(geometry.origin, dtype=float)
    lpsToRas = np.diag([-1.0, -1.0, 1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = lpsToRas.dot(direction * spacing)
    matrix[:3, 3] = lpsToRas.dot(origin)
    return matrix


def createLabelVolumeNode(array_ijk, geometry, name):
    """Add a ``vtkMRMLLabelMapVolumeNode`` holding ``array_ijk`` in *geometry*."""
    array = np.asarray(array_ijk)
    if array.min() < 0:
        logging.warning("GTReview: negative label values found, clamping them to 0")
        array = np.clip(array, 0, None)
    dtype = np.uint8 if int(array.max()) <= 255 else np.uint16
    array_kji = np.ascontiguousarray(array.transpose(2, 1, 0).astype(dtype, copy=False))
    node = slicer.util.addVolumeFromArray(
        array_kji,
        ijkToRASMatrixFromGeometry(geometry),
        name=name,
        nodeClassName="vtkMRMLLabelMapVolumeNode",
    )
    return node


def modifierImageFromMaskIJK(referenceVolumeNode, mask_ijk):
    """Build a modifier labelmap in *referenceVolumeNode*'s geometry.

    ``mask_ijk`` is a boolean/integer ``[i, j, k]`` array with the same shape as
    the reference volume.
    """
    image = slicer.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(
        referenceVolumeNode
    )
    image.UnRegister(None)  # the factory returns an extra reference
    # The factory may hand back an image that shares the volume node's scalar
    # buffer; reallocate so filling the modifier cannot clobber the node.
    image.AllocateScalars(image.GetScalarType(), 1)
    _fillOrientedImageFromIJK(image, np.asarray(mask_ijk) != 0)
    return image


def segmentationsLogicCall(methodName, *args):
    """Call a ``vtkSlicerSegmentationsModuleLogic`` method, static or not.

    Some members of that class are static and some are plain instance methods;
    which is which has changed between Slicer releases.  Try the class-level
    call first, fall back to the module logic instance.
    """
    method = getattr(slicer.vtkSlicerSegmentationsModuleLogic, methodName)
    try:
        return method(*args)
    except TypeError:
        logic = slicer.modules.segmentations.logic()
        return getattr(logic, methodName)(*args)


def focusedTextInput():
    """The focused text-entry widget, or None.

    Single-letter application shortcuts (n / p / s / j) would otherwise steal
    keystrokes from the directory box and the spin boxes.
    """
    widget = qt.QApplication.focusWidget()
    if widget is None:
        return None
    for widgetClass in (qt.QLineEdit, qt.QAbstractSpinBox, qt.QTextEdit, qt.QPlainTextEdit):
        try:
            if isinstance(widget, widgetClass):
                return widget
        except TypeError:  # pragma: no cover - binding quirk
            continue
    # ctkPathLineEdit is an editable combo box: its focus widget can be the
    # combo box itself, which forwards key events to its internal line edit.
    try:
        if isinstance(widget, qt.QComboBox) and bool(getattr(widget, "editable", False)):
            return widget
    except TypeError:  # pragma: no cover - binding quirk
        pass
    return None


class BusyCursor(object):
    """Context manager showing the wait cursor around a slow operation."""

    def __init__(self, statusMessage=None):
        self.statusMessage = statusMessage

    def __enter__(self):
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        if self.statusMessage:
            slicer.util.showStatusMessage(self.statusMessage)
        slicer.app.processEvents()
        return self

    def __exit__(self, excType, excValue, traceback):
        qt.QApplication.restoreOverrideCursor()
        if self.statusMessage:
            slicer.util.showStatusMessage("")
        return False


def guarded(what):
    """Decorate a slot so that an exception becomes an error popup, not a crash."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - deliberate catch-all for slots
                logging.exception("GTReview: %s failed", what)
                slicer.util.errorDisplay(
                    "{} failed.\n\n{}: {}".format(what, type(exc).__name__, exc),
                    windowTitle="GTReview",
                )
                return None

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# module
# --------------------------------------------------------------------------- #
class GTReview(ScriptedLoadableModule):
    """Module metadata."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("GT Review")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["Neosoma Inc. (https://www.neosomainc.com)"]
        self.parent.helpText = _(
            "Review and correct ground-truth segmentation masks lesion by lesion.\n"
            "Pick a batch directory, step through the cases, inspect the connected "
            "components of the mask, correct them with the segment editor and save a "
            "<case_id>_reviewed_seg.nii.gz next to the source data. The original files "
            "are never modified."
        )
        self.parent.helpText += parent.defaultDocumentationLink
        self.parent.acknowledgementText = _(
            "Developed by <a href=\"https://www.neosomainc.com\">Neosoma Inc.</a>"
        )


# --------------------------------------------------------------------------- #
# logic
# --------------------------------------------------------------------------- #
class GTReviewLogic(ScriptedLoadableModuleLogic):
    """All Slicer-side data handling: load a case, edit it, export it, save it.

    The logic owns the MRML nodes of the current case and knows the *source
    mask geometry*, which is what a saved reviewed mask must reproduce exactly.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.case = None
        self.maskPath = None
        self.maskGeometry = None          # maskio.MaskGeometry of the source mask
        self.geometryWarning = None       # str, set when image/mask geometry disagree
        self.volumeNodes = {}             # image key -> vtkMRMLScalarVolumeNode
        self.maskVolumeNode = None        # vtkMRMLLabelMapVolumeNode of the source mask
        self.referenceVolumeNode = None   # node whose geometry the mask lives in
        self.segmentationNode = None
        self.editorWidget = None          # set by the widget; the undo-aware vehicle
        self._ownedNodes = []             # nodes to remove from the scene on unload

    # ---------------------------------------------------------------- dataset
    @staticmethod
    def discoverCases(root):
        return dataset.discover_cases(root)

    # ------------------------------------------------------------- node hygiene
    def _track(self, node):
        if node is not None:
            self._ownedNodes.append(node)
        return node

    def unloadCase(self):
        """Remove every node this logic created; safe to call repeatedly."""
        editor = self.editorWidget
        if editor is not None:
            try:
                editor.setActiveEffect(None)
                editor.setSourceVolumeNode(None)
                editor.setSegmentationNode(None)
            except Exception:  # noqa: BLE001 - widget may be half torn down
                logging.debug("GTReview: detaching the editor failed", exc_info=True)
        for node in self._ownedNodes:
            try:
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            except Exception:  # noqa: BLE001
                logging.debug("GTReview: removing a node failed", exc_info=True)
        self._ownedNodes = []
        self.volumeNodes = {}
        self.maskVolumeNode = None
        self.referenceVolumeNode = None
        self.segmentationNode = None
        self.case = None
        self.maskPath = None
        self.maskGeometry = None
        self.geometryWarning = None

    # ------------------------------------------------------------------ loading
    def loadCase(self, case, maskPath=None):
        """Load every image of *case* plus one mask into the scene.

        ``maskPath`` defaults to ``case.default_mask_path()``.  A case with no
        mask at all degrades to an empty segmentation on the first image.
        Returns the loaded ``vtkMRMLSegmentationNode``.
        """
        if case is None:
            raise ValueError("no case to load")
        self.unloadCase()
        self.case = case

        for key in sorted(case.images, key=lambda k: (dataset.natural_key(k), k)):
            path = case.images[key]
            try:
                node = slicer.util.loadVolume(
                    path,
                    properties={
                        "name": "{}_{}".format(case.case_id, key),
                        "singleFile": True,
                        "show": False,
                    },
                )
            except Exception:  # noqa: BLE001 - one bad file must not kill the case
                logging.exception("GTReview: could not load image %s", path)
                continue
            # the reader uniquifies names against a counter the scene keeps even
            # after the old node is gone, so a reloaded case would show up as
            # "..._t1c_3" in the view corners; name it ourselves instead
            node.SetName("{}_{}".format(case.case_id, key))
            node.SetAttribute("GTReviewRole", "image")
            displayNode = node.GetDisplayNode()
            if displayNode is None:
                node.CreateDefaultDisplayNodes()
                displayNode = node.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetAutoWindowLevel(True)
            self.volumeNodes[key] = self._track(node)

        if maskPath is None:
            maskPath = case.default_mask_path()

        # The reference volume node is ALWAYS built by us from the maskio
        # geometry, so Slicer's IJK order == maskio's [i, j, k] order.
        if maskPath and os.path.isfile(maskPath):
            self.maskPath = maskPath
            maskArray, self.maskGeometry = maskio.read_mask(maskPath)
        else:
            self.maskPath = None
            referencePath = self._pathOfFirstImage()
            if referencePath is None:
                raise RuntimeError(
                    "case {} has neither an image nor a mask".format(case.case_id)
                )
            self.maskGeometry = maskio.read_geometry(referencePath)
            maskArray = np.zeros(
                tuple(int(n) for n in self.maskGeometry.size), dtype=np.uint8
            )
        maskNode = createLabelVolumeNode(
            maskArray, self.maskGeometry, "{}_maskref".format(case.case_id)
        )
        maskNode.SetAttribute("GTReviewRole", "maskReference")
        displayNode = maskNode.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetVisibility(False)
        self.maskVolumeNode = self._track(maskNode)
        self.referenceVolumeNode = maskNode

        self.geometryWarning = self._checkGeometries()
        self.segmentationNode = self._buildSegmentationNode()
        return self.segmentationNode

    def _pathOfFirstImage(self):
        for key in sorted(self.case.images, key=lambda k: (dataset.natural_key(k), k)):
            return self.case.images[key]
        return None

    def _checkGeometries(self):
        """Compare the mask geometry with every image; return a warning or None."""
        if self.maskGeometry is None or not self.case:
            return None
        problems = []
        for key in sorted(self.case.images, key=lambda k: (dataset.natural_key(k), k)):
            if key not in self.volumeNodes:
                continue
            try:
                imageGeometry = maskio.read_geometry(self.case.images[key])
            except Exception:  # noqa: BLE001
                logging.debug("GTReview: geometry probe failed", exc_info=True)
                continue
            reason = self.maskGeometry.mismatch_reason(imageGeometry)
            if reason:
                problems.append("{}: {}".format(key, reason))
        if not problems:
            return None
        return (
            "The mask and some image sequences do not share the same geometry. "
            "Editing and saving use the MASK geometry.\n  " + "\n  ".join(problems)
        )

    def _buildSegmentationNode(self):
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "{}_review".format(self.case.case_id)
        )
        self._track(node)
        node.CreateDefaultDisplayNodes()
        node.SetReferenceImageGeometryParameterFromVolumeNode(self.referenceVolumeNode)
        if self.maskVolumeNode is not None and self.maskPath:
            ok = segmentationsLogicCall(
                "ImportLabelmapToSegmentationNode", self.maskVolumeNode, node
            )
            if not ok:
                raise RuntimeError("importing {} failed".format(self.maskPath))
        self._normalizeSegments(node)
        self._ensureReviewLabels(node)
        displayNode = node.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetAllSegmentsVisibility(True)
            displayNode.SetVisibility2DFill(True)
            displayNode.SetVisibility2DOutline(True)
            displayNode.SetOpacity2DFill(0.5)
            # Show the surface in the 3D view without waiting for a click, but
            # keep the slice views on the binary labelmap: letting Slicer pick
            # the closed surface for 2D means every slice is re-cut with
            # vtkCutter, which is both slower and prone to seam artifacts.
            displayNode.SetPreferredDisplayRepresentationName2D(
                slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
            )
            displayNode.SetPreferredDisplayRepresentationName3D(
                slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            )
            displayNode.SetVisibility3D(True)
        try:
            # no smoothing: the 3D surface follows the voxels, like the 2D fill
            node.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
            node.CreateClosedSurfaceRepresentation()
        except Exception:  # noqa: BLE001 - never block loading over a preview
            logging.warning("GTReview: building the 3D surface failed", exc_info=True)
        return node

    def _ensureReviewLabels(self, segmentationNode):
        """Guarantee one segment per review label, listed in label order.

        12/50 batch_01 cases carry only label 2 and 17 pred_segs are empty,
        yet the reviewer must be able to paint either label.  An empty
        segment contributes nothing to the export (the self-test asserts it),
        so the saved file is unchanged by this.  Runs before the segmentation
        is observed, so it never marks the case as edited.
        """
        for value in sorted(LABEL_NAMES):
            if self.segmentIdForLabelValue(value, segmentationNode) is None:
                self.addLabel(value, segmentationNode=segmentationNode)
        # label-2-only masks would otherwise list "2" above "1"
        try:
            segmentation = segmentationNode.GetSegmentation()
            ordered = sorted(self.segmentLabelValues(segmentationNode), key=lambda p: p[1])
            for position, (segmentId, _value) in enumerate(ordered):
                segmentation.SetSegmentIndex(segmentId, position)
        except Exception:  # noqa: BLE001 - cosmetic only
            logging.debug("GTReview: reordering segments failed", exc_info=True)

    def _normalizeSegments(self, segmentationNode):
        """Stamp an explicit label value on every segment (name + tag + color)."""
        for segmentId, value in self.segmentLabelValues(segmentationNode):
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            segment.SetTag(SEGMENT_LABEL_TAG, str(value))
            try:
                segment.SetLabelValue(int(value))
            except Exception:  # noqa: BLE001 - older API
                logging.debug("GTReview: SetLabelValue unavailable", exc_info=True)
            name = nameForLabelValue(value)
            if segment.GetName() != name:
                segment.SetName(name)
            segment.SetColor(*colorForLabelValue(value))

    # ------------------------------------------------------- segment <-> label
    def segmentIds(self, segmentationNode=None):
        segmentationNode = segmentationNode or self.segmentationNode
        if segmentationNode is None:
            return []
        ids = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(ids)
        return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]

    @staticmethod
    def _segmentTag(segment, key):
        """``vtkSegment::GetTag`` with its C++ output argument, or None."""
        try:
            if not segment.HasTag(key):
                return None
            holder = vtk.reference("")
            if not segment.GetTag(key, holder):
                return None
            return str(holder.get() if hasattr(holder, "get") else holder)
        except Exception:  # noqa: BLE001 - tag API differences
            logging.debug("GTReview: reading segment tag failed", exc_info=True)
            return None

    @classmethod
    def _segmentLabelHint(cls, segment):
        """Best guess of a segment's integer label value, or None."""
        for candidate in (cls._segmentTag(segment, SEGMENT_LABEL_TAG), segment.GetName()):
            if candidate:
                try:
                    value = int(str(candidate).strip())
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
        try:
            value = int(segment.GetLabelValue())
        except Exception:  # noqa: BLE001
            return None
        return value if value > 0 else None

    def segmentLabelValues(self, segmentationNode=None):
        """``[(segmentId, labelValue), ...]`` — unique, positive, deterministic.

        This mapping is what makes the exported mask carry the ORIGINAL integer
        label values instead of segment ordinals.
        """
        segmentationNode = segmentationNode or self.segmentationNode
        if segmentationNode is None:
            return []
        segmentation = segmentationNode.GetSegmentation()
        pairs = []
        used = set()
        pending = []
        for segmentId in self.segmentIds(segmentationNode):
            segment = segmentation.GetSegment(segmentId)
            value = self._segmentLabelHint(segment) if segment else None
            if value is None or value in used:
                pending.append(segmentId)
                pairs.append((segmentId, None))
                continue
            used.add(value)
            pairs.append((segmentId, value))
        if pending:
            nextValue = 1
            resolved = []
            for segmentId, value in pairs:
                if value is None:
                    while nextValue in used:
                        nextValue += 1
                    value = nextValue
                    used.add(value)
                resolved.append((segmentId, value))
            pairs = resolved
        return pairs

    def labelValues(self, segmentationNode=None):
        return [value for _sid, value in self.segmentLabelValues(segmentationNode)]

    def segmentIdForLabelValue(self, value, segmentationNode=None):
        for segmentId, labelValue in self.segmentLabelValues(segmentationNode):
            if int(labelValue) == int(value):
                return segmentId
        return None

    def labelValueForSegmentId(self, segmentId, segmentationNode=None):
        for sid, labelValue in self.segmentLabelValues(segmentationNode):
            if sid == segmentId:
                return labelValue
        return None

    def nextFreeLabelValue(self, segmentationNode=None):
        used = set(self.labelValues(segmentationNode))
        value = 1
        while value in used:
            value += 1
        return value

    def addLabel(self, value=None, segmentationNode=None):
        """Create a new, empty segment carrying the next free integer label."""
        segmentationNode = segmentationNode or self.segmentationNode
        if segmentationNode is None:
            raise RuntimeError("no segmentation loaded")
        if value is None:
            value = self.nextFreeLabelValue(segmentationNode)
        value = int(value)
        if self.segmentIdForLabelValue(value, segmentationNode) is not None:
            raise ValueError("label value {} already exists".format(value))
        color = colorForLabelValue(value)
        segmentId = segmentationNode.GetSegmentation().AddEmptySegment(
            "", nameForLabelValue(value), color
        )
        segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
        segment.SetTag(SEGMENT_LABEL_TAG, str(value))
        try:
            segment.SetLabelValue(value)
        except Exception:  # noqa: BLE001
            logging.debug("GTReview: SetLabelValue unavailable", exc_info=True)
        return segmentId

    # ------------------------------------------------------------------ export
    def exportLabelmapArrayIJK(self):
        """Current segmentation as an ``[i, j, k]`` array of ORIGINAL labels."""
        if self.maskGeometry is None:
            raise RuntimeError("no case loaded")
        shape = tuple(int(n) for n in self.maskGeometry.size)
        pairs = self.segmentLabelValues()
        if not pairs:
            return np.zeros(shape, dtype=np.uint8)

        segmentIds = vtk.vtkStringArray()
        labelValues = vtk.vtkIntArray()
        for segmentId, value in pairs:
            segmentIds.InsertNextValue(segmentId)
            labelValues.InsertNextValue(int(value))

        merged = slicer.vtkOrientedImageData()
        segmentationsLogicCall(
            "GenerateMergedLabelmapInReferenceGeometry",
            self.segmentationNode,
            self.referenceVolumeNode,
            segmentIds,
            slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
            merged,
            labelValues,
        )
        array = _orientedImageToArrayIJK(merged)
        if tuple(array.shape) != shape:
            raise RuntimeError(
                "exported labelmap has shape {} but the source mask geometry is {}"
                .format(tuple(array.shape), shape)
            )
        return array

    # ----------------------------------------------------------------- lesions
    def computeLesions(self, minVoxels=1, connectivity=26):
        """``(component_map, [Lesion, ...])`` over the CURRENT segmentation."""
        array = self.exportLabelmapArrayIJK()
        spacing = tuple(self.maskGeometry.spacing)
        return lesions.find_lesions(
            array, spacing, connectivity=connectivity, min_voxels=int(minVoxels)
        )

    # ------------------------------------------------------- undo-aware edits
    def _effect(self):
        """A wired segment-editor effect used purely as an apply vehicle.

        ``effectByName`` returns an effect whose ``parameterSetNode`` is already
        the editor's, without activating it — so the user's current effect and
        the GUI are left alone (and it does not need slice views).
        """
        editor = self.editorWidget
        if editor is None:
            raise RuntimeError("the segment editor widget is not available")
        for name in ("Threshold", "Logical operators", "Margin", "Smoothing"):
            effect = editor.effectByName(name)
            if effect is not None:
                return effect
        raise RuntimeError("no segment editor effect available to apply the edit")

    def deleteLesionVoxels(self, lesionMaskIJK):
        """Remove the lesion's voxels from every segment, as ONE undo step."""
        if self.segmentationNode is None:
            raise RuntimeError("no segmentation loaded")
        editor = self.editorWidget
        effect = self._effect()
        modifier = modifierImageFromMaskIJK(self.referenceVolumeNode, lesionMaskIJK)
        modes = slicer.qSlicerSegmentEditorAbstractEffect
        editor.saveStateForUndo()
        for segmentId in self.segmentIds():
            effect.modifySegmentByLabelmap(
                self.segmentationNode,
                segmentId,
                modifier,
                modes.ModificationModeRemove,
                True,  # bypassMasking
            )

    def labelMaskIJK(self, labelValue):
        """Boolean ``[i, j, k]`` mask of every voxel currently at *labelValue*."""
        return self.exportLabelmapArrayIJK() == int(labelValue)

    def deleteLabelVoxels(self, labelValue):
        """Remove every voxel of *labelValue*, as ONE undo step.

        The label's segment is kept (now empty) so it can still be painted.
        Returns the number of voxels removed.
        """
        mask = self.labelMaskIJK(labelValue)
        removed = int(mask.sum())
        if removed:
            self.deleteLesionVoxels(mask)
        return removed

    def changeLesionLabel(self, lesionMaskIJK, targetLabelValue):
        """Move the lesion's voxels into the segment of *targetLabelValue*.

        Add + remove happen after a single ``saveStateForUndo``, so one undo
        reverts the whole relabel.
        """
        if self.segmentationNode is None:
            raise RuntimeError("no segmentation loaded")
        targetLabelValue = int(targetLabelValue)
        targetId = self.segmentIdForLabelValue(targetLabelValue)
        if targetId is None:
            targetId = self.addLabel(targetLabelValue)
        editor = self.editorWidget
        effect = self._effect()
        modifier = modifierImageFromMaskIJK(self.referenceVolumeNode, lesionMaskIJK)
        modes = slicer.qSlicerSegmentEditorAbstractEffect
        editor.saveStateForUndo()
        effect.modifySegmentByLabelmap(
            self.segmentationNode, targetId, modifier, modes.ModificationModeAdd, True
        )
        for segmentId in self.segmentIds():
            if segmentId == targetId:
                continue
            effect.modifySegmentByLabelmap(
                self.segmentationNode,
                segmentId,
                modifier,
                modes.ModificationModeRemove,
                True,
            )
        return targetId

    # -------------------------------------------------------------------- save
    def reviewedPath(self):
        return self.case.reviewed_path if self.case else None

    def saveReviewedMask(self):
        """Write ``<case_dir>/<case_id>_reviewed_seg.nii.gz``; return the path.

        No other path is ever written — the destination comes from the Case,
        never from the caller.
        """
        if self.case is None:
            raise RuntimeError("no case loaded")
        path = self.case.reviewed_path
        if not path:
            raise RuntimeError("case {} has no reviewed path".format(self.case.case_id))
        array = self.exportLabelmapArrayIJK()
        maskio.write_mask(path, array, self.maskGeometry, dtype=np.uint8)
        return path

    # ------------------------------------------------------------------ geometry
    def centroidToRAS(self, centroidIJK):
        """Voxel index -> world RAS, through the full IJK->RAS matrix."""
        node = self.referenceVolumeNode
        if node is None:
            raise RuntimeError("no reference volume")
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
        i, j, k = (float(v) for v in centroidIJK)
        ras = matrix.MultiplyPoint([i, j, k, 1.0])[:3]
        transformNode = node.GetParentTransformNode()
        if transformNode is not None:
            general = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                transformNode, None, general
            )
            ras = general.TransformPoint(ras)
        return [float(v) for v in ras]


# --------------------------------------------------------------------------- #
# widget
# --------------------------------------------------------------------------- #
class GTReviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """The GUI."""

    LESION_REFRESH_DEBOUNCE_MS = 1500
    #: the only effects offered in the brush grid
    EDITOR_EFFECTS = ("Paint", "Erase", SPHERE_THRESHOLD_EFFECT)
    #: brush is always absolute; lesions here are tiny (median 26 voxels)
    BRUSH_MM = 3.0
    BRUSH_MIN_MM = 1.0
    BRUSH_MAX_MM = 20.0
    BRUSH_STEP_MM = 1.0
    #: main-window actions whose shortcuts collide with ours while the module
    #: is open (Qt fires NEITHER owner of an ambiguous shortcut)
    MUTED_MAIN_WINDOW_ACTIONS = ("FileSaveSceneAction", "EditUndoAction", "EditRedoAction")
    #: a / d change the mask fill opacity by this much
    MASK_OPACITY_STEP = 0.1
    #: the lesion table grows with the list, then scrolls
    LESION_TABLE_MAX_ROWS = 10
    #: lesion table columns.  "Label" is gone: every lesion in the list already
    #: carries its colour in the views, and the column never varied enough to
    #: earn the width.
    LESION_COLUMN_NUMBER = 0
    LESION_COLUMN_VOXELS = 1
    LESION_COLUMN_VOLUME = 2
    LESION_COLUMN_DONE = 3
    #: last column: a per-row delete button, not data
    LESION_DELETE_COLUMN = 4
    #: "Paint over" sentinels; any other item data is a label value
    PAINT_OVER_ALL = -1
    PAINT_OVER_BACKGROUND = -2
    #: "Active label" sentinel; 0 is the background value in the saved NIfTI
    ACTIVE_LABEL_BACKGROUND = 0
    #: how many times a jumped-to lesion blinks, and how fast
    FLASH_BLINKS = 3
    FLASH_INTERVAL_MS = 170

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.editor = None
        self.segmentEditorNode = None
        self.effectFactorySingleton = None
        # applyViewLayers can run before the Display section is built
        self.windowLevelWidget = None
        self.lesionControls = None
        self.alignSlicesCheckBox = None
        self.activeLabelComboBox = None
        self.paintOverComboBox = None
        self.newLesionButton = None
        self.deleteReviewButton = None
        self._panelNarrowed = False
        self._effectOptionsBox = None
        # mask fingerprints taken at the start of each paint stroke, so one
        # Undo press can step back over the whole stroke
        self._strokeStarts = []
        self._redoTargets = []
        self._strokeObservers = []

        self.cases = []
        self.filteredCases = []
        self.currentCaseIndex = -1
        self.componentMap = None
        self.lesionList = []
        self.selectedLesionIndex = None
        self.selectedLesionSeed = None
        self.lesionsStale = True
        #: case directory -> {seed voxel of each lesion marked done}.  Seeds
        #: rather than indices, because indices are reassigned on every
        #: recount; kept in memory only, never written to disk.
        self.reviewedSeeds = {}
        self.unsavedChanges = False

        self._shortcuts = []
        self._shortcutHandlers = []
        self._updatingGui = False
        self._refreshTimer = None
        self._flashTimer = None
        self._flashNode = None
        self._flashesLeft = 0
        #: new-lesion mode: {"label": int, "before": {seed: voxel_count}} or None
        self._newLesion = None
        self._refreshing = False
        self._enforcingEditGate = False
        self._observedEditorNode = None
        self._mutedActions = []
        self._brushInitialised = False

    # ------------------------------------------------------------------- setup
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = GTReviewLogic()

        self._refreshTimer = qt.QTimer()
        self._refreshTimer.setSingleShot(True)
        self._refreshTimer.connect("timeout()", self.onDebouncedRefresh)

        self._flashTimer = qt.QTimer()
        self._flashTimer.connect("timeout()", self._onFlashTick)

        registerSphereThresholdEffect()
        self._buildDatasetSection()
        self._buildDisplaySection()
        self._buildLesionSection()
        self._buildEditingSection()
        self.layout.addStretch(1)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self._populateDatasetHistory()
        self._updateCaseControls()
        self._updateEditingControls()

    # -- dataset ------------------------------------------------------------- #
    def _buildDatasetSection(self):
        section = ctk.ctkCollapsibleButton()
        section.text = "Dataset"
        self._accentSection(section, self.ACCENT_DATASET, "Dataset")
        self.layout.addWidget(section)
        form = self._tighten(qt.QFormLayout(section))

        row = qt.QHBoxLayout()
        row.setSpacing(4)
        self.datasetPathEdit = ctk.ctkPathLineEdit()
        self.datasetPathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.datasetPathEdit.settingKey = "GTReview/DatasetDirectory"
        self.datasetPathEdit.toolTip = (
            "A batch directory holding one sub-directory per case.\n"
            "Type or pick a previous path here and press Enter to load it."
        )
        # One button, not two: ctkPathLineEdit's own "..." browses but does not
        # load, so browsing then loading took two clicks.  Hide it and let the
        # single button below browse *and* load.
        try:
            self.datasetPathEdit.showBrowseButton = False
        except Exception:  # noqa: BLE001 - older CTK without the property
            logging.debug("GTReview: showBrowseButton unavailable", exc_info=True)
        # ctkPathLineEdit has no returnPressed of its own; reach its inner
        # QLineEdit so Enter loads a typed or history-picked path directly.
        lineEdit = self.datasetPathEdit.findChild(qt.QLineEdit)
        if lineEdit is not None:
            lineEdit.connect("returnPressed()", self.onLoadDataset)
        else:  # pragma: no cover - unexpected CTK internals
            logging.debug("GTReview: no inner QLineEdit on the path edit")
        row.addWidget(self.datasetPathEdit)
        self.loadDatasetButton = qt.QPushButton("Browse && load...")
        self.loadDatasetButton.toolTip = (
            "Pick a batch directory and discover every case below it.\n"
            "Discovery is one level deep, so choose the batch folder itself."
        )
        self.loadDatasetButton.connect("clicked()", self.onBrowseAndLoad)
        row.addWidget(self.loadDatasetButton)
        form.addRow("Batch directory:", row)

        caseRow = qt.QHBoxLayout()
        caseRow.setSpacing(4)
        self.previousCaseButton = qt.QPushButton("< Prev")
        self.previousCaseButton.toolTip = "Previous case (p)"
        self.previousCaseButton.connect("clicked()", self.onPreviousCase)
        caseRow.addWidget(self.previousCaseButton)
        self.caseComboBox = qt.QComboBox()
        self.caseComboBox.toolTip = "Cases found in the batch directory."
        self.caseComboBox.connect("currentIndexChanged(int)", self.onCaseSelected)
        caseRow.addWidget(self.caseComboBox, 1)
        self.nextCaseButton = qt.QPushButton("Next >")
        self.nextCaseButton.toolTip = "Next case (n)"
        self.nextCaseButton.connect("clicked()", self.onNextCase)
        caseRow.addWidget(self.nextCaseButton)
        self.caseProgressLabel = qt.QLabel("0 / 0")
        caseRow.addWidget(self.caseProgressLabel)
        form.addRow("Case:", caseRow)

        self.skipReviewedCheckBox = qt.QCheckBox("Skip already reviewed cases")
        self.skipReviewedCheckBox.connect("toggled(bool)", self.onSkipReviewedToggled)
        form.addRow("", self.skipReviewedCheckBox)

        maskRow = qt.QHBoxLayout()
        maskRow.setSpacing(4)
        self.maskSourceComboBox = qt.QComboBox()
        self.maskSourceComboBox.toolTip = "Which mask of this case is being reviewed."
        self.maskSourceComboBox.connect("activated(int)", self.onMaskSourceChanged)
        maskRow.addWidget(self.maskSourceComboBox, 1)
        self.deleteReviewButton = qt.QPushButton("Delete review")
        self.deleteReviewButton.toolTip = (
            "Delete this case's <case_id>_reviewed_seg.nii.gz and start the case "
            "over from its original mask.\n"
            "This one really does erase a file from disk -- Ctrl+Z does not "
            "reach it."
        )
        self.deleteReviewButton.connect("clicked()", self.onDeleteReview)
        maskRow.addWidget(self.deleteReviewButton)
        form.addRow("Mask source:", maskRow)

        self.caseStatusLabel = qt.QLabel("No dataset loaded.")
        self.caseStatusLabel.wordWrap = False
        self.caseStatusLabel.textFormat = qt.Qt.RichText
        self.caseStatusLabel.textInteractionFlags = qt.Qt.TextSelectableByMouse
        form.addRow("", self.caseStatusLabel)

    # -- display ------------------------------------------------------------- #
    def _buildDisplaySection(self):
        section = ctk.ctkCollapsibleButton()
        section.text = "Display"
        self._accentSection(section, self.ACCENT_DISPLAY, "Display")
        self.layout.addWidget(section)
        form = self._tighten(qt.QFormLayout(section))

        self.displayForm = form

        self.backgroundComboBox = qt.QComboBox()
        self.backgroundComboBox.toolTip = (
            "The image sequence shown underneath the segmentation.\n"
            "This is also what the Segment Editor's intensity effects "
            "(Threshold, Level tracing, ...) read."
        )
        self.backgroundComboBox.connect("currentIndexChanged(int)", self.onViewLayersChanged)
        form.addRow("Image:", self.backgroundComboBox)

        self.foregroundComboBox = qt.QComboBox()
        self.foregroundComboBox.toolTip = (
            "A SECOND image sequence blended on top of the one above, so two\n"
            "sequences can be compared in place (e.g. t1c over flair).  Use the\n"
            "blend slider to fade between them.  Only useful when the case has\n"
            "more than one sequence -- these rows stay hidden when it has one."
        )
        self.foregroundComboBox.connect("currentIndexChanged(int)", self.onViewLayersChanged)
        form.addRow("Compare with:", self.foregroundComboBox)

        self.opacitySlider = ctk.ctkSliderWidget()
        self.opacitySlider.minimum = 0.0
        self.opacitySlider.maximum = 1.0
        self.opacitySlider.singleStep = 0.05
        self.opacitySlider.decimals = 2
        self.opacitySlider.value = 0.5
        self.opacitySlider.toolTip = (
            "0 = show only the image above, 1 = show only the compared sequence."
        )
        self.opacitySlider.connect("valueChanged(double)", self.onOpacityChanged)
        form.addRow("Blend:", self.opacitySlider)

        # Window/level of the Image layer.  Slicer's own widget is used rather
        # than a pair of sliders: it carries the auto/manual modes and the
        # modality presets, and it writes straight to the volume display node,
        # so nothing here has to mirror that state.
        self.windowLevelWidget = slicer.qMRMLWindowLevelWidget()
        self.windowLevelWidget.toolTip = (
            "Brightness / contrast of the image above.\n"
            "Auto follows the volume, Manual lets you drag the range.\n"
            "Purely a display setting -- it never touches the voxels or the mask."
        )
        # the widget carries its own "Window/Level:" caption, which would sit
        # next to the form's own label and say the same thing twice
        innerLabel = self.windowLevelWidget.findChild(qt.QLabel, "label")
        if innerLabel is not None:
            innerLabel.setVisible(False)
        form.addRow("Contrast:", self.windowLevelWidget)

        self.layoutComboBox = qt.QComboBox()
        for title, attributeName, fallback in LAYOUT_CHOICES:
            self.layoutComboBox.addItem(title, layoutId(attributeName, fallback))
        self.layoutComboBox.connect("activated(int)", self.onLayoutChanged)
        form.addRow("Layout:", self.layoutComboBox)

        # Ticked by default: in this data the mask is a median 8 degrees off the
        # anatomical axes, so the default anatomical slice planes cut the voxel
        # grid diagonally and a stroke drawn as a clean disc is committed as a
        # stair-stepped one.  Editing wants the acquisition plane.  Untick to
        # get true axial / coronal / sagittal back for judging anatomy.
        self.alignSlicesCheckBox = qt.QCheckBox("align to the image grid")
        self.alignSlicesCheckBox.checked = True
        self.alignSlicesCheckBox.toolTip = (
            "Rotate the slice views onto the mask's own voxel axes, so painting "
            "lands on whole voxels instead of stair-stepping.\n"
            "Untick for true axial / coronal / sagittal.  Display only -- it "
            "never changes a voxel."
        )
        self.alignSlicesCheckBox.connect("toggled(bool)", self.onAlignSlicesChanged)
        form.addRow("Slices:", self.alignSlicesCheckBox)

        maskRow = qt.QHBoxLayout()
        maskRow.setSpacing(6)
        self.maskVisibleCheckBox = qt.QCheckBox("show")
        self.maskVisibleCheckBox.checked = True
        self.maskVisibleCheckBox.toolTip = "Show / hide the mask in every view (s)"
        self.maskVisibleCheckBox.connect("toggled(bool)", self.onMaskDisplayChanged)
        maskRow.addWidget(self.maskVisibleCheckBox)
        self.maskOpacitySlider = ctk.ctkSliderWidget()
        self.maskOpacitySlider.minimum = 0.0
        self.maskOpacitySlider.maximum = 1.0
        self.maskOpacitySlider.singleStep = self.MASK_OPACITY_STEP
        self.maskOpacitySlider.pageStep = self.MASK_OPACITY_STEP
        self.maskOpacitySlider.decimals = 1
        self.maskOpacitySlider.value = 0.5
        self.maskOpacitySlider.toolTip = (
            "Fill opacity of the mask in the slice views.\n"
            "a = 10% more transparent, d = 10% more opaque, s = hide / show."
        )
        self.maskOpacitySlider.connect("valueChanged(double)", self.onMaskDisplayChanged)
        maskRow.addWidget(self.maskOpacitySlider, 1)
        form.addRow("Mask:", maskRow)

        self.fitViewsButton = qt.QPushButton("Reset field of view")
        self.fitViewsButton.connect("clicked()", self.onFitViews)
        form.addRow("", self.fitViewsButton)

    # -- lesions ------------------------------------------------------------- #
    def _buildLesionSection(self):
        section = ctk.ctkCollapsibleButton()
        section.text = "Lesions"
        self._accentSection(section, self.ACCENT_LESIONS, "Lesions")
        self.layout.addWidget(section)
        box = self._tighten(qt.QVBoxLayout(section), spacing=3)

        # The row below (min voxels / refresh / auto / flash / staleness) is
        # built but never shown: the defaults it carries -- auto-refresh on,
        # flash on, no size filter -- are the only ones the review workflow
        # uses, and the controls were more clutter than choice.  The widgets
        # stay alive because the refresh and flash logic reads them.
        self.lesionControls = qt.QWidget()
        controls = qt.QHBoxLayout(self.lesionControls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)
        controls.addWidget(qt.QLabel("Min voxels:"))
        self.minVoxelsSpinBox = qt.QSpinBox()
        self.minVoxelsSpinBox.minimum = 1
        self.minVoxelsSpinBox.maximum = 1000000
        self.minVoxelsSpinBox.value = 1
        self.minVoxelsSpinBox.toolTip = "Hide components smaller than this (display only)."
        self.minVoxelsSpinBox.connect("editingFinished()", self.onRefreshLesions)
        controls.addWidget(self.minVoxelsSpinBox)
        self.refreshLesionsButton = qt.QPushButton("Refresh lesions")
        self.refreshLesionsButton.toolTip = (
            "Recompute the lesion list from the current mask.\n"
            "Untick Auto on large cases: the recount runs on the UI thread."
        )
        self.refreshLesionsButton.connect("clicked()", self.onRefreshLesions)
        controls.addWidget(self.refreshLesionsButton)
        self.autoRefreshCheckBox = qt.QCheckBox("Auto")
        self.autoRefreshCheckBox.checked = True
        self.autoRefreshCheckBox.toolTip = (
            "Recompute the lesion list a moment after the mask stops changing."
        )
        controls.addWidget(self.autoRefreshCheckBox)
        self.flashLesionsCheckBox = qt.QCheckBox("Flash")
        self.flashLesionsCheckBox.checked = True
        self.flashLesionsCheckBox.toolTip = (
            "Blink the lesion a few times after jumping to it, so it is clear "
            "which one of several nearby components is selected."
        )
        controls.addWidget(self.flashLesionsCheckBox)
        self.staleLabel = qt.QLabel("")
        controls.addWidget(self.staleLabel)
        controls.addStretch(1)
        self.lesionControls.setVisible(False)
        box.addWidget(self.lesionControls)

        # New-lesion mode has existed in the logic all along with nothing to
        # start it: the button and label combo it reads were never built, so
        # onNewLesionToggled raised AttributeError on the missing combo.  The
        # label now comes from the Active label box instead.
        newRow = qt.QHBoxLayout()
        newRow.setSpacing(4)
        self.newLesionButton = qt.QPushButton("New lesion")
        self.newLesionButton.setIcon(self._historyIcon("plus"))
        self.newLesionButton.setIconSize(qt.QSize(18, 18))
        self.newLesionButton.setCheckable(True)
        self.newLesionButton.toolTip = (
            "Start a new lesion: paint anywhere the mask is still empty and the "
            "stroke becomes its own row once the list refreshes.\n"
            "It is painted with the Active label chosen below.  Esc cancels."
        )
        self.newLesionButton.connect("toggled(bool)", self.onNewLesionToggled)
        newRow.addWidget(self.newLesionButton)
        newRow.addStretch(1)
        box.addLayout(newRow)

        self.lesionTable = qt.QTableWidget()
        self.lesionTable.setColumnCount(5)
        self.lesionTable.setHorizontalHeaderLabels(
            ["#", "Voxels", "Volume (mm3)", "Done", ""]
        )
        self.lesionTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.lesionTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.lesionTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.lesionTable.verticalHeader().setVisible(False)
        self.lesionTable.setSortingEnabled(True)
        self.lesionTable.verticalHeader().setDefaultSectionSize(20)
        self.lesionTable.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        try:
            header = self.lesionTable.horizontalHeader()
            header.setSectionResizeMode(qt.QHeaderView.Stretch)
            # the delete column holds a button, not text: keep it button-wide
            header.setSectionResizeMode(
                self.LESION_DELETE_COLUMN, qt.QHeaderView.ResizeToContents
            )
            # sorting by a column of widgets would shuffle the rows by nothing
            header.connect(
                "sortIndicatorChanged(int,Qt::SortOrder)", self.onLesionSortChanged
            )
        except Exception:  # noqa: BLE001 - older Qt binding
            logging.debug("GTReview: header setup unavailable", exc_info=True)
        self.lesionTable.toolTip = (
            "One row per connected component of the mask.\n"
            "Select a row to act on that lesion; press j to jump the slice "
            "views to it."
        )
        self.lesionTable.connect("itemSelectionChanged()", self.onLesionSelectionChanged)
        # itemSelectionChanged does not fire when the already-selected row is
        # clicked again, so wire the click too: re-clicking re-flashes.
        self.lesionTable.connect("itemClicked(QTableWidgetItem*)", self.onLesionClicked)
        self.lesionTable.connect("itemChanged(QTableWidgetItem*)", self.onLesionItemChanged)
        box.addWidget(self.lesionTable)

        self.lesionSummaryLabel = qt.QLabel("No lesions.")
        box.addWidget(self.lesionSummaryLabel)

    # -- editing ------------------------------------------------------------- #
    def _buildEditingSection(self):
        section = ctk.ctkCollapsibleButton()
        section.text = "Editing"
        self._accentSection(section, self.ACCENT_EDITING, "Editing")
        self.layout.addWidget(section)
        box = self._tighten(qt.QVBoxLayout(section), spacing=3)

        # A compact history toolbar at the top (Undo / Redo / Reset as icons),
        # then the brush and the segment list.  Lesion actions live on the
        # keyboard (Del deletes a lesion) so the panel stays uncluttered.
        historyRow = qt.QHBoxLayout()
        historyRow.setSpacing(4)

        def iconButton(kind, text, tooltip, slot):
            """One drawn glyph per action, with the label spelled out.

            Slicer ships its undo and redo in one style and its restore icon in
            another, and the QStyle arrows before those read as back/forward
            navigation; three buttons in three styles look unrelated.  Drawing
            them keeps one stroke weight and one colour across the row.

            A framed QPushButton rather than a flat tool button: these sit in a
            panel of QPushButtons ("Reset field of view", "Save & next case")
            and a borderless row read as decoration next to them.
            """
            button = qt.QPushButton(text)
            button.setIcon(self._historyIcon(kind))
            button.setIconSize(qt.QSize(18, 18))
            button.toolTip = tooltip
            button.connect("clicked()", slot)
            historyRow.addWidget(button)
            return button

        self.undoButton = iconButton(
            "undo", "Undo", "Undo the last edit (Ctrl+Z)", self.onUndo)
        self.redoButton = iconButton(
            "redo", "Redo", "Redo (Ctrl+Y or Ctrl+Shift+Z)", self.onRedo)
        self.resetButton = iconButton(
            "reset", "Reset",
            "Reset all: reload the mask from disk, discarding all edits.",
            self.onReset)
        historyRow.addStretch(1)
        box.addLayout(historyRow)

        self.saveAndNextButton = qt.QPushButton("Save && next case")
        self.saveAndNextButton.toolTip = (
            "Write <case_id>_reviewed_seg.nii.gz next to the case and open the "
            "next one.  Ctrl+S saves without moving on."
        )
        font = self.saveAndNextButton.font
        font.setBold(True)
        self.saveAndNextButton.setFont(font)
        self.saveAndNextButton.setMinimumHeight(30)
        self.saveAndNextButton.connect("clicked()", self.onSaveAndNext)
        box.addWidget(self.saveAndNextButton)

        try:
            import qSlicerSegmentationsModuleWidgetsPythonQt as segmentationWidgets
        except ImportError:  # pragma: no cover - broken Slicer install
            logging.error("GTReview: the Segmentations module widgets are unavailable")
            box.addWidget(qt.QLabel("Segment editor unavailable."))
            return

        # ITK-SNAP's two questions, in its words: which label goes down, and
        # what may be painted over.  The second one is Slicer's masking
        # section, which this module hides -- driving it from one combo box
        # keeps the setting that matters without the other six.
        labelForm = self._tighten(qt.QFormLayout())

        self.activeLabelComboBox = qt.QComboBox()
        self.activeLabelComboBox.toolTip = (
            "The label the brush lays down.  The same choice as the segment "
            "list on the right; the two follow each other."
        )
        for value in sorted(LABEL_NAMES):
            self.activeLabelComboBox.addItem(
                self._labelIcon(value), nameForLabelValue(value), int(value)
            )
        # ITK-SNAP's "clear label": painting background is erasing, so this
        # entry is the Erase tool under the name the reviewer is looking for
        self.activeLabelComboBox.addItem(
            self._labelIcon(self.ACTIVE_LABEL_BACKGROUND),
            "Background (erase)",
            self.ACTIVE_LABEL_BACKGROUND,
        )
        self.activeLabelComboBox.connect("activated(int)", self.onActiveLabelChanged)
        labelForm.addRow("Active label:", self.activeLabelComboBox)

        self.paintOverComboBox = qt.QComboBox()
        self.paintOverComboBox.toolTip = (
            "Which voxels the brush is allowed to change.\n"
            "All labels: overwrite anything.\n"
            "Background only: never touch a voxel that already has a label.\n"
            "A single label: only that label may be painted over."
        )
        self.paintOverComboBox.addItem("All labels", self.PAINT_OVER_ALL)
        self.paintOverComboBox.addItem("Background only", self.PAINT_OVER_BACKGROUND)
        for value in sorted(LABEL_NAMES):
            self.paintOverComboBox.addItem(
                self._labelIcon(value),
                "Only {}".format(nameForLabelValue(value)),
                int(value),
            )
        self.paintOverComboBox.connect("activated(int)", self.onPaintOverChanged)
        labelForm.addRow("Paint over:", self.paintOverComboBox)
        box.addLayout(labelForm)

        self.editor = segmentationWidgets.qMRMLSegmentEditorWidget()
        # Painting is immediate (see _applyImmediatePaint), and Slicer saves an
        # undo state per brush stamp rather than per stroke, so one drag can eat
        # a dozen states.  20 would leave no history after a couple of strokes;
        # each state is a labelmap copy, so this is not free either.
        self.editor.setMaximumNumberOfUndoStates(60)
        # setUndoEnabled(True) CLEARS the undo history every time it is called
        # (verified on 5.10), so it runs exactly once, here, before any edit.
        # Its read-back getter only reports widget visibility, never rely on it.
        self.editor.setUndoEnabled(True)
        # The editor would otherwise rewrite the slice layers whenever its
        # source volume changes; GTReview manages Image / Compare with itself.
        self.editor.setAutoShowSourceVolumeNode(False)
        self.selectParameterNode()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor.setSegmentationNodeSelectorVisible(False)
        self.editor.setSourceVolumeNodeSelectorVisible(False)
        self.editor.setSwitchToSegmentationsButtonVisible(False)
        # no labels beyond 1 and 2, 3D is always on, geometry is the mask's
        self.editor.setAddRemoveSegmentButtonsVisible(False)
        self.editor.setShow3DButtonVisible(False)
        self.editor.setSpecifyGeometryButtonVisible(False)
        # must be set BEFORE the first effect activation to take effect
        self.editor.setMaskingSectionVisible(False)
        # Paint, Erase and our Sphere threshold only.  Slicer's own Threshold
        # is deliberately absent: its Apply replaces the whole label.
        self.editor.setEffectNameOrder(list(self.EDITOR_EFFECTS))
        self.editor.unorderedEffectsVisible = False  # a property, not a slot, in 5.10
        self.editor.setEffectButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        # one row: None, Paint, Erase, Sphere threshold
        self.editor.setEffectColumnCount(len(self.EDITOR_EFFECTS) + 1)
        box.addWidget(self.editor)
        # EffectHelpBrowser is the "Paint with a round brush... Show details."
        # line above every effect's options: prose about a tool the reviewer has
        # already chosen, re-read on every switch.
        for name in ("UndoRedoGroupBox", "MaskingGroupBox", "EffectHelpBrowser"):
            child = self.editor.findChild(qt.QWidget, name)
            if child is not None:
                child.setVisible(False)
        self._pinEffectGridLeft()
        self._moveEffectOptionsBelow(box)
        # With the segment list hidden and the options moved out, the editor
        # holds nothing but the tool row -- but it still had an expanding
        # vertical policy, so any spare height in the section opened up as a
        # band of nothing between the tools and their settings.
        self.editor.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        editorLayout = self.editor.layout()
        if editorLayout is not None:
            editorLayout.setContentsMargins(0, 0, 0, 0)
            editorLayout.setSpacing(0)
        self._configureSegmentsTable()
        self.logic.editorWidget = self.editor
        # keep the save button as the last thing in the section
        box.removeWidget(self.saveAndNextButton)
        box.addWidget(self.saveAndNextButton)

        # picking a row in the editor's segment list must move the combo too
        self.editor.connect(
            "currentSegmentIDChanged(QString)", self.onEditorSegmentChanged
        )

        self.effectFactorySingleton = slicer.qSlicerSegmentEditorEffectFactory.instance()
        self.effectFactorySingleton.connect(
            "effectRegistered(QString)", self.editorEffectRegistered
        )

    def editorEffectRegistered(self, effectName=None):
        del effectName
        if self.editor is not None:
            self.editor.updateEffectList()
            self._pinEffectGridLeft()  # updateEffectList rebuilds the grid

    def _moveEffectOptionsBelow(self, box):
        """Put each effect's options under the tool row instead of beside it.

        The editor lays the options group box out in the column next to the
        buttons, and every effect asks for a different width there -- so the
        tool row was a different size for each tool and the section reflowed on
        every click.  Underneath, the options get the full panel width and the
        row above them never moves.
        """
        if self.editor is None:
            return
        options = self.editor.findChild(qt.QWidget, "OptionsGroupBox")
        if options is None:
            return
        # reparenting takes it out of the editor's grid; the editor keeps its
        # own pointer to the widget and goes on filling it as effects change.
        # It also takes the box out of the editor's QObject tree, so anything
        # that reached the effect options through self.editor.findChild has to
        # start from here instead -- see _effectOptionsRoot.
        self._effectOptionsBox = options
        box.addWidget(options)

    def _effectOptionsRoot(self):
        """Where the active effect's option widgets live.

        The editor until _moveEffectOptionsBelow runs, the reparented group box
        afterwards.  Searching the wrong one silently finds nothing.
        """
        return self._effectOptionsBox if self._effectOptionsBox is not None else self.editor

    def _pinEffectGridLeft(self):
        """Stop the effect buttons resizing as the options beside them change.

        The buttons live in a QGridLayout whose columns share out whatever
        width the group box has, so a wider set of options next door made every
        button narrower and clicking around the row moved the targets.  An
        empty stretch column past the last one soaks up the spare width instead
        and leaves the buttons their natural size, against the left edge.
        """
        if self.editor is None:
            return
        effects = self.editor.findChild(qt.QWidget, "EffectsGroupBox")
        grid = effects.layout() if effects is not None else None
        if grid is None or not hasattr(grid, "setColumnStretch"):
            return
        try:
            grid.setColumnStretch(grid.columnCount(), 1)
        except Exception:  # noqa: BLE001 - layout without column stretch
            logging.debug("GTReview: pinning the effect grid failed", exc_info=True)

    def _configureSegmentsTable(self):
        """Hide the editor's segment list; the Active label box replaces it.

        The widget itself has to stay alive -- the editor keeps its current
        segment through it -- so the resizable frame around it is hidden rather
        than removed, and the rest of the tidying below still runs in case a
        future change shows it again.
        """
        table = self.editor.findChild(qt.QWidget, "SegmentsTableView") if self.editor else None
        if table is None:
            return
        for name in ("SegmentsTableResizableFrame", "SegmentsTableView"):
            widget = self.editor.findChild(qt.QWidget, name)
            if widget is not None:
                widget.setVisible(False)
        for name, value in (("opacityColumnVisible", False), ("statusColumnVisible", False),
                            ("visibilityColumnVisible", False), ("readOnly", True)):
            try:
                setattr(table, name, value)
            except Exception:  # noqa: BLE001 - property missing in another Slicer
                logging.debug("GTReview: segments table property %s unavailable", name)
        # readOnly leaves "Clear selected segments" in the right-click menu: it
        # zeroes a whole label, skips the edit gate and is not undoable
        table.setContextMenuPolicy(qt.Qt.NoContextMenu)
        inner = table.findChild(qt.QTableView, "SegmentsTable")
        if inner is not None:
            inner.setContextMenuPolicy(qt.Qt.NoContextMenu)
        # There are always exactly two rows; height the table to them so the
        # panel is not padded with empty space below label 2.
        self._compactSegmentsTable(table, inner)

    def _compactSegmentsTable(self, table, inner):
        rows = len(LABEL_NAMES)
        rowHeight = 24
        headerHeight = 26
        if inner is not None:
            try:
                inner.verticalHeader().setVisible(False)
                rowHeight = inner.verticalHeader().defaultSectionSize or rowHeight
                headerHeight = inner.horizontalHeader().height or headerHeight
            except Exception:  # noqa: BLE001 - binding differences
                logging.debug("GTReview: sizing the segments table failed", exc_info=True)
        height = headerHeight + rows * rowHeight + 6
        # the editor wraps the table in a drag-resizable frame; pin both
        for widget in (self.editor.findChild(qt.QWidget, "SegmentsTableResizableFrame"), table):
            if widget is not None:
                widget.setMinimumHeight(height)
                widget.setMaximumHeight(height)

    def selectParameterNode(self):
        node = slicer.mrmlScene.GetSingletonNode("GTReview", "vtkMRMLSegmentEditorNode")
        if node is None:
            node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            node.UnRegister(None)
            node.SetSingletonTag("GTReview")
            node = slicer.mrmlScene.AddNode(node)
        if self.segmentEditorNode is not node:
            self._unobserveSegmentEditorNode()
            self.segmentEditorNode = node
            if self.editor is not None:
                self.editor.setMRMLSegmentEditorNode(node)
            self._observeSegmentEditorNode()

    def _observeSegmentEditorNode(self):
        node = self.segmentEditorNode
        if node is None or self._observedEditorNode is node:
            return
        self.addObserver(node, vtk.vtkCommand.ModifiedEvent, self._onSegmentEditorNodeModified)
        self._observedEditorNode = node

    def _unobserveSegmentEditorNode(self):
        node = self._observedEditorNode
        if node is None:
            return
        try:
            self.removeObserver(node, vtk.vtkCommand.ModifiedEvent, self._onSegmentEditorNodeModified)
        except Exception:  # noqa: BLE001 - node already gone
            logging.debug("GTReview: removing the editor-node observer failed", exc_info=True)
        self._observedEditorNode = None

    def _onSegmentEditorNodeModified(self, caller=None, event=None):
        """The one barrier that really stops painting.

        Every path that activates an effect -- the grid buttons, digit keys,
        our own slots, the Python console -- ends up writing the active effect
        name into the editor node, which fires this.  If no lesion is selected
        and no new one is being painted, the effect is dropped again.
        """
        del caller, event
        if self._enforcingEditGate or self.editor is None or self.segmentEditorNode is None:
            return
        active = self.segmentEditorNode.GetActiveEffectName() or ""
        if active and not self._editingAllowed():
            self._enforcingEditGate = True
            try:
                self.editor.setActiveEffect(None)
                slicer.util.showStatusMessage(
                    "GTReview: select a lesion, or start a new one, before editing.", 3000
                )
            finally:
                self._enforcingEditGate = False
            return
        if active:
            # first-time size defaults + keep the brush absolute/bounded, but
            # leave the ACTIVE SEGMENT to the user: they choose which label to
            # paint by clicking a row in the segment list.  A lesion selection
            # sets a sensible default label; it is not re-forced here.
            self._initialiseBrush()
            self._constrainBrush()
            self._applyImmediatePaint()
        # the sphere effect reports what it did; relay it once it applied
        self._relaySphereThresholdResult()

    # -- save ---------------------------------------------------------------- #
    # --------------------------------------------------------- enter/exit/close
    def enter(self):
        if self.editor is not None:
            # The editor's own shortcuts are deliberately NOT installed: its
            # Ctrl+Z / Ctrl+Shift+Z duplicate ours (Qt then fires neither) and
            # its Z/Y/Q/W/I keys bypass the lesion gate.  Ours cover the rest.
            self.selectParameterNode()
            self.editor.updateWidgetFromMRML()
        self._muteMainWindowShortcuts()
        self.installShortcuts()
        self._setDataProbeVisible(False)
        self._narrowModulePanel()
        self._updateEditingControls()

    def _narrowModulePanel(self):
        """Shrink the module panel to the narrowest Slicer will allow, once.

        The views are where the work happens; the panel only has to be legible.
        Its floor is Slicer's, not ours -- the sections themselves fit in about
        385 px and the rest is the module selector and the dock's own margins.

        Only the first entry per session does this: Slicer remembers the dock
        width between runs, and a width dragged by hand should survive leaving
        the module and coming back.
        """
        if self._panelNarrowed:
            return
        self._panelNarrowed = True
        mainWindow = slicer.util.mainWindow()
        if mainWindow is None:
            return
        dock = mainWindow.findChild(qt.QDockWidget, "PanelDockWidget")
        if dock is None:
            return
        try:
            width = max(int(dock.minimumWidth), int(dock.minimumSizeHint.width()))
            mainWindow.resizeDocks([dock], [width], qt.Qt.Horizontal)
        except Exception:  # noqa: BLE001 - no resizeDocks in an older Qt
            logging.debug("GTReview: resizing the module panel failed", exc_info=True)

    def exit(self):
        self._cancelNewLesion()
        if self.editor is not None:
            self.editor.setActiveEffect(None)
            self.editor.removeViewObservations()
        self.removeShortcuts()
        self._restoreMainWindowShortcuts()
        self._setDataProbeVisible(True)

    def _muteMainWindowShortcuts(self):
        """Park the main window's Ctrl+S / Ctrl+Z / Ctrl+Y while we are open.

        A key sequence owned by two enabled shortcuts is "ambiguous" to Qt and
        reaches NEITHER, so without this Save and Undo keys are dead here.
        """
        self._restoreMainWindowShortcuts()
        mainWindow = slicer.util.mainWindow()
        if mainWindow is None:
            return
        for name in self.MUTED_MAIN_WINDOW_ACTIONS:
            action = mainWindow.findChild(qt.QAction, name)
            if action is None:
                continue
            try:
                self._mutedActions.append((action, qt.QKeySequence(action.shortcut)))
                action.setShortcut(qt.QKeySequence())
            except Exception:  # noqa: BLE001 - never block entering the module
                logging.debug("GTReview: muting %s failed", name, exc_info=True)

    def _restoreMainWindowShortcuts(self):
        muted, self._mutedActions = self._mutedActions, []
        for action, sequence in muted:
            try:
                action.setShortcut(sequence)
            except Exception:  # noqa: BLE001
                logging.debug("GTReview: restoring a main-window shortcut failed", exc_info=True)

    def _setDataProbeVisible(self, visible):
        """Hide Slicer's Data Probe panel; it only crowds this module.

        Restored on exit() so leaving GTReview does not change Slicer for the
        rest of the session.
        """
        mainWindow = slicer.util.mainWindow()
        if mainWindow is None:
            return
        widget = mainWindow.findChild(qt.QWidget, "DataProbeCollapsibleWidget")
        if widget is not None:
            widget.setVisible(bool(visible))

    def onSceneStartClose(self, caller=None, event=None):
        del caller, event
        self._cancelNewLesion()
        self.clearLesionHighlight()
        self._unobserveSegmentEditorNode()
        # StartCloseEvent cannot be vetoed, so this is Save-or-lose rather
        # than the usual Save / Discard / Cancel prompt.
        self._offerSaveBeforeSceneClose()
        self.segmentEditorNode = None
        self._stopObservingSegmentation()
        if self.editor is not None:
            self.editor.setSegmentationNode(None)
            self.editor.removeViewObservations()
        self.logic.unloadCase()
        self.unsavedChanges = False
        self.componentMap = None
        self.lesionList = []
        self._populateLesionTable()

    def _offerSaveBeforeSceneClose(self):
        """Last chance to keep unsaved edits before the scene is torn down."""
        if not self.unsavedChanges or self.logic.case is None:
            return
        if slicer.app.testingEnabled() or slicer.util.mainWindow() is None:
            logging.info(
                "GTReview: testing/headless mode -- scene closing, "
                "discarding unsaved edits without asking"
            )
            return
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Warning)
        box.setWindowTitle("GTReview -- unsaved edits")
        box.setText("Case {} has unsaved edits.".format(self.logic.case.case_id))
        box.setInformativeText(
            "The scene is closing and the edits cannot be kept in memory. "
            "Save them now?"
        )
        box.setStandardButtons(qt.QMessageBox.Save | qt.QMessageBox.Discard)
        box.setDefaultButton(qt.QMessageBox.Save)
        if box.exec_() == qt.QMessageBox.Save:
            try:
                self.saveCurrentCase()
            except Exception:  # noqa: BLE001 - never block the scene close
                logging.exception("GTReview: saving before scene close failed")

    def onSceneEndClose(self, caller=None, event=None):
        del caller, event
        if self.parent.isEntered:
            self.selectParameterNode()
            if self.editor is not None:
                self.editor.updateWidgetFromMRML()

    def cleanup(self):
        self._cancelNewLesion()
        self.clearLesionHighlight()
        self.removeShortcuts()
        self._restoreMainWindowShortcuts()
        self._setDataProbeVisible(True)
        self._unobserveSegmentEditorNode()
        self.removeObservers()
        if self._refreshTimer is not None:
            self._refreshTimer.stop()
        if self.effectFactorySingleton is not None:
            try:
                self.effectFactorySingleton.disconnect(
                    "effectRegistered(QString)", self.editorEffectRegistered
                )
            except Exception:  # noqa: BLE001
                logging.debug("GTReview: effect factory disconnect failed", exc_info=True)
        if self.editor is not None:
            self.editor.setActiveEffect(None)
            self.editor.removeViewObservations()
            self.editor.setMRMLScene(None)
            self.editor.setMRMLSegmentEditorNode(None)

    # ------------------------------------------------------------- shortcuts
    def installShortcuts(self):
        self.removeShortcuts()
        mainWindow = slicer.util.mainWindow()
        if not mainWindow:
            return
        control = qt.Qt.ControlModifier
        shift = qt.Qt.ShiftModifier
        none = qt.Qt.NoModifier
        bindings = (
            ("n", qt.Qt.Key_N, none, "n", self.onNextCase),
            ("p", qt.Qt.Key_P, none, "p", self.onPreviousCase),
            ("Ctrl+S", qt.Qt.Key_S, control, "", self.onSave),
            ("j", qt.Qt.Key_J, none, "j", self.onJumpToLesion),
            ("a", qt.Qt.Key_A, none, "a", self.onMaskMoreTransparent),
            ("s", qt.Qt.Key_S, none, "s", self.onToggleMaskVisible),
            ("d", qt.Qt.Key_D, none, "d", self.onMaskMoreOpaque),
            ("1", qt.Qt.Key_1, none, "1", functools.partial(self.onActivateEffect, "Paint")),
            ("2", qt.Qt.Key_2, none, "2", functools.partial(self.onActivateEffect, "Erase")),
            ("3", qt.Qt.Key_3, none, "3",
             functools.partial(self.onActivateEffect, SPHERE_THRESHOLD_EFFECT)),
            ("Esc", qt.Qt.Key_Escape, none, "", self.onStopEditing),
            ("Delete", qt.Qt.Key_Delete, none, "", self.onDeleteLesion),
            ("Ctrl+Z", qt.Qt.Key_Z, control, "", self.onUndo),
            ("Ctrl+Y", qt.Qt.Key_Y, control, "", self.onRedo),
            ("Ctrl+Shift+Z", qt.Qt.Key_Z, control | shift, "", self.onRedo),
        )
        for keys, key, modifiers, text, callback in bindings:
            shortcut = qt.QShortcut(qt.QKeySequence(keys), mainWindow)
            # Esc must keep closing dialogs, so it only applies to the main window
            shortcut.setContext(
                qt.Qt.WindowShortcut if keys == "Esc" else qt.Qt.ApplicationShortcut
            )
            shortcut.connect(
                "activated()", self._shortcutHandler(callback, key, modifiers, text)
            )
            self._shortcuts.append(shortcut)

    def _shortcutHandler(self, callback, key, modifiers, text):
        """Wrap a shortcut so it never eats a keystroke meant for a text box."""

        def handler():
            # A focused text box consumes printable keys before the shortcut
            # map sees them, so reaching here with a printable key (text) while
            # a text box is focused means the box declined it; re-sending it
            # would only re-fire this same shortcut (a RecursionError, verified
            # for Esc / Ctrl+S).  Typed letters are simply not acted upon; keys
            # without text (Esc, Ctrl+S, Del) always run.  key/modifiers are
            # kept in the signature for callers/readers but no longer used.
            if text and focusedTextInput() is not None:
                return
            callback()

        self._shortcutHandlers.append(handler)  # keep a python reference alive
        return handler

    def removeShortcuts(self):
        for shortcut in self._shortcuts:
            try:
                shortcut.disconnect("activated()")
                shortcut.setParent(None)
                shortcut.deleteLater()
            except Exception:  # noqa: BLE001
                logging.debug("GTReview: removing a shortcut failed", exc_info=True)
        self._shortcuts = []
        self._shortcutHandlers = []

    # ------------------------------------------------------------ dataset slots
    @guarded("Choosing a batch directory")
    def onBrowseAndLoad(self):
        """Browse for a batch directory and load it in one click."""
        start = self.datasetPathEdit.currentPath or ""
        if start and not os.path.isdir(start):
            start = os.path.dirname(start)
        directory = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "GTReview - pick a batch directory", start
        )
        if not directory:
            return
        self.datasetPathEdit.currentPath = directory
        self.onLoadDataset()

    def _datasetHistory(self):
        """Previously loaded batch directories, most recent first."""
        stored = slicer.app.userSettings().value(DATASET_HISTORY_KEY)
        if stored is None:
            return []
        if isinstance(stored, str):
            stored = [stored]
        return [str(path) for path in stored if str(path).strip()]

    def _rememberDatasetPath(self):
        """Persist the batch directory so it is there on the next start.

        ``ctkPathLineEdit.addCurrentPathToHistory`` fills the dropdown for this
        session but writes nothing to the settings file, so the history is kept
        here instead.
        """
        path = str(self.datasetPathEdit.currentPath or "")
        if not path:
            return
        try:
            history = [path] + [p for p in self._datasetHistory() if p != path]
            del history[DATASET_HISTORY_LIMIT:]
            settings = slicer.app.userSettings()
            settings.setValue(DATASET_HISTORY_KEY, history)
            settings.sync()
            self.datasetPathEdit.addCurrentPathToHistory()
            self._populateDatasetHistory(history, current=path)
        except Exception:  # noqa: BLE001 - never fail a load over the history
            logging.debug("GTReview: storing the dataset path failed", exc_info=True)

    def _populateDatasetHistory(self, history=None, current=None):
        """Fill the path edit's dropdown from the persisted history."""
        history = self._datasetHistory() if history is None else history
        if not history:
            return
        comboBox = self.datasetPathEdit.comboBox()
        if comboBox is None:  # pragma: no cover - unexpected CTK internals
            return
        previous = self._updatingGui
        self._updatingGui = True
        try:
            comboBox.clear()
            for path in history:
                comboBox.addItem(path)
            self.datasetPathEdit.currentPath = current or history[0]
        finally:
            self._updatingGui = previous

    @guarded("Loading the dataset")
    def onLoadDataset(self):
        root = self.datasetPathEdit.currentPath
        if not root:
            slicer.util.errorDisplay("Pick a batch directory first.", windowTitle="GTReview")
            return
        if not self._confirmDiscardIfNeeded():
            return
        with BusyCursor("GTReview: discovering cases in {} ...".format(root)):
            self.cases = self.logic.discoverCases(root)
        if not self.cases:
            self.caseStatusLabel.text = (
                "0 cases found in {} — did you mean a batch_NN folder? "
                "(discovery is one level deep)".format(root)
            )
            self.filteredCases = []
            self.currentCaseIndex = -1
            self._populateCaseComboBox()
            # currentCase() is None now -- tear the old case down as well
            self.loadCurrentCase()
            return
        # ctkPathLineEdit only persists its history when something asks it to;
        # its own browse button used to, and that button is hidden now.
        self._rememberDatasetPath()
        slicer.util.showStatusMessage(
            "GTReview: {} cases found.".format(len(self.cases)), 3000
        )
        self._applyCaseFilter()
        if self.filteredCases:
            self.setCurrentCaseIndex(0, force=True)

    def _filteredCaseList(self, skipReviewed=None):
        """The case list the browser would show for a skip-reviewed setting."""
        if skipReviewed is None:
            skipReviewed = self.skipReviewedCheckBox.checked
        if skipReviewed:
            return [case for case in self.cases if not case.is_reviewed]
        return list(self.cases)

    def _applyCaseFilter(self):
        self.filteredCases = self._filteredCaseList()
        self._populateCaseComboBox()

    def _populateCaseComboBox(self):
        previous = self._updatingGui
        self._updatingGui = True
        try:
            self.caseComboBox.clear()
            for case in self.filteredCases:
                title = case.case_id + ("  ✓" if case.is_reviewed else "")
                self.caseComboBox.addItem(title)
            if 0 <= self.currentCaseIndex < len(self.filteredCases):
                self.caseComboBox.currentIndex = self.currentCaseIndex
        finally:
            self._updatingGui = previous
        self._updateCaseControls()

    def currentCase(self):
        if 0 <= self.currentCaseIndex < len(self.filteredCases):
            return self.filteredCases[self.currentCaseIndex]
        return None

    @guarded("Switching case")
    def onCaseSelected(self, index):
        if self._updatingGui or index < 0 or index >= len(self.filteredCases):
            return
        if index == self.currentCaseIndex:
            return
        self.setCurrentCaseIndex(index)

    @guarded("Skip-reviewed filter")
    def onSkipReviewedToggled(self, checked):
        if self._updatingGui:
            return
        current = self.currentCase()
        # Ticking the box can filter the *open* case out of the list, which
        # unloads it.  That is a case switch like any other, so ask about
        # unsaved edits before anything is discarded -- and put the checkbox
        # back if the user cancels.
        if current is not None and not any(
            case is current for case in self._filteredCaseList(checked)
        ):
            if not self._confirmDiscardIfNeeded():
                self._setSkipReviewedChecked(not checked)
                return
        self._applyCaseFilter()
        index = -1
        if current is not None:
            for position, case in enumerate(self.filteredCases):
                if case.directory == current.directory:
                    index = position
                    break
        if index < 0:
            index = 0 if self.filteredCases else -1
        self.currentCaseIndex = index
        self._populateCaseComboBox()
        if index < 0:
            # the filter emptied the list -- do not leave the old case loaded
            self.loadCurrentCase()
        elif current is None or self.filteredCases[index] is not current:
            # unsaved edits were already dealt with above
            self.setCurrentCaseIndex(index, force=True)

    def _setSkipReviewedChecked(self, checked):
        """Set the checkbox without re-entering onSkipReviewedToggled."""
        previous = self._updatingGui
        self._updatingGui = True
        try:
            self.skipReviewedCheckBox.checked = bool(checked)
        finally:
            self._updatingGui = previous

    @guarded("Next case")
    def onNextCase(self):
        if self.currentCaseIndex + 1 < len(self.filteredCases):
            self.setCurrentCaseIndex(self.currentCaseIndex + 1)
        else:
            slicer.util.showStatusMessage("GTReview: this is the last case.", 2000)

    @guarded("Previous case")
    def onPreviousCase(self):
        if self.currentCaseIndex > 0:
            self.setCurrentCaseIndex(self.currentCaseIndex - 1)
        else:
            slicer.util.showStatusMessage("GTReview: this is the first case.", 2000)

    def setCurrentCaseIndex(self, index, force=False, maskPath=None):
        """Switch to case *index*, asking about unsaved edits first."""
        if not force and not self._confirmDiscardIfNeeded():
            self._syncCaseComboBox()
            return
        self.currentCaseIndex = index
        self._syncCaseComboBox()
        self.loadCurrentCase(maskPath=maskPath)
        if self.skipReviewedCheckBox.checked:
            self._refilterCases(keepCurrent=True)

    def _syncCaseComboBox(self):
        previous = self._updatingGui
        self._updatingGui = True
        try:
            if 0 <= self.currentCaseIndex < self.caseComboBox.count:
                self.caseComboBox.currentIndex = self.currentCaseIndex
        finally:
            self._updatingGui = previous

    def _confirmDiscardIfNeeded(self):
        """Save / Discard / Cancel dialog. True == it is OK to proceed."""
        if not self.unsavedChanges or self.logic.case is None:
            return True
        if slicer.app.testingEnabled() or slicer.util.mainWindow() is None:
            logging.info(
                "GTReview: testing/headless mode — discarding unsaved edits without asking"
            )
            return True
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setIcon(qt.QMessageBox.Warning)
        box.setWindowTitle("GTReview — unsaved edits")
        box.setText(
            "Case {} has unsaved edits.".format(self.logic.case.case_id)
        )
        box.setInformativeText("Save them before leaving this case?")
        box.setStandardButtons(
            qt.QMessageBox.Save | qt.QMessageBox.Discard | qt.QMessageBox.Cancel
        )
        box.setDefaultButton(qt.QMessageBox.Save)
        answer = box.exec_()
        if answer == qt.QMessageBox.Cancel:
            return False
        if answer == qt.QMessageBox.Save:
            return bool(self.saveCurrentCase())
        return True

    @guarded("Changing the mask source")
    def onMaskSourceChanged(self, index):
        if self._updatingGui or index < 0:
            return
        path = self.maskSourceComboBox.itemData(index)
        if not path or path == self.logic.maskPath:
            return
        if not self._confirmDiscardIfNeeded():
            self._updateMaskSourceComboBox()
            return
        self.loadCurrentCase(maskPath=path)

    # --------------------------------------------------------------- case load
    def loadCurrentCase(self, maskPath=None):
        case = self.currentCase()
        self._newLesion = None
        self._stopObservingSegmentation()
        self._clearLesionSelection()
        if case is None:
            self.logic.unloadCase()
            self.unsavedChanges = False
            self.componentMap = None
            self.lesionList = []
            self._populateLesionTable()
            self._updateCaseControls()
            self._updateEditingControls()
            return
        with BusyCursor("GTReview: loading {} ...".format(case.case_id)):
            self.logic.loadCase(case, maskPath=maskPath)
        self.unsavedChanges = False
        # fingerprints of the previous case's strokes mean nothing here
        self._strokeStarts = []
        self._redoTargets = []

        self._updateMaskSourceComboBox()
        self._populateVolumeComboBoxes()
        self._attachEditor()
        self.applyMaskDisplay()
        self._startObservingSegmentation()
        self.applyLayout()
        self.applyViewLayers()
        self.onFitViews()
        self.refreshLesions()
        self._updateCaseControls()
        self._updateEditingControls()

        if self.logic.geometryWarning:
            slicer.util.warningDisplay(self.logic.geometryWarning, windowTitle="GTReview")
        if not case.masks and not case.is_reviewed:
            slicer.util.showStatusMessage(
                "GTReview: {} has no mask — starting from an empty segmentation.".format(
                    case.case_id
                ),
                5000,
            )

    def _attachEditor(self):
        if self.editor is None:
            return
        segmentationNode = self.logic.segmentationNode
        if segmentationNode is not None:
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
                self.logic.referenceVolumeNode
            )
        self.editor.setSegmentationNode(segmentationNode)
        sourceNode = self._backgroundNode() or self.logic.referenceVolumeNode
        self.editor.setSourceVolumeNode(sourceNode)
        # the editor node is a scene singleton: an intensity mask or overwrite
        # mode left over from another case (or a saved scene) must not leak in
        editorNode = self.segmentEditorNode
        if editorNode is not None:
            try:
                editorNode.SetSourceVolumeIntensityMask(False)
                # The EditAllowed* enum lives on the segmentation node, not on
                # the editor node: reading PaintAllowedEverywhere off the editor
                # node raised AttributeError here, the except below swallowed it,
                # and neither the mask mode NOR the overwrite mode on the next
                # line was ever reset -- so a mode really could leak between
                # cases, which is the one thing this block exists to stop.
                editorNode.SetMaskMode(
                    slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
                editorNode.SetOverwriteMode(
                    slicer.vtkMRMLSegmentEditorNode.OverwriteAllSegments)  # one label per voxel
            except Exception:  # noqa: BLE001 - older node API
                logging.debug("GTReview: resetting the editor node failed", exc_info=True)
        segmentIds = self.logic.segmentIds()
        if segmentIds:
            self.editor.setCurrentSegmentID(segmentIds[0])
        # segment IDs are new for every case, so the Paint over choice has to be
        # translated again against this case's segments
        self.applyPaintOver()
        self._syncActiveLabelComboBox()

    def _updateMaskSourceComboBox(self):
        previous = self._updatingGui
        self._updatingGui = True
        try:
            self.maskSourceComboBox.clear()
            case = self.currentCase()
            if case is None:
                return
            entries = []
            if case.is_reviewed:
                entries.append(("reviewed_seg (resume)", case.reviewed_path))
            for key in sorted(case.masks, key=lambda k: (dataset.natural_key(k), k)):
                entries.append((key, case.masks[key]))
            if not entries:
                self.maskSourceComboBox.addItem("(no mask — empty segmentation)", "")
            for title, path in entries:
                self.maskSourceComboBox.addItem(title, path)
            for position in range(self.maskSourceComboBox.count):
                if self.maskSourceComboBox.itemData(position) == self.logic.maskPath:
                    self.maskSourceComboBox.currentIndex = position
                    break
        finally:
            self._updatingGui = previous

    # One accent colour per section, so the collapsible headers are easy to
    # pick out while scrolling.  The fill is derived from the accent at runtime
    # and blended towards the current theme, so it stays readable in both the
    # light and the dark Slicer themes instead of hard-coding one palette.
    ACCENT_DATASET = "#2f7fd0"    # blue   - pick the data
    ACCENT_DISPLAY = "#2f9f6f"    # green  - what you see
    ACCENT_LESIONS = "#d08a2f"    # amber  - what you inspect
    ACCENT_EDITING = "#8a5fd0"    # purple - what you change

    def _accentSection(self, section, color, name):
        """Tint a collapsible section header with its accent colour.

        Styled through an object-name selector so the rule cannot leak into the
        section's children (the Segment Editor has collapsibles of its own).
        """
        objectName = "GTReviewSection{}".format(name)
        section.setObjectName(objectName)
        if self._isDarkTheme():
            fill = self._blend(color, "#000000", 0.70)
            text = self._blend(color, "#ffffff", 0.62)
        else:
            fill = self._blend(color, "#ffffff", 0.80)
            text = self._blend(color, "#000000", 0.50)
        section.setStyleSheet(
            "#{name} {{ background-color: {fill}; color: {text};"
            " border: 1px solid {c}; border-left: 5px solid {c};"
            " border-radius: 3px; font-weight: bold; padding: 2px; }}".format(
                name=objectName, fill=fill, text=text, c=color
            )
        )
        return section

    @staticmethod
    def _tighten(layout, spacing=2):
        """Squeeze a section's layout so the panel needs less scrolling."""
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(spacing)
        try:  # QFormLayout only
            layout.setVerticalSpacing(spacing)
            layout.setHorizontalSpacing(6)
            layout.setLabelAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
        except AttributeError:
            pass
        return layout

    @staticmethod
    def _isDarkTheme():
        try:
            color = slicer.app.palette().color(qt.QPalette.Window)
        except Exception:  # noqa: BLE001 - no QApplication palette
            return False
        luma = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return luma < 128

    @staticmethod
    def _blend(hexColor, target, amount):
        """*hexColor* moved *amount* (0..1) of the way towards *target*."""
        a = qt.QColor(hexColor)
        b = qt.QColor(target)
        mix = lambda x, y: int(round(x + (y - x) * amount))  # noqa: E731
        return qt.QColor(
            mix(a.red(), b.red()), mix(a.green(), b.green()), mix(a.blue(), b.blue())
        ).name()

    @staticmethod
    def _setFormRowVisible(form, field, visible):
        """Show/hide a QFormLayout row (the field and its label)."""
        field.setVisible(bool(visible))
        try:
            label = form.labelForField(field)
        except Exception:  # noqa: BLE001 - older Qt binding
            label = None
        if label is not None:
            label.setVisible(bool(visible))

    def _populateVolumeComboBoxes(self):
        previous = self._updatingGui
        self._updatingGui = True
        try:
            keys = sorted(self.logic.volumeNodes, key=lambda k: (dataset.natural_key(k), k))
            # "Compare with" blends a second sequence over the first; with only
            # one sequence (the common case here) both rows are dead weight.
            comparable = len(keys) > 1
            self._setFormRowVisible(self.displayForm, self.foregroundComboBox, comparable)
            self._setFormRowVisible(self.displayForm, self.opacitySlider, comparable)
            for comboBox, none_first in (
                (self.backgroundComboBox, False),
                (self.foregroundComboBox, True),
            ):
                selected = comboBox.currentText
                comboBox.clear()
                if none_first:
                    comboBox.addItem("(none)", "")
                for key in keys:
                    comboBox.addItem(key, key)
                if not none_first:
                    comboBox.addItem("(none)", "")
                index = comboBox.findText(selected)
                comboBox.currentIndex = index if index >= 0 else 0
        finally:
            self._updatingGui = previous

    def _updateCaseControls(self):
        total = len(self.filteredCases)
        position = self.currentCaseIndex + 1 if self.currentCaseIndex >= 0 else 0
        self.caseProgressLabel.text = "{} / {}".format(position, total)
        self.previousCaseButton.enabled = self.currentCaseIndex > 0
        self.nextCaseButton.enabled = 0 <= self.currentCaseIndex < total - 1
        case = self.currentCase()
        if case is None:
            if self.cases:
                self.caseStatusLabel.text = "{} cases loaded, none selected.".format(
                    len(self.cases)
                )
            return
        self.caseStatusLabel.text = self._caseStatusHtml(case)
        self.caseStatusLabel.toolTip = self._caseStatusTooltip(case)
        if self.deleteReviewButton is not None:
            self.deleteReviewButton.enabled = case.is_reviewed

    @staticmethod
    def _elide(text, limit=52):
        """Middle-elide a long path so the panel never grows a scrollbar."""
        text = str(text)
        if len(text) <= limit:
            return text
        keep = (limit - 3) // 2
        return "{}...{}".format(text[:keep], text[-keep:])

    def _caseStatusHtml(self, case):
        """One compact table: the case, then its images and masks by name."""
        current = os.path.abspath(self.logic.maskPath) if self.logic.maskPath else ""
        rows = []
        for key in sorted(case.images):
            rows.append(("image", key, case.images[key]))
        for key in sorted(case.masks):
            rows.append(("mask", key, case.masks[key]))
        if case.is_reviewed:
            rows.append(("mask", "reviewed", case.reviewed_path))

        flags = []
        if case.is_reviewed:
            flags.append("<font color='#3c8c3c'>reviewed &#10003;</font>")
        if self.unsavedChanges:
            flags.append("<b><font color='#c86400'>UNSAVED</font></b>")

        html = [
            "<table cellspacing='0' cellpadding='0'>",
            "<tr><td colspan='3'><b>{}</b>&nbsp;&nbsp;{}</td></tr>".format(
                case.case_id, "&nbsp;&nbsp;".join(flags)
            ),
        ]
        for kind, key, path in rows:
            isCurrent = os.path.abspath(path) == current
            html.append(
                "<tr>"
                "<td><font size='-1' color='gray'>{marker}</font></td>"
                "<td><font size='-1' color='{color}'>&nbsp;{key}&nbsp;&nbsp;</font></td>"
                "<td><font size='-1'>{name}</font></td>"
                "</tr>".format(
                    marker="&#9679;" if isCurrent else "&nbsp;",
                    color="#1a4672" if kind == "image" else "#724c1a",
                    key=key,
                    name=os.path.basename(path),
                )
            )
        html.append(
            "<tr><td colspan='3'><font size='-2' color='gray'>{}</font></td></tr>".format(
                self._elide(case.directory, 64)
            )
        )
        html.append("</table>")
        return "".join(html)

    def _caseStatusTooltip(self, case):
        lines = [case.directory, ""]
        for key in sorted(case.images):
            lines.append("image  {}: {}".format(key, case.images[key]))
        for key in sorted(case.masks):
            lines.append("mask   {}: {}".format(key, case.masks[key]))
        if case.is_reviewed:
            lines.append("mask   reviewed: {}".format(case.reviewed_path))
        if self.logic.maskPath:
            lines.append("")
            lines.append("reviewing: {}".format(self.logic.maskPath))
        return "\n".join(lines)

    def _editingAllowed(self):
        """Voxels may change only for a selected lesion or a new one."""
        return (
            self.logic.segmentationNode is not None
            and self.editor is not None
            and (self.selectedLesion() is not None or self._newLesion is not None)
        )

    def _setChecked(self, button, checked):
        """Set a checkable button without firing its toggled handler."""
        previous = self._updatingGui
        self._updatingGui = True
        try:
            button.checked = bool(checked)
        finally:
            self._updatingGui = previous

    def _setEditorEditable(self, editable):
        """Lock / unlock the brush.

        ``setReadOnly`` greys the effect buttons and drops the active effect,
        but an effect activated afterwards still paints (verified on 5.10),
        so the real barrier is :meth:`_onSegmentEditorNodeModified`.
        """
        if self.editor is None:
            return
        try:
            self.editor.setReadOnly(not editable)
        except Exception:  # noqa: BLE001 - older editor
            logging.debug("GTReview: setReadOnly unavailable", exc_info=True)
        if not editable and self.editor.activeEffect() is not None:
            self.editor.setActiveEffect(None)
        if editable:
            self._configureSegmentsTable()  # setReadOnly(False) re-enables renames

    def _updateEditingControls(self):
        hasCase = self.logic.segmentationNode is not None
        hasEditor = self.editor is not None
        lesion = self.selectedLesion() if hasCase else None
        newMode = self._newLesion is not None
        for widget in (
            self.undoButton,
            self.redoButton,
            self.refreshLesionsButton,
        ):
            widget.enabled = hasCase and hasEditor
        self.resetButton.enabled = hasCase and bool(self.logic.maskPath)
        done, total = self._doneCount()
        allDone = self._allLesionsDone()
        self.saveAndNextButton.enabled = hasCase and hasEditor and allDone
        if not hasCase:
            self.saveAndNextButton.toolTip = "Load a case first."
        elif self.lesionsStale:
            self.saveAndNextButton.toolTip = (
                "The lesion list is stale -- refresh it and tick every lesion as Done."
            )
        elif allDone:
            self.saveAndNextButton.toolTip = (
                "Write <case_id>_reviewed_seg.nii.gz next to the case and open the "
                "next one.  Ctrl+S saves without moving on."
            )
        else:
            self.saveAndNextButton.toolTip = (
                "Tick every lesion as Done first ({} of {} done).".format(done, total)
            )

        # Del deletes the selected lesion -- keyboard only.
        self._setEditorEditable(hasCase and hasEditor and (lesion is not None or newMode))

    # ------------------------------------------------------------ display slots
    def _backgroundNode(self):
        key = self.backgroundComboBox.itemData(self.backgroundComboBox.currentIndex)
        if not key:
            return None
        return self.logic.volumeNodes.get(key)

    def _foregroundNode(self):
        key = self.foregroundComboBox.itemData(self.foregroundComboBox.currentIndex)
        if not key:
            return None
        return self.logic.volumeNodes.get(key)

    @guarded("Changing the displayed volumes")
    def onViewLayersChanged(self, index=None):
        del index
        if self._updatingGui:
            return
        self.applyViewLayers()
        if self.editor is not None and self.logic.segmentationNode is not None:
            self.editor.setSourceVolumeNode(
                self._backgroundNode() or self.logic.referenceVolumeNode
            )

    @guarded("Changing the foreground opacity")
    def onOpacityChanged(self, value):
        del value
        if self._updatingGui:
            return
        self.applyViewLayers()

    @guarded("Changing the mask display")
    def onMaskDisplayChanged(self, value=None):
        del value
        if self._updatingGui:
            return
        self.applyMaskDisplay()

    def applyMaskDisplay(self):
        """Push the show-checkbox and opacity slider onto the segmentation."""
        node = self.logic.segmentationNode
        displayNode = node.GetDisplayNode() if node is not None else None
        if displayNode is None:
            return
        displayNode.SetVisibility(bool(self.maskVisibleCheckBox.checked))
        displayNode.SetOpacity2DFill(float(self.maskOpacitySlider.value))

    def _stepMaskOpacity(self, delta):
        value = float(self.maskOpacitySlider.value) + delta
        value = max(0.0, min(1.0, round(value, 2)))
        self.maskOpacitySlider.value = value  # fires onMaskDisplayChanged
        slicer.util.showStatusMessage(
            "GTReview: mask opacity {:.0%}".format(value), 1500
        )

    @guarded("Mask opacity")
    def onMaskMoreTransparent(self):
        self._stepMaskOpacity(-self.MASK_OPACITY_STEP)

    @guarded("Mask opacity")
    def onMaskMoreOpaque(self):
        self._stepMaskOpacity(+self.MASK_OPACITY_STEP)

    @guarded("Mask visibility")
    def onToggleMaskVisible(self):
        self.maskVisibleCheckBox.checked = not self.maskVisibleCheckBox.checked
        slicer.util.showStatusMessage(
            "GTReview: mask {}".format(
                "shown" if self.maskVisibleCheckBox.checked else "hidden"
            ),
            1500,
        )

    def applySlicePlanes(self):
        """Outline the 2D slices inside the 3D view (always on).

        Only the frame is drawn: the textured slice itself is a large opaque
        quad that hides whatever lesion sits behind it, which is the opposite
        of what this is for.
        """
        for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            sliceNode.SetSliceVisible(False)
            try:
                sliceNode.SetSliceEdgeVisibility3D(True)
            except AttributeError:  # pragma: no cover - Slicer < 5.6
                logging.debug("GTReview: SetSliceEdgeVisibility3D unavailable")
                sliceNode.SetSliceVisible(True)
        self.applySliceOrientation()
        self._installStrokeObservers()

    def _installStrokeObservers(self):
        """Notice when a paint stroke begins, in every slice view.

        A stroke is a mouse-down, a drag and a mouse-up; Slicer saves an undo
        state per brush stamp in between.  Remembering what the mask looked
        like at mouse-down is what lets Undo step back over the whole stroke
        instead of one stamp at a time.
        """
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        wanted = []
        for name in layoutManager.sliceViewNames():
            widget = layoutManager.sliceWidget(name)
            if widget is None:
                continue
            try:
                interactor = widget.sliceView().interactorStyle().GetInteractor()
            except Exception:  # noqa: BLE001 - view still being built
                continue
            if interactor is not None:
                wanted.append(interactor)
        known = [interactor for interactor, _tag in self._strokeObservers]
        for interactor in wanted:
            if interactor in known:
                continue
            # ahead of the effects, so the fingerprint is the pre-stroke mask
            tag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent, self._onStrokeStart, 100.0
            )
            self._strokeObservers.append((interactor, tag))

    def _onStrokeStart(self, caller=None, event=None):
        """Mouse-down in a slice view with any effect armed starts an edit.

        Not just Paint and Erase: a Sphere threshold drag is one edit too, and
        leaving it unmarked made the next Undo walk back past it.
        """
        del caller, event
        if self.editor is None or self.editor.activeEffect() is None:
            return
        self._markEdit()

    def _markEdit(self):
        """Remember the mask as it stands, so one Undo press steps over the
        edit about to happen.

        Every logical edit needs a mark, not only the ones made with a brush.
        onUndo pops the top of this stack unconditionally, so an edit that left
        no mark of its own was undone together with whatever earlier stroke had
        left the mark it found -- two operations for one Ctrl+Z.
        """
        fingerprint = self._maskFingerprint()
        if fingerprint is None:
            return
        self._redoTargets = []  # a new edit invalidates the redo trail
        self._strokeStarts.append(fingerprint)
        del self._strokeStarts[: -self.MAX_STROKE_MARKS]

    def applySliceOrientation(self):
        """Point the slice views at the mask's voxel grid, or back at anatomy.

        The anatomical preset each view started from is stashed on the node
        itself, because rotating turns its orientation into "Reformat" and the
        name would otherwise be gone by the time the box is unticked.
        """
        if self.alignSlicesCheckBox is None:
            return
        aligned = bool(self.alignSlicesCheckBox.checked)
        volume = self.logic.referenceVolumeNode if self.logic is not None else None
        if aligned and volume is None:
            return  # nothing loaded yet; the next case load re-applies this
        for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
            name = sliceNode.GetOrientation()
            if aligned:
                if name != "Reformat":
                    sliceNode.SetAttribute("GTReview.Orientation", name)
                sliceNode.RotateToVolumePlane(volume)
            else:
                restored = sliceNode.GetAttribute("GTReview.Orientation")
                if restored:
                    sliceNode.SetOrientation(restored)

    @guarded("Aligning the slice views")
    def onAlignSlicesChanged(self, checked=None):
        del checked
        if self._updatingGui:
            return
        self.applySliceOrientation()

    def applyViewLayers(self):
        if not slicer.app.layoutManager():
            return
        background = self._backgroundNode()
        foreground = self._foregroundNode()
        opacity = float(self.opacitySlider.value)
        # the contrast widget edits the display node of whatever is in Image
        if self.windowLevelWidget is not None:
            self.windowLevelWidget.setMRMLVolumeNode(background)
            self.windowLevelWidget.setEnabled(background is not None)
        for compositeNode in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            compositeNode.SetBackgroundVolumeID(background.GetID() if background else "")
            compositeNode.SetForegroundVolumeID(foreground.GetID() if foreground else "")
            compositeNode.SetForegroundOpacity(opacity)
            compositeNode.SetLabelVolumeID("")
        # a new layout can bring new slice nodes, so re-apply this here
        self.applySlicePlanes()

    @guarded("Changing the layout")
    def onLayoutChanged(self, index=None):
        del index
        self.applyLayout()
        self.applyViewLayers()

    def applyLayout(self):
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return
        value = self.layoutComboBox.itemData(self.layoutComboBox.currentIndex)
        if value is None:
            return
        layoutManager.setLayout(int(value))

    @guarded("Resetting the field of view")
    def onFitViews(self):
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return
        for name in layoutManager.sliceViewNames():
            widget = layoutManager.sliceWidget(name)
            if widget is None:
                continue
            logic = widget.sliceLogic()
            if logic is not None:
                logic.FitSliceToAll()

    # ------------------------------------------------------------ lesion slots
    def _startObservingSegmentation(self):
        self._stopObservingSegmentation()
        node = self.logic.segmentationNode
        if node is None:
            return
        segmentation = node.GetSegmentation()
        for eventName in (
            "SourceRepresentationModified",
            "RepresentationModified",
            "SegmentModified",
            "SegmentAdded",
            "SegmentRemoved",
        ):
            event = getattr(slicer.vtkSegmentation, eventName, None)
            if event is None:
                continue
            self.addObserver(segmentation, event, self.onSegmentationModified)

    def _stopObservingSegmentation(self):
        self.removeObservers(method=self.onSegmentationModified)

    def onSegmentationModified(self, caller=None, event=None):
        del caller, event
        if self.logic.segmentationNode is None:
            return
        self.unsavedChanges = True
        self.setLesionsStale(True)
        if self._refreshTimer is not None and self.autoRefreshCheckBox.checked:
            self._refreshTimer.start(self.LESION_REFRESH_DEBOUNCE_MS)
        self._updateCaseControls()

    @guarded("Refreshing the lesion list")
    def onDebouncedRefresh(self):
        if not self.lesionsStale or self.logic.segmentationNode is None:
            return
        if not self.autoRefreshCheckBox.checked:
            return
        self.refreshLesions()

    def setLesionsStale(self, stale):
        changed = self.lesionsStale != bool(stale)
        self.lesionsStale = bool(stale)
        if changed and hasattr(self, "saveAndNextButton"):
            self._updateEditingControls()
        if stale:
            self.staleLabel.text = "<b><font color='#c86400'>stale</font></b>"
            self.staleLabel.toolTip = "The mask changed since this list was computed."
        else:
            self.staleLabel.text = "<font color='#3c8c3c'>up to date</font>"
            self.staleLabel.toolTip = ""

    @guarded("Refreshing the lesion list")
    def onRefreshLesions(self):
        self.refreshLesions()

    def refreshLesions(self):
        if self._refreshing:
            return  # BusyCursor pumps events; a pending timer must not nest us
        self._refreshing = True
        try:
            self._refreshLesions()
        finally:
            self._refreshing = False

    def _refreshLesions(self):
        if self.logic.segmentationNode is None:
            self.componentMap = None
            self.lesionList = []
            self._populateLesionTable()
            return
        start = time.time()
        with BusyCursor("GTReview: finding lesions ..."):
            self.componentMap, self.lesionList = self.logic.computeLesions(
                minVoxels=int(self.minVoxelsSpinBox.value)
            )
        logging.info(
            "GTReview: %d lesions in %.2f s", len(self.lesionList), time.time() - start
        )
        self.setLesionsStale(False)
        self._populateLesionTable()
        self._adoptNewLesion()

    def _populateLesionTable(self):
        previous = self._updatingGui
        self._updatingGui = True
        try:
            self.lesionTable.setSortingEnabled(False)
            self.lesionTable.clearContents()
            self.lesionTable.setRowCount(len(self.lesionList))
            for row, lesion in enumerate(self.lesionList):
                values = (
                    lesion.index,
                    lesion.voxel_count,
                    round(float(lesion.volume_mm3), 1),
                )
                for column, value in enumerate(values):
                    item = qt.QTableWidgetItem()
                    item.setData(qt.Qt.DisplayRole, value)
                    if column == self.LESION_COLUMN_NUMBER:
                        item.setData(qt.Qt.UserRole, int(lesion.index))
                    self.lesionTable.setItem(row, column, item)
                done = qt.QTableWidgetItem()
                done.setFlags(
                    qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable
                )
                done.setCheckState(
                    qt.Qt.Checked if self._isLesionReviewed(lesion) else qt.Qt.Unchecked
                )
                done.setData(qt.Qt.UserRole, int(lesion.index))
                done.setToolTip("Tick once you are happy with this lesion.")
                self.lesionTable.setItem(row, self.LESION_COLUMN_DONE, done)
            self.lesionTable.setSortingEnabled(True)
            column, order = self._lesionSortState()
            self.lesionTable.sortItems(column, order)
            # after the sort, not before: see _installLesionDeleteButtons
            self._installLesionDeleteButtons()
            self._resizeLesionTable()
            self._restoreLesionSelection()
        finally:
            self._updatingGui = previous
        self._updateLesionSummary()
        self._updateEditingControls()

    def _updateLesionSummary(self):
        totalVoxels = sum(lesion.voxel_count for lesion in self.lesionList)
        totalVolume = sum(lesion.volume_mm3 for lesion in self.lesionList)
        done, _total = self._doneCount()
        text = "{} lesions  |  {} voxels  |  {:.1f} mm3".format(
            len(self.lesionList), totalVoxels, totalVolume
        )
        if self.lesionList:
            color = "#3c8c3c" if done == len(self.lesionList) else "gray"
            text += "  |  <font color='{}'>{}/{} done</font>".format(
                color, done, len(self.lesionList)
            )
        self.lesionSummaryLabel.text = text

    def _doneCount(self):
        total = len(self.lesionList)
        done = sum(1 for lesion in self.lesionList if self._isLesionReviewed(lesion))
        return done, total

    def _allLesionsDone(self):
        """Saving is only offered once every listed lesion is ticked Done.

        An empty list (no lesions left) counts as done: deleting every false
        positive is a legitimate review outcome.  A stale list does not count.
        """
        if self.logic.segmentationNode is None or self.lesionsStale:
            return False
        done, total = self._doneCount()
        return done == total

    def _caseReviewedSeeds(self):
        case = self.currentCase()
        if case is None:
            return None
        return self.reviewedSeeds.setdefault(case.directory, set())

    def _isLesionReviewed(self, lesion):
        """True when one of this case's ticked seeds falls inside *lesion*."""
        seeds = self._caseReviewedSeeds()
        if not seeds or self.componentMap is None:
            return False
        for seed in seeds:
            try:
                if int(self.componentMap[seed]) == int(lesion.index):
                    return True
            except (IndexError, ValueError):
                continue
        return False

    def _setLesionReviewed(self, lesion, reviewed):
        seeds = self._caseReviewedSeeds()
        if seeds is None:
            return
        if reviewed:
            seeds.add(tuple(int(v) for v in lesion.centroid_ijk))
            return
        for seed in list(seeds):
            try:
                if int(self.componentMap[seed]) == int(lesion.index):
                    seeds.discard(seed)
            except (IndexError, ValueError):
                continue

    @guarded("Marking the lesion")
    def onLesionItemChanged(self, item=None):
        if self._updatingGui or item is None or item.column() != self.LESION_COLUMN_DONE:
            return
        index = item.data(qt.Qt.UserRole)
        if index is None:
            return
        lesion = next((l for l in self.lesionList if l.index == int(index)), None)
        if lesion is None:
            return
        self._setLesionReviewed(lesion, item.checkState() == qt.Qt.Checked)
        self._updateLesionSummary()
        self._updateEditingControls()

    def _lesionSortState(self):
        """The header's current sort, so a refresh does not reset the order."""
        header = self.lesionTable.horizontalHeader()
        try:
            column = int(header.sortIndicatorSection())
            order = header.sortIndicatorOrder()
        except Exception:  # noqa: BLE001 - binding differences
            column, order = -1, qt.Qt.DescendingOrder
        if (column < 0 or column >= self.lesionTable.columnCount
                or column == self.LESION_DELETE_COLUMN):
            return self.LESION_COLUMN_VOXELS, qt.Qt.DescendingOrder  # largest first
        return column, order

    def _installLesionDeleteButtons(self):
        """(Re)create the per-row delete buttons for the current row order.

        QTableWidget sorting moves the *items* but leaves cell widgets in the
        cells they were put in, so a button created before a sort would end up
        next to a different lesion than the one it deletes.  Rebuilding from
        the row order every time is cheap (the table caps at a screenful) and
        removes the whole class of bug.
        """
        table = self.lesionTable
        icon = qt.QIcon(":/Icons/Delete.png")
        if icon.pixmap(16, 16).isNull():
            icon = slicer.app.style().standardIcon(qt.QStyle.SP_TrashIcon)
        for row in range(table.rowCount):
            item = table.item(row, 0)
            if item is None:
                continue
            index = int(item.data(qt.Qt.UserRole))
            button = qt.QToolButton()
            button.setIcon(icon)
            button.setIconSize(qt.QSize(14, 14))
            button.setAutoRaise(True)
            button.toolTip = (
                "Delete lesion {} -- removes every one of its voxels from both "
                "labels.\nUndoable with Ctrl+Z; nothing is written to disk until "
                "you save.".format(index)
            )
            button.connect("clicked()", functools.partial(self._onDeleteLesionRow, index))
            table.setCellWidget(row, self.LESION_DELETE_COLUMN, button)

    def _onDeleteLesionRow(self, lesionIndex):
        """Delete the lesion this row's button belongs to.

        The row is selected first with the slots running normally, so the
        highlight and the brush label follow the click before anything is
        removed -- and a delete that bails out still leaves the selection
        where the user just pointed.
        """
        for row in range(self.lesionTable.rowCount):
            item = self.lesionTable.item(row, 0)
            if item is not None and item.data(qt.Qt.UserRole) == lesionIndex:
                self.lesionTable.selectRow(row)
                break
        else:
            return
        self.onDeleteLesion()

    def onLesionSortChanged(self, column, order):
        """Bounce a sort on the delete column back to the voxel-count sort."""
        del order
        if int(column) != self.LESION_DELETE_COLUMN or self._updatingGui:
            return
        self.lesionTable.sortItems(self.LESION_COLUMN_VOXELS, qt.Qt.DescendingOrder)

    def _resizeLesionTable(self):
        """Height the table to its contents, up to LESION_TABLE_MAX_ROWS."""
        table = self.lesionTable
        rows = table.rowCount
        shown = max(1, min(rows, self.LESION_TABLE_MAX_ROWS))
        rowHeight = table.verticalHeader().defaultSectionSize
        headerHeight = table.horizontalHeader().height
        height = headerHeight + shown * rowHeight + 2 * table.frameWidth
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)
        table.setVerticalScrollBarPolicy(
            qt.Qt.ScrollBarAsNeeded if rows > self.LESION_TABLE_MAX_ROWS
            else qt.Qt.ScrollBarAlwaysOff
        )

    def _selectRowForLesionIndex(self, index):
        """Select the table row of lesion *index* without firing the slots."""
        previous = self._updatingGui
        self._updatingGui = True
        try:
            for row in range(self.lesionTable.rowCount):
                item = self.lesionTable.item(row, 0)
                if item is not None and item.data(qt.Qt.UserRole) == index:
                    self.lesionTable.selectRow(row)
                    return True
        finally:
            self._updatingGui = previous
        return False

    def _clearLesionSelection(self):
        """Forget which lesion is selected (index *and* identity seed).

        Deliberately does NOT end new-lesion mode: the table rebuild calls
        this on every refresh while the new lesion is still being painted.
        """
        self.clearLesionHighlight()
        self.selectedLesionIndex = None
        self.selectedLesionSeed = None
        table = getattr(self, "lesionTable", None)
        if table is not None:
            table.clearSelection()

    def _lesionIndexAtSeed(self, seed):
        """Index of the lesion that currently owns voxel *seed*, else None."""
        if seed is None or self.componentMap is None:
            return None
        try:
            i, j, k = (int(value) for value in seed)
        except (TypeError, ValueError):
            return None
        shape = self.componentMap.shape
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            return None
        index = int(self.componentMap[i, j, k])
        return index if index > 0 else None

    def _restoreLesionSelection(self):
        """Re-select the previously selected lesion after a table rebuild.

        ``Lesion.index`` is an ordinal handed out *after* sorting by voxel
        count, so it is a position and not an identity: any edit that changes
        the relative lesion sizes renumbers every lesion.  Re-selecting by that
        number would therefore silently point the editing actions at a
        different lesion than the one the user picked and jumped to.

        The selection is instead keyed on a seed voxel known to lie inside the
        selected lesion.  After a refresh the lesion that still owns that voxel
        is the same lesion, whatever its new number.  If nothing owns it any
        more (erased, or the case was unloaded) the selection is dropped so the
        editing actions refuse to run rather than hit the wrong lesion.
        """
        index = self._lesionIndexAtSeed(self.selectedLesionSeed)
        if index is None:
            self._clearLesionSelection()
            return
        for row in range(self.lesionTable.rowCount):
            item = self.lesionTable.item(row, 0)
            if item is None:
                continue
            if item.data(qt.Qt.UserRole) == index:
                self.selectedLesionIndex = index
                self.lesionTable.selectRow(row)
                # re-pin the seed to the refreshed lesion's own centre voxel
                for lesion in self.lesionList:
                    if lesion.index == index:
                        self.selectedLesionSeed = tuple(
                            int(value) for value in lesion.centroid_ijk
                        )
                        break
                return
        self._clearLesionSelection()

    def selectedLesion(self):
        index = None
        selectionModel = self.lesionTable.selectionModel()
        rows = selectionModel.selectedRows() if selectionModel else []
        if rows:
            item = self.lesionTable.item(rows[0].row(), 0)
            if item is not None:
                index = item.data(qt.Qt.UserRole)
        if index is None:
            index = self.selectedLesionIndex
        if index is None:
            return None
        for lesion in self.lesionList:
            if lesion.index == int(index):
                return lesion
        return None

    @guarded("Selecting a lesion")
    def onLesionSelectionChanged(self):
        if self._updatingGui:
            return
        selectionModel = self.lesionTable.selectionModel()
        rows = selectionModel.selectedRows() if selectionModel else []
        if not rows:
            # a real deselect (Ctrl+click) must lock the brush again, so do not
            # fall back to the remembered index here
            self.selectedLesionIndex = None
            self.selectedLesionSeed = None
            self._updateEditingControls()
            return
        lesion = self.selectedLesion()
        if lesion is None:
            return
        self._newLesion = None  # picking a row ends new-lesion mode
        self.selectedLesionIndex = lesion.index
        self.selectedLesionSeed = tuple(int(value) for value in lesion.centroid_ijk)
        self._selectSegmentForLabel(lesion.label)
        self._updateEditingControls()
        self.jumpToLesion(lesion)

    @guarded("Selecting the lesion")
    def onLesionClicked(self, item=None):
        """Re-clicking the selected row re-flashes it."""
        del item
        if self._updatingGui:
            return
        lesion = self.selectedLesion()
        if lesion is not None:
            self.jumpToLesion(lesion)

    @staticmethod
    def _historyIcon(kind, size=18, scale=6):
        """Draw the undo / redo / reset / plus glyph.

        Rendered at *scale* times the logical size so it stays sharp on a hidpi
        panel, in the palette's button colour so it follows the Slicer theme.
        "undo" and "redo" are half arcs mirrored about the vertical, "reset" is
        a near-full circle and "plus" a cross -- drawn in one hand, at one
        stroke weight, so the buttons read as one family.
        """
        box = int(size * scale)
        pixmap = qt.QPixmap(box, box)
        pixmap.fill(qt.QColor(0, 0, 0, 0))
        painter = qt.QPainter()
        painter.begin(pixmap)
        try:
            painter.setRenderHint(qt.QPainter.Antialiasing, True)
            color = slicer.app.palette().color(qt.QPalette.ButtonText)
            pen = qt.QPen(color)
            pen.setWidthF(box * 0.12)
            pen.setCapStyle(qt.Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(qt.QBrush())
            if kind == "plus":
                arm = box * 0.30
                centre = box / 2.0
                painter.drawLine(qt.QPointF(centre - arm, centre),
                                 qt.QPointF(centre + arm, centre))
                painter.drawLine(qt.QPointF(centre, centre - arm),
                                 qt.QPointF(centre, centre + arm))
            else:
                radius = box * 0.30
                centreX = box / 2.0
                centreY = box / 2.0 + (box * 0.09 if kind != "reset" else 0.0)
                # the arcs stop short of the horizontal so the head has room: at
                # 20 px a head any smaller than this disappears into the stroke
                start, sweep = {
                    "undo": (10.0, 160.0),
                    "redo": (170.0, -160.0),
                }.get(kind, (250.0, 300.0))
                painter.drawArc(
                    qt.QRectF(centreX - radius, centreY - radius, 2 * radius, 2 * radius),
                    int(round(start * 16)), int(round(sweep * 16)),
                )
                # a filled head on the tangent at the far end of the arc
                end = math.radians(start + sweep)
                travel = 1.0 if sweep > 0 else -1.0
                tipX = centreX + radius * math.cos(end)
                tipY = centreY - radius * math.sin(end)
                dx = -math.sin(end) * travel
                dy = -math.cos(end) * travel
                head = box * 0.34
                path = qt.QPainterPath()
                path.moveTo(qt.QPointF(tipX + dx * head * 0.85, tipY + dy * head * 0.85))
                path.lineTo(qt.QPointF(tipX - dx * head * 0.15 - dy * head * 0.55,
                                       tipY - dy * head * 0.15 + dx * head * 0.55))
                path.lineTo(qt.QPointF(tipX - dx * head * 0.15 + dy * head * 0.55,
                                       tipY - dy * head * 0.15 - dx * head * 0.55))
                path.closeSubpath()
                painter.fillPath(path, qt.QBrush(color))
        finally:
            painter.end()
        return qt.QIcon(pixmap)

    @staticmethod
    def _labelIcon(value, size=12):
        """A filled swatch in the label's own colour, for the combo boxes.

        Label 0 is the background, which has no colour of its own: it gets an
        empty box with a slash through it, the way a "no fill" swatch is drawn.
        """
        if int(value) == GTReviewWidget.ACTIVE_LABEL_BACKGROUND:
            pixmap = qt.QPixmap(size, size)
            pixmap.fill(qt.QColor(0, 0, 0, 0))
            painter = qt.QPainter()
            painter.begin(pixmap)
            try:
                painter.setRenderHint(qt.QPainter.Antialiasing, True)
                pen = qt.QPen(slicer.app.palette().color(qt.QPalette.ButtonText))
                pen.setWidthF(1.2)
                painter.setPen(pen)
                painter.drawRect(qt.QRectF(0.75, 0.75, size - 1.5, size - 1.5))
                painter.drawLine(qt.QPointF(1.5, size - 1.5), qt.QPointF(size - 1.5, 1.5))
            finally:
                painter.end()
            return qt.QIcon(pixmap)
        red, green, blue = colorForLabelValue(int(value))
        pixmap = qt.QPixmap(size, size)
        pixmap.fill(qt.QColor.fromRgbF(red, green, blue))
        return qt.QIcon(pixmap)

    @guarded("Choosing the active label")
    def onActiveLabelChanged(self, index=None):
        del index
        if self._updatingGui or self.activeLabelComboBox is None:
            return
        value = self.activeLabelComboBox.itemData(self.activeLabelComboBox.currentIndex)
        if value is None:
            return
        if int(value) == self.ACTIVE_LABEL_BACKGROUND:
            self.onActivateEffect("Erase")
            return
        self._selectSegmentForLabel(int(value))
        # Coming back from Background: leaving Erase armed would keep taking
        # voxels away while the box says a label is being painted.  The effect
        # is switched directly rather than through onActivateEffect, which
        # re-selects the segment of the currently selected lesion and would
        # undo the label just picked here.
        effect = self.editor.activeEffect() if self.editor is not None else None
        if effect is not None and effect.name == "Erase":
            self.editor.setActiveEffectByName("Paint")
            self._syncActiveLabelComboBox()

    def onEditorSegmentChanged(self, segmentId=None):
        """Mirror the editor's segment list back into the Active label box."""
        del segmentId
        if self._updatingGui:
            return
        self._syncActiveLabelComboBox()

    def _syncActiveLabelComboBox(self):
        if self.activeLabelComboBox is None or self.editor is None:
            return
        effect = self.editor.activeEffect()
        erasing = effect is not None and effect.name == "Erase"
        current = self.editor.currentSegmentID()
        if not erasing and not current:
            return
        previous = self._updatingGui
        self._updatingGui = True
        try:
            # Erase IS painting the background, however it was reached -- the
            # Erase button, the 2 key or this box -- so the box says so
            wanted = self.ACTIVE_LABEL_BACKGROUND if erasing else next(
                (value for value in LABEL_NAMES
                 if self.logic.segmentIdForLabelValue(value) == current), None)
            if wanted is None:
                return
            index = self.activeLabelComboBox.findData(int(wanted))
            if index >= 0:
                self.activeLabelComboBox.currentIndex = index
        finally:
            self._updatingGui = previous

    @guarded("Choosing what may be painted over")
    def onPaintOverChanged(self, index=None):
        del index
        if self._updatingGui:
            return
        self.applyPaintOver()

    def applyPaintOver(self):
        """Translate the Paint over choice into the editor node's mask mode.

        "Background only" and "Only <label>" are Slicer's EditAllowedOutside-
        AllSegments and EditAllowedInsideSingleSegment.  The overwrite mode
        stays OverwriteAllSegments throughout: the two labels are mutually
        exclusive in the saved NIfTI, so painting one has to take the voxel
        away from the other -- that is not what Paint over is asking about.
        """
        node = self.segmentEditorNode
        if node is None or self.paintOverComboBox is None or self.logic is None:
            return
        choice = self.paintOverComboBox.itemData(self.paintOverComboBox.currentIndex)
        segmentations = slicer.vtkMRMLSegmentationNode
        try:
            if choice == self.PAINT_OVER_BACKGROUND:
                node.SetMaskMode(segmentations.EditAllowedOutsideAllSegments)
                node.SetMaskSegmentID("")
            elif choice is not None and int(choice) in LABEL_NAMES:
                segmentId = self.logic.segmentIdForLabelValue(int(choice))
                if not segmentId:
                    return  # no case loaded yet; _attachEditor re-applies this
                # the ID first: the node refuses EditAllowedInsideSingleSegment
                # while MaskSegmentID is still empty and silently stays on the
                # previous mode
                node.SetMaskSegmentID(segmentId)
                node.SetMaskMode(segmentations.EditAllowedInsideSingleSegment)
            else:
                node.SetMaskMode(segmentations.EditAllowedEverywhere)
                node.SetMaskSegmentID("")
            node.SetOverwriteMode(
                slicer.vtkMRMLSegmentEditorNode.OverwriteAllSegments
            )
        except Exception:  # noqa: BLE001 - older node API
            logging.debug("GTReview: setting the paint-over mask failed", exc_info=True)

    def _selectSegmentForLabel(self, labelValue):
        if self.editor is None:
            return
        segmentId = self.logic.segmentIdForLabelValue(labelValue)
        if segmentId:
            self.editor.setCurrentSegmentID(segmentId)
            # setCurrentSegmentID does not emit currentSegmentIDChanged, so
            # every programmatic path -- selecting a lesion above all -- has to
            # move the combo itself or the two disagree on screen
            self._syncActiveLabelComboBox()

    @guarded("Jumping to the lesion")
    def onJumpToLesion(self):
        lesion = self.selectedLesion()
        if lesion is None:
            slicer.util.showStatusMessage("GTReview: select a lesion first.", 2000)
            return
        self.jumpToLesion(lesion)

    def jumpToLesion(self, lesion):
        if self.logic.referenceVolumeNode is None:
            return
        ras = self.logic.centroidToRAS(lesion.centroid_ijk)
        if not slicer.app.layoutManager():
            return
        slicer.modules.markups.logic().JumpSlicesToLocation(ras[0], ras[1], ras[2], True)
        self.flashLesion(lesion)

    # ------------------------------------------------------------ lesion flash
    def flashLesion(self, lesion):
        """Blink the selected lesion so it is obvious which one is meant.

        Drawn as a throwaway segmentation node of its own rather than by
        touching the reviewed segmentation: nothing here may land on the undo
        stack, mark the case dirty or reach the saved mask.
        """
        self.clearLesionHighlight()
        if self.componentMap is None or self.logic.maskGeometry is None:
            return
        if not self.flashLesionsCheckBox.checked:
            return
        try:
            mask = lesions.lesion_mask(self.componentMap, lesion.index)
            # Crop to the lesion's own bounding box and shift the origin to
            # match.  A full-volume highlight costs ~0.6 s on a 94 M-voxel case
            # no matter how small the lesion is, which is far too slow for a
            # blink fired on every click in the lesion table.
            (i0, i1), (j0, j1), (k0, k1) = lesion.bbox_ijk
            box = np.ascontiguousarray(
                mask[i0:i1, j0:j1, k0:k1].astype(np.uint8, copy=False)
            )
            geometry = self._croppedGeometry(self.logic.maskGeometry, (i0, j0, k0), box.shape)
            labelNode = createLabelVolumeNode(box, geometry, "GTReviewFlashLabel")
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "GTReviewFlash"
            )
            node.SetSaveWithScene(False)
            node.CreateDefaultDisplayNodes()
            node.SetReferenceImageGeometryParameterFromVolumeNode(
                self.logic.referenceVolumeNode
            )
            segmentationsLogicCall("ImportLabelmapToSegmentationNode", labelNode, node)
            slicer.mrmlScene.RemoveNode(labelNode)  # only needed for the import
            displayNode = node.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetAllSegmentsVisibility(True)
                displayNode.SetVisibility2DFill(True)
                displayNode.SetVisibility2DOutline(True)
                displayNode.SetOpacity2DFill(0.9)
                displayNode.SetVisibility3D(False)
            for segmentId in self._segmentIdsOf(node):
                node.GetSegmentation().GetSegment(segmentId).SetColor(1.0, 1.0, 1.0)
        except Exception:  # noqa: BLE001 - a blink must never break the jump
            logging.debug("GTReview: building the lesion highlight failed", exc_info=True)
            self.clearLesionHighlight()
            return
        self._flashNode = node
        self._flashesLeft = self.FLASH_BLINKS * 2
        self._flashTimer.start(self.FLASH_INTERVAL_MS)

    @staticmethod
    def _croppedGeometry(geometry, start_ijk, shape):
        """*geometry* restricted to a sub-box starting at ``start_ijk``.

        The direction cosines and spacing are unchanged; only the origin moves,
        by the physical offset of the first voxel of the box.
        """
        direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
        spacing = np.asarray(geometry.spacing, dtype=np.float64)
        start = np.asarray(start_ijk, dtype=np.float64)
        origin = np.asarray(geometry.origin, dtype=np.float64) + direction.dot(spacing * start)
        return maskio.MaskGeometry(
            origin=tuple(float(v) for v in origin),
            spacing=tuple(float(v) for v in geometry.spacing),
            direction=tuple(float(v) for v in geometry.direction),
            size=tuple(int(n) for n in shape),
        )

    @staticmethod
    def _segmentIdsOf(segmentationNode):
        ids = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(ids)
        return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]

    def _onFlashTick(self):
        node = self._flashNode
        if node is None or self._flashesLeft <= 0:
            self.clearLesionHighlight()
            return
        displayNode = node.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetVisibility(not displayNode.GetVisibility())
        self._flashesLeft -= 1
        if self._flashesLeft <= 0:
            self.clearLesionHighlight()

    def clearLesionHighlight(self):
        """Stop any blink in progress and drop its throwaway node."""
        if self._flashTimer is not None:
            self._flashTimer.stop()
        self._flashesLeft = 0
        node, self._flashNode = self._flashNode, None
        if node is not None:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception:  # noqa: BLE001 - already gone with the scene
                logging.debug("GTReview: removing the highlight failed", exc_info=True)

    # ----------------------------------------------------------- editing slots
    def _requireLesion(self):
        if self.logic.segmentationNode is None:
            slicer.util.showStatusMessage("GTReview: load a case first.", 3000)
            return None
        if self._refreshing:
            slicer.util.showStatusMessage("GTReview: the lesion list is being recomputed.", 2000)
            return None
        if self.lesionsStale:
            hadSelection = self.selectedLesionSeed is not None
            self.refreshLesions()
            if hadSelection and self.selectedLesionIndex is None:
                slicer.util.showStatusMessage(
                    "GTReview: the mask changed and that lesion no longer exists -- "
                    "pick one in the refreshed table.", 4000,
                )
                return None
        lesion = self.selectedLesion()
        if lesion is None:
            slicer.util.showStatusMessage("GTReview: select a lesion in the table first.", 3000)
            return None
        return lesion

    @guarded("Deleting the lesion")
    def onDeleteLesion(self):
        lesion = self._requireLesion()
        if lesion is None:
            return
        # Both entry points are one gesture away from losing a lesion -- the
        # Delete key, and a trash button sitting next to the Done checkbox --
        # so the count and the label are spelled out before anything goes.
        if not slicer.util.confirmYesNoDisplay(
            "Delete lesion {} ({})?\n\n"
            "{} voxels, {:.1f} mm3, removed from both labels.\n"
            "This can be undone (Ctrl+Z), and nothing is written to disk "
            "until you save.".format(
                lesion.index,
                nameForLabelValue(int(lesion.label)),
                lesion.voxel_count,
                float(lesion.volume_mm3),
            ),
            windowTitle="GTReview",
        ):
            return
        mask = lesions.lesion_mask(self.componentMap, lesion.index)
        self._markEdit()
        with BusyCursor("GTReview: deleting lesion {} ...".format(lesion.index)):
            self.logic.deleteLesionVoxels(mask)
        # the deleted lesion is gone and the rest get renumbered
        self._clearLesionSelection()
        self.unsavedChanges = True
        self.refreshLesions()
        self._updateCaseControls()

    @guarded("Deleting the review")
    def onDeleteReview(self):
        """Remove this case's reviewed_seg file and reload from the original.

        Unlike everything else in this panel, this leaves the undo stack behind
        and touches the dataset itself, so it asks first and names the file.
        """
        case = self.currentCase()
        if case is None:
            slicer.util.showStatusMessage("GTReview: load a case first.", 3000)
            return
        if not case.is_reviewed:
            slicer.util.showStatusMessage(
                "GTReview: {} has no saved review.".format(case.case_id), 3000
            )
            return
        path = case.reviewed_path
        # The usual unsaved-edits dialog must NOT be used here.  Its Save
        # branch writes case.reviewed_path -- the very file this method is
        # about to remove -- so answering Save reported success, and the next
        # statement deleted the work.  The edits are named in the one prompt
        # instead, and they go with the file.
        pending = ""
        if self.unsavedChanges:
            pending = (
                "\n\nThe unsaved edits in the editor are discarded with it; "
                "saving them would only write the file being deleted."
            )
        if not slicer.util.confirmYesNoDisplay(
            "Delete the saved review of {}?\n\n{}\n\n"
            "The file is removed from disk and the case reopens from its "
            "original mask.  This cannot be undone.{}".format(
                case.case_id, path, pending
            ),
            windowTitle="GTReview",
        ):
            return
        try:
            os.remove(path)
        except OSError as error:
            slicer.util.errorDisplay(
                "Could not delete {}:\n{}".format(path, error), windowTitle="GTReview"
            )
            return
        self.unsavedChanges = False
        # The Done ticks were recorded against the review that just went away,
        # so leaving them would re-enable Save & next case on a case nobody has
        # looked at since.
        self.reviewedSeeds.pop(case.directory, None)
        # is_reviewed re-checks the disk, so the case object needs no patching;
        # the tick in the case list and the skip-reviewed filter do.  Re-running
        # the filter can move the case, so it is found again by id rather than
        # trusting the old index.
        caseId = case.case_id
        self._applyCaseFilter()
        for index, candidate in enumerate(self.filteredCases):
            if candidate.case_id == caseId:
                self.currentCaseIndex = index
                break
        self._populateCaseComboBox()
        self.loadCurrentCase()
        slicer.util.showStatusMessage(
            "GTReview: deleted the review of {}.".format(case.case_id), 4000
        )

    @guarded("Deleting the label")
    def onDeleteLabel(self):
        lesion = self._requireLesion()
        if lesion is None:
            return
        label = int(lesion.label)
        count = sum(1 for other in self.lesionList if int(other.label) == label)
        voxels = int(self.logic.labelMaskIJK(label).sum())
        if not voxels:
            slicer.util.showStatusMessage(
                "GTReview: label {} has no voxels.".format(label), 2000
            )
            return
        if not slicer.util.confirmYesNoDisplay(
            "Delete ALL of label {} in this case?\n\n"
            "That is {} voxels across {} lesion(s).  The other label is kept.\n"
            "This can be undone (Ctrl+Z).".format(
                nameForLabelValue(label), voxels, count
            ),
            windowTitle="GTReview",
        ):
            return
        self._markEdit()
        with BusyCursor("GTReview: deleting label {} ...".format(label)):
            removed = self.logic.deleteLabelVoxels(label)
        self._clearLesionSelection()
        self.unsavedChanges = True
        self.refreshLesions()
        self._updateCaseControls()
        self._updateEditingControls()
        slicer.util.showStatusMessage(
            "GTReview: removed {} voxels of label {}.".format(removed, label), 4000
        )

    @guarded("Flipping the lesion label")
    def onFlipLesionLabel(self):
        lesion = self._requireLesion()
        if lesion is None:
            return
        others = [v for v in sorted(LABEL_NAMES) if v != int(lesion.label)]
        if int(lesion.label) not in LABEL_NAMES or not others:
            slicer.util.showStatusMessage(
                "GTReview: lesion {} has label {}, only {} can be flipped.".format(
                    lesion.index, lesion.label, " / ".join(str(v) for v in sorted(LABEL_NAMES))
                ),
                3000,
            )
            return
        target = others[0]
        mask = lesions.lesion_mask(self.componentMap, lesion.index)
        with BusyCursor("GTReview: lesion {} -> label {} ...".format(lesion.index, target)):
            self._markEdit()
            self.logic.changeLesionLabel(mask, target)
        self.unsavedChanges = True
        self.refreshLesions()
        # the rebuild restores the selection silently; re-sync the brush label
        self._selectSegmentForLabel(target)
        self._updateCaseControls()
        self._updateEditingControls()

    def _relaySphereThresholdResult(self):
        """Echo the Sphere threshold effect's last apply in the status bar."""
        if self.editor is None:
            return
        effect = self.editor.effectByName(SPHERE_THRESHOLD_EFFECT)
        if effect is None or not hasattr(effect, "self"):
            return
        try:
            scripted = effect.self()
        except Exception:  # noqa: BLE001
            return
        result = getattr(scripted, "lastApplied", None)
        if not result:
            return
        scripted.lastApplied = None
        voxels, lower, upper, radius = result
        self.unsavedChanges = True
        self._updateCaseControls()

    @guarded("New lesion")
    def onNewLesionToggled(self, checked=None):
        if self._updatingGui:
            return
        if not checked:
            self._cancelNewLesion()
            return
        if self.logic.segmentationNode is None or self.editor is None or self._refreshing:
            slicer.util.showStatusMessage(
                "GTReview: load a case first." if self.logic.segmentationNode is None
                else "GTReview: the lesion list is being recomputed.", 3000
            )
            self._updateEditingControls()
            return
        if self.lesionsStale:
            self.refreshLesions()
        value = self._activeLabelValue()
        # Lesion numbers are re-assigned on every recount, so the lesions that
        # exist NOW are remembered by a seed voxel each (centroids always lie
        # inside their component) plus their size; see _adoptNewLesion.
        self._newLesion = {
            "label": value,
            "before": {
                tuple(int(v) for v in lesion.centroid_ijk): int(lesion.voxel_count)
                for lesion in self.lesionList
            },
        }
        self._clearLesionSelection()
        self._updateEditingControls()  # unlocks the brush for the new mode
        self._selectSegmentForLabel(value)
        self.onActivateEffect("Paint")
        slicer.util.showStatusMessage(
            "GTReview: paint the new lesion as {}; it is numbered and selected "
            "once the list refreshes.  Esc cancels.".format(nameForLabelValue(value)),
            6000,
        )

    def _activeLabelValue(self):
        """The label a new lesion should be painted with.

        The Active label box can be sitting on Background, which is the Erase
        tool and cannot start a lesion; fall back to the first real label.
        """
        value = None
        if self.activeLabelComboBox is not None:
            value = self.activeLabelComboBox.itemData(
                self.activeLabelComboBox.currentIndex
            )
        if value is None or int(value) not in LABEL_NAMES:
            return min(LABEL_NAMES)
        return int(value)

    def _cancelNewLesion(self):
        if self._newLesion is None:
            return
        self._newLesion = None
        if self.newLesionButton is not None:
            previous = self._updatingGui
            self._updatingGui = True
            try:
                self.newLesionButton.checked = False
            finally:
                self._updatingGui = previous
        self._updateEditingControls()  # locks the brush again unless a row is selected

    def _adoptNewLesion(self):
        """After a recount in new-lesion mode, select what was just painted."""
        state = self._newLesion
        if state is None or self.componentMap is None:
            return
        matched = {}
        for seed, count in state["before"].items():
            index = self._lesionIndexAtSeed(seed)
            if index is not None:
                matched[index] = matched.get(index, 0) + count  # merged lesions add up
        unmatched = [lesion for lesion in self.lesionList if lesion.index not in matched]
        chosen, merged = None, False
        if unmatched:
            chosen = max(unmatched, key=lambda lesion: (lesion.voxel_count, -lesion.index))
        else:
            grown = [
                (lesion.voxel_count - matched[lesion.index], lesion)
                for lesion in self.lesionList
                if lesion.index in matched and lesion.voxel_count > matched[lesion.index]
            ]
            if grown:
                chosen, merged = max(grown, key=lambda pair: pair[0])[1], True
        if chosen is None:
            return  # nothing painted yet (or below "Min voxels"): stay in the mode
        self._newLesion = None
        if self.newLesionButton is not None:
            previous = self._updatingGui
            self._updatingGui = True
            try:
                self.newLesionButton.checked = False
            finally:
                self._updatingGui = previous
        self.selectedLesionIndex = chosen.index
        self.selectedLesionSeed = tuple(int(v) for v in chosen.centroid_ijk)
        self._selectRowForLesionIndex(chosen.index)
        self._selectSegmentForLabel(chosen.label)
        self._updateEditingControls()
        if merged:
            slicer.util.showStatusMessage(
                "GTReview: the stroke touched lesion {} and merged into it.".format(chosen.index),
                5000,
            )
        else:
            slicer.util.showStatusMessage(
                "GTReview: new lesion #{} ({} voxels) selected.".format(
                    chosen.index, chosen.voxel_count
                ),
                4000,
            )

    @guarded("Activating a brush effect")
    def onActivateEffect(self, name):
        """Gated entry point for the digit keys and the new-lesion flow."""
        if self.editor is None or self.logic.segmentationNode is None:
            return
        if not self._editingAllowed():
            slicer.util.showStatusMessage(
                "GTReview: select a lesion, or start a new one, before editing.", 3000
            )
            return
        if self._newLesion is not None:
            self._selectSegmentForLabel(self._newLesion["label"])
        else:
            lesion = self.selectedLesion()
            if lesion is not None:
                self._selectSegmentForLabel(lesion.label)
        self.editor.setActiveEffectByName(name)
        # the segment was selected above, before the effect changed, so the
        # sync it triggered could not know Erase was about to become active
        self._syncActiveLabelComboBox()
        self._applyImmediatePaint()
        self._initialiseBrush()
        self._updateEditingControls()

    def _applyImmediatePaint(self):
        """Commit each brush stamp as it is drawn, not on mouse release.

        Slicer's Paint and Erase default to "delayed paint": the drag only
        leaves outlined circles behind and the labelmap is written once, when
        the button comes up.  Reviewers reading the result as they go want the
        segmentation to follow the cursor, the way ITK-SNAP paints, so the
        outlines are traded for a live fill.  Only the two C++ brush effects
        have the setting -- the scripted Sphere threshold has its own preview
        and is skipped.
        """
        if self.editor is None:
            return
        effect = self.editor.activeEffect()
        if effect is None or not hasattr(effect, "setDelayedPaint"):
            return
        try:
            effect.setDelayedPaint(False)
        except Exception:  # noqa: BLE001 - effect without the property
            logging.debug("GTReview: setDelayedPaint unavailable", exc_info=True)

    def _initialiseBrush(self):
        """A sensible absolute brush the first time a brush is activated."""
        if self._brushInitialised or self.editor is None:
            return
        effect = self.editor.activeEffect()
        if effect is None or effect.name not in ("Paint", "Erase"):
            return
        try:
            # common (unprefixed) parameters are shared by Paint and Erase and
            # drive the brush slider; setParameter would shadow them per effect
            effect.setCommonParameter("BrushAbsoluteDiameter", float(self.BRUSH_MM))
            self._brushInitialised = True
        except Exception:  # noqa: BLE001 - parameter API differs
            logging.debug("GTReview: setting the brush size failed", exc_info=True)

    def _constrainBrush(self):
        """Keep the brush absolute and bounded to [1, 20] mm in 1 mm steps.

        Run on every Paint/Erase activation because the effect rebuilds its
        options widgets and re-reads these common parameters each time.
        """
        if self.editor is None:
            return
        effect = self.editor.activeEffect()
        if effect is None or effect.name not in ("Paint", "Erase"):
            return
        try:
            effect.setCommonParameter("BrushDiameterIsRelative", 0)
            effect.setCommonParameter("BrushMinimumAbsoluteDiameter", float(self.BRUSH_MIN_MM))
            effect.setCommonParameter("BrushMaximumAbsoluteDiameter", float(self.BRUSH_MAX_MM))
            current = float(effect.doubleParameter("BrushAbsoluteDiameter"))
            clamped = min(max(current, self.BRUSH_MIN_MM), self.BRUSH_MAX_MM)
            if abs(clamped - current) > 1e-9:
                effect.setCommonParameter("BrushAbsoluteDiameter", clamped)
        except Exception:  # noqa: BLE001
            logging.debug("GTReview: bounding the brush failed", exc_info=True)
        # tighten the visible slider (step 1 mm, no decimals) and hide the
        # absolute/relative toggle so the brush cannot leave absolute mode
        root = self._effectOptionsRoot()
        options = root.findChild(qt.QWidget, "EffectsOptionsFrame")
        if options is None:
            return
        for slider in options.findChildren(qt.QWidget):
            if slider.className() not in ("qMRMLSliderWidget", "ctkSliderWidget"):
                continue
            if not slider.isVisibleTo(root):
                continue
            try:
                if "mm" not in str(slider.suffix):
                    continue
                slider.singleStep = self.BRUSH_STEP_MM
                slider.pageStep = self.BRUSH_STEP_MM
                slider.decimals = 0
            except Exception:  # noqa: BLE001 - property differences
                logging.debug("GTReview: tuning the brush slider failed", exc_info=True)
        for button in options.findChildren(qt.QToolButton):
            if str(button.text).strip().lower() in ("absolute", "relative"):
                button.setVisible(False)

    @guarded("Stopping the edit")
    def onStopEditing(self):
        """Esc: drop the active effect and leave new-lesion mode."""
        if self._newLesion is not None:
            self._cancelNewLesion()
        if self.editor is not None and self.editor.activeEffect() is not None:
            self.editor.setActiveEffect(None)
        self._updateEditingControls()

    @guarded("Undo")
    def onUndo(self):
        if self.editor is not None:
            here = self._maskFingerprint()
            # A mark whose edit changed nothing (a bare click that painted no
            # voxel, a cancelled drag) is already the current state; using it
            # as a target would step past a real edit looking for it.
            target = None
            while self._strokeStarts:
                candidate = self._strokeStarts.pop()
                if candidate != here:
                    target = candidate
                    break
            self._stepHistory(self.editor.undo, target)
            if target is not None and here is not None:
                self._redoTargets.append(here)
            self.setLesionsStale(True)
            if self.autoRefreshCheckBox.checked:
                self.refreshLesions()

    @guarded("Redo")
    def onRedo(self):
        if self.editor is not None:
            target = self._redoTargets.pop() if self._redoTargets else None
            here = self._maskFingerprint()
            self._stepHistory(self.editor.redo, target)
            if target is not None and here is not None:
                self._strokeStarts.append(here)
            self.setLesionsStale(True)
            if self.autoRefreshCheckBox.checked:
                self.refreshLesions()

    #: how many identical history states one Undo/Redo press will step over
    HISTORY_SKIP_LIMIT = 4
    #: how far Undo will walk back looking for the start of a stroke
    HISTORY_STROKE_LIMIT = 60
    #: how many stroke starts are remembered
    MAX_STROKE_MARKS = 60

    def _maskFingerprint(self):
        """Cheap content signature of the mask, to spot a no-op history step."""
        node = self.logic.segmentationNode if self.logic is not None else None
        if node is None:
            return None
        signature = []
        for segmentId in self.logic.segmentIds():
            try:
                array = slicer.util.arrayFromSegmentBinaryLabelmap(node, segmentId)
            except Exception:  # noqa: BLE001 - segment without a labelmap yet
                signature.append(None)
                continue
            if array is None:
                signature.append(None)
            else:
                signature.append((array.shape, int(np.count_nonzero(array))))
        return tuple(signature)

    def _stepHistory(self, step, target=None):
        """Walk the history until the mask really moves, or *target* is reached.

        Two problems are being papered over here, both from painting without
        Slicer's delayed paint.  Slicer calls paintApply once more when the
        mouse comes up and paintApply always saves a state first, so the top of
        the stack is a duplicate of the current mask and a single step looks
        like it did nothing.  And a stroke is not one state but one per brush
        stamp, so stepping once would rub out a few voxels of a long stroke.

        With *target* -- the fingerprint taken at mouse-down -- the walk
        continues until the mask matches it again, which undoes the stroke as a
        unit.  Without one, it only steps past states that change nothing.
        """
        before = self._maskFingerprint()
        if before is None:
            step()
            return
        if target is not None:
            for _ in range(self.HISTORY_STROKE_LIMIT):
                step()
                if self._maskFingerprint() == target:
                    return
            return  # ran out of history: leave it where it got to
        for _ in range(self.HISTORY_SKIP_LIMIT):
            step()
            if self._maskFingerprint() != before:
                return

    @guarded("Resetting the case")
    def onReset(self):
        case = self.currentCase()
        if case is None:
            return
        if not slicer.util.confirmYesNoDisplay(
            "Reload the mask of {} from disk?\nAll edits since the last save are lost.".format(
                case.case_id
            ),
            windowTitle="GTReview",
        ):
            return
        # the ticks describe lesions in the edited mask that is being thrown
        # away; keeping them would leave Save & next case unlocked over a mask
        # nobody has reviewed
        self.reviewedSeeds.pop(case.directory, None)
        self.loadCurrentCase(maskPath=self.logic.maskPath)

    # -------------------------------------------------------------- save slots
    def saveCurrentCase(self):
        """Save with an overwrite confirmation; return the path or None."""
        case = self.currentCase()
        if case is None or self.logic.segmentationNode is None:
            slicer.util.showStatusMessage("GTReview: load a case first.", 3000)
            return None
        path = self.logic.reviewedPath()
        if os.path.exists(path):
            if not slicer.util.confirmYesNoDisplay(
                "{} already exists.\nOverwrite it?".format(path), windowTitle="GTReview"
            ):
                return None
        with BusyCursor("GTReview: saving {} ...".format(os.path.basename(path))):
            written = self.logic.saveReviewedMask()
        self.unsavedChanges = False
        slicer.util.showStatusMessage("GTReview: saved {}".format(written), 6000)
        # "reviewed" means exactly "the reviewed file exists": re-derive the
        # check marks and the skip-reviewed filter from disk right now
        self._refilterCases(keepCurrent=True)
        self._updateCaseControls()
        self._updateMaskSourceComboBox()
        return written

    def _refilterCases(self, keepCurrent=True):
        """Re-apply the skip-reviewed filter without unloading the open case."""
        current = self.currentCase()
        allowed = self._filteredCaseList()
        if keepCurrent and current is not None and not any(c is current for c in allowed):
            # the case just saved stays listed until the reviewer moves on
            allowed = [c for c in self.cases if c is current or any(c is a for a in allowed)]
        self.filteredCases = allowed
        index = -1
        if current is not None:
            for position, case in enumerate(self.filteredCases):
                if case is current:
                    index = position
                    break
        self.currentCaseIndex = index
        self._populateCaseComboBox()

    def _refuseUnlessAllDone(self):
        """True when saving must wait; says why in the status bar."""
        if self.logic.segmentationNode is None:
            return False
        if self._allLesionsDone():
            return False
        done, total = self._doneCount()
        slicer.util.showStatusMessage(
            "GTReview: tick every lesion as Done before saving ({} of {} done{}).".format(
                done, total, ", list is stale" if self.lesionsStale else ""
            ),
            5000,
        )
        return True

    @guarded("Saving the reviewed mask")
    def onSave(self):
        if self._refuseUnlessAllDone():
            return
        self.saveCurrentCase()

    @guarded("Saving the reviewed mask")
    def onSaveAndNext(self):
        if self._refuseUnlessAllDone():
            return
        if self.saveCurrentCase() is None:
            return
        if self.currentCaseIndex + 1 < len(self.filteredCases):
            self.setCurrentCaseIndex(self.currentCaseIndex + 1, force=True)
        else:
            slicer.util.showStatusMessage("GTReview: that was the last case.", 3000)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
class GTReviewTest(ScriptedLoadableModuleTest):
    """Self-test on synthetic in-memory data — no dependency on any dataset."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.logic = None
        self.editor = None
        self.tempDir = None

    def tearDown(self):
        if self.logic is not None:
            self.logic.unloadCase()
        if self.editor is not None:
            try:
                self.editor.setMRMLScene(None)
                self.editor.setMRMLSegmentEditorNode(None)
            except Exception:  # noqa: BLE001
                logging.debug("GTReview test: editor teardown failed", exc_info=True)
            self.editor = None
        if self.tempDir and os.path.isdir(self.tempDir):
            import shutil

            shutil.rmtree(self.tempDir, ignore_errors=True)
        self.tempDir = None

    def say(self, message):
        logging.info("GTReview test: %s", message)
        if slicer.util.mainWindow():
            self.delayDisplay(message)

    def runTest(self):
        self.setUp()
        try:
            self.test_GTReview1()
        finally:
            self.tearDown()

    # -------------------------------------------------------------- fixtures
    def _makeSyntheticCase(self):
        import tempfile

        self.tempDir = tempfile.mkdtemp(prefix="GTReviewTest_")
        caseId = "TC_001"
        caseDir = os.path.join(self.tempDir, caseId)
        os.makedirs(caseDir)

        size = (24, 20, 16)
        geometry = maskio.MaskGeometry(
            origin=(-10.0, 3.0, 1.5),
            spacing=(0.5, 1.25, 3.0),
            direction=(0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0),
            size=size,
        )
        mask = np.zeros(size, dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1        # 64 voxels, label 1
        mask[14:17, 12:15, 8:11] = 2   # 27 voxels, label 2
        image = np.zeros(size, dtype=np.uint8)
        image[:] = 20
        image[mask > 0] = 200

        maskio.write_mask(os.path.join(caseDir, caseId + "_seg.nii.gz"), mask, geometry)
        maskio.write_mask(os.path.join(caseDir, caseId + "_t1c.nii.gz"), image, geometry)
        return caseDir, mask

    # ------------------------------------------------------------------ test
    def test_GTReview1(self):
        self.say("Starting the GTReview self-test")
        caseDir, sourceMask = self._makeSyntheticCase()

        cases = dataset.discover_cases(self.tempDir)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.case_id, "TC_001")
        self.assertIn("t1c", case.images)
        self.assertIn("seg", case.masks)

        self.logic = GTReviewLogic()
        self.logic.loadCase(case)
        self.assertIsNotNone(self.logic.segmentationNode)
        self.assertEqual(len(self.logic.volumeNodes), 1)
        self.say("Case loaded")

        # --- label mapping / round trip -----------------------------------
        self.assertEqual(sorted(self.logic.labelValues()), [1, 2])
        exported = self.logic.exportLabelmapArrayIJK()
        self.assertEqual(tuple(exported.shape), tuple(sourceMask.shape))
        self.assertTrue(
            np.array_equal(exported.astype(np.uint8), sourceMask),
            "the exported labelmap must reproduce the ORIGINAL label values",
        )
        self.say("Export preserves the original label values")

        # --- lesions -------------------------------------------------------
        componentMap, found = self.logic.computeLesions()
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0].voxel_count, 64)
        self.assertEqual(found[0].label, 1)
        self.assertEqual(found[1].voxel_count, 27)
        self.assertEqual(found[1].label, 2)
        self.assertAlmostEqual(found[0].volume_mm3, 64 * 0.5 * 1.25 * 3.0, places=4)
        for lesion in found:
            i, j, k = lesion.centroid_ijk
            self.assertTrue(sourceMask[i, j, k] != 0)
            ras = self.logic.centroidToRAS(lesion.centroid_ijk)
            self.assertEqual(len(ras), 3)
        self.say("Lesion detection OK (2 lesions)")

        # --- add a label ---------------------------------------------------
        newSegmentId = self.logic.addLabel()
        self.assertEqual(self.logic.labelValueForSegmentId(newSegmentId), 3)
        self.assertEqual(sorted(self.logic.labelValues()), [1, 2, 3])
        exported = self.logic.exportLabelmapArrayIJK()
        self.assertTrue(
            np.array_equal(exported.astype(np.uint8), sourceMask),
            "an empty extra segment must not renumber the exported labels",
        )
        self.say("Add label OK (label 3, no renumbering)")

        # --- undo-aware edits ---------------------------------------------
        # Only the widget being unavailable may skip the rest; anything that
        # fails once the editor exists is a real failure and must surface.
        try:
            import qSlicerSegmentationsModuleWidgetsPythonQt as segmentationWidgets

            editor = segmentationWidgets.qMRMLSegmentEditorWidget()
        except (ImportError, AttributeError):
            logging.warning(
                "GTReview test: segment editor widget unavailable, "
                "skipping the edit/undo checks", exc_info=True
            )
            self.say("Self-test finished (edit checks skipped)")
            return
        self.editor = editor
        self.editor.setMaximumNumberOfUndoStates(20)
        self.editor.setUndoEnabled(True)
        editorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.editor.setMRMLSegmentEditorNode(editorNode)
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor.setSegmentationNode(self.logic.segmentationNode)
        self.editor.setSourceVolumeNode(self.logic.referenceVolumeNode)
        self.assertIsNotNone(self.editor.effectByName("Threshold"))
        self.logic.editorWidget = self.editor

        biggest = found[0]
        self.logic.deleteLesionVoxels(lesions.lesion_mask(componentMap, biggest.index))
        exported = self.logic.exportLabelmapArrayIJK()
        self.assertEqual(int((exported == 1).sum()), 0)
        self.assertEqual(int((exported == 2).sum()), 27)
        self.say("Delete lesion OK")

        self.editor.undo()
        exported = self.logic.exportLabelmapArrayIJK()
        self.assertEqual(int((exported == 1).sum()), 64)
        self.say("Undo OK")

        self.editor.redo()
        self.assertEqual(int((self.logic.exportLabelmapArrayIJK() == 1).sum()), 0)
        self.editor.undo()
        self.assertEqual(int((self.logic.exportLabelmapArrayIJK() == 1).sum()), 64)
        self.say("Redo OK")

        # relabel 1 -> 3, as a single undo step
        componentMap, found = self.logic.computeLesions()
        biggest = [lesion for lesion in found if lesion.label == 1][0]
        self.logic.changeLesionLabel(lesions.lesion_mask(componentMap, biggest.index), 3)
        exported = self.logic.exportLabelmapArrayIJK()
        self.assertEqual(int((exported == 1).sum()), 0)
        self.assertEqual(int((exported == 3).sum()), 64)
        self.assertEqual(int((exported == 2).sum()), 27)
        self.editor.undo()
        exported = self.logic.exportLabelmapArrayIJK()
        self.assertEqual(int((exported == 1).sum()), 64)
        self.assertEqual(int((exported == 3).sum()), 0)
        self.say("Change label OK, one undo step")

        # --- save ----------------------------------------------------------
        written = self.logic.saveReviewedMask()
        self.assertEqual(written, case.reviewed_path)
        self.assertTrue(os.path.isfile(written))
        self.assertTrue(written.startswith(caseDir))
        readBack, readGeometry = maskio.read_mask(written)
        self.assertTrue(np.array_equal(readBack.astype(np.uint8), sourceMask))
        self.assertTrue(readGeometry.is_compatible(self.logic.maskGeometry))
        self.say("Save round trip OK")

        # --- image/mask geometry disagreement is reported, not fatal -------
        shiftedGeometry = maskio.MaskGeometry(
            origin=(-10.0, 3.0, 9.5),  # 8 mm off along the third axis
            spacing=(0.5, 1.25, 3.0),
            direction=(0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0),
            size=sourceMask.shape,
        )
        maskio.write_mask(
            os.path.join(caseDir, "TC_001_t1c.nii.gz"),
            np.zeros(sourceMask.shape, dtype=np.uint8),
            shiftedGeometry,
        )
        shiftedCase = dataset.parse_case_files(caseDir)
        self.logic.loadCase(shiftedCase, maskPath=shiftedCase.masks["seg"])
        self.assertIsNotNone(self.logic.geometryWarning)
        self.assertIn("t1c", self.logic.geometryWarning)
        self.assertTrue(
            self.logic.maskGeometry.is_compatible(maskio.read_geometry(shiftedCase.masks["seg"])),
            "the mask geometry must win over the image geometry",
        )
        self.assertTrue(
            np.array_equal(self.logic.exportLabelmapArrayIJK().astype(np.uint8), sourceMask)
        )
        self.say("Geometry mismatch warning OK")

        # --- a case without a mask degrades to an empty segmentation -------
        os.remove(case.masks["seg"])
        os.remove(case.reviewed_path)
        bareCase = dataset.parse_case_files(caseDir)
        self.assertEqual(bareCase.masks, {})
        self.assertIsNone(bareCase.default_mask_path())
        self.logic.loadCase(bareCase)
        self.assertEqual(sorted(self.logic.labelValues()), sorted(LABEL_NAMES))
        self.assertEqual(int(self.logic.exportLabelmapArrayIJK().sum()), 0)
        self.say("Empty-mask fallback OK (both review labels seeded)")

        self.logic.unloadCase()
        self.assertEqual(
            len(slicer.util.getNodesByClass("vtkMRMLSegmentationNode")),
            0,
            "unloadCase must not leak nodes",
        )
        self.say("GTReview self-test passed")
