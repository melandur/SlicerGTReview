"""Sphere threshold -- a Segment Editor effect for growing one lesion from its centre.

Click the centre of a lesion in a slice view: the intensity of that voxel sets
the threshold range (plus/minus a tolerance).  Drag outward to pull a physical
sphere; a circle shows its size and the status line counts the voxels that
qualify.  Release, and everything inside the sphere that lies in the range and
is connected to the seed is added to the current segment, as ONE undo step.
The result is a plain voxel mask: no smoothing, no interpolation.

This file is loaded by Slicer's ``qSlicerSegmentEditorScriptedEffect`` (see
``GTReview.registerSphereThresholdEffect``), so unlike the rest of
``GTReviewLib`` it imports ``slicer``/``qt``/``vtk``.  The array maths lives in
:func:`GTReviewLib.lesions.sphere_threshold_mask`, which is unit-tested
without Slicer.
"""

import logging
import math
import os

import numpy as np
import qt
import vtk
import slicer
from vtk.util import numpy_support as vtk_np

from SegmentEditorEffects import AbstractScriptedSegmentEditorEffect

try:
    from GTReviewLib import lesions
except ImportError:  # pragma: no cover - loaded outside the GTReview module path
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from GTReviewLib import lesions


EFFECT_NAME = "Sphere threshold"

MODE_SIMILAR = "similar"
MODE_BRIGHTER = "brighter"
MODE_DARKER = "darker"
MODES = (
    (MODE_SIMILAR, "similar to the seed (± tolerance)"),
    (MODE_BRIGHTER, "similar or brighter"),
    (MODE_DARKER, "similar or darker"),
)

#: a bare click (no drag) below this many voxels of radius is ignored
MIN_RADIUS_VOXELS = 0.5
#: guard against a runaway drag: the sphere never grows past this
MAX_RADIUS_MM = 60.0


def intensity_range(seed_value, tolerance_percent, mode):
    """``(lower, upper)`` around *seed_value* for the given mode."""
    value = float(seed_value)
    tolerance = max(abs(value) * float(tolerance_percent) / 100.0, 1e-6)
    if mode == MODE_BRIGHTER:
        return value - tolerance, float("inf")
    if mode == MODE_DARKER:
        return float("-inf"), value + tolerance
    return value - tolerance, value + tolerance


class SegmentEditorSphereThresholdEffect(AbstractScriptedSegmentEditorEffect):
    def __init__(self, scriptedEffect):
        scriptedEffect.name = EFFECT_NAME
        try:
            scriptedEffect.title = EFFECT_NAME
        except Exception:  # noqa: BLE001 - no title property in older Slicer
            pass
        scriptedEffect.perSegment = True
        scriptedEffect.requireSegments = True
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

        self.pipelines = {}  # slice widget -> _CirclePipeline
        self.dragging = False
        self.dragWidget = None
        self.seedRas = None
        self.seedIjk = None  # absolute image ijk (extent-aware)
        self.seedValue = None
        self.radiusMm = 0.0
        #: image axis the drag's slice view cuts across, captured at mouse-down
        #: because the apply happens after the drag state has been torn down
        self.flatAxis = None
        self._imageCache = None  # (image object, array_kji, extent)
        self.readoutLabel = None
        self.twoDCheckBox = None
        self.lastApplied = None  # (voxels, lower, upper, radius_mm) for GTReview's status line

    # ------------------------------------------------------------ boilerplate
    def clone(self):
        import qSlicerSegmentationsEditorEffectsPythonQt as effects

        clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(__file__.replace("\\", "/"))
        return clonedEffect

    def icon(self):
        iconPath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Resources", "Icons", "SphereThreshold.png",
        )
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def helpText(self):
        return (
            "<html><b>Click</b> the centre of a lesion, <b>drag</b> outward to pull a "
            "sphere, <b>release</b> to add every voxel in the sphere whose intensity "
            "is within the tolerance of the clicked voxel and connected to it.  Tick "
            "<b>2D</b> to keep only the slice you drew on.  One undo step; the result "
            "is a voxel mask.</html>"
        )

    # ------------------------------------------------------------ parameters
    def setMRMLDefaults(self):
        self.scriptedEffect.setParameterDefault("TolerancePercent", 25.0)
        self.scriptedEffect.setParameterDefault("Mode", MODE_SIMILAR)
        self.scriptedEffect.setParameterDefault("ConnectedOnly", 1)
        self.scriptedEffect.setParameterDefault("TwoD", 0)

    def tolerancePercent(self):
        return float(self.scriptedEffect.doubleParameter("TolerancePercent"))

    def mode(self):
        return self.scriptedEffect.parameter("Mode") or MODE_SIMILAR

    def connectedOnly(self):
        return bool(self.scriptedEffect.integerParameter("ConnectedOnly"))

    def twoDimensional(self):
        return bool(self.scriptedEffect.integerParameter("TwoD"))

    def setupOptionsFrame(self):
        self.toleranceSlider = slicer.qMRMLSliderWidget() if hasattr(slicer, "qMRMLSliderWidget") else None
        if self.toleranceSlider is None:
            import ctk

            self.toleranceSlider = ctk.ctkSliderWidget()
        self.toleranceSlider.minimum = 0.0
        self.toleranceSlider.maximum = 100.0
        self.toleranceSlider.singleStep = 1.0
        self.toleranceSlider.decimals = 0
        self.toleranceSlider.suffix = " %"
        self.toleranceSlider.setToolTip(
            "How far an intensity may deviate from the clicked voxel, as a "
            "percentage of that voxel's value."
        )
        self.scriptedEffect.addLabeledOptionsWidget("Tolerance:", self.toleranceSlider)

        self.modeComboBox = qt.QComboBox()
        for key, title in MODES:
            self.modeComboBox.addItem(title, key)
        self.modeComboBox.setToolTip(
            "Which intensities count: within the tolerance of the seed, or "
            "additionally everything brighter / darker than it."
        )
        self.scriptedEffect.addLabeledOptionsWidget("Include:", self.modeComboBox)

        self.connectedCheckBox = qt.QCheckBox("Only voxels connected to the seed")
        self.connectedCheckBox.setToolTip(
            "Leave unrelated bright or dark spots inside the sphere alone."
        )
        self.scriptedEffect.addOptionsWidget(self.connectedCheckBox)

        self.twoDCheckBox = qt.QCheckBox("2D: this slice only")
        self.twoDCheckBox.setToolTip(
            "Segment a disc on the slice you are looking at instead of a ball "
            "through the ones you are not.\n"
            "Use it when a lesion is only convincing on one slice: a sphere wide "
            "enough to cover it in-plane also reaches the slice above and below, "
            "and those voxels have to be erased again."
        )
        self.scriptedEffect.addOptionsWidget(self.twoDCheckBox)

        # No readout widget: the same sentences go to the status bar via
        # _setReadout, and a label that grows and reflows under the options
        # shifted every control below it after each drag.

        self.toleranceSlider.connect("valueChanged(double)", self.updateMRMLFromGUI)
        self.modeComboBox.connect("currentIndexChanged(int)", self.updateMRMLFromGUI)
        self.connectedCheckBox.connect("toggled(bool)", self.updateMRMLFromGUI)
        self.twoDCheckBox.connect("toggled(bool)", self.updateMRMLFromGUI)

    def updateGUIFromMRML(self):
        widgets = (self.toleranceSlider, self.modeComboBox, self.connectedCheckBox,
                   self.twoDCheckBox)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.toleranceSlider.value = self.tolerancePercent()
            index = self.modeComboBox.findData(self.mode())
            self.modeComboBox.currentIndex = index if index >= 0 else 0
            self.connectedCheckBox.checked = self.connectedOnly()
            self.twoDCheckBox.checked = self.twoDimensional()
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def updateMRMLFromGUI(self, *args):
        del args
        self.scriptedEffect.setParameter("TolerancePercent", float(self.toleranceSlider.value))
        self.scriptedEffect.setParameter(
            "Mode", str(self.modeComboBox.itemData(self.modeComboBox.currentIndex))
        )
        self.scriptedEffect.setParameter("ConnectedOnly", 1 if self.connectedCheckBox.checked else 0)
        self.scriptedEffect.setParameter("TwoD", 1 if self.twoDCheckBox.checked else 0)

    # ------------------------------------------------------------ lifecycle
    def activate(self):
        self.flatAxis = None
        self._imageCache = None
        self._setReadout("Click the centre of a lesion, then drag.")

    def deactivate(self):
        self._endDrag()
        for pipeline in list(self.pipelines.values()):
            pipeline.remove()
        self.pipelines = {}
        self._imageCache = None

    def cleanup(self):
        self.deactivate()

    # ------------------------------------------------------------ interaction
    def processInteractionEvents(self, callerInteractor, eventId, viewWidget):
        abortEvent = False
        if viewWidget.className() != "qMRMLSliceWidget":
            return abortEvent
        image = self.scriptedEffect.sourceVolumeImageData()
        if image is None:
            return abortEvent
        anyModifier = (
            callerInteractor.GetShiftKey()
            or callerInteractor.GetControlKey()
            or callerInteractor.GetAltKey()
        )
        if eventId == vtk.vtkCommand.LeftButtonPressEvent and not anyModifier:
            xy = callerInteractor.GetEventPosition()
            ras = self.xyToRas(xy, viewWidget)
            ijk = self.xyToIjk(xy, viewWidget, image)
            if not self._beginDrag(viewWidget, ras, ijk, image):
                return abortEvent
            abortEvent = True
        elif eventId == vtk.vtkCommand.MouseMoveEvent and self.dragging:
            if viewWidget is self.dragWidget:
                xy = callerInteractor.GetEventPosition()
                ras = self.xyToRas(xy, viewWidget)
                self._updateDrag(ras)
                abortEvent = True
        elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent and self.dragging:
            self._finishDrag()
            abortEvent = True
        return abortEvent

    def _beginDrag(self, viewWidget, ras, ijk, image):
        if not self._insideExtent(ijk, image):
            self._setReadout("That point is outside the image.")
            return False
        value = self._voxelValue(ijk, image)
        if value is None:
            return False
        self.dragging = True
        self.dragWidget = viewWidget
        self.flatAxis = self._flatAxis()
        self.seedRas = list(ras)
        self.seedIjk = tuple(int(v) for v in ijk)
        self.seedValue = float(value)
        self.radiusMm = 0.0
        pipeline = self.pipelines.get(viewWidget)
        if pipeline is None:
            pipeline = _CirclePipeline(self.scriptedEffect, viewWidget)
            self.pipelines[viewWidget] = pipeline
        pipeline.update(self.seedRas, 0.0)
        self._updateReadout(0)
        return True

    def _updateDrag(self, ras):
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(ras, self.seedRas)))
        self.radiusMm = min(float(distance), MAX_RADIUS_MM)
        pipeline = self.pipelines.get(self.dragWidget)
        if pipeline is not None:
            pipeline.update(self.seedRas, self.radiusMm)
        # a live count is cheap: the sphere is small
        box, mask = self._computeMask(self.seedIjk, self.radiusMm)
        self._updateReadout(int(mask.sum()) if mask is not None else 0)

    def _finishDrag(self):
        seedIjk, radiusMm = self.seedIjk, self.radiusMm
        self._endDrag()
        spacing = self._spacing()
        if radiusMm < MIN_RADIUS_VOXELS * min(spacing):
            self._setReadout("Drag outward from the centre to pull the sphere.")
            return
        self.applyAt(seedIjk, radiusMm)

    def _endDrag(self):
        self.dragging = False
        pipeline = self.pipelines.get(self.dragWidget) if self.dragWidget is not None else None
        if pipeline is not None:
            pipeline.hide()
        self.dragWidget = None

    # ------------------------------------------------------------ the maths
    def _spacing(self):
        image = self.scriptedEffect.sourceVolumeImageData()
        return tuple(float(v) for v in image.GetSpacing()) if image is not None else (1.0, 1.0, 1.0)

    @staticmethod
    def _insideExtent(ijk, image):
        extent = image.GetExtent()
        return all(extent[2 * a] <= ijk[a] <= extent[2 * a + 1] for a in range(3))

    def _imageArrayKJI(self, image):
        """The source volume as a ``[k, j, i]`` numpy view (cached per image)."""
        cache = self._imageCache
        if cache is not None and cache[0] is image and cache[2] == tuple(image.GetExtent()):
            return cache[1]
        scalars = image.GetPointData().GetScalars()
        if scalars is None:
            return None
        dims = image.GetDimensions()
        array = vtk_np.vtk_to_numpy(scalars)
        if scalars.GetNumberOfComponents() != 1:
            array = array[:, 0]
        array = array.reshape(dims[2], dims[1], dims[0])
        self._imageCache = (image, array, tuple(image.GetExtent()))
        return array

    def _voxelValue(self, ijk, image):
        array = self._imageArrayKJI(image)
        if array is None:
            return None
        extent = image.GetExtent()
        i, j, k = (int(ijk[a]) - extent[2 * a] for a in range(3))
        return float(array[k, j, i])

    def currentRange(self, seedValue=None):
        value = self.seedValue if seedValue is None else seedValue
        return intensity_range(value, self.tolerancePercent(), self.mode())

    def _computeMask(self, seedIjk, radiusMm):
        """``(box, mask)`` in the SOURCE image's ``[i, j, k]`` index space."""
        image = self.scriptedEffect.sourceVolumeImageData()
        if image is None:
            return None, None
        array_kji = self._imageArrayKJI(image)
        if array_kji is None:
            return None, None
        extent = image.GetExtent()
        seed_rel = tuple(int(seedIjk[a]) - extent[2 * a] for a in range(3))
        image_ijk = array_kji.transpose(2, 1, 0)
        lower, upper = self.currentRange(self._voxelValue(seedIjk, image))
        box, mask = lesions.sphere_threshold_mask(
            image_ijk, seed_rel, radiusMm, self._spacing(), lower, upper,
            connected=self.connectedOnly(),
        )
        if not self.twoDimensional() or mask is None:
            return box, mask
        return box, self._flattenToSeedSlice(box, mask, seedIjk, extent)

    def _flatAxis(self):
        """Which image axis the slice being drawn on cuts across, or None.

        The slice normal is a direction in world space; pushing it through the
        volume's world-to-image rotation says which of i, j, k it lines up with,
        and the largest component wins.  That is exact once the slice views are
        aligned to the image grid and still the best answer when they are not.
        """
        widget = self.dragWidget
        image = self.scriptedEffect.sourceVolumeImageData()
        if widget is None or image is None:
            return None
        try:
            sliceNode = widget.mrmlSliceNode()
        except Exception:  # noqa: BLE001 - a 3D view has no slice node
            return None
        if sliceNode is None:
            return None
        sliceToRas = sliceNode.GetSliceToRAS()
        normalWorld = [sliceToRas.GetElement(row, 2) for row in range(3)]
        imageToWorld = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(imageToWorld)
        worldToImage = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(imageToWorld, worldToImage)
        components = [
            abs(sum(worldToImage.GetElement(axis, col) * normalWorld[col]
                    for col in range(3)))
            for axis in range(3)
        ]
        return int(max(range(3), key=lambda axis: components[axis]))

    def _flattenToSeedSlice(self, box, mask, seedIjk, extent):
        """Keep only the plane of *mask* that holds the seed.

        The axis is the one captured at mouse-down: _finishDrag tears the drag
        state down before it applies, so asking the (now cleared) drag widget
        here would answer None and quietly hand back the whole ball.
        """
        axis = self.flatAxis
        if axis is None:
            axis = self._flatAxis()
        if axis is None:
            return mask  # drawn somewhere with no slice plane: leave the ball
        plane = int(seedIjk[axis]) - extent[2 * axis] - (box[axis].start or 0)
        if not 0 <= plane < mask.shape[axis]:
            logging.debug("GTReview: 2D sphere threshold seed outside its own box")
            return mask
        flat = np.zeros_like(mask)
        index = [slice(None)] * 3
        index[axis] = slice(plane, plane + 1)
        flat[tuple(index)] = mask[tuple(index)]
        return flat

    def applyAt(self, seedIjk, radiusMm):
        """Grow from *seedIjk* (absolute image ijk) with *radiusMm*; one undo step.

        Public so tests and scripts can drive the effect without a mouse.
        Returns the number of voxels added to the modifier (before masking).
        """
        image = self.scriptedEffect.sourceVolumeImageData()
        if image is None:
            return 0
        self.seedValue = self._voxelValue(seedIjk, image)
        box, mask = self._computeMask(seedIjk, radiusMm)
        if mask is None or not mask.any():
            self._setReadout("Nothing in range inside that sphere.")
            return 0
        modifier = slicer.vtkOrientedImageData()
        modifier.SetExtent(image.GetExtent())
        modifier.SetSpacing(image.GetSpacing())
        modifier.SetOrigin(image.GetOrigin())
        imageToWorld = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(imageToWorld)
        modifier.SetImageToWorldMatrix(imageToWorld)
        modifier.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        dims = modifier.GetDimensions()
        target = vtk_np.vtk_to_numpy(modifier.GetPointData().GetScalars()).reshape(
            dims[2], dims[1], dims[0]
        )
        target[:] = 0
        # write the [i,j,k] box into the [k,j,i] buffer
        target[box[2], box[1], box[0]] = mask.transpose(2, 1, 0)
        modifier.GetPointData().GetScalars().Modified()
        modifier.Modified()

        voxels = int(mask.sum())
        lower, upper = self.currentRange()
        self.scriptedEffect.saveStateForUndo()
        self.scriptedEffect.modifySelectedSegmentByLabelmap(
            modifier, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
        )
        self.lastApplied = (voxels, lower, upper, float(radiusMm))
        self._setReadout(
            "Added {} voxels within {} of the centre ({}).".format(
                voxels, self._radiusText(radiusMm), self._rangeText(lower, upper)
            )
        )
        return voxels

    # ------------------------------------------------------------ readouts
    def _radiusText(self, radiusMm):
        spacing = self._spacing()
        voxels = radiusMm / min(spacing) if min(spacing) > 0 else 0.0
        text = "{:.1f} voxels / {:.1f} mm".format(voxels, radiusMm)
        return text + ", this slice only" if self.twoDimensional() else text

    @staticmethod
    def _rangeText(lower, upper):
        lo = "-inf" if math.isinf(lower) else "{:g}".format(lower)
        hi = "inf" if math.isinf(upper) else "{:g}".format(upper)
        return "range [{}, {}]".format(lo, hi)

    def _updateReadout(self, voxelCount):
        lower, upper = self.currentRange()
        self._setReadout(
            "seed {:g} -> {}; radius {}; {} voxels".format(
                self.seedValue, self._rangeText(lower, upper),
                self._radiusText(self.radiusMm), voxelCount,
            )
        )

    def _setReadout(self, text):
        if self.readoutLabel is not None:
            self.readoutLabel.text = text
        try:
            slicer.util.showStatusMessage("Sphere threshold: " + text, 4000)
        except Exception:  # noqa: BLE001 - no status bar
            pass


class _CirclePipeline:
    """A yellow circle in one slice view: the sphere's cross-section."""

    def __init__(self, scriptedEffect, sliceWidget):
        self.scriptedEffect = scriptedEffect
        self.sliceWidget = sliceWidget

        self.source = vtk.vtkCylinderSource()
        self.source.SetResolution(48)

        self.brushToWorldOriginTransform = vtk.vtkTransform()
        self.brushToWorldOriginTransformer = vtk.vtkTransformPolyDataFilter()
        self.brushToWorldOriginTransformer.SetTransform(self.brushToWorldOriginTransform)
        self.brushToWorldOriginTransformer.SetInputConnection(self.source.GetOutputPort())

        self.worldOriginToWorldTransform = vtk.vtkTransform()
        self.worldOriginToWorldTransformer = vtk.vtkTransformPolyDataFilter()
        self.worldOriginToWorldTransformer.SetTransform(self.worldOriginToWorldTransform)
        self.worldOriginToWorldTransformer.SetInputConnection(
            self.brushToWorldOriginTransformer.GetOutputPort()
        )

        self.worldToSliceTransform = vtk.vtkTransform()
        self.worldToSliceTransformer = vtk.vtkTransformPolyDataFilter()
        self.worldToSliceTransformer.SetTransform(self.worldToSliceTransform)
        self.worldToSliceTransformer.SetInputConnection(
            self.worldOriginToWorldTransformer.GetOutputPort()
        )

        self.slicePlane = vtk.vtkPlane()
        self.slicePlane.SetNormal(0, 0, 1)
        self.slicePlane.SetOrigin(0, 0, 0)
        self.cutter = vtk.vtkCutter()
        self.cutter.SetCutFunction(self.slicePlane)
        self.cutter.SetInputConnection(self.worldToSliceTransformer.GetOutputPort())

        self.mapper = vtk.vtkPolyDataMapper2D()
        self.mapper.SetInputConnection(self.cutter.GetOutputPort())
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(self.mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(1.0, 1.0, 0.0)
        prop.SetLineWidth(2)
        self.actor.VisibilityOff()
        self.scriptedEffect.addActor2D(sliceWidget, self.actor)

    def update(self, centerRas, radiusMm):
        sliceNode = self.sliceWidget.sliceLogic().GetSliceNode()
        rasToSliceXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(sliceNode.GetXYToRAS(), rasToSliceXy)
        self.worldToSliceTransform.SetMatrix(rasToSliceXy)

        brushToWorldOrigin = vtk.vtkMatrix4x4()
        brushToWorldOrigin.DeepCopy(sliceNode.GetSliceToRAS())
        for row in range(3):
            brushToWorldOrigin.SetElement(row, 3, 0)
        self.brushToWorldOriginTransform.Identity()
        self.brushToWorldOriginTransform.Concatenate(brushToWorldOrigin)
        self.brushToWorldOriginTransform.RotateX(90)  # cylinder axis Y -> slice normal

        self.source.SetRadius(max(float(radiusMm), 0.01))
        self.source.SetHeight(self.scriptedEffect.sliceSpacing(self.sliceWidget))
        self.worldOriginToWorldTransform.Identity()
        self.worldOriginToWorldTransform.Translate(centerRas)
        self.actor.VisibilityOn()
        self.sliceWidget.sliceView().scheduleRender()

    def hide(self):
        self.actor.VisibilityOff()
        try:
            self.sliceWidget.sliceView().scheduleRender()
        except Exception:  # noqa: BLE001 - view already gone
            pass

    def remove(self):
        try:
            self.scriptedEffect.removeActor2D(self.sliceWidget, self.actor)
        except Exception:  # noqa: BLE001
            logging.debug("Sphere threshold: removing the circle failed", exc_info=True)
