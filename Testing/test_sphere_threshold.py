"""Unit tests for the maths behind the "Sphere threshold" segment editor effect.

The effect module itself (``GTReviewLib/SegmentEditorSphereThresholdEffect.py``)
imports ``slicer``, ``qt`` and ``vtk`` at the top, so it can never be imported
by this suite -- the suite has to stay runnable under a plain ``PythonSlicer``
interpreter.  Everything the effect actually computes therefore gets tested
where it lives:

* the sphere / threshold / connectivity maths in
  :func:`GTReviewLib.lesions.sphere_threshold_mask`, exercised here against an
  independent brute-force reference over the whole volume;
* the **2D flattening rule** (``_flattenToSeedSlice``), which is a method on the
  effect class.  It *is* reachable without Slicer, but only by parsing the file
  and compiling that single ``def`` -- see :func:`_load_flatten_rule`.  Nothing
  else in that file is executed, no Slicer module is touched, and the rule is
  not paraphrased here: a paraphrase would only prove itself self-consistent.
  ``_flatAxis`` (the other half of the 2D feature) is *not* testable -- it is
  pure VTK matrix plumbing -- so the axis is supplied as an input instead.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_sphere_threshold.py' -v
"""

import ast
import logging
import os
import sys
import unittest

import numpy as np

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

try:  # normal package import (GTReviewLib may be a namespace package)
    from GTReviewLib import lesions
    from GTReviewLib.lesions import sphere_threshold_mask
except ImportError:  # pragma: no cover - fallback: load straight from the file
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "gtreview_lesions", os.path.join(_MODULE_DIR, "GTReviewLib", "lesions.py")
    )
    lesions = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(lesions)
    sphere_threshold_mask = lesions.sphere_threshold_mask


ISO = (1.0, 1.0, 1.0)
#: the real acquisition grid in this project's corpus
CT = (0.4297, 0.4297, 0.5)

_EFFECT_PATH = os.path.join(
    _MODULE_DIR, "GTReviewLib", "SegmentEditorSphereThresholdEffect.py"
)


def _expand(box, mask, shape):
    """Paste a ``(box, mask)`` result back into a full-volume boolean array."""
    full = np.zeros(shape, dtype=bool)
    full[box] = mask
    return full


def _brute_force(image, seed, radius_mm, spacing, lower, upper):
    """Reference sphere+window mask, computed over the whole volume.

    No bounding box, no components: this is the definition the boxed
    implementation has to reproduce.  The ``1e-9`` mirrors the implementation's
    inclusive tolerance on the sphere surface; boundary inclusion itself is
    pinned separately by ``test_surface_voxel_is_inclusive``, so mirroring it
    here only keeps the other tests off a knife edge.
    """
    lo, hi = (lower, upper) if lower <= upper else (upper, lower)
    grid = np.indices(image.shape).astype(np.float64)
    d2 = sum(
        ((grid[axis] - seed[axis]) * spacing[axis]) ** 2 for axis in range(3)
    )
    return (d2 <= max(0.0, radius_mm) ** 2 + 1e-9) & (image >= lo) & (image <= hi)


def _load_flatten_rule():
    """Compile the effect's real ``_flattenToSeedSlice``, or return ``None``.

    ``ast.parse`` reads the file without running it, then only the one
    ``FunctionDef`` is compiled, with the two globals its body uses.  The
    module's ``import slicer`` never executes.
    """
    try:
        with open(_EFFECT_PATH, "r") as handle:
            tree = ast.parse(handle.read(), filename=_EFFECT_PATH)
    except (OSError, SyntaxError):  # pragma: no cover - file moved or broken
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_flattenToSeedSlice":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"np": np, "logging": logging}
            exec(compile(module, _EFFECT_PATH, "exec"), namespace)  # noqa: S102
            return namespace["_flattenToSeedSlice"]
    return None  # pragma: no cover - only if the method is renamed


_FLATTEN = _load_flatten_rule()


class _FakeEffect(object):
    """The only two things ``_flattenToSeedSlice`` reads off ``self``.

    ``flatAxis`` is what the effect captures at mouse-down; ``_flatAxis()`` is
    the live fallback, which needs a slice widget and so can only answer
    ``None`` here -- which is exactly the "drawn where there is no slice plane"
    case the rule has to handle.
    """

    def __init__(self, flat_axis=None):
        self.flatAxis = flat_axis

    def _flatAxis(self):
        return None


class SphereGeometryTest(unittest.TestCase):
    """Radius, box clipping and the sphere surface itself."""

    def setUp(self):
        # uniform in-range image: the mask is then purely a geometry question
        self.image = np.full((21, 21, 21), 100, dtype=np.int16)
        self.centre = (10, 10, 10)

    def _geometry(self, seed, radius_mm, spacing=ISO):
        box, mask = sphere_threshold_mask(self.image, seed, radius_mm, spacing, 0, 200)
        return box, _expand(box, mask, self.image.shape)

    def test_radius_reaches_exactly_that_many_voxels(self):
        _, full = self._geometry(self.centre, 3.0)
        self.assertTrue(full[13, 10, 10])
        self.assertFalse(full[14, 10, 10])
        self.assertTrue(full[7, 10, 10])
        self.assertFalse(full[6, 10, 10])

    def test_surface_voxel_is_inclusive(self):
        """A voxel at exactly ``radius_mm`` is inside -- the compare is ``<=``."""
        _, full = self._geometry(self.centre, 2.0)
        self.assertTrue(full[12, 10, 10], "distance == radius must be kept")
        # and the diagonal at sqrt(8) = 2.83 is genuinely outside r = 2
        self.assertFalse(full[12, 12, 10])
        _, full = self._geometry(self.centre, np.sqrt(8.0))
        self.assertTrue(full[12, 12, 10], "sqrt(8) diagonal is on the r=sqrt(8) shell")

    def test_it_is_a_ball_not_a_cube(self):
        _, full = self._geometry(self.centre, 4.0)
        self.assertFalse(full[14, 14, 14], "cube corner is 6.9 mm away")
        self.assertEqual(int(full.sum()), int(_brute_force(
            self.image, self.centre, 4.0, ISO, 0, 200).sum()))

    def test_box_is_the_tight_bounding_box(self):
        box, _ = self._geometry(self.centre, 2.6)
        # half-width is floor(2.6 / 1.0) = 2, so the box spans seed +/- 2
        self.assertEqual(box, (slice(8, 13), slice(8, 13), slice(8, 13)))

    def test_box_never_clips_the_sphere(self):
        """floor() on the half-width must not cut off a voxel that qualifies.

        Swept over radii that land between voxel steps, on both an isotropic and
        a real anisotropic grid, because that is where a truncation bug hides.
        """
        for spacing in (ISO, CT, (1.0, 0.3, 2.7)):
            for radius in np.arange(0.0, 5.0, 0.13):
                box, full = self._geometry(self.centre, float(radius), spacing)
                expected = _brute_force(
                    self.image, self.centre, float(radius), spacing, 0, 200
                )
                self.assertTrue(
                    np.array_equal(full, expected),
                    "radius %.2f spacing %s" % (radius, spacing),
                )

    def test_zero_radius_is_the_seed_voxel_alone(self):
        box, mask = sphere_threshold_mask(self.image, self.centre, 0.0, ISO, 0, 200)
        self.assertEqual(mask.shape, (1, 1, 1))
        self.assertEqual(box, (slice(10, 11), slice(10, 11), slice(10, 11)))
        self.assertTrue(mask[0, 0, 0])

    def test_negative_radius_is_clamped_to_zero(self):
        box, mask = sphere_threshold_mask(self.image, self.centre, -7.5, ISO, 0, 200)
        self.assertEqual(mask.shape, (1, 1, 1))
        self.assertTrue(mask[0, 0, 0])

    def test_radius_past_the_volume_is_clipped_not_an_error(self):
        box, mask = sphere_threshold_mask(self.image, (2, 3, 4), 500.0, ISO, 0, 200)
        self.assertEqual(box, (slice(0, 21), slice(0, 21), slice(0, 21)))
        self.assertTrue(mask.all(), "everything is in range and within 500 mm")

    def test_seed_on_an_edge(self):
        box, full = self._geometry((0, 10, 10), 3.0)
        self.assertEqual(box[0], slice(0, 4), "no negative start")
        self.assertTrue(full[0, 10, 10])
        self.assertTrue(full[3, 10, 10])
        self.assertTrue(np.array_equal(
            full, _brute_force(self.image, (0, 10, 10), 3.0, ISO, 0, 200)))

    def test_seed_on_a_corner(self):
        seed = (20, 0, 20)
        box, full = self._geometry(seed, 4.0)
        self.assertEqual(box, (slice(16, 21), slice(0, 5), slice(16, 21)))
        self.assertTrue(full[seed])
        self.assertTrue(np.array_equal(
            full, _brute_force(self.image, seed, 4.0, ISO, 0, 200)))

    def test_single_voxel_volume(self):
        tiny = np.array([[[42]]], dtype=np.int16)
        box, mask = sphere_threshold_mask(tiny, (0, 0, 0), 10.0, ISO, 0, 100)
        self.assertEqual(mask.shape, (1, 1, 1))
        self.assertTrue(mask[0, 0, 0])


class AnisotropyTest(unittest.TestCase):
    """The sphere is physical: mm, not voxels."""

    def setUp(self):
        self.image = np.full((25, 25, 25), 100, dtype=np.int16)
        self.centre = (12, 12, 12)

    def test_reach_scales_with_spacing(self):
        box, mask = sphere_threshold_mask(
            self.image, self.centre, 4.0, (1.0, 2.0, 0.5), 0, 200
        )
        full = _expand(box, mask, self.image.shape)
        self.assertTrue(full[16, 12, 12] and not full[17, 12, 12], "4 x 1.0 mm")
        self.assertTrue(full[12, 14, 12] and not full[12, 15, 12], "2 x 2.0 mm")
        self.assertTrue(full[12, 12, 20] and not full[12, 12, 21], "8 x 0.5 mm")

    def test_matches_brute_force_on_the_real_grid(self):
        box, mask = sphere_threshold_mask(self.image, self.centre, 3.0, CT, 0, 200)
        full = _expand(box, mask, self.image.shape)
        self.assertTrue(np.array_equal(
            full, _brute_force(self.image, self.centre, 3.0, CT, 0, 200)))

    def test_axis_coarser_than_the_radius_collapses_that_axis(self):
        """spacing 5 mm with a 3 mm radius: the sphere cannot leave the slice."""
        box, mask = sphere_threshold_mask(
            self.image, self.centre, 3.0, (1.0, 1.0, 5.0), 0, 200
        )
        self.assertEqual(box[2], slice(12, 13))
        full = _expand(box, mask, self.image.shape)
        self.assertEqual(int(full[:, :, 11].sum()) + int(full[:, :, 13].sum()), 0)
        self.assertTrue(full[15, 12, 12])

    def test_spacing_order_is_ijk_not_kji(self):
        """Guards the axis order at the maskio boundary: i is spacing[0]."""
        box, mask = sphere_threshold_mask(
            self.image, self.centre, 2.0, (2.0, 1.0, 1.0), 0, 200
        )
        full = _expand(box, mask, self.image.shape)
        self.assertTrue(full[13, 12, 12] and not full[14, 12, 12])
        self.assertTrue(full[12, 14, 12], "the fine axis reaches twice as far")


class IntensityWindowTest(unittest.TestCase):
    """The ``[lower, upper]`` window, including the open-ended effect modes."""

    def setUp(self):
        # a ramp along i so every threshold cuts at a predictable voxel
        self.image = np.zeros((21, 21, 21), dtype=np.int16)
        self.image += np.arange(21, dtype=np.int16)[:, None, None] * 10

    def test_window_is_inclusive_at_both_ends(self):
        box, mask = sphere_threshold_mask(
            self.image, (10, 10, 10), 5.0, ISO, 80, 120, connected=False
        )
        full = _expand(box, mask, self.image.shape)
        self.assertTrue(full[8, 10, 10], "value 80 == lower is kept")
        self.assertTrue(full[12, 10, 10], "value 120 == upper is kept")
        self.assertFalse(full[7, 10, 10])
        self.assertFalse(full[13, 10, 10])

    def test_swapped_bounds_are_normalised(self):
        args = (self.image, (10, 10, 10), 5.0, ISO)
        _, straight = sphere_threshold_mask(*args, lower=80, upper=120)
        _, swapped = sphere_threshold_mask(*args, lower=120, upper=80)
        self.assertTrue(np.array_equal(straight, swapped))

    def test_open_ended_ranges(self):
        """The brighter / darker modes hand in an infinite bound."""
        seed = (10, 10, 10)
        box, mask = sphere_threshold_mask(
            self.image, seed, 4.0, ISO, 95, float("inf"), connected=False
        )
        full = _expand(box, mask, self.image.shape)
        self.assertTrue(full[14, 10, 10])
        self.assertFalse(full[9, 10, 10], "value 90 is below the lower bound")

        box, mask = sphere_threshold_mask(
            self.image, seed, 4.0, ISO, float("-inf"), 105, connected=False
        )
        full = _expand(box, mask, self.image.shape)
        self.assertTrue(full[6, 10, 10])
        self.assertFalse(full[11, 10, 10], "value 110 is above the upper bound")

    def test_negative_intensities(self):
        """CT is signed: a window in the negatives must behave like any other."""
        image = np.full((11, 11, 11), -900, dtype=np.int16)
        image[5, 5, 5] = -500
        image[5, 5, 6] = -480
        box, mask = sphere_threshold_mask(image, (5, 5, 5), 3.0, ISO, -520, -450)
        full = _expand(box, mask, image.shape)
        self.assertEqual(sorted(map(tuple, np.argwhere(full))), [(5, 5, 5), (5, 5, 6)])

    def test_float_image(self):
        image = np.random.RandomState(0).uniform(0.0, 1.0, (15, 15, 15))
        image[7, 7, 7] = 0.5
        box, mask = sphere_threshold_mask(
            image, (7, 7, 7), 3.0, ISO, 0.4, 0.6, connected=False
        )
        full = _expand(box, mask, image.shape)
        self.assertTrue(np.array_equal(
            full, _brute_force(image, (7, 7, 7), 3.0, ISO, 0.4, 0.6)))

    def test_nan_voxels_are_excluded(self):
        image = np.full((9, 9, 9), 100.0)
        image[4, 4, 5] = np.nan
        box, mask = sphere_threshold_mask(image, (4, 4, 4), 2.0, ISO, 0, 200)
        full = _expand(box, mask, image.shape)
        self.assertFalse(full[4, 4, 5], "NaN compares false against both bounds")
        self.assertTrue(full[4, 4, 3])

    def test_seed_out_of_range_yields_nothing_when_connected(self):
        image = np.full((9, 9, 9), 100, dtype=np.int16)
        image[4, 4, 4] = 0  # the seed itself fails the window
        box, mask = sphere_threshold_mask(image, (4, 4, 4), 3.0, ISO, 50, 150)
        self.assertEqual(int(mask.sum()), 0)
        self.assertEqual(mask.shape, (7, 7, 7), "shape still describes the box")

    def test_seed_out_of_range_still_paints_when_not_connected(self):
        """Documented asymmetry -- see the notes in the test report."""
        image = np.full((9, 9, 9), 100, dtype=np.int16)
        image[4, 4, 4] = 0
        box, mask = sphere_threshold_mask(
            image, (4, 4, 4), 3.0, ISO, 50, 150, connected=False
        )
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(mask[3, 3, 3], "...but never the seed voxel itself")


class ConnectivityTest(unittest.TestCase):
    """``connected`` keeps the seed's component only."""

    def _two_blobs(self):
        image = np.zeros((15, 15, 15), dtype=np.int16)
        image[7, 7, 7] = 200          # the seed
        image[7, 7, 8] = 200          # face-touching -> same component
        image[7, 7, 11] = 200         # in the sphere, in range, separate
        return image

    def test_connected_drops_the_unrelated_blob(self):
        image = self._two_blobs()
        box, mask = sphere_threshold_mask(image, (7, 7, 7), 5.0, ISO, 150, 250)
        full = _expand(box, mask, image.shape)
        self.assertEqual(
            sorted(map(tuple, np.argwhere(full))), [(7, 7, 7), (7, 7, 8)]
        )

    def test_not_connected_keeps_everything_in_range(self):
        image = self._two_blobs()
        box, mask = sphere_threshold_mask(
            image, (7, 7, 7), 5.0, ISO, 150, 250, connected=False
        )
        full = _expand(box, mask, image.shape)
        self.assertEqual(int(full.sum()), 3)
        self.assertTrue(np.array_equal(
            full, _brute_force(image, (7, 7, 7), 5.0, ISO, 150, 250)))

    def test_connected_is_a_subset_of_unconnected(self):
        image = self._two_blobs()
        args = (image, (7, 7, 7), 5.0, ISO, 150, 250)
        _, strict = sphere_threshold_mask(*args, connected=True)
        _, loose = sphere_threshold_mask(*args, connected=False)
        self.assertEqual(int((strict & ~loose).sum()), 0)

    def test_connectivity_6_versus_26(self):
        image = np.zeros((11, 11, 11), dtype=np.int16)
        image[5, 5, 5] = 200
        image[6, 6, 6] = 200  # corner-touching only
        seed = (5, 5, 5)
        _, mask26 = sphere_threshold_mask(
            image, seed, 4.0, ISO, 150, 250, connectivity=26
        )
        _, mask6 = sphere_threshold_mask(
            image, seed, 4.0, ISO, 150, 250, connectivity=6
        )
        self.assertEqual(int(mask26.sum()), 2)
        self.assertEqual(int(mask6.sum()), 1, "corner neighbours are not 6-connected")

    def test_connectivity_18_splits_corners_but_not_edges(self):
        image = np.zeros((11, 11, 11), dtype=np.int16)
        image[5, 5, 5] = 200
        image[6, 6, 5] = 200  # edge-touching: 18-connected
        image[4, 4, 4] = 200  # corner-touching: not 18-connected
        _, mask = sphere_threshold_mask(
            image, (5, 5, 5), 4.0, ISO, 150, 250, connectivity=18
        )
        self.assertEqual(int(mask.sum()), 2)

    def test_connectivity_is_validated_only_when_connected(self):
        """Current behaviour, not necessarily the intended one (see notes)."""
        image = np.full((9, 9, 9), 100, dtype=np.int16)
        with self.assertRaises(ValueError):
            sphere_threshold_mask(image, (4, 4, 4), 2.0, ISO, 0, 200, connectivity=7)
        # ...but with connected=False the bad value is accepted silently
        _, mask = sphere_threshold_mask(
            image, (4, 4, 4), 2.0, ISO, 0, 200, connected=False, connectivity=7
        )
        self.assertGreater(int(mask.sum()), 0)

    def test_the_hole_in_a_lesion_splits_it(self):
        """A dark shell around the seed must stop the growth at the shell."""
        image = np.full((15, 15, 15), 200, dtype=np.int16)
        image[5:10, 5:10, 5:10] = 0
        image[7, 7, 7] = 200  # island inside the dark cube
        _, mask = sphere_threshold_mask(image, (7, 7, 7), 6.0, ISO, 150, 250)
        self.assertEqual(int(mask.sum()), 1)


class InputValidationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((10, 10, 10), 100, dtype=np.int16)

    def test_seed_outside_the_volume_raises(self):
        for seed in ((10, 0, 0), (0, -1, 0), (0, 0, 10), (-1, -1, -1)):
            with self.assertRaises(ValueError):
                sphere_threshold_mask(self.image, seed, 1.0, ISO, 0, 200)

    def test_non_3d_image_raises(self):
        for shape in ((10, 10), (10, 10, 10, 1), (10,)):
            with self.assertRaises(ValueError):
                sphere_threshold_mask(
                    np.zeros(shape, dtype=np.int16), (0,) * 3, 1.0, ISO, 0, 200
                )

    def test_bad_spacing_raises(self):
        for spacing in ((1.0, 1.0), (1.0, 0.0, 1.0), (1.0, -1.0, 1.0), (1.0,) * 4):
            with self.assertRaises(ValueError):
                sphere_threshold_mask(self.image, (5, 5, 5), 1.0, spacing, 0, 200)

    def test_source_image_is_never_modified(self):
        before = self.image.copy()
        sphere_threshold_mask(self.image, (5, 5, 5), 3.0, ISO, 0, 200)
        self.assertTrue(np.array_equal(self.image, before))

    def test_non_contiguous_input_is_accepted(self):
        """maskio hands over a transposed (F-contiguous) view."""
        view = np.full((10, 12, 14), 100, dtype=np.int16).transpose(2, 1, 0)
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        box, mask = sphere_threshold_mask(view, (5, 5, 5), 2.0, ISO, 0, 200)
        self.assertTrue(np.array_equal(
            _expand(box, mask, view.shape),
            _brute_force(np.asarray(view), (5, 5, 5), 2.0, ISO, 0, 200)))

    def test_float_seed_indices_are_truncated(self):
        _, from_float = sphere_threshold_mask(self.image, (5.0, 5.9, 5.2), 2.0, ISO, 0, 200)
        _, from_int = sphere_threshold_mask(self.image, (5, 5, 5), 2.0, ISO, 0, 200)
        self.assertTrue(np.array_equal(from_float, from_int))


class RandomisedAgreementTest(unittest.TestCase):
    """Random volumes, random seeds: boxed result == whole-volume definition."""

    def test_agrees_with_brute_force(self):
        rng = np.random.RandomState(20240827)
        for _ in range(25):
            shape = tuple(int(n) for n in rng.randint(5, 18, size=3))
            image = rng.randint(-200, 400, size=shape).astype(np.int16)
            seed = tuple(int(rng.randint(0, n)) for n in shape)
            radius = float(rng.uniform(0.0, 9.0))
            spacing = tuple(float(v) for v in rng.uniform(0.3, 2.5, size=3))
            lower = float(rng.randint(-200, 200))
            upper = lower + float(rng.randint(0, 300))
            box, mask = sphere_threshold_mask(
                image, seed, radius, spacing, lower, upper, connected=False
            )
            expected = _brute_force(image, seed, radius, spacing, lower, upper)
            self.assertTrue(
                np.array_equal(_expand(box, mask, shape), expected),
                "seed=%s r=%.2f spacing=%s window=(%s, %s)"
                % (seed, radius, spacing, lower, upper),
            )

    def test_connected_result_is_the_seed_component_of_the_loose_one(self):
        from scipy import ndimage

        rng = np.random.RandomState(7)
        for _ in range(15):
            image = rng.randint(0, 3, size=(14, 14, 14)).astype(np.int16)
            seed = (7, 7, 7)
            image[seed] = 2
            box, loose = sphere_threshold_mask(
                image, seed, 4.0, ISO, 2, 2, connected=False
            )
            _, strict = sphere_threshold_mask(image, seed, 4.0, ISO, 2, 2)
            labels, _n = ndimage.label(loose, ndimage.generate_binary_structure(3, 3))
            seed_rel = tuple(s - b.start for s, b in zip(seed, box))
            self.assertTrue(np.array_equal(strict, labels == labels[seed_rel]))


@unittest.skipIf(_FLATTEN is None, "_flattenToSeedSlice could not be compiled")
class TwoDFlattenRuleTest(unittest.TestCase):
    """The real ``_flattenToSeedSlice``, compiled out of the effect file.

    It takes ``(box, mask, seedIjk, extent)`` where ``box`` is in the source
    image's *relative* index space (the caller has already subtracted the VTK
    extent origin) and ``seedIjk`` is *absolute* -- so the plane index it has to
    reconstruct is ``seedIjk[axis] - extent[2 * axis] - box[axis].start``.
    """

    def setUp(self):
        self.mask = np.ones((5, 5, 5), dtype=bool)
        self.box = (slice(10, 15), slice(20, 25), slice(30, 35))
        self.zero_extent = (0, 99, 0, 99, 0, 99)

    def _seed_for(self, box, plane_offsets=(2, 2, 2), extent_origin=(0, 0, 0)):
        """Absolute seed sitting ``plane_offsets`` into ``box``."""
        return tuple(
            b.start + off + origin
            for b, off, origin in zip(box, plane_offsets, extent_origin)
        )

    def test_keeps_only_the_seed_plane(self):
        for axis in range(3):
            seed = self._seed_for(self.box)
            flat = _FLATTEN(_FakeEffect(axis), self.box, self.mask, seed, self.zero_extent)
            self.assertEqual(int(flat.sum()), 25, "one 5x5 plane survives")
            kept = np.argwhere(flat)[:, axis]
            self.assertTrue((kept == 2).all(), "and it is the seed's plane, axis %d" % axis)

    def test_kept_plane_is_the_untouched_mask_plane(self):
        mask = np.random.RandomState(3).rand(5, 5, 5) > 0.5
        seed = self._seed_for(self.box, (1, 3, 4))
        flat = _FLATTEN(_FakeEffect(1), self.box, mask, seed, self.zero_extent)
        self.assertTrue(np.array_equal(flat[:, 3, :], mask[:, 3, :]))
        self.assertEqual(int(flat.sum()), int(mask[:, 3, :].sum()))

    def test_extent_origin_is_subtracted(self):
        """A volume whose VTK extent does not start at 0 must still land right."""
        extent = (10, 200, 20, 200, 30, 200)
        origin = (extent[0], extent[2], extent[4])
        seed = self._seed_for(self.box, (0, 0, 4), origin)
        flat = _FLATTEN(_FakeEffect(2), self.box, self.mask, seed, extent)
        self.assertEqual(int(flat[:, :, 4].sum()), 25)
        self.assertEqual(int(flat.sum()), 25)

    def test_no_slice_plane_leaves_the_whole_ball(self):
        """Drawn in a 3D view: better to keep the ball than to guess an axis."""
        seed = self._seed_for(self.box)
        flat = _FLATTEN(_FakeEffect(None), self.box, self.mask, seed, self.zero_extent)
        self.assertTrue(np.array_equal(flat, self.mask))

    def test_seed_outside_its_own_box_leaves_the_mask_alone(self):
        seed = self._seed_for(self.box, (99, 2, 2))
        flat = _FLATTEN(_FakeEffect(0), self.box, self.mask, seed, self.zero_extent)
        self.assertTrue(np.array_equal(flat, self.mask))

    def test_mask_is_not_modified_in_place(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        seed = self._seed_for(self.box)
        _FLATTEN(_FakeEffect(0), self.box, mask, seed, self.zero_extent)
        self.assertTrue(mask.all(), "the ball is still needed for the readout")

    def test_flattening_a_real_sphere_gives_a_disc(self):
        """End to end: the maths module's ball, then the effect's 2D rule."""
        image = np.full((21, 21, 21), 100, dtype=np.int16)
        seed = (10, 10, 10)
        box, mask = sphere_threshold_mask(image, seed, 3.0, ISO, 0, 200)
        flat = _FLATTEN(_FakeEffect(2), box, mask, seed, (0, 20, 0, 20, 0, 20))
        full = _expand(box, flat, image.shape)
        # a radius-3 disc on a unit grid: 29 voxels (counted from the definition)
        disc = np.argwhere(full)
        self.assertTrue((disc[:, 2] == 10).all(), "one k-slice only")
        expected = int((
            (np.indices((21, 21))[0] - 10) ** 2 + (np.indices((21, 21))[1] - 10) ** 2
            <= 9 + 1e-9).sum())
        self.assertEqual(len(disc), expected)


class FlattenRuleReachableTest(unittest.TestCase):
    def test_rule_was_actually_loaded(self):
        """Fails loudly if the method is renamed, rather than silently skipping."""
        self.assertIsNotNone(
            _FLATTEN,
            "_flattenToSeedSlice is no longer compilable out of %s -- the 2D rule "
            "is then untested" % _EFFECT_PATH,
        )


if __name__ == "__main__":
    unittest.main()
