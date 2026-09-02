"""Unit tests for GTReviewLib.lesions -- plain unittest, no Slicer needed.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_lesions.py' -v
"""

import os
import sys
import time
import unittest

import numpy as np

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

try:  # normal package import (GTReviewLib may be a namespace package)
    from GTReviewLib import lesions
    from GTReviewLib.lesions import Lesion, find_lesions, lesion_mask
except ImportError:  # pragma: no cover - fallback: load straight from the file
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "gtreview_lesions", os.path.join(_MODULE_DIR, "GTReviewLib", "lesions.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    Lesion, find_lesions, lesion_mask = _mod.Lesion, _mod.find_lesions, _mod.lesion_mask


ISO = (1.0, 1.0, 1.0)


class TestDilate(unittest.TestCase):
    """``dilate`` bridges small gaps but never adds voxels to the result."""

    def _two_blobs(self, gap):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:4, 2:4, 2:4] = 1
        mask[2:4, 2:4, 4 + gap:6 + gap] = 2
        return mask

    def test_default_is_off(self):
        mask = self._two_blobs(gap=1)
        self.assertEqual(len(find_lesions(mask, ISO)[1]), 2)
        self.assertEqual(len(find_lesions(mask, ISO, dilate=0)[1]), 2)

    def test_one_voxel_gap_bridged(self):
        mask = self._two_blobs(gap=1)
        cmap, found = find_lesions(mask, ISO, dilate=1)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].voxel_count, 16)  # real voxels only
        self.assertTrue(np.array_equal(cmap != 0, mask != 0))
        self.assertEqual(found[0].bbox_ijk, ((2, 4), (2, 4), (2, 7)))

    def test_two_voxel_gap_bridged_by_one_voxel_dilation(self):
        # both sides grow by one, so a gap of two closes
        self.assertEqual(len(find_lesions(self._two_blobs(gap=2), ISO, dilate=1)[1]), 1)

    def test_three_voxel_gap_stays_apart(self):
        self.assertEqual(len(find_lesions(self._two_blobs(gap=3), ISO, dilate=1)[1]), 2)
        self.assertEqual(len(find_lesions(self._two_blobs(gap=3), ISO, dilate=2)[1]), 1)

    def test_min_voxels_counts_real_voxels(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[3, 3, 3] = 1  # one voxel, would be 27 after dilation
        self.assertEqual(len(find_lesions(mask, ISO, dilate=1, min_voxels=2)[1]), 0)

    def test_negative_dilate_is_off(self):
        self.assertEqual(len(find_lesions(self._two_blobs(gap=1), ISO, dilate=-1)[1]), 2)


class TestEmptyAndDegenerate(unittest.TestCase):
    def test_all_zero_mask(self):
        mask = np.zeros((8, 9, 10), dtype=np.int16)
        cmap, lesions = find_lesions(mask, ISO)
        self.assertEqual(lesions, [])
        self.assertEqual(cmap.shape, mask.shape)
        self.assertEqual(cmap.dtype, np.int32)
        self.assertEqual(int(cmap.max()), 0)

    def test_single_voxel_lesion_survives(self):
        """1-voxel specks are 6.7% of real ground-truth components -- keep them."""
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[2, 3, 4] = 2
        cmap, lesions = find_lesions(mask, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].voxel_count, 1)
        self.assertEqual(lesions[0].label, 2)
        self.assertEqual(lesions[0].centroid_ijk, (2, 3, 4))
        self.assertEqual(lesions[0].bbox_ijk, ((2, 3), (3, 4), (4, 5)))
        self.assertEqual(int(cmap[2, 3, 4]), 1)

    def test_bad_connectivity_raises(self):
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        for bad in (4, 8, 27, 0, -1, 3.5, None, "eight"):
            with self.assertRaises(ValueError):
                find_lesions(mask, ISO, connectivity=bad)
        # an int-like value that does name a supported neighbourhood is accepted
        self.assertEqual(find_lesions(mask, ISO, connectivity="26")[1], [])

    def test_non_3d_raises(self):
        with self.assertRaises(ValueError):
            find_lesions(np.zeros((4, 4), dtype=np.uint8), ISO)

    def test_bad_spacing_raises(self):
        with self.assertRaises(ValueError):
            find_lesions(np.zeros((4, 4, 4), dtype=np.uint8), (1.0, 1.0))


class TestTwoSeparatedCubes(unittest.TestCase):
    """A 27-voxel cube and an 8-voxel cube, far apart."""

    def setUp(self):
        self.mask = np.zeros((12, 12, 12), dtype=np.int16)
        self.mask[1:4, 1:4, 1:4] = 1  # 27 voxels
        self.mask[8:10, 8:10, 8:10] = 2  # 8 voxels
        self.cmap, self.lesions = find_lesions(self.mask, ISO)

    def test_counts_and_sort_order(self):
        self.assertEqual(len(self.lesions), 2)
        self.assertEqual([l.voxel_count for l in self.lesions], [27, 8])
        self.assertEqual([l.index for l in self.lesions], [1, 2])
        self.assertEqual([l.label for l in self.lesions], [1, 2])

    def test_component_map_matches_indices(self):
        self.assertEqual(int(self.cmap[2, 2, 2]), 1)
        self.assertEqual(int(self.cmap[9, 9, 9]), 2)
        self.assertEqual(int(np.count_nonzero(self.cmap == 1)), 27)
        self.assertEqual(int(np.count_nonzero(self.cmap == 2)), 8)
        self.assertEqual(int(np.count_nonzero(self.cmap)), 35)

    def test_bbox_and_centroid(self):
        big, small = self.lesions
        self.assertEqual(big.bbox_ijk, ((1, 4), (1, 4), (1, 4)))
        self.assertEqual(big.centroid_ijk, (2, 2, 2))
        self.assertEqual(small.bbox_ijk, ((8, 10), (8, 10), (8, 10)))
        # centre of mass (8.5, 8.5, 8.5) rounds to a voxel that is inside.
        self.assertTrue(self.mask[small.centroid_ijk] != 0)

    def test_volume_iso(self):
        self.assertAlmostEqual(self.lesions[0].volume_mm3, 27.0)
        self.assertAlmostEqual(self.lesions[1].volume_mm3, 8.0)

    def test_input_not_modified(self):
        before = self.mask.copy()
        find_lesions(self.mask, ISO)
        np.testing.assert_array_equal(self.mask, before)

    def test_lesion_mask(self):
        m1 = lesion_mask(self.cmap, 1)
        self.assertEqual(m1.dtype, np.bool_)
        self.assertEqual(int(m1.sum()), 27)
        np.testing.assert_array_equal(m1, self.mask == 1)
        # An index that does not exist yields all-False rather than raising.
        self.assertEqual(int(lesion_mask(self.cmap, 99).sum()), 0)

    def test_tie_break_is_deterministic(self):
        """Equal-size lesions keep scan order, and repeated runs agree."""
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[5, 5, 5] = 1
        mask[8, 8, 8] = 1
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual(
            [l.centroid_ijk for l in lesions], [(1, 1, 1), (5, 5, 5), (8, 8, 8)]
        )
        for _ in range(3):
            _, again = find_lesions(mask, ISO)
            self.assertEqual([l.centroid_ijk for l in again],
                             [l.centroid_ijk for l in lesions])


class TestConnectivity(unittest.TestCase):
    """26-connectivity is load-bearing: scipy's default is 6-connectivity."""

    def test_corner_touching(self):
        mask = np.zeros((6, 6, 6), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[2, 2, 2] = 1  # corner (vertex) neighbour only
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=26)[1]), 1)
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=18)[1]), 2)
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=6)[1]), 2)

    def test_edge_touching(self):
        mask = np.zeros((6, 6, 6), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[2, 2, 1] = 1  # edge neighbour: 18- and 26-connected, not 6
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=26)[1]), 1)
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=18)[1]), 1)
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=6)[1]), 2)

    def test_face_touching_always_one(self):
        mask = np.zeros((6, 6, 6), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[2, 1, 1] = 1
        for conn in (6, 18, 26):
            self.assertEqual(len(find_lesions(mask, ISO, connectivity=conn)[1]), 1)

    def test_diagonal_chain_merges_under_26(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        for n in range(5):
            mask[1 + n, 1 + n, 1 + n] = 1
        cmap26, l26 = find_lesions(mask, ISO, connectivity=26)
        self.assertEqual(len(l26), 1)
        self.assertEqual(l26[0].voxel_count, 5)
        self.assertEqual(int(cmap26.max()), 1)
        self.assertEqual(len(find_lesions(mask, ISO, connectivity=6)[1]), 5)


class TestCentroidSnapping(unittest.TestCase):
    """A C-shaped lesion: its centre of mass falls in the hollow, outside it."""

    def setUp(self):
        # C opening towards +j, lying in the k = 0 plane.
        self.mask = np.zeros((7, 7, 1), dtype=np.uint8)
        self.mask[0, 0:5, 0] = 1  # top bar
        self.mask[4, 0:5, 0] = 1  # bottom bar
        self.mask[0:5, 0, 0] = 1  # spine
        self.assertEqual(int(self.mask.sum()), 13)

    def test_centre_of_mass_is_really_outside(self):
        coords = np.argwhere(self.mask)
        com = coords.mean(axis=0)
        np.testing.assert_allclose(com, [2.0, 20.0 / 13.0, 0.0], atol=1e-9)
        rounded = tuple(int(v) for v in np.rint(com))
        self.assertEqual(rounded, (2, 2, 0))
        self.assertEqual(int(self.mask[rounded]), 0)  # the hollow

    def test_snaps_to_a_voxel_inside_the_lesion(self):
        _, lesions = find_lesions(self.mask, ISO)
        self.assertEqual(len(lesions), 1)
        c = lesions[0].centroid_ijk
        self.assertEqual(int(self.mask[c]), 1)
        # Nearest lesion voxel to (2, 1.538, 0) with isotropic spacing.
        self.assertEqual(c, (2, 0, 0))

    def test_snap_uses_physical_distance(self):
        """With j-spacing 10x, the physically nearest inside voxel changes."""
        _, lesions = find_lesions(self.mask, (1.0, 10.0, 1.0))
        c = lesions[0].centroid_ijk
        self.assertEqual(int(self.mask[c]), 1)
        # (0,2,0) and (4,2,0) tie; the lexicographically smallest wins.
        self.assertEqual(c, (0, 2, 0))

    def test_hollow_ring_every_lesion_centroid_is_inside(self):
        mask = np.zeros((9, 9, 9), dtype=np.uint8)
        mask[2:7, 2:7, 2:7] = 1
        mask[3:6, 3:6, 3:6] = 0  # hollow shell
        cmap, lesions = find_lesions(mask, (0.4297, 0.4297, 0.5))
        self.assertEqual(len(lesions), 1)
        c = lesions[0].centroid_ijk
        self.assertEqual(int(mask[c]), 1)
        self.assertEqual(int(cmap[c]), lesions[0].index)


class TestLabels(unittest.TestCase):
    def test_two_labels_stay_one_lesion(self):
        """24.6% of real GT components contain both label 1 and label 2."""
        mask = np.zeros((10, 10, 10), dtype=np.int16)
        mask[2:5, 2:5, 2:5] = 1  # 27 voxels
        mask[2:5, 5:8, 2:5] = 2  # 27 voxels, face-adjacent to the first block
        cmap, lesions = find_lesions(mask, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].voxel_count, 54)
        self.assertEqual(int(cmap.max()), 1)

    def test_dominant_label_wins(self):
        mask = np.zeros((10, 10, 10), dtype=np.int16)
        mask[2:5, 2:5, 2:5] = 2  # 27 voxels of label 2
        mask[2:5, 5:7, 2:5] = 1  # 18 voxels of label 1
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].voxel_count, 45)
        self.assertEqual(lesions[0].label, 2)

    def test_label_tie_goes_to_smallest_value(self):
        mask = np.zeros((8, 8, 8), dtype=np.int16)
        mask[2, 2, 2] = 3
        mask[2, 2, 3] = 1
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].label, 1)

    def test_labels_above_two_and_negative_values(self):
        mask = np.zeros((8, 8, 8), dtype=np.int16)
        mask[1:3, 1:3, 1:3] = 7
        mask[5, 5, 5] = -1  # non-zero, so it is still a lesion
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual([l.label for l in lesions], [7, -1])


class TestMinVoxels(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((14, 14, 14), dtype=np.uint8)
        self.mask[1:4, 1:4, 1:4] = 1  # 27
        self.mask[7:9, 7:9, 7:9] = 2  # 8
        self.mask[12, 12, 12] = 2  # 1

    def test_default_keeps_everything(self):
        _, lesions = find_lesions(self.mask, ISO)
        self.assertEqual([l.voxel_count for l in lesions], [27, 8, 1])

    def test_filters_and_relabels_contiguously(self):
        cmap, lesions = find_lesions(self.mask, ISO, min_voxels=8)
        self.assertEqual([l.voxel_count for l in lesions], [27, 8])
        self.assertEqual([l.index for l in lesions], [1, 2])
        self.assertEqual(sorted(np.unique(cmap).tolist()), [0, 1, 2])
        # the filtered speck is background in the map ...
        self.assertEqual(int(cmap[12, 12, 12]), 0)
        # ... but the source mask is untouched.
        self.assertEqual(int(self.mask[12, 12, 12]), 2)
        self.assertEqual(int(np.count_nonzero(cmap)), 35)

    def test_filter_everything(self):
        cmap, lesions = find_lesions(self.mask, ISO, min_voxels=1000)
        self.assertEqual(lesions, [])
        self.assertEqual(int(cmap.max()), 0)
        self.assertEqual(cmap.shape, self.mask.shape)

    def test_min_voxels_below_one_is_treated_as_one(self):
        for mv in (0, -5):
            _, lesions = find_lesions(self.mask, ISO, min_voxels=mv)
            self.assertEqual(len(lesions), 3)

    def test_map_index_agrees_with_lesion_index_after_filtering(self):
        cmap, lesions = find_lesions(self.mask, ISO, min_voxels=2)
        for les in lesions:
            self.assertEqual(int(np.count_nonzero(cmap == les.index)), les.voxel_count)
            self.assertEqual(int(cmap[les.centroid_ijk]), les.index)


class TestVolumeMaths(unittest.TestCase):
    def test_anisotropic_spacing(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1  # 8 voxels
        spacing = (0.4297, 0.4297, 0.5)
        _, lesions = find_lesions(mask, spacing)
        self.assertAlmostEqual(
            lesions[0].volume_mm3, 8 * 0.4297 * 0.4297 * 0.5, places=9
        )

    def test_one_millimetre_iso(self):
        mask = np.zeros((6, 6, 6), dtype=np.uint8)
        mask[0:3, 0:3, 0:3] = 1
        _, lesions = find_lesions(mask, (1.0, 1.0, 1.0))
        self.assertAlmostEqual(lesions[0].volume_mm3, 27.0)

    def test_volume_scales_with_voxel_count(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        mask[7, 7, 7] = 1
        spacing = (0.8984, 0.8984, 0.9)
        _, lesions = find_lesions(mask, spacing)
        vox_vol = 0.8984 * 0.8984 * 0.9
        self.assertAlmostEqual(lesions[0].volume_mm3, 27 * vox_vol, places=9)
        self.assertAlmostEqual(lesions[1].volume_mm3, 1 * vox_vol, places=9)


class TestAwkwardInputs(unittest.TestCase):
    def test_non_contiguous_view(self):
        big = np.zeros((10, 10, 20), dtype=np.int16)
        view = big[:, :, ::2]  # non-contiguous, shape (10, 10, 10)
        view[1:4, 1:4, 1:4] = 1
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        cmap, lesions = find_lesions(view, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].voxel_count, 27)
        self.assertEqual(cmap.shape, view.shape)

    def test_transposed_view(self):
        base = np.zeros((6, 7, 8), dtype=np.uint8)
        base[1:3, 1:3, 1:3] = 2
        view = np.transpose(base, (2, 1, 0))
        cmap, lesions = find_lesions(view, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].voxel_count, 8)
        self.assertEqual(lesions[0].label, 2)
        self.assertEqual(cmap.shape, view.shape)

    def test_float_mask(self):
        mask = np.zeros((8, 8, 8), dtype=np.float32)
        mask[2:4, 2:4, 2:4] = 2.0
        mask[6, 6, 6] = 1.0
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual([l.voxel_count for l in lesions], [8, 1])
        self.assertEqual([l.label for l in lesions], [2, 1])
        self.assertTrue(all(isinstance(l.label, int) for l in lesions))

    def test_bool_mask(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual(len(lesions), 1)
        self.assertEqual(lesions[0].label, 1)

    def test_nested_list_input(self):
        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[1, 1, 1] = 1
        _, lesions = find_lesions(data.tolist(), ISO)
        self.assertEqual(len(lesions), 1)

    def test_read_only_array(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        mask.setflags(write=False)
        _, lesions = find_lesions(mask, ISO)
        self.assertEqual(lesions[0].voxel_count, 8)

    def test_scalar_types_are_plain_python(self):
        mask = np.zeros((6, 6, 6), dtype=np.int8)
        mask[1:3, 1:3, 1:3] = 2
        _, lesions = find_lesions(mask, (0.5, 0.5, 0.5))
        les = lesions[0]
        self.assertIsInstance(les.index, int)
        self.assertIsInstance(les.label, int)
        self.assertIsInstance(les.voxel_count, int)
        self.assertIsInstance(les.volume_mm3, float)
        self.assertTrue(all(isinstance(v, int) for v in les.centroid_ijk))
        for lo, hi in les.bbox_ijk:
            self.assertIsInstance(lo, int)
            self.assertIsInstance(hi, int)


class TestPerformance(unittest.TestCase):
    def test_realistic_volume_is_fast(self):
        """192x256x232 int16 -- the SPEC's sample shape, ~11.4M voxels."""
        rng = np.random.default_rng(0)
        mask = np.zeros((192, 256, 232), dtype=np.int16)
        for _ in range(40):
            i, j, k = (
                int(rng.integers(5, 180)),
                int(rng.integers(5, 240)),
                int(rng.integers(5, 220)),
            )
            r = int(rng.integers(1, 6))
            mask[i:i + r, j:j + r, k:k + r] = int(rng.integers(1, 3))

        start = time.time()
        cmap, lesions = find_lesions(mask, (0.8984, 0.8984, 0.9))
        elapsed = time.time() - start

        self.assertGreater(len(lesions), 0)
        self.assertEqual(cmap.shape, mask.shape)
        # Measured well under 1 s; 5 s keeps the assertion CI-safe.
        self.assertLess(elapsed, 5.0, "find_lesions took %.3f s" % elapsed)

    def test_worst_case_shape_is_fast(self):
        """Largest real volume shape in the corpus: 512x512x360 = 94.4M voxels."""
        mask = np.zeros((512, 512, 360), dtype=np.int8)
        mask[100:140, 100:140, 100:140] = 2
        mask[300:310, 300:310, 300:310] = 1
        start = time.time()
        _, lesions = find_lesions(mask, (0.4297, 0.4297, 0.5))
        elapsed = time.time() - start
        self.assertEqual([l.voxel_count for l in lesions], [64000, 1000])
        self.assertLess(elapsed, 5.0, "find_lesions took %.3f s" % elapsed)


class SphereThresholdMaskTest(unittest.TestCase):
    """sphere_threshold_mask: physical sphere, seed-connected, voxel-exact."""

    def _scene(self):
        image = np.zeros((30, 30, 30), dtype=np.int16)
        image[10:20, 10:20, 10:20] = 200          # a bright 10^3 block
        image[15, 15, 15] = 210                   # the seed voxel
        image[12, 12, 12] = 5                     # a dark hole inside the block
        image[15, 15, 25] = 200                   # bright, isolated, 10 voxels away
        return image

    def test_sphere_clips_and_thresholds(self):
        image = self._scene()
        box, mask = lesions.sphere_threshold_mask(image, (15, 15, 15), 3.0, (1, 1, 1), 150, 250)
        self.assertEqual(box, (slice(12, 19), slice(12, 19), slice(12, 19)))
        full = np.zeros(image.shape, bool); full[box] = mask
        self.assertTrue(full[15, 15, 15])
        self.assertTrue(full[15, 15, 18] and not full[15, 15, 19], "radius 3 reaches 3 voxels, not 4")
        self.assertFalse(full[12, 12, 12], "out-of-range voxel inside the sphere is excluded")
        self.assertFalse(full[15, 15, 25], "outside the sphere")
        # every kept voxel is within 3 mm of the seed centre
        idx = np.argwhere(full)
        self.assertTrue((((idx - [15, 15, 15]) ** 2).sum(1) <= 9.0 + 1e-9).all())
        # and equals the brute-force definition
        ii, jj, kk = np.mgrid[0:30, 0:30, 0:30]
        brute = (((ii - 15) ** 2 + (jj - 15) ** 2 + (kk - 15) ** 2) <= 9) & (image >= 150) & (image <= 250)
        self.assertTrue(np.array_equal(full, brute))

    def test_anisotropic_spacing_is_a_physical_sphere(self):
        image = np.full((30, 30, 30), 100, dtype=np.int16)
        box, mask = lesions.sphere_threshold_mask(image, (15, 15, 15), 4.0, (1.0, 1.0, 2.0), 0, 200)
        full = np.zeros(image.shape, bool); full[box] = mask
        self.assertTrue(full[19, 15, 15] and not full[20, 15, 15], "4 voxels along a 1 mm axis")
        self.assertTrue(full[15, 15, 17] and not full[15, 15, 18], "2 voxels along a 2 mm axis")

    def test_connected_keeps_only_the_seed_component(self):
        image = self._scene()
        image[10:20, 10:20, 10:20] = 0
        image[15, 15, 15] = 200
        image[15, 15, 18] = 200                   # in range, in sphere, not connected
        box, mask = lesions.sphere_threshold_mask(image, (15, 15, 15), 4.0, (1, 1, 1), 150, 250)
        full = np.zeros(image.shape, bool); full[box] = mask
        self.assertEqual(np.argwhere(full).tolist(), [[15, 15, 15]])
        box, mask = lesions.sphere_threshold_mask(
            image, (15, 15, 15), 4.0, (1, 1, 1), 150, 250, connected=False
        )
        full = np.zeros(image.shape, bool); full[box] = mask
        self.assertEqual(int(full.sum()), 2)

    def test_edge_seed_zero_radius_and_swapped_range(self):
        image = self._scene()
        box, mask = lesions.sphere_threshold_mask(image, (0, 0, 0), 5.0, (1, 1, 1), -1, 1)
        self.assertEqual(box, (slice(0, 6), slice(0, 6), slice(0, 6)))
        self.assertTrue(mask[0, 0, 0])
        box, mask = lesions.sphere_threshold_mask(image, (15, 15, 15), 0.0, (1, 1, 1), 250, 150)
        self.assertEqual(mask.shape, (1, 1, 1)); self.assertTrue(mask[0, 0, 0])
        box, mask = lesions.sphere_threshold_mask(image, (15, 15, 15), 2.0, (1, 1, 1), 0, 100)
        self.assertEqual(int(mask.sum()), 0, "seed out of range -> nothing")
        with self.assertRaises(ValueError):
            lesions.sphere_threshold_mask(image, (30, 0, 0), 1.0, (1, 1, 1), 0, 1)


if __name__ == "__main__":
    unittest.main()
