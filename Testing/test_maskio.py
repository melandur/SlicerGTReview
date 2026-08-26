"""Unit tests for GTReviewLib.maskio.

Run with::

    PythonSlicer -m unittest discover -s Testing -p 'test_maskio.py' -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GTReview"),
)

from GTReviewLib import maskio  # noqa: E402
from GTReviewLib.maskio import MaskGeometry, read_geometry, read_mask, write_mask  # noqa: E402


# A geometry whose every number is exactly representable in float32, with a
# non-identity (axis-permuting + flipping) direction matrix and anisotropic
# spacing.  NIfTI stores the qform/sform in float32 and ITK re-derives the
# direction by normalising the sform columns in double precision, so only
# dyadic values survive a round trip bit-for-bit; a genuinely oblique matrix
# comes back ~1e-8 off (see test_roundtrip_oblique_geometry_is_compatible).
EXACT_GEOMETRY = dict(
    origin=(-10.5, 3.25, 7.0),
    spacing=(0.5, 1.25, 3.0),
    direction=(0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0),
    size=(3, 4, 5),
)

# Real geometry copied from
# 04_Groundtruthed/01_Yale/batch_01/YG_78CQZ7VA3H2G_27/..._seg.nii.gz
# (obliquely acquired, as all Yale data is).
OBLIQUE_DIRECTION = (
    0.9942999338705345,
    0.07371474992889149,
    -0.07703101581027359,
    0.07422982761792492,
    -0.9972337635910274,
    0.0038410000506768002,
    0.07653479076122086,
    0.009537106321932757,
    0.997021298329102,
)
OBLIQUE_ORIGIN = (-100.5965805053711, 118.34095764160156, -82.13697052001953)
OBLIQUE_SPACING = (0.898438036441803, 0.898438036441803, 0.9000005722045898)

REAL_CASE_DIR = (
    "/home/melandur/Neosoma Inc. Dropbox/Neosoma Inc. R&D AI/01_Annotation/METS/"
    "04_Groundtruthed/01_Yale/batch_01/YG_78CQZ7VA3H2G_27"
)
REAL_SEG = os.path.join(REAL_CASE_DIR, "YG_78CQZ7VA3H2G_27_seg.nii.gz")


def asymmetric_labelmap(shape=(3, 4, 5)):
    """A labelmap whose value encodes its own [i, j, k] index.

    ``value = 1 + i + 10*j + 100*k`` -> any axis swap or flip changes the array,
    so comparing it after a round trip pins the index order down completely.
    """
    i, j, k = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    return (1 + i + 10 * j + 100 * k).astype(np.uint16)


class GeometryTest(unittest.TestCase):
    def test_from_image(self):
        image = sitk.Image(3, 4, 5, sitk.sitkUInt8)
        image.SetOrigin(EXACT_GEOMETRY["origin"])
        image.SetSpacing(EXACT_GEOMETRY["spacing"])
        image.SetDirection(EXACT_GEOMETRY["direction"])

        geom = MaskGeometry.from_image(image)
        self.assertEqual(geom.size, (3, 4, 5))
        self.assertEqual(geom.origin, EXACT_GEOMETRY["origin"])
        self.assertEqual(geom.spacing, EXACT_GEOMETRY["spacing"])
        self.assertEqual(geom.direction, EXACT_GEOMETRY["direction"])
        self.assertEqual(geom, MaskGeometry(**EXACT_GEOMETRY))

    def test_comparable_and_hashable(self):
        a = MaskGeometry(**EXACT_GEOMETRY)
        b = MaskGeometry(**EXACT_GEOMETRY)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        self.assertNotEqual(a, MaskGeometry(**dict(EXACT_GEOMETRY, size=(3, 4, 6))))

    def test_coerces_sequences(self):
        geom = MaskGeometry(
            origin=np.array([0.0, 0.0, 0.0]),
            spacing=[1, 2, 3],
            direction=np.eye(3).ravel(),
            size=np.array([2, 3, 4]),
        )
        self.assertIsInstance(geom.origin, tuple)
        self.assertEqual(geom.spacing, (1.0, 2.0, 3.0))
        self.assertEqual(geom.size, (2, 3, 4))
        self.assertTrue(all(isinstance(v, int) for v in geom.size))
        self.assertEqual(geom.shape_ijk, (2, 3, 4))
        self.assertAlmostEqual(geom.voxel_volume_mm3, 6.0)

    def test_bad_lengths_rejected(self):
        with self.assertRaises(ValueError):
            MaskGeometry(origin=(0, 0), spacing=(1, 1, 1), direction=np.eye(3).ravel(), size=(1, 1, 1))
        with self.assertRaises(ValueError):
            MaskGeometry(origin=(0, 0, 0), spacing=(1, 1, 1), direction=(1, 0, 0, 1), size=(1, 1, 1))

    def test_is_compatible_tolerates_float32_qform_noise(self):
        """73% of real image/mask pairs differ at ~1e-8; that must not matter."""
        base = MaskGeometry(
            origin=OBLIQUE_ORIGIN,
            spacing=OBLIQUE_SPACING,
            direction=OBLIQUE_DIRECTION,
            size=(232, 256, 192),
        )
        noisy = MaskGeometry(
            origin=OBLIQUE_ORIGIN,
            spacing=tuple(s + 6e-8 for s in OBLIQUE_SPACING),
            direction=tuple(d + 5.2e-8 for d in OBLIQUE_DIRECTION),
            size=(232, 256, 192),
        )
        self.assertNotEqual(base, noisy)  # exact == would reject it
        self.assertTrue(base.is_compatible(noisy))
        self.assertTrue(noisy.is_compatible(base))
        self.assertIsNone(base.mismatch_reason(noisy))

    def test_is_compatible_rejects_real_differences(self):
        base = MaskGeometry(**EXACT_GEOMETRY)

        different_size = MaskGeometry(**dict(EXACT_GEOMETRY, size=(3, 4, 6)))
        self.assertFalse(base.is_compatible(different_size))
        self.assertIn("size", base.mismatch_reason(different_size))

        different_spacing = MaskGeometry(**dict(EXACT_GEOMETRY, spacing=(0.5, 1.25, 3.01)))
        self.assertFalse(base.is_compatible(different_spacing))
        self.assertIn("spacing", base.mismatch_reason(different_spacing))

        shifted = MaskGeometry(**dict(EXACT_GEOMETRY, origin=(-10.5, 3.25, 8.0)))
        self.assertFalse(base.is_compatible(shifted))
        self.assertIn("origin", base.mismatch_reason(shifted))

        rotated = MaskGeometry(**dict(EXACT_GEOMETRY, direction=tuple(np.eye(3).ravel())))
        self.assertFalse(base.is_compatible(rotated))
        self.assertIn("direction", base.mismatch_reason(rotated))

        self.assertFalse(base.is_compatible("not a geometry"))
        self.assertIsNotNone(base.mismatch_reason(object()))

    def test_is_compatible_honours_tol(self):
        base = MaskGeometry(**EXACT_GEOMETRY)
        off = MaskGeometry(**dict(EXACT_GEOMETRY, spacing=(0.5, 1.25, 3.0005)))
        self.assertFalse(base.is_compatible(off))  # default tol 1e-4
        self.assertTrue(base.is_compatible(off, tol=1e-2))

    def test_is_compatible_accepts_a_sitk_image(self):
        image = sitk.Image(3, 4, 5, sitk.sitkUInt8)
        image.SetOrigin(EXACT_GEOMETRY["origin"])
        image.SetSpacing(EXACT_GEOMETRY["spacing"])
        image.SetDirection(EXACT_GEOMETRY["direction"])
        self.assertTrue(MaskGeometry(**EXACT_GEOMETRY).is_compatible(image))

    def test_direction_matrix(self):
        matrix = MaskGeometry(**EXACT_GEOMETRY).direction_matrix()
        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(abs(float(np.linalg.det(matrix))), 1.0)


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.path = os.path.join(self.tmpdir, "case_reviewed_seg.nii.gz")

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_geometry_is_bit_identical(self):
        geom = MaskGeometry(**EXACT_GEOMETRY)
        array = asymmetric_labelmap(geom.size) % 3

        write_mask(self.path, array, geom)
        back, geom_back = read_mask(self.path)

        self.assertEqual(geom_back.size, geom.size)
        self.assertEqual(geom_back.origin, geom.origin)
        self.assertEqual(geom_back.spacing, geom.spacing)
        self.assertEqual(geom_back.direction, geom.direction)
        self.assertEqual(geom_back, geom)
        self.assertTrue(geom.is_compatible(geom_back))
        np.testing.assert_array_equal(back, array)

    def test_roundtrip_oblique_geometry_is_compatible(self):
        geom = MaskGeometry(
            origin=OBLIQUE_ORIGIN,
            spacing=OBLIQUE_SPACING,
            direction=OBLIQUE_DIRECTION,
            size=(4, 5, 6),
        )
        array = (asymmetric_labelmap(geom.size) % 3).astype(np.uint8)

        write_mask(self.path, array, geom)
        back, geom_back = read_mask(self.path)

        np.testing.assert_array_equal(back, array)
        self.assertEqual(geom_back.size, geom.size)
        self.assertTrue(geom.is_compatible(geom_back))
        for name in ("origin", "spacing", "direction"):
            delta = np.max(
                np.abs(np.asarray(getattr(geom_back, name)) - np.asarray(getattr(geom, name)))
            )
            self.assertLess(delta, 1e-6, "{} drifted by {:.3e}".format(name, delta))

    def test_ijk_index_order_survives_the_round_trip(self):
        geom = MaskGeometry(**EXACT_GEOMETRY)
        array = asymmetric_labelmap(geom.size)
        self.assertEqual(array.shape, (3, 4, 5))

        write_mask(self.path, array, geom)
        back, geom_back = read_mask(self.path)

        self.assertEqual(back.shape, array.shape)
        self.assertEqual(back.shape, tuple(geom_back.size))
        np.testing.assert_array_equal(back, array)
        self.assertEqual(int(back[1, 2, 3]), 1 + 1 + 20 + 300)

        # and the file itself is in SimpleITK's own [k, j, i] order
        raw = sitk.ReadImage(self.path)
        self.assertEqual(raw.GetSize(), (3, 4, 5))
        raw_kji = sitk.GetArrayFromImage(raw)
        self.assertEqual(raw_kji.shape, (5, 4, 3))
        np.testing.assert_array_equal(raw_kji.transpose(2, 1, 0), array)
        for i, j, k in ((0, 0, 0), (1, 2, 3), (2, 3, 4)):
            self.assertEqual(int(raw_kji[k, j, i]), int(array[i, j, k]))

    def test_read_mask_returns_contiguous_integer_array(self):
        geom = MaskGeometry(**EXACT_GEOMETRY)
        write_mask(self.path, asymmetric_labelmap(geom.size) % 3, geom)
        back, _ = read_mask(self.path)
        self.assertTrue(back.flags["C_CONTIGUOUS"])
        self.assertIn(back.dtype.kind, "iu")

    def test_read_geometry_matches_read_mask(self):
        geom = MaskGeometry(**EXACT_GEOMETRY)
        write_mask(self.path, np.zeros(geom.size, np.uint8), geom)
        self.assertEqual(read_geometry(self.path), read_mask(self.path)[1])
        self.assertEqual(read_geometry(self.path), geom)

    def test_all_zero_mask_roundtrips(self):
        geom = MaskGeometry(**EXACT_GEOMETRY)
        array = np.zeros(geom.size, np.uint8)
        write_mask(self.path, array, geom)
        back, _ = read_mask(self.path)
        np.testing.assert_array_equal(back, array)

    def test_float_nifti_is_rounded_and_cast(self):
        # a float volume written by something else entirely
        values_ijk = np.zeros((3, 4, 5), np.float32)
        values_ijk[0, 0, 0] = 1.2
        values_ijk[1, 0, 0] = 2.7
        values_ijk[2, 0, 0] = -0.4
        values_ijk[0, 1, 0] = np.nan
        values_ijk[0, 2, 0] = np.inf
        image = sitk.GetImageFromArray(values_ijk.transpose(2, 1, 0))
        MaskGeometry(**EXACT_GEOMETRY).apply_to(image)
        sitk.WriteImage(image, self.path, True)

        back, _ = read_mask(self.path)
        self.assertIn(back.dtype.kind, "iu")
        self.assertTrue(back.flags["C_CONTIGUOUS"])
        self.assertEqual(int(back[0, 0, 0]), 1)
        self.assertEqual(int(back[1, 0, 0]), 3)
        self.assertEqual(int(back[2, 0, 0]), 0)
        self.assertEqual(int(back[0, 1, 0]), 0)
        self.assertEqual(int(back[0, 2, 0]), 0)


class DtypeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.path = os.path.join(self.tmpdir, "m.nii.gz")
        self.geom = MaskGeometry(**EXACT_GEOMETRY)

    def tearDown(self):
        self._tmp.cleanup()

    def stored_pixel_type(self):
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.path)
        reader.ReadImageInformation()
        return reader.GetPixelIDValue()

    def test_default_is_uint8(self):
        array = np.zeros(self.geom.size, np.int16)
        array[0, 0, 0] = 2
        write_mask(self.path, array, self.geom)
        self.assertEqual(self.stored_pixel_type(), sitk.sitkUInt8)
        back, _ = read_mask(self.path)
        np.testing.assert_array_equal(back, array)

    def test_promotes_to_uint16_above_255(self):
        array = np.zeros(self.geom.size, np.int32)
        array[0, 0, 0] = 255
        array[1, 0, 0] = 300
        write_mask(self.path, array, self.geom)  # dtype defaults to uint8
        self.assertEqual(self.stored_pixel_type(), sitk.sitkUInt16)
        back, _ = read_mask(self.path)
        self.assertEqual(int(back[1, 0, 0]), 300)
        np.testing.assert_array_equal(back, array)

    def test_255_still_fits_uint8(self):
        array = np.zeros(self.geom.size, np.int32)
        array[0, 0, 0] = 255
        write_mask(self.path, array, self.geom)
        self.assertEqual(self.stored_pixel_type(), sitk.sitkUInt8)

    def test_explicit_dtype_is_honoured(self):
        array = np.zeros(self.geom.size, np.uint8)
        array[0, 0, 0] = 2
        write_mask(self.path, array, self.geom, dtype=np.uint16)
        self.assertEqual(self.stored_pixel_type(), sitk.sitkUInt16)

    def test_bool_array_is_accepted(self):
        array = np.zeros(self.geom.size, bool)
        array[1, 1, 1] = True
        write_mask(self.path, array, self.geom)
        back, _ = read_mask(self.path)
        self.assertEqual(int(back[1, 1, 1]), 1)
        self.assertEqual(int(back.sum()), 1)

    def test_float_array_is_rounded_on_write(self):
        array = np.zeros(self.geom.size, np.float32)
        array[0, 0, 0] = 1.4
        array[1, 0, 0] = 1.6
        write_mask(self.path, array, self.geom)
        back, _ = read_mask(self.path)
        self.assertEqual(int(back[0, 0, 0]), 1)
        self.assertEqual(int(back[1, 0, 0]), 2)

    def test_float_dtype_request_is_rejected(self):
        with self.assertRaises(ValueError):
            write_mask(self.path, np.zeros(self.geom.size, np.uint8), self.geom, dtype=np.float32)
        self.assertFalse(os.path.exists(self.path))

    def test_non_finite_write_is_rejected(self):
        array = np.zeros(self.geom.size, np.float32)
        array[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            write_mask(self.path, array, self.geom)
        self.assertFalse(os.path.exists(self.path))


class RefusalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.path = os.path.join(self.tmpdir, "m.nii.gz")
        self.geom = MaskGeometry(**EXACT_GEOMETRY)

    def tearDown(self):
        self._tmp.cleanup()

    def test_negative_labels_are_refused(self):
        array = np.zeros(self.geom.size, np.int16)
        array[0, 0, 0] = -1
        with self.assertRaises(ValueError) as ctx:
            write_mask(self.path, array, self.geom)
        self.assertIn("negative", str(ctx.exception).lower())
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_negative_float_labels_are_refused(self):
        array = np.zeros(self.geom.size, np.float32)
        array[0, 0, 0] = -1.2
        with self.assertRaises(ValueError):
            write_mask(self.path, array, self.geom)
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_shape_geometry_mismatch_is_refused(self):
        transposed = np.zeros(tuple(reversed(self.geom.size)), np.uint8)  # [k, j, i]
        with self.assertRaises(ValueError) as ctx:
            write_mask(self.path, transposed, self.geom)
        self.assertIn("[i, j, k]", str(ctx.exception))
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_non_3d_array_is_refused(self):
        with self.assertRaises(ValueError):
            write_mask(self.path, np.zeros((3, 4), np.uint8), self.geom)
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_geometry_must_be_a_geometry(self):
        with self.assertRaises(TypeError):
            write_mask(self.path, np.zeros(self.geom.size, np.uint8), (1, 2, 3))


class AtomicityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.path = os.path.join(self.tmpdir, "case_reviewed_seg.nii.gz")
        self.geom = MaskGeometry(**EXACT_GEOMETRY)

    def tearDown(self):
        self._tmp.cleanup()

    def test_successful_write_leaves_no_temp_files(self):
        write_mask(self.path, np.zeros(self.geom.size, np.uint8), self.geom)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ["case_reviewed_seg.nii.gz"])

    def test_overwrite_leaves_no_temp_files(self):
        first = np.zeros(self.geom.size, np.uint8)
        second = np.ones(self.geom.size, np.uint8)
        write_mask(self.path, first, self.geom)
        write_mask(self.path, second, self.geom)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ["case_reviewed_seg.nii.gz"])
        np.testing.assert_array_equal(read_mask(self.path)[0], second)

    def test_failed_write_keeps_the_previous_file_and_cleans_up(self):
        good = np.zeros(self.geom.size, np.uint8)
        good[0, 0, 0] = 2
        write_mask(self.path, good, self.geom)
        with open(self.path, "rb") as handle:
            before = handle.read()

        original_write = maskio.sitk.WriteImage

        def exploding_write(*args, **kwargs):
            # write some bytes into the temp file first, then die: exactly the
            # truncated-file scenario os.replace has to protect against
            with open(args[1], "wb") as handle:
                handle.write(b"\x1f\x8b truncated garbage")
            raise RuntimeError("disk went away")

        maskio.sitk.WriteImage = exploding_write
        try:
            with self.assertRaises(RuntimeError):
                write_mask(self.path, np.ones(self.geom.size, np.uint8), self.geom)
        finally:
            maskio.sitk.WriteImage = original_write

        self.assertEqual(sorted(os.listdir(self.tmpdir)), ["case_reviewed_seg.nii.gz"])
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        np.testing.assert_array_equal(read_mask(self.path)[0], good)

    def test_writes_are_compressed(self):
        big = np.zeros((64, 64, 64), np.uint8)
        geom = MaskGeometry(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=tuple(np.eye(3).ravel()),
            size=(64, 64, 64),
        )
        write_mask(self.path, big, geom)
        self.assertLess(os.path.getsize(self.path), big.nbytes / 10)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(2), b"\x1f\x8b")  # gzip magic

    def test_creates_missing_parent_directory(self):
        nested = os.path.join(self.tmpdir, "sub", "dir", "m.nii.gz")
        write_mask(nested, np.zeros(self.geom.size, np.uint8), self.geom)
        self.assertTrue(os.path.exists(nested))


@unittest.skipUnless(os.path.exists(REAL_SEG), "real sample data not available")
class RealDataTest(unittest.TestCase):
    """Reads one real ground-truth mask.  Never writes into the data tree."""

    def test_reads_real_ground_truth(self):
        array, geom = read_mask(REAL_SEG)

        self.assertEqual(array.shape, (232, 256, 192))  # [i, j, k]
        self.assertEqual(tuple(geom.size), array.shape)
        self.assertEqual(set(np.unique(array).tolist()), {0, 1, 2})
        self.assertIn(array.dtype.kind, "iu")
        self.assertTrue(array.flags["C_CONTIGUOUS"])
        # obliquely acquired: the direction matrix has real off-diagonal terms
        off_diagonal = np.abs(geom.direction_matrix() - np.diag(np.diag(geom.direction_matrix())))
        self.assertGreater(float(off_diagonal.max()), 0.01)

    def test_read_geometry_is_cheap_and_identical(self):
        header = read_geometry(REAL_SEG)
        self.assertEqual(header, read_mask(REAL_SEG)[1])

    def test_real_mask_is_compatible_with_its_image(self):
        """The float32 qform noise between t1c and seg must not be a mismatch."""
        image_path = os.path.join(REAL_CASE_DIR, "YG_78CQZ7VA3H2G_27_t1c.nii.gz")
        if not os.path.exists(image_path):
            self.skipTest("t1c missing")
        mask_geom = read_geometry(REAL_SEG)
        image_geom = read_geometry(image_path)
        self.assertTrue(
            mask_geom.is_compatible(image_geom), mask_geom.mismatch_reason(image_geom)
        )

    def test_real_mask_roundtrips_through_a_temp_dir(self):
        array, geom = read_mask(REAL_SEG)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "YG_78CQZ7VA3H2G_27_reviewed_seg.nii.gz")
            write_mask(out, array, geom)
            self.assertEqual(os.listdir(tmpdir), [os.path.basename(out)])
            back, geom_back = read_mask(out)
            np.testing.assert_array_equal(back, array)
            self.assertEqual(back.dtype, np.uint8)  # int16 source stored as uint8
            self.assertTrue(geom.is_compatible(geom_back), geom.mismatch_reason(geom_back))


if __name__ == "__main__":
    unittest.main(verbosity=2)
