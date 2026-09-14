"""Unit tests for GTReviewLib.maskio.

Run with::

    PythonSlicer -m unittest discover -s Testing -p 'test_maskio.py' -v
"""

import contextlib
import errno
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

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

    @contextlib.contextmanager
    def recorded_fsyncs(self, fsync_error=None):
        """Record every ``os.open`` and ``os.fsync`` while write_mask runs.

        ``opens`` holds ``(path, flags, fd)``; ``fsyncs`` holds
        ``(fd, flags that fd was opened with, path)``.  With *fsync_error* each
        fsync raises it instead of flushing.
        """
        real_open, real_fsync = os.open, os.fsync
        opens, fsyncs = [], []

        def recording_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            opens.append((os.fspath(path), flags, fd))
            return fd

        def recording_fsync(fd):
            opened = [(flags, path) for path, flags, number in opens if number == fd]
            flags, path = opened[-1] if opened else (None, None)
            fsyncs.append((fd, flags, path))
            if fsync_error is not None:
                raise fsync_error
            return real_fsync(fd)

        with mock.patch.object(maskio.os, "open", recording_open), \
                mock.patch.object(maskio.os, "fsync", recording_fsync):
            yield opens, fsyncs

    def test_the_temp_is_flushed_through_a_writable_descriptor(self):
        # Windows refuses to flush through a read-only handle, and the save
        # used to skip the flush there without a word
        with self.recorded_fsyncs() as (opens, fsyncs):
            write_mask(self.path, np.ones(self.geom.size, np.uint8), self.geom)

        self.assertEqual(len(fsyncs), 1)
        fd, flags, path = fsyncs[0]
        self.assertIsNotNone(flags, "fsync got a descriptor write_mask did not open")
        self.assertEqual(os.path.dirname(path), self.tmpdir)
        self.assertTrue(os.path.basename(path).startswith(".gtr-"), path)
        self.assertTrue(flags & (os.O_WRONLY | os.O_RDWR), oct(flags))
        # opening it must not truncate what SimpleITK just wrote
        self.assertFalse(flags & getattr(os, "O_TRUNC", 0), oct(flags))
        self.assertEqual(flags & getattr(os, "O_BINARY", 0), getattr(os, "O_BINARY", 0))
        np.testing.assert_array_equal(
            read_mask(self.path)[0], np.ones(self.geom.size, np.uint8)
        )
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ["case_reviewed_seg.nii.gz"])

    def test_a_filesystem_without_fsync_still_saves(self):
        error = OSError(errno.EINVAL, "Invalid argument")
        with self.recorded_fsyncs(fsync_error=error) as (opens, fsyncs):
            write_mask(self.path, np.ones(self.geom.size, np.uint8), self.geom)
        self.assertEqual(len(fsyncs), 1)
        # the descriptor is closed even though the flush failed; checked before
        # read_mask can open anything under the same number
        with self.assertRaises(OSError):
            os.fstat(fsyncs[0][0])
        np.testing.assert_array_equal(
            read_mask(self.path)[0], np.ones(self.geom.size, np.uint8)
        )
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ["case_reviewed_seg.nii.gz"])


# The real functions, captured before any test patches them, so a stand-in can
# refuse some calls and hand the rest on.
REAL_REPLACE = os.replace
REAL_REMOVE = os.remove

RETRY_SETTINGS = ("RETRY_ON_PERMISSION_ERROR", "RETRY_ATTEMPTS", "RETRY_DELAY_S")


def refused(times, real, error=None):
    """A stand-in for ``os.replace`` / ``os.remove`` that fails *times* times.

    Until then every call raises *error* (by default the ``PermissionError``
    Windows gives while another process holds the file); later calls go to
    *real*.  ``fake.calls`` records the arguments of every call.
    """
    calls = []

    def fake(*args):
        calls.append(args)
        if len(calls) <= times:
            if error is not None:
                raise error
            raise PermissionError(
                errno.EACCES,
                "The process cannot access the file because it is being used "
                "by another process",
                args[-1],
            )
        return real(*args)

    fake.calls = calls
    return fake


def is_read_only(path):
    return not os.stat(path).st_mode & stat.S_IWUSR


def comparable_mode(mode):
    """The part of *mode* this platform keeps.

    Windows stores only a read-only attribute and reports every file as 0o666
    or 0o444, so a 0o640 set there can only be checked by its owner write bit.
    """
    return mode & stat.S_IWUSR if os.name == "nt" else mode


def windows_replace(src, dst):
    """``os.replace`` as ``MoveFileEx`` behaves: a read-only destination is refused."""
    if os.path.exists(dst) and is_read_only(dst):
        raise PermissionError(errno.EACCES, "Access is denied", dst)
    return REAL_REPLACE(src, dst)


def windows_remove(path):
    """``os.remove`` as ``DeleteFile`` behaves: a read-only file is refused."""
    if is_read_only(path):  # a missing file raises FileNotFoundError, as os.remove does
        raise PermissionError(errno.EACCES, "Access is denied", path)
    return REAL_REMOVE(path)


@contextlib.contextmanager
def windows_file_semantics():
    """Make renames and deletes refuse read-only files the way Windows does."""
    with mock.patch.object(maskio.os, "replace", windows_replace), \
            mock.patch.object(maskio.os, "remove", windows_remove):
        yield


class RetrySettingsMixin:
    """Restores maskio's retry settings after each test and records the pauses.

    ``time.sleep`` is replaced for the test, so ``self.sleep`` holds one call
    per pause and no test actually waits.
    """

    def setUp(self):
        super().setUp()
        self.saved_settings = {name: getattr(maskio, name) for name in RETRY_SETTINGS}

        def restore():
            for name, value in self.saved_settings.items():
                setattr(maskio, name, value)

        self.addCleanup(restore)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = self._tmp.name
        maskio.RETRY_ATTEMPTS = 8
        maskio.RETRY_DELAY_S = 0.125
        patcher = mock.patch.object(maskio.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def write_bytes(self, name, payload):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def read_bytes(self, path):
        with open(path, "rb") as handle:
            return handle.read()


class RetryDefaultsTest(RetrySettingsMixin, unittest.TestCase):
    def test_retrying_is_on_for_windows_only(self):
        self.assertEqual(self.saved_settings["RETRY_ON_PERMISSION_ERROR"], os.name == "nt")

    def test_defaults_wait_about_a_second(self):
        self.assertGreaterEqual(self.saved_settings["RETRY_ATTEMPTS"], 5)
        self.assertLessEqual(self.saved_settings["RETRY_ATTEMPTS"], 10)
        self.assertGreaterEqual(self.saved_settings["RETRY_DELAY_S"], 0.1)
        self.assertLessEqual(self.saved_settings["RETRY_DELAY_S"], 0.2)


class ReplaceFileTest(RetrySettingsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.src = self.write_bytes(".gtr-new.nii.gz", b"new")
        self.dst = self.write_bytes("case_reviewed_seg.nii.gz", b"old")

    def test_replaces_the_destination(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        maskio.replace_file(self.src, self.dst)
        self.assertEqual(self.read_bytes(self.dst), b"new")
        self.assertFalse(os.path.exists(self.src))
        self.sleep.assert_not_called()

    def test_retries_until_the_other_process_lets_go(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        fake = refused(3, REAL_REPLACE)
        with mock.patch.object(maskio.os, "replace", fake):
            maskio.replace_file(self.src, self.dst)
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(self.read_bytes(self.dst), b"new")
        self.assertFalse(os.path.exists(self.src))
        self.assertEqual(self.sleep.call_args_list, [mock.call(0.125)] * 3)

    def test_gives_up_after_retry_attempts(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        maskio.RETRY_ATTEMPTS = 5
        fake = refused(100, REAL_REPLACE)
        with mock.patch.object(maskio.os, "replace", fake):
            with self.assertRaises(PermissionError):
                maskio.replace_file(self.src, self.dst)
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual(self.sleep.call_count, 4)
        self.assertEqual(self.read_bytes(self.dst), b"old")
        self.assertEqual(self.read_bytes(self.src), b"new")

    def test_does_not_retry_when_the_flag_is_off(self):
        maskio.RETRY_ON_PERMISSION_ERROR = False
        fake = refused(1, REAL_REPLACE)
        with mock.patch.object(maskio.os, "replace", fake):
            with self.assertRaises(PermissionError):
                maskio.replace_file(self.src, self.dst)
        self.assertEqual(len(fake.calls), 1)
        self.sleep.assert_not_called()
        self.assertEqual(self.read_bytes(self.dst), b"old")

    def test_other_os_errors_are_not_retried(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        for error in (OSError(errno.EIO, "I/O error"), OSError(errno.ENOSPC, "No space left")):
            with self.subTest(errno=error.errno):
                fake = refused(1, REAL_REPLACE, error=error)
                with mock.patch.object(maskio.os, "replace", fake):
                    with self.assertRaises(OSError) as ctx:
                        maskio.replace_file(self.src, self.dst)
                self.assertIs(ctx.exception, error)
                self.assertEqual(len(fake.calls), 1)
        with self.assertRaises(FileNotFoundError):
            maskio.replace_file(os.path.join(self.tmpdir, "missing.nii.gz"), self.dst)
        self.sleep.assert_not_called()


class RemoveFileTest(RetrySettingsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path = self.write_bytes("case_reviewed_seg.nii.gz", b"review")

    def tearDown(self):
        if os.path.exists(self.path):
            os.chmod(self.path, 0o600)

    def test_removes_the_file(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        maskio.remove_file(self.path)
        self.assertFalse(os.path.exists(self.path))
        self.sleep.assert_not_called()

    def test_retries_until_the_other_process_lets_go(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        fake = refused(2, REAL_REMOVE)
        with mock.patch.object(maskio.os, "remove", fake):
            maskio.remove_file(self.path)
        self.assertEqual(len(fake.calls), 3)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self.sleep.call_args_list, [mock.call(0.125)] * 2)

    def test_gives_up_after_retry_attempts(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        maskio.RETRY_ATTEMPTS = 6
        fake = refused(100, REAL_REMOVE)
        with mock.patch.object(maskio.os, "remove", fake):
            with self.assertRaises(PermissionError):
                maskio.remove_file(self.path)
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(self.sleep.call_count, 5)
        self.assertTrue(os.path.exists(self.path))

    def test_does_not_retry_when_the_flag_is_off(self):
        maskio.RETRY_ON_PERMISSION_ERROR = False
        fake = refused(1, REAL_REMOVE)
        with mock.patch.object(maskio.os, "remove", fake):
            with self.assertRaises(PermissionError):
                maskio.remove_file(self.path)
        self.assertEqual(len(fake.calls), 1)
        self.sleep.assert_not_called()
        self.assertTrue(os.path.exists(self.path))

    def test_other_os_errors_are_not_retried(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        error = OSError(errno.EIO, "I/O error")
        fake = refused(1, REAL_REMOVE, error=error)
        with mock.patch.object(maskio.os, "remove", fake):
            with self.assertRaises(OSError) as ctx:
                maskio.remove_file(self.path)
        self.assertIs(ctx.exception, error)
        self.assertEqual(len(fake.calls), 1)
        with self.assertRaises(FileNotFoundError):
            maskio.remove_file(os.path.join(self.tmpdir, "missing.nii.gz"))
        self.sleep.assert_not_called()

    def test_clears_the_read_only_attribute(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        os.chmod(self.path, 0o444)
        with windows_file_semantics():
            maskio.remove_file(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_a_delete_that_never_happens_keeps_the_file_read_only(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        os.chmod(self.path, 0o444)
        fake = refused(100, REAL_REMOVE)
        with mock.patch.object(maskio.os, "remove", fake):
            with self.assertRaises(PermissionError):
                maskio.remove_file(self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o444)
        self.assertEqual(self.read_bytes(self.path), b"review")

    def test_a_writable_locked_file_keeps_its_mode(self):
        maskio.RETRY_ON_PERMISSION_ERROR = True
        os.chmod(self.path, 0o640)
        fake = refused(100, REAL_REMOVE)
        with mock.patch.object(maskio.os, "remove", fake):
            with self.assertRaises(PermissionError):
                maskio.remove_file(self.path)
        self.assertEqual(comparable_mode(stat.S_IMODE(os.stat(self.path).st_mode)),
                         comparable_mode(0o640))

    def test_read_only_is_left_alone_when_the_flag_is_off(self):
        maskio.RETRY_ON_PERMISSION_ERROR = False
        os.chmod(self.path, 0o444)
        with windows_file_semantics():
            with self.assertRaises(PermissionError):
                maskio.remove_file(self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o444)


class WindowsSavingTest(RetrySettingsMixin, unittest.TestCase):
    """write_mask against the file semantics of Windows."""

    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmpdir, "case_reviewed_seg.nii.gz")
        self.geom = MaskGeometry(**EXACT_GEOMETRY)
        self.first = np.zeros(self.geom.size, np.uint8)
        self.first[0, 0, 0] = 2
        self.second = np.ones(self.geom.size, np.uint8)

    def tearDown(self):
        if os.path.exists(self.path):
            os.chmod(self.path, 0o600)

    def mode(self, path):
        return stat.S_IMODE(os.stat(path).st_mode)

    @contextlib.contextmanager
    def recorded_replaces(self):
        """Record ``(temp path, temp mode)`` for every replace write_mask makes."""
        original = maskio.replace_file
        records = []

        def recording(src, dst):
            records.append((src, dst, stat.S_IMODE(os.stat(src).st_mode)))
            return original(src, dst)

        with mock.patch.object(maskio, "replace_file", recording):
            yield records

    def test_overwrites_a_read_only_reviewed_mask(self):
        for flag in (True, False):
            with self.subTest(retry=flag):
                maskio.RETRY_ON_PERMISSION_ERROR = flag
                if os.path.exists(self.path):
                    os.chmod(self.path, 0o600)
                write_mask(self.path, self.first, self.geom)
                os.chmod(self.path, 0o444)

                with windows_file_semantics(), self.recorded_replaces() as records:
                    write_mask(self.path, self.second, self.geom)

                np.testing.assert_array_equal(read_mask(self.path)[0], self.second)
                self.assertEqual(self.mode(self.path), 0o444)
                self.assertEqual(os.listdir(self.tmpdir), [os.path.basename(self.path)])
                self.assertEqual(len(records), 1)
                # the temp never carried the read-only mode into the rename
                self.assertTrue(records[0][2] & stat.S_IWUSR)
                self.assertEqual(comparable_mode(records[0][2] & ~stat.S_IWUSR),
                                 comparable_mode(0o444 & ~stat.S_IWUSR))

    def test_overwrite_keeps_a_writable_mode(self):
        write_mask(self.path, self.first, self.geom)
        os.chmod(self.path, 0o640)
        write_mask(self.path, self.second, self.geom)
        self.assertEqual(comparable_mode(self.mode(self.path)), comparable_mode(0o640))

    def test_failed_replace_cleans_up_and_keeps_the_read_only_mask(self):
        write_mask(self.path, self.first, self.geom)
        os.chmod(self.path, 0o444)
        before = self.read_bytes(self.path)
        maskio.RETRY_ON_PERMISSION_ERROR = True
        maskio.RETRY_ATTEMPTS = 3
        locked = refused(100, REAL_REPLACE)

        with windows_file_semantics(), mock.patch.object(maskio.os, "replace", locked):
            with self.assertRaises(PermissionError):
                write_mask(self.path, self.second, self.geom)

        self.assertEqual(len(locked.calls), 3)
        self.assertEqual(os.listdir(self.tmpdir), [os.path.basename(self.path)])
        self.assertEqual(self.read_bytes(self.path), before)
        self.assertEqual(self.mode(self.path), 0o444)

    def test_failed_write_next_to_a_read_only_mask_leaves_no_temp(self):
        write_mask(self.path, self.first, self.geom)
        os.chmod(self.path, 0o444)
        before = self.read_bytes(self.path)

        def exploding_write(*args, **kwargs):
            with open(args[1], "wb") as handle:
                handle.write(b"\x1f\x8b truncated garbage")
            raise RuntimeError("disk went away")

        with windows_file_semantics(), \
                mock.patch.object(maskio.sitk, "WriteImage", exploding_write):
            with self.assertRaises(RuntimeError):
                write_mask(self.path, self.second, self.geom)

        self.assertEqual(os.listdir(self.tmpdir), [os.path.basename(self.path)])
        self.assertEqual(self.read_bytes(self.path), before)
        self.assertEqual(self.mode(self.path), 0o444)

    def test_rides_out_a_transient_lock_on_the_destination(self):
        write_mask(self.path, self.first, self.geom)
        maskio.RETRY_ON_PERMISSION_ERROR = True
        locked = refused(2, REAL_REPLACE)
        with mock.patch.object(maskio.os, "replace", locked):
            write_mask(self.path, self.second, self.geom)
        self.assertEqual(len(locked.calls), 3)
        self.assertEqual(os.listdir(self.tmpdir), [os.path.basename(self.path)])
        np.testing.assert_array_equal(read_mask(self.path)[0], self.second)

    def test_temp_name_is_short_and_next_to_the_destination(self):
        # Slicer on Windows has no long-path support, so the temp path must not
        # be longer than the reviewed path it is renamed to.
        names = (
            "a_reviewed_seg.nii.gz",
            "a_reviewed_seg.nii",
            "YG_78CQZ7VA3H2G_27_reviewed_seg.nii.gz",
        )
        for name in names:
            with self.subTest(name=name):
                path = os.path.join(self.tmpdir, name)
                with self.recorded_replaces() as records:
                    write_mask(path, self.first, self.geom)
                self.assertEqual(len(records), 1)
                temp = records[0][0]
                self.assertEqual(os.path.dirname(temp), os.path.dirname(os.path.abspath(path)))
                self.assertTrue(os.path.basename(temp).startswith(".gtr-"), temp)
                self.assertTrue(temp.endswith(".nii.gz" if name.endswith(".gz") else ".nii"))
                self.assertLessEqual(len(temp), len(os.path.abspath(path)))
                os.remove(path)


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
