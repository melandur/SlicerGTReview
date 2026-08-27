"""NIfTI mask I/O for GTReview — geometry-preserving, SimpleITK only.

**Index order — read this before touching an array anywhere in this codebase.**

Every array that crosses the boundary of this module is indexed ``[i, j, k]``,
i.e. the same axis order SimpleITK uses for ``GetSize()``, ``GetSpacing()`` and
``TransformIndexToPhysicalPoint()``.  SimpleITK's ``GetArrayFromImage()``
returns the *reversed* order ``[k, j, i]`` (numpy/ITK convention).

``read_mask`` transposes ``[k, j, i] -> [i, j, k]`` and ``write_mask``
transposes back.  **That transpose happens here and nowhere else.**  No other
module in GTReview may call ``sitk.GetArrayFromImage`` /
``sitk.GetImageFromArray`` directly; if you find yourself writing
``.transpose(2, 1, 0)`` outside this file, you have a bug.

Consequences for callers:

* ``array.shape == geometry.size`` always holds (both ``(nx, ny, nz)``).
* ``spacing[0]`` is the spacing along ``i``, so
  ``volume_mm3 = voxel_count * spacing[0] * spacing[1] * spacing[2]``.
* An IJK -> RAS/LPS position must go through the full direction matrix
  (``sitk.Image.TransformIndexToPhysicalPoint`` or Slicer's IJKToRAS matrix),
  never ``origin + spacing * index``: the real Yale data is obliquely acquired
  with off-diagonal direction terms up to ~0.33.

Other guarantees:

* ``read_mask`` always returns a C-contiguous **integer** array (float NIfTIs
  are rounded to nearest and cast; non-finite voxels become 0).
* ``write_mask`` writes **atomically** (temp file in the destination directory
  + ``os.replace``) so an interrupted save can never leave a truncated
  ``.nii.gz`` behind, restores origin/spacing/direction bit-for-bit from the
  supplied :class:`MaskGeometry`, writes compressed, picks ``uint8`` unless a
  label exceeds 255 (then ``uint16``), and refuses negative labels.
* Geometry comparison must be **tolerant**: NIfTI stores the qform/sform in
  float32, so two files describing the same volume routinely disagree in the
  8th decimal (observed max delta 6e-8 over the real corpus, in 73% of image /
  mask pairs).  Use :meth:`MaskGeometry.is_compatible`, never ``==``.

This module imports only ``numpy`` and ``SimpleITK`` — no ``slicer``, ``vtk``
or ``qt`` — so it is unit-testable under plain ``PythonSlicer -m unittest``.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

__all__ = [
    "MaskGeometry",
    "read_mask",
    "read_geometry",
    "write_mask",
    "NIFTI_EXTENSIONS",
]

#: Extensions ``write_mask`` accepts / ``read_mask`` expects.
NIFTI_EXTENSIONS = (".nii", ".nii.gz")

#: Default tolerance for geometry comparisons (see module docstring).
DEFAULT_TOL = 1e-4


def _as_float_tuple(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


def _as_int_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class MaskGeometry:
    """Voxel-grid geometry of a volume, in SimpleITK conventions.

    Attributes
    ----------
    origin:
        Physical (LPS) coordinate of voxel ``(0, 0, 0)``, length 3.
    spacing:
        Voxel size along ``i``, ``j``, ``k``, length 3.
    direction:
        Row-major 3x3 direction cosine matrix, length 9.
    size:
        Grid size ``(ni, nj, nk)`` — identical to the ``[i, j, k]`` array shape.

    Instances are frozen and hashable, so ``==`` works, but ``==`` is *exact*
    float equality and will report unequal for two files that merely round-trip
    the same geometry through NIfTI's float32 qform.  For "does this mask
    belong to this image volume?" use :meth:`is_compatible`.
    """

    origin: Tuple[float, ...]
    spacing: Tuple[float, ...]
    direction: Tuple[float, ...]
    size: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_float_tuple(self.origin))
        object.__setattr__(self, "spacing", _as_float_tuple(self.spacing))
        object.__setattr__(self, "direction", _as_float_tuple(self.direction))
        object.__setattr__(self, "size", _as_int_tuple(self.size))
        n = len(self.size)
        if len(self.origin) != n or len(self.spacing) != n:
            raise ValueError(
                "origin/spacing/size length mismatch: {} / {} / {}".format(
                    len(self.origin), len(self.spacing), n
                )
            )
        if len(self.direction) != n * n:
            raise ValueError(
                "direction must hold {} elements for a {}-D geometry, got {}".format(
                    n * n, n, len(self.direction)
                )
            )

    # -- construction ----------------------------------------------------

    @classmethod
    def from_image(cls, image: Any) -> "MaskGeometry":
        """Build from a ``SimpleITK.Image`` (or anything with the same getters)."""
        return cls(
            origin=image.GetOrigin(),
            spacing=image.GetSpacing(),
            direction=image.GetDirection(),
            size=image.GetSize(),
        )

    @classmethod
    def coerce(cls, other: Any) -> "MaskGeometry":
        """Return *other* as a :class:`MaskGeometry` (accepts a SimpleITK image)."""
        if isinstance(other, cls):
            return other
        if hasattr(other, "GetOrigin") and hasattr(other, "GetDirection"):
            return cls.from_image(other)
        raise TypeError(
            "expected a MaskGeometry or a SimpleITK image, got {!r}".format(type(other))
        )

    # -- derived ---------------------------------------------------------

    @property
    def dimension(self) -> int:
        return len(self.size)

    @property
    def shape_ijk(self) -> Tuple[int, ...]:
        """The numpy shape an ``[i, j, k]``-indexed array of this volume has."""
        return self.size

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(np.asarray(self.spacing, dtype=np.float64)))

    def direction_matrix(self) -> np.ndarray:
        n = self.dimension
        return np.asarray(self.direction, dtype=np.float64).reshape(n, n)

    # -- comparison ------------------------------------------------------

    def is_compatible(self, other: Any, tol: float = DEFAULT_TOL) -> bool:
        """True when *other* describes the same voxel grid, within *tol*.

        ``size`` must match exactly; ``origin``, ``spacing`` and ``direction``
        are compared with ``numpy.allclose(atol=tol, rtol=0)``.  The default
        ``tol=1e-4`` leaves three orders of magnitude of headroom over the
        float32 qform noise seen in real data (max observed delta 6e-8) while
        still catching any genuine misregistration, which in practice differs
        grossly (a different ``size``, a millimetre-scale origin shift).

        *other* may be a :class:`MaskGeometry` or a ``SimpleITK.Image``.
        """
        try:
            other = MaskGeometry.coerce(other)
        except TypeError:
            return False
        if tuple(self.size) != tuple(other.size):
            return False
        atol = float(tol)
        return bool(
            np.allclose(self.spacing, other.spacing, rtol=0.0, atol=atol)
            and np.allclose(self.origin, other.origin, rtol=0.0, atol=atol)
            and np.allclose(self.direction, other.direction, rtol=0.0, atol=atol)
        )

    def mismatch_reason(self, other: Any, tol: float = DEFAULT_TOL) -> Optional[str]:
        """Human-readable reason *other* is incompatible, or ``None`` if it is."""
        try:
            other = MaskGeometry.coerce(other)
        except TypeError as exc:
            return str(exc)
        if tuple(self.size) != tuple(other.size):
            return "size {} != {}".format(tuple(self.size), tuple(other.size))
        atol = float(tol)
        for name in ("spacing", "origin", "direction"):
            a = np.asarray(getattr(self, name), dtype=np.float64)
            b = np.asarray(getattr(other, name), dtype=np.float64)
            if not np.allclose(a, b, rtol=0.0, atol=atol):
                return "{} differs by up to {:.3e} (tol {:.3e})".format(
                    name, float(np.max(np.abs(a - b))), atol
                )
        return None

    # -- application -----------------------------------------------------

    def apply_to(self, image: Any) -> Any:
        """Stamp origin/spacing/direction onto *image*; returns *image*."""
        image.SetOrigin(tuple(self.origin))
        image.SetSpacing(tuple(self.spacing))
        image.SetDirection(tuple(self.direction))
        return image


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _to_integer_array(array: np.ndarray) -> np.ndarray:
    """Round/cast *array* to a compact signed-safe integer dtype."""
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    if array.dtype.kind in "iu":
        return array
    if array.dtype.kind not in "fc":
        raise ValueError("unsupported mask pixel type {}".format(array.dtype))
    values = np.asarray(array, dtype=np.float64)
    if array.dtype.kind == "c":  # pragma: no cover - no complex NIfTI in practice
        values = np.real(values)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.rint(values)
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    for dtype in (np.uint8, np.int16, np.uint16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if lo >= info.min and hi <= info.max:
            return values.astype(dtype)
    return values.astype(np.int64)  # pragma: no cover - unreachable in practice


def read_mask(path: str) -> Tuple[np.ndarray, MaskGeometry]:
    """Read a NIfTI mask.

    Returns ``(array_ijk, geometry)`` where ``array_ijk`` is a **C-contiguous
    integer** array indexed ``[i, j, k]`` with ``array_ijk.shape ==
    geometry.size``.  Float volumes are rounded to nearest (non-finite voxels
    become 0); native integer types are preserved as-is so a 94 M-voxel int8
    prediction does not silently quadruple in memory.
    """
    path = os.fspath(path)
    image = sitk.ReadImage(path)
    geometry = MaskGeometry.from_image(image)
    array_kji = sitk.GetArrayFromImage(image)
    if array_kji.ndim != 3:
        raise ValueError(
            "expected a 3-D volume, got {}-D from {}".format(array_kji.ndim, path)
        )
    array_ijk = np.ascontiguousarray(_to_integer_array(array_kji).transpose(2, 1, 0))
    return array_ijk, geometry


def read_geometry(path: str) -> MaskGeometry:
    """Read only the header geometry of *path* — cheap, no pixel data.

    Uses ``sitk.ImageFileReader.ReadImageInformation`` and falls back to a full
    read if that is unavailable for the file format.  The case browser uses
    this to check a mask against the reference image without decompressing
    tens of megabytes.
    """
    path = os.fspath(path)
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        return MaskGeometry(
            origin=reader.GetOrigin(),
            spacing=reader.GetSpacing(),
            direction=reader.GetDirection(),
            size=reader.GetSize(),
        )
    except AttributeError:  # pragma: no cover - very old SimpleITK
        return MaskGeometry.from_image(sitk.ReadImage(path))


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _resolve_output_dtype(array: np.ndarray, dtype: Any) -> np.dtype:
    """Pick the stored pixel type: *dtype*, promoted if it cannot hold the labels."""
    requested = np.dtype(dtype)
    if requested.kind not in "iub":
        raise ValueError(
            "write_mask needs an integer dtype, got {!r}".format(requested)
        )
    if requested.kind == "b":
        requested = np.dtype(np.uint8)
    hi = int(array.max()) if array.size else 0
    if hi <= np.iinfo(requested).max:
        return requested
    for candidate in (np.uint8, np.uint16, np.uint32, np.int64):
        if hi <= np.iinfo(candidate).max:
            return np.dtype(candidate)
    raise ValueError("label value {} is too large to store".format(hi))


def write_mask(
    path: str,
    array_ijk: np.ndarray,
    geometry: Any,
    dtype: Any = np.uint8,
) -> None:
    """Write ``array_ijk`` (indexed ``[i, j, k]``) as a compressed NIfTI mask.

    * ``geometry`` (a :class:`MaskGeometry`, or a ``SimpleITK.Image`` to copy
      from) is restored exactly — pass the geometry that came back from
      :func:`read_mask` for the *source mask*, not one round-tripped through
      Slicer's MRML/RAS conversion, which adds its own float noise.
    * ``geometry.size`` must equal ``array_ijk.shape``; a mismatch usually
      means someone handed over a ``[k, j, i]`` array and raises ``ValueError``.
    * Float arrays are rounded to nearest.  Negative labels are refused.
    * The stored type is *dtype* (default ``uint8``), promoted to ``uint16``
      when a label exceeds 255.
    * The write is atomic: a temp file in the destination directory is written
      and then ``os.replace``-d over *path*, so an interrupted or failing save
      leaves the previous file intact and no partial ``.nii.gz`` behind.
    """
    path = os.fspath(path)
    geometry = MaskGeometry.coerce(geometry)

    array = np.asarray(array_ijk)
    if array.ndim != 3:
        raise ValueError("expected a 3-D [i, j, k] array, got {}-D".format(array.ndim))
    if tuple(array.shape) != tuple(geometry.size):
        raise ValueError(
            "array shape {} does not match geometry size {} — the array must be "
            "indexed [i, j, k]".format(tuple(array.shape), tuple(geometry.size))
        )

    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    elif array.dtype.kind == "f":
        if not np.all(np.isfinite(array)):
            raise ValueError("mask contains non-finite values")
        array = np.rint(array)
    elif array.dtype.kind not in "iu":
        raise ValueError("unsupported array dtype {}".format(array.dtype))

    if array.size and float(array.min()) < 0:
        raise ValueError(
            "refusing to write negative label values (min = {})".format(
                int(array.min()) if array.dtype.kind != "f" else float(array.min())
            )
        )

    out_dtype = _resolve_output_dtype(array, dtype)
    # transpose back to SimpleITK's [k, j, i] — the only place this happens
    array_kji = np.ascontiguousarray(array.transpose(2, 1, 0).astype(out_dtype, copy=False))

    image = sitk.GetImageFromArray(array_kji)
    geometry.apply_to(image)

    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    # SimpleITK picks its format from the SUFFIX, so the scratch file has to
    # carry the destination's own: writing a ".nii" through a ".nii.gz" temp
    # produced a gzip stream under a plain-NIfTI name, which read_mask then
    # could not open.
    lowered = os.path.basename(path).lower()
    suffix = next(
        (ext for ext in sorted(NIFTI_EXTENSIONS, key=len, reverse=True)
         if lowered.endswith(ext)),
        ".nii.gz",
    )
    fd, tmp_path = tempfile.mkstemp(
        prefix=".{}.".format(os.path.basename(path)),
        suffix=suffix,
        dir=directory,
    )
    os.close(fd)
    try:
        try:
            sitk.WriteImage(image, tmp_path, suffix.endswith(".gz"))
        except TypeError:  # pragma: no cover - very old SimpleITK
            writer = sitk.ImageFileWriter()
            writer.SetFileName(tmp_path)
            writer.SetUseCompression(suffix.endswith(".gz"))
            writer.Execute(image)
        try:
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:  # pragma: no cover - fsync unsupported on some FS
            pass
        # mkstemp creates 0600; keep the mode the file would have had (or the
        # mode it already has, when overwriting a previous reviewed mask).
        try:
            if os.path.exists(path):
                mode = stat.S_IMODE(os.stat(path).st_mode)
            else:
                umask = os.umask(0)
                os.umask(umask)
                mode = 0o666 & ~umask
            os.chmod(tmp_path, mode)
        except OSError:  # pragma: no cover - chmod unsupported on some FS
            pass
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover
            pass
        raise
