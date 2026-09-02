"""Connected-component lesion analysis for GTReview.

Pure logic module: numpy + scipy.ndimage only.  It must stay importable under a
plain ``PythonSlicer`` interpreter with no Slicer / VTK / Qt available, so do NOT
add imports from ``slicer``, ``vtk`` or ``qt`` here.

Array index order is ``[i, j, k]`` everywhere in this codebase (see maskio.py,
which transposes at the SimpleITK boundary).  ``spacing_ijk`` is in the same
order, i.e. ``spacing_ijk[0]`` is the physical size of one voxel step along the
first array axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage

__all__ = ["Lesion", "find_lesions", "lesion_mask", "sphere_threshold_mask", "CONNECTIVITY_RANKS"]


# scipy's generate_binary_structure(rank=3, connectivity=r) yields:
#   r = 1 ->  6-neighbourhood (face)
#   r = 2 -> 18-neighbourhood (face + edge)
#   r = 3 -> 26-neighbourhood (face + edge + corner)
CONNECTIVITY_RANKS = {6: 1, 18: 2, 26: 3}


@dataclass
class Lesion:
    """One connected component of the binarised mask.

    Attributes
    ----------
    index:
        1-based lesion number.  Assigned *after* sorting by ``voxel_count``
        descending, so ``index == 1`` is always the largest lesion.  It matches
        the value this lesion has in the component map returned by
        :func:`find_lesions`.
    label:
        Dominant (most frequent) non-zero value of the source mask inside the
        component.  Ties resolve to the smallest label value.
    voxel_count:
        Number of voxels in the component.
    volume_mm3:
        ``voxel_count * spacing_i * spacing_j * spacing_k``.
    centroid_ijk:
        An ``(i, j, k)`` voxel index that is guaranteed to lie *inside* the
        component (see :func:`find_lesions` for the snapping rule).
    bbox_ijk:
        ``((i0, i1), (j0, j1), (k0, k1))`` half-open bounds, so
        ``arr[i0:i1, j0:j1, k0:k1]`` is the tight bounding box.
    """

    index: int
    label: int
    voxel_count: int
    volume_mm3: float
    centroid_ijk: Tuple[int, int, int]
    bbox_ijk: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _structure(connectivity: int) -> np.ndarray:
    try:
        rank = CONNECTIVITY_RANKS[int(connectivity)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "connectivity must be one of 6, 18, 26 (got %r)" % (connectivity,)
        )
    return ndimage.generate_binary_structure(3, rank)


def _as_mask_array(mask_ijk) -> np.ndarray:
    arr = np.asarray(mask_ijk)
    if arr.ndim != 3:
        raise ValueError("mask_ijk must be a 3-D array, got shape %r" % (arr.shape,))
    return arr


def _dominant_label(values: np.ndarray) -> int:
    """Most frequent non-zero value; ties -> smallest value.

    ``values`` are the raw mask values of one component.  They are already
    non-zero by construction (the component came from ``mask != 0``), but the
    zero filter is kept so the helper is safe on its own.
    """
    if values.size == 0:
        return 0
    if not np.issubdtype(values.dtype, np.integer):
        values = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    values = values[values != 0]
    if values.size == 0:
        return 0
    uniq, counts = np.unique(values, return_counts=True)  # uniq ascending
    return int(uniq[int(np.argmax(counts))])  # argmax -> first max -> smallest value


def _centroid_inside(
    component: np.ndarray,
    offset: Sequence[int],
    spacing_ijk: Sequence[float],
) -> Tuple[int, int, int]:
    """Centre of mass of ``component`` snapped to a voxel inside it.

    ``component`` is a boolean sub-volume (the lesion's bounding box) and
    ``offset`` is its origin in the full volume.  The centre of mass is rounded
    to the nearest voxel; if that voxel is not part of the component we snap to
    the component voxel that is physically closest (spacing-weighted euclidean
    distance) to the un-rounded centre of mass.  Ties break towards the
    lexicographically smallest ``(i, j, k)``, which keeps the result
    deterministic.

    The snap is fully vectorised over the component's voxels only -- there is no
    Python-level loop over voxels, and nothing scans the whole volume.
    """
    com = np.asarray(ndimage.center_of_mass(component), dtype=np.float64)
    rounded = np.rint(com).astype(np.int64)
    shape = np.asarray(component.shape, dtype=np.int64)
    rounded = np.clip(rounded, 0, shape - 1)
    if component[tuple(rounded)]:
        local = rounded
    else:
        coords = np.argwhere(component)  # C order -> lexicographically ascending
        scale = np.asarray(spacing_ijk, dtype=np.float64).reshape(1, 3)
        delta = (coords.astype(np.float64) - com.reshape(1, 3)) * scale
        d2 = np.einsum("ij,ij->i", delta, delta)
        local = coords[int(np.argmin(d2))]  # argmin -> first minimum
    full = np.asarray(offset, dtype=np.int64) + np.asarray(local, dtype=np.int64)
    return (int(full[0]), int(full[1]), int(full[2]))


def find_lesions(
    mask_ijk,
    spacing_ijk,
    connectivity: int = 26,
    min_voxels: int = 1,
    dilate: int = 0,
) -> Tuple[np.ndarray, List[Lesion]]:
    """Find lesions (connected components) in a segmentation mask.

    Parameters
    ----------
    mask_ijk:
        3-D array indexed ``[i, j, k]``.  Any array-like is accepted: it is not
        required to be contiguous, integer typed, or even writable.  Components
        are found over the **binarised** mask (``mask != 0``), so a lesion that
        spans two label values stays a single lesion.
    spacing_ijk:
        Three floats, the voxel size along ``i``, ``j``, ``k``, in mm.
    connectivity:
        6, 18 or 26.  26 (the default) treats corner-touching voxels as
        connected; anything else raises ``ValueError``.
    min_voxels:
        Components with fewer than ``min_voxels`` voxels are discarded.  Values
        below 1 are treated as 1.
    dilate:
        Grow the binarised mask by this many voxels (with the same
        ``connectivity`` structure) *before* labelling, so fragments separated
        by a gap of up to ``2 * dilate`` voxels are reported as one lesion.
        The labels are then mapped back onto the original voxels only:
        counts, volumes, centroids and boxes never include grown voxels and
        the component map is 0 wherever the input is 0.  0 disables it.

    Returns
    -------
    (component_map, lesions)
        ``component_map`` is an ``int32`` array with the same shape as the input:
        0 for background, ``n`` for the voxels of ``lesions[n - 1]``.  **The map
        is relabelled**: surviving lesions are numbered 1..N contiguously, in the
        same order as the returned list, and components removed by ``min_voxels``
        are 0 in the map (they are indistinguishable from background there --
        filtering is a display concern, so callers that must not lose those
        voxels should keep the original mask, which is never modified).

        ``lesions`` is sorted by ``voxel_count`` descending (ties broken by the
        raw scipy component id, i.e. by first occurrence in ``[i, j, k]`` scan
        order) and ``Lesion.index`` is assigned after that sort, so
        ``lesions[0].index == 1`` is the largest lesion.

        An all-zero (or entirely filtered) mask yields an all-zero map and an
        empty list; it never raises.
    """
    arr = _as_mask_array(mask_ijk)
    structure = _structure(connectivity)

    spacing = np.asarray(spacing_ijk, dtype=np.float64).reshape(-1)
    if spacing.size != 3:
        raise ValueError("spacing_ijk must have 3 elements, got %r" % (spacing_ijk,))
    voxel_volume = float(np.prod(spacing))

    min_voxels = max(1, int(min_voxels))

    # Make the working copy C-contiguous *before* binarising.  maskio hands us a
    # transposed (F-contiguous) view, and ndimage.label on a non-contiguous bool
    # array is ~6x slower (measured 2.9 s vs 0.5 s on a 94 M-voxel volume); doing
    # the copy on the source dtype is also cheaper than copying the bool result.
    # The caller's array is never written to.
    arr = np.ascontiguousarray(arr)
    binary = arr != 0

    dilate = max(0, int(dilate))
    if dilate:
        grown = ndimage.binary_dilation(binary, structure=structure, iterations=dilate)
        raw_map, n_raw = ndimage.label(grown, structure=structure)
        raw_map[~binary] = 0  # bridge the gaps, but keep only real voxels
    else:
        raw_map, n_raw = ndimage.label(binary, structure=structure)
    empty_map = np.zeros(arr.shape, dtype=np.int32)
    if n_raw == 0:
        return empty_map, []

    # bincount over the raw component map: counts[c] == voxels of component c.
    counts = np.bincount(raw_map.ravel(), minlength=n_raw + 1)
    raw_ids = np.arange(1, n_raw + 1)
    raw_counts = counts[1:]

    keep = raw_counts >= min_voxels
    if not np.any(keep):
        return empty_map, []

    kept_ids = raw_ids[keep]
    kept_counts = raw_counts[keep]

    # Sort by voxel count descending, ties by raw component id ascending.
    order = np.lexsort((kept_ids, -kept_counts))
    kept_ids = kept_ids[order]
    kept_counts = kept_counts[order]

    # Relabel: raw component id -> new contiguous 1..N index (0 = dropped).
    lut = np.zeros(n_raw + 1, dtype=np.int32)
    lut[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.int32)
    component_map = lut[raw_map]

    boxes = ndimage.find_objects(raw_map)

    lesions: List[Lesion] = []
    for new_index, (raw_id, count) in enumerate(zip(kept_ids, kept_counts), start=1):
        sl = boxes[int(raw_id) - 1]
        sub_component = raw_map[sl] == raw_id
        sub_values = arr[sl][sub_component]

        offset = (sl[0].start, sl[1].start, sl[2].start)
        bbox = (
            (int(sl[0].start), int(sl[0].stop)),
            (int(sl[1].start), int(sl[1].stop)),
            (int(sl[2].start), int(sl[2].stop)),
        )
        lesions.append(
            Lesion(
                index=int(new_index),
                label=_dominant_label(sub_values),
                voxel_count=int(count),
                volume_mm3=float(int(count) * voxel_volume),
                centroid_ijk=_centroid_inside(sub_component, offset, spacing),
                bbox_ijk=bbox,
            )
        )

    return component_map, lesions


def lesion_mask(component_map, index: int) -> np.ndarray:
    """Boolean mask of a single lesion.

    ``index`` is the 1-based :attr:`Lesion.index` (== the value in the component
    map returned by :func:`find_lesions`).  An index that is not present yields
    an all-False array rather than an error.
    """
    cmap = np.asarray(component_map)
    return cmap == int(index)


def sphere_threshold_mask(
    image_ijk,
    seed_ijk,
    radius_mm: float,
    spacing_ijk,
    lower: float,
    upper: float,
    connected: bool = True,
    connectivity: int = 26,
) -> Tuple[Tuple[slice, slice, slice], np.ndarray]:
    """Grow one lesion from a seed voxel: everything inside a sphere whose
    intensity lies in ``[lower, upper]``.

    The sphere is physical -- ``radius_mm`` around the centre of ``seed_ijk``
    with ``spacing_ijk`` mm per voxel -- so it stays a sphere on anisotropic
    grids; on the 0.9 mm isotropic data here that is the same as a voxel
    radius.  With ``connected`` only the component that contains the seed is
    kept (``connectivity`` as in :func:`find_lesions`), so an unrelated bright
    spot elsewhere in the sphere is not picked up.

    Returns ``(box, mask)``: ``box`` is a tuple of slices into the full
    volume (the sphere's bounding box, clipped) and ``mask`` a boolean array
    of that shape.  The result is a plain voxel mask -- no smoothing or
    interpolation -- so what is added is exactly what is shown.
    """
    image = np.asarray(image_ijk)
    if image.ndim != 3:
        raise ValueError("image_ijk must be a 3-D array, got shape {}".format(image.shape))
    seed = tuple(int(v) for v in seed_ijk)
    if len(seed) != 3 or not all(0 <= s < n for s, n in zip(seed, image.shape)):
        raise ValueError("seed {} lies outside the volume {}".format(seed, image.shape))
    spacing = np.asarray(spacing_ijk, dtype=np.float64).reshape(-1)
    if spacing.size != 3 or np.any(spacing <= 0):
        raise ValueError("spacing_ijk must be three positive floats, got %r" % (spacing_ijk,))
    radius_mm = float(max(0.0, radius_mm))
    lower, upper = float(lower), float(upper)
    if lower > upper:
        lower, upper = upper, lower

    half = [int(np.floor(radius_mm / sp)) for sp in spacing]
    box = tuple(
        slice(max(0, s - h), min(n, s + h + 1)) for s, h, n in zip(seed, half, image.shape)
    )
    sub = image[box]
    axes = [
        (np.arange(b.start, b.stop, dtype=np.float64) - s) * sp
        for b, s, sp in zip(box, seed, spacing)
    ]
    distance2 = (
        axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2
    )
    sphere = distance2 <= radius_mm ** 2 + 1e-9
    mask = sphere & (sub >= lower) & (sub <= upper)

    seed_rel = tuple(s - b.start for s, b in zip(seed, box))
    if connected:
        if not mask[seed_rel]:
            return box, np.zeros_like(mask)
        labels, _count = ndimage.label(mask, structure=_structure(connectivity))
        mask = labels == labels[seed_rel]
    return box, mask
