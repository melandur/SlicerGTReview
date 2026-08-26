"""Case discovery for the GTReview Slicer extension.

Standard library only — no ``slicer`` / ``vtk`` / ``qt`` / numpy imports, so this
module can be unit tested with plain ``PythonSlicer -m unittest``.

Naming rules (SPEC.md, implemented verbatim)
--------------------------------------------
For a file ``<stem>.nii.gz`` inside a case dir::

    key = stem[len(case_id) + 1:]   if stem starts with case_id + "_"
    key = stem                      otherwise

Classify by key (case-insensitive), first match wins:

1. ``reviewed_seg``                                   -> REVIEWED
2. key equals or ends with ``seg``/``mask``/``label``/``labels``/``gt`` -> MASK
3. otherwise                                          -> IMAGE

Rule 1 must be checked before rule 2 because ``reviewed_seg`` ends with ``seg``;
without it the tool would re-list its own output as an input mask on the next
discovery pass.  Likewise ``pred_seg`` ends with ``seg`` and is therefore a mask,
which a naive ``key == "seg"`` test would get wrong.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "IMAGE",
    "MASK",
    "REVIEWED",
    "REVIEWED_KEY",
    "REVIEWED_SUFFIX",
    "MASK_KEYS",
    "NIFTI_EXTENSIONS",
    "Case",
    "classify_key",
    "parse_case_files",
    "discover_cases",
    "is_nifti",
    "nifti_stem",
    "natural_key",
]

IMAGE, MASK, REVIEWED = "image", "mask", "reviewed"

#: key of the review output produced by this tool
REVIEWED_KEY = "reviewed_seg"
#: filename suffix (without extension) of the review output
REVIEWED_SUFFIX = "_" + REVIEWED_KEY
#: a key equal to, or ending with, one of these is a mask (checked after REVIEWED)
MASK_KEYS: Tuple[str, ...] = ("seg", "mask", "label", "labels", "gt")
#: accepted volume extensions, longest first so ``.nii.gz`` wins over ``.gz``
NIFTI_EXTENSIONS: Tuple[str, ...] = (".nii.gz", ".nii")

# Dropbox / OS artefacts that may appear next to real data in a synced tree.
_JUNK_PATTERNS = (
    re.compile(r"conflicted copy", re.IGNORECASE),
    re.compile(r"\(\d+\)\s*$"),          # "foo (1).nii.gz" -> stem ends with "(1)"
    re.compile(r"^~\$"),                 # office lock files
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def is_nifti(filename: str) -> bool:
    """True if *filename* looks like a usable NIfTI volume.

    Accepts ``.nii`` and ``.nii.gz`` (case-insensitively).  Rejects dotfiles and
    Dropbox conflict copies, which a live synced tree can grow at any time.
    """
    name = os.path.basename(filename)
    if not name or name.startswith("."):
        return False
    lowered = name.lower()
    if not any(lowered.endswith(ext) for ext in NIFTI_EXTENSIONS):
        return False
    stem = nifti_stem(name)
    return not any(pattern.search(stem) for pattern in _JUNK_PATTERNS)


def nifti_stem(filename: str) -> str:
    """``"a_b_seg.nii.gz"`` -> ``"a_b_seg"``.  Non-nifti names are returned as-is."""
    name = os.path.basename(filename)
    lowered = name.lower()
    for ext in NIFTI_EXTENSIONS:
        if lowered.endswith(ext):
            return name[: len(name) - len(ext)]
    return name


def natural_key(text: str) -> Tuple:
    """Sort key that orders embedded, unpadded integers numerically.

    Yale ids carry an unpadded trailing timepoint (``..._9``, ``..._10``), so a
    plain ``sorted()`` interleaves timepoints (``_10, _13, _9``) and Prev/Next
    looks scrambled to the annotator.  Digit runs compare as ``(0, int)`` and
    text runs as ``(1, str)`` so the tuple elements are always comparable.
    """
    parts = re.split(r"(\d+)", text or "")
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts if p != "")


def _key_for(stem: str, case_id: Optional[str]) -> str:
    """Apply the SPEC naming rule to one file stem."""
    if case_id:
        prefix = case_id + "_"
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _extension_rank(filename: str) -> int:
    """Prefer ``.nii.gz`` over ``.nii`` when both map to the same key."""
    return 0 if filename.lower().endswith(".nii.gz") else 1


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify_key(key: str) -> str:
    """Classify a filename key as :data:`IMAGE`, :data:`MASK` or :data:`REVIEWED`."""
    normalized = (key or "").strip().lower()

    # 1. the review output — checked first, it ends with "seg"
    if normalized == REVIEWED_KEY or normalized.endswith(REVIEWED_SUFFIX):
        return REVIEWED

    # 2. masks
    for mask_key in MASK_KEYS:
        if normalized == mask_key or normalized.endswith(mask_key):
            return MASK

    # 3. everything else is an image sequence
    return IMAGE


# --------------------------------------------------------------------------- #
# Case
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """One case directory: its image sequences, its masks and its review output."""

    case_id: str
    directory: str
    images: Dict[str, str] = field(default_factory=dict)
    masks: Dict[str, str] = field(default_factory=dict)
    reviewed_path: str = ""

    @property
    def is_reviewed(self) -> bool:
        """True when ``<case_id>_reviewed_seg.nii.gz`` already exists on disk."""
        try:
            return bool(self.reviewed_path) and os.path.isfile(self.reviewed_path)
        except OSError:  # pragma: no cover - unreadable mount
            return False

    def default_mask_path(
        self, preferred: Sequence[str] = ("seg", "gt", "pred_seg")
    ) -> Optional[str]:
        """reviewed_path if it exists, else first preferred key present, else any mask.

        In batch_01 ``seg`` exists for 46/50 cases; the fall-through to
        ``pred_seg`` covers the remaining 4, so both paths matter.
        """
        if self.is_reviewed:
            return self.reviewed_path

        lookup = {key.lower(): path for key, path in self.masks.items()}
        for key in preferred or ():
            path = lookup.get(str(key).strip().lower())
            if path:
                return path

        if not self.masks:
            return None
        # deterministic "any mask"
        first = sorted(self.masks, key=lambda k: (natural_key(k), k))[0]
        return self.masks[first]

    # convenience, used by the browser UI
    def has_masks(self) -> bool:
        return bool(self.masks)

    def has_images(self) -> bool:
        return bool(self.images)


# --------------------------------------------------------------------------- #
# parsing / discovery
# --------------------------------------------------------------------------- #
def _list_niftis(directory: str) -> List[str]:
    """Sorted list of nifti filenames directly inside *directory* (never raises)."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted((n for n in names if is_nifti(n)), key=lambda n: (natural_key(n), n))


def parse_case_files(case_dir: str, case_id: Optional[str] = None) -> Case:
    """Build a :class:`Case` from the nifti files directly inside *case_dir*.

    Non-nifti clutter, sub-directories, dotfiles and Dropbox conflict copies are
    ignored.  An unreadable or missing directory yields an empty ``Case`` rather
    than raising.
    """
    directory = os.path.abspath(os.path.expanduser(str(case_dir)))
    if not case_id:
        case_id = os.path.basename(directory.rstrip(os.sep)) or directory

    images: Dict[str, str] = {}
    masks: Dict[str, str] = {}

    for name in _list_niftis(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        key = _key_for(nifti_stem(name), case_id)
        if not key:
            continue
        kind = classify_key(key)
        if kind == REVIEWED:
            continue  # never listed as an input mask
        bucket = masks if kind == MASK else images
        existing = bucket.get(key)
        if existing is None or _extension_rank(name) < _extension_rank(existing):
            bucket[key] = path

    reviewed_path = os.path.join(directory, "{}{}.nii.gz".format(case_id, REVIEWED_SUFFIX))
    return Case(
        case_id=case_id,
        directory=directory,
        images=images,
        masks=masks,
        reviewed_path=reviewed_path,
    )


def _subdirectories(root: str) -> List[str]:
    try:
        entries = list(os.scandir(root))
    except OSError:
        return []
    out = []
    for entry in entries:
        name = entry.name
        if name.startswith("."):
            continue
        try:
            if not entry.is_dir():  # follows symlinks; a stray .txt is skipped here
                continue
        except OSError:  # pragma: no cover - broken symlink / permission
            continue
        out.append(os.path.join(root, name))
    return out


def discover_cases(root: str) -> List[Case]:
    """Sub-dirs of *root* holding >=1 nifti become cases, sorted by case_id.

    If *root* itself holds niftis and no such sub-dir exists, *root* is a single
    case.  A missing, empty or unreadable *root* returns ``[]`` without raising.

    Discovery is deliberately one level deep: pointing at a directory one level
    above the batch dirs yields ``[]``, which the UI should report as
    "0 cases found — did you mean a batch_NN folder?".
    """
    if not root:
        return []
    root = os.path.abspath(os.path.expanduser(str(root)))
    if not os.path.isdir(root):
        return []

    cases: List[Case] = []
    for sub in _subdirectories(root):
        if not _list_niftis(sub):
            continue
        cases.append(parse_case_files(sub))

    if not cases and _list_niftis(root):
        cases = [parse_case_files(root)]

    # case_id is NOT globally unique across batches, so break ties on the
    # absolute directory to keep the order stable.
    cases.sort(key=lambda c: (natural_key(c.case_id), c.case_id, c.directory))
    return cases


def iter_case_ids(cases: Iterable[Case]) -> List[str]:
    """Convenience for the combo box."""
    return [c.case_id for c in cases]
