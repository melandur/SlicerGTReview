"""Case discovery for the GTReview Slicer extension.

Standard library only — no ``slicer`` / ``vtk`` / ``qt`` / numpy imports, so this
module can be unit tested with plain ``PythonSlicer -m unittest``.

Naming rules (SPEC.md, implemented verbatim)
--------------------------------------------
For a file ``<stem>.nii.gz`` inside a case dir::

    key = stem[len(case_id) + 1:]   if stem starts with case_id + "_"
    key = stem                      otherwise

Classify by key (case-insensitive), first match wins:

1. key holds ``reviewed_seg`` as whole words (at its start or after ``_``,
   at its end or before ``_``), not right after ``not_``/``non_`` -> REVIEWED
2. key equals or ends with ``seg``/``mask``/``label``/``labels``/``gt`` -> MASK
3. otherwise                                          -> IMAGE

Rule 1 must be checked before rule 2 because ``reviewed_seg`` ends with ``seg``;
without it the tool would re-list its own output as an input mask on the next
discovery pass.  Likewise ``pred_seg`` ends with ``seg`` and is therefore a mask,
which a naive ``key == "seg"`` test would get wrong.

Unless the caller names it, ``case_id`` is the directory name as its parent
lists it, respelled to the file prefixes when the two differ only in letter
case.  A directory with no name of its own (a drive root) takes it from the
prefix its files share.
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
# The review key as whole words: it starts at the start of the key or after
# "_", ends at the end of the key or before "_", and is not negated by a
# "not_" or "non_" in front.  Matched on the lower-cased key, like the rest of
# classify_key.
_REVIEWED_KEY_PATTERN = re.compile(
    r"(?:^|_)(?<!not_)(?<!non_)" + re.escape(REVIEWED_KEY) + r"(?:_|$)"
)
#: a key equal to, or ending with, one of these is a mask (checked after REVIEWED)
MASK_KEYS: Tuple[str, ...] = ("seg", "mask", "label", "labels", "gt")
#: accepted volume extensions, longest first so ``.nii.gz`` wins over ``.gz``
NIFTI_EXTENSIONS: Tuple[str, ...] = (".nii.gz", ".nii")

# Dropbox / OS artefacts that may appear next to real data in a synced tree.
_JUNK_PATTERNS = (
    re.compile(r"conflicted copy", re.IGNORECASE),
    re.compile(r"\(\d+\)\s*$"),          # "foo (1).nii.gz" -> stem ends with "(1)"
    # Windows Explorer names a copy made in the same folder "foo - Copy", and
    # every later one "foo - Copy (2)", which the pattern above already catches.
    # Left in, a copied mask surfaces as an image called "seg - Copy".
    re.compile(r"\s-\scopy\s*$", re.IGNORECASE),
    re.compile(r"^~\$"),                 # office lock files
)

#: case id for a drive root whose files share no prefix to take one from.  A
#: fixed word rather than the drive letter, because a USB disk mounts under a
#: different letter from one day to the next and the review must still be found.
_ROOT_CASE_ID = "case"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def is_nifti(filename: str) -> bool:
    """True if *filename* looks like a usable NIfTI volume.

    Accepts ``.nii`` and ``.nii.gz`` (case-insensitively).  Rejects dotfiles,
    Dropbox conflict copies and Explorer ``" - Copy"`` duplicates, which a live
    synced tree can grow at any time.
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

    # 1. the review output — checked first, it ends with "seg".  Anywhere in
    #    the key counts, not only at its end: an older review kept by hand as
    #    "reviewed_seg_v2" or "reviewed_seg_old" is still a review, and offering
    #    it as an image or as a mask to start from would pass someone's earlier
    #    corrections off as input data.  It has to stand as whole words,
    #    though: a plain substring test also claimed "unreviewed_seg" and
    #    "prereviewed_seg", and a test for where a word starts still claimed
    #    "not_reviewed_seg" and "non_reviewed_seg".  All four are masks still
    #    waiting for a review, and taking them for reviews left those cases
    #    with nothing to start from.  "reviewed_segmentation" is some other
    #    word, not this tool's output.
    if _REVIEWED_KEY_PATTERN.search(normalized):
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
def _list_niftis(directory: str, strict: bool = False) -> List[str]:
    """Sorted list of nifti filenames directly inside *directory*.

    Never raises, unless *strict*: then a :class:`PermissionError` gets through,
    so the folder the user picked can be reported as denied instead of empty.
    """
    try:
        names = os.listdir(directory)
    except PermissionError:
        if strict:
            raise
        return []
    except OSError:
        return []
    return sorted((n for n in names if is_nifti(n)), key=lambda n: (natural_key(n), n))


def _on_disk_name(directory: str, typed_name: str) -> str:
    """*typed_name*, the last part of *directory*, spelled as its parent lists it.

    NTFS and APFS look names up case-insensitively, so a typed
    ``D:\\data\\yg_abc_1`` opens the folder ``YG_ABC_1``.  Taking the typed
    spelling as the case id would stop the files' prefix from matching it and
    save the review under a name nobody gave the case.  The typed name stands
    when the parent cannot be listed or holds no single matching entry.
    """
    try:
        siblings = os.listdir(os.path.dirname(directory))
    except OSError:
        return typed_name
    if typed_name in siblings:
        return typed_name
    folded = typed_name.lower()
    matches = [name for name in siblings if name.lower() == folded]
    return matches[0] if len(matches) == 1 else typed_name


def _prefix_spelling(case_id: str, stems: Sequence[str]) -> str:
    """*case_id* respelled like the file prefixes when only the letter case differs.

    A folder renamed on a case-insensitive disk, or a parent that could not be
    listed, leaves the directory name and the ``<case_id>_`` the files carry
    differing in case alone.  The files are what the naming rule reads, so their
    spelling wins: the keys still reduce to ``seg`` and ``pred_seg``, and the
    review lands next to them under the same spelling.  An exact match anywhere
    keeps *case_id* as it is, so a case-sensitive disk holding both spellings
    parses exactly as before.
    """
    prefix = case_id + "_"
    if any(stem.startswith(prefix) for stem in stems):
        return case_id
    folded = prefix.lower()
    counts: Dict[str, int] = {}
    for stem in stems:
        if stem[: len(prefix)].lower() == folded:
            spelling = stem[: len(case_id)]
            counts[spelling] = counts.get(spelling, 0) + 1
    if not counts:
        return case_id
    return min(counts, key=lambda spelling: (-counts[spelling], spelling))


def _case_id_from_stems(stems: Sequence[str]) -> str:
    """Case id for a directory with no name of its own, such as ``Z:\\``.

    A drive or share root holding one case's files directly used to get the
    path itself as its id and a review saved as a bare ``_reviewed_seg.nii.gz``.
    The files of one case share ``<case_id>_``, so the id is the prefix, cut at
    an underscore, that the most stems carry, the longest on a tie; a stray
    template next to the case cannot outvote it.  A review counts with its whole
    ``<case_id>``, so the id it was saved under is found again on the next pass.
    """
    support: Dict[str, int] = {}
    for stem in set(stems):
        candidates = set()
        if stem.lower().endswith(REVIEWED_SUFFIX):
            base = stem[: len(stem) - len(REVIEWED_SUFFIX)]
            if base:
                candidates.add(base)
        else:
            base = stem
        for match in re.finditer("_", base):
            if match.start():
                candidates.add(base[: match.start()])
        for candidate in candidates:
            support[candidate] = support.get(candidate, 0) + 1
    if not support:
        return _ROOT_CASE_ID
    return min(support, key=lambda c: (-support[c], -len(c), c))


def _default_case_id(directory: str, stems: Sequence[str], name_is_listed: bool) -> str:
    """The case id of *directory* when the caller does not name one."""
    name = os.path.basename(directory.rstrip(os.sep))
    if not name:
        return _case_id_from_stems(stems)
    if not name_is_listed:
        name = _on_disk_name(directory, name)
    return _prefix_spelling(name, stems)


def parse_case_files(case_dir: str, case_id: Optional[str] = None) -> Case:
    """Build a :class:`Case` from the nifti files directly inside *case_dir*.

    Non-nifti clutter, sub-directories, dotfiles and Dropbox conflict copies are
    ignored.  An unreadable or missing directory yields an empty ``Case`` rather
    than raising.
    """
    directory = os.path.abspath(os.path.expanduser(str(case_dir)))
    return _build_case(directory, _list_niftis(directory), case_id)


def _build_case(
    directory: str,
    names: Sequence[str],
    case_id: Optional[str] = None,
    name_is_listed: bool = False,
) -> Case:
    """The :class:`Case` for nifti *names* already listed from *directory*.

    Discovery lists each directory once and hands the names over, so a root it
    was refused is reported by that one listing rather than parsed a second
    time into an empty case.  *name_is_listed* says the directory's own name
    came from its parent's listing and needs no respelling.
    """
    names = [name for name in names if os.path.isfile(os.path.join(directory, name))]
    if not case_id:
        case_id = _default_case_id(
            directory, [nifti_stem(name) for name in names], name_is_listed
        )

    images: Dict[str, str] = {}
    masks: Dict[str, str] = {}

    for name in names:
        path = os.path.join(directory, name)
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
    # Only a refusal is passed on: macOS privacy protection and Windows ACLs
    # both answer with PermissionError, and the user can grant access, while a
    # root that vanished in the meantime is simply empty.
    try:
        with os.scandir(root) as iterator:
            entries = list(iterator)
    except PermissionError:
        raise
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
    case.  A missing or empty *root* returns ``[]`` without raising.  A *root*
    the process may not list or look at raises :class:`PermissionError`: macOS
    privacy protection over Desktop, Documents and Downloads does exactly that,
    and reporting it as "0 cases found" would hide the one thing the user can
    fix.  Sub-dirs that cannot be read are skipped.

    Discovery is deliberately one level deep: pointing at a directory one level
    above the batch dirs yields ``[]``, which the UI should report as
    "0 cases found — did you mean a batch_NN folder?".
    """
    if not root:
        return []
    root = os.path.abspath(os.path.expanduser(str(root)))
    if not os.path.isdir(root):
        # isdir() also answers False for a path it was refused a look at.
        try:
            os.stat(root)
        except PermissionError:
            raise
        except (OSError, ValueError):
            pass
        return []

    cases: List[Case] = []
    for sub in _subdirectories(root):
        names = _list_niftis(sub)
        if not names:
            continue
        cases.append(_build_case(sub, names, name_is_listed=True))

    if not cases:
        names = _list_niftis(root, strict=True)
        if names:
            cases = [_build_case(root, names)]

    # case_id is NOT globally unique across batches, so break ties on the
    # absolute directory to keep the order stable.
    cases.sort(key=lambda c: (natural_key(c.case_id), c.case_id, c.directory))
    return cases


def iter_case_ids(cases: Iterable[Case]) -> List[str]:
    """Convenience for the combo box."""
    return [c.case_id for c in cases]
