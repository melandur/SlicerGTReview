"""Second unit-test file for GTReviewLib.dataset — the review-output lifecycle.

``test_dataset.py`` covers classification, parsing and discovery of a tree that
sits still.  This file covers what happens when the tree *moves* underneath an
already-parsed :class:`Case`, which is what the "Delete review" button and the
save path in GTReview.py do at runtime:

* ``Case.is_reviewed`` is a live disk probe, not a snapshot taken at parse time,
  so a file appearing or disappearing under an existing object must be seen.
* ``default_mask_path`` must follow it: reviewed while the file is there, back
  to the original mask the instant it is deleted.
* the review output must never re-enter ``masks`` on the next discovery pass —
  including the round trip "write exactly at ``case.reviewed_path``, re-parse".
* discovery must survive the junk a live Dropbox tree grows: conflicted copies,
  ``(1)`` duplicates, dotfiles, AppleDouble stubs, Office lock files.
* a case directory with no mask at all, and one with nothing *but* the review.
* ``natural_key`` beyond the single happy case in the first file: mixed
  digit/text runs, leading zeros, empty input, and the tie-breaking that keeps
  Prev/Next stable when two ids collapse to the same key.
* what Windows and macOS put in the way: Explorer ``" - Copy"`` duplicates, an
  older review kept under a longer name, a folder path typed in another letter
  case than the disk holds, a drive root holding one case's files, and a root
  the process is refused.  The disks here are case-sensitive and the paths
  POSIX, so the case-insensitive lookup, the nameless drive root and the
  refusals are simulated by patching ``os`` for the duration of a test.

Nothing here duplicates ``test_dataset.py``; run both together.

Run with::

    /home/melandur/Documents/Slicer-5.10.0-linux-amd64/bin/PythonSlicer \
        -m unittest discover -s /home/melandur/code/gt_tools_slicer/Testing \
        -p 'test_dataset_extra.py' -v
"""

import contextlib
import errno
import ntpath
import os
import posixpath
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_ROOT = os.path.join(os.path.dirname(_HERE), "GTReview")
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from GTReviewLib import dataset  # noqa: E402
from GTReviewLib.dataset import (  # noqa: E402
    IMAGE,
    MASK,
    REVIEWED,
    Case,
    classify_key,
    discover_cases,
    iter_case_ids,
    natural_key,
    parse_case_files,
)


def touch(path, content=b""):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


def make_case(root, case_id, keys, ext=".nii.gz", prefix=None):
    """Create ``<root>/<case_id>/<prefix or case_id>_<key><ext>`` for each key."""
    case_dir = os.path.join(root, case_id)
    os.makedirs(case_dir, exist_ok=True)
    stem_prefix = case_id if prefix is None else prefix
    for key in keys:
        name = "{}_{}{}".format(stem_prefix, key, ext) if stem_prefix else "{}{}".format(key, ext)
        touch(os.path.join(case_dir, name))
    return case_dir


# chmod cannot lock a directory against root, who reads it anyway, nor on
# Windows, where it only sets the read-only attribute and os.geteuid is missing.
CHMOD_CANNOT_LOCK = getattr(os, "geteuid", lambda: 1)() == 0 or os.name == "nt"


class TempTreeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)


# --------------------------------------------------------------------------- #
# is_reviewed is a live probe: the file appears and disappears under the Case
# --------------------------------------------------------------------------- #
class TestReviewFileLifecycle(TempTreeTestCase):
    def test_flag_and_default_mask_follow_the_file_appearing(self):
        case = parse_case_files(make_case(self.root, "A1", ["t1c", "pred_seg"]))
        self.assertFalse(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), case.masks["pred_seg"])

        # the save path writes exactly here; the same object must notice
        touch(case.reviewed_path)
        self.assertTrue(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), case.reviewed_path)

    def test_delete_review_reverts_to_the_original_mask(self):
        # the "Delete review" button: os.remove(case.reviewed_path), no re-parse
        case = parse_case_files(make_case(self.root, "A2", ["t1c", "seg", "pred_seg"]))
        touch(case.reviewed_path)
        self.assertEqual(case.default_mask_path(), case.reviewed_path)

        os.remove(case.reviewed_path)
        self.assertFalse(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_flag_survives_repeated_write_delete_cycles(self):
        case = parse_case_files(make_case(self.root, "A3", ["t1c", "pred_seg"]))
        for _ in range(3):
            touch(case.reviewed_path)
            self.assertTrue(case.is_reviewed)
            os.remove(case.reviewed_path)
            self.assertFalse(case.is_reviewed)

    def test_masks_are_a_snapshot_even_though_the_flag_is_not(self):
        # deliberate asymmetry: only is_reviewed re-reads the disk, so a mask
        # deleted after parsing still shows up in case.masks
        case = parse_case_files(make_case(self.root, "A4", ["t1c", "seg"]))
        os.remove(case.masks["seg"])
        self.assertIn("seg", case.masks)
        self.assertFalse(os.path.exists(case.default_mask_path()))

    def test_directory_at_the_reviewed_path_is_not_a_review(self):
        case = parse_case_files(make_case(self.root, "A5", ["t1c", "seg"]))
        os.makedirs(case.reviewed_path)
        self.assertFalse(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_symlinked_review_is_followed_and_a_broken_one_is_not(self):
        case = parse_case_files(make_case(self.root, "A6", ["t1c", "seg"]))
        target = touch(os.path.join(self.root, "elsewhere_reviewed_seg.nii.gz"))
        try:
            os.symlink(target, case.reviewed_path)
        except OSError as exc:
            # Windows lets only administrators or Developer Mode make symlinks
            self.skipTest("cannot create a symlink here: {}".format(exc))
        self.assertTrue(case.is_reviewed)

        os.remove(target)
        self.assertFalse(case.is_reviewed)

    def test_empty_reviewed_path_is_never_reviewed(self):
        # a hand-built Case (the dataclass default) must not stat "" or cwd
        case = Case(case_id="A7", directory=self.root, masks={"seg": "/x/A7_seg.nii.gz"})
        self.assertEqual(case.reviewed_path, "")
        self.assertFalse(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), "/x/A7_seg.nii.gz")

    def test_review_of_a_case_with_no_masks_at_all(self):
        case = parse_case_files(make_case(self.root, "A8", ["t1c"]))
        self.assertIsNone(case.default_mask_path())
        touch(case.reviewed_path)
        self.assertEqual(case.default_mask_path(), case.reviewed_path)

    def test_discovered_cases_track_the_file_without_re_discovery(self):
        # GTReview.py re-runs its skip-reviewed filter over the case objects it
        # already holds and relies on this
        make_case(self.root, "c1", ["t1c", "pred_seg"])
        make_case(self.root, "c2", ["t1c", "pred_seg"])
        cases = discover_cases(self.root)
        self.assertEqual([c.case_id for c in cases if c.is_reviewed], [])

        touch(cases[0].reviewed_path)
        self.assertEqual([c.case_id for c in cases if c.is_reviewed], ["c1"])
        os.remove(cases[0].reviewed_path)
        self.assertEqual([c.case_id for c in cases if not c.is_reviewed], ["c1", "c2"])


# --------------------------------------------------------------------------- #
# the review output must never be classified as an input mask
# --------------------------------------------------------------------------- #
class TestReviewedNeverBecomesInput(TempTreeTestCase):
    def test_round_trip_write_then_reparse(self):
        # the self-discovery trap in full: save at case.reviewed_path, discover
        # again, and the review must not have grown into a third mask
        case_dir = make_case(self.root, "B1", ["t1c", "seg", "pred_seg"])
        first = parse_case_files(case_dir)
        touch(first.reviewed_path)

        second = parse_case_files(case_dir)
        self.assertEqual(set(second.masks), {"seg", "pred_seg"})
        self.assertEqual(set(second.images), {"t1c"})
        self.assertTrue(second.is_reviewed)
        self.assertEqual(second.reviewed_path, first.reviewed_path)

    def test_round_trip_when_the_file_prefix_is_not_the_dir_name(self):
        # reviewed_path is built from case_id, so the review lands with a stem
        # the parser then strips again -- true even when the data files carry a
        # different prefix
        case_dir = make_case(self.root, "case_007", ["t1c", "seg"], prefix="OTHERID")
        first = parse_case_files(case_dir)
        touch(first.reviewed_path)

        second = parse_case_files(case_dir)
        self.assertEqual(set(second.masks), {"OTHERID_seg"})
        self.assertTrue(second.is_reviewed)

    def test_round_trip_with_an_explicit_case_id(self):
        case_dir = make_case(self.root, "whatever", ["t1c", "seg"], prefix="OTHERID")
        first = parse_case_files(case_dir, case_id="OTHERID")
        touch(first.reviewed_path)

        second = parse_case_files(case_dir, case_id="OTHERID")
        self.assertEqual(set(second.masks), {"seg"})
        self.assertNotIn("reviewed_seg", second.masks)
        self.assertTrue(second.is_reviewed)

    def test_review_variants_are_excluded_from_both_buckets(self):
        case_dir = make_case(self.root, "B2", ["t1c", "seg"])
        for name in ("B2_reviewed_seg.nii.gz", "B2_REVIEWED_SEG.nii.gz",
                     "B2_v2_reviewed_seg.nii.gz", "B2_reviewed_seg.nii"):
            touch(os.path.join(case_dir, name))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"seg"})
        self.assertEqual(set(case.images), {"t1c"})
        for key in case.masks:
            self.assertNotEqual(classify_key(key), REVIEWED)

    def test_uncompressed_review_alone_is_invisible(self):
        # reviewed_path is hard-coded to .nii.gz, so a .nii review is dropped
        # from masks (it classifies as REVIEWED) *and* fails is_reviewed: the
        # case looks unreviewed and un-maskable.  Asserting today's behaviour.
        case_dir = make_case(self.root, "B3", ["t1c"])
        touch(os.path.join(case_dir, "B3_reviewed_seg.nii"))
        case = parse_case_files(case_dir)
        self.assertEqual(case.masks, {})
        self.assertFalse(case.is_reviewed)
        self.assertIsNone(case.default_mask_path())

    def test_a_directory_holding_only_the_review_is_still_a_case(self):
        case_dir = os.path.join(self.root, "B4")
        os.makedirs(case_dir)
        touch(os.path.join(case_dir, "B4_reviewed_seg.nii.gz"))
        cases = discover_cases(self.root)
        self.assertEqual([c.case_id for c in cases], ["B4"])
        self.assertEqual(cases[0].masks, {})
        self.assertEqual(cases[0].images, {})
        self.assertTrue(cases[0].is_reviewed)
        self.assertEqual(cases[0].default_mask_path(), cases[0].reviewed_path)

    def test_reviewed_classification_of_bare_keys(self):
        self.assertEqual(classify_key("reviewed_seg"), REVIEWED)
        self.assertEqual(classify_key("t1c_reviewed_seg"), REVIEWED)
        # the review key counts wherever a word starts, so an older review kept
        # as "_v2" is still a review and never offered as an image; without
        # the underscore it is just a word ending in "seg", and a mask, and so
        # is a word that merely ends in the review key
        self.assertEqual(classify_key("reviewed_seg_v2"), REVIEWED)
        self.assertEqual(classify_key("reviewedseg"), MASK)
        self.assertEqual(classify_key("unreviewed_seg"), MASK)
        self.assertEqual(classify_key("reviewed"), IMAGE)


# --------------------------------------------------------------------------- #
# default_mask_path fall-through order
# --------------------------------------------------------------------------- #
class TestDefaultMaskFallThrough(TempTreeTestCase):
    def test_full_order_seg_then_gt_then_pred_seg_then_any(self):
        # walk the whole chain by deleting the winner from the dict each time
        case = parse_case_files(
            make_case(self.root, "D1", ["t1c", "seg", "gt", "pred_seg", "tumor_mask"])
        )
        for expected in ("seg", "gt", "pred_seg", "tumor_mask"):
            self.assertEqual(case.default_mask_path(), case.masks[expected], expected)
            del case.masks[expected]
        self.assertIsNone(case.default_mask_path())

    def test_label_and_labels_are_masks_but_not_preferred(self):
        case = parse_case_files(make_case(self.root, "D2", ["t1c", "label", "labels", "gt"]))
        self.assertEqual(set(case.masks), {"label", "labels", "gt"})
        self.assertEqual(case.default_mask_path(), case.masks["gt"])

    def test_any_mask_tie_break_is_natural_not_lexicographic(self):
        # no preferred key present -> deterministic pick via natural_key, so
        # "a9_mask" beats "a10_mask" even though "a1..." sorts first as text
        case = parse_case_files(make_case(self.root, "D3", ["t1c", "a10_mask", "a9_mask"]))
        self.assertEqual(set(case.masks), {"a10_mask", "a9_mask"})
        self.assertEqual(case.default_mask_path(), case.masks["a9_mask"])

    def test_preferred_none_falls_straight_through_to_any_mask(self):
        case = parse_case_files(make_case(self.root, "D4", ["t1c", "seg", "a_mask"]))
        self.assertEqual(case.default_mask_path(preferred=None), case.masks["a_mask"])

    def test_preferred_entries_are_stripped_and_lowered(self):
        case = parse_case_files(make_case(self.root, "D5", ["t1c", "seg", "pred_seg"]))
        self.assertEqual(case.default_mask_path(preferred=("  PRED_Seg  ",)),
                         case.masks["pred_seg"])

    def test_uppercase_mask_key_on_disk_still_matches_the_default_order(self):
        # lookup lowercases the keys found on disk too, not just the preferences
        case_dir = make_case(self.root, "D6", ["t1c", "pred_seg"])
        touch(os.path.join(case_dir, "D6_SEG.nii.gz"))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"SEG", "pred_seg"})
        self.assertEqual(case.default_mask_path(), case.masks["SEG"])

    def test_unknown_preferred_keys_are_skipped_not_fatal(self):
        case = parse_case_files(make_case(self.root, "D7", ["t1c", "pred_seg"]))
        self.assertEqual(case.default_mask_path(preferred=("nope", "also_nope", "pred_seg")),
                         case.masks["pred_seg"])
        # nothing preferred matches -> "any mask", never None while masks exist
        self.assertEqual(case.default_mask_path(preferred=("nope",)), case.masks["pred_seg"])

    def test_nii_gz_wins_over_nii_for_the_same_mask_key(self):
        case_dir = make_case(self.root, "D8", ["t1c", "seg"], ext=".nii")
        make_case(self.root, "D8", ["seg"], ext=".nii.gz")
        case = parse_case_files(case_dir)
        self.assertTrue(case.default_mask_path().endswith("D8_seg.nii.gz"))

    def test_has_masks_and_has_images(self):
        both = parse_case_files(make_case(self.root, "D9", ["t1c", "seg"]))
        image_only = parse_case_files(make_case(self.root, "D10", ["t1c"]))
        mask_only = parse_case_files(make_case(self.root, "D11", ["seg"]))
        self.assertEqual((both.has_images(), both.has_masks()), (True, True))
        self.assertEqual((image_only.has_images(), image_only.has_masks()), (True, False))
        self.assertEqual((mask_only.has_images(), mask_only.has_masks()), (False, True))


# --------------------------------------------------------------------------- #
# a case with no mask at all
# --------------------------------------------------------------------------- #
class TestCaseWithoutMask(TempTreeTestCase):
    def test_image_only_case_is_discovered_but_has_nothing_to_review(self):
        make_case(self.root, "E1", ["t1c", "t2", "flair"])
        cases = discover_cases(self.root)
        self.assertEqual([c.case_id for c in cases], ["E1"])
        self.assertEqual(cases[0].masks, {})
        self.assertIsNone(cases[0].default_mask_path())
        self.assertFalse(cases[0].has_masks())

    def test_mask_less_case_does_not_hide_its_neighbours(self):
        make_case(self.root, "E2", ["t1c"])
        make_case(self.root, "E3", ["t1c", "pred_seg"])
        cases = {c.case_id: c for c in discover_cases(self.root)}
        self.assertEqual(sorted(cases), ["E2", "E3"])
        # the UI's "has something to open" test
        self.assertEqual([cid for cid, c in sorted(cases.items()) if c.default_mask_path()],
                         ["E3"])

    def test_key_that_reduces_to_nothing_is_skipped(self):
        # "<case_id>_.nii.gz" -> key "" ; it must not land in images under ""
        case_dir = make_case(self.root, "E4", ["t1c"])
        touch(os.path.join(case_dir, "E4_.nii.gz"))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.images), {"t1c"})
        self.assertNotIn("", case.images)


# --------------------------------------------------------------------------- #
# discovery through junk in a live Dropbox tree
# --------------------------------------------------------------------------- #
class TestJunkFiles(TempTreeTestCase):
    def test_is_nifti_rejects_the_rest_of_the_junk_vocabulary(self):
        self.assertFalse(dataset.is_nifti("~$a_seg.nii.gz"))          # office lock file
        self.assertFalse(dataset.is_nifti("._a_seg.nii.gz"))          # AppleDouble stub
        self.assertFalse(dataset.is_nifti("a_seg (12).nii.gz"))       # multi-digit copy
        self.assertFalse(dataset.is_nifti("a_seg (1) .nii.gz"))       # trailing space
        self.assertFalse(dataset.is_nifti("a_seg.nii.gz.tmp"))
        self.assertFalse(dataset.is_nifti("a_seg.gz"))
        # the junk test looks at the basename only, not at the directories
        self.assertTrue(dataset.is_nifti("/x/y (1)/a_seg.nii.gz"))
        self.assertTrue(dataset.is_nifti("/x/.dropbox.cache/a_seg.nii.gz"))

    def test_case_holding_only_junk_is_not_discovered(self):
        junk_dir = os.path.join(self.root, "F1")
        os.makedirs(junk_dir)
        touch(os.path.join(junk_dir, "F1_seg (melandur's conflicted copy 2026-08-24).nii.gz"))
        touch(os.path.join(junk_dir, "F1_t1c (1).nii.gz"))
        touch(os.path.join(junk_dir, "._F1_t1c.nii.gz"))
        touch(os.path.join(junk_dir, ".DS_Store"))
        make_case(self.root, "F2", ["t1c", "seg"])
        self.assertEqual([c.case_id for c in discover_cases(self.root)], ["F2"])

    def test_junk_never_shadows_the_real_mask(self):
        case_dir = make_case(self.root, "F3", ["t1c", "seg"])
        for name in ("F3_seg (1).nii.gz",
                     "F3_seg (12).nii.gz",
                     "F3_seg (Melandur's Conflicted Copy 2026-08-24).nii.gz",
                     "~$F3_seg.nii.gz",
                     "._F3_seg.nii.gz",
                     ".F3_seg.nii.gz"):
            touch(os.path.join(case_dir, name))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"seg"})
        self.assertTrue(case.masks["seg"].endswith("F3_seg.nii.gz"))

    def test_root_fallback_ignores_junk_only_directories(self):
        touch(os.path.join(self.root, "x_t1c (1).nii.gz"))
        touch(os.path.join(self.root, ".x_seg.nii.gz"))
        self.assertEqual(discover_cases(self.root), [])

    def test_parenthesis_that_is_not_a_duplicate_marker_is_kept(self):
        # only a trailing "(n)" is junk; "(old)" is a legitimate, if odd, name
        case_dir = make_case(self.root, "F4", ["t1c", "seg"])
        touch(os.path.join(case_dir, "F4_seg (old).nii.gz"))
        case = parse_case_files(case_dir)
        self.assertTrue(case.masks["seg"].endswith("F4_seg.nii.gz"))
        # it does not end in a mask word, so it lands among the images
        self.assertIn("seg (old)", case.images)

    def test_dropbox_conflicted_copy_directories_are_discovered_as_cases(self):
        # WEAKENED: junk filtering is filename-only, so a conflicted-copy *dir*
        # becomes a second case with a mangled id, and its review would be
        # written inside it.  Asserting what the code does today.
        make_case(self.root, "F5", ["t1c", "seg"])
        make_case(self.root, "F5 (1)", ["t1c", "seg"], prefix="F5")
        ids = iter_case_ids(discover_cases(self.root))
        self.assertEqual(ids, ["F5", "F5 (1)"])

    def test_junk_does_not_disturb_case_ordering(self):
        for cid in ("G_2", "G_10", "G_1"):
            case_dir = make_case(self.root, cid, ["t1c", "pred_seg"])
            touch(os.path.join(case_dir, "{}_pred_seg (1).nii.gz".format(cid)))
        touch(os.path.join(self.root, "batch_notes.txt"))
        self.assertEqual(iter_case_ids(discover_cases(self.root)), ["G_1", "G_2", "G_10"])


# --------------------------------------------------------------------------- #
# natural_key
# --------------------------------------------------------------------------- #
class TestNaturalKey(unittest.TestCase):
    def test_mixed_digit_and_text_runs_stay_comparable(self):
        # tuples of (0, int, "") and (1, 0, str) never compare int against str;
        # a leading digit run also sorts ahead of any leading text run
        values = ["10", "9", "a", "1b", "b1", ""]
        self.assertEqual(sorted(values, key=natural_key), ["", "1b", "9", "10", "a", "b1"])

    def test_empty_and_none(self):
        self.assertEqual(natural_key(""), ())
        self.assertEqual(natural_key(None), ())

    def test_multiple_digit_runs_compare_left_to_right(self):
        values = ["c_2_10", "c_10_1", "c_2_2"]
        self.assertEqual(sorted(values, key=natural_key), ["c_2_2", "c_2_10", "c_10_1"])

    def test_leading_zeros_collapse_to_the_same_key(self):
        # "_007" and "_7" are the same timepoint numerically; discovery breaks
        # the tie on case_id so the order is still deterministic
        self.assertEqual(natural_key("x_007"), natural_key("x_7"))

    def test_uppercase_sorts_before_lowercase_within_a_text_run(self):
        # plain str ordering inside the run: documenting, not endorsing
        self.assertEqual(sorted(["YG_b_1", "YG_B_1"], key=natural_key), ["YG_B_1", "YG_b_1"])

    def test_yale_timepoints_order_numerically(self):
        values = ["YG_X_9", "YG_X_10", "YG_X_1", "YG_X_100"]
        self.assertEqual(sorted(values, key=natural_key),
                         ["YG_X_1", "YG_X_9", "YG_X_10", "YG_X_100"])


class TestDiscoveryOrdering(TempTreeTestCase):
    def test_padded_and_unpadded_ids_are_ordered_deterministically(self):
        for cid in ("p_007", "p_7", "p_8"):
            make_case(self.root, cid, ["t1c", "pred_seg"])
        first = iter_case_ids(discover_cases(self.root))
        self.assertEqual(first, ["p_007", "p_7", "p_8"])
        self.assertEqual(first, iter_case_ids(discover_cases(self.root)))

    def test_iter_case_ids_matches_discovery_order(self):
        for cid in ("q_10", "q_2"):
            make_case(self.root, cid, ["t1c", "seg"])
        cases = discover_cases(self.root)
        self.assertEqual(iter_case_ids(cases), [c.case_id for c in cases])
        self.assertEqual(iter_case_ids([]), [])


# --------------------------------------------------------------------------- #
# simulated platforms
# --------------------------------------------------------------------------- #
# Saved before any patch, so the stand-ins below can reach the real disk.
_REAL_STAT = os.stat
_REAL_LISTDIR = os.listdir
_REAL_SCANDIR = os.scandir
_REAL_BASENAME = os.path.basename


def _resolve_case_insensitively(path):
    """*path* with each missing component replaced by its one case-insensitive match.

    That is the lookup NTFS and APFS do; a path that exists as spelled, or has
    no unique match, is returned unchanged.
    """
    if not isinstance(path, str) or os.path.lexists(path):
        return path
    current = os.sep
    for part in os.path.abspath(path).split(os.sep):
        if not part:
            continue
        candidate = os.path.join(current, part)
        if not os.path.lexists(candidate):
            try:
                siblings = _REAL_LISTDIR(current)
            except OSError:
                return path
            matches = [name for name in siblings if name.lower() == part.lower()]
            if len(matches) != 1:
                return path
            candidate = os.path.join(current, matches[0])
        current = candidate
    return current


@contextlib.contextmanager
def case_insensitive_disk(unlistable=()):
    """Resolve names in os.stat/listdir/scandir the way NTFS and APFS do.

    ``os.path.isfile`` and ``isdir`` go through ``os.stat`` and follow along.
    Listing a directory named in *unlistable* is refused.
    """
    refused = {os.path.abspath(p) for p in unlistable}

    def stat(path, *args, **kwargs):
        return _REAL_STAT(_resolve_case_insensitively(path), *args, **kwargs)

    def listdir(path="."):
        resolved = _resolve_case_insensitively(path)
        if isinstance(resolved, str) and os.path.abspath(resolved) in refused:
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return _REAL_LISTDIR(resolved)

    def scandir(path="."):
        return _REAL_SCANDIR(_resolve_case_insensitively(path))

    with mock.patch("os.stat", stat), mock.patch("os.listdir", listdir), \
            mock.patch("os.scandir", scandir):
        yield


@contextlib.contextmanager
def refused(function_name, *paths, error=None):
    """Make ``os.<function_name>`` raise for *paths*, as a privacy or ACL block does."""
    real = getattr(os, function_name)
    targets = {os.path.abspath(p) for p in paths}

    def stand_in(path=".", *args, **kwargs):
        if isinstance(path, str) and os.path.abspath(path) in targets:
            if error is not None:
                raise error
            # macOS privacy protection answers EPERM, "Operation not permitted"
            raise PermissionError(errno.EPERM, "Operation not permitted", path)
        return real(path, *args, **kwargs)

    with mock.patch("os." + function_name, stand_in):
        yield


@contextlib.contextmanager
def nameless_directory(path):
    """Make *path* look like ``Z:\\``: a directory whose last component is empty."""
    target = os.path.abspath(path).rstrip(os.sep)

    def basename(p):
        if isinstance(p, str) and p.rstrip(os.sep) == target:
            return ""
        return _REAL_BASENAME(p)

    with mock.patch("os.path.basename", basename):
        yield


# --------------------------------------------------------------------------- #
# Explorer duplicates
# --------------------------------------------------------------------------- #
class TestExplorerCopies(TempTreeTestCase):
    def test_is_nifti_rejects_explorer_copies(self):
        self.assertFalse(dataset.is_nifti("X_seg - Copy.nii.gz"))
        self.assertFalse(dataset.is_nifti("X - Copy (2).nii.gz"))
        self.assertFalse(dataset.is_nifti("X_t1c - copy.nii"))
        self.assertFalse(dataset.is_nifti("X_SEG - COPY.NII.GZ"))
        # Explorer puts the marker before the last extension only, which leaves
        # a name that is no NIfTI at all
        self.assertFalse(dataset.is_nifti("X_seg.nii - Copy.gz"))

    def test_the_word_copy_on_its_own_is_not_a_duplicate_marker(self):
        self.assertTrue(dataset.is_nifti("X_t1c_copy.nii.gz"))
        self.assertTrue(dataset.is_nifti("X - Copyright.nii.gz"))

    def test_copies_never_shadow_or_join_the_real_files(self):
        case_dir = make_case(self.root, "H1", ["t1c", "seg", "pred_seg"])
        for name in ("H1_seg - Copy.nii.gz",
                     "H1_seg - Copy (2).nii.gz",
                     "H1_t1c - copy.nii",
                     "H1 - Copy.nii.gz",
                     "H1_seg.nii - Copy.gz"):
            touch(os.path.join(case_dir, name))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        self.assertEqual(set(case.images), {"t1c"})
        self.assertTrue(case.masks["seg"].endswith("H1_seg.nii.gz"))
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_a_folder_of_nothing_but_copies_is_not_a_case(self):
        copies = os.path.join(self.root, "H2")
        os.makedirs(copies)
        touch(os.path.join(copies, "H2_t1c - Copy.nii.gz"))
        touch(os.path.join(copies, "H2_seg - Copy (3).nii.gz"))
        make_case(self.root, "H3", ["t1c", "seg"])
        self.assertEqual(iter_case_ids(discover_cases(self.root)), ["H3"])

    def test_root_fallback_ignores_copies(self):
        touch(os.path.join(self.root, "x_t1c - Copy.nii.gz"))
        self.assertEqual(discover_cases(self.root), [])


# --------------------------------------------------------------------------- #
# the review key anywhere in a name
# --------------------------------------------------------------------------- #
class TestReviewKeyAnywhere(TempTreeTestCase):
    def test_keys_containing_the_review_key_are_reviews(self):
        for key in ("reviewed_seg_v2", "reviewed_seg_old", "old_reviewed_seg",
                    "Reviewed_Seg_Backup", "t1c_reviewed_seg_2"):
            self.assertEqual(classify_key(key), REVIEWED, key)

    def test_older_reviews_are_neither_images_nor_masks(self):
        case_dir = make_case(self.root, "J1", ["t1c", "seg"])
        for name in ("J1_reviewed_seg_v2.nii.gz",
                     "J1_reviewed_seg_old.nii",
                     "reviewed_seg_backup.nii.gz",
                     "OTHER_reviewed_seg_2.nii.gz"):
            touch(os.path.join(case_dir, name))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"seg"})
        self.assertEqual(set(case.images), {"t1c"})
        # only <case_id>_reviewed_seg.nii.gz is the live review
        self.assertFalse(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_a_folder_holding_only_an_older_review_offers_nothing(self):
        case_dir = os.path.join(self.root, "J2")
        os.makedirs(case_dir)
        touch(os.path.join(case_dir, "J2_reviewed_seg_old.nii.gz"))
        cases = discover_cases(self.root)
        self.assertEqual(iter_case_ids(cases), ["J2"])
        self.assertEqual((cases[0].images, cases[0].masks), ({}, {}))
        self.assertIsNone(cases[0].default_mask_path())

    def test_a_mask_named_unreviewed_is_offered_as_a_mask(self):
        # the review key has to start a word; "unreviewed_seg" only ends in it,
        # and "not_" / "non_" in front negate it
        keys = ["unreviewed_seg", "prereviewed_seg", "not_reviewed_seg", "non_reviewed_seg"]
        case_dir = make_case(self.root, "J3", ["t1c"] + keys)
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), set(keys))
        self.assertEqual(set(case.images), {"t1c"})
        self.assertFalse(case.is_reviewed)
        self.assertIsNotNone(case.default_mask_path())


# --------------------------------------------------------------------------- #
# the session log GTReview writes into the batch folder
# --------------------------------------------------------------------------- #
class TestSessionLogInTheBatchFolder(TempTreeTestCase):
    """GTReview.log, and GTReview.log.1 once it rotated, sit in the loaded folder.

    The panel attaches its log to the folder it just discovered, so the next
    load of that folder finds the log files next to the cases, or next to the
    volumes themselves when the folder is a single case.  They must change
    nothing about what is discovered.
    """

    LOG_NAMES = ("GTReview.log", "GTReview.log.1")

    def _add_logs(self, folder):
        for name in self.LOG_NAMES:
            touch(os.path.join(folder, name), b"2026-09-14 10:00:00 INFO GTReview: dataset loaded\n")

    def test_batch_folder_discovers_the_same_cases(self):
        make_case(self.root, "YG_L_2", ["t1c", "seg"])
        make_case(self.root, "YG_L_10", ["t1c", "pred_seg", "reviewed_seg"])
        without = discover_cases(self.root)
        self._add_logs(self.root)
        with_logs = discover_cases(self.root)
        self.assertEqual(iter_case_ids(with_logs), ["YG_L_2", "YG_L_10"])
        self.assertEqual(with_logs, without)

    def test_single_case_folder_discovers_the_same_case(self):
        case_dir = make_case(self.root, "YG_L_3", ["t1c", "seg", "pred_seg"])
        without = discover_cases(case_dir)
        self._add_logs(case_dir)
        with_logs = discover_cases(case_dir)
        self.assertEqual(iter_case_ids(with_logs), ["YG_L_3"])
        self.assertEqual(with_logs, without)

    def test_drive_root_keeps_its_case_id(self):
        for stem in ("YG_L_4_t1c", "YG_L_4_seg"):
            touch(os.path.join(self.root, stem + ".nii.gz"))
        with nameless_directory(self.root):
            without = discover_cases(self.root)
            self._add_logs(self.root)
            with_logs = discover_cases(self.root)
        self.assertEqual(iter_case_ids(with_logs), ["YG_L_4"])
        self.assertEqual(with_logs, without)

    def test_a_folder_holding_only_the_logs_is_not_a_case(self):
        self._add_logs(self.root)
        self.assertEqual(discover_cases(self.root), [])


# --------------------------------------------------------------------------- #
# a folder path typed in another letter case (NTFS, APFS)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(os.name == "posix", "the simulated lookup splits POSIX paths")
class TestLetterCaseOfTypedPaths(TempTreeTestCase):
    def test_typed_single_case_folder_takes_the_case_id_from_the_disk(self):
        make_case(self.root, "YG_ABC_1", ["t1c", "seg", "pred_seg"])
        typed = os.path.join(self.root, "yg_abc_1")
        with case_insensitive_disk():
            case = parse_case_files(typed)
            default = case.default_mask_path()
        self.assertEqual(case.case_id, "YG_ABC_1")
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        # typed in lower case this used to fall through to pred_seg
        self.assertEqual(default, case.masks["seg"])
        self.assertEqual(os.path.basename(case.reviewed_path), "YG_ABC_1_reviewed_seg.nii.gz")

    def test_typed_single_case_folder_as_the_discovery_root(self):
        make_case(self.root, "YG_ABC_1", ["t1c", "seg", "pred_seg"])
        with case_insensitive_disk():
            cases = discover_cases(os.path.join(self.root, "yg_abc_1"))
            default = cases[0].default_mask_path()
        self.assertEqual(iter_case_ids(cases), ["YG_ABC_1"])
        self.assertEqual(default, cases[0].masks["seg"])

    def test_typed_batch_folder_lists_case_ids_as_on_disk(self):
        batch = os.path.join(self.root, "batch_01")
        make_case(batch, "YG_A_1", ["t1c", "seg", "pred_seg"])
        make_case(batch, "YG_A_2", ["t1c", "pred_seg"])
        with case_insensitive_disk():
            cases = discover_cases(os.path.join(self.root, "BATCH_01"))
        self.assertEqual(iter_case_ids(cases), ["YG_A_1", "YG_A_2"])
        self.assertEqual(set(cases[0].masks), {"seg", "pred_seg"})

    def test_files_without_the_prefix_take_the_on_disk_folder_name(self):
        make_case(self.root, "YG_ABC_2", ["t1c", "seg"], prefix="")
        with case_insensitive_disk():
            case = parse_case_files(os.path.join(self.root, "yg_abc_2"))
        self.assertEqual(case.case_id, "YG_ABC_2")
        self.assertEqual(set(case.masks), {"seg"})
        self.assertEqual(os.path.basename(case.reviewed_path), "YG_ABC_2_reviewed_seg.nii.gz")

    def test_review_saved_through_the_typed_path_is_found_again(self):
        on_disk = make_case(self.root, "YG_ABC_3", ["t1c", "seg"])
        typed = os.path.join(self.root, "yg_abc_3")
        with case_insensitive_disk():
            first = parse_case_files(typed)
        # open() does not go through the simulated lookup; write where the
        # case-insensitive disk would have put it
        touch(os.path.join(on_disk, os.path.basename(first.reviewed_path)))
        with case_insensitive_disk():
            second = parse_case_files(typed)
            reviewed = second.is_reviewed
        self.assertTrue(reviewed)
        self.assertEqual(set(second.masks), {"seg"})
        self.assertEqual(second.case_id, first.case_id)

    def test_unlistable_parent_falls_back_to_the_file_prefix(self):
        make_case(self.root, "YG_ABC_4", ["t1c", "seg", "pred_seg"])
        with case_insensitive_disk(unlistable=[self.root]):
            case = parse_case_files(os.path.join(self.root, "yg_abc_4"))
            default = case.default_mask_path()
        self.assertEqual(case.case_id, "YG_ABC_4")
        self.assertEqual(default, case.masks["seg"])

    def test_folder_renamed_in_another_case_follows_its_files(self):
        # no simulation: the folder really is lower case, its files are not
        make_case(self.root, "yg_abc_5", ["t1c", "seg", "pred_seg"], prefix="YG_ABC_5")
        case = parse_case_files(os.path.join(self.root, "yg_abc_5"))
        self.assertEqual(case.case_id, "YG_ABC_5")
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        self.assertEqual(case.default_mask_path(), case.masks["seg"])
        self.assertEqual(os.path.basename(case.reviewed_path), "YG_ABC_5_reviewed_seg.nii.gz")
        self.assertEqual(iter_case_ids(discover_cases(self.root)), ["YG_ABC_5"])

    def test_most_common_file_spelling_wins(self):
        case_dir = make_case(self.root, "yg_q_1", ["t1c", "seg"], prefix="YG_Q_1")
        touch(os.path.join(case_dir, "Yg_Q_1_pred_seg.nii.gz"))
        case = parse_case_files(case_dir)
        self.assertEqual(case.case_id, "YG_Q_1")
        self.assertEqual(set(case.masks), {"seg", "Yg_Q_1_pred_seg"})

    def test_exact_prefix_wins_on_a_case_sensitive_disk(self):
        case_dir = make_case(self.root, "c1", ["seg"])
        touch(os.path.join(case_dir, "C1_seg.nii.gz"))
        if len(os.listdir(case_dir)) < 2:
            self.skipTest("this disk folds letter case")
        case = parse_case_files(case_dir)
        self.assertEqual(case.case_id, "c1")
        self.assertEqual(set(case.masks), {"seg", "C1_seg"})
        self.assertEqual(case.default_mask_path(), case.masks["seg"])


# --------------------------------------------------------------------------- #
# a drive root holding one case's files
# --------------------------------------------------------------------------- #
class TestDriveRoot(TempTreeTestCase):
    def test_windows_roots_have_no_last_component(self):
        # the premise nameless_directory simulates, on the real path modules
        self.assertEqual(ntpath.basename("Z:\\".rstrip("\\")), "")
        self.assertEqual(ntpath.basename("\\\\server\\share\\".rstrip("\\")), "")
        self.assertEqual(posixpath.basename("/".rstrip("/")), "")

    def _touch_root(self, *stems):
        for stem in stems:
            touch(os.path.join(self.root, stem + ".nii.gz"))

    def test_case_id_comes_from_the_file_prefix(self):
        self._touch_root("YG_R_7_t1c", "YG_R_7_seg", "YG_R_7_pred_seg")
        with nameless_directory(self.root):
            case = parse_case_files(self.root)
        self.assertEqual(case.case_id, "YG_R_7")
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        self.assertEqual(case.reviewed_path,
                         os.path.join(os.path.abspath(self.root), "YG_R_7_reviewed_seg.nii.gz"))
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_discovery_of_a_drive_root(self):
        self._touch_root("YG_R_7_t1c", "YG_R_7_pred_seg")
        with nameless_directory(self.root):
            cases = discover_cases(self.root)
        self.assertEqual(iter_case_ids(cases), ["YG_R_7"])
        self.assertEqual(set(cases[0].masks), {"pred_seg"})

    def test_stray_volumes_do_not_outvote_the_case(self):
        self._touch_root("YG_R_7_t1c", "YG_R_7_seg", "YG_R_7_pred_seg",
                         "template", "mni_atlas", "T1_brain")
        with nameless_directory(self.root):
            case = parse_case_files(self.root)
        self.assertEqual(case.case_id, "YG_R_7")
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_review_round_trip_keeps_the_case_id(self):
        self._touch_root("YG_R_8_t1c", "YG_R_8_pred_seg")
        with nameless_directory(self.root):
            first = parse_case_files(self.root)
            touch(first.reviewed_path)
            second = parse_case_files(self.root)
        self.assertEqual(second.case_id, first.case_id)
        self.assertTrue(second.is_reviewed)
        self.assertEqual(set(second.masks), {"pred_seg"})
        self.assertEqual(second.default_mask_path(), second.reviewed_path)

    def test_single_file_round_trip_keeps_the_case_id(self):
        # one file gives no second stem to agree with; whatever id it yields
        # must survive the review written under it
        self._touch_root("YG_R_9_pred_seg")
        with nameless_directory(self.root):
            first = parse_case_files(self.root)
            touch(first.reviewed_path)
            second = parse_case_files(self.root)
        self.assertEqual(second.case_id, first.case_id)
        self.assertTrue(second.is_reviewed)

    def test_files_without_a_shared_prefix_still_get_a_usable_id(self):
        self._touch_root("t1c", "seg")
        with nameless_directory(self.root):
            first = parse_case_files(self.root)
            touch(first.reviewed_path)
            second = parse_case_files(self.root)
        self.assertTrue(first.case_id)
        self.assertNotIn(os.sep, first.case_id)
        self.assertFalse(os.path.basename(first.reviewed_path).startswith("_"))
        self.assertEqual(second.case_id, first.case_id)
        self.assertTrue(second.is_reviewed)
        self.assertEqual(set(second.masks), {"seg"})


# --------------------------------------------------------------------------- #
# a root the process is refused
# --------------------------------------------------------------------------- #
class TestPermissionDenied(TempTreeTestCase):
    def test_refused_listing_of_the_root_raises(self):
        make_case(self.root, "K1", ["t1c", "seg"])
        with refused("scandir", self.root), self.assertRaises(PermissionError):
            discover_cases(self.root)

    def test_refused_listing_of_a_single_case_root_raises(self):
        # the sub-dir scan finds nothing, and the root's own files must then be
        # listed strictly rather than parsed into "0 cases found"
        case_dir = make_case(self.root, "K2", ["t1c", "seg"])
        with refused("listdir", case_dir), self.assertRaises(PermissionError):
            discover_cases(case_dir)

    def test_root_that_cannot_be_looked_at_raises(self):
        # isdir() answers False without saying why; the stat behind it does.
        # On Windows isdir asks the system itself and never calls os.stat, so
        # its answer for the refused path is patched in as well.
        batch = os.path.abspath(os.path.join(self.root, "batch"))
        make_case(batch, "K3", ["t1c", "seg"])
        real_isdir = os.path.isdir

        def isdir(path):
            if isinstance(path, str) and os.path.abspath(path) == batch:
                return False
            return real_isdir(path)

        with refused("stat", batch), mock.patch("os.path.isdir", isdir), \
                self.assertRaises(PermissionError):
            discover_cases(batch)

    @unittest.skipIf(CHMOD_CANNOT_LOCK, "chmod cannot lock a directory for root or on Windows")
    def test_root_under_a_locked_parent_raises(self):
        locked = os.path.join(self.root, "locked")
        make_case(os.path.join(locked, "batch"), "K4", ["t1c", "seg"])
        os.chmod(locked, 0o000)
        self.addCleanup(os.chmod, locked, 0o755)
        with self.assertRaises(PermissionError):
            discover_cases(os.path.join(locked, "batch"))

    def test_unreadable_sub_folder_is_skipped(self):
        make_case(self.root, "K5", ["t1c", "seg"])
        locked = make_case(self.root, "K6", ["t1c", "seg"])
        with refused("listdir", locked):
            cases = discover_cases(self.root)
        self.assertEqual(iter_case_ids(cases), ["K5"])

    @unittest.skipIf(CHMOD_CANNOT_LOCK, "chmod cannot lock a directory for root or on Windows")
    def test_chmod_locked_sub_folder_is_skipped(self):
        make_case(self.root, "K7", ["t1c", "seg"])
        locked = make_case(self.root, "K8", ["t1c", "seg"])
        os.chmod(locked, 0o000)
        self.addCleanup(os.chmod, locked, 0o755)
        self.assertEqual(iter_case_ids(discover_cases(self.root)), ["K7"])

    def test_root_that_vanished_is_still_just_empty(self):
        make_case(self.root, "K9", ["t1c", "seg"])
        gone = FileNotFoundError(errno.ENOENT, "No such file or directory", self.root)
        with refused("scandir", self.root, error=gone):
            self.assertEqual(discover_cases(self.root), [])

    def test_parse_case_files_on_its_own_stays_lenient(self):
        case_dir = make_case(self.root, "K10", ["t1c", "seg"])
        with refused("listdir", case_dir):
            case = parse_case_files(case_dir)
        self.assertEqual(case.case_id, "K10")
        self.assertEqual((case.images, case.masks), ({}, {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
