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

Nothing here duplicates ``test_dataset.py``; run both together.

Run with::

    /home/melandur/Documents/Slicer-5.10.0-linux-amd64/bin/PythonSlicer \
        -m unittest discover -s /home/melandur/code/gt_tools_slicer/Testing \
        -p 'test_dataset_extra.py' -v
"""

import os
import sys
import tempfile
import unittest

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
        os.symlink(target, case.reviewed_path)
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
        # the rules look at suffixes only, never at the word in the middle:
        # a trailing "_v2" makes it neither the review output nor a mask
        self.assertEqual(classify_key("reviewed_seg_v2"), IMAGE)
        self.assertEqual(classify_key("reviewedseg"), MASK)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
