"""Unit tests for GTReviewLib.dataset — plain unittest, no pytest, no slicer.

Run with::

    /home/melandur/Documents/Slicer-5.10.0-linux-amd64/bin/PythonSlicer \
        -m unittest discover -s /home/melandur/code/gt_tool_slicer/Testing \
        -p 'test_dataset.py' -v
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
    parse_case_files,
)

REAL_BATCH = (
    "/home/melandur/Neosoma Inc. Dropbox/Neosoma Inc. R&D AI/01_Annotation/METS/"
    "04_Groundtruthed/01_Yale/batch_01"
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
# classify_key
# --------------------------------------------------------------------------- #
class TestClassifyKey(unittest.TestCase):
    def test_reviewed_wins_over_mask(self):
        # rule 1 must fire before rule 2: "reviewed_seg" also ends with "seg"
        self.assertEqual(classify_key("reviewed_seg"), REVIEWED)
        self.assertEqual(classify_key("REVIEWED_SEG"), REVIEWED)
        self.assertEqual(classify_key("  Reviewed_Seg "), REVIEWED)
        self.assertEqual(classify_key("v2_reviewed_seg"), REVIEWED)

    def test_masks(self):
        for key in ("seg", "pred_seg", "gt", "mask", "label", "labels",
                    "tumor_mask", "manual_gt", "SEG", "Pred_Seg"):
            self.assertEqual(classify_key(key), MASK, key)

    def test_pred_seg_is_a_mask_not_an_image(self):
        # naive `key == "seg"` would misclassify this
        self.assertEqual(classify_key("pred_seg"), MASK)

    def test_images(self):
        # the generic vocabulary never appears in the real corpus; cover it here
        for key in ("t1", "t1c", "t2", "flair", "adc", "dwi", "T1C", "t1ce", "swi"):
            self.assertEqual(classify_key(key), IMAGE, key)

    def test_empty_and_none(self):
        self.assertEqual(classify_key(""), IMAGE)
        self.assertEqual(classify_key(None), IMAGE)


# --------------------------------------------------------------------------- #
# parse_case_files
# --------------------------------------------------------------------------- #
class TestParseCaseFiles(TempTreeTestCase):
    def test_typical_groundtruthed_case(self):
        case_dir = make_case(self.root, "YG_78CQZ7VA3H2G_27", ["t1c", "seg", "pred_seg"])
        case = parse_case_files(case_dir)
        self.assertEqual(case.case_id, "YG_78CQZ7VA3H2G_27")
        self.assertEqual(case.directory, os.path.abspath(case_dir))
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        self.assertTrue(os.path.isabs(case.images["t1c"]))
        self.assertEqual(
            case.reviewed_path,
            os.path.join(case.directory, "YG_78CQZ7VA3H2G_27_reviewed_seg.nii.gz"),
        )
        self.assertFalse(case.is_reviewed)

    def test_cyprus_style_case_id(self):
        case_dir = make_case(self.root, "P39_2023-11-09", ["t1c", "pred_seg"])
        case = parse_case_files(case_dir)
        self.assertEqual(case.case_id, "P39_2023-11-09")
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"pred_seg"})

    def test_multi_sequence_case(self):
        case_dir = make_case(self.root, "C1", ["t1", "t1c", "t2", "flair", "adc", "dwi", "seg"])
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.images), {"t1", "t1c", "t2", "flair", "adc", "dwi"})
        self.assertEqual(set(case.masks), {"seg"})

    def test_uncompressed_nii_is_accepted(self):
        case_dir = make_case(self.root, "C2", ["t1c", "seg"], ext=".nii")
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg"})
        self.assertTrue(case.masks["seg"].endswith("C2_seg.nii"))

    def test_nii_gz_preferred_over_nii_for_same_key(self):
        case_dir = make_case(self.root, "C3", ["t1c"], ext=".nii")
        make_case(self.root, "C3", ["t1c"], ext=".nii.gz")
        case = parse_case_files(case_dir)
        self.assertTrue(case.images["t1c"].endswith(".nii.gz"))

    def test_prefix_differs_from_dir_name(self):
        # dead branch on the real corpus -> synthetic coverage for `key = stem`
        case_dir = make_case(self.root, "case_007", ["t1c", "seg"], prefix="OTHERID")
        case = parse_case_files(case_dir)
        self.assertEqual(case.case_id, "case_007")
        self.assertEqual(set(case.images), {"OTHERID_t1c"})
        self.assertEqual(set(case.masks), {"OTHERID_seg"})

    def test_explicit_case_id_overrides_dir_name(self):
        case_dir = make_case(self.root, "whatever", ["t1c", "seg"], prefix="OTHERID")
        case = parse_case_files(case_dir, case_id="OTHERID")
        self.assertEqual(case.case_id, "OTHERID")
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg"})
        self.assertTrue(case.reviewed_path.endswith("OTHERID_reviewed_seg.nii.gz"))

    def test_non_nifti_clutter_is_ignored(self):
        case_dir = make_case(self.root, "C4", ["t1c", "seg"])
        touch(os.path.join(case_dir, "notes.txt"))
        touch(os.path.join(case_dir, "batch_01_cases.txt"))
        touch(os.path.join(case_dir, "sidecar.json"))
        touch(os.path.join(case_dir, "C4_t1c.nii.gz.md5"))
        touch(os.path.join(case_dir, ".DS_Store"))
        touch(os.path.join(case_dir, ".hidden_seg.nii.gz"))
        os.makedirs(os.path.join(case_dir, "subdir_seg.nii.gz"), exist_ok=True)
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg"})

    def test_dropbox_conflict_copies_are_ignored(self):
        case_dir = make_case(self.root, "C5", ["t1c", "seg"])
        touch(os.path.join(case_dir, "C5_seg (melandur's conflicted copy 2026-08-24).nii.gz"))
        touch(os.path.join(case_dir, "C5_pred_seg (1).nii.gz"))
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"seg"})
        self.assertNotIn("pred_seg", case.masks)

    def test_reviewed_output_never_listed_as_input_mask(self):
        case_dir = make_case(self.root, "C6", ["t1c", "seg", "pred_seg", "reviewed_seg"])
        case = parse_case_files(case_dir)
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        self.assertNotIn("reviewed_seg", case.masks)
        self.assertNotIn("reviewed_seg", case.images)
        self.assertTrue(case.is_reviewed)

    def test_empty_case_dir(self):
        case_dir = os.path.join(self.root, "empty_case")
        os.makedirs(case_dir)
        case = parse_case_files(case_dir)
        self.assertEqual(case.images, {})
        self.assertEqual(case.masks, {})
        self.assertFalse(case.is_reviewed)
        self.assertIsNone(case.default_mask_path())

    def test_missing_dir_does_not_raise(self):
        case = parse_case_files(os.path.join(self.root, "does_not_exist"))
        self.assertEqual(case.case_id, "does_not_exist")
        self.assertEqual(case.images, {})
        self.assertEqual(case.masks, {})

    @unittest.skipIf(os.geteuid() == 0, "root can read any directory")
    def test_unreadable_dir_does_not_raise(self):
        case_dir = make_case(self.root, "locked", ["t1c", "seg"])
        os.chmod(case_dir, 0o000)
        self.addCleanup(os.chmod, case_dir, 0o755)
        case = parse_case_files(case_dir)
        self.assertEqual(case.images, {})
        self.assertEqual(case.masks, {})

    def test_trailing_separator_on_case_dir(self):
        case_dir = make_case(self.root, "C7", ["t1c", "seg"])
        case = parse_case_files(case_dir + os.sep)
        self.assertEqual(case.case_id, "C7")
        self.assertEqual(set(case.masks), {"seg"})


# --------------------------------------------------------------------------- #
# Case.default_mask_path / is_reviewed
# --------------------------------------------------------------------------- #
class TestDefaultMaskPath(TempTreeTestCase):
    def test_prefers_seg(self):
        case = parse_case_files(make_case(self.root, "A", ["t1c", "seg", "pred_seg"]))
        self.assertEqual(case.default_mask_path(), case.masks["seg"])

    def test_falls_through_to_pred_seg(self):
        # the hot path: only ~5% of real cases carry a `seg`
        case = parse_case_files(make_case(self.root, "B", ["t1c", "pred_seg"]))
        self.assertEqual(case.default_mask_path(), case.masks["pred_seg"])

    def test_gt_between_seg_and_pred_seg(self):
        case = parse_case_files(make_case(self.root, "C", ["t1c", "gt", "pred_seg"]))
        self.assertEqual(case.default_mask_path(), case.masks["gt"])

    def test_any_mask_when_no_preferred_key(self):
        case = parse_case_files(make_case(self.root, "D", ["t1c", "tumor_mask"]))
        self.assertEqual(case.default_mask_path(), case.masks["tumor_mask"])

    def test_none_when_no_masks(self):
        case = parse_case_files(make_case(self.root, "E", ["t1c"]))
        self.assertIsNone(case.default_mask_path())

    def test_reviewed_wins_when_present(self):
        case = parse_case_files(
            make_case(self.root, "F", ["t1c", "seg", "pred_seg", "reviewed_seg"])
        )
        self.assertTrue(case.is_reviewed)
        self.assertEqual(case.default_mask_path(), case.reviewed_path)

    def test_reviewed_path_absent_is_not_reviewed(self):
        case = parse_case_files(make_case(self.root, "G", ["t1c", "seg"]))
        self.assertFalse(case.is_reviewed)
        self.assertFalse(os.path.exists(case.reviewed_path))

    def test_custom_preferred_order(self):
        case = parse_case_files(make_case(self.root, "H", ["t1c", "seg", "pred_seg"]))
        self.assertEqual(case.default_mask_path(preferred=("pred_seg", "seg")),
                         case.masks["pred_seg"])
        self.assertEqual(case.default_mask_path(preferred=()), case.masks["pred_seg"])

    def test_preferred_lookup_is_case_insensitive(self):
        case = parse_case_files(make_case(self.root, "I", ["t1c", "pred_seg"]))
        self.assertEqual(case.default_mask_path(preferred=("SEG", "PRED_SEG")),
                         case.masks["pred_seg"])

    def test_case_dataclass_defaults(self):
        case = Case(case_id="X", directory=self.root)
        self.assertEqual(case.images, {})
        self.assertEqual(case.masks, {})
        self.assertFalse(case.is_reviewed)
        self.assertIsNone(case.default_mask_path())


# --------------------------------------------------------------------------- #
# discover_cases
# --------------------------------------------------------------------------- #
class TestDiscoverCases(TempTreeTestCase):
    def test_batch_dir(self):
        for cid in ("YG_A_1", "YG_A_2", "YG_B_1"):
            make_case(self.root, cid, ["t1c", "pred_seg"])
        touch(os.path.join(self.root, "batch_01_cases.txt"), b"free text\n")
        touch(os.path.join(self.root, "README.txt"))
        cases = discover_cases(self.root)
        self.assertEqual([c.case_id for c in cases], ["YG_A_1", "YG_A_2", "YG_B_1"])
        self.assertTrue(all(c.masks for c in cases))

    def test_sub_dirs_without_niftis_are_skipped(self):
        make_case(self.root, "good", ["t1c", "seg"])
        os.makedirs(os.path.join(self.root, "empty_dir"))
        os.makedirs(os.path.join(self.root, "docs"))
        touch(os.path.join(self.root, "docs", "readme.txt"))
        cases = discover_cases(self.root)
        self.assertEqual([c.case_id for c in cases], ["good"])

    def test_empty_root_returns_empty_list(self):
        self.assertEqual(discover_cases(self.root), [])

    def test_missing_root_returns_empty_list(self):
        self.assertEqual(discover_cases(os.path.join(self.root, "nope")), [])
        self.assertEqual(discover_cases(""), [])
        self.assertEqual(discover_cases(None), [])

    def test_root_that_is_a_file_returns_empty_list(self):
        path = touch(os.path.join(self.root, "a_file.txt"))
        self.assertEqual(discover_cases(path), [])

    @unittest.skipIf(os.geteuid() == 0, "root can read any directory")
    def test_unreadable_root_returns_empty_list(self):
        locked = os.path.join(self.root, "locked")
        make_case(locked, "c1", ["t1c", "seg"])
        os.chmod(locked, 0o000)
        self.addCleanup(os.chmod, locked, 0o755)
        self.assertEqual(discover_cases(locked), [])

    def test_discovery_is_one_level_deep(self):
        # pointing one level above the batch dir must yield 0 cases, not a crash
        batch = os.path.join(self.root, "batch_01")
        make_case(batch, "c1", ["t1c", "seg"])
        make_case(batch, "c2", ["t1c", "seg"])
        self.assertEqual(len(discover_cases(batch)), 2)
        self.assertEqual(discover_cases(self.root), [])

    def test_root_itself_is_a_single_case(self):
        case_dir = make_case(self.root, "YG_SINGLE_3", ["t1c", "seg", "pred_seg"])
        cases = discover_cases(case_dir)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "YG_SINGLE_3")
        self.assertEqual(set(cases[0].images), {"t1c"})
        self.assertEqual(set(cases[0].masks), {"seg", "pred_seg"})

    def test_root_single_case_fallback_ignores_non_niftis(self):
        # a batch dir holding only a .txt manifest must not become a case
        touch(os.path.join(self.root, "batch_01_cases.txt"))
        self.assertEqual(discover_cases(self.root), [])

    def test_sub_dir_cases_win_over_root_fallback(self):
        make_case(self.root, "c1", ["t1c", "seg"])
        touch(os.path.join(self.root, "stray_t1c.nii.gz"))
        cases = discover_cases(self.root)
        self.assertEqual([c.case_id for c in cases], ["c1"])

    def test_hidden_sub_dirs_are_skipped(self):
        make_case(self.root, "c1", ["t1c", "seg"])
        make_case(self.root, ".dropbox.cache", ["t1c", "seg"])
        self.assertEqual([c.case_id for c in discover_cases(self.root)], ["c1"])

    def test_ordering_is_natural_and_stable(self):
        for cid in ("YG_1LYOW0QK90GP_9", "YG_1LYOW0QK90GP_13", "YG_1LYOW0QK90GP_10",
                    "YG_83ATT6753PYK_11", "YG_83ATT6753PYK_4"):
            make_case(self.root, cid, ["t1c", "pred_seg"])
        ids = [c.case_id for c in discover_cases(self.root)]
        self.assertEqual(
            ids,
            ["YG_1LYOW0QK90GP_9", "YG_1LYOW0QK90GP_10", "YG_1LYOW0QK90GP_13",
             "YG_83ATT6753PYK_4", "YG_83ATT6753PYK_11"],
        )
        self.assertEqual(sorted(ids), sorted([c.case_id for c in discover_cases(self.root)]))

    def test_cases_are_keyed_on_directory_not_case_id(self):
        # the same case_id legitimately appears in several batch dirs
        a = make_case(os.path.join(self.root, "batch_01"), "YG_DUP_2", ["t1c", "pred_seg"])
        b = make_case(os.path.join(self.root, "batch_02"), "YG_DUP_2", ["t1c", "pred_seg"])
        first = discover_cases(os.path.dirname(a))[0]
        second = discover_cases(os.path.dirname(b))[0]
        self.assertEqual(first.case_id, second.case_id)
        self.assertNotEqual(first.directory, second.directory)

    def test_reviewed_flag_across_a_batch(self):
        make_case(self.root, "c1", ["t1c", "pred_seg"])
        make_case(self.root, "c2", ["t1c", "pred_seg", "reviewed_seg"])
        cases = {c.case_id: c for c in discover_cases(self.root)}
        self.assertFalse(cases["c1"].is_reviewed)
        self.assertTrue(cases["c2"].is_reviewed)
        # "skip already-reviewed" filter
        self.assertEqual([c.case_id for c in discover_cases(self.root) if not c.is_reviewed],
                         ["c1"])

    def test_reviewed_output_does_not_change_discovery(self):
        # self-discovery trap: reviewed_seg ends with "seg"
        make_case(self.root, "c1", ["t1c", "pred_seg"])
        before = discover_cases(self.root)[0]
        touch(os.path.join(self.root, "c1", "c1_reviewed_seg.nii.gz"))
        after = discover_cases(self.root)[0]
        self.assertEqual(set(after.masks), set(before.masks))
        self.assertEqual(set(after.images), set(before.images))
        self.assertTrue(after.is_reviewed)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class TestHelpers(unittest.TestCase):
    def test_is_nifti(self):
        self.assertTrue(dataset.is_nifti("a_seg.nii.gz"))
        self.assertTrue(dataset.is_nifti("a_seg.nii"))
        self.assertTrue(dataset.is_nifti("A_SEG.NII.GZ"))
        self.assertFalse(dataset.is_nifti("a_seg.nii.gz.md5"))
        self.assertFalse(dataset.is_nifti("notes.txt"))
        self.assertFalse(dataset.is_nifti(".a_seg.nii.gz"))
        self.assertFalse(dataset.is_nifti("a_seg (1).nii.gz"))
        self.assertFalse(dataset.is_nifti("a_seg (x's conflicted copy 2026-01-01).nii.gz"))
        self.assertFalse(dataset.is_nifti(""))

    def test_nifti_stem(self):
        self.assertEqual(dataset.nifti_stem("/x/y/a_b_seg.nii.gz"), "a_b_seg")
        self.assertEqual(dataset.nifti_stem("a_b_seg.nii"), "a_b_seg")
        self.assertEqual(dataset.nifti_stem("plain.txt"), "plain.txt")

    def test_natural_key_orders_unpadded_integers(self):
        values = ["x_10", "x_9", "x_1", "y_2"]
        self.assertEqual(sorted(values, key=dataset.natural_key),
                         ["x_1", "x_9", "x_10", "y_2"])


# --------------------------------------------------------------------------- #
# real data (skips cleanly when the Dropbox tree is not mounted)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(os.path.isdir(REAL_BATCH), "real batch_01 not available: %s" % REAL_BATCH)
class TestRealBatch01(unittest.TestCase):
    KNOWN_CASE = "YG_74M8KT2P9W2S_3"

    @classmethod
    def setUpClass(cls):
        cls.cases = discover_cases(REAL_BATCH)
        cls.by_id = {c.case_id: c for c in cls.cases}

    def test_discovers_cases(self):
        # The live tree grows and re-syncs, so assert the shape of the result
        # rather than a snapshot count.
        self.assertGreater(len(self.cases), 0)
        self.assertEqual(len(self.by_id), len(self.cases), "duplicate case ids")

    def test_known_case_keys(self):
        case = self.by_id[self.KNOWN_CASE]
        self.assertEqual(set(case.images), {"t1c"})
        self.assertEqual(set(case.masks), {"seg", "pred_seg"})
        self.assertEqual(case.default_mask_path(), case.masks["seg"])
        self.assertTrue(all(os.path.isfile(p) for p in case.images.values()))
        self.assertTrue(all(os.path.isfile(p) for p in case.masks.values()))

    def test_is_reviewed_agrees_with_the_files_on_disk(self):
        # Do not assert that nothing is reviewed: the extension writes
        # <case>_reviewed_seg.nii.gz into these very directories, so that
        # would start failing after the first real review session.
        for case in self.cases:
            self.assertEqual(
                bool(case.is_reviewed),
                os.path.isfile(case.reviewed_path),
                case.case_id,
            )

    def test_every_case_has_t1c_and_a_default_mask(self):
        for case in self.cases:
            self.assertEqual(set(case.images), {"t1c"}, case.case_id)
            self.assertIsNotNone(case.default_mask_path(), case.case_id)
            self.assertIn("pred_seg", case.masks, case.case_id)

    def test_cases_without_ground_truth_fall_back_to_pred_seg(self):
        no_gt = [c for c in self.cases if "seg" not in c.masks]
        for case in no_gt:
            self.assertEqual(case.default_mask_path(), case.masks["pred_seg"],
                             case.case_id)

    def test_parent_dir_yields_zero_cases(self):
        parent = os.path.dirname(REAL_BATCH)
        self.assertEqual(discover_cases(parent), [])

    def test_single_case_dir_as_root(self):
        case_dir = self.by_id[self.KNOWN_CASE].directory
        cases = discover_cases(case_dir)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, self.KNOWN_CASE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
