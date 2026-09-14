"""Unit tests for GTReviewLib.undobudget -- plain unittest, no Slicer needed.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_undobudget.py' -v
"""

import os
import sys
import unittest

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from GTReviewLib.undobudget import MIB, EditPeaks, states_for_budget  # noqa: E402


class StatesForBudgetTest(unittest.TestCase):
    def test_divides_the_budget_by_the_state_size(self):
        self.assertEqual(states_for_budget(10 * MIB, 1024 * MIB, 5, 200), 102)

    def test_small_states_are_capped_at_the_maximum(self):
        self.assertEqual(states_for_budget(MIB // 8, 1024 * MIB, 5, 200), 200)

    def test_huge_states_keep_the_minimum(self):
        self.assertEqual(states_for_budget(400 * MIB, 1024 * MIB, 5, 200), 5)

    def test_a_state_bigger_than_the_whole_budget_keeps_the_minimum(self):
        self.assertEqual(states_for_budget(2048 * MIB, 1024 * MIB, 5, 200), 5)

    def test_exact_fit_counts_the_last_state(self):
        self.assertEqual(states_for_budget(MIB, 7 * MIB, 1, 200), 7)
        self.assertEqual(states_for_budget(MIB, 7 * MIB - 1, 1, 200), 6)

    def test_an_empty_mask_costs_nothing(self):
        self.assertEqual(states_for_budget(0, 1024 * MIB, 5, 200), 200)
        self.assertEqual(states_for_budget(-3, 1024 * MIB, 5, 200), 200)

    def test_a_negative_budget_is_no_budget(self):
        self.assertEqual(states_for_budget(MIB, -1, 5, 200), 5)

    def test_minimum_never_exceeds_maximum(self):
        self.assertEqual(states_for_budget(400 * MIB, 1024 * MIB, 50, 20), 20)

    def test_minimum_is_at_least_one(self):
        # zero states would switch Slicer's undo off (SaveState refuses below 1)
        self.assertEqual(states_for_budget(400 * MIB, MIB, 0, 200), 1)

    def test_maximum_must_allow_a_state(self):
        with self.assertRaises(ValueError):
            states_for_budget(MIB, MIB, 1, 0)

    def test_result_is_a_plain_int_for_the_qt_setter(self):
        self.assertIs(type(states_for_budget(3.0 * MIB, 1024.0 * MIB, 5.0, 200.0)), int)


class EditPeaksTest(unittest.TestCase):
    def test_nothing_recorded_prices_the_mask_as_it_stands(self):
        peaks = EditPeaks(10)
        self.assertEqual(peaks.estimate(123), 123)

    def test_an_edit_is_priced_at_its_largest_size(self):
        peaks = EditPeaks(10)
        peaks.begin(100)
        peaks.observe(300)
        peaks.observe(200)
        peaks.end(250)
        self.assertEqual(peaks.estimate(50), 300)

    def test_states_saved_before_a_shrink_stay_expensive(self):
        peaks = EditPeaks(10)
        peaks.begin(1000)
        peaks.end(1000)
        peaks.begin(1000)  # a delete shrinks the labelmap
        peaks.end(10)
        self.assertEqual(peaks.estimate(10), 1000)

    def test_old_edits_age_out_after_window_edits(self):
        peaks = EditPeaks(3)
        peaks.begin(1000)
        peaks.end(1000)
        for _ in range(2):
            peaks.begin(10)
            peaks.end(10)
        self.assertEqual(peaks.estimate(10), 1000, "still inside the window")
        peaks.begin(10)
        peaks.end(10)
        self.assertEqual(peaks.estimate(10), 10, "pushed out by three newer edits")

    def test_the_edit_in_progress_counts(self):
        peaks = EditPeaks(10)
        peaks.begin(10)
        peaks.observe(500)
        self.assertTrue(peaks.editInProgress)
        self.assertEqual(peaks.estimate(20), 500)

    def test_a_growing_mask_counts_before_the_edit_ends(self):
        peaks = EditPeaks(10)
        peaks.begin(10)
        self.assertEqual(peaks.estimate(700), 700)

    def test_an_edit_that_never_ended_is_kept_when_the_next_begins(self):
        peaks = EditPeaks(10)
        peaks.begin(10)
        peaks.observe(900)  # button release lost
        peaks.begin(20)
        peaks.end(20)
        self.assertFalse(peaks.editInProgress)
        self.assertEqual(peaks.estimate(20), 900)

    def test_sizes_outside_an_edit_are_ignored(self):
        peaks = EditPeaks(10)
        peaks.observe(5000)  # an undo restoring a state that was already priced
        self.assertEqual(peaks.estimate(10), 10)

    def test_end_without_begin_records_the_size(self):
        peaks = EditPeaks(10)
        peaks.end(400)
        self.assertEqual(peaks.estimate(10), 400)

    def test_reset_forgets_everything(self):
        peaks = EditPeaks(10)
        peaks.begin(1000)
        peaks.end(1000)
        peaks.begin(2000)
        peaks.reset()
        self.assertFalse(peaks.editInProgress)
        self.assertEqual(peaks.estimate(7), 7)

    def test_negative_sizes_read_as_empty(self):
        peaks = EditPeaks(10)
        peaks.begin(-5)
        peaks.end(-1)
        self.assertEqual(peaks.estimate(-9), 0)

    def test_window_is_reported_and_validated(self):
        self.assertEqual(EditPeaks(200).window, 200)
        with self.assertRaises(ValueError):
            EditPeaks(0)

    def test_estimate_feeds_states_for_budget(self):
        peaks = EditPeaks(200)
        peaks.begin(30 * MIB)
        peaks.end(3 * MIB)  # far lesion deleted
        self.assertEqual(
            states_for_budget(peaks.estimate(3 * MIB), 1024 * MIB, 5, 200), 34,
            "priced at the 30 MiB states still in the history, not the 3 MiB mask",
        )


if __name__ == "__main__":
    unittest.main()
