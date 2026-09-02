"""Unit tests for GTReviewLib.layouts -- plain unittest, no Slicer needed.

Run with:
    PythonSlicer -m unittest discover -s Testing -p 'test_layouts.py' -v
"""

import os
import re
import sys
import unittest
import xml.etree.ElementTree as ET

_TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(os.path.dirname(_TESTING_DIR), "GTReview")
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from GTReviewLib import layouts  # noqa: E402
from GTReviewLib.layouts import (  # noqa: E402
    SEQUENCE_ORDER,
    VIEW_COLORS,
    grid_columns,
    sequence_order,
    sequences_layout_xml,
)


def _views(xml):
    """(singletontag, orientation, viewlabel, viewcolor) per view, in document order."""
    root = ET.fromstring(xml)
    out = []
    for view in root.iter("view"):
        props = {p.get("name"): p.text for p in view.findall("property")}
        out.append((
            view.get("singletontag"),
            props.get("orientation"),
            props.get("viewlabel"),
            props.get("viewcolor"),
        ))
    return out


def _rows(xml):
    """Number of views in each horizontal row."""
    root = ET.fromstring(xml)
    assert root.get("type") == "vertical"
    return [len(row.findall("item")) for row in root.findall("item/layout")]


class TestSequenceOrder(unittest.TestCase):
    def test_known_sequences_first_in_fixed_order(self):
        self.assertEqual(sequence_order(["flair", "t2", "t1c", "t1"]), ["t1", "t1c", "t2", "flair"])

    def test_matches_the_module_constant(self):
        self.assertEqual(tuple(sequence_order(reversed(SEQUENCE_ORDER))), SEQUENCE_ORDER)

    def test_unknown_keys_follow_naturally(self):
        self.assertEqual(
            sequence_order(["dwi", "t2", "adc", "t1", "b1000"]),
            ["t1", "t2", "adc", "b1000", "dwi"],
        )

    def test_unknown_keys_sort_numerically_not_lexically(self):
        self.assertEqual(sequence_order(["echo10", "echo2", "echo1"]), ["echo1", "echo2", "echo10"])

    def test_case_insensitive_match_keeps_original_spelling(self):
        self.assertEqual(sequence_order(["T2", "FLAIR", "t1"]), ["t1", "T2", "FLAIR"])

    def test_accepts_any_iterable(self):
        self.assertEqual(sequence_order({"t2": 1, "t1": 2}), ["t1", "t2"])
        self.assertEqual(sequence_order(iter(["t2", "t1"])), ["t1", "t2"])

    def test_empty(self):
        self.assertEqual(sequence_order([]), [])

    def test_does_not_modify_input(self):
        keys = ["t2", "t1"]
        sequence_order(keys)
        self.assertEqual(keys, ["t2", "t1"])


class TestGridColumns(unittest.TestCase):
    def test_shape(self):
        self.assertEqual([grid_columns(n) for n in range(0, 8)], [1, 1, 2, 2, 2, 3, 3, 3])


class TestSequencesLayoutXml(unittest.TestCase):
    def test_is_well_formed_xml(self):
        for n in range(0, 8):
            ET.fromstring(sequences_layout_xml(["s%d" % i for i in range(n)]))

    def test_one_axial_view_per_key_in_order(self):
        views = _views(sequences_layout_xml(["t1", "t1c", "t2", "flair"]))
        self.assertEqual([v[0] for v in views], ["t1", "t1c", "t2", "flair"])
        self.assertTrue(all(v[1] == "Axial" for v in views))
        self.assertEqual([v[2] for v in views], ["t1", "t1c", "t2", "flair"])

    def test_keys_are_used_verbatim_not_reordered(self):
        # ordering is sequence_order's job; the XML keeps what it is given
        self.assertEqual([v[0] for v in _views(sequences_layout_xml(["flair", "t1"]))], ["flair", "t1"])

    def test_grid_shapes(self):
        self.assertEqual(_rows(sequences_layout_xml(["a"])), [1])
        self.assertEqual(_rows(sequences_layout_xml(["a", "b"])), [2])
        self.assertEqual(_rows(sequences_layout_xml(["a", "b", "c"])), [2, 1])
        self.assertEqual(_rows(sequences_layout_xml(["a", "b", "c", "d"])), [2, 2])
        self.assertEqual(_rows(sequences_layout_xml(list("abcde"))), [3, 2])
        self.assertEqual(_rows(sequences_layout_xml(list("abcdefg"))), [3, 3, 1])

    def test_view_colors_cycle(self):
        keys = ["k%d" % i for i in range(len(VIEW_COLORS) + 2)]
        colors = [v[3] for v in _views(sequences_layout_xml(keys))]
        self.assertEqual(colors[: len(VIEW_COLORS)], list(VIEW_COLORS))
        self.assertEqual(colors[len(VIEW_COLORS):], list(VIEW_COLORS[:2]))
        self.assertTrue(all(re.match(r"^#[0-9A-Fa-f]{6}$", c) for c in colors))

    def test_duplicate_keys_collapse(self):
        views = _views(sequences_layout_xml(["t1", "t2", "t1"]))
        self.assertEqual([v[0] for v in views], ["t1", "t2"])

    def test_empty_gives_an_empty_row(self):
        xml = sequences_layout_xml([])
        self.assertEqual(_views(xml), [])
        self.assertEqual(_rows(xml), [0])

    def test_properties_are_defaults_so_the_user_can_change_them(self):
        root = ET.fromstring(sequences_layout_xml(["t1"]))
        actions = {p.get("action") for p in root.iter("property")}
        self.assertEqual(actions, {"default"})

    def test_deterministic(self):
        keys = ["t2", "t1", "adc"]
        self.assertEqual(sequences_layout_xml(keys), sequences_layout_xml(list(keys)))

    def test_module_exposes_the_pieces_the_widget_uses(self):
        self.assertTrue(callable(layouts.sequence_order))
        self.assertTrue(callable(layouts.sequences_layout_xml))
        self.assertEqual(layouts.SEQUENCE_ORDER, ("t1", "t1c", "t2", "flair"))


if __name__ == "__main__":
    unittest.main()
