"""Slicer layout XML for the "Sequences (axial)" layout -- pure python.

One axial slice view per image sequence of a case, in a grid.  Nothing here
imports ``slicer``: the widget registers the XML this module produces with
``vtkMRMLLayoutNode.AddLayoutDescription`` and reads each view's
``singletontag`` back as the sequence key it should display.
"""

from typing import Iterable, List

from . import dataset

#: sequences shown first, in this order; anything else follows naturally
SEQUENCE_ORDER = ("t1", "t1c", "t2", "flair")

#: view colours (the coloured bar above each slice view), cycled
VIEW_COLORS = ("#F34A33", "#EDD54C", "#6EB04B", "#4C7ED9", "#B361D6", "#4CC9C9")


def sequence_order(keys: Iterable[str]) -> List[str]:
    """t1, t1c, t2, flair first (in that order), then the rest naturally.

    Matching is case-insensitive; ties inside the "rest" are broken by
    :func:`dataset.natural_key` and then the raw key, so the order is stable.
    """
    known = {k: i for i, k in enumerate(SEQUENCE_ORDER)}
    return sorted(
        keys, key=lambda k: (known.get(k.lower(), len(known)), dataset.natural_key(k), k)
    )


def grid_columns(count: int) -> int:
    """1 column for a single view, 2 for up to four, 3 beyond."""
    if count < 2:
        return 1
    return 2 if count <= 4 else 3


def sequences_layout_xml(keys: Iterable[str]) -> str:
    """Layout XML with one axial slice view per key, rows of :func:`grid_columns`.

    ``keys`` are used verbatim as ``singletontag`` and ``viewlabel``; they are
    expected in display order already (see :func:`sequence_order`).  Duplicate
    keys are collapsed to the first occurrence, as Slicer singletons would be.
    """
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    columns = grid_columns(len(keys))
    rows = max(1, (len(keys) + columns - 1) // columns)
    lines = ['<layout type="vertical">']
    for row in range(rows):
        lines.append('  <item><layout type="horizontal">')
        for index, key in enumerate(keys[row * columns:(row + 1) * columns], start=row * columns):
            lines.append(
                '    <item><view class="vtkMRMLSliceNode" singletontag="{tag}">'
                '<property name="orientation" action="default">Axial</property>'
                '<property name="viewlabel" action="default">{label}</property>'
                '<property name="viewcolor" action="default">{color}</property>'
                "</view></item>".format(
                    tag=key, label=key, color=VIEW_COLORS[index % len(VIEW_COLORS)]
                )
            )
        lines.append("  </layout></item>")
    lines.append("</layout>")
    return "\n".join(lines)
