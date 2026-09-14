"""Keep the undo history inside a memory budget -- pure python.

Slicer's segment editor keeps a fixed NUMBER of undo states, and each state is
a full copy of every labelmap changed since the state before it
(``vtkSegmentation::CopySegment`` deep-copies a representation unless the
previous state already holds an up-to-date one).  What a copy costs swings by
three orders of magnitude between cases -- a box of a few hundred kilobytes
around the lesions, or tens of megabytes once edits reach both ends of the
head -- so no fixed count is deep enough for the first and safe for the second.

This module turns a byte budget into a count.  The widget prices one state
from the mask as it stands and asks :func:`states_for_budget` how many fit.
Lowering Slicer's limit then drops the OLDEST states first
(``vtkSegmentationHistory::RemoveAllObsoleteStates`` pops from the front).

A state costs what the mask cost when the state was SAVED, not what it costs
now: after a delete shrinks the labelmap, the states from before the delete
are still large.  :class:`EditPeaks` therefore remembers the largest size seen
during each recent edit, and the price of a state is the largest of those, the
edit in progress and the mask as it stands.  Every edit that saves a state
saves at least one, so a history of at most ``window`` states never reaches
further back than the last ``window`` edits.

Nothing here imports ``slicer``.
"""

from collections import deque
from typing import Optional

#: bytes in a mebibyte
MIB = 1 << 20


def states_for_budget(state_bytes: int, budget_bytes: int, minimum: int, maximum: int) -> int:
    """How many undo states of *state_bytes* each fit in *budget_bytes*.

    The answer is clamped to ``[minimum, maximum]``.  *maximum* is the depth
    wanted when memory is no concern.  *minimum* keeps a few steps of undo
    even when one state is a large share of the budget, so the history can
    exceed the budget by at most ``minimum * state_bytes``.  A state of zero
    bytes or less (an empty mask) costs nothing and gets *maximum*.
    """
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("maximum must be at least 1, got {}".format(maximum))
    minimum = max(1, min(int(minimum), maximum))
    state_bytes = int(state_bytes)
    if state_bytes <= 0:
        return maximum
    fits = max(0, int(budget_bytes)) // state_bytes
    return max(minimum, min(maximum, fits))


class EditPeaks:
    """The largest mask size seen during each of the last *window* edits.

    Sizes are in bytes.  An edit is opened with :meth:`begin` (the size before
    it), watched with :meth:`observe` while it runs, and closed with
    :meth:`end`.  :meth:`estimate` is what one undo state should be priced at.
    """

    def __init__(self, window: int):
        window = int(window)
        if window < 1:
            raise ValueError("window must be at least 1, got {}".format(window))
        self._peaks = deque(maxlen=window)
        self._open: Optional[int] = None

    @property
    def window(self) -> int:
        return self._peaks.maxlen

    @property
    def editInProgress(self) -> bool:
        return self._open is not None

    def reset(self) -> None:
        """Forget every edit: the history they priced has been cleared."""
        self._peaks.clear()
        self._open = None

    def begin(self, nbytes: int) -> None:
        """An edit starts on a mask of *nbytes*.

        An edit still open is closed first, at its peak so far: a stroke whose
        button release never arrived (focus lost mid-drag) still saved states.
        """
        if self._open is not None:
            self._peaks.append(self._open)
        self._open = max(0, int(nbytes))

    def observe(self, nbytes: int) -> None:
        """The mask measured *nbytes* part-way through the edit in progress.

        Ignored when no edit is open: sizes seen during an undo or redo belong
        to states that were already priced when they were saved.
        """
        if self._open is not None:
            self._open = max(self._open, int(nbytes))

    def end(self, nbytes: int) -> None:
        """The edit in progress finished on a mask of *nbytes*.

        Without an open edit the size is recorded as an edit of its own.
        """
        peak = max(0, int(nbytes))
        if self._open is not None:
            peak = max(peak, self._open)
        self._peaks.append(peak)
        self._open = None

    def estimate(self, current_bytes: int) -> int:
        """Price of one undo state, given the mask now measures *current_bytes*."""
        values = [max(0, int(current_bytes))]
        if self._open is not None:
            values.append(self._open)
        values.extend(self._peaks)
        return max(values)
