"""GTReviewLib — pure-python helpers for the GTReview Slicer extension.

Nothing in this package may import ``slicer``, ``vtk`` or ``qt``: the modules are
unit-testable under plain ``PythonSlicer -m unittest``.

``dataset`` is always available (standard library only).  ``lesions`` and
``maskio`` need numpy / scipy / SimpleITK and are imported defensively so that
importing this package never fails when those are absent.
"""

from . import dataset  # noqa: F401
from . import layouts  # noqa: F401

__all__ = ["dataset", "layouts"]

for _name in ("lesions", "maskio"):
    try:
        _mod = __import__("{}.{}".format(__name__, _name), fromlist=[_name])
    except Exception:  # pragma: no cover - optional deps / not yet implemented
        pass
    else:
        globals()[_name] = _mod
        __all__.append(_name)
del _name
