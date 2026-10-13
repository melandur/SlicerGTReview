"""GTReviewLib — pure-python helpers for the GTReview Slicer extension.

The pure modules import nothing of ``slicer``, ``vtk`` or ``qt``, so they are
unit-testable under plain ``PythonSlicer -m unittest``.  Two modules are the
Slicer side and are imported only inside Slicer, never from here: the Sphere
threshold effect, and ``recording`` (microphone and video capture), whose pure
halves are ``recordlog`` and ``wavfile``.

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
