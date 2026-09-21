"""torchff: force-field terms with custom CUDA kernels.

A CPU-only install (``TORCHFF_NO_CUDA=1``) has no compiled extensions; the submodules whose
import requires one are skipped here instead of breaking ``import torchff``, so that the
modules with pure-torch reference paths (``torchff.ffterms``) stay usable. Importing a
skipped submodule explicitly still raises the original ``ImportError``.
"""

import importlib as _importlib

__all__ = []

for _name in (
    "bond", "angle", "coulomb", "torsion", "vdw", "dispersion", "slater", "nblist", "nonbonded",
):
    try:
        _mod = _importlib.import_module(f".{_name}", __name__)
    except ImportError:  # extension not compiled (CPU-only install)
        continue
    _public = getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")])
    globals().update({n: getattr(_mod, n) for n in _public})
    __all__.extend(_public)

from . import ffterms  # noqa: E402  -- always importable; kernels optional

__all__.append("ffterms")
