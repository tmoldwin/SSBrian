"""Top-level package for SS_RNNs experiments.

This file allows subpackages such as `recurrent` and `feedforward`
to be imported using absolute paths (e.g., `brian_experiment.recurrent`)."""

import os
import sys

_PKG_DIR = os.path.dirname(__file__)
_EXTRA_PATHS = [
    os.path.join(_PKG_DIR, "shared"),
    os.path.join(_PKG_DIR, "shared", "core"),
    os.path.join(_PKG_DIR, "shared", "analysis"),
    os.path.join(_PKG_DIR, "shared", "plotting"),
]
for _path in _EXTRA_PATHS:
    if _path not in sys.path:
        sys.path.append(_path)

__all__ = ["recurrent", "feedforward"]