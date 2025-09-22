"""LatentDynamiX (LdX) - A library for modeling partially observed dynamical systems."""

from ldx.__version__ import __version__

# Import main modules for easy access
from ldx import models
from ldx import inference
from ldx import utils
from ldx import datasets

__all__ = [
    "__version__",
    "models",
    "inference",
    "utils",
    "datasets",
]