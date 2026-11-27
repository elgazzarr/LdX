"""Compatibility helpers across Flax NNX versions."""

from flax import nnx

# Flax >= 0.12 requires lists of submodules held as attributes to be wrapped in
# ``nnx.List`` (plain lists are treated as static data and raise an error).
# Older Flax versions have no ``nnx.List`` and handle plain lists directly.
ModuleList = getattr(nnx, "List", list)

__all__ = ["ModuleList"]
