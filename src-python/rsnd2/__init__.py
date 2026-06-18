"""nd2-compatible Python API backed by the rsnd2 parser."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore[assignment]
    version = None  # type: ignore[assignment]

from . import structures
from ._binary import BinaryLayer, BinaryLayers
from ._nd2file import ND2File, imread
from ._util import AXIS, is_legacy, is_supported_file, rescue_nd2

try:
    __version__ = version("rsnd2") if version else "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "AXIS",
    "BinaryLayer",
    "BinaryLayers",
    "ND2File",
    "__version__",
    "imread",
    "is_legacy",
    "is_supported_file",
    "rescue_nd2",
    "structures",
]
