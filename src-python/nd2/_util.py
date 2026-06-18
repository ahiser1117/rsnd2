from __future__ import annotations

from collections import namedtuple
from os import PathLike
from typing import Any

from . import _ffi


class AXIS:
    TIME = "T"
    POSITION = "P"
    CHANNEL = "C"
    Z = "Z"
    Y = "Y"
    X = "X"
    RGB = "S"

    _MAP = {
        "TimeLoop": TIME,
        "NETimeLoop": TIME,
        "XYPosLoop": POSITION,
        "ZStackLoop": Z,
        "ChannelLoop": CHANNEL,
    }

    @classmethod
    def frame_coords(cls) -> set[str]:
        return {cls.CHANNEL, cls.Y, cls.X, cls.RGB}


VoxelSize = namedtuple("VoxelSize", "x y z")


def is_supported_file(path: str | PathLike[str]) -> bool:
    try:
        probe = _ffi.version_probe(path)
    except Exception:
        return False
    return probe.get("variant") == "ModernChunked"


def is_legacy(path: str | PathLike[str]) -> bool:
    try:
        return _ffi.version_probe(path).get("variant") == "LegacyJp2Like"
    except Exception:
        return False


def rescue_nd2(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("Corrupt-frame rescue is not implemented in nd2-rs yet")
