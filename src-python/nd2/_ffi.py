from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_ubyte)),
        ("len", ctypes.c_size_t),
        ("status", ctypes.c_int),
        ("error", ctypes.c_void_p),
    ]


def _library_names() -> list[str]:
    if sys.platform == "win32":
        return ["nd2_rs.dll"]
    if sys.platform == "darwin":
        return ["libnd2_rs.dylib"]
    return ["libnd2_rs.so"]


def _candidate_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    source_root = here.parents[1]
    repo_root = here.parents[2]
    candidates: list[Path] = []
    for base in (
        here,
        source_root / "target" / "release",
        source_root / "target" / "debug",
        repo_root / "target" / "release",
        repo_root / "target" / "debug",
    ):
        candidates.extend(base / name for name in _library_names())
    if env_path := os.environ.get("ND2_RS_LIBRARY"):
        candidates.insert(0, Path(env_path))
    return candidates


def _load() -> ctypes.CDLL:
    errors: list[str] = []
    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        lib.nd2_rs_free_string.argtypes = [ctypes.c_void_p]
        lib.nd2_rs_free_string.restype = None
        lib.nd2_rs_free_buffer.argtypes = [_Buffer]
        lib.nd2_rs_free_buffer.restype = None
        for name in ("nd2_rs_version_probe_json", "nd2_rs_summary_json", "nd2_rs_index_json"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_char_p]
            fn.restype = ctypes.c_void_p
        lib.nd2_rs_read_plane_payload.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.nd2_rs_read_plane_payload.restype = _Buffer
        return lib
    detail = "; ".join(errors) if errors else "no compiled library was found"
    raise ImportError(
        "Could not load nd2-rs shared library. Run `cargo build --lib` from the "
        f"repository root or install the package with a build backend. Detail: {detail}"
    )


_LIB: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def _path_bytes(path: str | os.PathLike[str]) -> bytes:
    return os.fsencode(path)


def _json_call(name: str, path: str | os.PathLike[str]) -> dict[str, Any]:
    fn = getattr(lib(), name)
    ptr = fn(_path_bytes(path))
    if not ptr:
        raise RuntimeError(f"{name} returned a null pointer")
    try:
        raw = ctypes.string_at(ptr).decode("utf-8")
    finally:
        lib().nd2_rs_free_string(ptr)
    data = json.loads(raw)
    if "error" in data:
        raise ValueError(data["error"])
    return data


def version_probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _json_call("nd2_rs_version_probe_json", path)


def summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _json_call("nd2_rs_summary_json", path)


def index(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _json_call("nd2_rs_index_json", path)


def read_plane_payload(path: str | os.PathLike[str], sequence: int) -> bytes:
    buffer = lib().nd2_rs_read_plane_payload(_path_bytes(path), sequence)
    try:
        if buffer.status:
            error = ctypes.string_at(buffer.error).decode("utf-8") if buffer.error else "read failed"
            raise ValueError(error)
        return ctypes.string_at(buffer.ptr, buffer.len)
    finally:
        lib().nd2_rs_free_buffer(buffer)
