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


class _Status(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("error", ctypes.c_void_p),
    ]


class _OpenResult(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_void_p),
        ("status", ctypes.c_int),
        ("error", ctypes.c_void_p),
    ]


def _library_names() -> list[str]:
    if sys.platform == "win32":
        return ["rsnd2.dll"]
    if sys.platform == "darwin":
        return ["librsnd2.dylib"]
    return ["librsnd2.so"]


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
    if env_path := os.environ.get("RSND2_LIBRARY"):
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
        lib.rsnd2_free_string.argtypes = [ctypes.c_void_p]
        lib.rsnd2_free_string.restype = None
        lib.rsnd2_free_buffer.argtypes = [_Buffer]
        lib.rsnd2_free_buffer.restype = None
        for name in ("rsnd2_version_probe_json", "rsnd2_summary_json", "rsnd2_index_json"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_char_p]
            fn.restype = ctypes.c_void_p
        # Optional in older shared libraries; the Python wrapper falls back to
        # the full index parse when this lightweight (no per-plane) call is absent.
        meta = getattr(lib, "rsnd2_index_meta_json", None)
        if meta is not None:
            meta.argtypes = [ctypes.c_char_p]
            meta.restype = ctypes.c_void_p
        lib.rsnd2_read_plane_payload.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.rsnd2_read_plane_payload.restype = _Buffer
        lib.rsnd2_reader_open.argtypes = [ctypes.c_char_p]
        lib.rsnd2_reader_open.restype = _OpenResult
        lib.rsnd2_reader_free.argtypes = [ctypes.c_void_p]
        lib.rsnd2_reader_free.restype = None
        lib.rsnd2_reader_read_frames.argtypes = [
            ctypes.c_void_p,            # handle
            ctypes.c_void_p,            # indices (*const u64)
            ctypes.c_size_t,            # n_indices
            ctypes.c_uint64,            # prefix_len
            ctypes.c_void_p,            # out_ptr
            ctypes.c_size_t,            # out_len
            ctypes.c_size_t,            # n_threads
        ]
        lib.rsnd2_reader_read_frames.restype = _Status
        # Optional in older shared libraries (Linux-only O_DIRECT span read).
        span = getattr(lib, "rsnd2_reader_read_span_direct", None)
        if span is not None:
            span.argtypes = [
                ctypes.c_void_p,            # handle
                ctypes.c_uint64,            # offset
                ctypes.c_size_t,            # len
                ctypes.c_void_p,            # out_ptr
                ctypes.c_size_t,            # out_len
                ctypes.c_size_t,            # n_threads
            ]
            span.restype = _Status
        return lib
    detail = "; ".join(errors) if errors else "no compiled library was found"
    raise ImportError(
        "Could not load rsnd2 shared library. Run `cargo build --lib` from the "
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
        lib().rsnd2_free_string(ptr)
    data = json.loads(raw)
    if "error" in data:
        raise ValueError(data["error"])
    return data


def version_probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _json_call("rsnd2_version_probe_json", path)


def summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _json_call("rsnd2_summary_json", path)


def index(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _json_call("rsnd2_index_json", path)


def index_meta(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse file metadata (attributes, plane_count, chunk counts) without the
    per-plane record array. Falls back to the full index parse if the native
    library predates ``rsnd2_index_meta_json``."""
    if getattr(lib(), "rsnd2_index_meta_json", None) is None:
        return index(path)
    return _json_call("rsnd2_index_meta_json", path)


def read_plane_payload(path: str | os.PathLike[str], sequence: int) -> bytes:
    buffer = lib().rsnd2_read_plane_payload(_path_bytes(path), sequence)
    try:
        if buffer.status:
            error = ctypes.string_at(buffer.error).decode("utf-8") if buffer.error else "read failed"
            raise ValueError(error)
        return ctypes.string_at(buffer.ptr, buffer.len)
    finally:
        lib().rsnd2_free_buffer(buffer)


def _take_error(ptr: int | None, default: str) -> str:
    """Decode and free an owned error string from the Rust side exactly once,
    even if decoding fails."""
    if not ptr:
        return default
    try:
        return ctypes.string_at(ptr).decode("utf-8", "replace")
    finally:
        lib().rsnd2_free_string(ptr)


def _raise_status(status: _Status) -> None:
    if status.status:
        raise ValueError(_take_error(status.error, "rsnd2 call failed"))


class Reader:
    """Persistent handle for fast batched frame reads from one ND2 file.

    Wraps the Rust ``Nd2Reader`` (parsed index + open file descriptor) so that
    repeated batch reads avoid re-parsing the chunk map and re-opening the file.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        result = lib().rsnd2_reader_open(_path_bytes(path))
        if result.status or not result.handle:
            raise ValueError(_take_error(result.error, "open failed"))
        self._handle: int | None = result.handle

    def read_frames_into(
        self,
        indices_ptr: int,
        n_indices: int,
        prefix_len: int,
        out_ptr: int,
        out_len: int,
        n_threads: int,
    ) -> None:
        """Fill the caller-owned buffer at ``out_ptr`` (``out_len`` bytes) with the
        packed pixel bytes of the requested planes. ``indices_ptr`` points at an
        array of ``n_indices`` little-endian ``uint64`` plane sequence numbers."""
        if self._handle is None:
            raise ValueError("reader is closed")
        status = lib().rsnd2_reader_read_frames(
            self._handle,
            ctypes.c_void_p(indices_ptr),
            n_indices,
            prefix_len,
            ctypes.c_void_p(out_ptr),
            out_len,
            n_threads,
        )
        _raise_status(status)

    def read_span_direct(
        self,
        offset: int,
        length: int,
        out_ptr: int,
        out_len: int,
        n_threads: int,
    ) -> None:
        """Read the contiguous span ``[offset, offset+length)`` straight from the
        device into the page-locked buffer at ``out_ptr`` via ``O_DIRECT``,
        bypassing the page cache and ZFS ARC. ``offset`` and ``length`` must be
        4096-aligned and ``out_ptr`` page-aligned."""
        if self._handle is None:
            raise ValueError("reader is closed")
        fn = getattr(lib(), "rsnd2_reader_read_span_direct", None)
        if fn is None:
            raise NotImplementedError("native library lacks O_DIRECT span reads")
        status = fn(
            self._handle,
            ctypes.c_uint64(offset),
            length,
            ctypes.c_void_p(out_ptr),
            out_len,
            n_threads,
        )
        _raise_status(status)

    @staticmethod
    def has_direct() -> bool:
        return getattr(lib(), "rsnd2_reader_read_span_direct", None) is not None

    def close(self) -> None:
        # Clear the handle first so a failure (or re-entry via __del__) can never
        # free the same pointer twice.
        handle, self._handle = self._handle, None
        if handle is not None:
            lib().rsnd2_reader_free(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
