from __future__ import annotations

import math
import mmap
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, SupportsInt

from . import _ffi
from ._util import AXIS, VoxelSize, is_supported_file
from .structures import Attributes, Channel, ChannelMeta, Contents, FrameMetadata, Metadata, TextInfo

_PREFIX_LEN = 8


class ND2File:
    """Open and read an ND2 file using the nd2-rs parser.

    This class follows the public shape of the upstream ``nd2.ND2File`` API.
    The current Rust core supports modern chunked ND2 indexing and raw,
    uncompressed payload reads. Full Nikon metadata normalization and
    compressed pixel decoding are intentionally reported as unsupported.
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        validate_frames: bool = False,
        search_window: int = 100,
    ) -> None:
        if validate_frames:
            warnings.warn(
                "validate_frames is accepted for API compatibility but is not implemented",
                stacklevel=2,
            )
        self._path = Path(path)
        self._search_window = search_window
        self._closed = False
        self._index: dict[str, Any] | None = None
        self._version_probe: dict[str, Any] | None = None
        self._file = None
        self._mmap = None
        self._reader = None

    @staticmethod
    def is_supported_file(path: str | PathLike[str]) -> bool:
        return is_supported_file(path)

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def closed(self) -> bool:
        return self._closed

    def open(self) -> None:
        if self._closed:
            self._closed = False

    def close(self) -> None:
        self._closed = True
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._mmap is not None:
            try:
                self._mmap.close()
                self._mmap = None
            except BufferError:
                pass
        if self._file is not None and self._mmap is None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ND2File":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_closed"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _ensure_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed ND2File")

    def _ensure_index(self) -> dict[str, Any]:
        if self._index is None:
            self._index = _ffi.index(self._path)
        return self._index

    def _ensure_version_probe(self) -> dict[str, Any]:
        if self._version_probe is None:
            self._version_probe = _ffi.version_probe(self._path)
        return self._version_probe

    def _ensure_mmap(self):
        self._ensure_open()
        if self._mmap is None:
            self._file = open(self._path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def _ensure_reader(self):
        self._ensure_open()
        if self._reader is None:
            self._reader = _ffi.Reader(self._path)
        return self._reader

    def read_batch_to_torch(self, indices, *, device=None, n_threads: int | None = None):
        """Stream a batch of frames straight into a contiguous ``torch`` tensor
        of shape ``(len(indices), C, Y, X)``.

        A single Rust call packs every requested plane's pixel bytes into one
        contiguous host buffer (optionally using several concurrent positional
        reads), the buffer is copied to ``device`` in one transfer, and the
        per-pixel component interleave is undone on the destination device. This
        avoids the per-frame Python overhead and the strided host gather that a
        ``np.stack`` of individual frames incurs.
        """
        import os

        import torch

        np = _numpy()
        self._ensure_open()
        attrs = self.attributes
        idx_list = [int(i) for i in indices]

        # The fast packed path needs raw, uncompressed, unpadded rows so the
        # contiguous bytes reshape cleanly into (N, Y, X, components).
        itemsize = attrs.bitsPerComponentInMemory // 8
        packed_width_bytes = attrs.widthPx * attrs.componentCount * itemsize
        fast = (
            attrs.compressionType is None
            and attrs.widthBytes == packed_width_bytes
            and attrs.heightPx > 0
            and attrs.widthPx > 0
        )
        if not fast:
            arr = np.stack([np.asarray(self.read_frame(i)) for i in idx_list])
            t = torch.from_numpy(np.ascontiguousarray(arr))
            return t.to(device, non_blocking=True) if device is not None else t

        if n_threads is None:
            n_threads = int(os.environ.get("ND2_RS_READ_THREADS", "1"))

        n = len(idx_list)
        height = attrs.heightPx
        width = attrs.widthPx
        comp = attrs.componentCount
        pixel_len = height * attrs.widthBytes
        total = n * pixel_len

        idx = np.asarray(idx_list, dtype=np.uint64)
        host = np.empty(total, dtype=np.uint8)
        reader = self._ensure_reader()
        reader.read_frames_into(
            idx.ctypes.data, n, _PREFIX_LEN, host.ctypes.data, total, max(1, int(n_threads))
        )

        arr = host.view(self.dtype).reshape(n, height, width, comp)
        t = torch.from_numpy(arr)
        if device is not None:
            t = t.to(device, non_blocking=True)
        # (N, Y, X, C) -> (N, C, Y, X), undone on-device where it is cheap.
        return t.permute(0, 3, 1, 2).contiguous()

    @property
    def index(self) -> Mapping[str, Any]:
        try:
            return MappingProxyType(self._ensure_index())
        except ValueError as exc:
            return MappingProxyType({"error": str(exc)})

    @property
    def version(self) -> tuple[int, ...]:
        value = self._ensure_version_probe().get("signature_version")
        if isinstance(value, str) and value.lower().startswith("ver"):
            parts = []
            for part in value[3:].split("."):
                try:
                    parts.append(int(part))
                except ValueError:
                    break
            if parts:
                return tuple(parts)
        return (-1, -1)

    @property
    def is_legacy(self) -> bool:
        return self._ensure_version_probe().get("variant") == "LegacyJp2Like"

    @property
    def attributes(self) -> Attributes:
        parsed = self._ensure_index().get("attributes")
        if isinstance(parsed, dict):
            return Attributes(
                bitsPerComponentInMemory=int(parsed["bitsPerComponentInMemory"]),
                bitsPerComponentSignificant=int(parsed["bitsPerComponentSignificant"]),
                componentCount=int(parsed["componentCount"]),
                heightPx=int(parsed["heightPx"]),
                pixelDataType=str(parsed["pixelDataType"]),
                sequenceCount=int(parsed["sequenceCount"]),
                widthBytes=int(parsed["widthBytes"]),
                widthPx=int(parsed["widthPx"]),
                compressionLevel=parsed.get("compressionLevel"),
                compressionType=parsed.get("compressionType"),
                tileHeightPx=parsed.get("tileHeightPx"),
                tileWidthPx=parsed.get("tileWidthPx"),
                channelCount=int(parsed.get("channelCount") or 1),
            )

        height, width, dtype_name, bytes_per_component = self._guessed_image_layout()
        sequence_count = int(self._ensure_index().get("plane_count", 0))
        return Attributes(
            bitsPerComponentInMemory=bytes_per_component * 8,
            bitsPerComponentSignificant=bytes_per_component * 8,
            componentCount=1,
            heightPx=height,
            pixelDataType="unsigned" if dtype_name.startswith("uint") else dtype_name,
            sequenceCount=sequence_count,
            widthBytes=width * bytes_per_component,
            widthPx=width,
            compressionLevel=None,
            compressionType=None,
            tileHeightPx=None,
            tileWidthPx=None,
            channelCount=1,
        )

    @property
    def text_info(self) -> TextInfo:
        return TextInfo(description=f"ND2 parsed by nd2-rs from {self._path.name}")

    @property
    def rois(self) -> dict[int, Any]:
        return {}

    @property
    def experiment(self) -> list[Any]:
        return []

    def events(self, *, orient: str = "records", null_value: Any = float("nan")) -> Any:
        if orient == "records":
            return []
        if orient in {"dict", "list"}:
            return {}
        raise ValueError("orient must be one of 'records', 'dict', or 'list'")

    def unstructured_metadata(
        self,
        *,
        strip_prefix: bool = True,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        chunks = dict(self._ensure_index().get("chunk_name_counts", {}))
        if include is not None:
            chunks = {k: v for k, v in chunks.items() if k in include}
        if exclude is not None:
            chunks = {k: v for k, v in chunks.items() if k not in exclude}
        return {
            "nd2_rs_index": {
                "variant": self._ensure_index().get("variant"),
                "signature_version": self._ensure_index().get("signature_version"),
                "chunk_name_counts": chunks,
            }
        }

    @property
    def metadata(self) -> Metadata:
        return Metadata(
            contents=Contents(channelCount=1, frameCount=self.attributes.sequenceCount),
            channels=[Channel(ChannelMeta(name="Channel 0", index=0))],
        )

    def frame_metadata(self, seq_index: int | tuple[Any, ...]) -> FrameMetadata:
        if isinstance(seq_index, tuple):
            seq_index = int(self._seq_index_from_coords(seq_index))
        self._validate_sequence(int(seq_index))
        return FrameMetadata(
            contents=Contents(channelCount=1, frameCount=self.attributes.sequenceCount),
            channels=[],
        )

    @property
    def custom_data(self) -> dict[str, Any]:
        return {}

    def jobs(self) -> None:
        return None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.sizes.values())

    @property
    def sizes(self) -> Mapping[str, int]:
        attrs = self.attributes
        sizes: OrderedDict[str, int] = OrderedDict()
        if attrs.sequenceCount > 1:
            sizes[AXIS.TIME] = attrs.sequenceCount
        if attrs.componentCount > 1:
            sizes[AXIS.CHANNEL] = attrs.componentCount
        if attrs.heightPx > 1:
            sizes[AXIS.Y] = attrs.heightPx
        if attrs.widthPx > 1:
            sizes[AXIS.X] = attrs.widthPx
        return MappingProxyType(sizes)

    @property
    def is_rgb(self) -> bool:
        return self.components_per_channel in (3, 4)

    @property
    def components_per_channel(self) -> int:
        attrs = self.attributes
        return attrs.componentCount // (attrs.channelCount or 1)

    @property
    def size(self) -> int:
        return int(math.prod(self.shape)) if self.shape else 0

    @property
    def dtype(self):
        np = _numpy()
        attrs = self.attributes
        kind = "f" if attrs.pixelDataType == "float" else "u"
        return np.dtype(f"{kind}{attrs.bitsPerComponentInMemory // 8}")

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def voxel_size(self, channel: int = 0) -> VoxelSize:
        return VoxelSize(1.0, 1.0, 1.0)

    def asarray(self, position: int | None = None):
        self._ensure_open()
        np = _numpy()
        if position not in (None, 0):
            raise IndexError("Position is out of range. Only 1 position available")
        count = self.attributes.sequenceCount
        frames = [self.read_frame(i) for i in range(count)]
        if not frames:
            return np.asarray([], dtype=self.dtype)
        if count == 1:
            return frames[0]
        return np.stack(frames)

    def __array__(self):
        return self.asarray()

    def read_frame(self, frame_index: SupportsInt):
        self._ensure_open()
        np = _numpy()
        index = int(frame_index)
        self._validate_sequence(index)
        attrs = self.attributes
        plane = self._plane_record(index)
        frame_shape = self._raw_frame_shape()

        if attrs.compressionType == "lossless":
            import zlib

            payload = _ffi.read_plane_payload(self._path, index)
            arr = np.ndarray(
                shape=frame_shape,
                dtype=self.dtype,
                buffer=zlib.decompress(payload[_PREFIX_LEN:]),
                strides=self._strides(),
            )
        else:
            arr = np.ndarray(
                shape=frame_shape,
                dtype=self.dtype,
                buffer=self._ensure_mmap(),
                offset=int(plane["payload_offset"]) + _PREFIX_LEN,
                strides=self._strides(),
            )
        return arr.transpose((2, 0, 1, 3)).squeeze()

    def _get_frame(self, index: SupportsInt):
        warnings.warn(
            'Use of "_get_frame" is deprecated, use the public "read_frame" instead.',
            stacklevel=2,
        )
        return self.read_frame(index)

    @property
    def loop_indices(self) -> tuple[dict[str, int], ...]:
        return tuple({"T": i} for i in range(self.attributes.sequenceCount))

    @property
    def binary_data(self) -> None:
        return None

    def ome_metadata(self, *, include_unstructured: bool = True, tiff_file_name: str | None = None):
        raise NotImplementedError("OME metadata generation is not implemented in nd2-rs yet")

    def write_tiff(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("TIFF export is not implemented in nd2-rs yet")

    def write_ome_zarr(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("OME-Zarr export is not implemented in nd2-rs yet")

    def to_dask(self, wrapper: bool = True, copy: bool = True):
        import dask.array as da

        arr = self.asarray()
        return da.from_array(arr.copy() if copy else arr, chunks=tuple(1 for _ in arr.shape))

    def to_xarray(
        self,
        delayed: bool = True,
        squeeze: bool = True,
        position: int | None = None,
        copy: bool = True,
    ):
        import xarray as xr

        data = self.to_dask(copy=copy) if delayed else self.asarray(position)
        dims = list(self.sizes)
        xarr = xr.DataArray(
            data,
            dims=dims,
            attrs={
                "metadata": {
                    "metadata": self.metadata,
                    "experiment": self.experiment,
                    "attributes": self.attributes,
                    "text_info": self.text_info,
                }
            },
        )
        return xarr.squeeze() if squeeze else xarr

    def _seq_index_from_coords(self, coords: tuple[Any, ...]) -> int:
        if not coords:
            return 0
        return int(coords[0])

    def _validate_sequence(self, sequence: int) -> None:
        count = self.attributes.sequenceCount
        if sequence < 0 or sequence >= count:
            raise IndexError(f"frame index {sequence} out of range for {count} frames")

    def _plane_record(self, sequence: int) -> dict[str, Any]:
        planes = self._ensure_index().get("planes", [])
        if sequence < len(planes):
            plane = planes[sequence]
            if int(plane.get("sequence", -1)) == sequence:
                return plane
        for plane in planes:
            if int(plane.get("sequence", -1)) == sequence:
                return plane
        raise IndexError(f"frame index {sequence} is not present in the ND2 chunk map")

    def _raw_frame_shape(self) -> tuple[int, int, int, int]:
        attrs = self.attributes
        return (
            attrs.heightPx,
            attrs.widthPx or -1,
            attrs.channelCount or 1,
            self.components_per_channel,
        )

    def _strides(self) -> tuple[int, int, int, int] | None:
        attrs = self.attributes
        if not attrs.widthPx or not attrs.widthBytes:
            return None
        bytes_per_component = attrs.bitsPerComponentInMemory // 8
        expected_width_bytes = attrs.widthPx * bytes_per_component * attrs.componentCount
        if attrs.widthBytes == expected_width_bytes:
            return None
        return (
            attrs.widthBytes,
            attrs.componentCount * bytes_per_component,
            bytes_per_component,
            bytes_per_component,
        )

    def _guessed_image_layout(
        self, payload_len: int | None = None
    ) -> tuple[int, int, str, int]:
        if payload_len is None:
            planes = self._ensure_index().get("planes", [])
            if not planes:
                return (0, 0, "uint8", 1)
            payload_len = max(0, int(planes[0].get("payload_len", 0)) - _PREFIX_LEN)
        if payload_len and payload_len % 2 == 0:
            pixels = payload_len // 2
            side = int(math.isqrt(pixels))
            if side * side == pixels:
                return (side, side, "uint16", 2)
        return (1, payload_len, "uint8", 1)

    def __repr__(self) -> str:
        details = "closed" if self.closed else f"{self.dtype}: {dict(self.sizes)!r}"
        return f"<ND2File at {self.path!r} ({details})>"


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required for array output from nd2-rs") from exc
    return np


def imread(
    file: str | PathLike[str],
    *,
    dask: bool = False,
    xarray: bool = False,
    validate_frames: bool = False,
):
    with ND2File(file, validate_frames=validate_frames) as nd2:
        if xarray:
            return nd2.to_xarray(delayed=dask)
        if dask:
            return nd2.to_dask()
        return nd2.asarray()
