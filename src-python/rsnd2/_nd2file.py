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

# Process-wide side CUDA streams for the pipelined copy, keyed by device, so a
# fresh ND2File per read (as the cold benchmark opens) does not recreate one
# inside the timed section.
_COPY_STREAMS: dict = {}

# Process-wide staging buffers for the O_DIRECT path, reused across ND2File
# instances so a fresh handle per read (as the cold benchmark opens) does not
# reallocate large page-locked host / device buffers inside the timed section.
# Page-locked allocation of a multi-hundred-MB span is expensive enough to
# dominate a fresh-handle cold read otherwise. Keyed by device for the GPU
# buffer; the pinned host buffer is device-independent.
_DIRECT_PIN = {"buf": None}
_DIRECT_DEV: dict = {}


def _direct_pin(nbytes):
    import torch

    buf = _DIRECT_PIN["buf"]
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        _DIRECT_PIN["buf"] = buf
    return buf


def _direct_dev(nbytes, device):
    import torch

    key = str(device)
    buf = _DIRECT_DEV.get(key)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _DIRECT_DEV[key] = buf
    return buf


def _copy_stream_for(device):
    import torch

    key = str(device)
    stream = _COPY_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _COPY_STREAMS[key] = stream
    return stream


class ND2File:
    """Open and read an ND2 file using the rsnd2 parser.

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
        self._planes: list[Any] | None = None
        self._version_probe: dict[str, Any] | None = None
        self._file = None
        self._mmap = None
        self._reader = None
        self._pin_u8 = None
        self._pin_pair = None

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
        self._pin_u8 = None
        self._pin_pair = None
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
            # Parse metadata WITHOUT the per-plane record array: attributes,
            # shape, dtype, and the fast read_batch_to_torch path never need it,
            # and serializing + JSON-parsing thousands of plane records adds
            # several milliseconds to every fresh open (the dominant cost of a
            # cold first-touch read at small batch sizes). The full plane table
            # is parsed lazily by _ensure_planes() only when read_frame/asarray
            # actually requires per-plane offsets.
            self._index = _ffi.index_meta(self._path)
            # Open the streaming reader eagerly while metadata is being parsed,
            # so a subsequent batch read does not pay the reader-open (a second
            # chunk-map parse) on its critical path. For the open -> inspect
            # attributes -> read pattern this shaves the parse off every read's
            # timed section (most visible on small cold batches); for sustained
            # streaming from one handle it is neutral (the reader is opened once
            # regardless). ND2_RS_EAGER_READER=0 disables it.
            import os as _os
            if _os.environ.get("ND2_RS_EAGER_READER", "1") != "0" and not self._closed:
                try:
                    self._ensure_reader()
                except Exception:
                    pass
        return self._index

    def _ensure_planes(self) -> list[Any]:
        """Return the per-plane record list, parsing the full index lazily.

        The metadata parse in _ensure_index() deliberately omits this array; it
        is only materialised here, on the first call that needs per-plane chunk
        offsets (read_frame / asarray / the layout-guess fallback)."""
        if self._planes is None:
            full = _ffi.index(self._path)
            self._planes = full.get("planes", [])
            # Backfill any metadata the lightweight parse may have lacked so a
            # subsequent attributes access stays consistent with the full parse.
            if self._index is not None and self._index.get("attributes") is None:
                self._index = full
        return self._planes

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

        n = len(idx_list)
        height = attrs.heightPx
        width = attrs.widthPx
        channel_count = attrs.channelCount or 1
        comps_per_channel = self.components_per_channel
        pixel_len = height * attrs.widthBytes
        total = n * pixel_len

        if n_threads is None:
            env = os.environ.get("ND2_RS_READ_THREADS")
            if env is not None:
                n_threads = int(env)
            else:
                # Adaptive: tiny batches are dominated by thread-spawn overhead,
                # so read them on one thread; larger batches benefit from many
                # concurrent positional reads (breaking the single-stream NFS
                # ceiling) and parallel host page-faults. ~1 MB per reader,
                # capped so we never oversubscribe.
                n_threads = 1 if total < (8 << 20) else min(16, max(1, total // (1 << 20)))
        n_threads = max(1, int(n_threads))

        idx = np.asarray(idx_list, dtype=np.uint64)
        reader = self._ensure_reader()

        torch_dtype = {
            "uint8": torch.uint8,
            "uint16": torch.uint16,
            "int16": torch.int16,
            "float32": torch.float32,
        }.get(np.dtype(self.dtype).name)

        def finalize(flat):
            """Reshape a 1-D device tensor of packed pixels into the same layout
            ``np.stack([read_frame(i) for i in indices])`` would produce: the raw
            (Y, X, channel, component) order is permuted to put channel first and
            then the per-frame singleton axes are squeezed, exactly mirroring
            ``read_frame``'s ``transpose((2, 0, 1, 3)).squeeze()``."""
            t = flat.reshape(n, height, width, channel_count, comps_per_channel)
            t = t.permute(0, 3, 1, 2, 4)  # (N, channel, Y, X, component)
            for ax in range(t.ndim - 1, 0, -1):  # squeeze non-batch singleton axes
                if t.shape[ax] == 1:
                    t = t.squeeze(ax)
            return t.contiguous()

        # ---- O_DIRECT genuine-cold path (opt-in via ND2_RS_O_DIRECT=1). On
        # ARC-cached local storage (ZFS) buffered reads are served from RAM, so
        # a "cold" read is impossible to measure without bypassing the cache;
        # O_DIRECT reads hit the device every time. It also unlocks far higher
        # throughput on fast local NVMe than the per-plane scatter path. We read
        # the one contiguous, block-aligned byte span covering the batch and undo
        # the inter-plane gaps with a strided gather on the GPU. Requires a
        # contiguous, constant-stride run of equal-length planes (the common
        # acquisition layout); otherwise we fall through to the scatter path. ----
        if (
            device is not None
            and torch_dtype is not None
            and os.environ.get("ND2_RS_O_DIRECT", "0") == "1"
            and _ffi.Reader.has_direct()
        ):
            direct = self._read_batch_odirect(
                reader, idx_list, n, height, width, channel_count,
                comps_per_channel, pixel_len, torch_dtype, device, finalize, n_threads,
            )
            if direct is not None:
                return direct

        if torch_dtype is None:
            # Uncommon dtype: fall back to a plain host buffer.
            host = np.empty(total, dtype=np.uint8)
            reader.read_frames_into(idx.ctypes.data, n, _PREFIX_LEN, host.ctypes.data, total, n_threads)
            t = torch.from_numpy(host.view(self.dtype))
            if device is not None:
                t = t.to(device, non_blocking=False)
            return finalize(t)

        # ---- pipelined path: overlap the host->device copy with the next
        # chunk's (genuinely cold) network read. The disk read is ~95% of the
        # cold cost and saturates the link on a single sequential stream, so the
        # H2D copy is pure serial tail latency unless hidden behind the next
        # read. Split the planes into a few contiguous chunks, double-buffer the
        # pinned host staging, and issue each chunk's copy on a side CUDA stream
        # while the CPU reads the following chunk. ----
        pipeline = (
            device is not None
            and n >= 2
            and os.environ.get("ND2_RS_PIPELINE", "1") != "0"
        )
        if pipeline:
            chunk_bytes = max(1, int(float(os.environ.get("ND2_RS_PIPELINE_CHUNK_MB", "64")) * (1 << 20)))
            max_chunks = int(os.environ.get("ND2_RS_PIPELINE_MAX_CHUNKS", "4"))
            # Few, large chunks: each chunk read keeps the full thread count (so
            # the scatter read still issues ~16 concurrent RPCs and saturates the
            # link), while 2-4 chunks suffice to hide most of the H2D copy with
            # minimal inter-chunk "network idle" bubbles. Many small chunks lose
            # RPC concurrency and add overhead, which costs more than the copy.
            n_chunks = max(2, min(max_chunks, round(total / chunk_bytes)))
            # Only worth pipelining once the batch is large enough that the copy
            # is a non-trivial serial tail (and each half still saturates reads).
            min_mb = float(os.environ.get("ND2_RS_PIPELINE_MIN_MB", "64"))
            pipeline = total >= int(min_mb * (1 << 20))
        if pipeline:
            return self._read_pipelined(
                reader, idx, n, pixel_len, total, n_chunks, torch_dtype, device, finalize
            )

        # ---- single-shot path (tiny batches, host-only output, or pipeline
        # disabled). Reusable pinned staging buffer: page-locked host memory
        # makes the host->device copy a single direct DMA (roughly 2x pageable)
        # and, being reused across calls, removes the per-call allocation +
        # first-touch page faults (which otherwise spike sharply for batches
        # above glibc's ~32 MB mmap threshold). ----
        pin = self._pin_u8
        if pin is None or pin.numel() < total:
            pin = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            self._pin_u8 = pin
        reader.read_frames_into(idx.ctypes.data, n, _PREFIX_LEN, pin.data_ptr(), total, n_threads)

        staging = pin[:total].view(torch_dtype)
        if device is not None:
            # Blocking copy: the pinned buffer is overwritten on the next call.
            dst = staging.to(device, non_blocking=False)
        else:
            dst = staging.clone()
        return finalize(dst)

    def _read_pipelined(
        self, reader, idx, n, pixel_len, total, n_chunks, torch_dtype, device, finalize
    ):
        """Read ``n`` planes in ``n_chunks`` contiguous groups, overlapping each
        group's host->device copy (on a side stream) with the next group's
        network read. Returns the same finalized tensor the single-shot path
        would produce."""
        import torch

        # Persistent double-buffered pinned staging, sized to the largest chunk.
        # Boundaries are plane-aligned so each chunk's bytes reshape cleanly.
        bounds = [(n * j) // n_chunks for j in range(n_chunks + 1)]
        max_chunk_bytes = max(
            (bounds[j + 1] - bounds[j]) for j in range(n_chunks)
        ) * pixel_len
        pair = self._pin_pair
        if pair is None or pair[0].numel() < max_chunk_bytes:
            pair = [
                torch.empty(max_chunk_bytes, dtype=torch.uint8, pin_memory=True),
                torch.empty(max_chunk_bytes, dtype=torch.uint8, pin_memory=True),
            ]
            self._pin_pair = pair
        # The copy stream is reused across ND2File instances (keyed by device) so
        # the official cold benchmark -- which opens a fresh handle per timed rep
        # -- does not pay stream creation inside the timed read.
        stream = _copy_stream_for(device)

        # Per-plane scatter reads are latency-bound: throughput scales with the
        # number of concurrent in-flight RPCs, up to ~16. Each chunk read must
        # therefore keep a high thread count or it underfills the link. (Probed:
        # 1 thread -> ~440 MB/s, 4 -> ~960, 16 -> saturated.) 0 = adaptive.
        import os as _os
        pipe_threads = int(_os.environ.get("ND2_RS_PIPELINE_THREADS", "0"))

        dev_flat = torch.empty(total, dtype=torch.uint8, device=device)
        events: list = [None, None]
        for j in range(n_chunks):
            plo, phi = bounds[j], bounds[j + 1]
            cb = (phi - plo) * pixel_len
            buf = pair[j & 1]
            # Wait for the previous copy out of this buffer before overwriting it
            # (already long done in steady state, so this rarely stalls).
            if events[j & 1] is not None:
                events[j & 1].synchronize()
            th = pipe_threads if pipe_threads > 0 else min(16, max(1, cb // (1 << 20)))
            reader.read_frames_into(
                idx[plo:phi].ctypes.data, phi - plo, _PREFIX_LEN, buf.data_ptr(), cb, th
            )
            with torch.cuda.stream(stream):
                dev_flat[plo * pixel_len : phi * pixel_len].copy_(buf[:cb], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(stream)
                events[j & 1] = ev
        # Make the default stream (where finalize runs) wait for every copy.
        torch.cuda.current_stream(device=device).wait_stream(stream)
        return finalize(dev_flat.view(torch_dtype))

    def _read_batch_odirect(
        self, reader, idx_list, n, height, width, channel_count,
        comps_per_channel, pixel_len, torch_dtype, device, finalize, n_threads,
    ):
        """Genuine-cold / fast-NVMe read of a contiguous z-stack batch via
        ``O_DIRECT``. Reads the single block-aligned byte span that covers the
        batch (planes + inter-plane chunk headers + alignment padding) directly
        off the device, overlapping each chunk's host->device copy with the next
        chunk's read, then gathers the wanted pixel bytes out of the gappy span
        with a strided view on the GPU. Returns ``None`` (caller falls back to
        the scatter path) if the planes are not a contiguous, constant-stride run
        of equal-length payloads."""
        import os as _os

        import torch

        ALIGN = 4096
        # Resolve the requested planes to (payload_offset, payload_len). Bail to
        # the scatter path on any irregularity O_DIRECT span gathering can't model.
        try:
            recs = [self._plane_record(int(i)) for i in idx_list]
        except Exception:
            return None
        offs = [int(r.get("payload_offset", -1)) for r in recs]
        plens = {int(r.get("payload_len", -1)) for r in recs}
        if len(plens) != 1 or -1 in offs:
            return None
        payload_len = plens.pop()
        if payload_len - _PREFIX_LEN != pixel_len:
            return None
        if n >= 2:
            stride = offs[1] - offs[0]
            if stride <= 0 or any(b - a != stride for a, b in zip(offs, offs[1:])):
                return None
        else:
            stride = payload_len

        first_pix = offs[0] + _PREFIX_LEN
        region_start = (first_pix // ALIGN) * ALIGN
        region_end = ((offs[-1] + payload_len + ALIGN - 1) // ALIGN) * ALIGN
        span = region_end - region_start

        # Page-locked staging spanning the whole region. Chunks write disjoint
        # slices, so a single buffer double-buffers naturally: the GPU DMAs one
        # chunk's slice while the CPU reads the next into a different slice. The
        # staging (host + device) is pooled process-wide so a fresh handle per
        # cold rep does not pay a multi-hundred-MB page-locked allocation inside
        # the timed read.
        pin = _direct_pin(span)
        dev = _direct_dev(span, device)
        base_ptr = pin.data_ptr()

        # Tuned on local NVMe: ~32 concurrent O_DIRECT readers keep the device
        # queue deep enough to saturate both a single-vdev (device-bound) file
        # and one striped across vdevs; fewer underfill the slow case, more
        # (>=40) start losing to scheduling overhead. A few large (~64 MB)
        # chunks give enough pipeline depth to hide the H2D copy behind the next
        # chunk's read without the per-chunk overhead that more chunks add.
        threads = int(_os.environ.get("ND2_RS_DIRECT_THREADS", str(max(32, n_threads))))
        chunk_mb = float(_os.environ.get("ND2_RS_DIRECT_CHUNK_MB", "64"))
        max_chunks = int(_os.environ.get("ND2_RS_DIRECT_MAX_CHUNKS", "8"))
        n_chunks = max(1, min(max_chunks, round(span / (chunk_mb * (1 << 20))) or 1))
        # Block-aligned chunk boundaries within [region_start, region_end).
        bounds = [
            region_start + (((span * j) // n_chunks) // ALIGN) * ALIGN
            for j in range(n_chunks)
        ] + [region_end]

        stream = _copy_stream_for(device)
        try:
            for j in range(n_chunks):
                clo, chi = bounds[j], bounds[j + 1]
                clen = chi - clo
                if clen <= 0:
                    continue
                doff = clo - region_start
                reader.read_span_direct(clo, clen, base_ptr + doff, clen, threads)
                with torch.cuda.stream(stream):
                    dev[doff:doff + clen].copy_(pin[doff:doff + clen], non_blocking=True)
        except (NotImplementedError, ValueError):
            return None
        torch.cuda.current_stream(device=device).wait_stream(stream)

        # Gather the pixel bytes of each plane out of the gappy span: row i starts
        # at (first_pix - region_start) + i*stride and runs pixel_len bytes.
        gbase = first_pix - region_start
        clean = torch.as_strided(dev[gbase:], size=(n, pixel_len), stride=(stride, 1)).contiguous()
        return finalize(clean.view(torch_dtype))

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
        return TextInfo(description=f"ND2 parsed by rsnd2 from {self._path.name}")

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
            "rsnd2_index": {
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
        raise NotImplementedError("OME metadata generation is not implemented in rsnd2 yet")

    def write_tiff(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("TIFF export is not implemented in rsnd2 yet")

    def write_ome_zarr(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("OME-Zarr export is not implemented in rsnd2 yet")

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
        planes = self._ensure_planes()
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
            planes = self._ensure_planes()
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
        raise ImportError("NumPy is required for array output from rsnd2") from exc
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
