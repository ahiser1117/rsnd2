# High-Performance ND2 Reader Implementation Plan

## Purpose

Build a modern, high-performance ND2 reader optimized for large Nikon NIS-Elements `.nd2` microscopy files used in analysis, batch processing, and interactive visualization.

The goal is **not** to immediately replace Bio-Formats. The goal is to build a fast ND2 access engine that:

1. Reads common modern ND2 files much faster than the current compatibility-first readers.
2. Provides batch, lazy, and random-access APIs suitable for Python scientific computing.
3. Preserves a validation path against Bio-Formats and existing ND2 libraries.
4. Can fall back to Bio-Formats or another mature reader for obscure variants.
5. Separates raw file parsing from metadata normalization and high-level OME compatibility.

---

## Background

OME Bio-Formats’ `NativeND2Reader.java` / renamed `ND2Reader` is a compatibility-focused Java reader for Nikon ND2 files. It is integrated into the Bio-Formats `FormatReader` abstraction and retrieves pixels through `openBytes`-style plane-level access. Bio-Formats documentation states that raw pixels are retrieved one plane at a time and returned as raw byte arrays through `openBytes` methods.

The source reader recognizes ND2 magic values, maintains image offset arrays, tracks JPEG/JPEG2000 and zlib compression state, handles metadata offsets, stage position offsets, timestamps, channel colors, calibration values, and multiple metadata sources. It also supports older and newer ND2 structural variants.

Bio-Formats 7.0.0 removed the old legacy ND2 reader and made the native reader the sole ND2 path, which means the current reader is an important compatibility baseline rather than merely a simple parser.

A modern implementation should therefore treat Bio-Formats as a correctness oracle during development, while using a more performance-oriented architecture for the common fast path.

---

## Primary Design Conclusion

A much faster reader is feasible, but the gain should come from changing the **access model**, not merely rewriting Java code in a faster language.

Current dominant pattern:

```text
setId(file)
openBytes(plane_0)
openBytes(plane_1)
openBytes(plane_2)
...
```

Recommended pattern:

```text
open file
build or load persistent block index
plan requested reads
sort/coalesce by file offset
issue batched positional reads
decode in parallel
return into user-requested layout
```

The modern reader should be designed around:

1. Persistent block/plane indexing.
2. Batched positional reads.
3. Parallel native decompression.
4. Lazy metadata parsing.
5. Read-many and lazy-array APIs.
6. Storage-aware scheduling for NVMe, ZFS, NFS, Lustre, GPFS, SMB, and cloud/object storage.
7. Python-first scientific interface with a fast native core.

---

## Recommended Technology Stack

### Core Language

**Recommendation: Rust**

Rationale:

- Memory-safe low-level parsing.
- High-performance binary parsing.
- Excellent Python bindings through `PyO3` and `maturin`.
- Good CPU parallelism through `rayon`.
- Good control over memory layout and buffer reuse.
- Easier long-term maintainability than C++ for a mixed scientific software team.
- Better safety profile for parsing proprietary binary files.

### Python Interface

Use:

```text
PyO3 + maturin
```

Expose a Python package such as:

```text
fastnd2
```

Primary public API:

```python
from fastnd2 import ND2File

with ND2File("sample.nd2") as f:
    arr = f.read_plane(t=0, z=0, c=0)
    stack = f.read_many(t=slice(0, 100), z=0, c=0)
    lazy = f.to_dask()
    xr = f.to_xarray()
```

### Parallelism

Use:

```text
rayon
```

for CPU-bound decompression and pixel transformation.

Use async runtimes only where they actually help:

```text
tokio / async runtime:
  useful for cloud/object/network backends

pread/readv threadpool:
  better default for local filesystems
```

### I/O

Support multiple backends:

```text
Local file:
  pread / read_exact_at / readv-style batching

Local uncompressed file:
  mmap where safe and beneficial

Network filesystem:
  offset-sorted batched pread and larger read windows

Cloud/object storage:
  async range requests with bounded concurrency and local cache
```

### Codecs

Use backend abstraction:

```text
CodecBackend
├── Raw
├── Zlib
├── JPEG2000
└── Unknown / unsupported
```

Recommended native codecs:

```text
zlib / deflate:
  libdeflate first
  zlib-ng optional
  system zlib fallback

JPEG2000:
  OpenJPEG first
  Grok optional
  Kakadu optional if licensing permits
  nvJPEG2000 optional experimental backend
```

### Python Array Integration

Support:

```text
NumPy
Dask
xarray
Zarr
OME-Zarr conversion
```

Future optional:

```text
DLPack
CuPy
PyTorch
JAX
```

DLPack/GPU support should be secondary. Most ND2 speedups will come from I/O planning and CPU/native decompression, not immediate GPU transfer.

---

## Explicit Non-Goals for the First Version

The first version should **not** try to:

1. Fully replace Bio-Formats for all historical ND2 variants.
2. Fully normalize all metadata into OME-XML.
3. Provide a GUI.
4. Implement every obscure ND2 metadata field.
5. Support writing ND2 files.
6. Guarantee efficient partial-region reads for all compression types.
7. Implement GPU JPEG2000 decode before proving CPU bottlenecks.
8. Reimplement all ImageJ/Fiji Bio-Formats behaviors.

The first version should focus on fast read access for common modern ND2 files.

---

## Architectural Overview

```text
                  ┌───────────────────────┐
                  │        ND2 file        │
                  └───────────┬───────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────┐
│  Fast parser / scanner                              │
│  - magic bytes                                      │
│  - block table                                      │
│  - image payload records                            │
│  - metadata records                                 │
│  - auxiliary stream records                         │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Immutable ND2Index                                 │
│  - logical dimensions                               │
│  - plane → offset / size / compression              │
│  - channel / z / t / position mapping               │
│  - dtype / shape / endian                           │
│  - metadata pointers                                │
│  - auxiliary stream pointers                        │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
┌─────────────┐ ┌───────────────┐ ┌───────────────┐
│ Metadata API│ │ Pixel API     │ │ Lazy array API│
│ OME/xarray  │ │ read/read_many│ │ dask/zarr     │
└─────────────┘ └───────┬───────┘ └───────┬───────┘
                        │                 │
                        ▼                 ▼
              ┌────────────────────────────────┐
              │ Read scheduler                  │
              │ - sort by file offset           │
              │ - coalesce reads                │
              │ - prefetch                      │
              │ - bounded concurrency           │
              │ - storage-aware heuristics      │
              └───────────────┬────────────────┘
                              ▼
              ┌────────────────────────────────┐
              │ Decode workers                  │
              │ - raw                           │
              │ - zlib/libdeflate               │
              │ - JPEG2000/OpenJPEG/Grok/etc.   │
              │ - endian / dtype conversion     │
              └───────────────┬────────────────┘
                              ▼
              ┌────────────────────────────────┐
              │ Output                          │
              │ - NumPy                         │
              │ - Dask                          │
              │ - xarray                        │
              │ - Zarr / OME-Zarr               │
              └────────────────────────────────┘
```

---

## Core Data Model

### `ND2Index`

The central product of parsing should be an immutable index.

```rust
pub struct ND2Index {
    pub file_path: PathBuf,
    pub file_size: u64,
    pub file_fingerprint: FileFingerprint,
    pub nd2_variant: ND2Variant,
    pub dimensions: Dimensions,
    pub series: Vec<SeriesRecord>,
    pub channels: Vec<ChannelRecord>,
    pub planes: Vec<PlaneRecord>,
    pub metadata_blocks: Vec<MetadataBlockRecord>,
    pub aux_streams: Vec<AuxStreamRecord>,
    pub warnings: Vec<ParseWarning>,
}
```

### `PlaneRecord`

Each plane must be directly addressable.

```rust
pub struct PlaneRecord {
    pub logical_index: usize,
    pub series: usize,
    pub t: usize,
    pub z: usize,
    pub c: usize,
    pub position: Option<usize>,
    pub field: Option<usize>,

    pub offset: u64,
    pub compressed_size: u64,
    pub uncompressed_size: Option<u64>,

    pub width: u32,
    pub height: u32,
    pub dtype: PixelType,
    pub endianness: Endianness,
    pub compression: CompressionKind,

    pub metadata_ref: Option<MetadataRef>,
}
```

### `MetadataBlockRecord`

```rust
pub struct MetadataBlockRecord {
    pub block_type: MetadataBlockType,
    pub offset: u64,
    pub size: u64,
    pub encoding: Option<TextEncoding>,
    pub parsed: bool,
}
```

### `AuxStreamRecord`

Examples:

```rust
pub enum AuxStreamKind {
    StageX,
    StageY,
    StageZ,
    StageP,
    Pfs,
    PfsState,
    Exposure,
    Timestamp,
    Calibration,
    Custom(String),
}
```

---

## ND2 Variants to Support

The reader should explicitly classify the file instead of letting variant-specific logic leak across the codebase.

```rust
pub enum ND2Variant {
    ModernBlockScanned,
    LegacyJp2Like,
    Unknown,
}
```

### Modern Block-Scanned ND2

Fast path.

Known indicators from Bio-Formats source:

```text
ND2_MAGIC_BYTES_1 = 0xDACEBE0A
```

Expected block names / conceptual payloads to support first:

```text
ImageDataSeq
ImageText
ImageAttribu
CustomData|X
CustomData|Y
CustomData|Z
CustomData|P
metadata LV blocks
calibration data
PFS data
timestamp data
```

Tasks:

- Scan block headers.
- Identify image data payload blocks.
- Extract image offsets and sizes.
- Extract dimension metadata.
- Extract timestamps.
- Extract channel names and colors where possible.
- Extract physical pixel sizes and calibration where possible.
- Build `ND2Index`.

### Legacy JP2-Like ND2

Compatibility path.

Known indicator from Bio-Formats source:

```text
ND2_MAGIC_BYTES_2 = 0x6A502020
```

Tasks:

- Parse JP2-like box structure.
- Identify embedded JPEG2000 payloads.
- Extract trailing XML/text metadata where present.
- Build `ND2Index` where possible.
- If unsupported, return a structured error with fallback recommendation.

---

## Reader Layers

### Layer 1: Raw Binary Parser

Responsibilities:

- Open file.
- Identify ND2 variant.
- Scan block/box structure.
- Record offsets and sizes.
- Do minimal validation.
- Never decode pixels.
- Never normalize to OME metadata.
- Avoid large allocations.

Output:

```text
RawBlockTable
```

### Layer 2: Index Builder

Responsibilities:

- Convert raw block table into `ND2Index`.
- Map image blocks to logical dimensions.
- Resolve channels, Z, T, position, and field indices.
- Attach metadata pointers to plane records.
- Detect malformed or ambiguous structures.
- Emit warnings, not silent failures.

Output:

```text
ND2Index
```

### Layer 3: Pixel Access Engine

Responsibilities:

- Accept logical read requests.
- Resolve planes through `ND2Index`.
- Plan I/O.
- Batch and coalesce reads.
- Decode payloads.
- Return arrays.

Output:

```text
NumPy arrays / Rust buffers / lazy array chunks
```

### Layer 4: Metadata API

Responsibilities:

- Parse only requested metadata blocks.
- Provide structured metadata.
- Provide xarray coordinates.
- Provide optional OME metadata conversion.

Output:

```text
ND2Metadata
OME-compatible metadata
xarray attrs / coords
```

### Layer 5: Compatibility Adapters

Responsibilities:

- Offer Bio-Formats-like access where useful.
- Support validation tests against Bio-Formats.
- Provide fallback hooks.

---

## API Specification

### Python API

```python
from fastnd2 import ND2File

with ND2File("sample.nd2") as f:
    print(f.shape)
    print(f.sizes)
    print(f.dtype)
    print(f.channels)
    print(f.metadata())

    plane = f.read_plane(t=0, z=0, c=0)
    stack = f.read_many(t=slice(0, 100), z=0, c=0)

    lazy = f.to_dask(chunks={"t": 16, "z": 1, "c": 1})
    xr = f.to_xarray(chunks={"t": 16})
```

### Required Methods

```python
ND2File(path, index_cache=True, validate=False)

ND2File.index
ND2File.shape
ND2File.sizes
ND2File.dtype
ND2File.channels
ND2File.positions
ND2File.metadata(lazy=True)
ND2File.timestamps()
ND2File.stage_positions()
ND2File.read_plane(t=0, z=0, c=0, position=0, out=None)
ND2File.read_many(t=None, z=None, c=None, position=None, order="logical", out=None)
ND2File.to_dask(chunks=None)
ND2File.to_xarray(chunks=None)
ND2File.to_zarr(path, chunks=None, ome=True)
ND2File.close()
```

### Read Request Semantics

Support logical indexing:

```python
f.read_plane(t=10, z=2, c=1)
```

Support slice indexing:

```python
f.read_many(t=slice(0, 100), z=0, c=[0, 1])
```

Support explicit plane IDs:

```python
f.read_planes([0, 10, 20, 30], order="file")
```

Support output buffer reuse:

```python
out = np.empty((100, y, x), dtype=np.uint16)
f.read_many(t=slice(0, 100), z=0, c=0, out=out)
```

---

## Read Scheduler

### Goals

The scheduler should optimize file access and decompression for the requested workload.

### Pipeline

```text
logical read request
→ resolve plane records
→ group by compression type
→ sort by file offset
→ coalesce nearby reads
→ issue bounded concurrent reads
→ decode in worker pool
→ write into output layout
→ return in requested logical order
```

### Storage Profiles

Implement heuristics through a configurable profile.

```rust
pub enum StorageProfile {
    Auto,
    LocalNvme,
    LocalSsd,
    Zfs,
    Nfs,
    Smb,
    Lustre,
    Gpfs,
    CloudObject,
}
```

Default profile:

```text
Auto
```

Auto-detection should be conservative. Prefer simple, stable heuristics and allow user override.

### Example Heuristics

#### Local NVMe

```text
- Many outstanding reads.
- Medium coalescing window.
- Decode parallelism near physical CPU cores.
- Avoid over-prefetching if file is memory-mapped.
```

#### ZFS

```text
- Prefer larger sequential windows.
- Avoid excessive tiny random reads.
- Allow read-ahead and ARC to work.
- Do not oversubscribe I/O queue with hundreds of tiny requests.
```

#### NFS / SMB

```text
- Aggressively coalesce nearby reads.
- Use fewer concurrent reads.
- Prefer offset-sorted batch access.
- Increase cache size.
```

#### Lustre / GPFS

```text
- Allow higher concurrency.
- Consider stripe size and stripe count if detectable.
- Prefer large aligned reads.
```

#### Cloud/Object Storage

```text
- Use HTTP range requests.
- Use bounded async concurrency.
- Maintain local chunk cache.
- Avoid repeated metadata requests.
```

---

## Compression Strategy

### Raw / Uncompressed

Fast path:

- Use direct file read into output buffer where possible.
- Avoid intermediate allocation.
- Only perform endian conversion if needed.
- Only copy if output layout differs from on-disk layout.

### zlib / deflate

Use native backends:

1. `libdeflate`
2. `zlib-ng`
3. system zlib fallback

Implementation requirements:

- Reuse decompression buffers.
- Decode many planes in parallel.
- Record compressed and uncompressed byte counts.
- Validate decompressed size.
- Surface codec errors with file offset and plane identity.

### JPEG2000

Use backend abstraction.

Recommended order:

1. OpenJPEG backend.
2. Grok backend as optional feature.
3. Kakadu backend as optional proprietary feature.
4. nvJPEG2000 as experimental GPU feature.

Implementation requirements:

- Decode planes independently in parallel.
- Extract dtype/shape consistently.
- Validate lossless/lossy state where possible.
- Compare output against Bio-Formats and/or `nd2` Python library on test files.
- Add benchmark cases specifically for JPEG2000-heavy files.

---

## Metadata Strategy

### Principle

Do not parse all metadata eagerly.

### Required Metadata for MVP

Parse enough to support:

```text
- image dimensions
- dtype
- channel count
- timepoint count
- z count
- position count
- pixel physical size
- timestamps
- channel names
- channel colors
- stage positions when present
- exposure times when present
```

### Lazy Metadata

Large XML, LV, or custom metadata blocks should be parsed on demand.

```python
f.metadata(lazy=True)
f.metadata(raw=True)
f.metadata(block="ImageText")
```

### Metadata Normalization

Use a staged approach:

```text
Stage 1:
  expose raw structured metadata

Stage 2:
  map core metadata to xarray attrs and coords

Stage 3:
  optional OME metadata adapter

Stage 4:
  OME-Zarr export
```

Do not block fast pixel reads on full OME metadata normalization.

---

## Persistent Index Cache

### Motivation

Large files should not be rescanned every time they are opened.

### Sidecar Options

Preferred initial format:

```text
sample.nd2.fastnd2-index.json
```

Later optimized formats:

```text
sample.nd2.fastnd2-index.msgpack
sample.nd2.fastnd2-index.arrow
sample.nd2.fastnd2-index.sqlite
```

### Cache Validation

The sidecar must include:

```text
- absolute or canonical file path
- file size
- modified timestamp
- fast hash of selected file regions
- reader version
- index schema version
```

Do not require hashing the entire file by default.

Suggested fingerprint:

```text
file size
mtime
first 4 KiB hash
last 4 KiB hash
selected block-table hash if available
```

### Cache Behavior

```python
ND2File(path, index_cache=True)
ND2File(path, index_cache="read_only")
ND2File(path, index_cache=False)
ND2File(path, rebuild_index=True)
```

---

## Validation Strategy

### Correctness Oracles

Use multiple validation sources:

1. Bio-Formats / current ND2Reader.
2. Existing Python `nd2` library where applicable.
3. Nikon SDK-derived outputs if available.
4. Manually curated expected metadata for lab files.
5. Synthetic or minimally structured ND2-like fixtures where legally and technically possible.

### Pixel Validation

For each test file:

```text
- compare selected planes
- compare all planes for small files
- compare shape
- compare dtype
- compare min/max
- compare checksum
- compare exact bytes for lossless/raw files
- compare tolerance-based values for lossy JPEG2000 files
```

### Metadata Validation

Compare:

```text
- SizeX, SizeY, SizeZ, SizeC, SizeT
- physical pixel size
- channel names
- channel colors
- timestamps
- exposure times
- stage X/Y/Z/P
- position count
- series count
```

### Fuzz and Robustness Testing

Add parser robustness tests:

```text
- truncated file
- corrupted block size
- invalid offset
- bad compression payload
- missing metadata block
- inconsistent plane count
- unsupported variant
```

The parser must fail with structured errors, not panic.

### Regression Corpus

Maintain test files by category:

```text
tests/data/
├── modern_uncompressed/
├── modern_zlib/
├── modern_jpeg2000/
├── multi_channel/
├── multi_z/
├── multi_t/
├── multi_position/
├── large_metadata/
├── legacy_jp2_like/
├── malformed/
└── unsupported/
```

If files cannot be committed for size/licensing reasons, store a manifest with checksums and retrieval instructions.

---

## Benchmark Strategy

### Benchmarks to Implement

1. Open/index time.
2. Reopen with cached index.
3. Single random plane read.
4. 100 random plane reads.
5. Sequential full time series read.
6. Multi-channel stack read.
7. Full-file conversion to Zarr.
8. JPEG2000 decode throughput.
9. zlib decode throughput.
10. Metadata-only read.

### Comparison Targets

Compare against:

```text
Bio-Formats Java reader
Python nd2 library
Nikon SDK if available
Current lab workflow
```

### Metrics

Collect:

```text
- wall-clock time
- CPU time
- peak RSS
- read throughput MB/s
- decompressed throughput MB/s
- planes/s
- I/O wait
- number of syscalls where possible
- cache hit/miss behavior
```

### Benchmark Matrix

Run on:

```text
- local NVMe
- ZFS pool
- NFS mount
- HPC parallel filesystem if relevant
- optionally cloud/object storage
```

### Benchmark Harness

Provide a command:

```bash
fastnd2-bench sample.nd2 --compare bioformats --compare nd2 --profile zfs
```

Output:

```text
JSON summary
Markdown report
optional flamegraph/profile artifacts
```

---

## Implementation Phases

## Phase 0 — Repository and Tooling

### Goal

Create the project skeleton.

### Tasks

- Create Rust workspace.
- Create Python package.
- Add `maturin` build.
- Add CI for Linux.
- Add formatting/linting.
- Add basic docs.
- Add issue templates for unsupported files.
- Add benchmark harness skeleton.

### Suggested Layout

```text
fastnd2/
├── Cargo.toml
├── crates/
│   ├── fastnd2-core/
│   ├── fastnd2-codecs/
│   ├── fastnd2-python/
│   └── fastnd2-cli/
├── python/
│   └── fastnd2/
├── tests/
│   ├── data_manifest/
│   └── fixtures/
├── benches/
├── docs/
└── README.md
```

### Acceptance Criteria

- `pip install -e .` works through maturin.
- `import fastnd2` works.
- `cargo test` passes.
- CI runs formatting and tests.

---

## Phase 1 — File Identification and Block Scanner

### Goal

Detect ND2 files and scan top-level structure.

### Tasks

- Implement magic byte detection.
- Classify modern block-scanned vs legacy JP2-like vs unknown.
- Implement sequential scanner for modern blocks.
- Record block name, offset, size, and payload offset.
- Add structured warnings.
- Add parser error types.

### Acceptance Criteria

- Can open representative modern ND2 files.
- Produces a block table.
- Does not decode pixels.
- Does not parse all metadata.
- Handles truncated files gracefully.

---

## Phase 2 — Index Builder

### Goal

Build `ND2Index` with plane records.

### Tasks

- Identify image data blocks.
- Extract image offsets.
- Extract compressed sizes.
- Infer logical dimensions.
- Attach channel/Z/T/position indices.
- Extract dtype and dimensions.
- Store metadata block references.
- Serialize index to JSON sidecar.
- Validate sidecar freshness.

### Acceptance Criteria

- `ND2File(path).index` returns valid plane records.
- Plane count matches expected dimensions.
- Cached reopen is faster than cold open.
- Index JSON round-trips.

---

## Phase 3 — Raw and zlib Pixel Reads

### Goal

Read uncompressed and zlib-compressed planes.

### Tasks

- Implement `read_plane`.
- Implement `read_many`.
- Implement direct positional reads.
- Implement output-buffer reuse.
- Implement raw decode path.
- Implement zlib/libdeflate decode path.
- Add endian conversion if necessary.
- Add NumPy output via Python bindings.

### Acceptance Criteria

- Reads raw files correctly.
- Reads zlib files correctly.
- Matches Bio-Formats or Python `nd2` on selected test planes.
- `read_many` is faster than repeated `read_plane` for contiguous requests.

---

## Phase 4 — Read Scheduler

### Goal

Optimize batch reading.

### Tasks

- Resolve logical requests to plane records.
- Sort by file offset.
- Coalesce adjacent reads.
- Bound concurrency.
- Decode in Rayon worker pool.
- Reorder output to logical order.
- Add storage profile settings.
- Add prefetching.

### Acceptance Criteria

- Batch random reads outperform naive plane-by-plane access.
- Sequential stack reads avoid excessive seeks.
- Scheduler does not oversubscribe CPU by default.
- User can override thread count and storage profile.

---

## Phase 5 — JPEG2000 Support

### Goal

Support JPEG2000-compressed ND2 files.

### Tasks

- Add OpenJPEG backend.
- Add feature-gated optional Grok backend.
- Implement JPEG2000 plane decode.
- Validate lossy vs lossless behavior.
- Add JPEG2000 benchmark.
- Add meaningful error messages for unsupported codestreams.

### Acceptance Criteria

- JPEG2000 test files decode correctly.
- Parallel JPEG2000 decode scales across cores.
- Error messages include plane ID, offset, and codec backend.

---

## Phase 6 — Metadata API

### Goal

Expose essential metadata without slowing hot-path pixel reads.

### Tasks

- Parse core dimensions.
- Parse channel names.
- Parse timestamps.
- Parse pixel sizes.
- Parse exposure times.
- Parse stage positions.
- Parse colors.
- Expose raw metadata blocks.
- Add lazy metadata loading.
- Add xarray coordinate generation.

### Acceptance Criteria

- `f.metadata()` returns structured metadata.
- `f.timestamps()` works.
- `f.stage_positions()` works when present.
- Pixel reads do not eagerly parse large metadata blocks.
- Metadata agrees with validation sources for test corpus.

---

## Phase 7 — Dask, xarray, and Zarr

### Goal

Integrate with Python analysis workflows.

### Tasks

- Implement `to_dask`.
- Implement `to_xarray`.
- Implement chunked read adapter.
- Implement `to_zarr`.
- Implement basic OME-Zarr metadata writing.
- Expose chunk-size recommendations.

### Acceptance Criteria

- `to_dask()` supports lazy reads.
- `to_xarray()` includes dimensions and coordinates.
- `to_zarr()` writes a valid readable Zarr store.
- Dask access does not open excessive file handles.

---

## Phase 8 — CLI and Conversion Tools

### Goal

Provide useful command-line workflows.

### Commands

```bash
fastnd2 info sample.nd2
fastnd2 index sample.nd2
fastnd2 validate sample.nd2 --against bioformats
fastnd2 bench sample.nd2
fastnd2 convert sample.nd2 output.zarr
```

### Acceptance Criteria

- CLI tools work without Python scripting.
- `info` prints dimensions, compression, channels, timestamps, and warnings.
- `validate` compares against available backends.
- `convert` can write chunked Zarr.

---

## Phase 9 — Compatibility and Fallback Layer

### Goal

Handle unsupported files gracefully.

### Tasks

- Detect unsupported variants early.
- Provide structured errors.
- Add fallback adapter to Bio-Formats where installed.
- Add fallback adapter to Python `nd2` where installed.
- Add option to force native-only mode.

### Acceptance Criteria

- Unsupported files fail with actionable messages.
- User can choose fallback behavior.
- Fallback path is clearly marked in metadata/result object.

---

## Error Handling

Use structured error types.

```rust
pub enum ND2Error {
    Io { path: PathBuf, source: io::Error },
    UnsupportedVariant { magic: [u8; 8] },
    CorruptBlock { offset: u64, reason: String },
    InvalidPlaneRecord { plane: usize, reason: String },
    CodecError { plane: usize, offset: u64, codec: CompressionKind, reason: String },
    MetadataError { block: String, offset: u64, reason: String },
    IndexCacheStale,
}
```

Python exceptions should preserve the same information.

Example:

```python
try:
    f.read_plane(t=100, z=0, c=0)
except fastnd2.CodecError as e:
    print(e.plane)
    print(e.offset)
    print(e.codec)
```

---

## Performance Guidelines for Coding Agent

### Do

- Minimize allocations in hot paths.
- Reuse buffers.
- Separate parsing from decoding.
- Keep metadata parsing lazy.
- Use positional reads.
- Batch reads where possible.
- Sort random plane reads by file offset.
- Use bounded concurrency.
- Benchmark every optimization.
- Preserve exact error context.
- Validate against known readers.

### Do Not

- Parse full metadata during every pixel read.
- Read planes through one synchronous API internally when a batch is requested.
- Spawn unbounded tasks.
- Depend on global mutable state.
- Silently ignore corrupt offsets.
- Panic on malformed files.
- Convert to OME metadata before returning pixels.
- Assume every ND2 file uses the same block structure.
- Assume compressed payloads can be partially decoded efficiently.
- Assume async I/O improves local NVMe performance.

---

## Hardware-Aware Defaults

### Default Threading

```text
decode_threads = min(physical_cores, 16) initially
io_concurrency = storage-profile dependent
```

Allow environment overrides:

```bash
FASTND2_DECODE_THREADS=8
FASTND2_IO_CONCURRENCY=4
FASTND2_STORAGE_PROFILE=zfs
FASTND2_INDEX_CACHE=1
```

### ZFS / Lab Server Recommendation

For a ZFS pool with many large microscopy files:

```text
- prefer batched offset-sorted reads
- avoid many tiny random reads
- use larger coalescing windows
- keep decode threads below total cores if downstream analysis is also threaded
- expose conversion to Zarr for repeated analysis
```

### NFS Recommendation

```text
- increase coalescing
- reduce concurrency
- enable local index cache
- optionally enable local compressed payload cache for repeated interactive browsing
```

### NVMe Recommendation

```text
- higher I/O concurrency
- lower coalescing threshold
- focus on decode throughput
```

---

## Suggested Configuration Object

```python
config = fastnd2.ReaderConfig(
    index_cache=True,
    storage_profile="auto",
    decode_threads=None,
    io_concurrency=None,
    codec_backend={
        "zlib": "libdeflate",
        "jpeg2000": "openjpeg",
    },
    metadata_policy="lazy",
    validate_index=True,
)
```

---

## Minimum Impressive Demo

The first impressive demo should show:

1. Opening a large ND2 file.
2. Building and caching the index.
3. Reading a time-series stack through `read_many`.
4. Showing speedup over repeated plane reads.
5. Loading the same data lazily with Dask/xarray.
6. Converting to Zarr.
7. Validating selected planes against Bio-Formats or Python `nd2`.

Demo script:

```python
from fastnd2 import ND2File
import time

path = "large_timelapse.nd2"

t0 = time.perf_counter()
with ND2File(path, index_cache=True) as f:
    print(f.sizes)
    print(f.channels)

    stack = f.read_many(t=slice(0, 100), z=0, c=0)
    print(stack.shape, stack.dtype)

    xr = f.to_xarray(chunks={"t": 16})
    print(xr)

    f.to_zarr("large_timelapse.zarr", chunks={"t": 16, "z": 1, "c": 1})

print("elapsed", time.perf_counter() - t0)
```

---

## Acceptance Criteria for MVP

The MVP is complete when:

1. It installs as a Python package on Linux.
2. It can open common modern ND2 files.
3. It builds a persistent sidecar index.
4. It reads raw and zlib-compressed planes.
5. It supports `read_plane` and `read_many`.
6. It exposes NumPy arrays.
7. It exposes Dask lazy arrays.
8. It parses core dimensions and timestamps.
9. It validates selected planes against Bio-Formats or Python `nd2`.
10. It includes benchmarks showing improvement for at least one realistic workload.

JPEG2000 support can be MVP+ if codec integration slows development, but the architecture must reserve the codec abstraction from the beginning.

---

## Stretch Goals

1. JPEG2000 GPU decode.
2. Direct DLPack output.
3. JAX/PyTorch integration.
4. Interactive thumbnail pyramid cache.
5. OME-Zarr multiscale export.
6. Cloud object storage backend.
7. WebAssembly parser for browser-side metadata inspection.
8. Bio-Formats-compatible Java wrapper.
9. Automatic storage-profile tuning.
10. File-layout visualizer.

---

## Major Risks

### Proprietary Format Variability

ND2 has multiple historical variants and undocumented edge cases.

Mitigation:

- Start with common modern files.
- Keep fallback path.
- Maintain a growing corpus.
- Return structured unsupported errors.

### JPEG2000 Complexity

JPEG2000 may dominate performance and compatibility.

Mitigation:

- Abstract codec backends.
- Start with OpenJPEG.
- Add Grok/Kakadu as optional features.
- Benchmark decode independently.

### Metadata Complexity

Full metadata normalization can consume a large amount of engineering time.

Mitigation:

- Keep raw metadata access.
- Parse core metadata first.
- Implement OME mapping later.

### Async Overengineering

Async I/O may not improve local performance.

Mitigation:

- Use staged scheduler.
- Use pread threadpool for local files.
- Use async only for network/cloud.

### Validation Difficulty

No single oracle is perfect.

Mitigation:

- Compare against multiple readers.
- Validate core metadata and pixels separately.
- Keep test files from real acquisition systems.

---

## Suggested First Coding-Agent Tasks

Assign the coding agent the following initial tasks in order.

### Task 1: Create Project Skeleton

Build a Rust + Python package using `maturin`.

Deliverables:

```text
- Rust workspace
- Python module
- CI
- basic ND2File class
- import test
```

### Task 2: Implement Magic Detection

Implement:

```rust
detect_nd2_variant(path: &Path) -> Result<ND2Variant, ND2Error>
```

Recognize:

```text
0xDACEBE0A
0x6A502020
```

Deliverables:

```text
- tests with synthetic byte fixtures
- structured unsupported error
```

### Task 3: Implement Modern Block Scanner

Implement a scanner that records candidate blocks:

```rust
scan_blocks(path: &Path) -> Result<Vec<BlockRecord>, ND2Error>
```

Deliverables:

```text
- block offsets
- block sizes
- block names where decoded
- warnings for malformed blocks
```

### Task 4: Implement Index Serialization

Implement:

```rust
ND2Index::to_json()
ND2Index::from_json()
ND2Index::is_fresh_for(path)
```

Deliverables:

```text
- JSON schema version
- file fingerprint
- round-trip tests
```

### Task 5: Implement Plane Read Prototype

Implement reading for one uncompressed plane from a known `PlaneRecord`.

Deliverables:

```text
- Rust unit test
- Python `read_plane`
- NumPy output
```

### Task 6: Add Batch Read Planning

Implement:

```rust
plan_reads(plane_records, storage_profile) -> ReadPlan
```

Deliverables:

```text
- sort by offset
- coalesce nearby reads
- preserve logical output order
- unit tests
```

### Task 7: Add zlib Decode

Implement zlib/libdeflate decompression behind codec trait.

Deliverables:

```text
- codec trait
- zlib backend
- decompressed-size validation
- Python tests
```

### Task 8: Add Benchmark Harness

Implement:

```bash
fastnd2 bench sample.nd2
```

Deliverables:

```text
- open/index timing
- single-plane timing
- read-many timing
- JSON output
```

---

## Coding Standards

### Rust

- Use `thiserror` for errors.
- Use `tracing` for debug/instrumentation.
- Use `rayon` for CPU parallelism.
- Avoid unsafe unless justified and isolated.
- Prefer explicit endian parsing.
- Keep parser allocation-light.
- Use property tests for binary parsing where useful.

### Python

- Type annotate public API.
- Provide docstrings.
- Use NumPy typing where practical.
- Avoid hiding fallback behavior.
- Keep Dask/xarray optional dependencies.
- Keep core import lightweight.

### Documentation

Required docs:

```text
README.md
docs/design.md
docs/format_notes.md
docs/performance.md
docs/validation.md
docs/fallbacks.md
```

---

## Documentation for Users

README should state clearly:

```text
fastnd2 is a high-performance ND2 reader optimized for modern Nikon ND2 files.
It is not affiliated with Nikon.
It does not initially guarantee support for every historical ND2 variant.
For unsupported files, use Bio-Formats or Python nd2 fallback.
```

---

## References for Agent

- OME Bio-Formats `NativeND2Reader.java` source.
- Bio-Formats developer documentation: raw pixels are read plane-by-plane through `openBytes`.
- Bio-Formats 7.0.0 release notes: legacy ND2 reader removed; native reader became the sole ND2 reader.
- Existing Python `nd2` package: useful comparison target; supports dask/xarray workflows.
- OME-Zarr specification for later export support.

---

## Final Recommendation

Build this as a **Rust-core, Python-first, index-driven ND2 access engine**.

Do not start by reproducing every Bio-Formats metadata behavior. Start by making common modern ND2 files fast:

```text
index once
read by offset
batch requests
decode in parallel
return NumPy/Dask/xarray
cache for repeated use
validate against Bio-Formats
fallback for obscure files
```

This design gives the largest chance of meaningful speedups while keeping compatibility risk controlled.
