# rsnd2

A fast ND2 (Nikon NIS-Elements) file reader with a Rust core and Python bindings.

`rsnd2` parses ND2 files in Rust and exposes them through a Python API that is
compatible with the popular [`nd2`](https://pypi.org/project/nd2/) package, so
existing code can switch readers with minimal changes. The package is imported
as `rsnd2`

## Project layout

| Path | Description |
| --- | --- |
| `src/` | Rust core. `src/lib.rs` exposes the parser, index, and FFI helpers; `src/main.rs` provides the CLI. |
| `src-python/rsnd2/` | Python package. `_ffi.py` loads the native library, `_nd2file.py` provides `ND2File`, `structures.py` holds API-compatible data classes. |
| `tests/` | Python (`unittest`) and Rust tests. |
| `Cargo.toml`, `pyproject.toml`, `setup.py` | Packaging metadata. |

## Python usage

```python
import rsnd2

with rsnd2.ND2File("path/to/file.nd2") as f:
    print(f.shape, f.dtype)
    arr = f.asarray()        # requires the [array] extra

# Or read straight into an array
arr = rsnd2.imread("path/to/file.nd2")
```

## Comparison with the PyPI `nd2` package

`rsnd2` is benchmarked against the upstream [`nd2`](https://pypi.org/project/nd2/)
package on the disk→GPU batch-streaming path: the end-to-end time to stream a
batch of **z-stacks** into a contiguous CUDA tensor. A single plane returned by
`read_frame(i)` is one z-slice (what PyPI `nd2` calls a "frame"); the acquisition
unit is a z-stack of `z_per_frame` z-slices (default 80), so a batch of *B*
reads *B × 80* contiguous z-slices. Test data is 3× real ND2 files (4800
z-slices each = 60 z-stacks of 80; `80×2×212×322` uint16 per z-stack, on NFS)
read on an RTX 6000 Ada. The harness runs both readers in separate subprocesses
(both packages import as `nd2`). The full workspace lives alongside this repo in
`../nd2-pypi-comparison`; see `../nd2-pypi-comparison/results/RESULTS.md` for the
complete write-up.

### Headline (batch = 8 z-stacks = 640 z-slices)

- **Warm / sustained streaming: 19.0× faster** (226.1 ms → 11.88 ms) — the
  steady-state regime for an ML data loader on a large-RAM host. The warm
  speedup grows with batch size and plateaus around 19×.
- **Cold / first-touch from NFS: 3.9× faster** (667.0 ms → 169.9 ms), at the NFS
  single-client bandwidth ceiling (~1 GB/s). The cold ratio holds ~3.7–4.7×
  across batch sizes and varies run-to-run with PyPI's single-stream speed.

### All batch sizes

| cache | batch (z-stacks) | z-slices | PyPI `nd2` (ms) | `rsnd2` (ms) | speedup |
| --- | --- | --- | --- | --- | --- |
| warm | 1 | 80 | 27.23 | 1.739 | 15.7× |
| warm | 2 | 160 | 54.42 | 3.052 | 17.8× |
| warm | 4 | 320 | 109.82 | 5.884 | 18.7× |
| warm | 8 | 640 | 226.10 | 11.883 | **19.0×** |
| warm | 16 | 1280 | 455.40 | 23.694 | 19.2× |
| cold | 1 | 80 | 104.17 | 22.392 | 4.7× |
| cold | 2 | 160 | 161.33 | 42.927 | 3.8× |
| cold | 4 | 320 | 318.85 | 85.338 | 3.7× |
| cold | 8 | 640 | 666.95 | 169.855 | **3.9×** |
| cold | 16 | 1280 | 1315.25 | 336.752 | 3.9× |

`rsnd2` values are means across the final-build (Opt C) measurement runs over the
3 files. The warm advantage is largely fixed-overhead amortization: PyPI `nd2`
reads each z-slice individually, while `rsnd2` packs the whole z-stack batch in
one Rust call. Cold reads are bounded by single-client NFS bandwidth.

![z-stack disk→GPU streaming: rsnd2 vs PyPI nd2](../nd2-pypi-comparison/results/zstack_streaming.png)

### Correctness

Pixel data is byte-for-byte identical to PyPI `nd2`: blake2b hashes matched for
all 30/30 batch/file/cache cases in the run above. The batch layout mirrors
`read_frame` exactly, verified for single-channel, multi-channel, and RGB.

## Command-line interface

Build and run the Rust CLI with `cargo`:

```bash
cargo build                       # debug build of the library and CLI
cargo run -- inspect FILE.nd2     # inspect one or more ND2 files
cargo run -- scan ROOT            # scan directories for ND2 files
cargo run -- bench-read FILE.nd2  # benchmark pixel reads
```

## Building the native library

```bash
cargo build --release --lib   # builds the native lib copied into the Python package
```

## Testing

```bash
cargo test                                      # Rust tests
uv run python -m unittest discover -s tests     # Python API tests
```

Run both before submitting a change.

## Contributing

See [AGENTS.md](AGENTS.md) for coding style, testing, and pull-request
guidelines. Keep binary parsing logic in Rust, run `cargo fmt` before
submitting Rust changes.
