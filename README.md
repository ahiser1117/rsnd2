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
package on the **disk→GPU batch-streaming** path: the end-to-end wall time to
stream a batch of z-stacks from an `.nd2` file into one contiguous `torch.cuda`
tensor of shape `(batch × z_per_frame, C, Y, X)`.

### What is measured

- A **z-slice** is one plane returned by `read_frame(i)` (what PyPI `nd2` calls a
  "frame"). A **z-stack** is `z_per_frame` contiguous z-slices (default 80), so a
  batch of *B* z-stacks reads *B × 80* contiguous z-slices.
- **Warm:** the file is resident in cache — the steady-state regime for an ML
  data loader on a large-RAM host.
- **Cold:** the client page cache is dropped (`posix_fadvise(DONTNEED)`) before
  every timed read, so the read comes from the backing store.
- Time is end-to-end and includes the host→device (PCIe) transfer and the layout
  transform, not just the disk read.

### Assumptions

- **Fair end-to-end path.** `rsnd2` uses `read_batch_to_torch` (one packed Rust
  read of the whole batch, then a single pinned host→device copy, with the copy
  pipelined behind the read); PyPI `nd2` uses `np.stack([read_frame(i) …])`
  followed by one host→device copy. Both produce a byte-identical CUDA tensor.
- **File format = the fast path:** modern chunked ND2, uncompressed, with packed
  (unpadded) rows. Compressed or legacy-JPEG2000 files take a slower fallback and
  are not represented by these numbers.
- **One client, one GPU** (RTX 6000 Ada), GPU otherwise idle; both readers run in
  separate subprocesses (both distributions import as `nd2`).
- **Test data:** 3× real ND2 files, 4800 z-slices each (60 z-stacks of 80),
  `80×2×212×322` uint16 per z-stack (~21.8 MB). Reported values are the **mean
  across the 3 files**.
- Cold throughput is **storage-bandwidth-bound**, so the speedup over PyPI `nd2`
  depends on the medium (see the two cases below). Warm throughput is bound by
  host memory / PCIe and by PyPI's per-frame Python overhead.

The benchmark harness and full write-up live in the sibling `rsnd2-testing/`
workspace (`gpu_stream_bench.py`, `results/RESULTS.md`).

### Networked storage (NFS, single 10 GbE link)

Test host on a single 10 GbE NFS4 mount (no `nconnect`, `rsize=1 MB`).

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

Warm is ~19× faster — largely fixed-overhead amortization: PyPI `nd2` reads each
z-slice individually, while `rsnd2` packs the whole z-stack batch in one Rust
call. Cold reads saturate the single-client NFS link (~1.1 GB/s on 10 GbE), so
both readers hit the same network ceiling and the ratio is ~3.7–4.7×. The cold
streaming path was further tuned (the batch read is split into a few large,
plane-aligned chunks whose host→device copies overlap the next chunk's read, and
the reader is opened during metadata parse), recovering ~3–4% toward the network
ceiling — see `rsnd2-testing/results/RESULTS.md`.

![NFS disk→GPU z-stack streaming: rsnd2 vs PyPI nd2](docs/zstack_streaming.png)

### Local storage (NVMe, ZFS)

Same files copied to a local NVMe-backed ZFS dataset.

| cache | batch (z-stacks) | z-slices | PyPI `nd2` (ms) | `rsnd2` (ms) | speedup | `rsnd2` GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| warm | 1 | 80 | 29.5 | 1.83 | 16× | 11.9 |
| warm | 2 | 160 | 59.3 | 2.96 | 20× | 14.8 |
| warm | 4 | 320 | 115.3 | 5.21 | 22× | 16.8 |
| warm | 8 | 640 | 226.1 | 9.77 | **23×** | 17.9 |
| warm | 16 | 1280 | 453.8 | 19.38 | 23× | 18.0 |
| cold | 1 | 80 | 71.5 | 2.03 | 35× | 10.7 |
| cold | 2 | 160 | 140.9 | 3.18 | 44× | 13.7 |
| cold | 4 | 320 | 276.4 | 5.36 | 52× | 16.3 |
| cold | 8 | 640 | 544.8 | 9.86 | **55×** | 17.7 |
| cold | 16 | 1280 | 1090.5 | 19.58 | 56× | 17.9 |

On local storage `rsnd2` sustains **~18 GB/s** and is **16–23× faster warm** and
**35–56× faster cold** than PyPI `nd2`. Two local-specific notes:

- **ZFS ARC cannot be evicted via `posix_fadvise(DONTNEED)`** (it needs root
  `drop_caches`), so "cold" here is *page-cache-dropped but ARC-resident*
  (RAM-served), not a genuine first-touch from NVMe. This is the realistic local
  steady state once data is hot in ARC.
- The much larger **cold** speedup is because PyPI `nd2`'s mmap-based per-frame
  reads re-fault every page when the page cache is dropped (1.09 s at batch 16),
  whereas `rsnd2`'s `pread`-based reader serves straight from ARC.
- At this speed the bottleneck has moved off storage onto the **PCIe host→device
  copy (~26 GB/s)**: for large batches the H2D copy, not the read, is the long
  pole — analogous to the 10 GbE ceiling in the NFS case.

![Local NVMe/ZFS disk→GPU z-stack streaming: rsnd2 vs PyPI nd2](docs/local_nvme_streaming.png)

### Correctness

Pixel data is byte-for-byte identical to PyPI `nd2`: blake2b hashes matched for
all batch/file/cache cases in both runs above. The batch layout mirrors
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
