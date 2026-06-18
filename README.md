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
