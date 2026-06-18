# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Rust ND2 reader core with Python bindings. Rust source lives in `src/`: `src/lib.rs` exposes parser, index, and FFI helpers, while `src/main.rs` provides CLI commands such as `inspect`, `scan`, and `bench-read`. Python package code lives in `src-python/nd2/`; `_ffi.py` loads the native library, `_nd2file.py` provides `ND2File`, and `structures.py` contains API-compatible data classes. Tests are in `tests/`. Packaging metadata is in `Cargo.toml`, `pyproject.toml`, and `setup.py`.

## Build, Test, and Development Commands

- `cargo build` builds the Rust library and CLI in debug mode.
- `cargo build --release --lib` builds the native library copied into the Python package by `setup.py`.
- `cargo test` runs Rust tests when present.
- `uv run python -m unittest discover -s tests` runs the Python API tests.
- `uv pip install -e .` installs the package in editable mode and triggers the release Rust library build.
- `cargo run -- inspect path/to/file.nd2` runs the CLI against an ND2 file.

## Coding Style & Naming Conventions

Use Rust 2024 edition idioms and run `cargo fmt` before submitting Rust changes. Keep Rust types in `UpperCamelCase`, functions and fields in `snake_case`, and constants in `SCREAMING_SNAKE_CASE`. Python uses 4-space indentation, type annotations for public APIs, `snake_case` functions and modules, and `UpperCamelCase` classes. Prefer `uv` for Python package management and command execution. Keep binary parsing logic in Rust.

## Testing Guidelines

Python tests use the standard `unittest` framework and should be named `test_*.py`. Keep generated fixtures small and local to tests, following `tests/test_python_api.py`. Add Rust tests near parser code for low-level validation and Python tests for user-facing API behavior. Run both `cargo test` and `uv run python -m unittest discover -s tests` before a pull request.

## Commit & Pull Request Guidelines

The history currently contains only `first commit`, so no strict convention is established. Use short, imperative subjects such as `add chunk map validation` or `fix ND2File close handling`. Pull requests should describe behavior changes, list test commands run, and mention ND2 sample files or edge cases used for validation. Include CLI output snippets for changes to `src/main.rs`.

## Security & Configuration Tips

Do not commit proprietary ND2 datasets or large binary samples. Keep test fixtures synthetic or minimal. Avoid unchecked reads from file offsets; validate sizes before allocation and preserve clear `Invalid` versus `Unsupported` errors.
