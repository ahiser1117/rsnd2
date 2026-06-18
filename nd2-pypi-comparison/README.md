# nd2-rs vs PyPI nd2 comparison

This workspace contains a uv-managed notebook environment for comparing this
repository's Rust-backed `nd2-rs` Python API with the upstream PyPI `nd2`
package on local ND2 files under:

- `/store1/prj_sexsharedneurons/data_raw`
- `/store1/prj_rim/data_raw`

Setup:

```bash
cd notebooks/nd2-pypi-comparison
uv sync
cd ../..
cargo build --release --lib
cd notebooks/nd2-pypi-comparison
uv run python nd2_rs_vs_pypi_nd2.py
```

The percent-format Python file can also be opened interactively in VS Code,
JupyterLab, or another editor that understands `# %%` cells. It writes tables,
figures, and a short interpretation to `outputs/`.

The comparison intentionally runs the two readers in subprocesses because both
packages import as `nd2`. The PyPI reader is imported from the uv environment;
the local reader is imported from this repository's `src-python` directory with
`ND2_RS_LIBRARY` pointing at the compiled Rust shared library.
