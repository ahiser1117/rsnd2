"""Compare the local Rust-backed ``nd2-rs`` Python API against the upstream PyPI
``nd2`` package using ND2 files already present on this filesystem.

Both distributions import as ``nd2``, so each backend runs in a separate Python
subprocess. The PyPI backend imports from this uv environment. The local backend
imports from this repository's ``src-python`` directory and loads the compiled
``target/release`` Rust shared library.

Run directly as a script::

    uv run python nd2_rs_vs_pypi_nd2.py

CSV and PNG outputs are written to the ``outputs/`` directory next to this file.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
OUTPUT_DIR = WORKSPACE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = WORKSPACE.parents[1]
LOCAL_SRC = REPO_ROOT / "src-python"
DATA_ROOTS = [
    Path("/store1/prj_sexsharedneurons/data_raw"),
    Path("/store1/prj_rim/data_raw"),
]
SAMPLE_PER_ROOT = int(os.environ.get("ND2_COMPARE_SAMPLE_PER_ROOT", "8"))
FEW_FRAME_COUNT = int(os.environ.get("ND2_COMPARE_FEW_FRAME_COUNT", "4"))
BATCH_FRAME_COUNTS = tuple(
    int(part)
    for part in os.environ.get("ND2_COMPARE_BATCH_FRAME_COUNTS", "16,64").split(",")
    if part.strip()
)
INCLUDE_COLD_CACHE = os.environ.get("ND2_COMPARE_COLD_CACHE", "1") != "0"


def local_library_path() -> Path | None:
    names = (
        ["nd2_rs.dll"]
        if sys.platform == "win32"
        else ["libnd2_rs.dylib"]
        if sys.platform == "darwin"
        else ["libnd2_rs.so"]
    )
    for profile in ("release", "debug"):
        for name in names:
            candidate = REPO_ROOT / "target" / profile / name
            if candidate.exists():
                return candidate
    return None


LOCAL_LIBRARY = local_library_path()
print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
print(f"Workspace: {WORKSPACE}")
print(f"Repo root: {REPO_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Local nd2 source: {LOCAL_SRC}")
print(f"Local Rust library: {LOCAL_LIBRARY or 'not built yet'}")


def short_text(value: object, limit: int = 240) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."

# ## Discover local ND2 files
#
# The requested data roots include many large acquisitions. The comparison uses
# the smallest files from each root by default; adjust `SAMPLE_PER_ROOT` or the
# `ND2_COMPARE_SAMPLE_PER_ROOT` environment variable for broader coverage.

def discover_nd2_files(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            rows.append(
                {
                    "root": str(root),
                    "path": None,
                    "size_bytes": None,
                    "error": "root does not exist",
                }
            )
            continue
        for path in root.rglob("*.nd2"):
            try:
                size = path.stat().st_size
            except OSError as exc:
                rows.append(
                    {
                        "root": str(root),
                        "path": str(path),
                        "size_bytes": None,
                        "error": repr(exc),
                    }
                )
                continue
            rows.append(
                {
                    "root": str(root),
                    "path": str(path),
                    "size_bytes": size,
                    "error": None,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["root", "path", "size_bytes", "size_gb", "error"])
    df["size_gb"] = df["size_bytes"].astype(float) / 1024**3
    return df.sort_values(["root", "size_bytes", "path"], na_position="last").reset_index(
        drop=True
    )


files = discover_nd2_files(DATA_ROOTS)
files.to_csv(OUTPUT_DIR / "discovered_nd2_files.csv", index=False)
print(files.groupby("root", dropna=False).agg(files=("path", "count"), min_gb=("size_gb", "min"), max_gb=("size_gb", "max")))
print(files.dropna(subset=["path"]).head(12).to_string(index=False))

sample_paths = (
    files.dropna(subset=["path"])
    .groupby("root", group_keys=False)
    .head(SAMPLE_PER_ROOT)["path"]
    .map(Path)
    .tolist()
)
print("Metadata sample paths:")
for path in sample_paths:
    print(f"  {path}")

# ## Backend runner
#
# Each probe returns JSON from a fresh Python interpreter. This avoids the
# shared `nd2` module-name collision and makes failures explicit per backend
# and per file.

PROBE_CODE = r'''
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as md
import json
import os
import sys
import time
import traceback
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]

def drop_file_cache(target):
    """Best-effort eviction of a file from the OS page cache (no root needed)."""
    fd = os.open(str(target), os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

def pack(value):
    if dataclasses.is_dataclass(value):
        return pack(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

def safe(label, fn):
    try:
        return {label: pack(fn())}
    except Exception as exc:
        return {label + "_error": f"{type(exc).__name__}: {exc}"}

result = {"path": str(path), "mode": mode}
try:
    import nd2
    result["module_file"] = getattr(nd2, "__file__", None)
    result["module_version"] = getattr(nd2, "__version__", None)
    for dist_name in ("nd2", "nd2-rs"):
        try:
            result[f"dist_{dist_name}_version"] = md.version(dist_name)
        except md.PackageNotFoundError:
            pass

    t0 = time.perf_counter()
    result.update(safe("is_supported_file", lambda: nd2.is_supported_file(path)))
    result["support_elapsed_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with nd2.ND2File(path) as f:
        result["open_elapsed_s"] = time.perf_counter() - t0
        for label, fn in (
            ("repr", lambda: repr(f)),
            ("version", lambda: tuple(f.version)),
            ("is_legacy", lambda: f.is_legacy),
            ("sizes", lambda: dict(f.sizes)),
            ("shape", lambda: tuple(f.shape)),
            ("ndim", lambda: f.ndim),
            ("dtype", lambda: str(f.dtype)),
            ("attributes", lambda: f.attributes),
            ("metadata_repr", lambda: repr(f.metadata)[:1200]),
        ):
            result.update(safe(label, fn))

        if hasattr(f, "index"):
            def index_summary():
                idx = dict(f.index)
                counts = idx.get("chunk_name_counts", {}) or {}
                return {
                    "variant": idx.get("variant"),
                    "signature_version": idx.get("signature_version"),
                    "plane_count": idx.get("plane_count"),
                    "chunk_count": sum(int(v) for v in counts.values()),
                    "top_chunks": sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:8],
                }
            result.update(safe("index_summary", index_summary))

        if mode == "frame0":
            import numpy as np
            reader = getattr(f, "read_frame", None) or getattr(f, "_get_frame", None)
            if reader is None:
                raise AttributeError("ND2File has no read_frame or _get_frame method")
            t1 = time.perf_counter()
            arr = np.asarray(reader(0))
            result["frame0_elapsed_s"] = time.perf_counter() - t1
            result["frame0_shape"] = tuple(int(x) for x in arr.shape)
            result["frame0_dtype"] = str(arr.dtype)
            result["frame0_min"] = float(np.nanmin(arr)) if arr.size else None
            result["frame0_max"] = float(np.nanmax(arr)) if arr.size else None
            result["frame0_mean"] = float(np.nanmean(arr)) if arr.size else None
            result["frame0_blake2b16"] = hashlib.blake2b(
                np.ascontiguousarray(arr).view(np.uint8), digest_size=16
            ).hexdigest()
        elif mode.startswith("frames:") or mode.startswith("coldframes:"):
            import numpy as np
            cold = mode.startswith("coldframes:")
            requested_count = int(mode.split(":", 1)[1])
            reader = getattr(f, "read_frame", None) or getattr(f, "_get_frame", None)
            if reader is None:
                raise AttributeError("ND2File has no read_frame or _get_frame method")
            attrs = f.attributes
            sequence_count = int(getattr(attrs, "sequenceCount", requested_count))
            actual_count = min(requested_count, sequence_count)
            if cold:
                # Evict the file from the page cache right before the timed read
                # so the read measures disk I/O rather than cached pages.
                drop_file_cache(path)
            t1 = time.perf_counter()
            arr = np.stack([np.asarray(reader(i)) for i in range(actual_count)])
            result["batch_elapsed_s"] = time.perf_counter() - t1
            result["cache_state"] = "cold" if cold else "warm"
            result["frame_count_requested"] = requested_count
            result["frame_count_read"] = actual_count
            result["batch_shape"] = tuple(int(x) for x in arr.shape)
            result["batch_dtype"] = str(arr.dtype)
            result["batch_bytes"] = int(arr.nbytes)
            result["batch_min"] = float(np.nanmin(arr)) if arr.size else None
            result["batch_max"] = float(np.nanmax(arr)) if arr.size else None
            result["batch_mean"] = float(np.nanmean(arr)) if arr.size else None
            result["batch_blake2b16"] = hashlib.blake2b(
                np.ascontiguousarray(arr).view(np.uint8), digest_size=16
            ).hexdigest()
    result["ok"] = True
except Exception as exc:
    result["ok"] = False
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback_tail"] = traceback.format_exc().splitlines()[-8:]

print(json.dumps(result, sort_keys=True))
'''


def run_backend(backend: str, path: Path, mode: str = "metadata", timeout: int = 120) -> dict[str, object]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-c", PROBE_CODE, str(path), mode]
    if backend == "pypi-nd2":
        cmd = [sys.executable, "-I", "-c", PROBE_CODE, str(path), mode]
    elif backend == "nd2-rs":
        env["PYTHONPATH"] = str(LOCAL_SRC)
        if LOCAL_LIBRARY is not None:
            env["ND2_RS_LIBRARY"] = str(LOCAL_LIBRARY)
    else:
        raise ValueError(f"unknown backend: {backend}")

    try:
        completed = subprocess.run(
            cmd,
            cwd=WORKSPACE,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "backend": backend,
            "path": str(path),
            "mode": mode,
            "ok": False,
            "error": f"timeout after {timeout}s",
        }

    stdout = completed.stdout.strip().splitlines()
    if not stdout:
        return {
            "backend": backend,
            "path": str(path),
            "mode": mode,
            "ok": False,
            "error": completed.stderr[-2000:],
            "returncode": completed.returncode,
        }
    try:
        payload = json.loads(stdout[-1])
    except json.JSONDecodeError:
        payload = {"ok": False, "error": stdout[-1], "stderr": completed.stderr[-2000:]}
    payload["backend"] = backend
    payload["returncode"] = completed.returncode
    if completed.stderr.strip():
        payload["stderr_tail"] = completed.stderr.strip()[-1000:]
    return payload

# ## Compare metadata/index behavior

metadata_rows: list[dict[str, object]] = []
for path in sample_paths:
    for backend in ("pypi-nd2", "nd2-rs"):
        print(f"metadata: {backend} -> {path}")
        metadata_rows.append(run_backend(backend, path, mode="metadata", timeout=180))

metadata = pd.json_normalize(metadata_rows)
if "error" in metadata.columns:
    metadata["error_short"] = metadata["error"].map(short_text)
metadata.to_csv(OUTPUT_DIR / "metadata_comparison.csv", index=False)
summary_cols = [
    "backend",
    "ok",
    "path",
    "module_version",
    "module_file",
    "is_supported_file",
    "version",
    "sizes",
    "shape",
    "dtype",
    "open_elapsed_s",
    "index_summary.variant",
    "index_summary.signature_version",
    "index_summary.plane_count",
    "error_short",
]
with pd.option_context("display.max_colwidth", 120):
    print(metadata[[c for c in summary_cols if c in metadata.columns]].to_string(index=False))


def valid_paths_from_metadata(metadata_df: pd.DataFrame) -> list[Path]:
    if metadata_df.empty:
        return []
    shaped = metadata_df[
        (metadata_df["ok"] == True)  # noqa: E712
        & metadata_df.get("shape", pd.Series(index=metadata_df.index, dtype=object)).map(
            lambda value: isinstance(value, list) and bool(value)
        )
        & metadata_df.get("dtype", pd.Series(index=metadata_df.index, dtype=object)).map(
            lambda value: isinstance(value, str) and bool(value)
        )
    ]
    valid_paths: list[Path] = []
    for path, group in shaped.groupby("path"):
        if set(group["backend"]) == {"pypi-nd2", "nd2-rs"}:
            valid_paths.append(Path(path))
    return valid_paths


valid_frame_paths = valid_paths_from_metadata(metadata)
print("Valid frame benchmark paths:")
for path in valid_frame_paths:
    print(f"  {path}")

if "open_elapsed_s" in metadata.columns:
    timing = metadata.copy()
    timing["file"] = timing["path"].map(lambda p: Path(p).name if isinstance(p, str) else p)
    table = timing.pivot_table(
        index="file", columns="backend", values="open_elapsed_s", aggfunc="first"
    )
    ax = table.plot(kind="bar", figsize=(10, 4))
    ax.set_ylabel("open + metadata seconds")
    ax.set_title("Metadata open time by backend")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    metadata_plot = OUTPUT_DIR / "metadata_open_time.png"
    plt.savefig(metadata_plot, dpi=160)
    plt.close()
    print(f"Saved {metadata_plot}")

# ## Compare a single-frame read
#
# This reads frame 0 from every sampled file that both backends can shape as a
# valid image. It records shape, dtype, aggregate values, and a content hash
# without loading full time series into memory.

frame_rows: list[dict[str, object]] = []
for path in valid_frame_paths:
    for backend in ("pypi-nd2", "nd2-rs"):
        print(f"frame0: {backend} -> {path}")
        frame_rows.append(run_backend(backend, path, mode="frame0", timeout=240))

frames = pd.json_normalize(frame_rows)
if "error" in frames.columns:
    frames["error_short"] = frames["error"].map(short_text)
frames.to_csv(OUTPUT_DIR / "frame0_comparison.csv", index=False)
frame_cols = [
    "backend",
    "ok",
    "path",
    "frame0_shape",
    "frame0_dtype",
    "frame0_min",
    "frame0_max",
    "frame0_mean",
    "frame0_blake2b16",
    "frame0_elapsed_s",
    "error_short",
]
with pd.option_context("display.max_colwidth", 120):
    print(frames[[c for c in frame_cols if c in frames.columns]].to_string(index=False))

if "frame0_elapsed_s" in frames.columns:
    plot_df = frames.copy()
    plot_df["file"] = plot_df["path"].map(lambda p: Path(p).name if isinstance(p, str) else p)
    table = plot_df.pivot_table(
        index="file", columns="backend", values="frame0_elapsed_s", aggfunc="first"
    )
    ax = table.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("frame 0 read seconds")
    ax.set_title("Single-frame read time by backend")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    frame_plot = OUTPUT_DIR / "frame0_read_time.png"
    plt.savefig(frame_plot, dpi=160)
    plt.close()
    print(f"Saved {frame_plot}")

# ## Compare multi-frame reads
#
# This benchmark reads batches of 4, 16, and 64 frames from every valid sampled
# file.  The batch hash is computed over the contiguous stacked array so
# correctness is checked beyond aggregate statistics.
#
# Each batch size is read twice: once cold (the file is evicted from the OS page
# cache right before the timed read, so it measures disk I/O) and once warm (the
# pages just populated by the cold read are reused).  Set ND2_COMPARE_COLD_CACHE=0
# to skip the cold pass.  The timing plot is split into one panel per batch size,
# a companion plot tracks the relative speed as the batch size increases, and a
# dedicated figure compares the cold and warm cache modes.

# Cold reads come first for each size so the warm read is genuinely served from
# cache.  Frame counts are read smallest-to-largest within each cache state.
frame_counts = (FEW_FRAME_COUNT, *BATCH_FRAME_COUNTS)
if INCLUDE_COLD_CACHE:
    batch_modes = [
        mode
        for count in frame_counts
        for mode in (f"coldframes:{count}", f"frames:{count}")
    ]
else:
    batch_modes = [f"frames:{count}" for count in frame_counts]
batch_rows: list[dict[str, object]] = []
for path in valid_frame_paths:
    for mode in batch_modes:
        for backend in ("pypi-nd2", "nd2-rs"):
            print(f"{mode}: {backend} -> {path}")
            batch_rows.append(run_backend(backend, path, mode=mode, timeout=600))

batches = pd.json_normalize(batch_rows)
if not batches.empty:
    if "error" in batches.columns:
        batches["error_short"] = batches["error"].map(short_text)
    batches["file"] = batches["path"].map(lambda p: Path(p).name if isinstance(p, str) else p)
    batches["read_label"] = batches["frame_count_requested"].map(
        lambda value: f"{int(value)} frames" if pd.notna(value) else ""
    )
    batches["throughput_MB_s"] = (
        pd.to_numeric(batches.get("batch_bytes"), errors="coerce")
        / pd.to_numeric(batches.get("batch_elapsed_s"), errors="coerce")
        / 1e6
    )
batches.to_csv(OUTPUT_DIR / "batch_read_comparison.csv", index=False)
batch_cols = [
    "backend",
    "ok",
    "path",
    "read_label",
    "frame_count_read",
    "batch_shape",
    "batch_dtype",
    "batch_blake2b16",
    "batch_elapsed_s",
    "throughput_MB_s",
    "error_short",
]
with pd.option_context("display.max_colwidth", 120):
    if not batches.empty:
        print(batches[[c for c in batch_cols if c in batches.columns]].to_string(index=False))
    else:
        print("No valid files available for multi-frame benchmark.")

# Tag each batch read with its cache state, deriving it from the probe mode for
# any rows that errored before recording it.
if not batches.empty:
    if "cache_state" not in batches.columns:
        batches["cache_state"] = pd.NA
    batches["cache_state"] = batches["cache_state"].fillna(
        batches["mode"].map(
            lambda m: "cold" if isinstance(m, str) and m.startswith("coldframes:") else "warm"
        )
    )


def speedup_by_size(df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-file PyPI/nd2-rs speedup indexed by batch size (rows) and file (cols)."""
    table = df.pivot_table(
        index=["file", "frame_count_requested"],
        columns="backend",
        values="batch_elapsed_s",
        aggfunc="first",
    )
    if not {"pypi-nd2", "nd2-rs"}.issubset(table.columns):
        return None
    return (table["pypi-nd2"] / table["nd2-rs"]).unstack("file").sort_index()


if not batches.empty and "batch_elapsed_s" in batches.columns:
    plot_df = batches[batches["ok"] == True].copy()  # noqa: E712
    if not plot_df.empty:
        plot_df["frame_count_requested"] = pd.to_numeric(
            plot_df["frame_count_requested"], errors="coerce"
        )
        warm_df = plot_df[plot_df["cache_state"] == "warm"]
        batch_sizes = sorted(plot_df["frame_count_requested"].dropna().unique())

        # Per-file read+stack time, one panel per batch size (warm cache).
        fig, axes = plt.subplots(
            1,
            len(batch_sizes),
            figsize=(max(5, 4.5 * len(batch_sizes)), 5),
            squeeze=False,
        )
        for ax, size in zip(axes[0], batch_sizes):
            sub = warm_df[warm_df["frame_count_requested"] == size]
            table = sub.pivot_table(
                index="file", columns="backend", values="batch_elapsed_s", aggfunc="first"
            )
            table.plot(kind="bar", ax=ax, legend=(ax is axes[0][0]))
            ax.set_title(f"{int(size)} frames")
            ax.set_xlabel("")
            ax.set_ylabel("read + stack seconds")
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle("Multi-frame read time by batch size (warm cache)")
        fig.tight_layout()
        batch_plot = OUTPUT_DIR / "batch_read_time.png"
        fig.savefig(batch_plot, dpi=160)
        plt.close(fig)
        print(f"Saved {batch_plot}")

        # Relative speed vs batch size: thin per-file lines plus a bold median,
        # drawn separately for each cache state so the I/O effect is explicit.
        fig, ax = plt.subplots(figsize=(9, 6))
        legend_handles = []
        for state, color in (("warm", "tab:blue"), ("cold", "tab:orange")):
            wide = speedup_by_size(plot_df[plot_df["cache_state"] == state])
            if wide is None or wide.empty:
                continue
            for file_name in wide.columns:
                ax.plot(
                    wide.index, wide[file_name],
                    marker="o", markersize=3, linewidth=0.8, color=color, alpha=0.25,
                )
            median_speedup = wide.median(axis=1)
            ax.plot(
                median_speedup.index, median_speedup.values,
                marker="s", linewidth=2.5, color=color,
            )
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=2.5, marker="s",
                       label=f"{state} cache: median across files")
            )
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
        legend_handles.append(Line2D([0], [0], color="grey", linestyle="--", label="parity (1x)"))
        ax.set_xscale("log", base=2)
        ax.set_xticks([int(s) for s in batch_sizes])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("batch size (frames)")
        ax.set_ylabel("PyPI nd2 seconds / nd2-rs seconds")
        ax.set_title("nd2-rs speedup vs batch size by cache state")
        ax.legend(handles=legend_handles)
        fig.tight_layout()
        speedup_plot = OUTPUT_DIR / "batch_read_speedup.png"
        fig.savefig(speedup_plot, dpi=160)
        plt.close(fig)
        print(f"Saved {speedup_plot}")

def matched_timing_table(
    df: pd.DataFrame,
    value_col: str,
    case_cols: list[str],
    *,
    ok_only: bool = True,
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if ok_only and "ok" in work.columns:
        work = work[work["ok"] == True]  # noqa: E712
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col, "backend", *case_cols])
    if work.empty:
        return pd.DataFrame()
    table = work.pivot_table(
        index=case_cols,
        columns="backend",
        values=value_col,
        aggfunc="first",
    )
    if {"pypi-nd2", "nd2-rs"}.issubset(table.columns):
        return table.dropna(subset=["pypi-nd2", "nd2-rs"])
    return pd.DataFrame()


def _case_label(index_value: object) -> str:
    if isinstance(index_value, tuple):
        parts = list(index_value)
    else:
        parts = [index_value]
    label_parts = []
    for part in parts:
        if isinstance(part, str) and part.endswith(".nd2"):
            label_parts.append(Path(part).name)
        elif isinstance(part, str) and "/" in part:
            label_parts.append(Path(part).name)
        elif pd.notna(part):
            label_parts.append(str(part))
    return "\n".join(label_parts)


def _time_label(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} us"
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def plot_timing_subplot(ax, table: pd.DataFrame, title: str, ylabel: str) -> None:
    if table.empty:
        ax.text(0.5, 0.5, "No matched valid timings", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return
    labels = [_case_label(value) for value in table.index]
    x = np.arange(len(table))
    width = 0.38
    pypi = table["pypi-nd2"].astype(float).to_numpy()
    rs = table["nd2-rs"].astype(float).to_numpy()
    pypi_bars = ax.bar(x - width / 2, pypi, width, label="PyPI nd2", color="#4c78a8")
    rs_bars = ax.bar(x + width / 2, rs, width, label="nd2-rs", color="#f58518")
    max_value = max(float(np.nanmax(pypi)), float(np.nanmax(rs)))
    text_offset = max_value * 0.025 if max_value > 0 else 0.001
    for bars in (pypi_bars, rs_bars):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + text_offset,
                _time_label(height),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    for idx, (pypi_s, rs_s) in enumerate(zip(pypi, rs)):
        if rs_s > 0:
            speedup = pypi_s / rs_s
            top = max(pypi_s, rs_s)
            ax.text(
                idx,
                top + text_offset * 6 if top > 0 else 0.001,
                f"{speedup:.1f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.margins(y=0.34)
    ax.grid(axis="y", alpha=0.25)


# The summary timing figures describe the warm-cache reads; the cold pass gets
# its own dedicated comparison further below.
if "cache_state" in batches.columns:
    warm_batches = batches[batches["cache_state"] == "warm"]
    cold_batches = batches[batches["cache_state"] == "cold"]
else:
    warm_batches = batches
    cold_batches = batches.iloc[0:0]

metadata_summary = matched_timing_table(metadata, "open_elapsed_s", ["path"])
frame0_summary = matched_timing_table(frames, "frame0_elapsed_s", ["path"])
batch_summary = matched_timing_table(warm_batches, "batch_elapsed_s", ["path", "read_label"])

fig, axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=False)
plot_timing_subplot(
    axes[0],
    metadata_summary,
    "Metadata Open/Index Time",
    "seconds",
)
plot_timing_subplot(
    axes[1],
    frame0_summary,
    "Frame 0 Read Time",
    "seconds",
)
plot_timing_subplot(
    axes[2],
    batch_summary,
    "Batch Read + Stack Time",
    "seconds",
)
handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.975))
fig.suptitle("nd2-rs vs PyPI nd2 timing summary: absolute time and speedup", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.94])
summary_plot = OUTPUT_DIR / "timing_summary_subplots.png"
fig.savefig(summary_plot, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved {summary_plot}")


def average_timing_row(table: pd.DataFrame, benchmark: str) -> dict[str, float | str | int]:
    if table.empty:
        return {
            "benchmark": benchmark,
            "cases": 0,
            "pypi-nd2_mean_s": np.nan,
            "nd2-rs_mean_s": np.nan,
            "speedup_from_means": np.nan,
            "mean_case_speedup": np.nan,
        }
    pypi = table["pypi-nd2"].astype(float)
    rs = table["nd2-rs"].astype(float)
    case_speedup = pypi / rs.replace(0, np.nan)
    pypi_mean = float(pypi.mean())
    rs_mean = float(rs.mean())
    return {
        "benchmark": benchmark,
        "cases": len(table),
        "pypi-nd2_mean_s": pypi_mean,
        "nd2-rs_mean_s": rs_mean,
        "speedup_from_means": pypi_mean / rs_mean if rs_mean > 0 else np.nan,
        "mean_case_speedup": float(case_speedup.mean()),
    }


average_rows = [
    average_timing_row(metadata_summary, "Metadata open/index"),
    average_timing_row(frame0_summary, "Frame 0 read"),
    average_timing_row(batch_summary, "Batch read + stack"),
]
average_timings = pd.DataFrame(average_rows)
average_timings.to_csv(OUTPUT_DIR / "average_timing_summary.csv", index=False)
print(average_timings.to_string(index=False))


def average_table(row: pd.Series) -> pd.DataFrame:
    if int(row["cases"]) <= 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "pypi-nd2": [float(row["pypi-nd2_mean_s"])],
            "nd2-rs": [float(row["nd2-rs_mean_s"])],
        },
        index=["Average"],
    )


def batch_average_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Mean read+stack time per backend for each batch size, averaged over files."""
    matched = matched_timing_table(df, "batch_elapsed_s", ["path", "frame_count_requested"])
    if matched.empty:
        return pd.DataFrame()
    means = matched.groupby(level="frame_count_requested").mean().sort_index()
    means.index = [f"x{int(size)}" for size in means.index]
    return means[["pypi-nd2", "nd2-rs"]]


batch_size_means = batch_average_by_size(warm_batches)

fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=False)
plot_timing_subplot(
    axes[0],
    average_table(average_timings.iloc[0]),
    f"Metadata open/index average across {int(average_timings.iloc[0]['cases'])} matched cases",
    "mean seconds",
)
plot_timing_subplot(
    axes[1],
    average_table(average_timings.iloc[1]),
    f"Frame 0 read average across {int(average_timings.iloc[1]['cases'])} matched cases",
    "mean seconds",
)
plot_timing_subplot(
    axes[2],
    batch_size_means,
    "Batch read + stack mean across files, by batch size (warm cache)",
    "mean seconds",
)
handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.97))
fig.suptitle("Average nd2-rs vs PyPI nd2 timing summary", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.93])
average_plot = OUTPUT_DIR / "average_timing_summary_subplots.png"
fig.savefig(average_plot, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved {average_plot}")


# Dedicated cold- vs warm-cache comparison: the same per-batch-size backend
# averages, with one panel per cache state so the I/O cost is read off directly.
cold_size_means = batch_average_by_size(cold_batches)
if not cold_size_means.empty:
    # Independent y-axes: cold reads are ~100x slower, so a shared scale would
    # flatten the warm panel to invisibility.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    plot_timing_subplot(
        axes[0],
        batch_size_means,
        "Warm cache: mean read + stack by batch size",
        "mean seconds",
    )
    plot_timing_subplot(
        axes[1],
        cold_size_means,
        "Cold cache: mean read + stack by batch size",
        "mean seconds",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("nd2-rs vs PyPI nd2: cold vs warm cache by batch size", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    cache_plot = OUTPUT_DIR / "batch_cache_comparison.png"
    fig.savefig(cache_plot, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cache_plot}")

# ## Analysis

def _fmt_seconds(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "n/a"


def render_analysis(
    files_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    batches_df: pd.DataFrame,
    average_df: pd.DataFrame,
) -> str:
    lines: list[str] = ["# nd2-rs vs PyPI nd2 analysis", ""]
    lines.append("## Data scanned")
    for root, group in files_df.dropna(subset=["path"]).groupby("root"):
        lines.append(
            f"- `{root}`: {len(group)} files, "
            f"{group['size_gb'].min():.3f} GiB smallest, {group['size_gb'].max():.3f} GiB largest."
        )
    lines.append("")

    lines.append("## Metadata/index comparison")
    for backend, group in metadata_df.groupby("backend"):
        ok = int(group["ok"].fillna(False).sum())
        total = len(group)
        elapsed = pd.to_numeric(group.get("open_elapsed_s"), errors="coerce")
        lines.append(
            f"- `{backend}`: {ok}/{total} metadata probes succeeded; "
            f"median open/index time {_fmt_seconds(elapsed.median())}."
        )
    failures = metadata_df[metadata_df["ok"] != True]  # noqa: E712
    if not failures.empty:
        lines.append("- Metadata failures:")
        for row in failures.itertuples(index=False):
            lines.append(
                f"  - `{getattr(row, 'backend')}` on `{Path(getattr(row, 'path')).name}`: "
                f"{short_text(getattr(row, 'error', 'unknown error'))}"
            )
    lines.append("")

    partial_cols = [
        col
        for col in ("shape_error", "dtype_error", "attributes_error", "metadata_repr_error")
        if col in metadata_df.columns
    ]
    if partial_cols:
        partials = metadata_df[
            metadata_df[partial_cols].apply(
                lambda row: any(bool(short_text(value)) for value in row), axis=1
            )
        ]
        if not partials.empty:
            lines.append("## Partial metadata/API errors")
            for _, row in partials.iterrows():
                details = [
                    f"{col}={short_text(row[col], 160)}"
                    for col in partial_cols
                    if short_text(row.get(col))
                ]
                lines.append(
                    f"- `{row['backend']}` on `{Path(row['path']).name}`: "
                    + "; ".join(details)
                )
            lines.append("")

    shape_cols = [c for c in ("backend", "path", "shape", "dtype", "index_summary.variant") if c in metadata_df.columns]
    if shape_cols:
        lines.append("## Shape and dtype observations")
        for path, group in metadata_df[metadata_df["ok"] == True].groupby("path"):  # noqa: E712
            bits = []
            for _, row in group.iterrows():
                bits.append(
                    f"{row['backend']} shape={row.get('shape')} dtype={row.get('dtype')}"
                )
            lines.append(f"- `{Path(path).name}`: " + "; ".join(bits) + ".")
        lines.append(
            "- PyPI `nd2` and `nd2-rs` should report the same shape and dtype for "
            "supported uncompressed modern ND2 files. Differences here indicate a "
            "metadata parsing or frame-layout compatibility gap."
        )
        lines.append("")

    lines.append("## Frame-0 comparison")
    for backend, group in frames_df.groupby("backend"):
        ok = int(group["ok"].fillna(False).sum())
        total = len(group)
        elapsed = pd.to_numeric(group.get("frame0_elapsed_s"), errors="coerce")
        lines.append(
            f"- `{backend}`: {ok}/{total} frame-0 probes succeeded; "
            f"median read time {_fmt_seconds(elapsed.median())}."
        )
    frame_failures = frames_df[frames_df["ok"] != True]  # noqa: E712
    if not frame_failures.empty:
        lines.append("- Frame read failures:")
        for row in frame_failures.itertuples(index=False):
            lines.append(
                f"  - `{getattr(row, 'backend')}` on `{Path(getattr(row, 'path')).name}`: "
                f"{short_text(getattr(row, 'error', 'unknown error'))}"
            )
    lines.append("")

    lines.append("## Multi-frame read comparison")
    if batches_df.empty:
        lines.append("- No valid files were available for multi-frame benchmarking.")
    else:
        cache_states = ["warm", "cold"] if "cache_state" in batches_df.columns else [None]
        for state in cache_states:
            scoped = (
                batches_df[batches_df["cache_state"] == state]
                if state is not None
                else batches_df
            )
            if scoped.empty:
                continue
            label = f" ({state} cache)" if state is not None else ""
            for backend, group in scoped.groupby("backend"):
                ok = int(group["ok"].fillna(False).sum())
                total = len(group)
                elapsed = pd.to_numeric(group.get("batch_elapsed_s"), errors="coerce")
                throughput = pd.to_numeric(group.get("throughput_MB_s"), errors="coerce")
                lines.append(
                    f"- `{backend}`{label}: {ok}/{total} multi-frame probes succeeded; "
                    f"median read+stack time {_fmt_seconds(elapsed.median())}; "
                    f"median throughput {throughput.median():.1f} MB/s."
                )

        ok_batches = batches_df[batches_df["ok"] == True].copy()  # noqa: E712
        if not ok_batches.empty:
            if "cache_state" not in ok_batches.columns:
                ok_batches["cache_state"] = "warm"
            hash_matches = []
            speedup_by_state: dict[object, list[float]] = {}
            group_keys = ["path", "frame_count_requested", "cache_state"]
            for (path, requested, state), group in ok_batches.groupby(group_keys):
                by_backend = group.set_index("backend")
                if {"pypi-nd2", "nd2-rs"}.issubset(by_backend.index):
                    pypi_hash = by_backend.loc["pypi-nd2", "batch_blake2b16"]
                    rs_hash = by_backend.loc["nd2-rs", "batch_blake2b16"]
                    hash_matches.append(bool(pypi_hash == rs_hash))
                    pypi_elapsed = float(by_backend.loc["pypi-nd2", "batch_elapsed_s"])
                    rs_elapsed = float(by_backend.loc["nd2-rs", "batch_elapsed_s"])
                    if rs_elapsed > 0:
                        speedup_by_state.setdefault(state, []).append(pypi_elapsed / rs_elapsed)
            if hash_matches:
                lines.append(
                    f"- Batch hashes matched for {sum(hash_matches)}/{len(hash_matches)} "
                    "backend/file/count comparisons."
                )
            for state in ("warm", "cold"):
                values = speedup_by_state.get(state)
                if values:
                    lines.append(
                        f"- Median nd2-rs speedup over PyPI nd2 ({state} cache): "
                        f"{pd.Series(values).median():.1f}x."
                    )
    batch_failures = batches_df[batches_df["ok"] != True] if not batches_df.empty else batches_df  # noqa: E712
    if not batch_failures.empty:
        lines.append("- Multi-frame failures:")
        for row in batch_failures.itertuples(index=False):
            lines.append(
                f"  - `{getattr(row, 'backend')}` on `{Path(getattr(row, 'path')).name}` "
                f"mode `{getattr(row, 'mode')}`: "
                f"{short_text(getattr(row, 'error', 'unknown error'))}"
            )
    lines.append("")

    lines.append("## Average timing summary")
    if average_df.empty:
        lines.append("- No matched timing averages were available.")
    else:
        for _, row in average_df.iterrows():
            lines.append(
                f"- `{row['benchmark']}` over {int(row['cases'])} "
                f"matched cases: PyPI nd2 mean {_fmt_seconds(row['pypi-nd2_mean_s'])}, "
                f"nd2-rs mean {_fmt_seconds(row['nd2-rs_mean_s'])}, "
                f"speedup from means {float(row['speedup_from_means']):.1f}x."
            )
    lines.append("")

    lines.append("## Generated files")
    for name in (
        "discovered_nd2_files.csv",
        "metadata_comparison.csv",
        "frame0_comparison.csv",
        "batch_read_comparison.csv",
        "average_timing_summary.csv",
        "metadata_open_time.png",
        "frame0_read_time.png",
        "batch_read_time.png",
        "batch_read_speedup.png",
        "batch_cache_comparison.png",
        "timing_summary_subplots.png",
        "average_timing_summary_subplots.png",
    ):
        lines.append(f"- `{OUTPUT_DIR / name}`")
    lines.append("")
    return "\n".join(lines)


analysis = render_analysis(files, metadata, frames, batches, average_timings)
analysis_path = OUTPUT_DIR / "analysis.md"
analysis_path.write_text(analysis)
print(analysis)
print(f"Saved {analysis_path}")
