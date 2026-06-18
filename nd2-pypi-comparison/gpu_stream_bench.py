"""GPU streaming benchmark: stream batches of ND2 frames from disk into a CUDA
torch tensor, comparing the local Rust-backed ``nd2-rs`` against the upstream
PyPI ``nd2`` package.

The metric of interest is *end-to-end* wall time to get ``batch`` frames from an
``.nd2`` file on disk into a contiguous ``torch.cuda`` tensor (shape
``(batch, C, Y, X)``), measured in both a cold page-cache state (genuine disk /
NFS read) and a warm state (file resident in the OS page cache).

Both distributions import as ``nd2``, so each backend runs in its own Python
subprocess. The PyPI backend imports from this uv environment; the local backend
imports from this repository's ``src-python`` directory and loads the compiled
``target/release`` shared library.

Each measurement is appended to ``results/gpu_stream_results.csv`` tagged with a
commit hash, a wall-clock timestamp, and a short human description of the change
under test, so the progress plot can reconstruct the full optimization history.

Usage::

    uv run python gpu_stream_bench.py --desc "baseline naive stack" --tag <commit-or-FAILED>

Environment knobs:
    ND2_GPU                 CUDA device ordinal to expose (default 1; GPU 0 is busy)
    ND2_BENCH_BATCHES       comma list of batch sizes (default 1,8,16,32,64,128)
    ND2_BENCH_WARM_REPS     warm-cache timed repeats per batch (default 5)
    ND2_BENCH_COLD_REPS     cold-cache timed repeats per batch (default 3)
    ND2_BENCH_FILES         comma list of .nd2 paths (default: built-in rim set)
    ND2_BENCH_NO_COLD       set to 1 to skip cold-cache passes (fast iteration)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parent
LOCAL_SRC = REPO_ROOT / "src-python"
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "gpu_stream_results.csv"

DEFAULT_FILES = [
    "/store1/prj_rim/data_raw/2023-06-02-03.nd2",
    "/store1/prj_rim/data_raw/2023-06-02-04.nd2",
    "/store1/prj_rim/data_raw/2023-06-03-11.nd2",
]

GPU = os.environ.get("ND2_GPU", "2")
BATCHES = [int(x) for x in os.environ.get("ND2_BENCH_BATCHES", "1,8,16,32,64,128").split(",") if x.strip()]
WARM_REPS = int(os.environ.get("ND2_BENCH_WARM_REPS", "5"))
COLD_REPS = int(os.environ.get("ND2_BENCH_COLD_REPS", "3"))
INCLUDE_COLD = os.environ.get("ND2_BENCH_NO_COLD", "0") != "1"
FILES = [p for p in os.environ.get("ND2_BENCH_FILES", ",".join(DEFAULT_FILES)).split(",") if p.strip()]

CSV_FIELDS = [
    "tag", "timestamp", "desc", "backend", "path", "batch", "cache",
    "method", "elapsed_mean_s", "elapsed_min_s", "elapsed_median_s",
    "reps", "bytes", "throughput_MB_s", "shape", "dtype", "hash", "ok", "error",
]


def local_library_path() -> Path | None:
    for profile in ("release", "debug"):
        cand = REPO_ROOT / "target" / profile / "libnd2_rs.so"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# The probe runs inside a subprocess with the chosen backend on PYTHONPATH.
# It loops over all batch sizes / cache states for one file, importing torch
# once, and prints one JSON line per measurement to stdout.
# ---------------------------------------------------------------------------
PROBE_CODE = r'''
import hashlib, json, os, statistics, sys, time
from pathlib import Path

path = Path(sys.argv[1])
batches = [int(x) for x in sys.argv[2].split(",")]
warm_reps = int(sys.argv[3])
cold_reps = int(sys.argv[4])
include_cold = sys.argv[5] == "1"
backend = sys.argv[6]

import numpy as np
import torch
import nd2

DEV = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES already restricts to one GPU
torch.cuda.init()
# warm up the cuda context / allocator so the first timed run is not penalised
_ = torch.empty(1024, 1024, device=DEV); torch.cuda.synchronize(); del _


def drop_cache(target):
    fd = os.open(str(target), os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def emit(row):
    sys.stdout.write(json.dumps(row) + "\n")
    sys.stdout.flush()


# --- the nd2-rs optimized streaming path, if this build exposes one ----------
FORCE_NAIVE = os.environ.get("ND2_RS_FORCE_NAIVE", "0") == "1"


def stream_rs(f, indices):
    """Return a contiguous (N,C,Y,X) CUDA tensor for the given frame indices,
    using the fastest method this build of nd2-rs provides."""
    fn = getattr(f, "read_batch_to_torch", None)
    if fn is not None and not FORCE_NAIVE:
        return fn(indices, device=DEV)
    # fallback / baseline: naive per-frame read + host stack + single H2D copy
    arr = np.stack([np.asarray(f.read_frame(i)) for i in indices])
    return torch.from_numpy(np.ascontiguousarray(arr)).to(DEV, non_blocking=True)


def stream_pypi(f, indices):
    reader = getattr(f, "read_frame", None) or getattr(f, "_get_frame", None)
    arr = np.stack([np.asarray(reader(i)) for i in indices])
    return torch.from_numpy(np.ascontiguousarray(arr)).to(DEV, non_blocking=True)


def timed_stream(f, indices):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t = maker(f, indices)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, t


result_method = (
    "read_batch_to_torch"
    if (backend == "nd2-rs" and hasattr(nd2.ND2File, "read_batch_to_torch") and not FORCE_NAIVE)
    else "naive_stack"
)
maker = stream_rs if backend == "nd2-rs" else stream_pypi


def emit_stat(batch, cache, samples, meta):
    emit({
        "ok": True, "backend": backend, "path": str(path), "batch": batch,
        "cache": cache, "method": result_method,
        "elapsed_mean_s": statistics.fmean(samples),
        "elapsed_min_s": min(samples),
        "elapsed_median_s": statistics.median(samples),
        "reps": len(samples), "bytes": meta["bytes"],
        "throughput_MB_s": meta["bytes"] / statistics.fmean(samples) / 1e6,
        "shape": meta["shape"], "dtype": meta["dtype"], "hash": meta["hash"],
    })


try:
    # discover sequence count once
    with nd2.ND2File(path) as f0:
        seqcount = int(getattr(f0.attributes, "sequenceCount", 0))

    for batch in batches:
        n = min(batch, seqcount)
        indices = list(range(n))
        try:
            # ---- warm cache: one persistent handle, pages resident ----
            with nd2.ND2File(path) as f:
                _, t = timed_stream(f, indices)  # correctness reference
                ref = t.detach().to("cpu").contiguous()
                meta = {
                    "hash": hashlib.blake2b(ref.numpy().view(np.uint8), digest_size=16).hexdigest(),
                    "shape": [int(x) for x in t.shape],
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "bytes": int(ref.numpy().nbytes),
                }
                del t, ref
                # Untimed warm-ups so the CUDA caching allocator has expanded to
                # this batch size before timing; otherwise a one-off allocation
                # pollutes the mean of the (few) timed reps.
                for _ in range(3):
                    _, t = timed_stream(f, indices); del t
                warm = []
                for _ in range(warm_reps):
                    dt, t = timed_stream(f, indices)
                    warm.append(dt); del t
                emit_stat(batch, "warm", warm, meta)

            # ---- cold cache: genuine per-rep eviction. The file is evicted
            # with NO live mapping (the previous ND2File is closed), a fresh
            # handle is opened (metadata only, not timed), then the pixel
            # stream is timed reading genuinely-cold bytes from disk/NFS. ----
            if include_cold:
                cold = []
                for _ in range(cold_reps):
                    drop_cache(path)
                    with nd2.ND2File(path) as f:
                        _ = f.attributes  # warm metadata region only (not timed)
                        dt, t = timed_stream(f, indices)
                        cold.append(dt); del t
                emit_stat(batch, "cold", cold, meta)
        except Exception as exc:
            import traceback
            emit({
                "ok": False, "backend": backend, "path": str(path), "batch": batch,
                "cache": "warm", "method": result_method,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-6:],
            })
except Exception as exc:
    import traceback
    emit({
        "ok": False, "backend": backend, "path": str(path), "batch": None,
        "cache": None, "method": result_method,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc().splitlines()[-8:],
    })
'''


def run_backend(backend: str, path: str) -> list[dict]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    cmd = [sys.executable, "-c", PROBE_CODE, path,
           ",".join(str(b) for b in BATCHES), str(WARM_REPS), str(COLD_REPS),
           "1" if INCLUDE_COLD else "0", backend]
    if backend == "pypi-nd2":
        # use the uv venv's own interpreter, isolated from local src
        pass
    elif backend == "nd2-rs":
        env["PYTHONPATH"] = str(LOCAL_SRC)
        lib = local_library_path()
        if lib is not None:
            env["ND2_RS_LIBRARY"] = str(lib)
    else:
        raise ValueError(backend)

    proc = subprocess.run(cmd, cwd=WORKSPACE, env=env, capture_output=True, text=True, timeout=3600)
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows and proc.returncode != 0:
        rows.append({"ok": False, "backend": backend, "path": path,
                     "error": proc.stderr[-1500:], "batch": None, "cache": None})
    if proc.stderr.strip():
        sys.stderr.write(f"[{backend} {Path(path).name}] stderr tail:\n{proc.stderr.strip()[-800:]}\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True, help="one-line description of the change under test")
    ap.add_argument("--tag", default="WIP", help="commit hash, or FAILED for a slower attempt")
    ap.add_argument("--backends", default="pypi-nd2,nd2-rs")
    args = ap.parse_args()

    sys.executable_real = sys.executable
    ts = datetime.now(timezone.utc).isoformat()
    print(f"GPU={GPU} batches={BATCHES} warm_reps={WARM_REPS} cold_reps={COLD_REPS} cold={INCLUDE_COLD}")
    print(f"tag={args.tag} desc={args.desc!r} ts={ts}")
    print(f"local lib: {local_library_path()}")

    all_rows: list[dict] = []
    for backend in args.backends.split(","):
        for path in FILES:
            print(f"  run {backend} {Path(path).name} ...", flush=True)
            t0 = time.time()
            rows = run_backend(backend, path)
            for r in rows:
                r.setdefault("backend", backend)
                r.setdefault("path", path)
                r["tag"] = args.tag
                r["timestamp"] = ts
                r["desc"] = args.desc
                all_rows.append(r)
            print(f"    -> {len(rows)} rows in {time.time()-t0:.1f}s", flush=True)

    # append to the persistent CSV
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in all_rows:
            row = dict(r)
            if isinstance(row.get("shape"), (list, tuple)):
                row["shape"] = "x".join(str(s) for s in row["shape"])
            w.writerow(row)

    # quick summary + correctness check at batch=64
    print("\n=== summary (mean seconds) ===")
    summarize(all_rows)


def summarize(rows: list[dict]) -> None:
    by = {}
    hashes = {}
    for r in rows:
        if not r.get("ok"):
            if r.get("error"):
                print(f"  ERROR {r.get('backend')} {Path(str(r.get('path'))).name} "
                      f"batch={r.get('batch')}: {str(r.get('error'))[:200]}")
            continue
        key = (r["backend"], r["batch"], r["cache"])
        by.setdefault(key, []).append(r["elapsed_mean_s"])
        hashes.setdefault((r["batch"], Path(r["path"]).name, r["cache"]), {})[r["backend"]] = r["hash"]

    for cache in ("warm", "cold"):
        print(f"\n  -- {cache} cache --")
        print(f"  {'batch':>6} {'pypi_s':>10} {'rs_s':>10} {'speedup':>8}")
        for batch in sorted({k[1] for k in by if k[2] == cache}):
            p = by.get(("pypi-nd2", batch, cache))
            r = by.get(("nd2-rs", batch, cache))
            if p and r:
                pm = sum(p) / len(p); rm = sum(r) / len(r)
                print(f"  {batch:>6} {pm:>10.5f} {rm:>10.5f} {pm/rm:>7.2f}x")

    # correctness
    mism = [k for k, v in hashes.items() if len(v) == 2 and len(set(v.values())) != 1]
    if mism:
        print(f"\n  !! HASH MISMATCH (nd2-rs != pypi) for: {mism[:10]}")
    else:
        n = sum(1 for v in hashes.values() if len(v) == 2)
        print(f"\n  correctness: {n} batch/file/cache pairs hashed identical across backends")


if __name__ == "__main__":
    main()
