"""Render the optimization-progress plot for the GPU streaming work.

For the 64-frame batch, plot nd2-rs average disk->GPU runtime against the commit
timestamp of each change. Successful changes (each a new best, committed) are
joined by a solid line; failed changes (slower, not committed) are drawn as
low-opacity red markers branching off the best line. Each point is annotated
with a one-line description and its speedup over PyPI nd2 in the same run.

    uv run python make_progress_plot.py
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = WORKSPACE / "results" / "gpu_stream_results.csv"
OUT_DIR = WORKSPACE / "outputs"
OUT_DIR.mkdir(exist_ok=True)
BATCH = 64

# Commit time (x position) for each committed change; failed attempts fall back
# to their measurement timestamp. Times are git commit times for this branch.
COMMIT_TIME = {
    "baseline": "2026-06-18T16:00:00-04:00",
    "optA": "2026-06-18T16:46:31-04:00",
    "optB": "2026-06-18T16:49:29-04:00",
    "optC": "2026-06-18T16:53:58-04:00",
}
ORDER = ["baseline", "optA", "optB", "optC"]
SHORT = {
    "baseline": "baseline\n(naive np.stack + H2D)",
    "optA": "Opt A: one contiguous\nRust batch read",
    "optB": "Opt B: adaptive\nparallel reads",
    "optC": "Opt C: pinned\nreusable buffer",
    "FAILED-overlap": "overlap H2D\n(chunk overhead > gain)",
    "FAILED-threads64": "fixed 64 threads\n(spawn overhead)",
}


def load():
    df = pd.read_csv(RESULTS)
    df = df[(df.batch == BATCH) & (df.ok == True)].copy()  # noqa: E712
    # Collapse the repeated optC measurements (optC / optC-verify / optC-final,
    # all the same final code) into one point so its value is a stable central
    # estimate rather than a single NFS-noisy run.
    df["tag"] = df["tag"].apply(lambda t: "optC" if str(t).startswith("optC") else t)
    return df


def agg(df, backend, cache):
    sub = df[(df.backend == backend) & (df.cache == cache)]
    g = sub.groupby("tag")["elapsed_mean_s"].mean() * 1000.0  # ms
    return g


def xtime(df, tag):
    if tag in COMMIT_TIME:
        return pd.Timestamp(COMMIT_TIME[tag]).tz_convert("UTC")
    ts = df[df.tag == tag]["timestamp"].iloc[0]
    return pd.Timestamp(ts).tz_convert("UTC")


# Annotation placement (offset in points from the point) + leader arrows, hand
# tuned to drop labels into the empty middle of the timeline where the points
# cluster on the right.
ANN = {
    "baseline": (12, 18),
    "optA": (-150, 150),
    "optB": (-200, 95),
    "optC": (-150, 45),
    "FAILED-overlap": (30, 70),
    "FAILED-threads64": (45, 25),
}


def panel(ax, df, cache, title):
    rs = agg(df, "nd2-rs", cache)
    pypi = agg(df, "pypi-nd2", cache)
    pypi_mean = float(pypi.reindex(ORDER).dropna().mean())

    # reference lines: PyPI nd2 and the >=4x target
    ax.axhline(pypi_mean, color="#888888", ls="--", lw=1.2, zorder=1)
    ax.text(0.015, pypi_mean, f"  PyPI nd2 ~{pypi_mean:.1f} ms", color="#666666",
            va="bottom", ha="left", fontsize=8.5, transform=ax.get_yaxis_transform())
    target = pypi_mean / 4.0
    ax.axhline(target, color="#2ca02c", ls=":", lw=1.6, zorder=1)
    ax.text(0.015, target, f"  ≥4x target ~{target:.2f} ms", color="#2ca02c",
            va="bottom", ha="left", fontsize=8.5, transform=ax.get_yaxis_transform())

    # successful trajectory (solid line through the committed bests)
    xs = [xtime(df, t) for t in ORDER if t in rs.index]
    ys = [rs[t] for t in ORDER if t in rs.index]
    ax.plot(xs, ys, "-o", color="#1f3b8c", lw=2.4, ms=8, zorder=4, label="committed (new best)")

    optc_x, optc_y = xtime(df, "optC"), rs.get("optC")

    # failed attempts: faint red, branching off the optC best
    failed = [t for t in rs.index if str(t).startswith("FAILED")]
    for i, t in enumerate(failed):
        fx, fy = xtime(df, t), rs[t]
        ax.plot([optc_x, fx], [optc_y, fy], "--", color="red", alpha=0.30, lw=1.5, zorder=2)
        ax.plot([fx], [fy], "X", color="red", alpha=0.55, ms=11, zorder=3,
                label="failed (slower, reverted)" if i == 0 else None)

    def speedup(tag):
        return pypi_mean / rs[tag] if tag in rs.index else float("nan")

    for t in [*ORDER, *failed]:
        if t not in rs.index:
            continue
        is_fail = str(t).startswith("FAILED")
        color = "red" if is_fail else "#1f3b8c"
        label = f"{SHORT.get(t, t)}\n{speedup(t):.1f}x vs PyPI"
        ax.annotate(
            label, (xtime(df, t), rs[t]), textcoords="offset points",
            xytext=ANN.get(t, (10, 12)), fontsize=8, color=color,
            alpha=0.8 if is_fail else 1.0, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.45, lw=1.0),
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(f"avg disk→GPU runtime, batch={BATCH} (ms)")
    ax.set_xlabel("commit timestamp (UTC)")
    ax.set_ylim(bottom=0, top=max(ys) * 1.18)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", fontsize=9)
    if cache == "cold":
        ax.text(0.99, 0.02,
                "cold first-touch is bounded by the NFS single-client\n"
                "bandwidth ceiling (~1 GB/s); ratio varies 3.6–4.6x with\n"
                "PyPI's single-stream speed (mean of optC runs shown)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="#555555", style="italic")


def main():
    df = load()
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    panel(axes[0], df, "warm", "Warm cache (page-cache resident) — sustained streaming")
    panel(axes[1], df, "cold", "Cold cache (genuine NFS read) — first-touch from disk")
    fig.suptitle(
        "nd2-rs batched frame → GPU streaming optimization (batch=64)\n"
        "solid = committed improvements (each a new best); faint red ✕ = failed attempts (reverted)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for dest in (OUT_DIR / "optimization_progress.png", WORKSPACE / "results" / "optimization_progress.png"):
        fig.savefig(dest, dpi=160)
        print(f"saved {dest}")
    plt.close(fig)


if __name__ == "__main__":
    main()
