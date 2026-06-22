"""Figures: single PIT histogram, 2×4 cross-asset grid, KS-vs-kurtosis scatter.

matplotlib only; renders from saved arrays / summary rows, never the model.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Kaggle / CI safe)
import matplotlib.pyplot as plt
import numpy as np


def pit_hist(ax, pit: np.ndarray, title: str, bins: int = 9) -> None:
    """Draw one PIT histogram with the uniform reference line onto ``ax``."""
    pit = np.asarray(pit, dtype=float)
    ax.hist(pit, bins=bins, range=(0, 1), density=True,
            edgecolor="black", color="steelblue", alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")


def pit_grid(
    asset_pits: dict[str, np.ndarray],
    ks_by_asset: dict[str, float] | None = None,
    ncols: int = 4,
    save_path: str | Path | None = None,
):
    """2×4 grid of PIT histograms — the first cross-asset thesis figure."""
    items = list(asset_pits.items())
    n = len(items)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, pit) in zip(axes, items):
        title = name
        if ks_by_asset and name in ks_by_asset:
            title = f"{name}\nKS={ks_by_asset[name]:.4f}"
        pit_hist(ax, pit, title)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Zero-shot PIT — TimesFM 2.5 (flip-invariance OFF, log returns)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def ks_vs_kurtosis(rows: list[dict], save_path: str | Path | None = None):
    """Scatter KS vs excess kurtosis — tests 'miscalibration scales with tail
    thickness'. ``rows`` are summary dicts from eval.summarize_run."""
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r.get("excess_kurtosis", np.nan) for r in rows]
    ys = [r.get("ks", np.nan) for r in rows]
    names = [r.get("asset", "") for r in rows]
    ax.scatter(xs, ys, s=60, color="darkorange", edgecolor="black", zorder=3)
    for x, y, nm in zip(xs, ys, names):
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(nm, (x, y), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
    # OLS trend (guard against <2 finite points).
    fin = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(fin) >= 2:
        fx, fy = np.array(fin).T
        b, a = np.polyfit(fx, fy, 1)
        xr = np.linspace(min(fx), max(fx), 50)
        ax.plot(xr, b * xr + a, "--", color="gray",
                label=f"slope={b:.4f}")
        ax.legend()
    ax.set_xlabel("excess kurtosis of returns")
    ax.set_ylabel("KS statistic (PIT vs uniform)")
    ax.set_title("Calibration error vs tail thickness")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
