#!/usr/bin/env python3
"""Publication-style plots for Lab 6 baseball elimination experiments.

The script reads CSV files produced by baseball_elimination_experiment.py and
generates report-ready PNG/PDF figures with academic styling.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Colorblind-safe Okabe-Ito inspired palette.
COLOR_TRIVIAL = "#56B4E9"
COLOR_NONTRIVIAL = "#E69F00"
COLOR_EK = "#0072B2"
COLOR_DINIC = "#009E73"
COLOR_ORIGINAL = "#5B4B8A"
COLOR_PRUNED = "#F59E0B"
COLOR_GRID = "#D8DEE9"
COLOR_TEXT = "#222222"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#333333",
            "axes.labelcolor": COLOR_TEXT,
            "axes.titlecolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def short_dataset_label(name: str) -> str:
    return name.removesuffix(".txt")


def add_value_labels(ax: plt.Axes, bars, fmt="{:.0f}", dy=0.04) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_TEXT,
        )


def plot_elimination_type_count() -> None:
    """Diverging horizontal bar chart — trivial left, non-trivial right."""
    rows = read_csv(RESULTS_DIR / "princeton_correctness.csv")
    labels = [short_dataset_label(row["dataset"]) for row in rows]
    trivial = [int(row["trivial"]) for row in rows]
    nontrivial = [int(row["nontrivial"]) for row in rows]
    y = list(range(len(labels)))

    fig_height = max(4.8, 0.28 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))

    # Trivial → left (negative), Non-trivial → right (positive)
    ax.barh(
        y,
        [-t for t in trivial],
        height=0.66,
        color=COLOR_TRIVIAL,
        edgecolor="white",
        linewidth=0.8,
        label="Trivial elimination",
    )
    ax.barh(
        y,
        nontrivial,
        height=0.66,
        color=COLOR_NONTRIVIAL,
        edgecolor="white",
        linewidth=0.8,
        label="Non-trivial elimination",
    )

    # Value labels
    max_val = max(max(trivial), max(nontrivial), 1)
    offset = max_val * 0.04
    for idx, t in enumerate(trivial):
        if t:
            ax.text(-t - offset, idx, str(t), ha="right", va="center", fontsize=8)
    for idx, n in enumerate(nontrivial):
        if n:
            ax.text(n + offset, idx, str(n), ha="left", va="center", fontsize=8)

    # Zero center line
    ax.axvline(0, color=COLOR_TEXT, linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Number of eliminated teams")
    ax.set_ylabel("Dataset")
    fig.suptitle("Trivial vs Non-trivial eliminations across baseball datasets", y=0.985, fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Symmetric x-axis: same range left and right
    limit = max_val * 1.25
    ax.set_xlim(-limit, limit)
    # Tick labels show absolute counts
    ax.set_xticks([-t for t in range(max_val + 1, 0, -2)] + list(range(0, max_val + 1, 2)))
    ax.set_xticklabels([str(abs(int(x))) for x in ax.get_xticks()])

    ax.invert_yaxis()
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.18, right=0.94)

    save_figure(fig, "elimination_type_count_academic")


def _runtime_scale(values: list[float]) -> str:
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return "linear"
    ratio = max(positive) / min(positive)
    return "log" if ratio >= 20 else "linear"


def plot_maxflow_runtime_comparison() -> None:
    """Horizontal grouped bar chart for Edmonds-Karp vs Dinic runtime."""
    rows = read_csv(RESULTS_DIR / "maxflow_runtime.csv")
    labels = [short_dataset_label(row["dataset"]) for row in rows]
    ek = [float(row["edmonds_karp_ms"]) for row in rows]
    dinic = [float(row["dinic_ms"]) for row in rows]
    speedup = [float(row["speedup"]) for row in rows]

    ek_std = [float(row.get("edmonds_karp_std_ms", 0) or 0) for row in rows]
    dinic_std = [float(row.get("dinic_std_ms", 0) or 0) for row in rows]
    use_error = any(v > 0 for v in ek_std + dinic_std)

    y = list(range(len(labels)))
    height = 0.34
    fig_height = max(4.5, 0.34 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    scale = _runtime_scale(ek + dinic)

    error_kw = {"elinewidth": 0.8, "capsize": 3, "capthick": 0.8, "ecolor": "#333333"}
    ax.barh(
        [i - height / 2 for i in y],
        ek,
        height=height,
        color=COLOR_EK,
        edgecolor="white",
        linewidth=0.8,
        xerr=ek_std if use_error else None,
        error_kw=error_kw if use_error else None,
        label="Edmonds-Karp",
    )
    ax.barh(
        [i + height / 2 for i in y],
        dinic,
        height=height,
        color=COLOR_DINIC,
        edgecolor="white",
        linewidth=0.8,
        xerr=dinic_std if use_error else None,
        error_kw=error_kw if use_error else None,
        label="Dinic",
    )

    if scale == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    else:
        ax.set_xlim(0, max(ek + dinic + [1e-9]) * 1.35)

    for idx, ratio in enumerate(speedup):
        if math.isfinite(ratio) and ratio > 0:
            x_pos = max(ek[idx], dinic[idx])
            if scale == "log":
                x_pos *= 1.22
            else:
                x_pos += max(ek + dinic + [1e-9]) * 0.035
            ax.text(x_pos, idx, f"{ratio:.2f}x", ha="left", va="center", fontsize=8, color=COLOR_TEXT)

    ax.set_xlabel("Average runtime (ms)")
    ax.set_ylabel("Dataset")
    fig.suptitle("Runtime comparison of maximum-flow solvers", y=0.985, fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.18, right=0.92)

    save_figure(fig, "maxflow_runtime_comparison_academic")


def plot_network_scale_comparison() -> None:
    """Horizontal grouped bar chart with log scale — Original vs Pruned flow edges."""
    rows = read_csv(RESULTS_DIR / "network_scale.csv")
    datasets = sorted({row["dataset"] for row in rows},
                      key=lambda d: (int("".join(c for c in d if c.isdigit()) or "999"), d))

    labels = [short_dataset_label(d) for d in datasets]
    original = [int(next(r["flow_edges"] for r in rows if r["dataset"] == ds and r["build_mode"] == "original")) for ds in datasets]
    pruned = [int(next(r["flow_edges"] for r in rows if r["dataset"] == ds and r["build_mode"] == "pruned")) for ds in datasets]

    y = list(range(len(labels)))
    height = 0.34
    fig_height = max(4.8, 0.34 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))

    ax.barh(
        [i - height / 2 for i in y],
        original,
        height=height,
        color=COLOR_ORIGINAL,
        edgecolor="white",
        linewidth=0.8,
        label="Original (all game pairs)",
    )
    ax.barh(
        [i + height / 2 for i in y],
        pruned,
        height=height,
        color=COLOR_PRUNED,
        edgecolor="white",
        linewidth=0.8,
        label="Pruned ($g_{ij} > 0$ only)",
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.set_xlabel("Flow edges (log scale)")
    ax.set_ylabel("Dataset")
    fig.suptitle("Network scale before and after zero-game pruning", y=0.985, fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)

    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.18, right=0.92)

    save_figure(fig, "network_scale_comparison_academic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate academic figures for Lab 6.")
    parser.add_argument(
        "figure",
        nargs="?",
        default="all",
        choices=["all", "elimination", "runtime", "network"],
    )
    args = parser.parse_args()

    setup_style()
    if args.figure in ("all", "elimination"):
        plot_elimination_type_count()
    if args.figure in ("all", "runtime"):
        plot_maxflow_runtime_comparison()
    if args.figure in ("all", "network"):
        plot_network_scale_comparison()


if __name__ == "__main__":
    main()
