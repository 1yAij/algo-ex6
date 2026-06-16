#!/usr/bin/env python3
"""深入研究实验：赛程稀疏度对建图剪枝效果的影响分析

本实验基于 ex6/results/network_scale.csv 和 princeton_correctness.csv 数据，
分析每个数据集的剩余比赛稀疏度与剪枝收益之间的关系，回答以下问题：
—— "赛程越稀疏，剪枝收益是否线性增长？"
—— "是否存在剪枝无效的临界稀疏度？"

输出：
  ex6/figures/sparsity_pruning_scatter.png/pdf  — 散点拟合图
  ex6/figures/sparsity_pruning_bar.png/pdf      — 分组柱状图

使用方法：
  python3 sparsity_experiment.py
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

# 色盲友好配色
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#5B4B8A"
GREEN = "#009E73"
GRID = "#D8DEE9"
TEXT = "#222222"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 180, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 读取数据
    scale_rows = read_csv(RESULTS_DIR / "network_scale.csv")

    # 计算每个数据集的指标
    datasets = sorted({r["dataset"] for r in scale_rows},
                      key=lambda d: (int("".join(c for c in d if c.isdigit()) or "999"), d))

    records = []
    for ds in datasets:
        orig_r = next(r for r in scale_rows if r["dataset"] == ds and r["build_mode"] == "original")
        prun_r = next(r for r in scale_rows if r["dataset"] == ds and r["build_mode"] == "pruned")
        n = int(orig_r["n"])
        max_pairs_total = n * (n - 1) * (n - 2) // 2 if n >= 2 else 1  # n×C(n-1,2)
        orig_edges = int(orig_r["flow_edges"])
        prun_edges = int(prun_r["flow_edges"])
        orig_games = int(orig_r["game_nodes"])

        if max_pairs_total == 0:
            continue  # skip n=1

        sparsity = 1.0 - (orig_games / max_pairs_total)  # 0=完全稠密, 1=完全稀疏
        pruning_ratio = (orig_edges - prun_edges) / orig_edges if orig_edges > 0 else 0

        records.append({
            "dataset": ds.removesuffix(".txt"),
            "n": n,
            "max_pairs": max_pairs_total,
            "orig_games": orig_games,
            "orig_edges": orig_edges,
            "prun_edges": prun_edges,
            "sparsity": sparsity,
            "pruning_ratio": pruning_ratio,
        })

    # ===== 图 1: 散点拟合图 — 稀疏度 vs 剪枝收益 =====
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.0))

    sparsity_vals = [r["sparsity"] for r in records]
    ratio_vals = [r["pruning_ratio"] for r in records]
    labels = [r["dataset"] for r in records]
    sizes = [r["n"] * 3 for r in records]  # bubble size ∝ team count

    # 散点
    scatter = ax1.scatter(sparsity_vals, ratio_vals, s=sizes, c=BLUE,
                          alpha=0.7, edgecolors="white", linewidth=0.8, zorder=5)

    # 标注关键数据集
    for r in records:
        if r["pruning_ratio"] > 0.7 or r["sparsity"] > 0.9 or r["sparsity"] < 0.05:
            offset_x = 0.015 if r["sparsity"] < 0.9 else -0.06
            ax1.annotate(r["dataset"], (r["sparsity"], r["pruning_ratio"]),
                         textcoords="offset points", xytext=(offset_x * 700, 6),
                         fontsize=7, color=TEXT, alpha=0.85)

    # 线性拟合
    valid = [(s, r) for s, r in zip(sparsity_vals, ratio_vals) if r > 0.001]
    if len(valid) > 3:
        sx = [v[0] for v in valid]
        sy = [v[1] for v in valid]
        coeff = np.polyfit(sx, sy, 1)
        fit_fn = np.poly1d(coeff)
        x_line = np.linspace(min(sx), max(sx), 100)
        ax1.plot(x_line, fit_fn(x_line), color=ORANGE, linewidth=1.8, linestyle="--",
                 label=f"Linear fit (slope={coeff[0]:.2f})", zorder=4)
        # R²
        ss_res = sum((sy[i] - fit_fn(sx[i])) ** 2 for i in range(len(sx)))
        ss_tot = sum((y - np.mean(sy)) ** 2 for y in sy)
        r2 = 1 - ss_res / ss_tot
        ax1.text(0.95, 0.12, f"R² = {r2:.3f}", transform=ax1.transAxes,
                 fontsize=10, ha="right", color=ORANGE, fontweight="600")

    ax1.set_xlabel("Schedule Sparsity (1 - actual pairs / max possible pairs)")
    ax1.set_ylabel("Pruning Benefit (edges reduced / original edges)")
    ax1.set_title("Schedule Sparsity vs Pruning Benefit", fontsize=14)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xlim(-0.02, 1.08)
    ax1.set_ylim(-0.04, 1.08)
    ax1.xaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "sparsity_pruning_scatter.png", bbox_inches="tight", facecolor="white")
    fig1.savefig(FIGURES_DIR / "sparsity_pruning_scatter.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print("图 1 已保存: sparsity_pruning_scatter.png/pdf")

    # ===== 图 2: 分组柱状图 — 各数据集剪枝前后对比 (top 15 by edge count) =====
    sorted_recs = sorted(records, key=lambda r: r["orig_edges"], reverse=True)[:15]
    fig2, ax2 = plt.subplots(figsize=(8, 5.0))
    x = np.arange(len(sorted_recs))
    width = 0.36
    ax2.bar(x - width / 2, [r["orig_edges"] for r in sorted_recs], width,
            color=PURPLE, edgecolor="white", linewidth=0.6, label="Original")
    ax2.bar(x + width / 2, [r["prun_edges"] for r in sorted_recs], width,
            color=ORANGE, edgecolor="white", linewidth=0.6, label="Pruned")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["dataset"] for r in sorted_recs], rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Flow Edges (log scale)")
    ax2.set_title("Network Scale Before and After Pruning (Top 15)", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "sparsity_pruning_bar.png", bbox_inches="tight", facecolor="white")
    fig2.savefig(FIGURES_DIR / "sparsity_pruning_bar.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("图 2 已保存: sparsity_pruning_bar.png/pdf")

    # ===== 表格输出 =====
    print("\n数据集稀疏度与剪枝收益一览表")
    print(f"{'数据集':<22} {'球队数':>4} {'最大对数':>7} {'实际对数':>7} {'稀疏度':>7} {'剪枝比':>7}")
    print("-" * 60)
    for r in records:
        print(f"{r['dataset']:<22} {r['n']:>4} {r['max_pairs']:>7} {r['orig_games']:>7} "
              f"{r['sparsity']:>6.1%} {r['pruning_ratio']:>6.1%}")

    # 关键结论
    high_sparse = [r for r in records if r["sparsity"] > 0.8]
    dense = [r for r in records if r["sparsity"] < 0.01]
    print(f"\n结论概要:")
    print(f"  - 高稀疏度 (>{0.8}) 数据集: {len(high_sparse)} 个, 平均剪枝比 = "
          f"{sum(r['pruning_ratio'] for r in high_sparse)/max(len(high_sparse),1):.1%}")
    print(f"  - 完全稠密 (稀疏度<{0.01}) 数据集: {len(dense)} 个, 平均剪枝比 = "
          f"{sum(r['pruning_ratio'] for r in dense)/max(len(dense),1):.1%}")
    print(f"  - 线性拟合 R² = {r2:.3f}")


if __name__ == "__main__":
    main()
