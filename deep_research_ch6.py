#!/usr/bin/env python3
"""第六章深度研究：建图剪枝策略的多维分析

在第四章实验四的基础上，本章从三个维度深入分析剪枝效果：
  实验一：稀疏度与剪枝收益的定量关系（散点拟合 + 残差分析）
  实验二：剪枝对运行时间的加速效果（横向条形图 + 加速比排位）
  实验三：网络规模缩减的边际效应（球队数量 vs 缩减率折线图）

输出：
  ex6/figures/ch6_exp1_scatter.png/pdf       — 实验一：散点拟合 + 残差
  ex6/figures/ch6_exp2_speedup.png/pdf       — 实验二：剪枝加速比条形图
  ex6/figures/ch6_exp3_marginal.png/pdf      — 实验三：边际效应折线图
  ex6/results/ch6_exp2_speedup.csv           — 实验二数据表
  ex6/results/ch6_exp3_marginal.csv          — 实验三数据表

使用方法：
  python3 deep_research_ch6.py
"""

from __future__ import annotations

import csv
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
RED = "#D55E00"
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scale_rows = read_csv(RESULTS_DIR / "network_scale.csv")

    datasets = sorted({r["dataset"] for r in scale_rows},
                      key=lambda d: (int("".join(c for c in d if c.isdigit()) or "999"), d))

    records = []
    for ds in datasets:
        orig_r = next(r for r in scale_rows if r["dataset"] == ds and r["build_mode"] == "original")
        prun_r = next(r for r in scale_rows if r["dataset"] == ds and r["build_mode"] == "pruned")
        n = int(orig_r["n"])
        max_pairs = n * (n - 1) * (n - 2) // 2 if n >= 2 else 1
        orig_edges = int(orig_r["flow_edges"])
        prun_edges = int(prun_r["flow_edges"])
        orig_games = int(orig_r["game_nodes"])
        orig_time = float(orig_r["avg_time_ms"])
        prun_time = float(prun_r["avg_time_ms"])

        sparsity = 1.0 - (orig_games / max_pairs) if max_pairs > 0 else 0.0
        pruning_ratio = (orig_edges - prun_edges) / orig_edges if orig_edges > 0 else 0
        speedup = orig_time / prun_time if prun_time > 0 else 1.0
        edge_reduction = orig_edges - prun_edges

        records.append({
            "dataset": ds.removesuffix(".txt"), "n": n,
            "max_pairs": max_pairs, "orig_games": orig_games,
            "orig_edges": orig_edges, "prun_edges": prun_edges,
            "orig_time": orig_time, "prun_time": prun_time,
            "sparsity": sparsity, "pruning_ratio": pruning_ratio,
            "speedup": speedup, "edge_reduction": edge_reduction,
        })

    # ===== 实验一：稀疏度与剪枝收益散点拟合 + 残差分析 =====
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(10, 4.5),
                                       gridspec_kw={'width_ratios': [2.2, 1]})

    sparsity_vals = [r["sparsity"] for r in records]
    ratio_vals = [r["pruning_ratio"] for r in records]
    sizes = [max(r["n"] * 2, 15) for r in records]

    # 散点 + 拟合
    ax1a.scatter(sparsity_vals, ratio_vals, s=sizes, c=BLUE, alpha=0.65,
                 edgecolors="white", linewidth=0.6, zorder=5)
    valid = [(s, r) for s, r in zip(sparsity_vals, ratio_vals) if r > 0.001]
    sx, sy = [v[0] for v in valid], [v[1] for v in valid]
    if len(sx) > 3:
        coeff = np.polyfit(sx, sy, 1)
        fn = np.poly1d(coeff)
        xl = np.linspace(min(sx), max(sx), 100)
        ax1a.plot(xl, fn(xl), color=ORANGE, linewidth=1.8, linestyle="--",
                  label=f"Linear fit (R²={1-sum((sy[i]-fn(sx[i]))**2 for i in range(len(sx)))/sum((y-np.mean(sy))**2 for y in sy):.3f})")
        # 残差
        residuals = [sy[i] - fn(sx[i]) for i in range(len(sx))]
        ax1b.barh(range(len(residuals)), residuals, color=[RED if r > 0 else GREEN for r in residuals],
                  edgecolor="white", linewidth=0.5, height=0.7)
        ax1b.axvline(0, color=TEXT, linewidth=0.8)
        ax1b.set_xlabel("Residual"); ax1b.set_title("Residual Distribution", fontsize=11)
    ax1a.legend(loc="upper left", fontsize=8)
    ax1a.set_xlabel("Schedule Sparsity"); ax1a.set_ylabel("Pruning Benefit")
    ax1a.set_title("Exp 1: Schedule Sparsity vs Pruning Benefit", fontsize=13)
    ax1a.set_xlim(-0.02, 0.90); ax1a.set_ylim(-0.04, 1.04)
    ax1a.xaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax1a.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax1a.set_axisbelow(True); ax1b.set_axisbelow(True)
    ax1a.spines["top"].set_visible(False); ax1a.spines["right"].set_visible(False)
    ax1b.spines["top"].set_visible(False); ax1b.spines["right"].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "ch6_exp1_scatter.png", bbox_inches="tight", facecolor="white")
    fig1.savefig(FIGURES_DIR / "ch6_exp1_scatter.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print("实验一图表已保存")

    # ===== 实验二：剪枝节省的绝对运行时间 vs 网络规模 =====
    valid_r = [r for r in records if r["pruning_ratio"] > 0.01]
    valid_r.sort(key=lambda r: r["n"])

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ns2 = [r["n"] for r in valid_r]
    time_saved = [r["orig_time"] - r["prun_time"] for r in valid_r]
    sizes2 = [max(r["pruning_ratio"] * 200, 12) for r in valid_r]
    labels2 = [r["dataset"] for r in valid_r]

    scatter = ax2.scatter(ns2, time_saved, s=sizes2, c=BLUE, alpha=0.65,
                          edgecolors="white", linewidth=0.6, zorder=5)
    # 标注关键点
    for r in valid_r:
        saved = r["orig_time"] - r["prun_time"]
        if saved > 50 or r["pruning_ratio"] > 0.8 or r["n"] > 50:
            ax2.annotate(r["dataset"], (r["n"], saved),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=7, color=TEXT, alpha=0.85)
    # 参考线: 1ms
    ax2.axhline(1.0, color=ORANGE, linewidth=0.8, linestyle="--", alpha=0.5,
                label="1 ms threshold (above: perceptible gain)")
    ax2.set_xlabel("Number of Teams n")
    ax2.set_ylabel("Absolute Time Saved by Pruning (ms)")
    ax2.set_title("Exp 2: Absolute Time Saved vs Network Scale", fontsize=13)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_yscale("log")
    ax2.xaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    # 图例说明气泡
    ax2.text(0.98, 0.96, "Bubble size ∝ pruning ratio",
             transform=ax2.transAxes, fontsize=8, ha="right", va="top",
             color=TEXT, alpha=0.7)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "ch6_exp2_timesaved.png", bbox_inches="tight", facecolor="white")
    fig2.savefig(FIGURES_DIR / "ch6_exp2_timesaved.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    # 保存 CSV
    with open(RESULTS_DIR / "ch6_exp2_timesaved.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n", "orig_time_ms", "prun_time_ms",
                                           "time_saved_ms", "pruning_ratio_pct", "meaningful"])
        w.writeheader()
        for r in valid_r:
            saved = r["orig_time"] - r["prun_time"]
            w.writerow({"dataset": r["dataset"], "n": r["n"],
                        "orig_time_ms": f"{r['orig_time']:.4f}",
                        "prun_time_ms": f"{r['prun_time']:.4f}",
                        "time_saved_ms": f"{saved:.4f}",
                        "pruning_ratio_pct": f"{r['pruning_ratio']*100:.1f}",
                        "meaningful": "Yes" if saved > 1.0 else "No"})
    # 统计
    meaningful = [r for r in valid_r if (r["orig_time"] - r["prun_time"]) > 1.0]
    trivial = [r for r in valid_r if (r["orig_time"] - r["prun_time"]) <= 0.1]
    print("实验二图表及数据已保存")

    # ===== 实验三：边际效应——球队规模 vs 边减少量 =====
    records_by_n = sorted(records, key=lambda r: r["n"])
    fig3, ax3a = plt.subplots(figsize=(7.5, 4.2))
    ns = [r["n"] for r in records_by_n]
    reductions = [r["edge_reduction"] for r in records_by_n]
    ratios = [r["pruning_ratio"] for r in records_by_n]
    labels3 = [r["dataset"] for r in records_by_n]

    # 双Y轴：柱状图为边减少量（对数），折线为缩减比例
    ax3b = ax3a.twinx()
    bars = ax3a.bar(range(len(ns)), reductions, color=PURPLE, edgecolor="white",
                    linewidth=0.5, alpha=0.75, label="Flow Edges Reduced")
    ax3a.set_yscale("log")
    ax3a.set_ylabel("Flow Edges Reduced (log scale)", color=PURPLE)
    ax3a.tick_params(axis='y', labelcolor=PURPLE)
    ax3b.plot(range(len(ns)), [r * 100 for r in ratios], color=ORANGE, linewidth=2,
              marker='o', markersize=7, markerfacecolor=ORANGE, markeredgecolor='white',
              label="Pruning Reduction Ratio (%)")
    ax3b.set_ylabel("Pruning Reduction Ratio (%)", color=ORANGE)
    ax3b.tick_params(axis='y', labelcolor=ORANGE)
    ax3b.set_ylim(-5, 105)

    ax3a.set_xticks(range(len(ns)))
    ax3a.set_xticklabels(labels3, rotation=40, ha="right", fontsize=8)
    ax3a.set_title("Exp 3: Team Scale vs Marginal Pruning Benefit", fontsize=13)
    lines1, lbl1 = ax3a.get_legend_handles_labels()
    lines2, lbl2 = ax3b.get_legend_handles_labels()
    fig3.legend(lines1 + lines2, lbl1 + lbl2, ncol=2, loc="upper center",
                bbox_to_anchor=(0.5, 0.97), fontsize=9)
    ax3a.spines["top"].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "ch6_exp3_marginal.png", bbox_inches="tight", facecolor="white")
    fig3.savefig(FIGURES_DIR / "ch6_exp3_marginal.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    # 保存 CSV
    with open(RESULTS_DIR / "ch6_exp3_marginal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n", "orig_edges", "prun_edges",
                                           "edge_reduction", "pruning_ratio_pct"])
        w.writeheader()
        for r in records:
            w.writerow({"dataset": r["dataset"], "n": r["n"],
                        "orig_edges": r["orig_edges"], "prun_edges": r["prun_edges"],
                        "edge_reduction": r["edge_reduction"],
                        "pruning_ratio_pct": f"{r['pruning_ratio']*100:.1f}"})
    print("实验三图表及数据已保存")

    # 统计摘要
    print(f"\n=== 第六章实验统计 ===")
    print(f"数据集总数: {len(records)}")
    print(f"剪枝有效数据集 (pruning_ratio > 1%): {len(valid_r)}")
    print(f"剪枝无效数据集 (pruning_ratio = 0): {len([r for r in records if r['pruning_ratio'] < 0.001])}")
    print(f"总共节省流边: {sum(r['edge_reduction'] for r in records):,}")
    print(f"绝对时间节省 > 1ms 的数据集: {len(meaningful)} / {len(valid_r)}")
    print(f"绝对时间节省 < 0.1ms 的数据集: {len(trivial)} / {len(valid_r)}")


if __name__ == "__main__":
    main()
