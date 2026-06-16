#!/usr/bin/env python3
"""第六章实验：当前弧优化的真伪命题 —— Dinic_Standard vs Dinic_Naive

假设：在棒球赛流网络（深度≤4层）中，当前弧优化对DFS的加速收益极低，
      而维护cur数组的额外访存开销可能抵消甚至逆转这一收益。

测试集：
  - teams4.txt    (极小规模, n=4)
  - teams60.txt   (大规模稀疏, n=60, 启用剪枝)
  - teams50.txt   (大规模极端稠密, n=50, 不剪枝 — 核心观测点)

输出：
  ex6/results/ch6_arc_comparison.csv
  ex6/figures/ch6_arc_comparison.png/pdf
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

# 导入实验模块
import sys
sys.path.insert(0, str(ROOT))
from baseball_elimination_experiment import (
    read_baseball_file, classify_team, BaseballData,
    OFFICIAL_DATA_DIR, RESULTS_DIR as _, FIGURES_DIR as __,
)

# 色盲友好配色
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#5B4B8A"
RED = "#D55E00"
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


def run_comparison(data: BaseballData, repeats: int = 10) -> dict:
    """对指定数据集的每支球队，交替运行 Dinic_Standard 和 Dinic_Naive 各 repeats 次。"""
    timings: dict[str, list[float]] = {"dinic_std": [], "dinic_nv": []}
    n = data.n

    for _ in range(repeats):
        # 交替运行以减少系统负载波动影响
        for algo in ("dinic", "dinic_naive"):
            start = time.perf_counter()
            for target in range(n):
                classify_team(data, target, algorithm=algo, prune_zero_games=True)
            timings["dinic_std" if algo == "dinic" else "dinic_nv"].append(
                (time.perf_counter() - start) * 1000
            )

    return {
        "dataset": data.dataset,
        "n": n,
        "dinic_std_mean": statistics.mean(timings["dinic_std"]),
        "dinic_std_std": statistics.stdev(timings["dinic_std"]) if repeats > 1 else 0.0,
        "dinic_nv_mean": statistics.mean(timings["dinic_nv"]),
        "dinic_nv_std": statistics.stdev(timings["dinic_nv"]) if repeats > 1 else 0.0,
        "repeats": repeats,
    }


def plot(results: list[dict]):
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [r["dataset"].removesuffix(".txt") for r in results]
    std_vals = [r["dinic_std_mean"] for r in results]
    nv_vals = [r["dinic_nv_mean"] for r in results]
    std_errs = [r["dinic_std_std"] for r in results]
    nv_errs = [r["dinic_nv_std"] for r in results]
    n_vals = [r["n"] for r in results]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(datasets))
    width = 0.32

    bars1 = ax.bar(x - width/2, std_vals, width, color=PURPLE, edgecolor="white",
                   linewidth=0.6, label="Dinic (with current-arc opt.)")
    bars2 = ax.bar(x + width/2, nv_vals, width, color=ORANGE, edgecolor="white",
                   linewidth=0.6, label="Dinic_Naive (no current-arc opt.)")

    # 误差线 + 数值标注
    for i in range(len(datasets)):
        ax.errorbar(x[i] - width/2, std_vals[i], yerr=std_errs[i],
                    fmt='none', ecolor=PURPLE, capsize=4, capthick=0.8, linewidth=0.8)
        ax.errorbar(x[i] + width/2, nv_vals[i], yerr=nv_errs[i],
                    fmt='none', ecolor=ORANGE, capsize=4, capthick=0.8, linewidth=0.8)
        # 标注差值
        diff = nv_vals[i] - std_vals[i]
        diff_pct = (diff / std_vals[i]) * 100 if std_vals[i] > 0 else 0
        y_max = max(std_vals[i], nv_vals[i])
        sign = "+" if diff > 0 else ""
        ax.text(x[i], y_max + max(std_vals) * 0.04,
                f"{sign}{diff:.2f}ms ({sign}{diff_pct:.1f}%)",
                ha="center", fontsize=8, color=RED if diff > 0 else GREEN, fontweight="600")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(datasets, n_vals)], fontsize=10)
    ax.set_ylabel("Total Runtime (ms)")
    ax.set_title("Real Benefit of Current-Arc Optimization: Dinic vs Dinic_Naive", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ch6_arc_comparison.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_DIR / "ch6_arc_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("图表已保存: ch6_arc_comparison.png/pdf")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 测试数据集
    test_files = ["teams4.txt", "teams60.txt", "teams50.txt"]
    repeats = 10

    results = []
    for fname in test_files:
        path = OFFICIAL_DATA_DIR / fname
        if not path.exists():
            print(f"  [跳过] {fname} 不存在")
            continue
        data = read_baseball_file(path)
        print(f"正在测试 {fname} (n={data.n}, repeats={repeats})...")
        r = run_comparison(data, repeats=repeats)
        results.append(r)
        print(f"  Dinic_Std: {r['dinic_std_mean']:.3f} ± {r['dinic_std_std']:.3f} ms")
        print(f"  Dinic_Nv:  {r['dinic_nv_mean']:.3f} ± {r['dinic_nv_std']:.3f} ms")
        diff = r["dinic_nv_mean"] - r["dinic_std_mean"]
        pct = (diff / r["dinic_std_mean"]) * 100 if r["dinic_std_mean"] > 0 else 0
        winner = "Std 更快" if diff > 0 else ("Naive 更快" if diff < 0 else "持平")
        print(f"  差异: {diff:+.3f} ms ({pct:+.1f}%) — {winner}")

    # 保存 CSV
    csv_path = RESULTS_DIR / "ch6_arc_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "n", "dinic_std_mean_ms", "dinic_std_std_ms",
            "dinic_nv_mean_ms", "dinic_nv_std_ms", "repeats"
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "dataset": r["dataset"], "n": r["n"],
                "dinic_std_mean_ms": f"{r['dinic_std_mean']:.4f}",
                "dinic_std_std_ms": f"{r['dinic_std_std']:.4f}",
                "dinic_nv_mean_ms": f"{r['dinic_nv_mean']:.4f}",
                "dinic_nv_std_ms": f"{r['dinic_nv_std']:.4f}",
                "repeats": r["repeats"],
            })

    # 绘图
    if results:
        plot(results)

    print(f"\n结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
