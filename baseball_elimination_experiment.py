#!/usr/bin/env python3
"""Baseball elimination experiment suite for Algorithm Design Lab 6.

The script implements baseball elimination with two max-flow algorithms:
Edmonds-Karp and Dinic. It can run the four-team sample, batch-test data files
under ex6/baseball, compare runtime, and export CSV/figure artifacts for the lab
report.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OFFICIAL_DATA_DIR = ROOT / "baseball"
FALLBACK_DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

BLUE = "#1f4e79"
LIGHT_BLUE = "#7db7e8"
ORANGE = "#f59e0b"
GREEN = "#2e7d32"
RED = "#b22222"
PURPLE = "#5b4b8a"


@dataclass(frozen=True)
class BaseballData:
    names: list[str]
    wins: list[int]
    losses: list[int]
    remaining: list[int]
    games: list[list[int]]
    dataset: str

    @property
    def n(self) -> int:
        return len(self.names)


@dataclass
class FlowEdge:
    to: int
    rev: int
    cap: int
    original_cap: int


class FlowNetwork:
    def __init__(self, n: int) -> None:
        self.graph: list[list[FlowEdge]] = [[] for _ in range(n)]

    @property
    def n(self) -> int:
        return len(self.graph)

    @property
    def edge_count(self) -> int:
        return sum(len(adj) for adj in self.graph) // 2

    def add_edge(self, u: int, v: int, cap: int) -> None:
        if cap < 0:
            raise ValueError(f"negative capacity on edge {(u, v)}: {cap}")
        forward = FlowEdge(v, len(self.graph[v]), cap, cap)
        backward = FlowEdge(u, len(self.graph[u]), 0, 0)
        self.graph[u].append(forward)
        self.graph[v].append(backward)


@dataclass
class BuildResult:
    network: FlowNetwork
    source: int
    sink: int
    total_game_capacity: int
    team_node: dict[int, int]
    game_nodes: int
    team_nodes: int
    pruned_zero_games: bool


@dataclass
class EliminationResult:
    dataset: str
    team: str
    algorithm: str
    eliminated: bool
    elimination_type: str
    certificate: tuple[str, ...]
    max_wins: int
    total_game_capacity: int
    maxflow: int
    full_flow: bool | None
    game_nodes: int
    team_nodes: int
    flow_nodes: int
    flow_edges: int
    runtime_ms: float


def read_baseball_file(path: Path) -> BaseballData:
    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    n = int(lines[0])
    if len(lines) < n + 1:
        raise ValueError(f"{path} ended early: expected {n} team lines")

    names: list[str] = []
    wins: list[int] = []
    losses: list[int] = []
    remaining: list[int] = []
    games: list[list[int]] = []
    for idx in range(n):
        parts = lines[idx + 1].split()
        if len(parts) != 4 + n:
            raise ValueError(f"{path}:{idx + 2} expected {4+n} fields, got {len(parts)}")
        names.append(parts[0])
        wins.append(int(parts[1]))
        losses.append(int(parts[2]))
        remaining.append(int(parts[3]))
        games.append([int(x) for x in parts[4:]])

    return BaseballData(names, wins, losses, remaining, games, path.name)


def write_baseball_file(path: Path, data: BaseballData) -> None:
    lines = [str(data.n)]
    for i, name in enumerate(data.names):
        row = [name, str(data.wins[i]), str(data.losses[i]), str(data.remaining[i])]
        row.extend(str(x) for x in data.games[i])
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def natural_dataset_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**9, path.name)


def default_data_dir() -> Path:
    official_files = list(OFFICIAL_DATA_DIR.glob("teams*.txt"))
    if official_files:
        return OFFICIAL_DATA_DIR
    return FALLBACK_DATA_DIR


def ensure_sample_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    teams4 = BaseballData(
        names=["Atlanta", "Philadelphia", "New_York", "Montreal"],
        wins=[83, 80, 78, 77],
        losses=[71, 79, 78, 82],
        remaining=[8, 3, 6, 3],
        games=[
            [0, 1, 6, 1],
            [1, 0, 0, 2],
            [6, 0, 0, 0],
            [1, 2, 0, 0],
        ],
        dataset="teams4.txt",
    )
    teams4_path = data_dir / "teams4.txt"
    if not teams4_path.exists():
        write_baseball_file(teams4_path, teams4)

    teams5 = BaseballData(
        names=["New_York", "Baltimore", "Boston", "Toronto", "Detroit"],
        wins=[75, 71, 69, 63, 49],
        losses=[59, 63, 66, 72, 86],
        remaining=[28, 28, 27, 27, 27],
        games=[
            [0, 3, 8, 7, 3],
            [3, 0, 2, 7, 7],
            [8, 2, 0, 0, 3],
            [7, 7, 0, 0, 3],
            [3, 7, 3, 3, 0],
        ],
        dataset="teams5.txt",
    )
    teams5_path = data_dir / "teams5.txt"
    if not teams5_path.exists():
        write_baseball_file(teams5_path, teams5)


def build_network(data: BaseballData, target: int, prune_zero_games: bool = True) -> BuildResult:
    n = data.n
    source = 0
    next_node = 1
    team_node: dict[int, int] = {}
    for i in range(n):
        if i == target:
            continue
        team_node[i] = next_node
        next_node += 1

    game_pairs: list[tuple[int, int]] = []
    for i in range(n):
        if i == target:
            continue
        for j in range(i + 1, n):
            if j == target:
                continue
            if prune_zero_games and data.games[i][j] == 0:
                continue
            game_pairs.append((i, j))

    game_node: dict[tuple[int, int], int] = {}
    for pair in game_pairs:
        game_node[pair] = next_node
        next_node += 1

    sink = next_node
    network = FlowNetwork(sink + 1)
    max_wins = data.wins[target] + data.remaining[target]
    total_game_capacity = 0
    infinite = max(sum(sum(row) for row in data.games), 1)

    for i, j in game_pairs:
        games_left = data.games[i][j]
        node = game_node[(i, j)]
        network.add_edge(source, node, games_left)
        network.add_edge(node, team_node[i], infinite)
        network.add_edge(node, team_node[j], infinite)
        total_game_capacity += games_left

    for i, node in team_node.items():
        network.add_edge(node, sink, max_wins - data.wins[i])

    return BuildResult(
        network=network,
        source=source,
        sink=sink,
        total_game_capacity=total_game_capacity,
        team_node=team_node,
        game_nodes=len(game_pairs),
        team_nodes=len(team_node),
        pruned_zero_games=prune_zero_games,
    )


def edmonds_karp(network: FlowNetwork, source: int, sink: int) -> int:
    maxflow = 0
    n = network.n
    while True:
        parent: list[tuple[int, int] | None] = [None] * n
        parent[source] = (source, -1)
        q = deque([source])
        while q and parent[sink] is None:
            u = q.popleft()
            for idx, edge in enumerate(network.graph[u]):
                if edge.cap > 0 and parent[edge.to] is None:
                    parent[edge.to] = (u, idx)
                    q.append(edge.to)
                    if edge.to == sink:
                        break
        if parent[sink] is None:
            break

        bottleneck = math.inf
        v = sink
        while v != source:
            u, idx = parent[v]  # type: ignore[misc]
            bottleneck = min(bottleneck, network.graph[u][idx].cap)
            v = u

        delta = int(bottleneck)
        v = sink
        while v != source:
            u, idx = parent[v]  # type: ignore[misc]
            edge = network.graph[u][idx]
            edge.cap -= delta
            network.graph[v][edge.rev].cap += delta
            v = u
        maxflow += delta

    return maxflow


def dinic(network: FlowNetwork, source: int, sink: int) -> int:
    maxflow = 0
    n = network.n

    def bfs() -> list[int]:
        level = [-1] * n
        level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for edge in network.graph[u]:
                if edge.cap > 0 and level[edge.to] < 0:
                    level[edge.to] = level[u] + 1
                    q.append(edge.to)
        return level

    def dfs(u: int, pushed: int, level: list[int], it: list[int]) -> int:
        if u == sink:
            return pushed
        while it[u] < len(network.graph[u]):
            idx = it[u]
            edge = network.graph[u][idx]
            if edge.cap > 0 and level[edge.to] == level[u] + 1:
                flow = dfs(edge.to, min(pushed, edge.cap), level, it)
                if flow:
                    edge.cap -= flow
                    network.graph[edge.to][edge.rev].cap += flow
                    return flow
            it[u] += 1
        return 0

    while True:
        level = bfs()
        if level[sink] < 0:
            break
        it = [0] * n
        while True:
            pushed = dfs(source, 10**18, level, it)
            if not pushed:
                break
            maxflow += pushed

    return maxflow


def dinic_naive(network: FlowNetwork, source: int, sink: int) -> int:
    """Dinic without current-arc optimization — DFS always scans from list head."""
    maxflow = 0
    n = network.n

    def bfs() -> list[int]:
        level = [-1] * n
        level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for edge in network.graph[u]:
                if edge.cap > 0 and level[edge.to] < 0:
                    level[edge.to] = level[u] + 1
                    q.append(edge.to)
        return level

    def dfs(u: int, pushed: int, level: list[int]) -> int:
        if u == sink:
            return pushed
        for edge in network.graph[u]:
            if edge.cap > 0 and level[edge.to] == level[u] + 1:
                flow = dfs(edge.to, min(pushed, edge.cap), level)
                if flow:
                    edge.cap -= flow
                    network.graph[edge.to][edge.rev].cap += flow
                    return flow
        return 0

    while True:
        level = bfs()
        if level[sink] < 0:
            break
        while True:
            pushed = dfs(source, 10**18, level)
            if not pushed:
                break
            maxflow += pushed

    return maxflow


def reachable_nodes(network: FlowNetwork, source: int) -> set[int]:
    seen = {source}
    q = deque([source])
    while q:
        u = q.popleft()
        for edge in network.graph[u]:
            if edge.cap > 0 and edge.to not in seen:
                seen.add(edge.to)
                q.append(edge.to)
    return seen


def copy_network(network: FlowNetwork) -> FlowNetwork:
    clone = FlowNetwork(network.n)
    for u, adj in enumerate(network.graph):
        clone.graph[u] = [FlowEdge(e.to, e.rev, e.cap, e.original_cap) for e in adj]
    return clone


def classify_team(
    data: BaseballData,
    target: int,
    algorithm: str = "dinic",
    prune_zero_games: bool = True,
) -> EliminationResult:
    start = time.perf_counter()
    max_wins = data.wins[target] + data.remaining[target]

    trivial_cert = [data.names[i] for i in range(data.n) if i != target and data.wins[i] > max_wins]
    if trivial_cert:
        runtime_ms = (time.perf_counter() - start) * 1000
        return EliminationResult(
            dataset=data.dataset,
            team=data.names[target],
            algorithm=algorithm,
            eliminated=True,
            elimination_type="trivial",
            certificate=tuple(trivial_cert),
            max_wins=max_wins,
            total_game_capacity=0,
            maxflow=0,
            full_flow=None,
            game_nodes=0,
            team_nodes=data.n - 1,
            flow_nodes=0,
            flow_edges=0,
            runtime_ms=runtime_ms,
        )

    build = build_network(data, target, prune_zero_games=prune_zero_games)
    network = build.network
    if algorithm == "edmonds_karp":
        flow = edmonds_karp(network, build.source, build.sink)
    elif algorithm == "dinic":
        flow = dinic(network, build.source, build.sink)
    elif algorithm == "dinic_naive":
        flow = dinic_naive(network, build.source, build.sink)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    full_flow = flow == build.total_game_capacity
    eliminated = not full_flow
    certificate: tuple[str, ...] = ()
    if eliminated:
        reachable = reachable_nodes(network, build.source)
        certificate = tuple(
            data.names[i]
            for i, node in build.team_node.items()
            if node in reachable
        )
        if not certificate:
            certificate = ("nontrivial_min_cut",)

    runtime_ms = (time.perf_counter() - start) * 1000
    return EliminationResult(
        dataset=data.dataset,
        team=data.names[target],
        algorithm=algorithm,
        eliminated=eliminated,
        elimination_type="nontrivial" if eliminated else "none",
        certificate=certificate,
        max_wins=max_wins,
        total_game_capacity=build.total_game_capacity,
        maxflow=flow,
        full_flow=full_flow,
        game_nodes=build.game_nodes,
        team_nodes=build.team_nodes,
        flow_nodes=network.n,
        flow_edges=network.edge_count,
        runtime_ms=runtime_ms,
    )


def load_all_data(data_dir: Path) -> list[BaseballData]:
    if data_dir == FALLBACK_DATA_DIR:
        ensure_sample_data(data_dir)
    datasets = []
    for path in sorted(data_dir.glob("teams*.txt"), key=natural_dataset_key):
        datasets.append(read_baseball_file(path))
    if not datasets:
        raise FileNotFoundError(f"No teams*.txt datasets found in {data_dir}")
    return datasets


def result_row(result: EliminationResult) -> dict[str, object]:
    return {
        "dataset": result.dataset,
        "team": result.team,
        "algorithm": result.algorithm,
        "eliminated": result.eliminated,
        "elimination_type": result.elimination_type,
        "certificate": ",".join(result.certificate),
        "max_wins": result.max_wins,
        "total_game_capacity": result.total_game_capacity,
        "maxflow": result.maxflow,
        "full_flow": "" if result.full_flow is None else result.full_flow,
        "game_nodes": result.game_nodes,
        "team_nodes": result.team_nodes,
        "flow_nodes": result.flow_nodes,
        "flow_edges": result.flow_edges,
        "runtime_ms": f"{result.runtime_ms:.6f}",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_four_team(data_dir: Path) -> list[EliminationResult]:
    data = read_baseball_file(data_dir / "teams4.txt")
    rows = [classify_team(data, i, "dinic") for i in range(data.n)]
    write_csv(RESULTS_DIR / "four_team_result.csv", [result_row(r) for r in rows])
    return rows


def run_correctness(datasets: list[BaseballData]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []
    for data in datasets:
        dinic_results = [classify_team(data, i, "dinic") for i in range(data.n)]
        ek_results = [classify_team(data, i, "edmonds_karp") for i in range(data.n)]
        consistent = all(
            d.eliminated == e.eliminated and d.elimination_type == e.elimination_type
            for d, e in zip(dinic_results, ek_results)
        )
        eliminated = sum(r.eliminated for r in dinic_results)
        trivial = sum(r.elimination_type == "trivial" for r in dinic_results)
        nontrivial = sum(r.elimination_type == "nontrivial" for r in dinic_results)
        rows.append(
            {
                "dataset": data.dataset,
                "n": data.n,
                "eliminated": eliminated,
                "trivial": trivial,
                "nontrivial": nontrivial,
                "ek_consistent": consistent,
                "dinic_consistent": True,
                "reference_matched": "",
            }
        )
        detailed_rows.extend(result_row(r) for r in dinic_results)

    write_csv(RESULTS_DIR / "princeton_correctness.csv", rows)
    write_csv(RESULTS_DIR / "team_level_results.csv", detailed_rows)
    return rows


def run_runtime(datasets: list[BaseballData], repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for data in datasets:
        timings: dict[str, list[float]] = {"edmonds_karp": [], "dinic": []}
        consistency = True
        for algorithm in ("edmonds_karp", "dinic"):
            for _ in range(repeats):
                start = time.perf_counter()
                results = [classify_team(data, i, algorithm) for i in range(data.n)]
                timings[algorithm].append((time.perf_counter() - start) * 1000)
                if algorithm == "edmonds_karp":
                    ek_flags = [(r.eliminated, r.elimination_type) for r in results]
                else:
                    dinic_flags = [(r.eliminated, r.elimination_type) for r in results]
            if "ek_flags" in locals() and "dinic_flags" in locals():
                consistency = ek_flags == dinic_flags
        ek_avg = statistics.mean(timings["edmonds_karp"])
        dinic_avg = statistics.mean(timings["dinic"])
        ek_std = statistics.stdev(timings["edmonds_karp"]) if repeats > 1 else 0.0
        dinic_std = statistics.stdev(timings["dinic"]) if repeats > 1 else 0.0
        rows.append(
            {
                "dataset": data.dataset,
                "n": data.n,
                "edmonds_karp_ms": f"{ek_avg:.6f}",
                "edmonds_karp_std_ms": f"{ek_std:.6f}",
                "dinic_ms": f"{dinic_avg:.6f}",
                "dinic_std_ms": f"{dinic_std:.6f}",
                "speedup": f"{(ek_avg / dinic_avg) if dinic_avg else 0:.3f}",
                "result_consistent": consistency,
            }
        )
    write_csv(RESULTS_DIR / "maxflow_runtime.csv", rows)
    return rows


def run_network_scale(datasets: list[BaseballData], repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for data in datasets:
        for prune in (False, True):
            mode = "pruned" if prune else "original"
            game_nodes = 0
            flow_edges = 0
            timings: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                for target in range(data.n):
                    max_wins = data.wins[target] + data.remaining[target]
                    if any(data.wins[i] > max_wins for i in range(data.n) if i != target):
                        continue
                    build = build_network(data, target, prune_zero_games=prune)
                    game_nodes += build.game_nodes
                    flow_edges += build.network.edge_count
                    network = copy_network(build.network)
                    dinic(network, build.source, build.sink)
                timings.append((time.perf_counter() - start) * 1000)
            rows.append(
                {
                    "dataset": data.dataset,
                    "n": data.n,
                    "build_mode": mode,
                    "game_nodes": game_nodes // repeats,
                    "flow_edges": flow_edges // repeats,
                    "avg_time_ms": f"{statistics.mean(timings):.6f}",
                }
            )
    write_csv(RESULTS_DIR / "network_scale.csv", rows)
    return rows


def plot_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    correctness_path = RESULTS_DIR / "princeton_correctness.csv"
    if correctness_path.exists() and correctness_path.read_text(encoding="utf-8").strip():
        with correctness_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        labels = [r["dataset"] for r in rows]
        trivial = [int(r["trivial"]) for r in rows]
        nontrivial = [int(r["nontrivial"]) for r in rows]
        x = range(len(labels))
        plt.figure(figsize=(8, 4.8))
        plt.bar(x, trivial, label="Trivial", color=LIGHT_BLUE)
        plt.bar(x, nontrivial, bottom=trivial, label="Non-trivial", color=ORANGE)
        plt.xticks(list(x), labels, rotation=25, ha="right")
        plt.ylabel("Eliminated Teams")
        plt.title("Elimination Type Count by Dataset")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "elimination_type_count.png", dpi=180)
        plt.close()

    runtime_path = RESULTS_DIR / "maxflow_runtime.csv"
    if runtime_path.exists() and runtime_path.read_text(encoding="utf-8").strip():
        with runtime_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        labels = [r["dataset"] for r in rows]
        ek = [float(r["edmonds_karp_ms"]) for r in rows]
        dn = [float(r["dinic_ms"]) for r in rows]
        x = list(range(len(labels)))
        width = 0.36
        plt.figure(figsize=(8, 4.8))
        plt.bar([i - width / 2 for i in x], ek, width, label="Edmonds-Karp", color=BLUE)
        plt.bar([i + width / 2 for i in x], dn, width, label="Dinic", color=GREEN)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Average Runtime (ms)")
        plt.title("Max-Flow Runtime Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "maxflow_runtime_comparison.png", dpi=180)
        plt.close()

    scale_path = RESULTS_DIR / "network_scale.csv"
    if scale_path.exists() and scale_path.read_text(encoding="utf-8").strip():
        with scale_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        datasets = sorted({r["dataset"] for r in rows})
        modes = ["original", "pruned"]
        values = {
            mode: [int(next(r for r in rows if r["dataset"] == ds and r["build_mode"] == mode)["flow_edges"]) for ds in datasets]
            for mode in modes
        }
        x = list(range(len(datasets)))
        width = 0.36
        plt.figure(figsize=(8, 4.8))
        plt.bar([i - width / 2 for i in x], values["original"], width, label="Original", color=PURPLE)
        plt.bar([i + width / 2 for i in x], values["pruned"], width, label="Pruned", color=ORANGE)
        plt.xticks(x, datasets, rotation=25, ha="right")
        plt.ylabel("Flow Edges")
        plt.title("Network Scale Before and After Pruning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "network_scale_comparison.png", dpi=180)
        plt.close()


def run_all(data_dir: Path, repeats: int, runtime_max_teams: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_all_data(data_dir)
    runtime_datasets = [data for data in datasets if data.n <= runtime_max_teams]
    run_four_team(data_dir)
    run_correctness(datasets)
    run_runtime(runtime_datasets, repeats=repeats)
    run_network_scale(datasets, repeats=repeats)
    plot_figures()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseball elimination experiments.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "sample", "correctness", "runtime", "scale", "plot"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing Princeton-style teams*.txt files. Default: ex6/baseball if present, else ex6/data.",
    )
    parser.add_argument(
        "--runtime-max-teams",
        type=int,
        default=12,
        help="Largest team count included in Edmonds-Karp vs Dinic runtime comparison.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_all_data(args.data_dir)
    runtime_datasets = [data for data in datasets if data.n <= args.runtime_max_teams]

    if args.command == "all":
        run_all(args.data_dir, args.repeats, args.runtime_max_teams)
    elif args.command == "sample":
        run_four_team(args.data_dir)
    elif args.command == "correctness":
        run_correctness(datasets)
    elif args.command == "runtime":
        run_runtime(runtime_datasets, args.repeats)
    elif args.command == "scale":
        run_network_scale(datasets, args.repeats)
    elif args.command == "plot":
        plot_figures()


if __name__ == "__main__":
    main()
