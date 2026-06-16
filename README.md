# 实验六 最大流应用问题：棒球赛淘汰

本目录存放实验六的代码、数据、实验结果和图表。

## 目录结构

```text
ex6/
  baseball_elimination_experiment.py  # 主实验脚本
  baseball/                           # Princeton 官方 baseball 数据集
  data/                               # 兜底样例数据目录
  results/                            # CSV 实验结果
  figures/                            # Python 绘制图表
```

## 数据格式

数据文件采用 Princeton Baseball Elimination 的文本格式：

```text
n
TeamName wins losses remaining g_0 g_1 ... g_{n-1}
...
```

其中 `g_j` 表示该球队与第 `j` 支球队之间剩余比赛数量。

当前实验默认读取 `ex6/baseball/` 下的 Princeton 官方数据集。该目录中共有 23 个
`teams*.txt` 数据文件，规模从 1 支队伍到 60 支队伍不等：

```text
teams1.txt, teams4.txt, teams4a.txt, teams4b.txt,
teams5.txt, teams5a.txt, teams5b.txt, teams5c.txt,
teams7.txt, teams8.txt, teams10.txt,
teams12.txt, teams12-allgames.txt,
teams24.txt, teams29.txt, teams30.txt, teams32.txt,
teams36.txt, teams42.txt, teams48.txt, teams50.txt,
teams54.txt, teams60.txt
```

若 `ex6/baseball/` 不存在，脚本会退回 `ex6/data/`，并在该目录中补充
`teams4.txt` 和 `teams5.txt` 两个基础样例。

## 运行方法

运行全部实验：

```bash
python3 ex6/baseball_elimination_experiment.py all
```

显式指定数据目录：

```bash
python3 ex6/baseball_elimination_experiment.py all --data-dir ex6/baseball
```

只运行四队样例：

```bash
python3 ex6/baseball_elimination_experiment.py sample
```

只运行批量正确性测试：

```bash
python3 ex6/baseball_elimination_experiment.py correctness
```

只运行 Edmonds-Karp 与 Dinic 时间对比：

```bash
python3 ex6/baseball_elimination_experiment.py runtime --repeats 5
```

运行时间对比默认只纳入 `N <= 12` 的数据集，避免 Edmonds-Karp 在大规模数据上
产生过长等待；可通过 `--runtime-max-teams` 调整。

只运行建图剪枝规模测试：

```bash
python3 ex6/baseball_elimination_experiment.py scale --repeats 5
```

只根据已有 CSV 重新绘图：

```bash
python3 ex6/baseball_elimination_experiment.py plot
```

## 输出文件

```text
results/four_team_result.csv
results/princeton_correctness.csv
results/team_level_results.csv
results/maxflow_runtime.csv
results/network_scale.csv
figures/elimination_type_count.png
figures/maxflow_runtime_comparison.png
figures/network_scale_comparison.png
figures/elimination_type_count_academic.png
figures/elimination_type_count_academic.pdf
figures/maxflow_runtime_comparison_academic.png
figures/maxflow_runtime_comparison_academic.pdf
```

这些文件对应实验报告第四章中的“实验结果”表格和 Python 绘图结果。
