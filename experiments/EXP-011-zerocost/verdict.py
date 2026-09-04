"""EXP-011 闭环配对裁决：各 NN 组 vs 手写默认 / 手写冠军 / 009 冠军控制组。

用法：``python3 verdict.py <artifact 根目录> <组标签> [<组标签> ...]``

口径与 EXP-004/004b/008b/009b/010b 完全一致：525 计划 × 8 局 = 4200 局/组，seed 基 61444，
race_shield on，special_mode=canonical；同 (计划, 局) 共享规则层随机数 → 逐行配对。

判据（PRINCIPLES §0.4）：**配对 t>2 且 Δ>0** 才算超过某基线。
锚点：
  - hw-default = 65438.2、hw-champion = 65554.2（Δ+116.1 t+5.91）——**硬校验**，
    EXP-010b 已在本仓本 patch 链逐位复现过，再漂移说明本轮口径与历史不可比，直接失败；
  - ctrl-009-0831 = 66734.8（全仓纪录）——**软校验**，它是在主仓跑的 bench，
    patch 链若有细微差异会整体平移，故只报偏差、不阻塞出表（对比仍以本轮实测控制组为准）。

⚠ 本脚本只认闭环评价分。regret 是「学生像不像教师」的抄写分，永不作为进步证据
（EXP-010 教训：regret 191.5→186.5 看着变好，闭环 63922 反比手写低 1516）。
"""

from __future__ import annotations

import csv
import glob
import math
import sys
from pathlib import Path

BASELINES = ("hw-default", "hw-champion", "ctrl-009-0831")
HARD_ANCHORS = ("hw-default", "hw-champion")
ANCHOR_MEANS = {"hw-default": 65438.2, "hw-champion": 65554.2, "ctrl-009-0831": 66734.8}
ANCHOR_TOLERANCE = 0.05
TARGET = 70000.0


def load(root: str, tag: str) -> list[dict[str, str]]:
    """读取某一组的 bench CSV。"""

    hits = glob.glob(f"{root}/EXP-011-bench-{tag}/**/exp011_{tag}.csv", recursive=True)
    if not hits:
        raise SystemExit(f"缺 {tag} 的 CSV（在 {root}/EXP-011-bench-{tag} 下递归查找无果）")
    with open(hits[0], newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stats(rows: list[dict[str, str]]):
    """返回 (逐局分数, n, 均分, 标准误)。"""

    scores = [float(row["score"]) for row in rows]
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((value - mean) ** 2 for value in scores) / (n - 1)
    return scores, n, mean, math.sqrt(variance) / math.sqrt(n)


def paired(deltas: list[float]):
    """配对差值的均值与 t 值。"""

    n = len(deltas)
    mean = sum(deltas) / n
    variance = sum((value - mean) ** 2 for value in deltas) / (n - 1)
    se = math.sqrt(variance) / math.sqrt(n)
    return mean, (mean / se if se > 0 else 0.0)


def verdict(delta: float, t_value: float) -> str:
    """§0.4 判据。"""

    return "超过 ✅" if (t_value > 2.0 and delta > 0.0) else "未超过"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = argv[0]
    tags = argv[1:]
    missing = [tag for tag in BASELINES if tag not in tags]
    if missing:
        raise SystemExit(f"缺基线组 {missing}，无法配对裁决")
    nn_tags = [tag for tag in tags if tag not in BASELINES]
    if not nn_tags:
        raise SystemExit("没有待裁决的 NN 组")

    lines: list[str] = []
    references: dict[str, tuple[list[float], list[str]]] = {}
    drift_failures: list[str] = []
    print("=== 基线（锚点复现校验）===")
    for tag in BASELINES:
        rows = load(root, tag)
        scores, n, mean, se = stats(rows)
        anchor = ANCHOR_MEANS[tag]
        drift = mean - anchor
        if abs(drift) <= ANCHOR_TOLERANCE:
            flag = "✅ 复现"
        elif tag in HARD_ANCHORS:
            flag = "❌ 硬锚点漂移"
            drift_failures.append(f"{tag}: 实测 {mean:.1f} vs 锚点 {anchor}")
        else:
            flag = "⚠ 软锚点漂移（不阻塞，以本轮实测为准）"
        fails = sum(row["free_race_ok"] == "0" for row in rows)
        print(f"{tag}: n={n} mean={mean:.1f} se={se:.1f} 自选未达标={fails}（锚点 {anchor} 偏差 {drift:+.1f} {flag}）")
        lines.append(f"{tag}: mean={mean:.1f} se={se:.1f} fail={fails} anchor={anchor} drift={drift:+.2f} {flag}")
        references[tag] = (scores, [row["seed"] for row in rows])
    if drift_failures:
        raise SystemExit("硬锚点未复现，本轮口径与历史不可比，先查环境：\n  " + "\n  ".join(drift_failures))

    base_scores, base_seeds = references["hw-default"]
    champion_scores, _ = references["hw-champion"]
    control_scores, _ = references["ctrl-009-0831"]
    delta, t_value = paired([a - b for a, b in zip(champion_scores, base_scores)])
    print(f"冠军 vs 默认：Δ={delta:+.1f} t={t_value:+.2f}（锚点应复现 +116.1 t+5.91）")
    lines.append(f"hw-champion vs hw-default: d={delta:+.1f} t={t_value:+.2f}")

    print(f"\n=== 待裁决组（{len(nn_tags)} 个，全部 vs 三基线逐行配对）===")
    print(f"{'组':<12} {'均分':>9} | {'Δ默认':>8} {'t':>7} | {'Δ冠军':>8} {'t':>7} | {'Δ控制':>8} {'t':>7} | 判定")
    results: list[tuple[str, float]] = []
    for tag in nn_tags:
        rows = load(root, tag)
        if len(rows) != len(base_scores):
            raise SystemExit(f"{tag}: 行数 {len(rows)} != {len(base_scores)}，配对失效")
        if [row["seed"] for row in rows] != base_seeds:
            raise SystemExit(f"{tag}: seed 列与基线不逐行一致，配对失效")
        scores, _, mean, se = stats(rows)
        results.append((tag, mean))
        fails = sum(row["free_race_ok"] == "0" for row in rows)
        d1, t1 = paired([a - b for a, b in zip(scores, base_scores)])
        d2, t2 = paired([a - b for a, b in zip(scores, champion_scores)])
        d3, t3 = paired([a - b for a, b in zip(scores, control_scores)])
        call = f"{verdict(d1, t1)}默认 / {verdict(d2, t2)}冠军 / {verdict(d3, t3)}控制"
        print(f"{tag:<12} {mean:>9.1f} | {d1:>+8.1f} {t1:>+7.2f} | {d2:>+8.1f} {t2:>+7.2f} | "
              f"{d3:>+8.1f} {t3:>+7.2f} | {call}（se={se:.1f} 自选未达标={fails}）")
        lines.append(f"{tag}: mean={mean:.1f} se={se:.1f} d_default={d1:+.1f} t={t1:+.2f} "
                     f"d_champ={d2:+.1f} t2={t2:+.2f} d_ctrl={d3:+.1f} t3={t3:+.2f} fail={fails} {call}")

    best_tag, best_mean = max(results, key=lambda item: item[1])
    control_mean = references["ctrl-009-0831"][0] and stats(load(root, "ctrl-009-0831"))[2]
    print(f"\n本轮最好组 = {best_tag} {best_mean:.1f}"
          f"（vs 本轮控制 {control_mean:.1f} = {best_mean - control_mean:+.1f}；距 {TARGET:.0f} = {TARGET - best_mean:+.1f}）")
    print("历史链：gen1 −5516~−9514 → gen2 −1139~−613 → gen3 0832 +231 → gen4(009b) 66734.8 纪录 "
          "→ gen5(自蒸馏 76k) 63922 判负 → 本轮=零采集训练侧（集成/EMA/更长日程）")
    lines.append(f"best={best_tag} mean={best_mean:.1f} ctrl={control_mean:.1f} "
                 f"d_ctrl={best_mean - control_mean:+.1f} gap_to_70000={TARGET - best_mean:+.1f}")

    Path("SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SUMMARY.txt 已写盘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
