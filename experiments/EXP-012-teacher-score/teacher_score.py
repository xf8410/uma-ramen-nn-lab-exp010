#!/usr/bin/env python3
"""EXP-012：从 EXP-009 现成标签里算「教师自己的闭环分」——零采集、零训练、只读数据。

背景（为什么这个数要命）
    学生（NN）是背教师的动作长大的，纪录 66734.8；但**教师自己从没进过考场**。
    若教师 ≈ 学生 → 这批标签已榨干，70000 在这批标签上不可能达到，训练侧全是抠零头。
    若教师 >> 学生 → 分数在拟合端，训练侧才有肉。

从哪读
    crates/umasim/src/training_sample.rs：value_target 是 3 维
        [scoreMean, scoreStdev, value]
    其中 scoreMean = 教师在该局面下、经蒙特卡洛搜索后对**最终育成分**的估计。

怎么算（关键口径）
    最终分是随回合累积的，所以「全体样本的 scoreMean 均值」会被早期回合拉低，
    **不能**当教师分（EXP-009 归一化 center[0]=60993.8 就是这个被拉低的全体均值）。
    正确量 = 按局分组后：
      · 每局**第一个**样本（最早回合）的 scoreMean → 教师对该局最终分的**事前估计**
        对全体局求平均 = 教师的闭环水平（本实验的主答案）
      · 每局**最后一个**样本的 scoreMean → 终局前一刻的估计（应接近真实终局分，做交叉校验）

诚实声明
    采集的 index 空间与 bench 的 525 计划×8 局**未必同分布**，故本数是「教师在其自身
    采集分布上的自估分」，与 66734.8 同量级可比、但非严格同口径。脚本会打印每局样本数
    分布与 scoreMean 的局内单调性，供判断口径是否成立。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# 对照锚点（同仓已实查的闭环分）
HW_DEFAULT = 65438.2
HW_CHAMPION = 65554.2
STUDENT_RECORD = 66734.8  # EXP-009b seed20260831，全仓纪录
TARGET = 70000.0


def die(msg: str) -> None:
    print(f"\n❌ {msg}")
    sys.exit(1)


def list_npy(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*.npy") if p.is_file())
    print(f"=== 目录 {root} 下的 .npy 文件（共 {len(files)} 个）===")
    for p in files:
        try:
            a = np.load(p, mmap_mode="r")
            print(f"  {p.relative_to(root)}  shape={a.shape} dtype={a.dtype} "
                  f"size={p.stat().st_size/1e6:.1f}MB")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.relative_to(root)}  读取失败: {e}")
    if not files:
        die("一个 .npy 都没找到，artifact 结构与预期不符")
    return files


def find_json_files(root: Path) -> None:
    print("\n=== 随包 JSON（溯源用：search_n / 区间 / 配方）===")
    found = False
    for p in sorted(root.rglob("*.json"))[:20]:
        found = True
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.name}: 读不动 {e}")
            continue
        print(f"  --- {p.relative_to(root)} ({len(txt)} 字符) ---")
        print("  " + txt[:1200].replace("\n", "\n  "))
    if not found:
        print("  （无 JSON）")


def pick(files: list[Path], *keywords: str) -> Path | None:
    """按文件名关键词挑一个 .npy（后面的关键词优先度更低）。"""
    for kw in keywords:
        for p in files:
            if kw in p.name.lower():
                return p
    return None


def group_positions(ep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按局分组，返回 (每局首样本位置, 每局末样本位置, 每局样本数)。

    用 stable argsort，保证局内顺序与原始写入顺序一致（局内应按回合递增）。
    """
    order = np.argsort(ep, kind="stable")
    s = ep[order]
    change = np.concatenate(([True], s[1:] != s[:-1]))
    first_pos = order[change]
    uniq_ids = s[change]
    change_next = np.concatenate((s[1:] != s[:-1], [True]))
    last_pos = order[change_next]
    counts = np.diff(np.concatenate((first_pos, [len(ep)]))) if len(first_pos) else np.array([])
    # 上面 counts 依赖「局在数组里连续」，不成立时用 bincount 兜底
    if counts.size != uniq_ids.size or counts.min() <= 0:
        _, inv = np.unique(ep, return_inverse=True)
        counts = np.bincount(inv)
    return first_pos, last_pos, counts


def stats(name: str, v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f"{name}: 全非有限值"
    q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return (f"{name}: n={v.size} 均值={v.mean():.1f} 标准差={v.std(ddof=1):.1f} "
            f"min={v.min():.1f} max={v.max():.1f} | "
            f"p1={q[0]:.0f} p5={q[1]:.0f} p25={q[2]:.0f} 中位={q[3]:.0f} "
            f"p75={q[4]:.0f} p95={q[5]:.0f} p99={q[6]:.0f}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    if not root.is_dir():
        die(f"目录不存在: {root}")

    files = list_npy(root)
    find_json_files(root)

    # ---------- 1. 找 value_target（3 维：scoreMean, scoreStdev, value）----------
    vpath = pick(files, "value")
    if vpath is None:
        die("没找到名字含 value 的 .npy —— 标签里没有 value_target，本实验无法进行")
    value = np.load(vpath, mmap_mode="r")
    print(f"\nvalue_target = {vpath.name}  shape={value.shape}")
    if value.ndim != 2:
        die(f"value_target 维度异常: {value.ndim}，期望 2")
    if value.shape[1] != 3:
        if value.shape[0] == 3:
            print("  检测到转置 (3,N)，已转置")
            value = np.ascontiguousarray(value.T)
        else:
            die(f"value_target 最后一维={value.shape[1]}，期望 3 [scoreMean,scoreStdev,value]")
    n = value.shape[0]

    score_mean = np.asarray(value[:, 0], dtype=np.float64)
    score_stdev = np.asarray(value[:, 1], dtype=np.float64)
    value_col = np.asarray(value[:, 2], dtype=np.float64)

    # ---------- 2. 找分组键 ----------
    ipath = pick(files, "index", "episode", "game", "position")
    ep = None
    if ipath is not None:
        cand = np.load(ipath, mmap_mode="r")
        print(f"\n候选分组键 = {ipath.name}  shape={cand.shape} dtype={cand.dtype}")
        if cand.ndim == 2 and cand.shape[1] == 1:
            cand = cand.ravel()
        if cand.ndim == 1 and cand.shape[0] == n:
            ep = np.asarray(cand).ravel()
        else:
            print(f"  长度 {cand.shape} 与样本数 {n} 不符 → 不用它分组")
    if ep is None:
        die("找不到可用的按局分组键（index.npy 缺失或长度不匹配）")

    uniq = np.unique(ep)
    print(f"  唯一局数 = {uniq.size}，样本数 = {n}")
    if uniq.size == n:
        die(f"分组键每个样本一个唯一值（像是全局样本号而非局号）→ 无法按局分组，"
            f"需要真正的 episode id 或 turn 数组")

    first_pos, last_pos, counts = group_positions(ep)
    print(f"  每局样本数: 均值={counts.mean():.1f} min={counts.min()} max={counts.max()} "
          f"（TOTAL_TURN=78，含多阶段则 >78 属正常）")

    # ---------- 3. 主答案 ----------
    print("\n" + "=" * 78)
    print("教师闭环分（scoreMean 口径）")
    print("=" * 78)
    first = score_mean[first_pos]
    last = score_mean[last_pos]
    lines = [
        stats("① 每局首样本 scoreMean（教师事前估计 = 主答案）", first),
        stats("② 每局末样本 scoreMean（终局前一刻，交叉校验）", last),
        stats("③ 全体样本 scoreMean（会被早期回合拉低，仅对照）", score_mean),
        stats("④ 每局首样本 scoreStdev（教师自估波动）", score_stdev[first_pos]),
        stats("⑤ 每局首样本 value 列（第三维，含义待核）", value_col[first_pos]),
    ]
    for ln in lines:
        print("  " + ln)

    # 局内单调性：scoreMean 应随回合递增（分数在累积）
    mono = 0
    tot = 0
    for f, l in zip(first_pos, last_pos, strict=False):
        if l > f:
            seg = score_mean[f:l + 1]
            tot += 1
            if np.all(np.diff(seg) >= -1.0):
                mono += 1
    if tot:
        print(f"  局内 scoreMean 单调不降的局占比 = {mono}/{tot} = {mono/tot:.1%}"
              f"（高 → 分组与回合顺序成立；低 → 口径可疑，结论要打折）")

    teacher = float(np.mean(first[np.isfinite(first)]))
    teacher_hi = float(np.mean(last[np.isfinite(last)]))

    # ---------- 4. 判定 ----------
    print("\n" + "=" * 78)
    print("天花板判定")
    print("=" * 78)
    verdict = (
        f"教师事前估计（首样本）= {teacher:.1f}\n"
        f"教师终局前估计（末样本）= {teacher_hi:.1f}\n"
        f"对照：手写默认 {HW_DEFAULT} / 手写冠军 {HW_CHAMPION} / "
        f"学生纪录 {STUDENT_RECORD} / 目标 {TARGET:.0f}\n"
    )
    if teacher > STUDENT_RECORD + 500:
        verdict += (f"→ 教师明显强于学生（+{teacher - STUDENT_RECORD:.0f}）："
                    f"标签里还有大量分数未兑现，训练侧（011 那类）确有肉；"
                    f"但更该做的是把教师自己的分先测实。")
    elif teacher > STUDENT_RECORD - 500:
        verdict += (f"→ 教师与学生基本持平（Δ={teacher - STUDENT_RECORD:+.0f}）："
                    f"学生已贴顶，这批标签榨干了。训练侧调参是抠零头，"
                    f"要上 70000 必须换教师（把 search_n 从 64 拉回 1024 重采）。")
    elif teacher > HW_DEFAULT:
        verdict += (f"→ 教师弱于学生（Δ={teacher - STUDENT_RECORD:+.0f}）但仍强于手写默认："
                    f"学生已超过教师的自估水平，说明标签里真正被学到的是动作而非分数；"
                    f"训练侧余量存疑，优先换教师。")
    else:
        verdict += (f"→ 教师自估分低于手写默认 {HW_DEFAULT}："
                    f"「教师严格强于基线」这个蒸馏前提**根本不成立**，"
                    f"整条蒸馏线的地基有问题，011 不必再跑。")
    print(verdict)

    out = Path("SUMMARY.txt")
    out.write_text(
        "EXP-012 教师闭环分（零采集零训练，只读 EXP-009 标签）\n"
        f"样本数={n} 局数={uniq.size} 每局样本数均值={counts.mean():.1f}\n\n"
        + "\n".join(lines) + "\n\n" + verdict + "\n",
        encoding="utf-8",
    )
    print(f"\nSUMMARY.txt 已写盘（{out.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
