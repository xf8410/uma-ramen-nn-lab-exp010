#!/usr/bin/env python3
"""EXP-012 v2：从 EXP-009 现成标签里算「教师自己的闭环分」——零采集、零训练、只读数据。

v1 失败原因（run 33896281219，34 秒红）：index.npy 是**全局样本号**（157871 个唯一值 =
样本数），不是局号，无法按它分组。v1 的防御断言正确地拒绝了假分组，没有给假数。

v2 改用 npy_v3/turn.npy（实测 shape=(157871,) int16）：
  最终分随回合累积 ⇒ 每局**最早回合**那条样本的 scoreMean = 教师对该局最终分的事前估计。
  不需要知道局边界，只要按 turn 取值即可，且天然免疫"每局几条样本"的问题。

口径为什么可比（v1 里我担心的那点已被 meta.json 消掉）：
  npy_v3/meta.json 实载 plan_count=525、sampling_space_hash=6c30529e9333fb94、
  git_commit=d27a6ebd —— 与 ramen_space_bench 的 525 计划**同一采样空间、同一 pin**。
  所以本数与 65438.2 / 66734.8 同口径可比，不是"另一世界的自估分"。

诚实边界：
  scoreMean 是教师**自估**（搜索一步 + 按采集 rollout 策略走完，cross-fit 去乐观偏差），
  不是真实终局分。它与 bench 实测同空间，但一个是"教师认为能考多少"，一个是"实际考了多少"。
  两者之差本身就是有用读数（教师对自己高估还是低估）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HW_DEFAULT = 65438.2
HW_CHAMPION = 65554.2
STUDENT_RECORD = 66734.8
TARGET = 70000.0

STAGE_NAMES = {0: "RamenSelect", 1: "Begin/其他", 2: "Train", 3: "SuperRamenSelect", 4: "RegionSelect"}


def die(msg: str) -> None:
    print(f"\n❌ {msg}")
    sys.exit(1)


def find(root: Path, *parts: str) -> Path | None:
    for p in root.rglob("*.npy"):
        if p.name in parts:
            return p
    return None


def stats(name: str, v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f"{name}: 无有限值"
    q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return (f"{name}: n={v.size} 均值={v.mean():.1f} sd={v.std(ddof=1):.1f} "
            f"min={v.min():.0f} max={v.max():.0f} | p5={q[1]:.0f} 中位={q[3]:.0f} "
            f"p95={q[5]:.0f}  (p1={q[0]:.0f} p25={q[2]:.0f} p75={q[4]:.0f} p99={q[6]:.0f})")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    if not root.is_dir():
        die(f"目录不存在: {root}")

    print("=== 目录下的 .npy ===")
    for p in sorted(root.rglob("*.npy")):
        a = np.load(p, mmap_mode="r")
        print(f"  {p.relative_to(root)}  shape={a.shape} dtype={a.dtype}")

    meta = next(root.rglob("meta.json"), None)
    if meta:
        print("\n=== npy_v3/meta.json（口径证据）===")
        print(json.dumps(json.loads(meta.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)[:1500])
    ljson = next(root.rglob("labels.json"), None)
    if ljson:
        d = json.loads(ljson.read_text(encoding="utf-8"))
        print("\n=== labels_v3/labels.json 关键字段 ===")
        print(f"  rollouts={d.get('rollouts')} bootstrap_draws={d.get('config',{}).get('bootstrap_draws')} "
              f"radical_factor={d.get('config',{}).get('radical_factor')}")
        print(f"  value_mean={d.get('value_mean')}")
        for k, v in (d.get("stages") or {}).items():
            print(f"  stage {k}({STAGE_NAMES.get(int(k),'?')}): samples={v['samples']} "
                  f"entropy={v['policy_entropy_mean']:.3f} selector_stability={v['selector_stability_mean']:.4f}")

    vp = find(root, "value_target.npy")
    tp = find(root, "turn.npy")
    if vp is None:
        die("缺 value_target.npy")
    if tp is None:
        die("缺 turn.npy —— v2 的分组依据，没有它退回 v1 的困境")

    value = np.load(vp, mmap_mode="r")
    if value.ndim != 2 or value.shape[1] != 3:
        die(f"value_target 形状异常 {value.shape}，期望 (N,3)")
    sm = np.asarray(value[:, 0], dtype=np.float64)
    sd = np.asarray(value[:, 1], dtype=np.float64)
    v2 = np.asarray(value[:, 2], dtype=np.float64)
    turn = np.asarray(np.load(tp, mmap_mode="r")).ravel().astype(np.int64)
    n = sm.size
    if turn.size != n:
        die(f"turn 长度 {turn.size} 与样本数 {n} 不符")

    sp = find(root, "stage.npy")
    stage = np.asarray(np.load(sp, mmap_mode="r")).ravel().astype(np.int64) if sp else None

    print(f"\n样本数={n}  turn 范围=[{turn.min()}, {turn.max()}]  不同 turn 数={len(np.unique(turn))}")

    # ---------- 主曲线：scoreMean 随 turn 的增长 ----------
    print("\n" + "=" * 78)
    print("教师自估最终分随回合的增长（每 turn 的 scoreMean 均值）")
    print("=" * 78)
    rows = []
    for t in np.unique(turn):
        m = turn == t
        rows.append((int(t), int(m.sum()), float(sm[m].mean())))
    print(f"{'turn':>5} {'样本数':>8} {'scoreMean均值':>14}")
    shown = rows if len(rows) <= 90 else rows[:: max(1, len(rows) // 45)]
    for t, c, m in shown:
        print(f"{t:>5} {c:>8} {m:>14.1f}")

    # ---------- 主答案 ----------
    print("\n" + "=" * 78)
    print("教师闭环分（三个口径，互相校验）")
    print("=" * 78)
    t0 = int(turn.min())
    first = sm[turn == t0]
    early = sm[turn <= t0 + 1]
    last = sm[turn == int(turn.max())]
    lines = [
        stats(f"A 最早回合 turn={t0}（教师事前估计 = 主答案）", first),
        stats(f"B turn<={t0+1}（放宽一回合，扩样本降方差）", early),
        stats(f"C turn={int(turn.max())}（终局前一刻，应≈真实终局分）", last),
        stats("D 全体样本（被早期回合拉低，仅对照）", sm),
    ]
    for ln in lines:
        print("  " + ln)
    if stage is not None:
        print("\n  按阶段看最早回合样本的构成：")
        for s in np.unique(stage[turn == t0]):
            m = (turn == t0) & (stage == s)
            print(f"    stage {int(s)}({STAGE_NAMES.get(int(s),'?')}): n={int(m.sum())} "
                  f"scoreMean均值={sm[m].mean():.1f}")

    teacher = float(first.mean()) if first.size else float(early.mean())
    teacher_last = float(last.mean()) if last.size else float("nan")
    n_eps = int(first.size)

    print("\n" + "=" * 78)
    print("天花板判定")
    print("=" * 78)
    verdict = (
        f"教师事前估计（口径 A，n={n_eps} 个最早回合样本）= {teacher:.1f}\n"
        f"教师终局前估计（口径 C）= {teacher_last:.1f}\n"
        f"对照：手写默认 {HW_DEFAULT} / 手写冠军 {HW_CHAMPION} / 学生纪录 {STUDENT_RECORD} / 目标 {TARGET:.0f}\n"
        f"教师 − 学生 = {teacher - STUDENT_RECORD:+.1f}    教师 − 手写默认 = {teacher - HW_DEFAULT:+.1f}\n"
    )
    if teacher > STUDENT_RECORD + 500:
        verdict += (f"→ 教师明显强于学生（+{teacher - STUDENT_RECORD:.0f}）：标签里还有大量分数未兑现，"
                    f"训练侧（011 那类）确有肉，且值得继续把教师优势榨进网络。")
    elif teacher > STUDENT_RECORD - 500:
        verdict += ("→ 教师与学生基本持平：学生已贴顶，这批标签榨干。训练侧调参是抠零头，"
                    "要上 70000 必须换更强的教师（更强 rollout / 更深搜索 / 修模拟器后重采）。")
    elif teacher > HW_DEFAULT:
        verdict += (f"→ 教师自估低于学生但仍高于手写默认：学生已超过教师的自估水平。"
                    f"要么教师自估偏保守（看口径 C 是否更高），要么学生学到了搜索之外的东西。"
                    f"训练侧余量存疑。")
    else:
        verdict += (f"→ 教师自估低于手写默认 {HW_DEFAULT}：「教师严格强于基线」这个蒸馏前提**不成立**，"
                    f"整条蒸馏线的地基有问题，011 不必再跑。")
    print(verdict)

    Path("SUMMARY.txt").write_text(
        "EXP-012 v2 教师闭环分（只读 EXP-009 标签，零采集零训练）\n"
        f"样本数={n} turn范围=[{turn.min()},{turn.max()}] 最早回合样本数={n_eps}\n\n"
        + "\n".join(lines) + "\n\n" + verdict + "\n",
        encoding="utf-8",
    )
    print("\nSUMMARY.txt 已写盘")
    return 0


if __name__ == "__main__":
    sys.exit(main())
