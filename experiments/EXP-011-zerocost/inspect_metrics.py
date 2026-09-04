"""打印若干模型目录的训练摘要（轮数 / best_regret / 结构 / value 归一化），供 CI 日志核对事实。

用法：``python3 inspect_metrics.py <模型目录> [<模型目录> ...]``

存在的理由：EXP-011 的集成臂依赖两个**必须实查**的事实，不能靠记忆——
1. 成员模型的 value 归一化是否同坐标系（不同 split_seed → 不同 center/scale → 必须仿射校正）；
2. EXP-009 那三个模型到底训了多少轮（早停还是跑满 400 轮），决定「更长日程」这条臂值不值得排。

目录层级随 artifact 打包方式变化，故一律递归查找。只读不改。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _find(root: Path, name: str) -> Path | None:
    """在目录下（含任意层 artifact 前缀）找指定文件。"""

    direct = root / name
    if direct.is_file():
        return direct
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def _records(root: Path) -> list[dict]:
    """读 metrics.jsonl（每轮一行）。"""

    path = _find(root, "metrics.jsonl")
    if path is None:
        raise SystemExit(f"{root}: 缺 metrics.jsonl，无法核对训练事实")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _describe(root: Path) -> dict | None:
    """打印单个模型目录的摘要，返回其 value_normalization（供跨模型比对）。"""

    print(f"=== {root} ===")
    if not root.exists():
        raise SystemExit(f"目录不存在: {root}")

    records = _records(root)
    best = min(float(rec["best_regret"]) for rec in records)
    last = records[-1]
    last_regret = float(last["evaluation"]["overall"]["expected_regret"])
    print(f"  epochs_run={len(records)} best_regret={best:.2f} last_regret={last_regret:.2f} "
          f"last_step={last.get('global_step')} last_stale_epochs={last.get('stale_epochs')}")

    normalization = None
    run_path = _find(root, "run.json")
    if run_path is not None:
        run_info = json.loads(run_path.read_text(encoding="utf-8"))
        schedule = run_info.get("schedule", {})
        split = run_info.get("split", {})
        print(f"  schedule: steps_per_epoch={schedule.get('steps_per_epoch')} "
              f"patience_epochs={schedule.get('patience_epochs')} max_epochs={schedule.get('max_epochs')} "
              f"total_steps={schedule.get('total_steps')} early_stop={schedule.get('early_stop')} "
              f"lr_schedule={schedule.get('lr_schedule')}")
        print(f"  split: by={split.get('split_by')} train={split.get('train')} validation={split.get('validation')}")
        print(f"  parameters={run_info.get('parameters')} seeds={json.dumps(run_info.get('seeds'))}")
        normalization = run_info.get("value_normalization")
        print(f"  model_config={json.dumps(run_info.get('model_config'), ensure_ascii=False)}")
    else:
        print("  ⚠ 缺 run.json（旧 artifact？归一化改从 onnx sidecar 读）")

    sidecar = _find(root, "model.onnx.json")
    if sidecar is not None:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if normalization is None:
            normalization = meta.get("value_normalization")
        print(f"  onnx operators={meta.get('operators')} max_abs_error={meta.get('max_abs_error')}")
    print(f"  value_normalization={json.dumps(normalization, ensure_ascii=False)}")
    return normalization


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    norms = [_describe(Path(item)) for item in argv]
    usable = [norm for norm in norms if norm is not None]
    if not usable:
        raise SystemExit("一个模型的归一化都没读到，集成口径无法确认")
    identical = all(norm == usable[0] for norm in usable[1:])
    print(f"\n各模型 value_normalization 是否一致: {identical}")
    if not identical:
        print("→ 成员处在不同仿射空间：集成必须走 compose_ensemble.py 的仿射校正，直接平均会串味")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
