#!/usr/bin/env python3
"""EXP-010 补丁：自蒸馏 v4——搜索引擎 rollout 评估器接入 009 冠军 NN。

背景（证据链）：
- EXP-009b：gen4 NN 3 种子 66085.5/66734.8/65595.5（均 66138.6），0831=66734.8 新全仓纪录
- 距 70000：最好种子差 3265，3 种子均值差 3861
- 数据量/训练时长的边际收益已兑现，下一量级=标签质量再上台阶
- 蒸馏增益链实证：教师 +116（008）→ 闭环最优种子 +347；教师端每 +1 分，NN 侧放大 ~3 倍

设计（上游既有能力接线，非新算法）：
- searchable.rs `RamenGame::default_rollout_trainer()` 的类型是 trait 关联类型
  `RolloutTrainer`（编译期定死），无法按环境变量在手写/NN 间运行时切换——
  所以本轮直接把关联类型的构造函数体换成 NN 加载器
- RamenNnTrainer 是 `Trainer<RamenGame>` 完整实现：policy argmax + race_shield
  硬守门 + choice 头委托手写 fallback（for_rollout()），rollout 语义兼容
- `RAMEN_NN_MODEL` 环境变量指定 ONNX 路径（CI 注入）；缺失/加载失败直接 panic
  ——教师评估器**必须**显式就位，静默回退手写会污染本轮全部标签（PRINCIPLES §4）

口径警告：
- 本轮是唯一变量实验：教师评估器 手写冠军 → NN 66734.8
- regret 口径随教师变，Python 侧 regret 只记不裁决
- 最终裁决 = EXP-010b 闭环四基线配对（t>2 且 Δ>0 才判超），回主仓跑
"""
from pathlib import Path
import sys

TARGET = Path("crates/umasim/src/search/searchable.rs")

OLD = """    /// rollout 专用实例：三份年的 breakdown 全部关闭
    fn default_rollout_trainer() -> Self::RolloutTrainer {
        crate::trainer::RecommendedRamenTrainer::for_rollout()
    }"""

NEW = """    /// rollout 专用实例：三份年的 breakdown 全部关闭
    ///
    /// EXP-010 自蒸馏 v4：rollout 评估器切 NN（009b 冠军 66734.8）。
    /// - `RAMEN_NN_MODEL` 环境变量指定 ONNX 路径；缺失/加载失败即 panic
    ///   （禁静默回退：教师悄悄退回手写 = 本轮标签口径污染，白跑）
    /// - choice 头未训练，RamenNnTrainer 内部已委托手写 fallback，rollout 兼容
    /// - race_shield 默认开：自选比赛硬守门与真实对局同构
    fn default_rollout_trainer() -> Self::RolloutTrainer {
        let path = std::env::var("RAMEN_NN_MODEL")
            .expect("EXP-010: RAMEN_NN_MODEL 未设置——NN 教师模式必须显式提供模型路径");
        crate::trainer::RamenNnTrainer::load(std::path::Path::new(&path))
            .expect("EXP-010: RamenNnTrainer 加载失败（检查 ONNX + 旁路 JSON 是否齐全）")
    }"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        print(f"PATCH FAIL: 锚点出现 {count} 次（应为 1）——pin 漂移或上游变更，拒绝打补丁")
        return 1
    text = text.replace(OLD, NEW)
    TARGET.write_text(text, encoding="utf-8")
    print("PATCH OK: RamenGame::default_rollout_trainer → RamenNnTrainer::load($RAMEN_NN_MODEL)（EXP-010 自蒸馏 v4 教师）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
