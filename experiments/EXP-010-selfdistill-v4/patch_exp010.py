#!/usr/bin/env python3
"""EXP-010 补丁：自蒸馏 v4——搜索引擎 rollout 评估器接入 009 冠军 NN。

背景（证据链）：
- EXP-009b：gen4 NN 3 种子 66085.5/66734.8/65595.5（均 66138.6），0831=66734.8 新全仓纪录
- 距 70000：最好种子差 3265；数据量/训练时长边际已兑现 → 下一量级=标签质量再上台阶
- 蒸馏放大链实证：教师 +116（008）→ 闭环最优种子 +347（~3×）

设计（上游既有能力接线，非新算法）：
- `FlatSearchGame::RolloutTrainer` 是**关联类型**（编译期定死），无法运行时切换
  手写/NN → 必须同时改类型声明与构造函数体
- RamenNnTrainer（onnx feature）已是 `Trainer<RamenGame>` 完整实现：policy argmax
  + race_shield 硬守门 + choice 头委托手写 fallback，rollout 语义兼容
- **cfg 双分支**：onnx 构建 → RamenNnTrainer::load($RAMEN_NN_MODEL)；
  非 onnx 构建 → 原 for_rollout()（保证 ramen_export_npy 等无 onnx 的 bin 仍可编译）
- **Send+Sync 绑定**：RamenNnTrainer 的模型字段 Arc<SimplePlan> 推理走 &self，
  镜像温泉 NeuralNetEvaluator 的既有 unsafe impl 先例；smoke 的双跑逐位一致
  断言是经验防线（若 tract 有内部可变缓存，双跑必现差异 → 红）
- **禁静默回退**：RAMEN_NN_MODEL 缺失/加载失败即 panic（教师悄悄退回手写=白跑）

锚点（3 处，各恰好 1 次断言防 pin 漂移）：
1. searchable.rs `type RolloutTrainer = crate::trainer::RecommendedRamenTrainer;`
2. searchable.rs `fn default_rollout_trainer() ... for_rollout() }`（含 doc 行）
3. ramen_nn_trainer.rs 追加 unsafe impl Send/Sync（锚 `pub struct RamenNnTrainer` 存在性）
"""
from pathlib import Path
import sys

SEARCHABLE = Path("crates/umasim/src/search/searchable.rs")
NN_TRAINER = Path("crates/umasim/src/trainer/ramen_nn_trainer.rs")

TYPE_OLD = """    type RolloutTrainer = crate::trainer::RecommendedRamenTrainer;"""

TYPE_NEW = """    /// EXP-010：onnx 构建下 rollout 评估器 = RamenNnTrainer（$RAMEN_NN_MODEL）；
    /// 非 onnx 构建保持原 RecommendedRamenTrainer（无 onnx 的 bin 仍可编译）
    #[cfg(feature = "onnx")]
    type RolloutTrainer = crate::trainer::RamenNnTrainer;
    #[cfg(not(feature = "onnx"))]
    type RolloutTrainer = crate::trainer::RecommendedRamenTrainer;"""

FN_OLD = """    /// rollout 专用实例：三份年的 breakdown 全部关闭
    fn default_rollout_trainer() -> Self::RolloutTrainer {
        crate::trainer::RecommendedRamenTrainer::for_rollout()
    }"""

FN_NEW = """    /// rollout 专用实例：三份年的 breakdown 全部关闭
    ///
    /// EXP-010 自蒸馏 v4：onnx 构建 = RamenNnTrainer::load($RAMEN_NN_MODEL)。
    /// 模型缺失/加载失败即 panic（禁静默回退——教师悄悄退回手写=标签口径污染）。
    #[cfg(feature = "onnx")]
    fn default_rollout_trainer() -> Self::RolloutTrainer {
        let path = std::env::var("RAMEN_NN_MODEL")
            .expect("EXP-010: RAMEN_NN_MODEL 未设置——NN 教师模式必须显式提供模型路径");
        crate::trainer::RamenNnTrainer::load(std::path::Path::new(&path))
            .expect("EXP-010: RamenNnTrainer 加载失败（检查 ONNX + 旁路 JSON 是否齐全）")
    }

    #[cfg(not(feature = "onnx"))]
    fn default_rollout_trainer() -> Self::RolloutTrainer {
        crate::trainer::RecommendedRamenTrainer::for_rollout()
    }"""

SEND_SYNC = """
// ============================================================================
// EXP-010：Send + Sync 绑定（镜像温泉 NeuralNetEvaluator 的既有先例）
//
// 模型字段是 Arc<SimplePlan>，推理走 &self；温泉侧 NeuralNetEvaluator 已用同一
// 论证 unsafe impl 过（"通过 Arc 共享模型，是线程安全的"）。本类型作为
// FlatSearchGame::RolloutTrainer 被 rayon 并行 rollout 共享，需要同样绑定。
// 经验防线：EXP-010 smoke 的双跑逐位一致断言——若 tract run 存在内部可变竞态，
// 同 index 两次采集的 part 哈希必然不同，smoke 直接红。
// ============================================================================
unsafe impl Send for RamenNnTrainer {}
unsafe impl Sync for RamenNnTrainer {}
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        print(f"PATCH FAIL: [{label}] 锚点出现 {count} 次（应为 1）——pin 漂移或上游变更，拒绝打补丁")
        sys.exit(1)
    return text.replace(old, new)


def main() -> int:
    text = SEARCHABLE.read_text(encoding="utf-8")
    text = replace_once(text, TYPE_OLD, TYPE_NEW, "RolloutTrainer 类型声明")
    text = replace_once(text, FN_OLD, FN_NEW, "default_rollout_trainer 函数体")
    SEARCHABLE.write_text(text, encoding="utf-8")

    nn = NN_TRAINER.read_text(encoding="utf-8")
    if "pub struct RamenNnTrainer" not in nn:
        print("PATCH FAIL: ramen_nn_trainer.rs 找不到 pub struct RamenNnTrainer——pin 漂移")
        return 1
    if "unsafe impl Sync for RamenNnTrainer" in nn:
        print("PATCH SKIP: Send/Sync 绑定已存在（重复应用？）")
        return 1
    nn += SEND_SYNC
    NN_TRAINER.write_text(nn, encoding="utf-8")

    print("PATCH OK: RolloutTrainer(onnx)=RamenNnTrainer::load($RAMEN_NN_MODEL) + Send/Sync 绑定（EXP-010 自蒸馏 v4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
