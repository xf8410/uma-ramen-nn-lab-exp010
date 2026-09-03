# EXP-010 立项：自蒸馏 v4——NN 当搜索引擎评估器（飞轮闭环）

## 动机（证据链）

- EXP-009b：gen4 NN 3 种子 66085.5 / **66734.8** / 65595.5，均值 66138.6（+949 vs gen3）
- 距 70000：最好种子 −3265，均值 −3861
- **教师→学生放大系数 ~3×**（008 教师 +116 → 学生最优种子 +347）已被实证
- 160k 数据/ep400 的边际收益已吃完；下一个教师端大跃迁 = **搜索标签本身更强**

## 本轮唯一变量

教师 rollout 评估器：手写冠军（65554.2）→ **NN 66734.8（gen4 最优）**

搜索标签=教师 rollout 的终局均分。评估器从 65554 档升到 66735 档，理论上标签精度
再抬 ~1180 分（若放大系数保持），对学生侧是足够冲 70000 的增量。实际收益受三个
折扣因素压制（§风险），预期保守按 1/2~2/3 记。

## 改点（1 处）

`crates/umasim/src/search/searchable.rs` → `RamenGame::default_rollout_trainer()`：
`RecommendedRamenTrainer::for_rollout()` → `RamenNnTrainer::load($RAMEN_NN_MODEL)`

- **为什么不用 008 的 with_tokens 方案**：008 是换手写配方（类型不变），010 是换
  评估器**类型**（RecommendedRamenTrainer → RamenNnTrainer），trait 关联类型
  `RolloutTrainer` 必须编译期定死，无法运行时切换
- RamenNnTrainer 已是 `Trainer<RamenGame>` 完整实现：policy argmax + race_shield
  硬守门 + choice 头委托手写（内部 fallback=for_rollout()），行为兼容 rollout
- **禁静默回退**：`RAMEN_NN_MODEL` 缺失/加载失败即 panic——否则教师悄悄退回手写
  相当于白跑全轮采集，必须 fail-fast

## 工作流（exp-010.yml）

1. **collect 前置：下载 009b 冠军模型 artifact**（主仓 run 33716171772 的 `EXP-009-model-seed20260831`，41MB，onnx+pt+metrics 全套）
2. **collect**：80 分片 × CHUNK=1000（80k），patch 链 006c→d→e→fix→**010**
   - 环境：`RAMEN_NN_MODEL=models/champion/model.onnx`
   - patch 后可执行文件必须链上 onnx feature（cargo build --features onnx）
3. **assemble/convert/labels/train**：同 009 配方（3 种子 ep400）
4. **裁决 EXP-010b**：回主仓跑，四基线（默认 65438.2 / 冠军 65554.2 / gen3 65669.2 / gen4 66734.8）

## 成本估算

- NN 评估器推理： tract ONNX 单帧 ~1ms，每 rollout ~170 决策点 ≈ 170ms
- vs 手写 ~1.2ms/rollout → **~140× 单 rollout 成本**
- search_n=64 × 候选 10 × 160000 样本 ≈ 1.02 亿次 rollout → 无法在免费 CI 承受

### 降本路径（本轮采用：search_n 降档 + 分片量减半）

- search_n 64 → **16**（EXP-003b 实证 sn16 相对 sn64 regret 降幅 <10%，可接受）
- 数据 160k → **80k**（gen3→gen4 的增量已证不靠数据量，80k 足够收敛）
- 总 rollout = 80k × 10 × 16 = 1280 万，单 rollout 170ms ≈ **604 CPU 小时**
- 80 分片 × 8 核 ≈ 每片 ~1h 墙钟，CI 可承受

## 风险与折扣

1. **NN 评估器有 race_shield 守门 + 状态分布偏移**：rollout 阶段网络见到的局面
   分布与训练时（手写世界）不一致，可能外推
2. **CRN 配对下 NN/handwritten 的差异**：CRN 只对齐规则层随机性，策略层的差异
   仍然保留——搜索排序质量取决于 rollout 期望值的**相对排序**是否更准
3. **标签绝对值漂移**：NN 评估器的 rollout 分数分布与手写不同，value 头的
   归一化常数（center/scale）要重新从 labels.json 读，labels.py 已按数据自适应

## 止损条件

- smoke 阶段 NN 教师 rollout 失败率 >5%：回滚 patch 链到 009 配方，本轮中止
- 采集完成后 Python regret 相对 009b 的 regret 恶化 >20：先查教师 rollout 是否
  系统性偏低，再决定是否裁决
- **最终裁决仍以 EXP-010b 闭环为准**：t>2 且 Δ>0 才判超，Python regret 只记不裁决
