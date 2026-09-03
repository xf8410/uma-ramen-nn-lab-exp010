#!/usr/bin/env bash
# EXP-010：自蒸馏 v4 教师采分片（NN 评估器 + search_n=16）——分片采集
# 唯一变量 vs EXP-009：教师 rollout 评估器 手写冠军 → NN 66734.8
# 用法: run.sh <shard_id> [CHUNK] [START_OFFSET]
# ★ --count 是累计长度：区间 = [--start, --start + --count)（INCIDENT-20260902 教训）
set -euo pipefail

SHARD="${1:?用法: run.sh <shard_id> [CHUNK] [START_OFFSET]}"
SN=16
CHUNK="${2:-1000}"
START=$((SHARD * CHUNK))
DATA="experiment_output/shard_${SHARD}"

: "${RAMEN_NN_MODEL:?EXP-010: RAMEN_NN_MODEL 未设置——NN 教师模式必须显式提供模型路径}"
[ -f "$RAMEN_NN_MODEL" ] || { echo "EXP-010: 模型不存在: $RAMEN_NN_MODEL"; exit 1; }
[ -f "$RAMEN_NN_MODEL.json" ] || { echo "EXP-010: 模型旁路 JSON 不存在: $RAMEN_NN_MODEL.json"; exit 1; }

mkdir -p "$DATA" experiment_output

{
  echo "arm: EXP-010 selfdistill-v4  shard: ${SHARD}  chunk: ${CHUNK}"
  echo "index_range: [${START}, $((START + CHUNK)))"
  echo "teacher: RamenNnTrainer::load($RAMEN_NN_MODEL)（009b 冠军 66734.8）"
  echo "search_n: ${SN}  shard_size: 256  radical_factor_max: 1.4  quota_permille: 20,30"
  echo "upstream_commit: $(git rev-parse HEAD)"
  echo "gamedata_signature: $(cat gamedata/*.json gamedata/*.toml 2>/dev/null | sha256sum | cut -c1-16)"
  echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} | tee "experiment_output/env_shard_${SHARD}.txt"

echo "=== 教师配置自检（期望见到 RamenNnTrainer::load） ==="
grep -n "RamenNnTrainer::load" crates/umasim/src/search/searchable.rs

# NN 评估器需要 onnx feature
cargo build --release --features onnx -p umasim --bin ramen_teacher_collect

echo "=== 单条冒烟（index=${START}） ==="
./target/release/ramen_teacher_collect \
  --count 1 --start "$START" --search-n "$SN" \
  --output-dir "$DATA" --shard-size 256 2>&1 | tail -6

echo "=== 本分片全量 [${START}, $((START + CHUNK))) sn=${SN} ==="
./target/release/ramen_teacher_collect \
  --count "$CHUNK" --start "$START" --search-n "$SN" \
  --output-dir "$DATA" --shard-size 256 2>&1 | tail -20

cp "$DATA/manifest.json" "experiment_output/manifest_shard_${SHARD}.json"
echo "=== shard ${SHARD} 完成 ==="
