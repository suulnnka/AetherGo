#!/usr/bin/env bash
# AetherGo N3:自对弈强化循环(学生网自己的 selfplay → 训练 → gatekeeper 淘汰)
# 用法:run_n3_loop.sh <baseDir> <studentCkptDir>
#   baseDir:N3 工作区(selfplay 数据 / gatekeeper / 训练)
#   studentCkptDir:N2 产出的学生训练目录(trainrun,内含续训 checkpoint)
#
# 每一代:①学生 selfplay → ②shuffle+train(续训)→ ③gatekeeper 对上一代,
# 赢家成为新的 selfplay 模型。全部是主仓库现成命令,本脚本只做调度。
set -euo pipefail

BASE="${1:?用法: run_n3_loop.sh <baseDir> <studentCkptDir>}"
CKPTDIR="${2:?用法: run_n3_loop.sh <baseDir> <studentCkptDir>}"
KATAGO=/home/a/go/KataGo
CFGDIR=$(cd "$(dirname "$0")" && pwd)
ORT_LIB=/home/a/go/tools/onnxruntime-linux-x64-gpu_cuda13-1.30.0/lib
export LD_LIBRARY_PATH="$ORT_LIB:$LD_LIBRARY_PATH"

GEN=0
while true; do
  GEN=$((GEN + 1))
  echo "===== 第 $GEN 代 ====="

  # ① 学生自对弈(挂 30 分钟;量级由 MAXGAMES 控制)
  mkdir -p "$BASE"/selfplay
  timeout "${N3_SELFPLAY_SECONDS:-1800}" \
    "$KATAGO"/build-onnx/katago selfplay \
      -models-dir "$BASE"/models \
      -config "$CFGDIR"/selfplay_aether9.cfg \
      -output-dir "$BASE"/selfplay \
      -max-games-total "${N3_GAMES_PER_GEN:-600}" \
      -override-config "logDir=$BASE/logs-selfplay" || true

  # ② shuffle + 续训
  bash "$CFGDIR"/run_n2.sh "$BASE"/selfplay "$BASE"/train-gen$GEN "${N3_TRAIN_EPOCHS:-4}"

  # ③ gatekeeper:新模型 vs models/ 里的现任
  NEWBIN=$(ls -t "$BASE"/train-gen$GEN/models/*.bin.gz | head -1)
  mkdir -p "$BASE"/gatekeeper
  timeout "${N3_GATEKEEPER_SECONDS:-1800}" \
    "$KATAGO"/build-onnx/katago gatekeeper \
      -models-dir "$BASE"/models \
      -test-model "$NEWBIN" \
      -config "$CFGDIR"/gatekeeper_aether9.cfg \
      -tdata-dir "$BASE"/gatekeeper \
      -sgf-output-dir "$BASE"/gatekeeper/sgfs || true

  # gatekeeper 通过(日志出现 "is probably stronger")→ 新模型入列
  if grep -q "probably stronger" "$BASE"/logs-gatekeeper/*.log 2>/dev/null; then
    cp "$NEWBIN" "$BASE"/models/
    echo "第 $GEN 代通过守门,已发布"
  else
    echo "第 $GEN 代未通过守门,保留现任(本代数据仍在窗口里,下一轮继续)"
  fi
done
