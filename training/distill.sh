#!/usr/bin/env bash
# AetherGo 纯 PyTorch 蒸馏一键脚本(主路线,2026-10-01 拍板):
#   SGF 局面(JS 编码器,已对拍验证)→ 教师 ONNX 一次前向(ORT-GPU)
#   → 学生 b8c96h3tfrs-fson-silu 直接拟合(PyTorch,KD 损失)
#   → 导出 bin.gz → dumponnx → AetherGo/models/student_9.onnx
# 不经过搜索、不经过 npz/KataGo train.py。
#
# 用法:distill.sh <sgfs目录> <工作目录> [epochs] [batch]
# 依赖:bleed 环境(torch-cu + onnxruntime-gpu)、教师 teacher_9.onnx(首跑自动导出)
set -euo pipefail

SGFDIR="${1:?用法: distill.sh <sgfs目录> <工作目录> [epochs] [batch]}"
WORK="${2:?用法: distill.sh <sgfs目录> <工作目录> [epochs] [batch]}"
EPOCHS="${3:-4}"
BATCH="${4:-256}"
PY=/home/a/miniconda3/envs/bleed/bin/python
KATAGO=/home/a/go/KataGo
ORT_LIB=/home/a/go/tools/onnxruntime-linux-x64-gpu_cuda13-1.30.0/lib
export LD_LIBRARY_PATH="$ORT_LIB:$LD_LIBRARY_PATH"
HERE=$(cd "$(dirname "$0")" && pwd)
NODE=$(command -v node || echo /home/a/.local/opt/node-v22.23.3-linux-x64/bin/node)

mkdir -p "$WORK"

# 教师 9×9 ONNX(一次性,300MB)
if [ ! -f /home/a/go/aether/teacher_9.onnx ]; then
  echo "== 导出教师 ONNX =="
  "$KATAGO"/build-onnx/katago dumponnx -model /home/a/go/models/b11c768.bin.gz \
    -out /home/a/go/aether/teacher_9.onnx -nn-x-len 9 -nn-y-len 9 -require-exact-nnlen
fi

echo "== 1/4 局面特征导出(JS 编码器)=="
"$NODE" "$HERE"/dump_positions.mjs "$SGFDIR" "$WORK/positions.bin"

echo "== 2/4 教师前向 + 学生蒸馏 ==" 
"$PY" "$HERE"/distill_pytorch.py "$WORK/positions.bin" /home/a/go/aether/teacher_9.onnx \
  "$WORK/student.pt" --epochs "$EPOCHS" --batch "$BATCH"

echo "== 3/4 导出 bin.gz =="
rm -rf "$WORK/models"; mkdir -p "$WORK/models"
"$PY" "$KATAGO"/python/export_model_pytorch.py \
  -checkpoint "$WORK/student.pt" \
  -export-dir "$WORK/models" -model-name b8c96h3tfrs-fson-silu -filename-prefix student
BIN=$(ls "$WORK"/models/*.bin | head -1)
gzip -k "$BIN"

echo "== 4/4 dumponnx → AetherGo/models =="
"$KATAGO"/build-onnx/katago dumponnx -model "$BIN.gz" \
  -out /home/a/go/AetherGo/models/student_9.onnx \
  -nn-x-len 9 -nn-y-len 9 -require-exact-nnlen
echo "完成:AetherGo/models/student_9.onnx(验证:node test/nn-e2e.mjs models/student_9.onnx)"
