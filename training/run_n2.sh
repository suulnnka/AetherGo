#!/usr/bin/env bash
# AetherGo N2/N3 管线:老师自对弈数据 → shuffle → 蒸馏训练(学生 b8c96h3tfrs-fson-silu)
#   → 导出 .bin.gz → dumponnx 导 9×9 ONNX → AetherGo/models/student_9.onnx
#
# 用法:run_n2.sh <dataDir> <outDir> [maxEpochs]
#   dataDir:katago selfplay 的 -output-dir(内含 <model>/tdata/*.npz 与 sgfs)
#   outDir :本管线工作目录(shuffle/训练/导出)
#   maxEpochs:本次训练实例最多跑的 epoch(冒烟用小值;正式训练去掉该参数)
set -euo pipefail

DATADIR="${1:?用法: run_n2.sh <dataDir> <outDir> [maxEpochs]}"
OUTDIR="${2:?用法: run_n2.sh <dataDir> <outDir> [maxEpochs]}"
MAXEPOCH="${3:-}"

KATAGO=/home/a/go/KataGo
PY=python3
ORT_LIB=/home/a/go/tools/onnxruntime-linux-x64-gpu_cuda13-1.30.0/lib
MODEL_KIND=b8c96h3tfrs-fson-silu
BATCH=128

EXTRA=()
if [ -n "$MAXEPOCH" ]; then EXTRA+=("-max-epochs-this-instance" "$MAXEPOCH"); fi

rm -rf "$OUTDIR"/shuffle
mkdir -p "$OUTDIR"/shuffle/train "$OUTDIR"/shuffle/val "$OUTDIR"/models

echo "== 1/4 shuffle =="
"$PY" "$KATAGO"/python/shuffle.py \
  "$DATADIR" \
  -expand-window-per-row 0.4 \
  -taper-window-exponent 0.65 \
  -out-dir "$OUTDIR"/shuffle/train \
  -out-tmp-dir "$OUTDIR"/shuffle/tmp \
  -approx-rows-per-out-file 70000 \
  -num-processes 8 \
  -keep-target-rows all \
  -min-rows 1000

# shuffle.py 只产 train;匀 5% 给 val(train.py 需要 val 目录)
VDIR="$OUTDIR"/shuffle/val
if [ -z "$(ls -A "$VDIR" 2>/dev/null)" ]; then
  mkdir -p "$VDIR"
  i=0
  for f in "$OUTDIR"/shuffle/train/*.npz; do
    i=$((i+1)); [ $((i % 20)) -eq 0 ] && mv "$f" "$VDIR"/
  done
fi

echo "== 2/4 训练 =="
# train.py 用 -traindir/-datadir 旗标(非位置参数);冒烟建议加
#   -epochs-per-export 1 -samples-per-epoch <小值>
# 让导出尽快出现;正式训练去掉这两个参数即可。
"$PY" "$KATAGO"/python/train.py \
  -traindir "$OUTDIR"/trainrun \
  -datadir "$OUTDIR"/shuffle \
  -pos-len 9 \
  -batch-size "$BATCH" \
  -model-kind "$MODEL_KIND" \
  -use-muon \
  -no-compile \
  -exportdir "$OUTDIR"/torchmodels_toexport \
  -exportprefix "$MODEL_KIND" \
  ${N2_TRAIN_EXTRA:-} \
  "${EXTRA[@]}"

echo "== 3/4 导出 bin.gz =="
CKPT=$(ls -t "$OUTDIR"/torchmodels_toexport/*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "! torchmodels_toexport 没有 checkpoint —— 训练没跑够导出周期"
  echo "  (可先用 -export-random-initialized-model 验证导出链,或加长训练)"
  exit 1
fi
"$PY" "$KATAGO"/python/export_model_pytorch.py \
  -checkpoint "$CKPT" \
  -export-dir "$OUTDIR"/models \
  -model-name "$MODEL_KIND" \
  -filename-prefix "$MODEL_KIND"
# 导出产物是未压缩 .bin —— dumponnx 前压一下
for b in "$OUTDIR"/models/*.bin; do
  [ -f "$b" ] && [ ! -f "$b.gz" ] && gzip -k "$b" && mv "$b.gz" "$b-s.bin.gz"
done
BIN=$(ls -t "$OUTDIR"/models/*.bin.gz | head -1)
"$KATAGO"/build-onnx/katago dumponnx \
  -model "$BIN" \
  -out /home/a/go/AetherGo/models/student_9.onnx \
  -nn-x-len 9 -nn-y-len 9 -require-exact-nnlen
echo "完成:AetherGo/models/student_9.onnx"

"$KATAGO"/build-onnx/katago dumponnx \
  -model "$BIN" \
  -out /home/a/go/AetherGo/models/student_9.onnx \
  -nn-x-len 9 -nn-y-len 9 -require-exact-nnlen
echo "完成:AetherGo/models/student_9.onnx"
