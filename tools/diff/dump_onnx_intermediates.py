#!/usr/bin/env python3
"""dump_onnx_intermediates.py — ONNX 检查点张量导出(cpuref 分段对拍用)。

读 positions.json([{spatial:[22*361], global:[19]}...]),把图内检查点张量
临时加为图输出,跑一遍,逐项存 /tmp/aewn_dbg/<name>.npy(展平 float32)。
"""
import json
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

MODEL = '/home/a/go/AetherGo/models/b8c96h3tfrs_19.onnx'
ATTN_RES = [43, 99, 155, 211, 267, 323, 379, 435]
FFN_RES = [59, 115, 171, 227, 283, 339, 395, 451]
CHECKPOINTS = [f'model.blocks.{2*b}/{n}' for b, n in enumerate(ATTN_RES)]
CHECKPOINTS += [f'model.blocks.{2*b+1}/{n}' for b, n in enumerate(FFN_RES)]
CHECKPOINTS += [
    'trunk/initbias/2',                          # stem(NCHW)
    'trunk/trunk/tonhwc/transpose/3',            # NHWC
    'model.blocks.0.norm1/scaled/12',            # norm0
    'model.blocks.0.q_proj/nhwc/13',             # q0
    'model.blocks.0/qrope/rope_out/28',          # qrope0 (1,3,361,32)
    'model.blocks.0/scoresscaled/36',            # scores0
    'model.blocks.0/probs/37',                   # probs0
    'model.blocks.0/attnnhwc/reshape/41',        # attn0
    'model.blocks.0.out_proj/nhwc/42',           # outproj0
    'model.blocks.0/43',                         # res0
    'model.blocks.1.ffn_linear1/nhwc/53',        # ffn0 half1
    'model.blocks.1/swiglu/57',                  # hidden0
    'model.blocks.1.ffn_linear2/nhwc/58',        # ffn0 out
    'model.blocks.1/59',                         # res1
    'model.blocks.15/451',                       # trunk end
    'model.act_trunkfinal/455',                  # trunkfinal
    'model.policy_head.conv1p/456',              # p1 (NCHW)
    'model.policy_head.actg/460',                # actg (NCHW)
    'model.policy_head/g/gpconcat/464',          # gpp
    'model.policy_head.act2/469',                # act2p
    'model.value_head/v/gpconcat/482',           # gpv
    'model.value_head.bias_valuehead/487',       # value
    'model.policy_head.linear_pass2/474',         # pass out
]

def main(pos_file):
    m = onnx.load(MODEL)
    g = m.graph
    vi = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None) for name in CHECKPOINTS]
    del g.output[:]
    g.output.extend(vi)
    onnx.save(m, '/tmp/aewn_dbg_model.onnx')
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 保真(禁图改写)
    sess = ort.InferenceSession('/tmp/aewn_dbg_model.onnx', so, providers=['CPUExecutionProvider'])
    rows = json.load(open(pos_file))
    n = len(rows)
    sp = np.array([r['spatial'] for r in rows], dtype=np.float32).reshape(n, 22, 19, 19)
    gl = np.array([r['global'] for r in rows], dtype=np.float32).reshape(n, 19, 1, 1)
    mk = np.ones((n, 1, 19, 19), dtype=np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: sp, sess.get_inputs()[1].name: gl, sess.get_inputs()[2].name: mk})
    import os
    os.makedirs('/tmp/aewn_dbg', exist_ok=True)
    for name, arr in zip(CHECKPOINTS, outs):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        a.tofile(f'/tmp/aewn_dbg/{name.replace("/", "_")}.npy')
        print(f'{name}: shape={np.asarray(arr).shape} mean={a.mean():.4f} std={a.std():.4f}')

if __name__ == '__main__':
    main(sys.argv[1])
