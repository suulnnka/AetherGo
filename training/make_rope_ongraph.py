#!/usr/bin/env python3
"""把 AetherGo ONNX 模型里预计算的 RoPE 查找表换成图内现场计算。

背景(AetherGo models/b8c96h3tfrs_19.onnx 实测):
  - 8 个注意力块 × q/k × (ropecos/ropesinsigned/ropeswapidx) = 48 个 initializer,
    全部是同一张表的复制,合计 4.44 MB ≈ 模型体积的 54%。
  - 表的生成公式(KataGo_Transformer train/model_pytorch.py precompute_freqs_cos_sin_2d,
    theta=100,head_dim=32,19×19 行主序 2D 棋盘 RoPE),已用 numpy 精确重建(误差 6e-8):
        freqs = 100^(-[0,2,..,14]/16)                  # (8,)
        angle_pair(p, i) = (p//19)·freqs[i]  (i<8)     # 行半
                         = (p%19) ·freqs[i-8] (i>=8)   # 列半
        cos[2i]=cos[2i+1]=cos(angle_i);sin_signed[2i]=-sin(angle_i),[2i+1]=+sin(angle_i)
  - 应用侧结构不变:out = x·cos + Gather(x, swapidx, axis=-1)·sin_signed。

改图:
  - 删 48 个 rope initializer,新增一次性的图内子图(Mul/Concat/Unsqueeze/Reshape/Sin/Cos,
    全部是 ORT-Web WebGPU 原生 kernel 的基础算子)计算共享表 rope/cos、rope/sinsigned (1,1,361,32);
    swapidx 保留为共享 initializer(32×i64)。
  - 所有 rope_swap / rope_t1 / rope_t2 节点的表输入改指向共享张量(广播 1→3 头不变)。

用法:
  python3 make_rope_ongraph.py <in.onnx> <out.onnx>
输出同时打印新旧文件体积与表重建最大误差。
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

POS_LEN, HEAD_DIM, THETA = 19, 32, 100.0


def build_rope_tables():
    freqs = (1.0 / (THETA ** (np.arange(0, HEAD_DIM // 2, 2, dtype=np.float32) / (HEAD_DIM // 2)))).astype(np.float32)
    p = np.arange(POS_LEN * POS_LEN, dtype=np.float32)
    h_idx = np.floor(p / POS_LEN).astype(np.float32)
    w_idx = (p % POS_LEN).astype(np.float32)
    return freqs, h_idx, w_idx


def reference_tables():
    """float64 重建,用于和被删掉的存储表对比误差。"""
    freqs, h_idx, w_idx = build_rope_tables()
    freqs = freqs.astype(np.float64)
    pairs = np.concatenate([np.outer(h_idx, freqs), np.outer(w_idx, freqs)], axis=1)  # (361,16)
    cos = np.empty((POS_LEN * POS_LEN, HEAD_DIM))
    sin_signed = np.empty((POS_LEN * POS_LEN, HEAD_DIM))
    ang = np.repeat(pairs, 2, axis=1)
    cos[:, 0::2] = np.cos(ang)[:, 0::2]
    cos[:, 1::2] = np.cos(ang)[:, 0::2]
    s = np.sin(pairs)
    sin_signed[:, 0::2] = -s
    sin_signed[:, 1::2] = s
    return cos, sin_signed


def main(inp, outp):
    m = onnx.load(inp)
    g = m.graph
    nodes, inits = list(g.node), {i.name: numpy_helper.to_array(i) for i in g.initializer}

    # ---- 1. 校验待删表的一致性,并和公式重建对比 ----
    rope_keys = [k for k in inits if '/qrope/' in k or '/krope/' in k]
    ref_cos, ref_sin = reference_tables()
    for k in rope_keys:
        arr = inits[k]
        if arr.dtype == np.float32 and arr.ndim == 4 and arr.shape[-1] == HEAD_DIM:
            r = ref_cos if 'ropecos' in k else ref_sin
            err = np.abs(arr[0, 0] - r).max()
            assert err < 1e-5, f"{k} 与重建表不一致: {err}"
    print(f"待删 rope initializer: {len(rope_keys)} 个,合计 "
          f"{sum(inits[k].nbytes for k in rope_keys)/1e6:.2f} MB;公式重建校验通过 (<1e-5)")

    # ---- 2. 新增共享常量与现场计算子图 ----
    freqs, h_idx, w_idx = build_rope_tables()
    new_inits = [
        numpy_helper.from_array(freqs, 'rope/freqs'),
        numpy_helper.from_array(h_idx, 'rope/hidx'),
        numpy_helper.from_array(w_idx, 'rope/widx'),
        numpy_helper.from_array(np.arange(HEAD_DIM, dtype=np.int64) ^ 1, 'rope/swapidx'),
        numpy_helper.from_array(np.array(-1.0, np.float32), 'rope/neg1'),
        numpy_helper.from_array(np.array([1, 1, POS_LEN * POS_LEN, HEAD_DIM], np.int64), 'rope/shape'),
        numpy_helper.from_array(np.array([1], np.int64), 'rope/ax1'),
        numpy_helper.from_array(np.array([2], np.int64), 'rope/ax2'),
    ]
    new_nodes = [
        helper.make_node('Unsqueeze', ['rope/hidx', 'rope/ax1'], ['rope/h_col'], 'rope/uh'),
        helper.make_node('Unsqueeze', ['rope/widx', 'rope/ax1'], ['rope/w_col'], 'rope/uw'),
        helper.make_node('Mul', ['rope/h_col', 'rope/freqs'], ['rope/h_ang'], 'rope/hang'),
        helper.make_node('Mul', ['rope/w_col', 'rope/freqs'], ['rope/w_ang'], 'rope/wang'),
        helper.make_node('Concat', ['rope/h_ang', 'rope/w_ang'], ['rope/pairs'], 'rope/pairs', axis=1),
        # cos:[c,c] 成对展开 → (361,16,2) → (1,1,361,32)
        helper.make_node('Unsqueeze', ['rope/pairs', 'rope/ax2'], ['rope/p3'], 'rope/up'),
        helper.make_node('Concat', ['rope/p3', 'rope/p3'], ['rope/ang_d'], 'rope/angd', axis=2),
        helper.make_node('Reshape', ['rope/ang_d', 'rope/shape'], ['rope/ang32'], 'rope/ang32'),
        helper.make_node('Cos', ['rope/ang32'], ['rope/cos'], 'rope/cos'),
        # sin_signed:[-s,+s] 交错 → (361,16,2) → (1,1,361,32)
        helper.make_node('Sin', ['rope/pairs'], ['rope/s_p'], 'rope/sin'),
        helper.make_node('Mul', ['rope/s_p', 'rope/neg1'], ['rope/ns_p'], 'rope/neg'),
        helper.make_node('Unsqueeze', ['rope/ns_p', 'rope/ax2'], ['rope/ns3'], 'rope/uns_n'),
        helper.make_node('Unsqueeze', ['rope/s_p', 'rope/ax2'], ['rope/s3'], 'rope/uns_p'),
        helper.make_node('Concat', ['rope/ns3', 'rope/s3'], ['rope/ss_d'], 'rope/ssd', axis=2),
        helper.make_node('Reshape', ['rope/ss_d', 'rope/shape'], ['rope/sinsigned'], 'rope/sinsigned'),
    ]
    value_info = [
        helper.make_tensor_value_info('rope/cos', TensorProto.FLOAT, (1, 1, POS_LEN * POS_LEN, HEAD_DIM)),
        helper.make_tensor_value_info('rope/sinsigned', TensorProto.FLOAT, (1, 1, POS_LEN * POS_LEN, HEAD_DIM)),
    ]

    # ---- 3. 重写 16 个应用点的表输入,删除旧 initializer ----
    rewritten = 0
    for n in nodes:
        if n.name.endswith('/qrope/rope_swap') or n.name.endswith('/krope/rope_swap'):
            n.input[1] = 'rope/swapidx'; rewritten += 1
        elif n.name.endswith('/qrope/rope_t1') or n.name.endswith('/krope/rope_t1'):
            n.input[1] = 'rope/cos'; rewritten += 1
        elif n.name.endswith('/qrope/rope_t2') or n.name.endswith('/krope/rope_t2'):
            n.input[1] = 'rope/sinsigned'; rewritten += 1
    assert rewritten == 48, f"预期改写 48 个节点,实际 {rewritten}"
    keep = [i for i in g.initializer if i.name not in rope_keys]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    del g.node[:]
    g.node.extend(new_nodes + nodes)          # 计算子图放图首
    del g.value_info[:]
    g.value_info.extend(value_info)

    onnx.checker.check_model(m)
    onnx.save(m, outp)

    import os
    print(f"改写节点: {rewritten};体积 {os.path.getsize(inp)/1e6:.2f} MB → {os.path.getsize(outp)/1e6:.2f} MB")
    print(f"输出: {outp}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
