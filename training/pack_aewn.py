#!/usr/bin/env python3
"""pack_aewn.py — 把 b8c96h3tfrs ONNX 权重打包成 aethernn 私有格式 .aewn。

自研 WebGPU 引擎(src/nn/webgpu/)的唯一权重来源。执行计划硬编码为 JS 数据表,
packer 负责把通用 ONNX 图重排成引擎要的布局,并吸收四类图外预处理:

  1. 融合投影:q/k/v 三个 (96,96) 拼成 qkv (96,288);ffn_linear1+gate 拼成
     gate (96,512) —— 与 PyTorch 训练侧 fused_qkv_proj / fused_gate_proj 同构
     (model_pytorch.py TransformerAttentionBlock/TransformerFFNBlock)。
  2. BiasMask 折叠:头部的 ×scale + bias 里,scale 是逐出通道常数,直接乘进
     1×1 conv 权重(conv1g、value conv1),bias 原样保留;policy bias2 /
     trunkfinal 的 scale 乘的是「和」(gpbias),不能折,按原样存,由内核
     epilogue 处理。
  3. 头部 1×1 conv 权重转置:trunk 的 Wnhwc 已是 [inC][outC] k 主序,头部仍是
     Conv 布局 [outC][inC] —— 统一转成 [inC][outC],引擎 GEMM 单一读法。
  4. RoPE 表:图内 Sin/Cos/Gather 子图(make_rope_ongraph.py 的职责)换成
     打包期预算的 cos/sin 表(361×32),语义 = PyTorch precompute_freqs_cos_sin_2d
     (theta=100) + rotate_every_two,与 ONNX 常量逐位重建验证(cos diff 0)。

两种 dtype(INT8_QUANT_RESEARCH.md §4.1 方案 A 的引擎侧实现):

  --dtype f32(缺省)  header.dtype=0;全 fp32,文件 ≈3.82MB。
  --dtype i8f16       header.dtype=1;trunk 大权重(attn qkv/out + ffn gate/ffn2
                      ×8 + stem 3×3 conv ≈ 91.2 万参数)按「逐输出通道对称 int8」
                      量化:q = clip(round(w/scale), -127, 127),scale = max|w|/127,
                      4×int8 打进一个 u32(LSB 在前,shader 移位取字节);scale 存
                      f32 单独张量(`<名>.s`)。头部(policy/value 全部 conv+linear)、
                      RMSNorm、RoPE、全部 bias 按排除清单留 fp32(用户可见的目差/
                      死子标注不动)。激活由引擎以 f16 存储/f32 累加(katago-webgpu
                      同款形态)。文件 ≈1.1MB。

权重布局约定(引擎内核按此硬编码,见 src/nn/webgpu/plan.js):
  - 线性层一律 [inC][outC] k 主序,GEMM 形式 out[m,o] = Σ_k in[m,k]·W[k,o]。
  - conv_spatial 保持 [outC][inC][3][3](直接卷积读法;量化轴 = 首维)。
  - blob 内每个张量 256B 对齐(WebGPU 存储缓冲 binding offset 对齐要求)。

格式(两 dtype 同构):
  magic "AEWN" | u32 version=1 | u32 dtype | u32 metaJsonLen | u32 nTensors
  nTensors × { u32 nameLen, utf8 name, u32 ndim, u32×ndim dims, u64 offset, u32 nbytes, u32 pad }
  metaJson(utf8;dtype=1 时含 quant 映射 {张量名: scale张量名} 与 quantAxis)
  payload

用法(bleed 环境):
  python3 pack_aewn.py models/b8c96h3tfrs_19.onnx models/b8c96h3tfrs_19.aewn [--dtype f32|i8f16]
"""
import json
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"AEWN"
VERSION = 1
ALIGN = 256
POS, HEAD_DIM, THETA = 19, 32, 100.0
NUM_BLOCKS = 8


def rope_tables(freqs):
    """PyTorch precompute_freqs_cos_sin_2d(32, 19, theta=100) 的逐位重建。

    cos[p,2i]=cos[p,2i+1]=cos(ang[p,i]);sin_signed[p,2i]=-sin(ang[p,i]),[2i+1]=+sin;
    ang[p,i] = h(p)·freqs[i] (i<8) / w(p)·freqs[i-8] (i>=8)。应用侧
    out[j] = x[j]·cos[j] + x[j^1]·sin[j](swapidx=j^1 已折进 sin 的符号)。
    """
    freqs = freqs.astype(np.float32)
    p = np.arange(POS * POS, dtype=np.float32)
    h = np.floor(p / POS).astype(np.float32)
    w = (p % POS).astype(np.float32)
    pairs = np.concatenate([np.outer(h, freqs), np.outer(w, freqs)], axis=1)  # (361,16)
    ang = np.repeat(pairs, 2, axis=1)                                          # (361,32)
    cos = np.cos(ang).astype(np.float32)
    s = np.sin(pairs).astype(np.float32)
    sin = np.empty((POS * POS, HEAD_DIM), np.float32)
    sin[:, 0::2] = -s
    sin[:, 1::2] = s
    return cos, sin


def quantize_per_oc(w2d, name, axis):
    """逐输出通道对称 int8(INT8 报告 §4.1)。

    axis='last'(GEMM [K][O]):输出通道 = 列,scale = max(w, axis=0),q = w/scale
    按行广播;axis='first'(conv [OC][rest]):输出通道 = 行,scale = max(w, axis=1),
    q = w/scale[:,None] 按列广播。返回 (u32 打包数组, scale f32)。4×int8 按 LSB
    在前打进 u32,与 shader 的 `(u >> (i*8)) & 0xFF` 取字节序一致。逐权重断言:
    回代误差 ≤ 半步 scale。
    """
    assert w2d.ndim >= 1
    if axis == 'first':
        scale = np.abs(w2d).max(axis=1).astype(np.float32)
        scale = np.maximum(scale, 1e-12) / np.float32(127.0)
        q = np.rint(w2d / scale[:, None]).astype(np.int64)
        err = np.abs(q * scale[:, None] - w2d)
    else:
        scale = np.abs(w2d).max(axis=0).astype(np.float32)
        scale = np.maximum(scale, 1e-12) / np.float32(127.0)
        q = np.rint(w2d / scale[None, :]).astype(np.int64)
        err = np.abs(q * scale[None, :] - w2d)
    assert q.min() >= -127 and q.max() <= 127, name
    sb = scale[:, None] if axis == 'first' else scale[None, :]
    assert (err <= sb * 0.5 * 1.001).all(), f'{name}: 量化回代超差 max={err.max()}'
    q8 = q.astype(np.int8).reshape(-1)
    assert q8.size % 4 == 0, f'{name}: {q8.size} 不能 4 对齐'
    b = q8.view(np.uint8).reshape(-1, 4).astype(np.uint32)
    u32 = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
    return u32.astype(np.uint32), scale.astype(np.float32)


class Pack:
    def __init__(self):
        self.items = []            # (name, dims, np array float32/uint32)
        self.map = {}

    def add(self, name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        assert a.ndim >= 1, name
        self.items.append((name, a.shape, a))
        self.map[name] = a
        return a

    def add_u32(self, name, arr_u32, dims):
        a = np.ascontiguousarray(arr_u32, dtype=np.uint32)
        self.items.append((name, tuple(dims), a))
        self.map[name] = a
        return a

    def add_f16(self, name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float16)
        self.items.append((name, a.shape, a))
        self.map[name] = a
        return a

    def finalize(self, meta, dtype):
        n = len(self.items)
        meta_b = json.dumps(meta, ensure_ascii=False).encode('utf-8')
        # 头部:magic | version | dtype | metaJsonLen | nTensors(与 plan.js 解析严格一致)
        head = MAGIC + struct.pack('<IIII', VERSION, dtype, len(meta_b), n)

        def dir_size():
            total = 0
            for name, dims, a in self.items:
                nb = name.encode('utf-8')
                total += 4 + len(nb) + 4 + 4 * len(dims) + 16
            return total
        payload_start = (len(head) + dir_size() + len(meta_b) + ALIGN - 1) // ALIGN * ALIGN

        dir_b = bytearray()
        payload = bytearray()
        for name, dims, a in self.items:
            while len(payload) % ALIGN:
                payload.append(0)
            off = payload_start + len(payload)
            payload += a.tobytes()
            nb = name.encode('utf-8')
            dir_b += struct.pack('<I', len(nb)) + nb
            dir_b += struct.pack('<I', len(dims))
            dir_b += struct.pack(f'<{len(dims)}I', *dims)
            dir_b += struct.pack('<QII', off, a.nbytes, 0)
        pad = payload_start - (len(head) + len(dir_b) + len(meta_b))
        return head + bytes(dir_b) + meta_b + b'\x00' * pad + bytes(payload)


def main(inp, outp, dtype='f32', exclude=()):
    assert dtype in ('f32', 'i8f16', 'f16')
    dt_flag = {'f32': 0, 'i8f16': 1, 'f16': 2}[dtype]
    m = onnx.load(inp)
    g = m.graph
    I = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    # 形状断言:模型结构与计划表互验的第一道闸(架构不符直接拒绝打包)
    assert I['model.conv_spatial.W'].shape == (96, 22, 3, 3)
    assert I['model.linear_global.W'].shape == (96, 19, 1, 1)
    for i in range(NUM_BLOCKS):
        a = I[f'model.blocks.{2*i}.norm1.weightnhwc']
        assert a.shape == (1, 1, 1, 96), (i, a.shape)
        for k in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            assert I[f'model.blocks.{2*i}.{k}.Wnhwc'].shape == (96, 96)
        assert I[f'model.blocks.{2*i+1}.norm.weightnhwc'].shape == (1, 1, 1, 96)
        assert I[f'model.blocks.{2*i+1}.ffn_linear1.Wnhwc'].shape == (96, 256)
        assert I[f'model.blocks.{2*i+1}.ffn_linear_gate.Wnhwc'].shape == (96, 256)
        assert I[f'model.blocks.{2*i+1}.ffn_linear2.Wnhwc'].shape == (256, 96)

    P = Pack()
    quant_map = {}      # 量化张量名 → scale 张量名
    quant_axis = {}     # 量化轴:'last'(GEMM [K][O])| 'first'(conv [OC][...])
    n_qparams = 0

    def add_weight(name, arr):
        """按 dtype 决定存储:f32 直存 / int8 逐输出通道(clip=1.0,quant_explore
        校准搜索证明无敏感离群值)/ f16 权重(回退产物;计算本就是 f16)。"""
        nonlocal n_qparams
        if dt_flag == 2:
            if name in exclude:
                P.add(name, arr)
            else:
                P.add_f16(name, arr)
                quant_map[name] = 'f16'
            return
        if dt_flag == 0 or name in exclude:
            P.add(name, arr)
            return
        w = np.ascontiguousarray(arr, dtype=np.float32)
        # stem 两个矩阵都是 [OC][rest] 布局(输出通道在首维);其余 GEMM 是 [K][O]
        axis = 'first' if name.startswith('stem.') else 'last'
        w2d = w.reshape(w.shape[0], -1) if axis == 'first' else w.reshape(-1, w.shape[-1])
        u32, scale = quantize_per_oc(w2d, name, axis)
        P.add_u32(name, u32, w.shape)
        P.add(name + '.s', scale)
        quant_map[name] = name + '.s'
        quant_axis[name] = axis
        n_qparams += w.size

    # ---- stem(conv_spatial + linear_global,报告 58 层口径)----
    # 两矩阵成对量化/排除(引擎 stem 内核只备了全 i8 与全 f32 两种形态)
    if 'stem.conv_w' in exclude or 'stem.global_w' in exclude:
        exclude = tuple(set(exclude) | {'stem.conv_w', 'stem.global_w'})
    add_weight('stem.conv_w', I['model.conv_spatial.W'])
    add_weight('stem.global_w', I['model.linear_global.W'].reshape(96, 19))

    # ---- RoPE 表(图内子图 → 打包期预算)----
    cos, sin = rope_tables(I['rope/freqs'])
    P.add('rope.cos', cos)
    P.add('rope.sin', sin)

    # ---- trunk:8 × (attn + ffn);dtype=1 时量化(INT8 报告 §4.1)----
    for i in range(NUM_BLOCKS):
        q = I[f'model.blocks.{2*i}.q_proj.Wnhwc']
        k = I[f'model.blocks.{2*i}.k_proj.Wnhwc']
        v = I[f'model.blocks.{2*i}.v_proj.Wnhwc']
        P.add(f'attn{i}.norm', I[f'model.blocks.{2*i}.norm1.weightnhwc'].reshape(96))
        add_weight(f'attn{i}.qkv', np.concatenate([q, k, v], axis=1))     # (96,288)
        add_weight(f'attn{i}.out', I[f'model.blocks.{2*i}.out_proj.Wnhwc'])
        f1 = I[f'model.blocks.{2*i+1}.ffn_linear1.Wnhwc']
        fg = I[f'model.blocks.{2*i+1}.ffn_linear_gate.Wnhwc']
        P.add(f'ffn{i}.norm', I[f'model.blocks.{2*i+1}.norm.weightnhwc'].reshape(96))
        add_weight(f'ffn{i}.gate', np.concatenate([f1, fg], axis=1))      # (96,512)
        add_weight(f'ffn{i}.ffn2', I[f'model.blocks.{2*i+1}.ffn_linear2.Wnhwc'])

    # ---- trunkfinal(fixup norm:scale+bias+relu,无归一化)----
    P.add('trunkfinal.scale', I['model.norm_trunkfinal.scale'].reshape(96))
    P.add('trunkfinal.bias', I['model.norm_trunkfinal.bias'].reshape(96))

    # ---- policy head(头部全部留 f32,INT8 报告 §4.1 排除清单)----
    # conv1g 的 biasg.scale 折进权重(biasg 输出只进 relu,无其它消费方)
    w1g = I['model.policy_head.conv1g.W'].reshape(32, 96) * I['model.policy_head.biasg.scale'].reshape(32, 1)
    P.add('policy.conv1p', I['model.policy_head.conv1p.W'].reshape(32, 96).T)
    P.add('policy.conv1g', w1g.T)
    P.add('policy.conv1g_b', I['model.policy_head.biasg.bias'].reshape(32))
    P.add('policy.gp_ling', I['model.policy_head.linear_g.W'].reshape(32, 96).T)
    P.add('policy.bias2_scale', I['model.policy_head.bias2.scale'].reshape(32))
    P.add('policy.bias2_bias', I['model.policy_head.bias2.bias'].reshape(32))
    P.add('policy.conv2p', I['model.policy_head.conv2p.W'].reshape(2, 32).T)
    P.add('policy.pass_w', I['model.policy_head.linear_pass.W'].reshape(32, 96).T)
    P.add('policy.pass_b', I['model.policy_head.linear_pass_bias.b'].reshape(32))
    P.add('policy.pass2', I['model.policy_head.linear_pass2.W'].reshape(2, 32).T)

    # ---- value head(conv1 的 bias1.scale 同理折进权重;留 f32)----
    wv1 = I['model.value_head.conv1.W'].reshape(32, 96) * I['model.value_head.bias1.scale'].reshape(32, 1)
    P.add('value.conv1', wv1.T)
    P.add('value.conv1_b', I['model.value_head.bias1.bias'].reshape(32))
    P.add('value.v2', I['model.value_head.linear2.W'].reshape(64, 96).T)
    P.add('value.v2_b', I['model.value_head.bias2.b'].reshape(64))
    P.add('value.vh', I['model.value_head.linear_valuehead.W'].reshape(3, 64).T)
    P.add('value.vh_b', I['model.value_head.bias_valuehead.b'].reshape(3))
    P.add('value.misc', I['model.value_head.linear_miscvaluehead.W'].reshape(6, 64).T)
    P.add('value.misc_b', I['model.value_head.bias_miscvaluehead.b'].reshape(6))
    P.add('value.own', I['model.value_head.conv_ownership.W'].reshape(1, 32).T)

    # 注意力 scale(1/√32)与 rms eps 从图里读出存进 meta,内核不再硬编码数值来源
    def find_one(prefix):
        hits = [v for k, v in I.items() if k.startswith(prefix)]
        assert len(hits) == 1, (prefix, len(hits))
        return float(hits[0].flatten()[0])
    attn_scale = find_one('model.blocks.0/scale/')
    eps = find_one('model.blocks.0.norm1/eps/')
    meta = {
        'model': 'b8c96h3tfrs',
        'posLen': POS, 'channels': 96, 'heads': 3, 'headDim': HEAD_DIM,
        'ffn': 256, 'blocks': NUM_BLOCKS, 'spatialC': 22, 'globalC': 19,
        'attnScale': attn_scale, 'eps': eps,
        'dtype': dtype,
        'quant': quant_map, 'quantAxis': quant_axis,
        'quantExclude': sorted(exclude),
        'source': inp.split('/')[-1],
    }
    blob = P.finalize(meta, dt_flag)
    with open(outp, 'wb') as f:
        f.write(blob)
    if dt_flag == 1:
        excl = f';排除 {",".join(sorted(exclude))}' if exclude else ''
        print(f'{outp}: {len(blob)/1e6:.2f} MB(f32 版 3.82MB 的 {len(blob)/3.82e6:.0%});'
              f'量化 {n_qparams/1e6:.2f}M 参数(trunk GEMM),头部/norm/rope 留 f32{excl}')
    elif dt_flag == 2:
        print(f'{outp}: {len(blob)/1e6:.2f} MB;trunk 权重 f16 存储'
              f'(计算 f16,回退产物;fp16 纯变体 Top1 99.67% ≈ 无损,见 quant 报告)')
    else:
        print(f'{outp}: {len(blob)/1e6:.2f} MB,{len(P.map)} 张量 '
              f'(onnx {inp.split("/")[-1]};attnScale={attn_scale:.10f} eps={eps:.3e})')


if __name__ == '__main__':
    args = sys.argv[1:]
    dt = 'f32'
    exclude = ()
    if '--dtype' in args:
        i = args.index('--dtype')
        dt = args[i + 1]
        del args[i:i + 2]
    if '--exclude' in args:
        i = args.index('--exclude')
        exclude = tuple(x for x in args[i + 1].split(',') if x)
        del args[i:i + 2]
    main(args[0], args[1], dt, exclude)
