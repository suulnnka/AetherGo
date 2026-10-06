#!/usr/bin/env python3
"""pack_aewn.py — 把 b8c96h3tfrs ONNX 权重打包成 aethernn 私有格式 .aewn。

自研 WebGPU 引擎(src/nn/webgpu/)的唯一权重来源。执行计划硬编码为 JS 数据表,
packer 负责把通用 ONNX 图重排成引擎要的布局,并吸收三类图外预处理:

  1. 融合投影:q/k/v 三个 (96,96) 拼成 qkv (96,288);ffn_linear1+gate 拼成
     gate (96,512) —— 与 PyTorch 训练侧 fused_qkv_proj / fused_gate_proj 同构
     (model_pytorch.py TransformerAttentionBlock/TransformerFFNBlock)。
  2. BiasMask 折叠:头部的 ×scale + bias 里,scale 是逐出通道常数,直接乘进
     1×1 conv 权重(conv1g、value conv1),bias 原样保留;policy bias2 /
     trunkfinal 的 scale 乘的是「和」(gpbias),不能折,按原样存,由内核
     epilogue 处理。
  3. RoPE 表:图内 Sin/Cos/Gather 子图(make_rope_ongraph.py 的职责)换成
     打包期预算的 cos/sin 表(361×32),语义 = PyTorch precompute_freqs_cos_sin_
     sin_2d(theta=100) + rotate_every_two,已用 numpy 对 ONNX 常量逐位重建
     (cos diff 0,sin 经 swapidx 折叠后 diff 0)。

权重布局约定(引擎内核按此硬编码,见 src/nn/webgpu/plan.js):
  - 线性层一律 [inC][outC] k 主序(Wnhwc),GEMM 形式 out[m,o] = Σ_k in[m,k]·W[k,o]。
  - conv_spatial 保持 [outC][inC][3][3](直接卷积读法)。
  - blob 内每个张量 256B 对齐(存储缓冲 binding offset 对齐安全余量)。

格式:
  magic "AEWN" | u32 version | u32 dtype(0=f32) | u32 metaJsonLen | u32 nTensors
  nTensors × { u32 nameLen, utf8 name, u32 ndim, u32×ndim dims, u64 offset, u32 nbytes, u32 pad }
  metaJson (utf8) —— 板径/通道/头数/eps 等,引擎加载时与计划表互验
  payload

用法(bleed 环境):
  python3 pack_aewn.py models/b8c96h3tfrs_19.onnx models/b8c96h3tfrs_19.aewn
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


class Pack:
    def __init__(self):
        self.items = []            # (name, dims, np array)
        self.map = {}

    def add(self, name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        assert a.ndim >= 1, name
        self.items.append((name, a.shape, a))
        self.map[name] = a
        return a

    def finalize(self, meta):
        n = len(self.items)
        meta_b = json.dumps(meta, ensure_ascii=False).encode('utf-8')
        # 头部:magic | version | dtype | metaJsonLen | nTensors(与 plan.js 解析严格一致)
        head = MAGIC + struct.pack('<IIII', VERSION, 0, len(meta_b), n)

        # 第一遍:先算目录长度 → payload 起始(绝对偏移需 256 对齐 —— WebGPU
        # 存储缓冲绑定偏移按 256 对齐校验)
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


def main(inp, outp):
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

    # ---- stem ----
    P.add('stem.conv_w', I['model.conv_spatial.W'])                 # (96,22,3,3)
    P.add('stem.global_w', I['model.linear_global.W'].reshape(96, 19))

    # ---- RoPE 表(图内子图 → 打包期预算)----
    cos, sin = rope_tables(I['rope/freqs'])
    P.add('rope.cos', cos)
    P.add('rope.sin', sin)

    # ---- trunk:8 × (attn + ffn) ----
    for i in range(NUM_BLOCKS):
        q = I[f'model.blocks.{2*i}.q_proj.Wnhwc']
        k = I[f'model.blocks.{2*i}.k_proj.Wnhwc']
        v = I[f'model.blocks.{2*i}.v_proj.Wnhwc']
        P.add(f'attn{i}.norm', I[f'model.blocks.{2*i}.norm1.weightnhwc'].reshape(96))
        P.add(f'attn{i}.qkv', np.concatenate([q, k, v], axis=1))     # (96,288)
        P.add(f'attn{i}.out', I[f'model.blocks.{2*i}.out_proj.Wnhwc'])
        f1 = I[f'model.blocks.{2*i+1}.ffn_linear1.Wnhwc']
        fg = I[f'model.blocks.{2*i+1}.ffn_linear_gate.Wnhwc']
        P.add(f'ffn{i}.norm', I[f'model.blocks.{2*i+1}.norm.weightnhwc'].reshape(96))
        P.add(f'ffn{i}.gate', np.concatenate([f1, fg], axis=1))      # (96,512)
        P.add(f'ffn{i}.ffn2', I[f'model.blocks.{2*i+1}.ffn_linear2.Wnhwc'])

    # ---- trunkfinal(fixup norm:scale+bias+relu,无归一化)----
    P.add('trunkfinal.scale', I['model.norm_trunkfinal.scale'].reshape(96))
    P.add('trunkfinal.bias', I['model.norm_trunkfinal.bias'].reshape(96))

    # ---- policy head ----
    # 头部全是 1×1 conv,ONNX 权重是 Conv 布局 [outC][inC](与 trunk 的 Wnhwc
    # [inC][outC] 相反!);引擎 GEMM 统一按 k 主序 [inC][outC] 读 → 全部转置。
    # conv1g 的 biasg.scale 折进权重(biasg 输出只进 relu,无其它消费方)。
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

    # ---- value head(conv1 的 bias1.scale 同理折进权重)----
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
        'source': inp.split('/')[-1],
    }
    blob = P.finalize(meta)
    with open(outp, 'wb') as f:
        f.write(blob)
    n_w = sum(int(np.prod(v.shape)) for v in P.map.values())
    print(f'{outp}: {len(blob)/1e6:.2f} MB,{len(P.map)} 张量,{n_w/1e6:.2f}M 参数 '
          f'(onnx {inp.split("/")[-1]};attnScale={attn_scale:.10f} eps={eps:.3e})')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
