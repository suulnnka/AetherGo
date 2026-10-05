#!/usr/bin/env python3
"""AetherGo 纯蒸馏器:教师(ONNX/ORT-GPU)直接前向 → 学生(PyTorch)拟合。

不经过搜索、不经过 npz/KataGo train.py —— 教师的原始 policy/value/ownership/
scoremean 就是监督目标。特征由已对拍验证的 JS 编码器导出(dump_positions.mjs)。

用法(bleed 环境,torch-cu + onnxruntime-gpu):
  python distill_pytorch.py <positions.bin> <teacher.onnx> <out_ckpt.pt> \
      [--epochs N] [--batch N] [--lr F] [--max-rows N]

产出 checkpoint 直接走:
  export_model_pytorch.py -checkpoint out_ckpt.pt ... → bin.gz → dumponnx → 浏览器
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/a/go/KataGo/python')
from katago.train import modelconfigs  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402

STUDENT = 'b8c96h3tfrs-fson-silu'
ROW_F32 = 22 * 81 + 19


def load_positions(path, max_rows=None):
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw[:8].tobytes() == b'AEPOS001', '不是 dump_positions.mjs 的输出'
    n = int(np.frombuffer(raw[8:12], dtype='<u4')[0])
    row_f32 = int(np.frombuffer(raw[12:16], dtype='<u4')[0])
    assert row_f32 == ROW_F32 + 1
    n = min(n, max_rows or n)
    rows = np.frombuffer(raw[16:16 + n * (ROW_F32 + 1) * 4], dtype='<f4').reshape(n, ROW_F32 + 1)
    spatial = rows[:, :22 * 81].copy().reshape(n, 22, 9, 9)
    glob = rows[:, 22 * 81:ROW_F32].copy().reshape(n, 19)
    return spatial.astype(np.float32), glob.astype(np.float32)


def teacher_session(onnx_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, so, providers=prov)
    names_in = [i.name for i in sess.get_inputs()]
    names_out = [o.name for o in sess.get_outputs()]
    pol = next(n for n in names_out if 'Policy' in n and 'Pass' not in n)
    polp = next(n for n in names_out if 'PolicyPass' in n)
    val = next(n for n in names_out if 'Value' in n and 'Score' not in n)
    sv = next(n for n in names_out if 'ScoreValue' in n)
    own = next((n for n in names_out if 'Ownership' in n), None)
    return sess, names_in, pol, polp, val, sv, own


def teacher_targets(sess, names_in, pol_n, polp_n, val_n, sv_n, own_n, sp, gl, bs=512):
    n = sp.shape[0]
    probs82 = np.zeros((n, 82), np.float32)
    vprobs = np.zeros((n, 3), np.float32)
    own_t = np.zeros((n, 81), np.float32)
    smean = np.zeros((n, 1), np.float32)
    ones = np.ones((bs, 1, 9, 9), np.float32)
    for i in range(0, n, bs):
        j = min(i + bs, n)
        m = np.ones((j - i, 1, 9, 9), np.float32)
        out = sess.run(None, {
            names_in[0]: sp[i:j], names_in[1]: gl[i:j].reshape(j - i, 19, 1, 1),
            names_in[2]: m,
        })
        d = {o.name: v for o, v in zip(sess.get_outputs(), out)}
        pol = d[pol_n].reshape(j - i, -1, 81)       # (b, C, 9*9) 摊平
        polp = d[polp_n].reshape(j - i, -1)         # (b, C)
        vlog = d[val_n].reshape(j - i, -1)          # (b, 3)
        pc = pol[:, 0]                                 # 通道 0 = 主策略
        pp = polp[:, :1]                               # 通道 0 = pass
        logits = np.concatenate([pc, pp], axis=1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs82[i:j] = e / e.sum(axis=1, keepdims=True)
        ev = np.exp(vlog - vlog.max(axis=1, keepdims=True))
        vprobs[i:j] = ev / ev.sum(axis=1, keepdims=True)
        if own_n:
            own_t[i:j] = d[own_n][:, 0].reshape(j - i, -1)
        sv = d[sv_n].reshape(j - i, -1)             # (b, 6) → 通道 0 = scoremean(归一化)
        smean[i:j, 0] = sv[:, 0]
    return probs82, vprobs, own_t, smean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('positions')
    ap.add_argument('teacher_onnx')
    ap.add_argument('out_ckpt')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--dev', default='cuda')
    a = ap.parse_args()

    t0 = time.time()
    sp, gl = load_positions(a.positions, a.max_rows)
    n = sp.shape[0]
    print(f'读入 {n} 局面({time.time()-t0:.1f}s)')

    print('教师前向(ORT)…')
    t0 = time.time()
    sess, nin, pol_n, polp_n, val_n, sv_n, own_n = teacher_session(a.teacher_onnx)
    probs82, vprobs, own_t, smean = teacher_targets(
        sess, nin, pol_n, polp_n, val_n, sv_n, own_n, sp, gl)
    print(f'教师目标就绪({time.time()-t0:.1f}s)')

    cfg = modelconfigs.config_of_name[STUDENT]
    model = Model(cfg, 9)
    model.initialize()
    model.to(a.dev)
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f'学生 {STUDENT}:{nparams/1e6:.2f}M 参数')

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    steps = 0
    for ep in range(a.epochs):
        idx = rng.permutation(n)
        tot = {'loss': 0.0, 'pol': 0.0, 'val': 0.0, 'own': 0.0, 'sm': 0.0}
        nb = 0
        for i in range(0, n, a.batch):
            b = idx[i:i + a.batch]
            if len(b) < 8:
                continue
            x = torch.from_numpy(sp[b]).to(a.dev)
            g = torch.from_numpy(gl[b]).to(a.dev)
            tp = torch.from_numpy(probs82[b]).to(a.dev)
            tv = torch.from_numpy(vprobs[b]).to(a.dev)
            to = torch.from_numpy(own_t[b]).to(a.dev)
            ts = torch.from_numpy(smean[b]).to(a.dev)

            out = model(x, g)
            outs = out[0] if isinstance(out, tuple) and len(out) == 1 else out
            out_policy, out_value, out_miscvalue = outs[0], outs[1], outs[2]
            out_ownership = outs[4]
            pol_logits = out_policy[:, 0]                # (N, 82):81 点 + pass,通道 0 = 主策略
            loss_pol = -(tp * torch.log_softmax(pol_logits, dim=1)).sum(1).mean()
            loss_val = -(tv * torch.log_softmax(out_value, dim=1)).sum(1).mean()
            loss_own = F.mse_loss(out_ownership.reshape(len(b), -1), to)
            loss_sm = F.mse_loss(out_miscvalue[:, 0], ts[:, 0])
            loss = loss_pol + loss_val + 0.25 * loss_own + 0.25 * loss_sm

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot['loss'] += loss.item(); tot['pol'] += loss_pol.item()
            tot['val'] += loss_val.item(); tot['own'] += loss_own.item(); tot['sm'] += loss_sm.item()
            nb += 1; steps += 1
            if nb % 50 == 0:
                print(f'  ep{ep} step{steps} loss={tot["loss"]/nb:.4f} pol={tot["pol"]/nb:.4f} '
                      f'val={tot["val"]/nb:.4f} own={tot["own"]/nb:.4f} sm={tot["sm"]/nb:.4f}')
        print(f'epoch {ep} 完成:loss={tot["loss"]/max(nb,1):.4f}')

    torch.save({'model': model.state_dict(), 'config': cfg}, a.out_ckpt)
    print(f'已存 {a.out_ckpt}(students 参数 {nparams/1e6:.2f}M)')


if __name__ == '__main__':
    main()
