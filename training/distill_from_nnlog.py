#!/usr/bin/env python3
"""AetherGo 蒸馏器(nnlog 版):直接消费 katago ONNX 后端的旁路记录。

数据 = 自对弈内部「教师输入→教师原始输出」对(KATA_NNLOG 采集,见
onnxbackend.cpp 的 nnEvalLogBatch)。memmap 零拷贝读取,教师 logits 的
softmax 在 GPU 上现算 —— 无预处理遍、无 npz、无第二次教师前向。

用法:python distill_from_nnlog.py <collect.binl> <out_ckpt.pt>
        [--epochs N] [--batch N] [--lr F] [--max-rows N] [--dev cuda]
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


def open_log(path):
    f = open(path, 'rb')
    magic = f.read(8)
    assert magic == b'AENNL001', f'不是 nnEvalLogBatch 的输出:{magic}'
    hdr = np.frombuffer(f.read(32), '<u4')
    sp_elts, gl_elts, pol_elts, pass_elts, val_elts, sv_elts, own_elts, _ = hdr
    row_bytes = int(sp_elts + (gl_elts + pol_elts + pass_elts + val_elts + sv_elts + own_elts) * 4)
    return f, dict(sp=int(sp_elts), gl=int(gl_elts), pol=int(pol_elts), pas=int(pass_elts),
                   val=int(val_elts), sv=int(sv_elts), own=int(own_elts), row=row_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('out_ckpt')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--dev', default='cuda')
    a = ap.parse_args()

    f, dims = open_log(a.log)
    import os
    n_rows = (os.path.getsize(a.log) - 40) // dims['row']
    if a.max_rows:
        n_rows = min(n_rows, a.max_rows)
    print(f"日志 {n_rows:,} 行,每行 {dims['row']}B(空间{dims['sp']} policyC{dims['pol']//81})")
    mm = np.memmap(a.log, dtype=np.uint8, mode='r')

    off = 40
    sp_n, gl_n, pas_n = dims['sp'], dims['gl'], dims['pas']
    pol_n, val_n, sv_n, own_n = dims['pol'], dims['val'], dims['sv'], dims['own']

    def row_slices(i):
        b = off + i * dims['row']
        e = b + dims['row']
        return b, e

    cfg = modelconfigs.config_of_name[STUDENT]
    model = Model(cfg, 9)
    model.initialize()
    model.to(a.dev)
    model.train()
    print(f'学生 {STUDENT}:{sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数')

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * max(1, n_rows // a.batch))
    rng = np.random.default_rng(0)

    # 预生成每行的字节偏移,批内解码向量化(u8 空间 + f32 其余)
    offs = off + np.arange(n_rows, dtype=np.int64) * dims['row']
    f32_off = sp_n                                     # 行内 f32 区起点(字节)
    gl_off, pas_off = f32_off, f32_off + gl_n * 4
    pol_off = pas_off + pas_n * 4
    val_off = pol_off + pol_n * 4
    sv_off = val_off + val_n * 4
    own_off = sv_off + sv_n * 4

    steps = 0
    t0 = time.time()
    for ep in range(a.epochs):
        idx = rng.permutation(n_rows)
        tot, nb = 0.0, 0
        for i in range(0, n_rows - a.batch, a.batch):
            b = idx[i:i + a.batch]
            base = offs[b]
            rows = np.stack([mm[x:x + dims['row']] for x in base])   # (B, row)
            sp = torch.from_numpy(np.ascontiguousarray(rows[:, :sp_n])).to(a.dev).float().view(len(b), 22, 9, 9)
            gl = torch.from_numpy(rows[:, gl_off:pas_off].copy().view('<f4')).to(a.dev)
            pas = torch.from_numpy(rows[:, pas_off:pol_off].copy().view('<f4')).to(a.dev)
            pol = torch.from_numpy(rows[:, pol_off:val_off].copy().view('<f4')).to(a.dev)
            val = torch.from_numpy(rows[:, val_off:sv_off].copy().view('<f4')).to(a.dev)
            sv = torch.from_numpy(rows[:, sv_off:own_off].copy().view('<f4')).to(a.dev)
            own = torch.from_numpy(rows[:, own_off:own_off + own_n * 4].copy().view('<f4')).to(a.dev)

            # 教师目标(GPU 上 softmax)
            with torch.no_grad():
                t_pol = torch.softmax(torch.cat([pol[:, :81], pas[:, :1]], dim=1), dim=1)
                t_val = torch.softmax(val, dim=1)
                t_sm = sv[:, 0]

            out = model(sp, gl)
            outs = out[0] if isinstance(out, tuple) and len(out) == 1 else out
            s_pol_logits, s_val, s_misc, s_own = outs[0], outs[1], outs[2], outs[4]
            loss_pol = -(t_pol * torch.log_softmax(s_pol_logits[:, 0], dim=1)).sum(1).mean()
            loss_val = -(t_val * torch.log_softmax(s_val, dim=1)).sum(1).mean()
            loss_own = F.mse_loss(s_own.reshape(len(b), -1), own)
            loss_sm = F.mse_loss(s_misc[:, 0], t_sm)
            loss = loss_pol + loss_val + 0.25 * loss_own + 0.25 * loss_sm

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1; steps += 1
            if nb % 100 == 0:
                r = n_rows and (i + a.batch) / n_rows
                print(f'  ep{ep} {r:.0%} step{steps} loss={tot/nb:.4f} '
                      f'({(time.time()-t0):.0f}s)', flush=True)
        print(f'epoch {ep} 完成:loss={tot/max(nb,1):.4f}', flush=True)

    torch.save({'model': model.state_dict(), 'config': cfg}, a.out_ckpt)
    print(f'已存 {a.out_ckpt}')


if __name__ == '__main__':
    main()
