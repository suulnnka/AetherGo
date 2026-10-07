#!/usr/bin/env python3
"""b8c96h3tfrs INT4 + QAT 实验:PTQ 损失 vs QAT 恢复,协议与 quant_explore.py 完全同口径。

变体:
  int4w       - 逐输出通道对称 int4(minmax,scale=amax/7),反量化 fp16 运算
  int4w_clip  - 逐层裁剪比例校准搜索(最小化该层输出 MSE)
  int4qat     - STE 直通估计器微调(固定 scale,Trunk 假量化 + 头部 fp32 一起训),
                训完按真 int4 网格硬量化再评估(部署形态)

评估协议与 quant_explore.py 一致:种子 20261006,校准 60 局 / 评估 200 局,
batch 128,评估 64 batch;指标 = 训练损失 Δ + policy KL/Top1/胜率 MAE/目差 MAE/形势 RMSE。
"""
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("KATAGO_FUSED_QKV_PROJ", "0")
sys.path.insert(0, "/home/a/go/KataGo/python")
from katago.train import data_processing_pytorch as dp
from katago.train import load_model as km
from katago.train.metrics_pytorch import Metrics

CKPT = "/home/a/go/trainrun/b8c96h3tfrs_run/checkpoint.ckpt"
DATA_GLOB = "/home/a/go/kata1_data/*npzs/*/*.npz"
OUT_DIR = "/home/a/go/trainrun/quant"
QMIN, QMAX = -7, 7
QAT_STEPS = int(os.environ.get("QAT_STEPS", "1500"))
QAT_LR = float(os.environ.get("QAT_LR", "1e-4"))
QAT_BS = 128


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def quant_weight(w, bits, clip_frac=1.0):
    flat = w.reshape(w.shape[0], -1)
    amax = flat.abs().amax(dim=1, keepdim=True) * clip_frac
    scale = (amax / QMAX).clamp_min(1e-12)
    q = (flat / scale).round().clamp(QMIN, QMAX)
    return (q * scale).reshape(w.shape).to(w.dtype), scale.squeeze(1)


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        s = scale.reshape(-1, *([1] * (w.dim() - 1)))
        return torch.clamp((w / s).round(), QMIN, QMAX) * s

    @staticmethod
    def backward(ctx, g):
        return g, None


class STEWrap(nn.Module):
    def __init__(self, mod, scale):
        super().__init__()
        self.mod = mod
        self.register_buffer("qscale", scale.reshape(-1, *([1] * (mod.weight.dim() - 1))).float())

    def forward(self, x):
        w = FakeQuant.apply(self.mod.weight.float(), self.qscale)
        if isinstance(self.mod, nn.Conv2d):
            return F.conv2d(x, w, self.mod.bias, self.mod.stride, self.mod.padding, self.mod.dilation, self.mod.groups)
        return F.linear(x, w, self.mod.bias)


def find_quantizable(model):
    out = []
    for name, mod in model.named_modules():
        if name.startswith(("policy_head", "value_head")):
            continue
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and mod.weight.dim() >= 2:
            out.append((name, mod))
    return out


def materialize(files, max_batches, batch_size, model_config, has_meta):
    batches = []
    for batch in dp.read_npz_training_data(
        files, batch_size, 1, 0, 19, torch.device("cpu"),
        randomize_symmetries=False, include_meta=has_meta,
        model_config=model_config, prefetch_depth=2,
    ):
        batches.append(batch)
        if len(batches) >= max_batches:
            break
    return batches


def batch_to(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def forward_batch(model, batch, device, has_meta):
    dtype = next(model.parameters()).dtype
    bin_x = batch["binaryInputNCHW"].to(dtype)
    glob_x = batch["globalInputNC"].to(dtype)
    meta_x = batch["metadataInputNC"].to(dtype) if has_meta else None
    outputs = model(bin_x, glob_x, input_meta=meta_x)
    outputs = tuple(tuple(t.float() for t in head) if isinstance(head, tuple) else head.float()
                    for head in outputs)
    return model.postprocess_output(outputs)


def agreement(base, var, batch, pos_len=19):
    hw = pos_len * pos_len
    mask = batch["binaryInputNCHW"][:, 0].flatten(1)
    pmask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1).bool()
    out = {}
    p_base = torch.softmax(base[0][:, 0, :], dim=1) * pmask
    p_var = torch.softmax(var[0][:, 0, :], dim=1) * pmask
    p_base = p_base / p_base.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p_var = p_var / p_var.sum(dim=1, keepdim=True).clamp_min(1e-12)
    out["policy_kl"] = (p_base * (torch.log(p_base.clamp_min(1e-12)) - torch.log(p_var.clamp_min(1e-12)))).sum(dim=1).mean().item()
    out["top1_agree"] = (p_base.argmax(dim=1) == p_var.argmax(dim=1)).float().mean().item()
    v_base = torch.softmax(base[1], dim=1)[:, 0]
    v_var = torch.softmax(var[1], dim=1)[:, 0]
    out["win_mae"] = (v_base - v_var).abs().mean().item()
    out["score_mae"] = (base[8] - var[8]).abs().mean().item()
    own_base = torch.tanh(base[4]).flatten(1)
    own_var = torch.tanh(var[4]).flatten(1)
    out["own_rmse"] = torch.sqrt(((own_base - own_var) ** 2).mean()).item()
    return out


MAIN_METRICS = ["loss_sum", "p0loss", "vloss", "oloss", "pacc1"]
LOSS_KEYS = {"loss_sum": "loss_sum", "p0loss": "p0loss_sum", "vloss": "vloss_sum",
             "oloss": "oloss_sum", "pacc1": "pacc1_sum"}


def eval_model(model, metrics_obj, eval_batches, base_outputs, device, has_meta):
    acc, w_acc, agree_acc = defaultdict(float), defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for i, b in enumerate(eval_batches):
            b = batch_to(b, device)
            out = forward_batch(model, b, device, has_meta)
            m = metrics_obj.metrics_dict_batchwise(
                model, out, None, b, is_training=False,
                soft_policy_weight_scale=1.0, disable_optimistic_policy=False,
                meta_kata_only_soft_policy=False, value_loss_scale=1.0,
                td_value_loss_scales=[0.4, 0.4, 0.4], seki_loss_scale=1.0,
                variance_time_loss_scale=1.0, main_loss_scale=1.0,
                intermediate_loss_scale=None, include_model_norms=False,
            )
            m = {k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in m.items()}
            n = b["binaryInputNCHW"].shape[0]
            for disp, src in LOSS_KEYS.items():
                if src in m:
                    acc[disp] += m[src] if isinstance(m[src], (int, float)) else 0
                    w_acc[disp] += n
            if base_outputs is not None:
                a = agreement(base_outputs[i], out[0], b)
                for k, v in a.items():
                    agree_acc[k] += v * n
    metrics = {k: round(acc[k] / w_acc[k], 6) for k in acc}
    agree = {k: round(v / (w_acc.get("loss_sum", 1)), 6) for k, v in agree_acc.items()}
    return metrics, agree


def calibrate_clip_int4(model, quantizable, calib_batches, device, has_meta,
                        candidates=(1.0, 0.98, 0.95, 0.90, 0.85, 0.80)):
    best_frac = {}
    for ci, (name, mod) in enumerate(quantizable):
        orig = mod.weight.detach().clone()
        refs = []
        for b in calib_batches:
            b = batch_to(b, device)
            h = mod.register_forward_hook(lambda m, i, o: refs.append(o.detach().clone()))
            forward_batch(model, b, device, has_meta)
            h.remove()
        best, best_mse = 1.0, None
        for frac in candidates:
            mod.weight.data.copy_(quant_weight(orig, 4, frac)[0])
            sq, cnt = 0.0, 0
            for bi, b in enumerate(calib_batches):
                b = batch_to(b, device)
                outs = []
                h = mod.register_forward_hook(lambda m, i, o: outs.append(o.detach()))
                forward_batch(model, b, device, has_meta)
                h.remove()
                diff = outs[0] - refs[bi]
                sq += diff.pow(2).sum().item()
                cnt += diff.numel()
            if best_mse is None or sq / cnt < best_mse:
                best_mse, best = sq / cnt, frac
        mod.weight.data.copy_(orig)
        best_frac[name] = best
        if (ci + 1) % 24 == 0:
            log(f"  clip 校准 {ci + 1}/{len(quantizable)}")
    return best_frac


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)

    log("加载 checkpoint ...")
    model, swa_model, _ = km.load_model(CKPT, use_swa=False, device=device, pos_len=19)
    model.eval()
    model.disable_fused_attention_kernels()
    model_config = model.config
    has_meta = model.get_has_metadata_encoder()
    metrics_obj = Metrics(1, model)
    original_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    quantizable = find_quantizable(model)
    w_elems = sum(m.weight.numel() for _, m in quantizable)
    log(f"可量化层 {len(quantizable)},权重元素 {w_elems}(int4 ≈ {w_elems/2/1e6:.2f} MB vs int8 {w_elems/1e6:.2f} MB)")

    all_files = sorted(glob.glob(DATA_GLOB))
    rng = random.Random(20261006)
    rng.shuffle(all_files)
    calib_files = all_files[:60]
    eval_files = all_files[60:260]
    log(f"校准 {len(calib_files)} 局,评估 {len(eval_files)} 局")
    calib_batches = materialize(calib_files, 8, 128, model_config, has_meta)
    eval_batches = materialize(eval_files, 64, 128, model_config, has_meta)
    log(f"校准 {len(calib_batches)} batch / 评估 {len(eval_batches)} batch")

    results = {}
    log("评估 fp32 基线 ...")
    base_metrics, _ = eval_model(model, metrics_obj, eval_batches, None, device, has_meta)
    base_outputs = []
    with torch.no_grad():
        for b in eval_batches:
            base_outputs.append(forward_batch(model, batch_to(b, device), device, has_meta)[0])
    results["fp32"] = base_metrics
    log(f"[fp32] loss_sum={base_metrics.get('loss_sum')}")

    def eval_quant(sd_fn, name):
        sd = {}
        for k, v in original_sd.items():
            sd[k] = v
        for n, mod in quantizable:
            frac = clip_fracs.get(n, 1.0)
            q, _ = quant_weight(mod.weight.detach(), 4, frac)
            sd[n + ".weight"] = q.to(torch.float16)
        model.load_state_dict(sd)
        model.to(torch.float16)
        for pn, p in model.named_parameters():
            if pn.startswith(("policy_head.", "value_head.")):
                p.data = p.data.float()
        for pn, pbuf in model.named_buffers():
            if pn.startswith(("policy_head.", "value_head.")):
                pbuf.data = pbuf.data.float()
        m, a = eval_model(model, metrics_obj, eval_batches, base_outputs, device, has_meta)
        results[name] = m
        results[name + "_agree"] = a
        log(f"[{name}] loss_sum={m.get('loss_sum')} Δ={m.get('loss_sum', 0) - base_metrics['loss_sum']:+.4f} "
            f"top1={a.get('top1_agree', 0) * 100:.2f}% KL={a.get('policy_kl', 0):.2e}")
        model.float()
        model.load_state_dict(original_sd)

    log("评估 int4w(minmax)...")
    clip_fracs = {}
    eval_quant(None, "int4w")

    log("校准:逐层 int4 裁剪搜索 ...")
    clip_fracs = calibrate_clip_int4(model, quantizable, calib_batches[:4], device, has_meta)
    log("评估 int4w_clip ...")
    eval_quant(None, "int4w_clip")

    log(f"QAT:STE 微调 {QAT_STEPS} 步(lr={QAT_LR},bs={QAT_BS})...")
    model.float()
    model.load_state_dict(original_sd)
    model.train()
    scales = {n: quant_weight(mod.weight.detach(), 4, 1.0)[1] for n, mod in quantizable}
    wrappers = {}
    for n, mod in quantizable:
        parts = n.split(".")
        parent = model.get_submodule(".".join(parts[:-1]))
        w = STEWrap(mod, scales[n].float())
        setattr(parent, parts[-1], w)
        wrappers[n] = (parent, parts[-1], mod)
    opt = torch.optim.AdamW(model.parameters(), lr=QAT_LR, weight_decay=1e-4)
    qat_batches = materialize(calib_files, 40, QAT_BS, model_config, has_meta)
    if not qat_batches:
        raise SystemExit("QAT batch 不足")
    t0 = time.time()
    loss_hist = []
    for step in range(QAT_STEPS):
        b = batch_to(qat_batches[step % len(qat_batches)], device)
        out = forward_batch(model, b, device, has_meta)
        m = metrics_obj.metrics_dict_batchwise(
            model, out, None, b, is_training=True,
            soft_policy_weight_scale=1.0, disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False, value_loss_scale=1.0,
            td_value_loss_scales=[0.4, 0.4, 0.4], seki_loss_scale=1.0,
            variance_time_loss_scale=1.0, main_loss_scale=1.0,
            intermediate_loss_scale=None, include_model_norms=False,
        )
        loss = m["loss_sum"] if torch.is_tensor(m["loss_sum"]) else torch.tensor(m["loss_sum"], device=device, requires_grad=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        if (step + 1) % 100 == 0:
            log(f"  step {step + 1}/{QAT_STEPS} loss_sum={loss.item():.4f}(近100均值 {sum(loss_hist[-100:]) / 100:.4f}){'' if step != QAT_STEPS - 1 else f' 用时 {time.time() - t0:.0f}s'}")
    # 解包:训练在 mod.weight 原位进行,解包后 state_dict 直接就是训练后的权重
    for n, (parent, leaf, mod) in wrappers.items():
        setattr(parent, leaf, mod)
    qat_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.eval()

    log("评估 int4qat(硬量化到 int4 网格,部署形态)...")
    sd = {k: v.clone() for k, v in qat_sd.items()}   # QAT 训练后的全部参数(头/norm 一并)
    for k in list(sd.keys()):
        if k.endswith(".weight"):
            base_name = k[:-len(".weight")]
            if base_name in scales:
                q, _ = quant_weight(sd[k].float(), 4, 1.0)
                sd[k] = q.to(torch.float16)
    model.load_state_dict(sd)
    model.to(torch.float16)
    for pn, p in model.named_parameters():
        if pn.startswith(("policy_head.", "value_head.")):
            p.data = p.data.float()
    for pn, pbuf in model.named_buffers():
        if pn.startswith(("policy_head.", "value_head.")):
            pbuf.data = pbuf.data.float()
    m, a = eval_model(model, metrics_obj, eval_batches, base_outputs, device, has_meta)
    results["int4qat"] = m
    results["int4qat_agree"] = a
    log(f"[int4qat] loss_sum={m.get('loss_sum')} Δ={m.get('loss_sum', 0) - base_metrics['loss_sum']:+.4f} "
        f"top1={a.get('top1_agree', 0) * 100:.2f}% KL={a.get('policy_kl', 0):.2e}")

    log("QAT 后无损微调对照(硬量化模型再训,即 QAT-then-PTQ 逆,可跳过)") if False else None
    report = {"fp32": results["fp32"], "int4w": results["int4w"], "int4w_clip": results["int4w_clip"],
              "int4qat": results["int4qat"], "int4w_clip_agree": results.get("int4w_clip_agree"),
              "int4w_agree": results.get("int4w_agree"), "int4qat_agree": results.get("int4qat_agree"),
              "qat_steps": QAT_STEPS, "qat_lr": QAT_LR,
              "int8_reference": {"top1_agree": 0.9788, "policy_kl": 8.2e-4, "win_mae": 0.0056, "delta_p0loss": 0.0008}}
    with open(os.path.join(OUT_DIR, "int4_qat_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("报告写入 " + os.path.join(OUT_DIR, "int4_qat_report.json"))


if __name__ == "__main__":
    main()
