#!/usr/bin/env python3
"""行协议推理服务:stdin 每行一个 JSON 批,stdout 每行回同序结果批。
供 Node 侧(nn 搜索)在无 onnxruntime-node 的环境里做端到端验证:
    python3 ort_server.py <model.onnx>
协议:请求 {"rows":[{"spatial":[22*L*L],"global":[19]},...]},盘径 L 从首行
     长度自适应(9 路 22*81 / 19 路 22*361),
     响应 [{"policy":[L*L],"policyPass":f,"winLoss":f,"scoreLead":f,
          "ownership":[L*L]}, ...](scoreLead 为行棋方视角网端目差,可能为 null)
"""
import json
import math
import os
import sys

import numpy as np
import onnxruntime as ort

model_path = sys.argv[1]

# GPU 优先:CUDA EP 靠 dlopen 找 cuda/cudnn,先把 bleed torch 自带的
# nvidia 库用全路径 ctypes 预载(进程启动后再改 LD_LIBRARY_PATH 无效)
def _preload_cuda_libs():
    import glob
    import ctypes
    for pat in ('cublas/lib/libcublas.so*', 'cublas/lib/libcublasLt.so*',
                'cuda_runtime/lib/libcudart.so*', 'cudnn/lib/libcudnn.so*',
                'cufft/lib/libcufft.so*', 'nvrtc/lib/libnvrtc.so*',
                'curand/lib/libcurand.so*', 'cusparse/lib/libcusparse.so*',
                'nvjitlink/lib/libnvJitLink.so*'):
        for p in sorted(glob.glob(f'/home/a/miniconda3/envs/bleed/lib/python3.10/site-packages/nvidia/{pat}')):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

_want_gpu = os.environ.get('ORT_CPU') != '1'
if _want_gpu:
    _preload_cuda_libs()
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_provs = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if _want_gpu
          else ['CPUExecutionProvider'])
sess = ort.InferenceSession(model_path, so, providers=_provs)
sys.stderr.write(f"[ort_server] 实际后端: {sess.get_providers()}\n")
sys.stderr.flush()
names_in = [i.name for i in sess.get_inputs()]
names_out = [o.name for o in sess.get_outputs()]
POL = next(n for n in names_out if 'Policy' in n and 'Pass' not in n)
POLP = next(n for n in names_out if 'PolicyPass' in n)
VAL = next(n for n in names_out if 'Value' in n and 'Score' not in n)
OWN = next((n for n in names_out if 'Ownership' in n), None)
SV = next((n for n in names_out if 'ScoreValue' in n), None)

L = -1                                             # 盘径(从首行自适应)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rows = req['rows']
    n = len(rows)
    if L < 0:
        L = math.isqrt(len(rows[0]['spatial']) // 22)
        sys.stderr.write(f"[ort_server] 盘径 L={L}\n")
        sys.stderr.flush()
    sp = np.array([r['spatial'] for r in rows], dtype=np.float32).reshape(n, 22, L, L)
    gl = np.array([r['global'] for r in rows], dtype=np.float32).reshape(n, 19, 1, 1)
    mk = np.ones((n, 1, L, L), dtype=np.float32)
    feeds = {names_in[0]: sp, names_in[1]: gl, names_in[2]: mk}
    out = sess.run(None, feeds)
    pol = out[names_out.index(POL)]
    polp = out[names_out.index(POLP)]
    val = out[names_out.index(VAL)]
    own = out[names_out.index(OWN)] if OWN else None
    sv = out[names_out.index(SV)] if SV else None
    polc = pol.shape[1] // (L * L) if pol.ndim == 4 else 1
    passc = max(1, polp.size // n)
    res = []
    for i in range(n):
        logits = val[i]
        m = logits.max()
        e = np.exp(logits - m)
        p = e / e.sum()
        res.append({
            'policy': [float(x) for x in pol[i, :L * L].reshape(-1)] if pol.ndim == 4 else [float(x) for x in pol[i][:L * L]],
            'policyPass': float(polp.reshape(n, -1)[i, 0]),
            'winLoss': float(p[0] - p[1]),
            # 网端 lead:OutputScoreValue 通道 2 × leadMultiplier(20,desc.cpp v≥13
            # 默认,b6c96 v8 与 b8c96h3tfrs v17 头均核实)。行棋方视角,已含贴目。
            'scoreLead': float(sv[i][2]) * 20.0 if sv is not None else None,
            'ownership': [float(x) for x in own[i, 0].reshape(-1)] if own is not None else None,
        })
    sys.stdout.write(json.dumps(res) + '\n')
    sys.stdout.flush()
