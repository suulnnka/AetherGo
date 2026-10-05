#!/usr/bin/env python3
"""完整 session.evalBatch 契约的行协议推理服务(AetherGo 对战驱动用)。
每行请求 {"rows":[{"spatial":[...], "global":[...], "optimism":f}, ...]}
回同序:[{policy(插值后 logits), policyPass, winLoss, scoreMean, scoreStdev,
        scoreLead, shorttermWinlossError, shorttermScoreError, ownership}]
后处理数值口径与 src/nn/session.js 完全一致(20/20/20、sqrt150、0.5、softplus)。
"""
import json, math, sys
import numpy as np
import glob, ctypes
for pat in ("cublas/lib/libcublas.so*","cublas/lib/libcublasLt.so*","cuda_runtime/lib/libcudart.so*",
            "cudnn/lib/libcudnn.so*","cufft/lib/libcufft.so*","nvrtc/lib/libnvrtc.so*",
            "curand/lib/libcurand.so*","cusparse/lib/libcusparse.so*","nvjitlink/lib/libnvJitLink.so*"):
    for p in sorted(glob.glob("/home/a/miniconda3/envs/bleed/lib/python3.10/site-packages/nvidia/" + pat)):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError: pass
import onnxruntime as ort


model_path = sys.argv[1]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
sys.stderr.write("[server] providers: %s\n" % sess.get_providers()); sys.stderr.flush()
names_in = [i.name for i in sess.get_inputs()]
names_out = [o.name for o in sess.get_outputs()]
POL  = next(n for n in names_out if "Policy" in n and "Pass" not in n)
POLP = next(n for n in names_out if "PolicyPass" in n)
VAL  = next(n for n in names_out if "Value" in n and "Score" not in n)
OWN  = next(n for n in names_out if "Ownership" in n)
SV   = next(n for n in names_out if "ScoreValue" in n)

def softplus(x): return x if x > 30 else math.log1p(math.exp(x))
N2 = 361
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rows = req["rows"]
    n = len(rows)
    sp = np.array([r["spatial"] for r in rows], dtype=np.float32).reshape(n, 22, 19, 19)
    gl = np.array([r["global"] for r in rows], dtype=np.float32).reshape(n, 19, 1, 1)
    mk = np.ones((n, 1, 19, 19), dtype=np.float32)
    out = sess.run(None, {names_in[0]: sp, names_in[1]: gl, names_in[2]: mk})
    pol = out[names_out.index(POL)]; polp = out[names_out.index(POLP)]
    val = out[names_out.index(VAL)]; own = out[names_out.index(OWN)]
    sv  = out[names_out.index(SV)]
    polc = pol[0].size // N2
    valf = val.reshape(n, -1)
    svf = sv.reshape(n, -1)
    passc = max(1, polp.size // n)
    res = []
    for i in range(n):
        lam = rows[i].get("optimism", 1.0)
        def interp(ch, base, optv):
            return base if (polc < 2 or lam == 1.0) else base + (optv - base) * lam
        polf = pol[i].reshape(polc, N2) if pol.ndim == 4 else pol[i][:N2]
        p0 = polf[0]
        if polc >= 2 and lam != 1.0:
            policy = [float(p0[p] + (polf[1][p] - p0[p]) * lam) for p in range(N2)]
        else:
            policy = [float(x) for x in p0]
        pb = float(polp.reshape(n, -1)[i, 0])
        if polc >= 2 and lam != 1.0:
            po = float(polp.reshape(n, -1)[i, 1]); policyPass = pb + (po - pb) * lam
        else:
            policyPass = pb
        l0, l1 = float(valf[i][0]), float(valf[i][1])
        m = max(l0, l1); e0 = math.exp(l0 - m); e1 = math.exp(l1 - m)
        b = 0
        res.append({
            "policy": policy, "policyPass": policyPass,
            "winLoss": (e0 - e1) / (e0 + e1),
            "scoreMean": float(svf[i][0]) * 20.0,
            "scoreStdev": softplus(float(svf[i][1])) * 20.0,
            "scoreLead": float(svf[i][2]) * 20.0,
            "shorttermWinlossError": softplus(float(svf[i][4]) * 0.5) * 0.5,
            "shorttermScoreError": softplus(float(svf[i][5]) * 0.5) * math.sqrt(150),
            "ownership": [float(x) for x in own[i][0].reshape(-1)],
        })
    sys.stdout.write(json.dumps(res) + "\n"); sys.stdout.flush()
