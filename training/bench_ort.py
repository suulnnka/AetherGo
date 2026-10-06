#!/usr/bin/env python3
"""ort 基准:同一张 b8c96h3tfrs_19.onnx,与 test/aewnn-bench.mjs 同口径分批计时。

默认 CPU EP(本机无可用 GPU;浏览器真机 A/B 见 docs/WEBGPU_ENGINE_RESEARCH.md §9.4)。
输出各批档 min-of-10 毫秒与 rows/ms,直接对表 aewnn 基准。

用法:python3 training/bench_ort.py [models/b8c96h3tfrs_19.onnx]
"""
import sys
import time

import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/b8c96h3tfrs_19.onnx'

t0 = time.perf_counter()
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
t_load = time.perf_counter() - t0
print(f'ort 会话创建(CPU EP,含图优化): {t_load * 1000:.0f}ms;providers={sess.get_providers()}')

HW, SP, GL = 361, 22, 19
sp0 = np.zeros((1, SP, 19, 19), np.float32)
sp0[0, 0] = 1.0
gl0 = np.zeros((1, GL, 1, 1), np.float32)
gl0[0, 5] = 7.5 / 20
mk0 = np.ones((1, 1, 19, 19), np.float32)

for n in (1, 2, 4, 8, 16, 32):
    sp = np.repeat(sp0, n, axis=0)
    gl = np.repeat(gl0, n, axis=0)
    mk = np.repeat(mk0, n, axis=0)
    feeds = {sess.get_inputs()[0].name: sp, sess.get_inputs()[1].name: gl, sess.get_inputs()[2].name: mk}
    sess.run(None, feeds)                              # 预热
    best = float('inf')
    for _ in range(10):
        t = time.perf_counter()
        sess.run(None, feeds)
        best = min(best, time.perf_counter() - t)
    print(f'批 {n:2d}: 总 {best * 1000:8.2f}ms = {n / best:8.2f} rows/ms')
