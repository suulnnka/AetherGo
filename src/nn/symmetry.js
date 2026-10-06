/* ============================================================
 * AetherGo NN —— 8 对称置换(KataGo nnEvaluator 对称同款)
 *
 * 从 search.js 抽出:自研 WebGPU 引擎(src/nn/webgpu/)要在打包/初始化期
 * 生成「逐对称的输入 gather 表」(stem 卷积直接按变换后坐标取输入),
 * search.js 与 webgpu 两侧共用同一份定义,避免循环 import。
 *
 * SYM8[s][p] = q:恒等坐标系位置 p 经 s 号对称变换后的位置。
 * 约定与 nninputs.cpp 一致:特征按 dst[perm[p]] = src[p] 前向变换,
 * 输出按 policy[p] = out.policy[perm[p]] 逆变换(unpermuteOut)。
 * ============================================================ */
import { N, N2 } from '../engine.js';

export const SYM8 = (() => {
  const syms = [];
  for (let s = 0; s < 8; s++) {
    const map = new Int32Array(N2);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        let rr = r, cc = c;
        for (let k = 0; k < (s & 3); k++) { const t = rr; rr = cc; cc = N - 1 - t; }
        if (s & 4) cc = N - 1 - cc;
        map[r * N + c] = rr * N + cc;
      }
    }
    syms.push(map);
  }
  return syms;
})();

/** 前向变换:dst[perm[p]] = src[p](通道整体,spatial NCHW 平面) */
export function permuteSpatial(src, perm, dst) {
  const ch = (src.length / N2) | 0;
  for (let c = 0; c < ch; c++) {
    const off = c * N2;
    for (let p = 0; p < N2; p++) dst[off + perm[p]] = src[off + p];
  }
  return dst;
}

/** 输出逆变换:policy[p] = out.policy[perm[p]](ownership 同) */
export function unpermuteOut(out, perm) {
  const policy = new Float32Array(N2);
  for (let p = 0; p < N2; p++) policy[p] = out.policy[perm[p]];
  const ownership = out.ownership ? (() => {
    const o = new Float32Array(N2);
    for (let p = 0; p < N2; p++) o[p] = out.ownership[perm[p]];
    return o;
  })() : out.ownership;
  return { ...out, policy, ownership };
}

/**
 * stem 卷积的输入 gather 表(自研引擎专用)。
 * 变换后局面在位置 q 的 3×3 邻域 = 原局面在 invs[q+δ] 的取值,
 * 其中 invs = SYM8[s] 的逆置换。返回 Uint32Array(8 * N2 * 9):
 *   table[(s*N2 + q)*9 + δ] = inv_s[q 的 δ 邻居] 或 ZERO_SLOT(盘外)。
 * ZERO_SLOT 指向上传缓冲末尾追加的恒 0 槽位(见 webgpu/session.js)。
 */
export function buildStemGatherTable(zeroSlot) {
  const table = new Uint32Array(8 * N2 * 9);
  for (let s = 0; s < 8; s++) {
    const perm = SYM8[s];
    const inv = new Int32Array(N2);
    for (let p = 0; p < N2; p++) inv[perm[p]] = p;
    for (let q = 0; q < N2; q++) {
      const r = (q / N) | 0, c = q % N;
      for (let d = 0; d < 9; d++) {
        const dr = ((d / 3) | 0) - 1, dc = (d % 3) - 1;
        const rr = r + dr, cc = c + dc;
        table[(s * N2 + q) * 9 + d] =
          (rr >= 0 && rr < N && cc >= 0 && cc < N) ? inv[rr * N + cc] : zeroSlot;
      }
    }
  }
  return table;
}
