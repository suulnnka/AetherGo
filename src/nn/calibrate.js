/* ============================================================
 * AetherGo NN 批大小校准(KataGo benchmark 的加载时自动化)
 *
 * 从 session.js 抽出:ort 路径与自研 aewnn 路径共用同一测速器与同口径
 * 的 maxBatch 语义(吞吐 ≥ 最优 90% 的最小批),天然成为两后端 A/B 的
 * 标准测速器。evalBatch 契约见 src/nn/session.js。
 * ============================================================ */
import { N2 } from '../engine.js';

/** 纯选择:entries = [[size, rows/ms]...](升序)→ 最优吞吐 × tol 的最小 size */
export function pickBatchSizeFromThroughput(entries, tol = 0.9) {
  let best = 0;
  for (const [, t] of entries) if (t > best) best = t;
  const threshold = best * tol;
  for (const [size, t] of entries) if (t >= threshold) return size;
  return entries.length ? entries[0][0] : null;
}

/** 计时校准:对 evalBatch 逐档测吞吐,返回批上限。timeBudgetMs 兜底慢设备。 */
export async function calibrateMaxBatch(evalBatch, opt = {}) {
  const sizes = opt.sizes ?? [2, 4, 8, 16, 32];
  const iters = opt.iters ?? 6;
  const now = () => performance.now();
  const deadline = now() + (opt.timeBudgetMs ?? 2500);
  const protoRow = { spatial: new Float32Array(22 * N2), global: new Float32Array(19) };
  const mkRows = (n) => Array.from({ length: n }, () => protoRow);

  const entries = [];
  for (const size of sizes) {
    const rows = mkRows(size);
    const w0 = now();
    await evalBatch(rows);                       // 预热:首推理含 shader/pipeline 编译,丢弃
    if (now() + (now() - w0) > deadline) break;  // 剩余时间不够测下一档:到此为止
    let minMs = Infinity;
    for (let k = 0; k < iters && now() < deadline; k++) {
      const t0 = now();
      await evalBatch(rows);
      const ms = now() - t0;
      if (ms < minMs) minMs = ms;                // min = 最少受调度干扰的纯速度估计
    }
    if (Number.isFinite(minMs) && minMs > 0) entries.push([size, size / minMs]);
  }
  return pickBatchSizeFromThroughput(entries) ?? sizes[0];
}
