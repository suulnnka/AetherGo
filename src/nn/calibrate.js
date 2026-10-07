/* ============================================================
 * AetherGo NN 批大小校准(KataGo benchmark 的加载时自动化)
 *
 * 从 session.js 抽出:ort 路径与自研 aewnn 路径共用同一测速器与同口径
 * 的 maxBatch 语义(吞吐 ≥ 最优 90% 的最小批),天然成为两后端 A/B 的
 * 标准测速器。evalBatch 契约见 src/nn/session.js。
 *
 * 2026-10-07 修复(实测复现的三类失真,见 test/browser-ab/results*.json):
 *   1. 冷启动截断乱选:首推理含一次性管线编译(ORT 整图 ~2s),旧版把它
 *      计入预算 → deadline 截断后 entries 可能只剩 1 档,而「≥90%最优」
 *      对单档恒成立 → 任意选小批(实测 onnx 冷启动选中批 2,真值 16)。
 *      修复:一次性编译以预算前全局预热吸收;单档/空集回落 null,
 *      调用方(session)回落默认批,不再信退化样本集。
 *   2. min-of-N 估计不稳:吞吐高原曲线上相邻档差 <10%,纯 min 对单次
 *      调度抖动过敏(同会话三次校准 16/8/8 漂移)。修复:改取最快 3 次
 *      均值(仍拒慢模式污染,方差约为纯 min 的一半),iters 6→12、预算 2.5s→4s(提升慢模式窗口外的覆盖轮数)。
 *   3. 预算截断档的少样本污染:deadline 尾部只采到 1-2 样本的档会被当
 *      锚点。修复:样本 <3 的档整档丢弃。
 *   4. 慢模式串扰:ORT 的双峰时序(快/慢模式)会连续占据一档的全部样本,
 *      逐档连续采样时档间比值失真(实测 onnx 批16 被整档污染 → 误选批8)。
 *      修复:轮转交错采样(每轮扫全部档),慢模式波及整轮,fast3 取各档
 *      各自的快样本,比值可比。
 *   5. 峰值锚刀口:锚 = 前二快档均值(pickBatchSizeFromThroughput),相邻档
 *      差 < 噪声的高原上阈值不再随单档噪声跳变(批8 判定余量 0.1% → 2%+)。
 * ============================================================ */
import { N2 } from '../engine.js';

/** 纯选择:entries = [[size, rows/ms]...](升序)→ 峰值 × tol 的最小 size。
 * 峰值锚 = 前二快档均值(而非单一最大值):吞吐高原上相邻档差 <噪声时,
 * 单档锚会让阈值跳变、选档随之漂移;前二均值把锚的方差减半,刀口档的
 * 判定余量从 ~0.1% 提到 ~2%。真实悬崖(如 ORT 批32)不受影响。 */
export function pickBatchSizeFromThroughput(entries, tol = 0.9) {
  if (!entries.length) return null;
  const ts = entries.map(([, t]) => t).sort((a, b) => b - a);
  const best = ts.length >= 2 ? (ts[0] + ts[1]) / 2 : ts[0];
  const threshold = best * tol;
  for (const [size, t] of entries) if (t >= threshold) return size;
  return entries[0][0];
}

/**
 * 计时校准:对 evalBatch 逐档测吞吐,返回批上限。
 * 返回 null 表示无法可信校准(单档/空集/异常),调用方应回落默认批。
 */
export async function calibrateMaxBatch(evalBatch, opt = {}) {
  const sizes = opt.sizes ?? [2, 4, 8, 16, 32];
  const iters = opt.iters ?? 12;
  const now = () => performance.now();
  const protoRow = { spatial: new Float32Array(22 * N2), global: new Float32Array(19) };
  const mkRows = (n) => Array.from({ length: n }, () => protoRow);

  /* 一次性管线编译(ORT 整图 / aewnn 84 管线)不计入预算 */
  await evalBatch(mkRows(sizes[0]));
  const deadline = now() + (opt.timeBudgetMs ?? 4000);

  /* 逐档预热(形状 JIT)+ 资格判定 */
  const warm = [];
  for (const size of sizes) {
    const w0 = now();
    await evalBatch(mkRows(size));
    if (now() + (now() - w0) > deadline) break;
    warm.push(size);
  }

  /* 轮转交错采样:每轮扫全部档。慢模式(驱动/调度状态)会连续占据多次
   * 调用 —— 逐档连续采样时整档被污染、档间比值失真;交错后波及的是整轮,
   * fast3 取各档各自的快样本,比值保持可比。 */
  const times = new Map(warm.map((s) => [s, []]));
  for (let k = 0; k < iters && now() < deadline; k++) {
    for (const size of warm) {
      const t0 = now();
      await evalBatch(mkRows(size));
      times.get(size).push(now() - t0);
    }
  }

  const entries = [];
  const minSamples = Math.min(3, iters);              // iters<3 时按 iters 收敛(10e 兼容)
  for (const size of warm) {
    const ts = times.get(size);
    if (ts.length < minSamples) continue;             // 采样过少的档不可信
    ts.sort((a, b) => a - b);
    const kf = Math.min(3, ts.length);
    let fast3 = 0;                                    // 最快 kf 均值:稳,且拒慢模式
    for (let i = 0; i < kf; i++) fast3 += ts[i];
    fast3 /= kf;
    if (fast3 > 0) entries.push([size, size / fast3]);
  }
  /* 单档无法做「≥90%最优」比较(任何值都自动过阈)→ null 让调用方回落默认批 */
  if (entries.length < 2) return null;
  return pickBatchSizeFromThroughput(entries);
}
