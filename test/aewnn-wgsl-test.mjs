/* aewnn WGSL 内核 vs CPU 参考解释器对拍(同计划、同权重、同输入,i8f16 口径)。
 *
 * cpuref-Q 的「纯权重精度效应」已由 test/aewnn-quant-test.mjs 第 2 层对拍过
 * ort golden;本文件把同一批输入喂给真实 WGSL 内核(Dawn),验证内核数学与
 * 宿主(bindgroup 布局、uniform 填充、读写缓冲规划)。容差按「f16 存储 +
 * 累加序差异」量级定(同 quant-test 第 3 层闸门):GEMM 平铺与 flash
 * attention 在线 softmax 都会重排浮点累加序,f16 存储将其放大。
 *
 * 运行:node test/aewnn-wgsl-test.mjs(需 webgpu 包:node_modules/
 *   npm install --no-save webgpu  # Dawn 预编译二进制,仅测试用)
 */
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

let dawn;
try {
  const require2 = createRequire(import.meta.url);
  dawn = require2('webgpu');
} catch {
  console.error('需要 Dawn 的 Node 绑定:npm install --no-save webgpu(仅测试依赖)');
  process.exit(77);                                   // SKIP
}
const { create, globals } = dawn;
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: create([]) }, configurable: true });

const { N, N2, BLACK, WHITE, newBoard, make } = await import(pathToFileURL(join(ROOT, 'src/engine.js')).href);
const { encodeFeatures } = await import(pathToFileURL(join(ROOT, 'src/nn/features.js')).href);
const { createCpuRefSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/cpuref.js')).href);
const { createAewnnSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/session.js')).href);

let failed = 0;
const check = (name, cond, extra) => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  — ' + (extra ?? '')}`);
  if (!cond) failed++;
};

import { ensureBlob } from './blob-helper.mjs';
const blob = ensureBlob('b8c96h3tfrs_19.i8.aewn');

const cpu = await createCpuRefSession(blob);
const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });

/* 局面:空盘 + 中盘 + 官子(与 quant-test 同源口径) */
const SEQ_MID = [
  [3, 3], [15, 15], [3, 15], [15, 3], [9, 9], [3, 9], [15, 9], [9, 3],
  [9, 15], [5, 5], [13, 13], [5, 13], [13, 5], [7, 7], [11, 11], [7, 11],
  [11, 7], [2, 8], [16, 8], [8, 2], [8, 16],
];
const pos = newBoard();
for (let i = 0; i < SEQ_MID.length; i++) {
  const [r, c] = SEQ_MID[i];
  make(pos, r * N + c, i % 2 === 0 ? BLACK : WHITE);
}
const posEnd = newBoard();
const SEQ_END = [[0, 0], [0, 1], [18, 18], [18, 17], [1, 1], [17, 17], [9, 9], [9, 10], [10, 9], [10, 10]];
for (let i = 0; i < SEQ_END.length; i++) {
  const [r, c] = SEQ_END[i];
  make(posEnd, r * N + c, i % 2 === 0 ? BLACK : WHITE);
}
const POSITIONS = [
  { bd: newBoard(), side: BLACK, moves: [] },
  { bd: pos, side: WHITE, moves: SEQ_MID.map(([r, c]) => r * N + c) },
  { bd: posEnd, side: BLACK, moves: SEQ_END.map(([r, c]) => r * N + c) },
];

/* 测试组:混批(不同位置 × 不同 sym × λ)一次下发 */
const rows = [];
for (const { bd, side, moves } of POSITIONS) {
  for (const sym of [0, 2, 5, 7]) {
    rows.push({ bd, side, moves: moves.slice(), sym, optimism: sym === 0 ? 0.2 : 1.0 });
  }
}

const enc = rows.map((r) => encodeFeatures(r.bd, r.side, { recentMoves: r.moves, komi: 7.5 }));
const inRows = rows.map((r, i) => ({ spatial: enc[i].spatial, global: enc[i].global, sym: r.sym, optimism: r.optimism }));

const [cpuOut, gpuOut] = await Promise.all([cpu.evalBatch(inRows), gpu.evalBatch(inRows)]);

/* f16 存储 + 累加序重排的量级带(同 quant-test L3 闸门口径) */
const TOL = { policy: 5e-2, policyPass: 5e-2, winLoss: 5e-3, score: 0.15, ownership: 2e-2 };
let worst = { policy: 0, winLoss: 0, ownership: 0 };
rows.forEach((r, i) => {
  const c = cpuOut[i], g = gpuOut[i];
  const tag = `pos${i}(sym${r.sym},λ${r.optimism})`;
  let dp = 0;
  for (let p = 0; p < N2; p++) dp = Math.max(dp, Math.abs(c.policy[p] - g.policy[p]));
  let dov = 0;
  for (let p = 0; p < N2; p++) dov = Math.max(dov, Math.abs(c.ownership[p] - g.ownership[p]));
  const dw = Math.abs(c.winLoss - g.winLoss);
  worst.policy = Math.max(worst.policy, dp);
  worst.winLoss = Math.max(worst.winLoss, dw);
  worst.ownership = Math.max(worst.ownership, dov);
  check(`${tag} policy`, dp < TOL.policy, `max|Δ|=${dp.toExponential(2)}`);
  check(`${tag} policyPass`, Math.abs(c.policyPass - g.policyPass) < TOL.policyPass,
    `${c.policyPass.toFixed(4)} vs ${g.policyPass.toFixed(4)}`);
  check(`${tag} winLoss`, dw < TOL.winLoss, `|Δ|=${dw.toExponential(2)}`);
  check(`${tag} scoreLead`, Math.abs(c.scoreLead - g.scoreLead) < TOL.score,
    `${c.scoreLead.toFixed(3)} vs ${g.scoreLead.toFixed(3)}`);
  check(`${tag} scoreMean`, Math.abs(c.scoreMean - g.scoreMean) < TOL.score);
  check(`${tag} ownership`, dov < TOL.ownership, `max|Δ|=${dov.toExponential(2)}`);
});
console.log(`WGSL-Q vs cpuref-Q 最差偏差: policy=${worst.policy.toExponential(2)} winLoss=${worst.winLoss.toExponential(2)} ownership=${worst.ownership.toExponential(2)}`);
console.log(`dispatches: ${gpu.dispatchCount}`);

/* ==================== B 系(高批)路径 ====================
 * 同一输入:低批(n<8,原内核)与高批(n≥8,B 系)必须各自对拍 cpuref-Q。
 * GEMM B 系累加序与低批一致;flashB 重排累加序 → 交叉比对按同容差判定
 * (非逐位)。边界批(末 workgroup 不满 R 行)不越界。 */
{
  const diffRow = (a, b) => {
    let d = 0;
    for (let p = 0; p < N2; p++) {
      d = Math.max(d, Math.abs(a.policy[p] - b.policy[p]), Math.abs(a.ownership[p] - b.ownership[p]));
    }
    return Math.max(d, Math.abs(a.winLoss - b.winLoss), Math.abs(a.policyPass - b.policyPass),
      Math.abs(a.scoreLead - b.scoreLead), Math.abs(a.scoreMean - b.scoreMean));
  };

  /* 低批路径回归:n=4(原内核)对拍 cpuref-Q */
  const lowOut = await gpu.evalBatch(inRows.slice(0, 4));
  let lowWorst = 0;
  for (let i = 0; i < 4; i++) lowWorst = Math.max(lowWorst, diffRow(cpuOut[i], lowOut[i]));
  check(`B系 低批路径(n=4)对拍 cpuref-Q`, lowWorst < TOL.policy, `max|Δ|=${lowWorst.toExponential(2)}`);

  /* 12 行各自单行评估(n=1,低路径)作为交叉基准 */
  const low1 = [];
  for (const r of inRows) low1.push((await gpu.evalBatch([r]))[0]);

  /* 高批:整 R(8)/末组 1 行(9)/跨组(15)/整批倍数(16)/近 CAP(31) */
  for (const n of [8, 9, 15, 16, 31]) {
    const rowsN = Array.from({ length: n }, (_, i) => inRows[i % inRows.length]);
    const [cpuN, gpuN] = await Promise.all([cpu.evalBatch(rowsN), gpu.evalBatch(rowsN)]);
    let worstCpu = 0, worstX = 0;
    for (let i = 0; i < n; i++) {
      worstCpu = Math.max(worstCpu, diffRow(cpuN[i], gpuN[i]));
      worstX = Math.max(worstX, diffRow(low1[i % low1.length], gpuN[i]));
    }
    check(`B系 n=${n} 对拍 cpuref-Q`, worstCpu < TOL.policy, `max|Δ|=${worstCpu.toExponential(2)}`);
    check(`B系 n=${n} 与低批一致(容差)`, worstX < TOL.policy, `max|Δ|=${worstX.toExponential(2)}`);
  }
}

gpu.dispose();
process.exit(failed ? 1 : 0);
