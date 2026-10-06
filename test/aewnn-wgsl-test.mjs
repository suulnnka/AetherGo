/* aewnn WGSL 内核 vs CPU 参考解释器对拍(同计划、同权重、同输入)。
 *
 * cpuref 已由 test/aewnn-cpuref-test.mjs 对拍过 ort golden;本文件把同一批
 * 输入喂给真实 WGSL 内核(Dawn),验证内核数学与宿主(bindgroup 布局、
 * uniform 填充、读写缓冲规划)。容差按「累加序差异量级」定:GEMM 平铺与
 * flash attention 在线 softmax 都会重排浮点累加序。
 *
 * 运行:node test/aewnn-wgsl-test.mjs(需 webgpu 包:node_modules/
 *   npm install --no-save webgpu  # Dawn 预编译二进制,仅测试用)
 */
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

let dawn;
try {
  const require2 = createRequire(import.meta.url);
  const pkg = require2('webgpu');
  dawn = pkg;
} catch {
  console.error('需要 Dawn 的 Node 绑定:npm install --no-save webgpu(仅测试依赖)');
  process.exit(77);                                   // SKIP
}
const { create, globals } = dawn;
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: create([]) }, configurable: true });

const { N, N2, BLACK, WHITE, PASS, newBoard, make } = await import(join(ROOT, 'src/engine.js'));
const { encodeFeatures } = await import(join(ROOT, 'src/nn/features.js'));
const { SYM8 } = await import(join(ROOT, 'src/nn/symmetry.js'));
const { createCpuRefSession } = await import(join(ROOT, 'src/nn/webgpu/cpuref.js'));
const { createAewnnSession } = await import(join(ROOT, 'src/nn/webgpu/session.js'));

let failed = 0;
const check = (name, cond, extra) => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  — ' + (extra ?? '')}`);
  if (!cond) failed++;
};

const blobBuf = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.aewn'));
const blob = blobBuf.buffer.slice(blobBuf.byteOffset, blobBuf.byteOffset + blobBuf.byteLength);

const cpu = await createCpuRefSession(blob);
const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });

/* 局面:空盘 + 中盘 + 官子(与 cpuref 测试同源口径) */
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
const tags = [];
for (const { bd, side, moves } of POSITIONS) {
  for (const sym of [0, 2, 5, 7]) {
    rows.push({ bd, side, moves: moves.slice(), sym, optimism: sym === 0 ? 0.2 : 1.0 });
    tags.push(`sym${sym}`);
  }
}

const fbufs = rows.map(() => encodeFeatures(newBoard(), BLACK, {}));
const enc = rows.map((r) => encodeFeatures(r.bd, r.side, { recentMoves: r.moves, komi: 7.5 }));
const inRows = rows.map((r, i) => ({ spatial: enc[i].spatial, global: enc[i].global, sym: r.sym, optimism: r.optimism }));

const [cpuOut, gpuOut] = await Promise.all([cpu.evalBatch(inRows), gpu.evalBatch(inRows)]);

const TOL = { policy: 2e-4, policyPass: 2e-4, winLoss: 1e-4, score: 2e-3, ownership: 2e-4 };
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
console.log(`WGSL vs cpuref 最差偏差: policy=${worst.policy.toExponential(2)} winLoss=${worst.winLoss.toExponential(2)} ownership=${worst.ownership.toExponential(2)}`);
console.log(`dispatches: ${gpu.dispatchCount}`);

gpu.dispose();
void fbufs; void existsSync;
process.exit(failed ? 1 : 0);
