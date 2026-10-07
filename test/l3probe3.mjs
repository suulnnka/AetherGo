/* L3 质量探针(临时):WGSL-Q(真实 GPU 输出)vs ort CPU golden 的对弈级指标。
 *
 * 背景:l3probe/l3probe2 已证明 WGSL-Q 内核逐位忠实(f32 模式 2.5e-5;Q 模式
 * stem 出 100% ≤1ULP),但 f16 激活存储与 f32 累加序差异耦合成「舍入边界
 * 混沌」,16 层放大后 WGSL-Q 对 f64 参考的输出偏差 ~1e-2 量级 —— 与 cpuref-Q
 * 的窄容差对拍在原理上不可能通过,L3 闸门需要换成「对 golden 的对弈级指标」。
 * 本探针即用 L2 同款口径(policy KL / top1 / top5 / winLoss MAE / scoreLead
 * MAE / ownership MAE)直接量化 WGSL-Q vs golden,与 L2(int8 权重纯效应)
 * 对比,给出浏览器实际引擎的真实质量带。
 *
 * 运行:PYTHON_BIN=python node test/l3probe3.mjs
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const PY = process.env.PYTHON_BIN ?? 'python';
const MODEL = join(ROOT, 'models/b8c96h3tfrs_19.onnx');

const dawn = await import('webgpu');
Object.assign(globalThis, dawn.globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: dawn.create([]) }, configurable: true });

const { N, N2, BLACK, WHITE, PASS, newBoard, make } = await import(pathToFileURL(join(ROOT, 'src/engine.js')).href);
const { encodeFeatures } = await import(pathToFileURL(join(ROOT, 'src/nn/features.js')).href);
const { SYM8, permuteSpatial } = await import(pathToFileURL(join(ROOT, 'src/nn/symmetry.js')).href);
const { createAewnnSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/session.js')).href);

/* ---------- golden 服务(同 quant-test) ---------- */
const server = spawn(PY, [join(ROOT, 'tools/diff/full_ort_server.py'), MODEL], {
  stdio: ['pipe', 'pipe', 'inherit'],
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '' },
});
let buf = '';
const pending = [];
server.stdout.on('data', (d) => {
  buf += d.toString();
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
    const t = line.trim();
    if (!t) continue;
    if (!t.startsWith('[')) continue;
    pending.shift().resolve(JSON.parse(t));
  }
});
const golden = (rows) => new Promise((resolve, reject) => {
  pending.push({ resolve, reject });
  server.stdin.write(JSON.stringify({ rows }) + '\n');
});

/* ---------- 局面集(与 quant-test 完全同源,30 例) ---------- */
const SEQ_MID = [
  [3, 3], [15, 15], [3, 15], [15, 3], [9, 9], [3, 9], [15, 9], [9, 3],
  [9, 15], [5, 5], [13, 13], [5, 13], [13, 5], [7, 7], [11, 11], [7, 11],
  [11, 7], [2, 8], [16, 8], [8, 2], [8, 16],
];
const SEQ_END = [
  [0, 0], [0, 1], [18, 18], [18, 17], [1, 1], [1, 2], [17, 17], [17, 16],
  [0, 18], [0, 17], [18, 0], [18, 1], [9, 9], [9, 10], [10, 9], [10, 10],
  PASS, [4, 4], [14, 14], [4, 14], [14, 4], PASS,
];
function positions() {
  const out = [{ bd: newBoard(), side: BLACK, moves: [] }];
  const mk = (seq) => {
    const bd = newBoard();
    for (let i = 0; i < seq.length; i++) {
      const mv = seq[i][0] === undefined ? PASS : seq[i][0] * N + seq[i][1];
      make(bd, mv, i % 2 === 0 ? BLACK : WHITE);
    }
    return bd;
  };
  const mvs = (seq) => seq.map((m) => (m[0] === undefined ? PASS : m[0] * N + m[1]));
  out.push({ bd: mk(SEQ_MID.slice(0, 10)), side: BLACK, moves: mvs(SEQ_MID.slice(0, 10)) });
  out.push({ bd: mk(SEQ_MID), side: WHITE, moves: mvs(SEQ_MID) });
  out.push({ bd: mk([...SEQ_MID, ...SEQ_END.slice(0, 12)]), side: BLACK, moves: mvs([...SEQ_MID, ...SEQ_END.slice(0, 12)]) });
  const bd2 = mk(SEQ_END);
  out.push({ bd: bd2, side: WHITE, moves: mvs(SEQ_END) });
  return out;
}
const CASES = [];
for (const { bd, side, moves } of positions()) {
  for (const sym of [0, 1, 5]) {
    for (const optimism of [1.0, 0.2]) {
      CASES.push({ bd, side, moves: moves.slice(), sym, optimism });
    }
  }
}

const blobBuf = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.i8.aewn'));
const blob = blobBuf.buffer.slice(blobBuf.byteOffset, blobBuf.byteOffset + blobBuf.byteLength);
const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });

let sumKL = 0, top1Agree = 0, nearTieFlips = 0, realFlip = 0, top5J = 0;
let sumWL = 0, sumLead = 0, sumOwn = 0, sumPass = 0;
const top1Disagree = [];
for (let c0 = 0; c0 < CASES.length; c0 += 8) {
  const group = CASES.slice(c0, c0 + 8);
  const gRows = group.map(({ bd: b, side, moves: mv, sym, optimism }) => {
    const f = encodeFeatures(b, side, { recentMoves: mv, komi: 7.5 });
    return {
      spatial: Array.from(sym ? permuteSpatial(f.spatial, SYM8[sym], new Float32Array(f.spatial.length)) : f.spatial),
      global: Array.from(f.global),
      optimism,
    };
  });
  const [gold, refs] = await Promise.all([
    golden(gRows),
    gpu.evalBatch(group.map(({ bd: b, side, moves: mv, sym, optimism }) => {
      const f = encodeFeatures(b, side, { recentMoves: mv, komi: 7.5 });
      return { spatial: f.spatial, global: f.global, sym, optimism };
    })),
  ]);
  group.forEach((cs, i) => {
    const g = gold[i], r = refs[i];
    const lgG = new Float64Array(N2 + 1), lgQ = new Float64Array(N2 + 1);
    for (let p = 0; p < N2; p++) { lgG[p] = g.policy[p]; lgQ[p] = r.policy[p]; }
    lgG[N2] = g.policyPass; lgQ[N2] = r.policyPass;
    const mxG = Math.max(...lgG), mxQ = Math.max(...lgQ);
    let kl = 0, sumG = 0, sumQ = 0;
    for (let p = 0; p <= N2; p++) { sumG += Math.exp(lgG[p] - mxG); sumQ += Math.exp(lgQ[p] - mxQ); }
    for (let p = 0; p <= N2; p++) {
      const pG = Math.exp(lgG[p] - mxG) / sumG, pQ = Math.exp(lgQ[p] - mxQ) / sumQ;
      kl += pG * Math.log(pG / Math.max(pQ, 1e-300));
    }
    sumKL += kl;
    const argmax = (lg) => lg.indexOf(Math.max(...lg));
    const tG = argmax(Array.from(lgG)), tQ = argmax(Array.from(lgQ));
    if (tG === tQ) top1Agree++;
    else {
      const second = Math.max(...lgG.filter((_, p) => p !== tG));
      const gap = lgG[tG] - second;
      if (gap < 0.5) nearTieFlips++;
      else { top1Disagree.push(`case${c0 + i}:${tG}→${tQ}(gap ${gap.toFixed(2)})`); realFlip++; }
    }
    const top5 = (lg) => new Set(Array.from(lg).map((v, p) => [v, p]).sort((a, b) => b[0] - a[0]).slice(0, 5).map(([, p]) => p));
    const s5 = top5(Array.from(lgG)), sQ5 = top5(Array.from(lgQ));
    let inter = 0;
    for (const p of s5) if (sQ5.has(p)) inter++;
    top5J += inter / 5;
    sumWL += Math.abs(g.winLoss - r.winLoss);
    sumLead += Math.abs(g.scoreLead - r.scoreLead);
    sumOwn += g.ownership.reduce((a, v, p) => a + Math.abs(v - r.ownership[p]), 0) / N2;
    sumPass += Math.abs(g.policyPass - r.policyPass);
  });
}
const nCases = CASES.length;
console.log(`\n== WGSL-Q(真实 GPU)vs ort golden,${nCases} 例 ==`);
console.log(`policy KL      = ${ (sumKL / nCases).toExponential(3)} nat   [L2(int8 权重)参考 9.65e-4;研究 8192 盘 int8w 7.9e-4]`);
console.log(`top1 一致率    = ${((top1Agree + nearTieFlips) / nCases * 100).toFixed(2)}%(近平局 ${nearTieFlips} + 实质 ${realFlip})[L2 100%]`);
if (top1Disagree.length) console.log('  实质分歧: ' + top1Disagree.join(' '));
console.log(`top5 重合      = ${(top5J / nCases * 100).toFixed(2)}%   [L2 98.67%]`);
console.log(`winLoss MAE    = ${(sumWL / nCases).toExponential(3)}        [L2 6.23e-3;研究 5.1e-3]`);
console.log(`scoreLead MAE  = ${(sumLead / nCases).toFixed(4)} 目     [L2 0.115;研究 0.061]`);
console.log(`ownership MAE  = ${(sumOwn / nCases).toExponential(3)}        [L2 3.82e-3]`);
console.log(`pass logits 偏差 = ${(sumPass / nCases).toExponential(3)}`);
gpu.dispose();
server.stdin.end();
server.kill();
