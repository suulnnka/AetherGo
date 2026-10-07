/* L3 诊断探针 v2(临时):WGSL-Q 与 f16 仿真参考的**逐 stage** ULP 级对照。
 *
 * v1 结论:WGSL-Q 对 f16emul / cpuref-Q 的 policy rms 偏差都是 ~2.6e-2,而
 * f16emul vs cpuref 只有 3.8e-3 —— WGSL 的有效舍入比「存储点 f16 舍入」粗。
 * 两个候选解释:
 *   A. 舍入边界混沌:WGSL 与参考的 f32 累加序差(~1e-6 相对)使每个存储点
 *      ~4% 的值跨过 f16 舍入界翻转 1 ULP,16 层级联放大 → 输出 ~1e-2;
 *   B. 某个 Q 模式内核有真 bug(如 i8 反量化 GEMM)→ 某 stage 出现 >>1ULP
 *      的结构性偏差。
 * 判别:逐 stage 快照,按 f16 ULP 归类(=0 / ≤1ULP / >1ULP)。A 的特征是
 * 偏差始终 ≤1ULP 但翻转率随深度增长;B 会在特定 stage 出现大量 >1ULP。
 *
 * 运行:node test/l3probe2.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

const dawn = await import('webgpu');
Object.assign(globalThis, dawn.globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: dawn.create([]) }, configurable: true });

const { N, BLACK, WHITE, newBoard, make } = await import(pathToFileURL(join(ROOT, 'src/engine.js')).href);
const { encodeFeatures } = await import(pathToFileURL(join(ROOT, 'src/nn/features.js')).href);
const { createAewnnSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/session.js')).href);
const {
  HW, C_TRUNK, NUM_HEADS, HEAD_DIM, FFN, FFN_FUSED, QKV_FUSED,
  NUM_BLOCKS, SPATIAL_C, GLOBAL_C, HEAD_C, V2_C, ATTN_SCALE, RMS_EPS,
  parseAewn, makeStemTables,
} = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/plan.js')).href);

const f16 = (x) => Math.f16round(x);

/* f16 ULP(绝对步长)在 v 处:f16 相邻可表示数间隔 */
function f16ulp(v) {
  const a = Math.abs(v);
  if (!Number.isFinite(a) || a === 0) return 6e-8;           // 亚法线最小步长
  const e = Math.floor(Math.log2(a));
  const exp = Math.max(-14, Math.min(15, e));                // f16 规格化指数域
  return 2 ** (exp - 10);
}
function decodeF16(u32arr, nWords) {
  const out = new Float32Array(nWords * 2);
  for (let i = 0; i < nWords; i++) {
    const w = u32arr[i];
    out[i * 2] = f16bits((w) & 0xFFFF);
    out[i * 2 + 1] = f16bits((w >>> 16) & 0xFFFF);
  }
  return out;
}
function f16bits(h) {
  const sg = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, m = h & 0x03ff;
  if (e === 0) return (sg ? -1 : 1) * m * 2 ** -24;
  if (e === 31) return m ? NaN : (sg ? -Infinity : Infinity);
  return (sg ? -1 : 1) * (1 + m / 1024) * 2 ** (e - 15);
}

/* ---------- f16 仿真参考(v1 同款 + 快照) ---------- */
function createF16EmulSession(blobBuffer, snapSet) {
  const parsed = parseAewn(blobBuffer);
  const { meta, w: weights } = parsed;
  for (const [name, sName] of Object.entries(meta.quant ?? {})) {
    const packed = weights.get(name);
    const scale = weights.get(sName);
    const dims = parsed.dims.get(name);
    const total = packed.length * 4;
    const out = new Float32Array(total);
    const firstAxis = meta.quantAxis?.[name] === 'first';
    const O = firstAxis ? dims[0] : dims[dims.length - 1];
    const inner = total / O;
    for (let i = 0; i < total; i++) {
      const b = (packed[i >> 2] >>> ((i & 3) * 8)) & 0xFF;
      const sv = b >= 128 ? b - 256 : b;
      out[i] = sv * scale[firstAxis ? (i / inner) | 0 : i % O];
    }
    weights.set(name, out);
  }
  const w = { get: (k) => { const v = weights.get(k); if (!v) throw new Error(`blob 缺张量 ${k}`); return v; }, stemTables: makeStemTables() };
  const cosT = w.get('rope.cos'), sinT = w.get('rope.sin');
  const snap = (name, arr) => { if (snapSet.has(name)) snapSet.get(name)(Float32Array.from(arr)); };

  function stemConv(n, sym, spatialIn, globalIn, trunk) {
    const { table } = w.stemTables;
    const W = w.get('stem.conv_w'), WG = w.get('stem.global_w');
    const gBase = n * GLOBAL_C;
    for (let q = 0; q < HW; q++) {
      const tBase = ((sym * HW) + q) * 9;
      for (let oc = 0; oc < C_TRUNK; oc++) {
        let acc = 0;
        const wBase = oc * SPATIAL_C * 9;
        for (let ic = 0; ic < SPATIAL_C; ic++) {
          const inBase = (n * SPATIAL_C + ic) * HW;
          const kW = wBase + ic * 9;
          for (let d = 0; d < 9; d++) {
            const pos = table[tBase + d];
            const v = pos < SPATIAL_C * HW ? spatialIn[inBase + pos] : 0;
            acc += v * W[kW + d];
          }
        }
        let g = 0;
        for (let j = 0; j < GLOBAL_C; j++) g += globalIn[gBase + j] * WG[oc * GLOBAL_C + j];
        trunk[(n * HW + q) * C_TRUNK + oc] = f16(acc + g);
      }
    }
  }
  function rmsNorm(x, gamma, n, out) {
    for (let q = 0; q < HW; q++) {
      const base = (n * HW + q) * C_TRUNK;
      let ss = 0;
      for (let c = 0; c < C_TRUNK; c++) { const v = x[base + c]; ss += v * v; }
      const r = 1 / Math.sqrt(ss / C_TRUNK + RMS_EPS);
      for (let c = 0; c < C_TRUNK; c++) out[base + c] = f16(x[base + c] * r * gamma[c]);
    }
  }
  function qkvGemm(x, W, n, out) {
    for (let q = 0; q < HW; q++) {
      const xBase = (n * HW + q) * C_TRUNK;
      const oBase = (n * HW + q) * QKV_FUSED;
      for (let o = 0; o < QKV_FUSED; o++) {
        let acc = 0;
        for (let k = 0; k < C_TRUNK; k++) acc += x[xBase + k] * W[k * QKV_FUSED + o];
        out[oBase + o] = f16(acc);
      }
    }
  }
  function ropeScatter(qkv, n, qh, kh, vh) {
    for (let q = 0; q < HW; q++) {
      const src = (n * HW + q) * QKV_FUSED;
      const cBase = q * HEAD_DIM;
      for (let o = 0; o < C_TRUNK; o++) {
        const h = o >> 5, d = o & 31;
        const x0 = qkv[src + o];
        const x1 = qkv[src + (o ^ 1)];
        qh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = f16(x0 * cosT[cBase + d] + x1 * sinT[cBase + d]);
        const k0 = qkv[src + C_TRUNK + o], k1 = qkv[src + C_TRUNK + (o ^ 1)];
        kh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = f16(k0 * cosT[cBase + d] + k1 * sinT[cBase + d]);
        vh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = qkv[src + 2 * C_TRUNK + o];
      }
    }
  }
  function attention(qh, kh, vh, n, scores, attn) {
    const seq2 = HW * HW;
    for (let h = 0; h < NUM_HEADS; h++) {
      const qBase = ((n * NUM_HEADS + h) * HW) * HEAD_DIM;
      const sBase = (n * NUM_HEADS + h) * seq2;
      for (let qi = 0; qi < HW; qi++) {
        const rowBase = sBase + qi * HW;
        const qq = qBase + qi * HEAD_DIM;
        for (let ki = 0; ki < HW; ki++) {
          const kk = qBase + ki * HEAD_DIM;
          let acc = 0;
          for (let d = 0; d < HEAD_DIM; d++) acc += qh[qq + d] * kh[kk + d];
          scores[rowBase + ki] = acc * ATTN_SCALE;
        }
        let mx = -3e38;
        for (let ki = 0; ki < HW; ki++) { const s = scores[rowBase + ki]; if (s > mx) mx = s; }
        let sum = 0;
        for (let ki = 0; ki < HW; ki++) { const e = Math.exp(scores[rowBase + ki] - mx); scores[rowBase + ki] = e; sum += e; }
        const inv = 1 / sum;
        for (let ki = 0; ki < HW; ki++) scores[rowBase + ki] *= inv;
        for (let e = 0; e < HEAD_DIM; e++) {
          let acc = 0;
          for (let ki = 0; ki < HW; ki++) acc += scores[rowBase + ki] * vh[qBase + ki * HEAD_DIM + e];
          attn[(n * HW + qi) * C_TRUNK + h * HEAD_DIM + e] = f16(acc);
        }
      }
    }
  }
  function gemm(x, W, n, K, O, out, epi, residual, bias) {
    for (let q = 0; q < HW; q++) {
      const xBase = (n * HW + q) * K;
      const oBase = (n * HW + q) * O;
      for (let o = 0; o < O; o++) {
        let acc = 0;
        for (let k = 0; k < K; k++) acc += x[xBase + k] * W[k * O + o];
        if (epi === 'res') acc += residual[oBase + o];
        if (epi === 'biasrelu') acc = Math.max(acc + bias[o], 0);
        out[oBase + o] = f16(acc);
      }
    }
  }
  function swiglu(gate, n, hidden) {
    for (let q = 0; q < HW; q++) {
      const gBase = (n * HW + q) * FFN_FUSED;
      const hBase = (n * HW + q) * FFN;
      for (let i = 0; i < FFN; i++) {
        const a = gate[gBase + i];
        const s = 0.5 * (1.0 + Math.tanh(0.5 * a));
        hidden[hBase + i] = f16(a * s * gate[gBase + FFN + i]);
      }
    }
  }
  function trunkFinal(x, scale, bias, n) {
    for (let q = 0; q < HW; q++) {
      const base = (n * HW + q) * C_TRUNK;
      for (let c = 0; c < C_TRUNK; c++) x[base + c] = f16(Math.max(x[base + c] * scale[c] + bias[c], 0));
    }
  }

  async function evalBatch(rows) {
    const n = rows.length;
    let trunk = new Float32Array(n * HW * C_TRUNK);
    const normed = new Float32Array(n * HW * C_TRUNK);
    const attn = new Float32Array(n * HW * C_TRUNK);
    let proj = new Float32Array(n * HW * C_TRUNK);
    const qkv = new Float32Array(n * HW * QKV_FUSED);
    const qh = new Float32Array(n * NUM_HEADS * HW * HEAD_DIM);
    const kh = new Float32Array(n * NUM_HEADS * HW * HEAD_DIM);
    const vh = new Float32Array(n * NUM_HEADS * HW * HEAD_DIM);
    const scores = new Float32Array(n * NUM_HEADS * HW * HW);
    const gate = new Float32Array(n * HW * FFN_FUSED);
    const hidden = new Float32Array(n * HW * FFN);

    const spatialBat = new Float32Array(n * SPATIAL_C * HW);
    const globalBat = new Float32Array(n * GLOBAL_C);
    for (let i = 0; i < n; i++) {
      spatialBat.set(rows[i].spatial, i * SPATIAL_C * HW);
      globalBat.set(rows[i].global, i * GLOBAL_C);
    }
    for (let i = 0; i < n; i++) {
      const r = rows[i];
      const sym = r.sym ?? 0;
      stemConv(i, sym, spatialBat, globalBat, trunk);
      if (i === 0) snap('trunk0', trunk);
      for (let b = 0; b < NUM_BLOCKS; b++) {
        rmsNorm(trunk, w.get(`attn${b}.norm`), i, normed);
        if (i === 0 && b === 0) snap('normed0', normed);
        qkvGemm(normed, w.get(`attn${b}.qkv`), i, qkv);
        ropeScatter(qkv, i, qh, kh, vh);
        if (i === 0 && b === 0) snap('qh0', qh);
        attention(qh, kh, vh, i, scores, attn);
        if (i === 0 && b === 0) snap('attn0', attn);
        gemm(attn, w.get(`attn${b}.out`), i, C_TRUNK, C_TRUNK, proj, 'res', trunk);
        [trunk, proj] = [proj, trunk];
        if (i === 0 && b === 0) snap('trunkA0', trunk);
        if (i === 0 && b === 3) snap('trunkA3', trunk);
        rmsNorm(trunk, w.get(`ffn${b}.norm`), i, normed);
        gemm(normed, w.get(`ffn${b}.gate`), i, C_TRUNK, FFN_FUSED, gate, 'plain');
        swiglu(gate, i, hidden);
        gemm(hidden, w.get(`ffn${b}.ffn2`), i, FFN, C_TRUNK, proj, 'res', trunk);
        [trunk, proj] = [proj, trunk];
        if (i === 0 && b === 0) snap('trunkB0', trunk);
        if (i === 0 && b === 3) snap('trunkB3', trunk);
        if (i === 0 && b === 7) snap('trunkB7', trunk);
      }
      trunkFinal(trunk, w.get('trunkfinal.scale'), w.get('trunkfinal.bias'), i);
      if (i === 0) snap('trunkfinal', trunk);
    }
    return n;
  }
  return { evalBatch };
}

/* ---------- 局面(case8:中盘,sym1,λ1) ---------- */
const SEQ_MID = [
  [3, 3], [15, 15], [3, 15], [15, 3], [9, 9], [3, 9], [15, 9], [9, 3],
  [9, 15], [5, 5], [13, 13], [5, 13], [13, 5], [7, 7], [11, 11], [7, 11],
  [11, 7], [2, 8], [16, 8], [8, 2], [8, 16],
];
const bd = newBoard();
for (let i = 0; i < SEQ_MID.length; i++) make(bd, SEQ_MID[i][0] * N + SEQ_MID[i][1], i % 2 === 0 ? BLACK : WHITE);
const moves = SEQ_MID.map(([r, c]) => r * N + c);
const f = encodeFeatures(bd, WHITE, { recentMoves: moves, komi: 7.5 });
const row = { spatial: f.spatial, global: f.global, sym: 1, optimism: 1.0 };

/* ---------- 跑三方 ---------- */
const blobBuf = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.i8.aewn'));
const blob = blobBuf.buffer.slice(blobBuf.byteOffset, blobBuf.byteOffset + blobBuf.byteLength);

const snapSet = new Map();
for (const nm of ['trunk0', 'normed0', 'qh0', 'attn0', 'trunkA0', 'trunkA3', 'trunkB0', 'trunkB3', 'trunkB7', 'trunkfinal']) {
  snapSet.set(nm, (arr) => { snapSet['v_' + nm] = arr; });
}
const emul = createF16EmulSession(blob.slice(0), snapSet);
await emul.evalBatch([row]);

const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });
await gpu.evalBatch([row]);

/* GPU 快照(经 __debugCopy,u32 读 + f16 解码) */
const TR = HW * C_TRUNK / 2, QH = NUM_HEADS * HW * HEAD_DIM / 2;
const stages = [
  ['trunk0', 'trunk', 1, TR],
  ['normed0', 'normed', 2, TR],
  ['qh0', 'qh', 4, QH],
  ['attn0', 'attn', 5, TR],
  ['trunkA0', 'proj', 6, TR],
  ['trunkB0', 'trunk', 10, TR],
  ['trunkA3', 'proj', 33, TR],
  ['trunkB3', 'trunk', 37, TR],
  ['trunkB7', 'trunk', 73, TR],
  ['trunkfinal', 'normed', 74, TR],
];
const gpuSnap = {};
for (const [name, bname, upto, words] of stages) {
  const d = await gpu.__debugCopy([{ name, buf: bname, floats: 0, u32: true, count: words, bytes: words * 4 }], [row], upto);
  gpuSnap[name] = decodeF16(d[name], words);
}

/* ---------- 逐 stage ULP 分析 ---------- */
console.log(`stage 对照(row=case8 中盘 sym1;GPU-Q vs f16emul,按 f16 ULP 归类)`);
console.log('stage       N      =0%    ≤1ULP%  >1ULP%   max|Δ|/ULP   rms|Δ|');
for (const [name] of stages) {
  const g = gpuSnap[name], e = snapSet['v_' + name];
  if (!g || !e) { console.log(`${name} 缺快照`); continue; }
  let eq0 = 0, ulp1 = 0, over = 0, mxr = 0, sum2 = 0;
  const n = Math.min(g.length, e.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(g[i] - e[i]);
    const u = f16ulp(g[i]);
    const r = d / u;
    sum2 += d * d;
    if (d === 0) eq0++;
    else if (r <= 1.5) ulp1++;
    else over++;
    if (r > mxr && Number.isFinite(r)) mxr = r;
  }
  console.log(`${name.padEnd(10)} ${String(n).padStart(6)}  ${(100 * eq0 / n).toFixed(1).padStart(5)}  ${(100 * (eq0 + ulp1) / n).toFixed(1).padStart(6)}  ${(100 * over / n).toFixed(2).padStart(6)}   ${mxr.toFixed(1).padStart(8)}   ${Math.sqrt(sum2 / n).toExponential(2)}`);
}

/* 确定性:同输入重跑 evalBatch,输出应逐位一致 */
const o1 = await gpu.evalBatch([row]);
const o2 = await gpu.evalBatch([row]);
let det = true;
for (let p = 0; p < HW; p++) if (o1[0].policy[p] !== o2[0].policy[p]) { det = false; break; }
console.log(`\nGPU 重复推理逐位一致: ${det ? '是' : '否(非确定!)'}`);
gpu.dispose();
