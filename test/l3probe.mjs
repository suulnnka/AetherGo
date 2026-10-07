/* L3 诊断探针(临时):WGSL-Q 与「f16 舍入仿真版 cpuref」逐点对拍。
 *
 * 背景:aewnn-quant-test 第 3 层(WGSL-Q vs cpuref-Q)在本机(RTX 5060,Dawn
 * D3D12)报 policy max|Δ|=7.15e-2 > 0.05 闸门。cpuref-Q 是「int8 权重反量化 +
 * 纯 f32 数学」,不建模 f16 激活存储;本探针在 cpuref 同构循环里按 WGSL Q 模式
 * 的**精确存储点**注入 Math.f16round:
 *   stem 出 / rms 出 / qkvGemm 出(gemmQkv 直写 f16(acc) 后 rope 再旋转)/
 *   rope 出 / flash 出 / gemmRes 出(残差在 f16 载体上)/ gemmPlain(gate)出 /
 *   swiglu 出 / trunkFinal 出 / 头部 gemmPlain32·biasrelu32 出 / ling 出。
 * gp / pol / pass / val / misc / own 保持 f32(与 session 缓冲规划一致)。
 *
 * 判读:
 *   - WGSL-Q vs f16emul ≈ f32 模式基线(~2.5e-5)→ 内核逐位忠实,7e-2 是
 *     「f16 存储效应」本身(L3 参照未建模),闸门/参照需修;
 *   - 仍 >> 基线 → 内核真 bug,继续用 gpu-probe 定位首个发散 stage。
 *
 * 运行:node test/l3probe.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

const dawn = await import('webgpu');
Object.assign(globalThis, dawn.globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: dawn.create([]) }, configurable: true });

const { N, N2, BLACK, WHITE, PASS, newBoard, make } = await import(pathToFileURL(join(ROOT, 'src/engine.js')).href);
const { encodeFeatures } = await import(pathToFileURL(join(ROOT, 'src/nn/features.js')).href);
const { createCpuRefSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/cpuref.js')).href);
const { createAewnnSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/session.js')).href);
const {
  HW, C_TRUNK, NUM_HEADS, HEAD_DIM, FFN, FFN_FUSED, QKV_FUSED,
  NUM_BLOCKS, SPATIAL_C, GLOBAL_C, HEAD_C, V2_C, ATTN_SCALE, RMS_EPS,
  parseAewn, makeStemTables,
} = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/plan.js')).href);

/* ---------- adapter 信息(GPU 落点证据) ---------- */
const adapter = await navigator.gpu.requestAdapter();
console.log('== adapter ==');
console.log('  features:', [...adapter.features].join(','));
console.log('  info:', JSON.stringify(adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}), (_k, v) => typeof v === 'bigint' ? String(v) : v));

const f16 = (x) => Math.f16round(x);
const silu = (a) => a * 0.5 * (1.0 + Math.tanh(0.5 * a));

/* ---------- f16 仿真参考(与 cpuref.js 逐算子同构,仅加存储点舍入) ---------- */
function createF16EmulSession(blobBuffer) {
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

  function stemConv(w_, n, sym, spatialIn, globalIn, trunk) {
    const { table } = w_.stemTables;
    const W = w_.get('stem.conv_w'), WG = w_.get('stem.global_w');
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
  function ropeScatter(qkv, cosT_, sinT_, n, qh, kh, vh) {
    for (let q = 0; q < HW; q++) {
      const src = (n * HW + q) * QKV_FUSED;
      const cBase = q * HEAD_DIM;
      for (let o = 0; o < C_TRUNK; o++) {
        const h = o >> 5, d = o & 31;
        const x0 = qkv[src + o];
        const x1 = qkv[src + (o ^ 1)];
        qh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = f16(x0 * cosT_[cBase + d] + x1 * sinT_[cBase + d]);
        const k0 = qkv[src + C_TRUNK + o], k1 = qkv[src + C_TRUNK + (o ^ 1)];
        kh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = f16(k0 * cosT_[cBase + d] + k1 * sinT_[cBase + d]);
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
  function poolPolicy(actg, n, gp) {
    const base = n * HW * HEAD_C;
    for (let c = 0; c < HEAD_C; c++) {
      let sum = 0, mx = -3e38;
      for (let q = 0; q < HW; q++) { const v = actg[base + q * HEAD_C + c]; sum += v; if (v > mx) mx = v; }
      const mean = sum / HW;
      gp[n * 3 * HEAD_C + c] = mean;
      gp[n * 3 * HEAD_C + HEAD_C + c] = mean * 0.5;
      gp[n * 3 * HEAD_C + 2 * HEAD_C + c] = mx;
    }
  }
  function lingFused(p1, gp, Wlg, s2, b2, n, act2) {
    const gpBase = n * 3 * HEAD_C;
    for (let c = 0; c < HEAD_C; c++) {
      let g2 = 0;
      for (let j = 0; j < 3 * HEAD_C; j++) g2 += gp[gpBase + j] * Wlg[j * HEAD_C + c];
      for (let q = 0; q < HW; q++) {
        const o = (n * HW + q) * HEAD_C + c;
        act2[o] = f16(Math.max((g2 + p1[o]) * s2[c] + b2[c], 0));
      }
    }
  }
  function passFused(gp, Wpw, bp, Wp2, n, passOut) {
    const gpBase = n * 3 * HEAD_C;
    const h = new Float32Array(HEAD_C);
    for (let c = 0; c < HEAD_C; c++) {
      let acc = 0;
      for (let j = 0; j < 3 * HEAD_C; j++) acc += gp[gpBase + j] * Wpw[j * HEAD_C + c];
      h[c] = Math.max(acc + bp[c], 0);
    }
    for (let o = 0; o < 2; o++) {
      let acc = 0;
      for (let k = 0; k < HEAD_C; k++) acc += h[k] * Wp2[k * 2 + o];
      passOut[n * 2 + o] = acc;
    }
  }
  function poolValue(act1, n, gp) {
    const base = n * HW * HEAD_C;
    for (let c = 0; c < HEAD_C; c++) {
      let sum = 0;
      for (let q = 0; q < HW; q++) sum += act1[base + q * HEAD_C + c];
      const mean = sum / HW;
      gp[n * 3 * HEAD_C + c] = mean;
      gp[n * 3 * HEAD_C + HEAD_C + c] = mean * 0.5;
      gp[n * 3 * HEAD_C + 2 * HEAD_C + c] = mean * 0.15;
    }
  }
  function valueMlp(gp, Wv2, v2b, Wvh, vhb, Wm, mb, n, valOut, miscOut) {
    const gpBase = n * 3 * HEAD_C;
    const a = new Float32Array(V2_C);
    for (let j = 0; j < V2_C; j++) {
      let acc = 0;
      for (let k = 0; k < 3 * HEAD_C; k++) acc += gp[gpBase + k] * Wv2[k * V2_C + j];
      a[j] = Math.max(acc + v2b[j], 0);
    }
    for (let o = 0; o < 3; o++) {
      let acc = 0;
      for (let k = 0; k < V2_C; k++) acc += a[k] * Wvh[k * 3 + o];
      valOut[n * 3 + o] = acc + vhb[o];
    }
    for (let o = 0; o < 6; o++) {
      let acc = 0;
      for (let k = 0; k < V2_C; k++) acc += a[k] * Wm[k * 6 + o];
      miscOut[n * 6 + o] = acc + mb[o];
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
    const p1 = new Float32Array(n * HW * HEAD_C);
    const actg = new Float32Array(n * HW * HEAD_C);
    const v1 = new Float32Array(n * HW * HEAD_C);
    const gp = new Float32Array(n * 3 * HEAD_C);
    const act2 = new Float32Array(n * HW * HEAD_C);
    const pol = new Float32Array(n * HW * 2);
    const pass = new Float32Array(n * 2);
    const val = new Float32Array(n * 3);
    const misc = new Float32Array(n * 6);
    const own = new Float32Array(n * HW);

    const spatialBat = new Float32Array(n * SPATIAL_C * HW);
    const globalBat = new Float32Array(n * GLOBAL_C);
    for (let i = 0; i < n; i++) {
      spatialBat.set(rows[i].spatial, i * SPATIAL_C * HW);
      globalBat.set(rows[i].global, i * GLOBAL_C);
    }
    for (let i = 0; i < n; i++) {
      const r = rows[i];
      const sym = r.sym ?? 0;
      stemConv(w, i, sym, spatialBat, globalBat, trunk);
      for (let b = 0; b < NUM_BLOCKS; b++) {
        rmsNorm(trunk, w.get(`attn${b}.norm`), i, normed);
        qkvGemm(normed, w.get(`attn${b}.qkv`), i, qkv);
        ropeScatter(qkv, cosT, sinT, i, qh, kh, vh);
        attention(qh, kh, vh, i, scores, attn);
        gemm(attn, w.get(`attn${b}.out`), i, C_TRUNK, C_TRUNK, proj, 'res', trunk);
        [trunk, proj] = [proj, trunk];
        rmsNorm(trunk, w.get(`ffn${b}.norm`), i, normed);
        gemm(normed, w.get(`ffn${b}.gate`), i, C_TRUNK, FFN_FUSED, gate, 'plain');
        swiglu(gate, i, hidden);
        gemm(hidden, w.get(`ffn${b}.ffn2`), i, FFN, C_TRUNK, proj, 'res', trunk);
        [trunk, proj] = [proj, trunk];
      }
      trunkFinal(trunk, w.get('trunkfinal.scale'), w.get('trunkfinal.bias'), i);
      gemm(trunk, w.get('policy.conv1p'), i, C_TRUNK, HEAD_C, p1, 'plain');
      gemm(trunk, w.get('policy.conv1g'), i, C_TRUNK, HEAD_C, actg, 'biasrelu', null, w.get('policy.conv1g_b'));
      poolPolicy(actg, i, gp);
      lingFused(p1, gp, w.get('policy.gp_ling'), w.get('policy.bias2_scale'), w.get('policy.bias2_bias'), i, act2);
      for (let q = 0; q < HW; q++) {
        const xBase = (i * HW + q) * HEAD_C;
        const oBase = (i * HW + q) * 2;
        for (let o = 0; o < 2; o++) {
          let acc = 0;
          for (let k = 0; k < HEAD_C; k++) acc += act2[xBase + k] * w.get('policy.conv2p')[k * 2 + o];
          pol[oBase + o] = acc;
        }
      }
      passFused(gp, w.get('policy.pass_w'), w.get('policy.pass_b'), w.get('policy.pass2'), i, pass);
      gemm(trunk, w.get('value.conv1'), i, C_TRUNK, HEAD_C, v1, 'biasrelu', null, w.get('value.conv1_b'));
      poolValue(v1, i, gp);
      valueMlp(gp, w.get('value.v2'), w.get('value.v2_b'), w.get('value.vh'), w.get('value.vh_b'),
        w.get('value.misc'), w.get('value.misc_b'), i, val, misc);
      for (let q = 0; q < HW; q++) {
        const xBase = (i * HW + q) * HEAD_C;
        let acc = 0;
        for (let k = 0; k < HEAD_C; k++) acc += v1[xBase + k] * w.get('value.own')[k];
        own[i * HW + q] = acc;
      }
    }

    const res = new Array(n);
    for (let i = 0; i < n; i++) {
      const optimism = rows[i].optimism ?? 1.0;
      const policy = new Float32Array(HW);
      for (let p = 0; p < HW; p++) {
        const p0 = pol[(i * HW + p) * 2], pOpt = pol[(i * HW + p) * 2 + 1];
        policy[p] = (optimism !== 1.0) ? p0 + (pOpt - p0) * optimism : p0;
      }
      const pb = pass[i * 2], pbOpt = pass[i * 2 + 1];
      const policyPass = (optimism !== 1.0) ? pb + (pbOpt - pb) * optimism : pb;
      const l0 = val[i * 3], l1 = val[i * 3 + 1];
      const m = Math.max(l0, l1);
      const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
      const softPlus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));
      const b = i * 6;
      res[i] = {
        policy, policyPass,
        winLoss: (e0 - e1) / (e0 + e1),
        ownership: Float32Array.from(own.subarray(i * HW, (i + 1) * HW)),
        scoreMean: misc[b] * 20,
        scoreStdev: softPlus(misc[b + 1]) * 20,
        scoreLead: misc[b + 2] * 20,
      };
    }
    return res;
  }
  return { ep: 'f16emul', evalBatch, meta };
}

/* ---------- 局面集(同 quant-test 口径,取前 12 例) ---------- */
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
const rows = CASES.slice(0, 12).map(({ bd, side, moves, sym, optimism }) => {
  const f = encodeFeatures(bd, side, { recentMoves: moves, komi: 7.5 });
  return { spatial: f.spatial, global: f.global, sym, optimism };
});

/* ---------- 三方对拍 ---------- */
const blobBuf = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.i8.aewn'));
const blob = blobBuf.buffer.slice(blobBuf.byteOffset, blobBuf.byteOffset + blobBuf.byteLength);

const cpuQ = createCpuRefSession(blob.slice(0));
const emul = createF16EmulSession(blob.slice(0));
const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });

const [cq, eq, gq] = await Promise.all([
  cpuQ.evalBatch(rows.map((r) => ({ ...r }))),
  emul.evalBatch(rows.map((r) => ({ ...r }))),
  gpu.evalBatch(rows.map((r) => ({ ...r }))),
]);

const stat = (a, b, sel) => {
  let mx = 0, imx = -1, sum2 = 0, cnt = 0;
  for (let i = 0; i < rows.length; i++) {
    const va = sel(a[i]), vb = sel(b[i]);
    for (let p = 0; p < va.length; p++) {
      const d = Math.abs(va[p] - vb[p]);
      if (d > mx) { mx = d; imx = p; }
      sum2 += d * d; cnt++;
    }
  }
  return { mx, rms: Math.sqrt(sum2 / cnt), imx };
};
const pol = (r) => r.policy;
const own = (r) => r.ownership;

const gvc = stat(gq, cq, pol);
const gve = stat(gq, eq, pol);
const evc = stat(eq, cq, pol);
console.log('\n== policy(max |Δ| / RMS,12 例 × 361 点) ==');
console.log(`WGSL-Q vs f16emul : max ${gve.mx.toExponential(3)}  rms ${gve.rms.toExponential(3)}  @${gve.imx}`);
console.log(`WGSL-Q vs cpuref-Q: max ${gvc.mx.toExponential(3)}  rms ${gvc.rms.toExponential(3)}  @${gvc.imx}`);
console.log(`f16emul vs cpuref : max ${evc.mx.toExponential(3)}  rms ${evc.rms.toExponential(3)}  @${evc.imx}`);
const gvcO = stat(gq, cq, own);
const gveO = stat(gq, eq, own);
console.log('\n== ownership ==');
console.log(`WGSL-Q vs f16emul : max ${gveO.mx.toExponential(3)}  rms ${gveO.rms.toExponential(3)}`);
console.log(`WGSL-Q vs cpuref-Q: max ${gvcO.mx.toExponential(3)}  rms ${gvcO.rms.toExponential(3)}`);

/* 逐例 policy max|Δ|(WGSL vs 两个参照)—— 看发散是否集中在个别局面 */
console.log('\n== 逐例 policy max|Δ| ==');
rows.forEach((_, i) => {
  let dg = 0, dc = 0;
  for (let p = 0; p < HW; p++) {
    dg = Math.max(dg, Math.abs(gq[i].policy[p] - eq[i].policy[p]));
    dc = Math.max(dc, Math.abs(gq[i].policy[p] - cq[i].policy[p]));
  }
  console.log(`  case${i}(sym${rows[i].sym},λ${rows[i].optimism}):  vs-emul ${dg.toExponential(2)}  vs-cpuref ${dc.toExponential(2)}`);
});

/* 逐层对照:把 f16emul 的关键中间量与 GPU 的分段快照对齐成本高,
 * 这里仅输出 winLoss / scoreLead 三方对照作 sanity */
console.log('\n== winLoss / scoreLead(行 0..5)==');
for (let i = 0; i < 6; i++) {
  console.log(`  case${i}: winLoss gq ${gq[i].winLoss.toFixed(5)} / emul ${eq[i].winLoss.toFixed(5)} / cpu ${cq[i].winLoss.toFixed(5)}   lead ${gq[i].scoreLead.toFixed(3)} / ${eq[i].scoreLead.toFixed(3)} / ${cq[i].scoreLead.toFixed(3)}`);
}
gpu.dispose();
