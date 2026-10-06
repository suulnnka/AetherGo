/* ============================================================
 * aethernn CPU 参考解释器(测试专用,不进运行时依赖图)
 *
 * 与 src/nn/webgpu/kernels.js 的 WGSL 内核**逐算子同构**的纯 JS 实现:
 * 同一执行计划(plan.js)、同一权重 blob(.aewn)、同一融合结构、同一累加序。
 * 用途:
 *   1. 验证 packer 产出的权重映射与计划数学(对拍 ort CPU golden);
 *   2. WGSL 内核的对拍参照(同输入跑两侧,容差内必须一致);
 *   3. 无 GPU 环境下的计划回归测试。
 *
 * 不追求性能(每评估全 JS 标量循环),只在测试里跑。
 * 输入支持 sym(GPU gather 表同款语义,stem 内做置换)。
 * ============================================================ */
import {
  POS_LEN, HW, C_TRUNK, NUM_HEADS, HEAD_DIM, FFN, FFN_FUSED, QKV_FUSED,
  NUM_BLOCKS, SPATIAL_C, GLOBAL_C, HEAD_C, V2_C, ATTN_SCALE, RMS_EPS,
  parseAewn, assertPlanMeta, makeStemTables,
} from './plan.js';
import { N2 } from '../../engine.js';

const silu = (a) => a * 0.5 * (1.0 + Math.tanh(0.5 * a));
const relu = (a) => (a > 0 ? a : 0);

/* 逐内核实现 —— 每个函数与 WGSL 同名内核同构(见 kernels.js 对应注释) */

function stemConv(w, n, sym, spatialIn, globalIn, trunk) {
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
          const v = pos < SPATIAL_C * HW ? spatialIn[inBase + pos] : 0;   // 哨兵 = 盘外
          acc += v * W[kW + d];
        }
      }
      let g = 0;
      for (let j = 0; j < GLOBAL_C; j++) g += globalIn[gBase + j] * WG[oc * GLOBAL_C + j];
      trunk[(n * HW + q) * C_TRUNK + oc] = acc + g;
    }
  }
}

function rmsNorm(x, gamma, n, out) {
  for (let q = 0; q < HW; q++) {
    const base = (n * HW + q) * C_TRUNK;
    let ss = 0;
    for (let c = 0; c < C_TRUNK; c++) { const v = x[base + c]; ss += v * v; }
    const r = 1 / Math.sqrt(ss / C_TRUNK + RMS_EPS);
    for (let c = 0; c < C_TRUNK; c++) out[base + c] = x[base + c] * r * gamma[c];
  }
}

/* 融合 qkv GEMM(96→288),NHWC 输入/输出(head-major 散排由 ropeScatter 做) */
function qkvGemm(x, W, n, out) {
  for (let q = 0; q < HW; q++) {
    const xBase = (n * HW + q) * C_TRUNK;
    const oBase = (n * HW + q) * QKV_FUSED;
    for (let o = 0; o < QKV_FUSED; o++) {
      let acc = 0;
      for (let k = 0; k < C_TRUNK; k++) acc += x[xBase + k] * W[k * QKV_FUSED + o];
      out[oBase + o] = acc;
    }
  }
}

/* rope(q,k 旋转)+ head-major 散排(v 直拷)—— WGSL ropeScatter 同构 */
function ropeScatter(qkv, cosT, sinT, n, qh, kh, vh) {
  for (let q = 0; q < HW; q++) {
    const src = (n * HW + q) * QKV_FUSED;
    const cBase = q * HEAD_DIM, sBase = q * HEAD_DIM;
    for (let o = 0; o < C_TRUNK; o++) {
      const h = o >> 5, d = o & 31;
      const x0 = qkv[src + o];
      const x1 = qkv[src + (o ^ 1)];
      qh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = x0 * cosT[cBase + d] + x1 * sinT[sBase + d];
      const k0 = qkv[src + C_TRUNK + o], k1 = qkv[src + C_TRUNK + (o ^ 1)];
      kh[((n * NUM_HEADS + h) * HW + q) * HEAD_DIM + d] = k0 * cosT[cBase + d] + k1 * sinT[sBase + d];
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
        const kk = qBase + ki * HEAD_DIM;   // k 与 q 同 head-major 基址(各 buffer)
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
        attn[(n * HW + qi) * C_TRUNK + h * HEAD_DIM + e] = acc;
      }
    }
  }
}

/* GEMM + epilogue:NHWC (n·HW, K) × (K, O)。epi: 'plain' | 'res'(+residual) | 'biasrelu' */
function gemm(x, W, n, K, O, out, epi, residual, bias) {
  for (let q = 0; q < HW; q++) {
    const xBase = (n * HW + q) * K;
    const oBase = (n * HW + q) * O;
    for (let o = 0; o < O; o++) {
      let acc = 0;
      for (let k = 0; k < K; k++) acc += x[xBase + k] * W[k * O + o];
      if (epi === 'res') acc += residual[oBase + o];
      if (epi === 'biasrelu') acc = relu(acc + bias[o]);
      out[oBase + o] = acc;
    }
  }
}

function swiglu(gate, n, hidden) {
  for (let q = 0; q < HW; q++) {
    const gBase = (n * HW + q) * FFN_FUSED;
    const hBase = (n * HW + q) * FFN;
    for (let i = 0; i < FFN; i++) hidden[hBase + i] = silu(gate[gBase + i]) * gate[gBase + FFN + i];
  }
}

function trunkFinal(x, scale, bias, n) {
  for (let q = 0; q < HW; q++) {
    const base = (n * HW + q) * C_TRUNK;
    for (let c = 0; c < C_TRUNK; c++) x[base + c] = relu(x[base + c] * scale[c] + bias[c]);
  }
}

/* policy 池化:gp[n, 0..31]=mean, [32..63]=mean·0.5, [64..95]=max(mask 常数已折叠进图) */
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

/* linear_g + gpbias + bias2 + relu 融合(单 kernel:WGSL 里每线程一通道) */
function lingFused(p1, gp, Wlg, s2, b2, n, act2) {
  const gpBase = n * 3 * HEAD_C;
  for (let c = 0; c < HEAD_C; c++) {
    let g2 = 0;
    for (let j = 0; j < 3 * HEAD_C; j++) g2 += gp[gpBase + j] * Wlg[j * HEAD_C + c];
    for (let q = 0; q < HW; q++) {
      const o = (n * HW + q) * HEAD_C + c;
      act2[o] = relu((g2 + p1[o]) * s2[c] + b2[c]);
    }
  }
}

/* pass 路(单线程逐行:gp → linear_pass + b → relu → linear_pass2) */
function passFused(gp, Wpw, bp, Wp2, n, passOut) {
  const gpBase = n * 3 * HEAD_C;
  const h = new Float32Array(HEAD_C);
  for (let c = 0; c < HEAD_C; c++) {
    let acc = 0;
    for (let j = 0; j < 3 * HEAD_C; j++) acc += gp[gpBase + j] * Wpw[j * HEAD_C + c];
    h[c] = relu(acc + bp[c]);
  }
  for (let o = 0; o < 2; o++) {
    let acc = 0;
    for (let k = 0; k < HEAD_C; k++) acc += h[k] * Wp2[k * 2 + o];
    passOut[n * 2 + o] = acc;
  }
}

/* value 池化:mean, mean·0.5, mean·0.15(InputMask/scale、/quad 常数,mask 全 1 已折叠) */
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

/* value MLP(单线程逐行:gp → v2 + b → relu → vh/misc + b) */
function valueMlp(gp, Wv2, v2b, Wvh, vhb, Wm, mb, n, valOut, miscOut) {
  const gpBase = n * 3 * HEAD_C;
  const a = new Float32Array(V2_C);
  for (let j = 0; j < V2_C; j++) {
    let acc = 0;
    for (let k = 0; k < 3 * HEAD_C; k++) acc += gp[gpBase + k] * Wv2[k * V2_C + j];
    a[j] = relu(acc + v2b[j]);
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

/**
 * 创建 CPU 参考会话:同 evalBatch 契约(rows 可带 sym; optimism 同 ort 路径)。
 */
export function createCpuRefSession(blobBuffer, opt = {}) {
  const { meta, w: weights } = parseAewn(blobBuffer);
  assertPlanMeta(meta);
  const w = { get: (k) => { const v = weights.get(k); if (!v) throw new Error(`blob 缺张量 ${k}`); return v; }, stemTables: makeStemTables() };
  const cosT = w.get('rope.cos'), sinT = w.get('rope.sin');
  const EPS = meta.eps ?? RMS_EPS;

  /* 调试钩子:opt.debugTensors = ['stem','norm0',...] 时,evalBatch 顺带把
   * 指定中间张量(第 0 行)快照到 session.__debug[name](各 stage 语义见
   * evalBatch 内的 snap 调用点;与 ONNX 检查点张量一一对应)。 */
  const dbg = new Set(opt.debugTensors ?? []);
  const out0 = {};

  async function evalBatch(rows) {
    const n = rows.length;
    const snap = (name, arr) => { if (n > 0 && dbg.has(name)) out0[name] = Float32Array.from(arr); };
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

    /* 输入拼批:内核按 (n·22·361)/(n·19) 批布局索引,逐行数组先归位 */
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
      if (i === 0) snap('stem', trunk);

      /* PyTorch TransformerAttentionBlock ×8(前半)与 TransformerFFNBlock ×8(后半) */
      for (let b = 0; b < NUM_BLOCKS; b++) {
        rmsNorm(trunk, w.get(`attn${b}.norm`), i, normed);
        if (i === 0 && b === 0) snap('norm0', normed);
        qkvGemm(normed, w.get(`attn${b}.qkv`), i, qkv);
        if (i === 0 && b === 0) snap('q0', qkv.subarray(0, HW * C_TRUNK));
        ropeScatter(qkv, cosT, sinT, i, qh, kh, vh);
        if (i === 0 && b === 0) snap('qrope0', qh);
        attention(qh, kh, vh, i, scores, attn);
        if (i === 0 && b === 0) { snap('scores0', scores); snap('attn0', attn); }
        gemm(attn, w.get(`attn${b}.out`), i, C_TRUNK, C_TRUNK, proj, 'res', trunk);
        [trunk, proj] = [proj, trunk];
        snap(`resA${b}`, trunk);
        if (i === 0 && b === 0) snap('res0', trunk);

        rmsNorm(trunk, w.get(`ffn${b}.norm`), i, normed);
        gemm(normed, w.get(`ffn${b}.gate`), i, C_TRUNK, FFN_FUSED, gate, 'plain');
        if (i === 0 && b === 0) snap('gate0', gate);
        swiglu(gate, i, hidden);
        if (i === 0 && b === 0) snap('hidden0', hidden);
        gemm(hidden, w.get(`ffn${b}.ffn2`), i, FFN, C_TRUNK, proj, 'res', trunk);
        [trunk, proj] = [proj, trunk];
        snap(`resB${b}`, trunk);
        if (i === 0 && b === 0) snap('res1', trunk);
      }

      trunkFinal(trunk, w.get('trunkfinal.scale'), w.get('trunkfinal.bias'), i);
      if (i === 0) snap('trunkfinal', trunk);

      /* PolicyHead:conv1p(无偏) ‖ conv1g(折 scale)+biasg→relu → 池化 →
       * linear_g+gpbias+bias2→relu → conv2p;pass 路独立 MLP */
      gemm(trunk, w.get('policy.conv1p'), i, C_TRUNK, HEAD_C, p1, 'plain');
      gemm(trunk, w.get('policy.conv1g'), i, C_TRUNK, HEAD_C, actg, 'biasrelu', null, w.get('policy.conv1g_b'));
      if (i === 0) snap('actg', actg);
      poolPolicy(actg, i, gp);
      if (i === 0) { snap('gpp', gp); snap('p1', p1); }
      lingFused(p1, gp, w.get('policy.gp_ling'), w.get('policy.bias2_scale'), w.get('policy.bias2_bias'), i, act2);
      gemm(act2, w.get('policy.conv2p'), i, HEAD_C, 2, pol, 'plain');
      passFused(gp, w.get('policy.pass_w'), w.get('policy.pass_b'), w.get('policy.pass2'), i, pass);

      /* ValueHead:conv1(折 scale)+b→relu → 池化 → v2+relu → vh/misc;ownership 从 act1 */
      gemm(trunk, w.get('value.conv1'), i, C_TRUNK, HEAD_C, v1, 'biasrelu', null, w.get('value.conv1_b'));
      if (i === 0) snap('v1', v1);
      poolValue(v1, i, gp);
      if (i === 0) snap('gpv', gp);
      valueMlp(gp, w.get('value.v2'), w.get('value.v2_b'), w.get('value.vh'), w.get('value.vh_b'),
        w.get('value.misc'), w.get('value.misc_b'), i, val, misc);
      gemm(v1, w.get('value.own'), i, HEAD_C, 1, own, 'plain');

      void EPS; // eps 已在 meta 互验,RMS 内核用 plan 常量(与 f32(1e-6) 一致)
    }

    /* ---- JS 后处理:与 ort 路径(src/nn/session.js)同口径 ---- */
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
        policy,
        policyPass,
        winLoss: (e0 - e1) / (e0 + e1),
        ownership: Float32Array.from(own.subarray(i * HW, (i + 1) * HW)),
        scoreMean: misc[b] * 20,
        scoreStdev: softPlus(misc[b + 1]) * 20,
        scoreLead: misc[b + 2] * 20,
        shorttermScoreError: softPlus(misc[b + 5] * 0.5) * Math.sqrt(150),
        shorttermWinlossError: softPlus(misc[b + 4] * 0.5) * 0.5,
      };
    }
    return res;
  }

  return { ep: 'cpuref', evalBatch, meta, get __debug() { return out0; } };
}

export { N2 };
