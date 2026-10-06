/* ============================================================
 * AetherGo NN 会话门面 —— 引擎选择 + onnxruntime-web 路径(A/B 保留)
 *
 * 自研引擎(2026-10-06 立项,docs/WEBGPU_ENGINE_RESEARCH.md):
 *   默认走 src/nn/webgpu/session.js(aethernn,仅支持 b8c96h3tfrs,权重经
 *   training/pack_aewn.py 打包为 .aewn);本文件的 ort 路径保留为 A/B 与
 *   逃生舱,通过 ?engine=ort 强制。evalBatch 契约两路完全一致:
 *
 * - IO 契约按 katago dumponnx 实测(训练/浏览器两侧同一张图):
 *     输入 InputSpatial (N,22,19,19) f32 / InputGlobal (N,19,1,1) f32 /
 *          InputMask (N,1,19,19) f32(require-exact-nnlen 无掩码图,恒喂 1)
 *     输出 OutputPolicy (N,C,19,19) 策略 logits,C=2 v≥12([0]主+[1]乐观)
 *          OutputPolicyPass (N,C,1,1) 取通道 0(插值同上)
 *          OutputValue (N,3) 行棋方视角 胜/负/无结果 logits
 *          OutputScoreValue (N,6) 行棋方视角 分数通道(裸值,后处理见下)
 *          OutputOwnership (N,1,19,19) 行棋方视角逐点归属(裸 pretanh 值)
 *   ★ 视角约定:模型输出一律「行棋方」视角,+ = 行棋方优;本侧不转,
 *     消费方(search / worker)各自按 side 换算。
 *   ★ scoreValue 后处理(裸值 ×/softplus;本模型 20/20/20/40/0.25/150):
 *     scoreMean = raw0×20;scoreStdev = softplus(raw1)×20;
 *     scoreLead = raw2×20(已含贴目);stScoreErr = softplus(raw5×0.5)×√150;
 *     stWinlossErr = softplus(raw4×0.5)×0.5。
 *   ★ 乐观策略插值:每落点与 pass 做 p + (pOpt−p)×λ,logits 空间、softmax 前;
 *     λ 逐行随请求传入(rows[i].optimism,缺省 1.0)。
 *   ★ 8 对称:rows[i].sym(可省,缺省 0)随行下发 —— aewnn 在 GPU 侧置换
 *     (stem gather);ort 路径在本文件内做 CPU 置换(成本同旧 search 侧)。
 * - 批大小校准:src/nn/calibrate.js,两路共用同一测速器与 maxBatch 语义。
 * - 权重精度:aewnn 缺省加载 int8 量化版(models/b8c96h3tfrs_19.i8.aewn,
 *   W8A16 + f16 激活存储,INT8 报告 §4.1 方案 A);?weights=f32 切 fp32 版。
 * ============================================================ */
import { N, N2 } from '../engine.js';
import { SYM8, permuteSpatial } from './symmetry.js';
import { calibrateMaxBatch as calibrateMaxBatchLocal } from './calibrate.js';

export { pickBatchSizeFromThroughput, calibrateMaxBatch } from './calibrate.js';

const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;

/* scoreValue 裸通道 → 真值(目)的后处理乘数(desc.cpp v≥13 默认值,本模型头核实) */
const SCORE_MEAN_MULT = 20, SCORE_STDEV_MULT = 20, SCORE_LEAD_MULT = 20;
const ST_SCORE_ERR_MULT = Math.sqrt(150);
const ST_WL_ERR_MULT = 0.5;                     // sqrt(0.25),nneval.cpp v≥14
const softPlus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));

/* 引擎选择:opt.engine > URL ?engine= > 缺省 aethernn(ort 为 A/B 逃生舱) */
function pickEngine(opt) {
  if (opt?.engine) return opt.engine;
  if (typeof location !== 'undefined') {
    const q = new URLSearchParams(location.search).get('engine');
    if (q === 'ort') return 'ort';
  }
  return 'aewnn';
}

let ortP = null;
async function loadOrt() {
  if (ortP) return ortP;
  ortP = (async () => {
    /* Worker(module)与主线程都能用动态 import;必须取 .mjs —— dist/ort.all.min.js
     * 是 UMD/CJS(浏览器 import() 下命名空间为空,ort.env 为 undefined),
     * 只有 .mjs 有具名导出 env/Tensor/InferenceSession */
    const ort = await import(/* @vite-ignore */ `${ORT_CDN}/ort.all.min.mjs`);
    ort.env.wasm.wasmPaths = `${ORT_CDN}/`;
    return ort;
  })();
  return ortP;
}

/**
 * 创建推理会话。
 * opt: { modelUrl, onStatus(s), calibrate = true, engine? = 'aewnn' | 'ort',
 *         aewnUrl?(aewnn 路径,缺省由 modelUrl 换后缀) }
 * 返回 { ep, evalBatch(rows), maxBatch, dispose() }:
 *   rows: [{ spatial: Float32Array(22*361), global: Float32Array(19),
 *            sym?: number(8 对称,缺省 0), optimism?: number(乐观插值 λ,缺省 1.0) }]
 *   evalBatch 返回同序数组:[{ policy: Float32Array(361)(插值后 logits,
 *                             softmax 在 search 侧), policyPass: number(同),
 *                             winLoss: number(-1..1,行棋方视角),
 *                             scoreLead: number(目,已含贴目,行棋方视角),
 *                             scoreMean/scoreStdev/shorttermScoreError: number(目),
 *                             ownership: Float32Array(361)(行棋方视角,裸 pretanh) }]
 *   WebGPU 不可用或权重不符时抛错(消息带上原始原因,UI 直接显示)。
 */
export async function createSession(opt) {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('当前环境没有 WebGPU(需 Chrome/Edge 113+ 等启用 WebGPU 的浏览器)');
  }
  const engine = pickEngine(opt);
  if (engine !== 'ort') {
    if (engine !== 'aewnn') throw new Error(`未知引擎: ${engine}`);
    const { createAewnnSession } = await import('./webgpu/session.js');
    /* 权重精度:缺省 int8 量化版(.i8.aewn,W8A16 + f16 激活);
     * opt.weightsF32 或 ?weights=f32 强制 fp32 版(.aewn)。 */
    let w32 = opt.weightsF32;
    if (w32 === undefined && typeof location !== 'undefined') {
      w32 = new URLSearchParams(location.search).get('weights') === 'f32';
    }
    return createAewnnSession({ ...opt, weightsF32: !!w32 });
  }
  return createOrtSession(opt);
}

/* ==================== ort-web 路径(A/B 与逃生舱保留) ==================== */

async function createOrtSession(opt) {
  const ort = await loadOrt();
  const status = opt.onStatus ?? (() => {});
  status('加载 onnxruntime-web');

  status('创建 WebGPU 会话');
  const session = await ort.InferenceSession.create(opt.modelUrl, {
    executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
  });
  status('模型就绪(WebGPU)');

  const nIn = session.inputNames, nOut = session.outputNames;
  const maskBuf = () => new Float32Array(N2).fill(1);
  const permBuf = new Float32Array(22 * N2);        // 对称置换工作缓冲

  /* 输出形状自适应:policy 通道数按模型版本(C=1/2/4),取通道 0 = 主策略 */
  async function evalBatch(rows) {
    const n = rows.length;
    const sp = new Float32Array(n * 22 * N2);
    const gl = new Float32Array(n * 19);
    const mk = new Float32Array(n * N2);
    for (let i = 0; i < n; i++) {
      const sym = rows[i].sym ?? 0;
      const src = sym ? permuteSpatial(rows[i].spatial, SYM8[sym], permBuf) : rows[i].spatial;
      sp.set(src, i * 22 * N2);
      gl.set(rows[i].global, i * 19);
      mk.fill(1, i * N2, (i + 1) * N2);
    }
    const feeds = {
      [nIn[0]]: new ort.Tensor('float32', sp, [n, 22, N, N]),
      [nIn[1]]: new ort.Tensor('float32', gl, [n, 19, 1, 1]),
      [nIn[2]]: new ort.Tensor('float32', mk, [n, 1, N, N]),
    };
    const out = await session.run(feeds);
    const pol = out[nOut.find((nm) => nm.includes('Policy') && !nm.includes('Pass'))].data;
    const polPass = out[nOut.find((nm) => nm.includes('PolicyPass'))].data;
    const val = out[nOut.find((nm) => nm.includes('Value') && !nm.includes('Score'))].data;
    const own = out[nOut.find((nm) => nm.includes('Ownership'))]?.data;
    const sv = out[nOut.find((nm) => nm.includes('ScoreValue'))]?.data;
    /* policy 通道数:(N,C,19,19) → C = pol.length / (n*361) */
    const polC = pol.length / (n * N2);
    const passC = polPass.length / n;
    const res = new Array(n);
    for (let i = 0; i < n; i++) {
      const optimism = rows[i].optimism ?? 1.0;   // 树内 1.0;根评估 0.2(GTP 配方)
      /* 乐观插值:C≥2 时 [1] 为乐观面,logits 空间线性插值(C++ onnxbackend 同);
       * C=1 老网无乐观面,原样返回(等价 λ 不生效)。softmax 在 search 侧做。 */
      const policy = new Float32Array(N2);
      if (polC >= 2 && optimism !== 1.0) {
        for (let p = 0; p < N2; p++) {
          const p0 = pol[i * polC * N2 + p], pOpt = pol[i * polC * N2 + N2 + p];
          policy[p] = p0 + (pOpt - p0) * optimism;
        }
      } else {
        for (let p = 0; p < N2; p++) policy[p] = pol[i * polC * N2 + p];
      }
      const passBase = polPass[i * passC];
      const policyPass = (polC >= 2 && optimism !== 1.0)
        ? passBase + (polPass[i * passC + 1] - passBase) * optimism
        : passBase;
      /* value:3 logits(行棋方 胜/负/无结果)→ 行棋方视角胜率差。
       * 面积计分 + 非 simple ko 下无结果不可能,C++ 侧把无结果 logit 压 −1e5
       * (nneval.cpp)—— 本引擎恒中国规则,等价实现:直接按 (e0+e1) 归一。 */
      const l0 = val[i * 3], l1 = val[i * 3 + 1];
      const m = Math.max(l0, l1);
      const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
      const ownership = own ? new Float32Array(N2) : null;
      if (own) for (let p = 0; p < N2; p++) ownership[p] = own[i * N2 + p];
      /* scoreValue:行棋方视角分数(裸值 ×20 / softplus 后处理,见文件头)。
       * 短期目差误差 = softplus(raw5×0.5)×√150(nneval.cpp v≥14 分支)。 */
      let scoreMean, scoreStdev, scoreLead, shorttermScoreError, shorttermWinlossError;
      if (sv) {
        const b = i * 6;
        scoreMean = sv[b] * SCORE_MEAN_MULT;
        scoreStdev = softPlus(sv[b + 1]) * SCORE_STDEV_MULT;
        scoreLead = sv[b + 2] * SCORE_LEAD_MULT;
        shorttermScoreError = softPlus(sv[b + 5] * 0.5) * ST_SCORE_ERR_MULT;
        shorttermWinlossError = softPlus(sv[b + 4] * 0.5) * ST_WL_ERR_MULT;
      }
      res[i] = {
        policy,
        policyPass,
        winLoss: (e0 - e1) / (e0 + e1),
        ownership,
        scoreMean, scoreStdev, scoreLead, shorttermScoreError, shorttermWinlossError,
      };
    }
    return res;
  }

  /* 批大小校准:对当前设备现测一次,产出常量上限(失败回落 null → 搜索侧用默认) */
  let maxBatch = null;
  if (opt.calibrate !== false) {
    try {
      status('校准批次大小');
      maxBatch = await calibrateMaxBatchLocal((rows) => evalBatch(rows));
      status(`校准完成(批上限 ${maxBatch})`);
    } catch {
      maxBatch = null;                           // 校准失败不影响可用性
    }
  }

  return {
    ep: 'webgpu',
    evalBatch,
    maxBatch,
    dispose() { return session.release(); },
  };
}
