/* ============================================================
 * AetherGo NN 会话封装 —— onnxruntime-web 加载与推理(仅 WebGPU)
 *
 * - 执行后端只走 WebGPU(2026-10-02 用户拍板去掉 WASM/CPU 回退):
 *   WebGPU 不可用直接报错,UI 显示原因 —— 不做慢速降级。
 * - IO 契约按 katago dumponnx 实测(训练/浏览器两侧同一张图):
 *     输入 InputSpatial (N,22,19,19) f32 / InputGlobal (N,19,1,1) f32 /
 *          InputMask (N,1,19,19) f32(require-exact-nnlen 无掩码图,恒喂 1)
 *     输出 OutputPolicy (N,C,19,19) 策略 logits,现取通道 0(主策略)。
 *          C 按模型版本(NEURAL_PLAN §4,2026-10-03 核实):C=1 v<12 老网(无乐观面);
 *          C=2 v≥12([0]主+[1]乐观);C=4 v16/v17-q 再加 2 通道 q 值。
 *          乐观插值(p + (pOpt−p)×λ,logits 空间、softmax 前)已拍板照抄 KataGo,待接。
 *          OutputPolicyPass (N,C,1,1) 取通道 0(插值同上)
 *          OutputValue (N,3) 行棋方视角 胜/负/无结果 logits
 *          OutputScoreValue (N,6) 行棋方视角 分数通道(裸值,后处理见下)
 *          OutputOwnership (N,1,19,19) 行棋方视角逐点归属(裸 pretanh 值,
 *          与 KataGo C++ ownerMap 同口径)
 *   ★ 视角约定(新旧模型双实测 + C++ 对证):模型输出一律「行棋方」视角,
 *     + = 行棋方优 —— nneval.cpp「the neural net gives us back the value from
 *     the perspective of the player」,训练目标即行棋方视角(trainingwrite.cpp
 *     fillValueTDTargets 按行棋方翻转)。C++ 拿到后再统一转白方视角存 NNOutput;
 *     本侧不转,消费方(search / worker)各自按 side 换算。
 *   ★ scoreValue 后处理(裸值 ×/softplus;乘数来自 .bin.gz 头,本模型
 *     20/20/20/40/0.25/150,outputScale=1;公式对齐 nneval.cpp v≥14 分支):
 *     scoreMean = raw0×20;scoreStdev = softplus(raw1)×20;
 *     scoreLead = raw2×20(已含贴目);stScoreErr = softplus(raw5×0.5)×√150;
 *     stWinlossErr = softplus(raw4×0.5)×0.5(即 sqrt(softplus²×0.25),nneval v≥14)。
 *   ★ 乐观策略插值(2026-10-03 拍板照抄 KataGo,onnxbackend.cpp 权威):
 *     每落点与 pass 做 p + (pOpt−p)×λ,**logits 空间、softmax 之前**;
 *     λ 逐行随请求传入(rows[i].optimism,缺省 1.0 = 树内;根评估传 0.2);
 *     C=1 老网无乐观面,λ 不生效(KataGo 同)。softmax 仍在 search 侧展开时做。
 * - ort-web 从 CDN 懒加载(本仓库无构建链;体积闸门 N4 再核定)。
 * - 批大小校准(2026-10-04):createSession 现测各档吞吐,产出常量上限 maxBatch,
 *   详见文件中部「批大小校准」注释块。
 * ============================================================ */

import { N, N2 } from '../engine.js';

const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;

/* scoreValue 裸通道 → 真值(目)的后处理乘数(desc.cpp v≥13 默认值,本模型头核实) */
const SCORE_MEAN_MULT = 20, SCORE_STDEV_MULT = 20, SCORE_LEAD_MULT = 20;
const ST_SCORE_ERR_MULT = Math.sqrt(150);
const ST_WL_ERR_MULT = 0.5;                     // sqrt(0.25),nneval.cpp v≥14
const softPlus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));

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

/* ==================== 批大小校准(KataGo benchmark 的加载时自动化) ====================
 * KataGo 的批上限由并发数静态推导 + 离线 CLI 基准(katago benchmark → 人写 config);
 * 浏览器单机跑在未知设备上,等价物是 createSession 时现测一次:纯前向、预热丢
 * shader 编译、取各档吞吐(rows/ms),选「达到最优吞吐 90% 的最小批」——
 * 强设备吞吐曲线平坦 → 选出小批(stale 低);弱设备大批明显更快 → 选出大批。
 * 只测当前后端、只产出常量上限,运行时零热调;换自研后端时本函数不改(只依赖
 * evalBatch 契约),最优值随新后端自动刷新。 */

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

/**
 * 创建推理会话。
 * opt: { modelUrl, onStatus(s), calibrate = true } —— onStatus 回报加载阶段(供 UI);
 * calibrate = false 跳过批大小校准(默认开,耗时约 0.3~2.5s,有硬性时间预算兜底)。
 * 返回 { ep: 'webgpu', evalBatch(rows), maxBatch, dispose() }:
 *   maxBatch:加载时校准的批大小上限(吞吐 ≥ 最优 90% 的最小批);校准失败为 null。
 *   rows: [{ spatial: Float32Array(22*361), global: Float32Array(19),
 *            optimism?: number(乐观插值 λ,缺省 1.0) }]
 *   evalBatch 返回同序数组:[{ policy: Float32Array(361)(插值后 logits,
 *                             softmax 在 search 侧), policyPass: number(同),
 *                             winLoss: number(-1..1,行棋方视角),
 *                             scoreLead: number(目,已含贴目,行棋方视角),
 *                             scoreMean/scoreStdev/shorttermScoreError: number(目),
 *                             ownership: Float32Array(361)(行棋方视角,裸 pretanh) }]
 *   WebGPU 不可用时抛错(消息带上原始原因,UI 直接显示)。
 */
export async function createSession(opt) {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('当前环境没有 WebGPU(需 Chrome/Edge 113+ 等启用 WebGPU 的浏览器)');
  }
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

  /* 输出形状自适应:policy 通道数按模型版本(C=1/2/4),取通道 0 = 主策略 */
  async function evalBatch(rows) {
    const n = rows.length;
    const sp = new Float32Array(n * 22 * N2);
    const gl = new Float32Array(n * 19);
    const mk = new Float32Array(n * N2);
    for (let i = 0; i < n; i++) {
      sp.set(rows[i].spatial, i * 22 * N2);
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
      maxBatch = await calibrateMaxBatch((rows) => evalBatch(rows));
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
