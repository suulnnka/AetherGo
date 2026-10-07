/* ============================================================
 * AetherGo NN 会话门面 —— 唯一引擎:aethernn(自研 WebGPU 推理)
 *
 * 历史注记:onnxruntime-web 路径(?engine=ort)与 fp32 权重通道
 * (?weights=f32)为 A/B 逃生舱,2026-10-08 拍板移除 —— 引擎仅支持
 * i8 权重 + f16 激活存储/f32 累加(W8A16,INT8 报告 §4.1 方案 A),
 * 加载 models/b8c96h3tfrs_19.i8.aewn,无任何回退。
 *
 * evalBatch 契约(实现见 src/nn/webgpu/session.js):
 * - IO 契约按 katago dumponnx 实测(训练/浏览器两侧同一张图):
 *     输入 InputSpatial (N,22,19,19) f32 / InputGlobal (N,19,1,1) f32
 *     (require-exact-nnlen 无掩码图,掩码恒 1 已折叠)
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
 *   ★ 8 对称:rows[i].sym(可省,缺省 0)随行下发,aewnn 在 GPU 侧置换
 *     (stem gather,见 webgpu/session.js)。
 * - 批大小校准:src/nn/calibrate.js。
 * ============================================================ */

export { pickBatchSizeFromThroughput, calibrateMaxBatch } from './calibrate.js';

/**
 * 创建推理会话(aethernn,仅 WebGPU)。
 * opt: { modelUrl, onStatus(s), calibrate = true, aewnUrl?(缺省即 modelUrl) }
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
  const { createAewnnSession } = await import('./webgpu/session.js');
  return createAewnnSession(opt);
}
