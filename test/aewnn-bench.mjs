/* aewnn 吞吐基准(Dawn,dawn-node 环境)。
 *
 * 注意:WSL2 开发机上 Dawn 通常落在 llvmpipe(软件 Vulkan),数字只反映
 * 软件渲染下限,不代表目标设备 —— 正式口径在浏览器(桌面 Chrome + 中端
 * Android)实测。本基准的用途:
 *   1. 满容量批(CAP=32)与全 dispatch 链的烟测;
 *   2. 校准口径(≥最优 90% 的最小批)在真设备上的现成测速器;
 *   3. 会话创建耗时(shader 编译 × 84 + 权重上传)。
 * 运行:node test/aewnn-bench.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const dawn = await import('webgpu');
Object.assign(globalThis, dawn.globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: dawn.create([]) }, configurable: true });

const { BLACK, newBoard } = await import(pathToFileURL(join(ROOT, 'src/engine.js')).href);
const { encodeFeatures } = await import(pathToFileURL(join(ROOT, 'src/nn/features.js')).href);
const { createAewnnSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/session.js')).href);
const { pickBatchSizeFromThroughput } = await import(pathToFileURL(join(ROOT, 'src/nn/session.js')).href);

const { ensureBlob } = await import('./blob-helper.mjs');
const blob = ensureBlob('b8c96h3tfrs_19.i8.aewn');

const adapter = await navigator.gpu.requestAdapter();
const ai = adapter.info ?? {};
console.log(`adapter: ${ai.vendor ?? '?'} ${ai.architecture ?? ''} ${ai.device ?? ''}(${ai.description ?? ''})`);
const t0 = performance.now();
const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });
const loadMs = performance.now() - t0;
console.log(`会话创建(i8f16,含 84 pipeline 编译 + 1.1MB 权重上传): ${loadMs.toFixed(0)}ms,dispatch ${gpu.dispatchCount}`);

/* 中盘特征 ×32 份做满容量验证与吞吐 */
const f = encodeFeatures(newBoard(), BLACK, { recentMoves: [], komi: 7.5 });

const proto = { spatial: f.spatial, global: f.global, sym: 0, optimism: 1.0 };
const entries = [];
for (const size of [1, 2, 4, 8, 16, 32]) {
  const rows = Array.from({ length: size }, () => proto);
  await gpu.evalBatch(rows);                        // 预热
  let minMs = Infinity, minSync = Infinity;
  for (let k = 0; k < 10; k++) {
    const t = performance.now();
    const p = gpu.evalBatch(rows);                  // 同步前缀:输入 writeBuffer + uniform 补丁
    const tSync = performance.now() - t;            //   = JS 侧准备工作(不含 GPU 执行)
    await p;                                        //   余下 = GPU 执行 + 读回 map + 后处理
    const tTotal = performance.now() - t;
    if (tTotal < minMs) { minMs = tTotal; minSync = tSync; }
  }
  entries.push([size, size / minMs]);
  console.log(`批 ${String(size).padStart(2)}: 总 ${minMs.toFixed(2)}ms(JS 准备 ${minSync.toFixed(2)}ms)= ${(size / minMs).toFixed(2)} rows/ms`);
}
console.log(`校准口径(≥最优90% 最小批)→ maxBatch = ${pickBatchSizeFromThroughput(entries)}`);
gpu.dispose();
process.exit(0);
