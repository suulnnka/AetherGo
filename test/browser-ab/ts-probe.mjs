/* 临时探针:timestampWrites(pass 起止)+ 按内核族跳过 dispatch 的消融法,
 * 定位高批边际成本所在内核家族。
 * 用法:node test/browser-ab/ts-probe.mjs [n=32] [batchHi=8] [rounds=7] */
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: create([]) }, configurable: true });

const N_ROWS = Number(process.argv[2] ?? 32);
const BATCH_HI = Number(process.argv[3] ?? 8);
const ROUNDS = Number(process.argv[4] ?? 7);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'timestamp-query'],
  requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(9, adapter.limits.maxStorageBuffersPerShaderStage) },
});
/* session 会自建设备:劫持 requestAdapter 使其复用本探针设备(带 timestamp) */
navigator.gpu.requestAdapter = async () => ({
  features: adapter.features,
  limits: adapter.limits,
  requestDevice: async () => device,
});

const codeOf = new WeakMap(), nameOf = new WeakMap();
const origShader = device.createShaderModule.bind(device);
device.createShaderModule = (d) => {
  const m = origShader(d);
  codeOf.set(m, d.code);
  return m;
};
const origPipe = device.createComputePipeline.bind(device);
device.createComputePipeline = (d) => {
  const p = origPipe(d);
  const code = codeOf.get(d.compute.module) ?? '';
  const m = code.match(/fn\s+(\w+)\s*\(/);
  nameOf.set(p, m ? m[1] : '?');
  return p;
};

let skipRe = null;                 // 命中即跳过该 dispatch(消融)
let passNames = null;
const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
const tsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
const rbTs = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const origPass = GPUCommandEncoder.prototype.beginComputePass;
GPUCommandEncoder.prototype.beginComputePass = function (...a) {
  a[0] = { ...a[0], timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } };
  const pass = origPass.apply(this, a);
  passNames = [];
  const origSet = pass.setPipeline.bind(pass);
  pass.setPipeline = (p) => { passNames.push(nameOf.get(p) ?? '?'); return origSet(p); };
  const origDisp = pass.dispatchWorkgroups.bind(pass);
  pass.dispatchWorkgroups = (...args) => {
    const nm = passNames[passNames.length - 1];
    if (skipRe && skipRe.test(nm)) return undefined;
    return origDisp(...args);
  };
  const origEnd = pass.end.bind(pass);
  pass.end = () => {
    origEnd();
    this.resolveQuerySet(qs, 0, 2, tsBuf, 0);
    this.copyBufferToBuffer(tsBuf, 0, rbTs, 0, 16);
  };
  return pass;
};

const { N2 } = await import('../../src/engine.js');
const { createAewnnSession } = await import('../../src/nn/webgpu/session.js');
const { ensureBlob } = await import('../blob-helper.mjs');
const blob = ensureBlob('b8c96h3tfrs_19.i8.aewn', ['i8']);
const s = await createAewnnSession({ blob, calibrate: false, batchHi: BATCH_HI, onStatus: () => {} });
const mk = (n) => Array.from({ length: n }, () => ({ spatial: new Float32Array(22 * N2), global: new Float32Array(19) }));
await s.evalBatch(mk(N_ROWS));

const period = device.limits.timestampPeriod ?? 1;
async function timedOnce(skip) {
  skipRe = skip;
  const t0 = performance.now();
  await s.evalBatch(mk(N_ROWS));
  const wall = performance.now() - t0;
  await rbTs.mapAsync(GPUMapMode.READ);
  const f = new Float64Array(rbTs.getMappedRange().slice(0));
  rbTs.unmap();
  const gpuUs = (f[1] - f[0]) * period / 1000;
  skipRe = null;
  return { wall, gpuUs, dispatches: passNames.length, names: passNames };
}
/* 预热一次 */
await timedOnce(null);
const CONFIGS = [
  ['全量', null],
  ['跳 GEMM', /^gemm/],
  ['跳 flash', /^flash$/],
  ['跳 elementwise', /^(rms|swiglu|rope|trunkFinal)$/],
  ['跳 stem', /^stem$/],
  ['跳 头部小核', /^(poolPolicy|ling|passHead|valueMlp|gemmSmall|poolValue)$/],
];
const out = {};
for (const [name, re] of CONFIGS) {
  let gpuMin = Infinity, wallMin = Infinity, last = null;
  for (let r = 0; r < ROUNDS; r++) {
    last = await timedOnce(re);
    gpuMin = Math.min(gpuMin, last.gpuUs); wallMin = Math.min(wallMin, last.wall);
  }
  out[name] = { gpuMin, wallMin, dispatches: last.dispatches };
  console.log(`${name.padEnd(10)} GPU ${gpuMin.toFixed(0).padStart(6)}µs  墙钟 ${wallMin.toFixed(1).padStart(6)}ms  dispatches ${last.dispatches}`);
}
const full = out['全量'].gpuMin;
console.log(`\n占比(GPU):GEMM ≈ ${((full - out['跳 GEMM'].gpuMin) / full * 100).toFixed(0)}%  flash ≈ ${((full - out['跳 flash'].gpuMin) / full * 100).toFixed(0)}%  elementwise ≈ ${((full - out['跳 elementwise'].gpuMin) / full * 100).toFixed(0)}%  stem ≈ ${((full - out['跳 stem'].gpuMin) / full * 100).toFixed(0)}%  头部 ≈ ${((full - out['跳 头部小核'].gpuMin) / full * 100).toFixed(0)}%`);
s.dispose();
process.exit(0);
