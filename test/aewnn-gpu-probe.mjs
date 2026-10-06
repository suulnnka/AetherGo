/* GPU 分段探针:1 行中盘(sym0,λ1),evalBatch 后把各中间缓冲拷出,
 * 与 cpuref 的同名快照逐级比对 —— 定位 GPU 侧首个发散 stage。
 * 运行:node test/aewnn-gpu-probe.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

const dawn = (await import('webgpu'));
const { create, globals } = dawn;
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: create([]) }, configurable: true });
globalThis.GPUBufferUsage = globals.GPUBufferUsage;
globalThis.GPUMapMode = globals.GPUMapMode;

const { N, BLACK, WHITE, newBoard, make } = await import(join(ROOT, 'src/engine.js'));
const { encodeFeatures } = await import(join(ROOT, 'src/nn/features.js'));
const { createCpuRefSession } = await import(join(ROOT, 'src/nn/webgpu/cpuref.js'));
const { createAewnnSession } = await import(join(ROOT, 'src/nn/webgpu/session.js'));

const blobBuf = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.aewn'));
const blob = blobBuf.buffer.slice(blobBuf.byteOffset, blobBuf.byteOffset + blobBuf.byteLength);

const SEQ = [[3,3],[15,15],[3,15],[15,3],[9,9],[3,9],[15,9],[9,3],[9,15],[5,5],[13,13],[5,13],[13,5],[7,7],[11,11],[7,11],[11,7],[2,8],[16,8],[8,2],[8,16]];
const bd = newBoard();
for (let i = 0; i < SEQ.length; i++) make(bd, SEQ[i][0] * N + SEQ[i][1], i % 2 === 0 ? BLACK : WHITE);
const moves = SEQ.map(([r, c]) => r * N + c);
const side = WHITE;
const f = encodeFeatures(bd, side, { recentMoves: moves, komi: 7.5 });
const SYM_TEST = Number(process.argv[2] ?? 0);
const row = { spatial: f.spatial, global: f.global, sym: SYM_TEST, optimism: 1.0 };

const dbgNames = ['stem', 'norm0', 'q0', 'qrope0', 'attn0', 'res0', 'hidden0', 'res1', 'trunkfinal', 'gpp', 'gpv', 'p1', 'actg', 'v1',
  ...Array.from({ length: 8 }, (_, b) => `resA${b}`), ...Array.from({ length: 8 }, (_, b) => `resB${b}`)];
const cpu = await createCpuRefSession(blob, { debugTensors: dbgNames });
const gpu = await createAewnnSession({ blob, calibrate: false, onStatus: () => {} });

const rows2 = [
  row,
  { spatial: f.spatial, global: f.global, sym: (SYM_TEST + 2) % 8, optimism: 1.0 },
];
await Promise.all([cpu.evalBatch(rows2), gpu.evalBatch(rows2)]);

/* GPU 中间缓冲拷回:trunk(A: stem 后)/normed/qkv/qh/attn/gate/hidden/gp/pol/pass/val/misc/own */
const HW = 361, C = 96, dev = gpu.__buffers;
const buf = (await import('node:fs'), null);
const device = (await import(join(ROOT, 'src/nn/webgpu/session.js')), null);
void buf; void device;

/* 通过 session 内部 device 不可达 —— 改用一次独立 evalBatch 后无法拷贝,
 * 这里用 Dawn 的 queue 已 idle 保证:直接建编码器需要 device 句柄。
 * 因此这里采用折衷:__buffers 上的每个 GPUBuffer 无法脱离 device 读,
 * 改为对 outputs 做 map —— 但 outputs 也非 mappable。
 * => 依赖 session 暴露 __debugCopy(names):见下。 */
/* 分段:upto=1 → stem 后;2 → norm0 后;3 → qkv;4 → rope;5 → attn;6 → out_proj+res;
 * 每块 9 dispatch;块0结束=9;trunkfinal=74;p1=75;actg=76;pool=77;ling=78;conv2p=79;pass=80;v1=81;poolv=82;mlp=83;own=84 */
const dumps = {};
const stages = [
  ['stem', 'trunk', 1, HW * C],
  ['norm0', 'normed', 2, HW * C],
  ['qrope0', 'qh', 4, 3 * HW * 32],
  ['attn0', 'attn', 5, HW * C],
  ['resA0', 'proj', 6, HW * C],
  ['resB0', 'trunk', 10, HW * C],
];
for (let b = 1; b < 8; b++) {
  stages.push([`resA${b}`, 'proj', 6 + 9 * b, HW * C]);
  stages.push([`resB${b}`, 'trunk', 10 + 9 * b, HW * C]);
}
stages.push(['trunkfinal', 'normed', 74, HW * C]);
stages.push(['p1', 'p1', 75, HW * 32]);
stages.push(['actg', 'actg', 76, HW * 32]);
stages.push(['gpp', 'gp', 77, 96]);
stages.push(['act2', 'act2', 78, HW * 32]);
stages.push(['v1', 'v1', 81, HW * 32]);
stages.push(['gpv', 'gp', 82, 96]);
for (const [name, bname, upto, floats] of stages) {
  const d = await gpu.__debugCopy([{ name, buf: bname, floats: floats * 2 }], rows2, upto);
  dumps[name] = d[name].subarray(0, floats);          // 行 0
  dumps[name + '_r1'] = d[name].subarray(floats);     // 行 1
}
const dumps2 = await gpu.__debugCopy([
  { name: 'spatialIn', buf: 'spatial', floats: 7942 },
  { name: 'unif', buf: 'uniform', floats: 0, bytes: 32, u32: true, count: 8 },
  { name: 'pol', buf: 'pol', floats: HW * 2 },
  { name: 'own', buf: 'own', floats: HW },
], [row]);
Object.assign(dumps, dumps2);

/* 行 1 单独跑 cpu,得到行 1 参照 */
await cpu.evalBatch([rows2[1]]);
console.log('uniform stem 槽(words):', Array.from(dumps.unif ?? []));
const CMP = {
  stem: 'stem', norm0: 'norm0', qrope0: 'qrope0', attn0: 'attn0', resA0: 'res0', resB0: 'res1',
};
for (let b = 1; b < 8; b++) { CMP[`resA${b}`] = `resA${b}`; CMP[`resB${b}`] = `resB${b}`; }
Object.assign(CMP, { trunkfinal: 'trunkfinal', p1: 'p1', actg: 'actg', gpp: 'gpp', v1: 'v1', gpv: 'gpv' });
const ROW1 = process.argv[3] === 'r1';
for (const [gname, cname] of Object.entries(CMP)) {
  const g = ROW1 ? dumps[gname + '_r1'] : dumps[gname];
  if (!g) continue;
  const c = cname ? cpu.__debug[cname] : null;
  if (!c) continue;
  {
    /* 行 1(不同 sym)也比对:cpu 对应行 = 第二次 eval?此处先只比行 0 */
  }
  let mx = 0, idx = -1;
  for (let i = 0; i < Math.min(g.length, c.length); i++) {
    const d = Math.abs(g[i] - c[i]);
    if (d > mx) { mx = d; idx = i; }
  }
  console.log(`${gname.padEnd(8)} vs ${cname.padEnd(8)} max|Δ|=${mx.toExponential(3)} @${idx} gpu=${g[idx]} cpu=${c[idx]}`);
}
{
  const nz = dumps.spatialIn.reduce((a, v) => a + (v !== 0 ? 1 : 0), 0);
  console.log(`gpu spatial 非零数: ${nz} / 7942(期望 = ch0 的 361 + 其它)`);
  console.log('gpu spatial[0..6]:', Array.from(dumps.spatialIn.slice(0, 7)));
  console.log('gpu trunk[0..6]:', Array.from(dumps.stem.slice(0, 7)));
  console.log('cpu stem[0..6]:', Array.from(cpu.__debug.stem.slice(0, 7)));
}
console.log('expect ch0 ones:', Array.from(f.spatial.slice(0, 6)), ' ch1:', Array.from(f.spatial.slice(361, 367)));
{
  const g = dumps.attn0, c = cpu.__debug.attn0;
  for (const [tag, arr] of [['gpu', g], ['cpu', c]]) {
    let line = '';
    for (let h = 0; h < 3; h++) {
      let nz = 0, mx = 0;
      for (let q = 0; q < 361; q++) for (let e = 0; e < 32; e++) {
        const v = arr[q * 96 + h * 32 + e];
        if (v !== 0) nz++;
        mx = Math.max(mx, Math.abs(v));
      }
      line += ` h${h}: nz=${nz} max=${mx.toExponential(2)}`;
    }
    console.log(`attn0 ${tag}:${line}`);
  }
  console.log('gpu attn0 [64..70]:', Array.from(g.slice(64, 70)));
  console.log('cpu attn0 [64..70]:', Array.from(c.slice(64, 70)));
}
console.log('\npol head:', Array.from(dumps.pol.slice(0, 8)));
console.log('own head:', Array.from(dumps.own.slice(0, 5)));
gpu.dispose();
process.exit(0);
