/* 分段对拍:cpuref 中间张量 vs ONNX 检查点(定位系统性偏差的第一站)。
 * 用法:node test/aewnn-stage-diff.mjs
 */
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const PY = process.env.PYTHON_BIN ?? '/home/a/miniconda3/envs/bleed/bin/python';

const { N, N2, BLACK, newBoard } = await import(join(ROOT, 'src/engine.js'));
const { encodeFeatures } = await import(join(ROOT, 'src/nn/features.js'));
const { createCpuRefSession } = await import(join(ROOT, 'src/nn/webgpu/cpuref.js'));

/* 空盘(sym=0,λ=1):输入确定,分段 diff 从 stem 开始 */
const bd = newBoard();
const f = encodeFeatures(bd, BLACK, { recentMoves: [], komi: 7.5 });

/* 1) 给 ONNX 侧喂同一输入 */
writeFileSync('/tmp/aewn_dbg_pos.json', JSON.stringify([
  { spatial: Array.from(f.spatial), global: Array.from(f.global) },
]));
execFileSync(PY, [join(ROOT, 'tools/diff/dump_onnx_intermediates.py'), '/tmp/aewn_dbg_pos.json'], { stdio: 'inherit' });

/* 2) cpuref 跑同一输入,收集检查点 */
const blobBuf = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.aewn'));
const blob = blobBuf.buffer.slice(blobBuf.byteOffset, blobBuf.byteOffset + blobBuf.byteLength);
const session = createCpuRefSession(blob, { debugTensors: ['stem', 'norm0', 'q0', 'qrope0', 'scores0', 'attn0', 'res0', 'gate0', 'hidden0', 'res1', 'trunkfinal', 'gpp', 'p1', 'gpv', 'actg', ...Array.from({length: 8}, (_, b) => `resA${b}`), ...Array.from({length: 8}, (_, b) => `resB${b}`)] });
await session.evalBatch([{ spatial: f.spatial, global: f.global }]);

/* 3) 逐检查点比对 */
const NAME_MAP = {};
for (let b = 0; b < 8; b++) {
  NAME_MAP[`resA${b}`] = `model.blocks.${2 * b}_` + [43, 99, 155, 211, 267, 323, 379, 435][b];
  NAME_MAP[`resB${b}`] = `model.blocks.${2 * b + 1}_` + [59, 115, 171, 227, 283, 339, 395, 451][b];
}
NAME_MAP.stem = 'trunk_initbias_2';
Object.assign(NAME_MAP, {
  stem: 'trunk_initbias_2',            // ONNX 是 NCHW (96,361) → 需转 NHWC 对比
  'trunk/trunk/tonhwc/transpose/3': null,
  norm0: 'model.blocks.0.norm1_scaled_12',
  q0: 'model.blocks.0.q_proj_nhwc_13',
  qrope0: 'model.blocks.0_qrope_rope_out_28',
  scores0: 'model.blocks.0_scoresscaled_36',
  attn0: 'model.blocks.0_attnnhwc_reshape_41',
  res0: 'model.blocks.0_43',
  hidden0: 'model.blocks.1_swiglu_57',
  res1: 'model.blocks.1_59',
  trunkfinal: 'model.act_trunkfinal_455',
  gpp: 'model.policy_head_g_gpconcat_464',
  gpv: 'model.value_head_v_gpconcat_482',
  p1: 'model.policy_head.conv1p_456',
  actg: 'model.policy_head.actg_460',
});
/* ONNX 头部已转回 NCHW 的检查点(NCHW→NHWC 后比较) */
const NCHW_GOLD = new Map([['trunkfinal', 96], ['p1', 32], ['actg', 32]]);
const HW = 361, C = 96, H = 3, D = 32;

function nchwToNhwc32(a, C2) {          // (C2,361) → (361,C2)
  const out = new Float32Array(HW * C2);
  for (let c = 0; c < C2; c++) for (let p = 0; p < HW; p++) out[p * C2 + c] = a[c * HW + p];
  return out;
}

function nchwToNhwc(a) {                // (96,361) → (361,96)
  const out = new Float32Array(HW * C);
  for (let c = 0; c < C; c++) for (let p = 0; p < HW; p++) out[p * C + c] = a[c * HW + p];
  return out;
}

const load = (name) => {
  const b = readFileSync(`/tmp/aewn_dbg/${name}.npy`);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

function diff(tag, ref, gold) {
  let mx = 0, idx = -1;
  const n = Math.min(ref.length, gold.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(ref[i] - gold[i]);
    if (d > mx) { mx = d; idx = i; }
  }
  console.log(`${tag.padEnd(12)} max|Δ|=${mx.toExponential(3)} @${idx} ref=${ref[idx]} gold=${gold[idx]}`);
  return mx;
}

let worst = {};
for (const [key, file] of Object.entries(NAME_MAP)) {
  const ref = session.__debug[key];
  if (!ref) { console.log(`${key}: 无 cpuref 快照`); continue; }
  if (!file) continue;
  const gold = load(file);
  let r = ref, g = gold;
  if (key === 'stem') { g = nchwToNhwc(gold); }
  if (NCHW_GOLD.has(key)) { g = nchwToNhwc32(gold, NCHW_GOLD.get(key)); }
  if (key === 'qrope0' || key === 'scores0') {
    /* head-major (3,361,32)/(3,361,361) vs ONNX (1,3,361,·) → 同布局展平 */
    r = ref;
  }
  worst[key] = diff(key, r, g);
}
console.log('\nworst:', JSON.stringify(worst, (k, v) => (typeof v === 'number' ? v.toExponential(2) : v)));
