/* 引擎级一致性探针(临时):完整 PUCT 搜索下,三个推理后端的选点对拍。
 *
 *   A. ort CPU fp32 golden(tools/diff/full_ort_server.py,onnx)
 *   B. aethernn GPU int8 量化版(.i8.aewn —— 浏览器缺省路径)
 *   C. aethernn GPU fp32 golden(.aewn)
 *
 * 每个局面 × 每引擎:clearEvalCache + reuseTree=false + 固定 rngSeed +
 * 温度 0(LCB 选点)→ 同引擎确定性;跨引擎只允许 f16/量化噪声级分歧。
 * 预期:B/C 与 A 的 top1 选点基本一致,分歧集中在近平局(gap 小),
 * winRate/scoreLead 差在引擎侧指标同带。
 *
 * 运行:PYTHON_BIN=python node test/engine-align-probe.mjs
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const PY = process.env.PYTHON_BIN ?? 'python';
const MODEL = join(ROOT, 'models/b8c96h3tfrs_19.onnx');
const VISITS = Number(process.env.VISITS ?? 64);

const E = await import(pathToFileURL(join(ROOT, 'src/engine.js')).href);
const S = await import(pathToFileURL(join(ROOT, 'src/nn/search.js')).href);
const { encodeFeatures } = await import(pathToFileURL(join(ROOT, 'src/nn/features.js')).href);
const { createAewnnSession } = await import(pathToFileURL(join(ROOT, 'src/nn/webgpu/session.js')).href);

const dawn = await import('webgpu');
Object.assign(globalThis, dawn.globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: dawn.create([]) }, configurable: true });

/* ---------- golden 服务 ---------- */
const server = spawn(PY, [join(ROOT, 'tools/diff/full_ort_server.py'), MODEL], {
  stdio: ['pipe', 'pipe', 'inherit'],
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '' },
});
let buf = '';
const pending = [];
server.stdout.on('data', (d) => {
  buf += d.toString();
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
    const t = line.trim();
    if (!t || !t.startsWith('[')) continue;
    pending.shift().resolve(JSON.parse(t));
  }
});
const goldenEval = (rows) => new Promise((resolve, reject) => {
  pending.push({ resolve, reject });
  server.stdin.write(JSON.stringify({
    rows: rows.map((r) => ({ spatial: Array.from(r.spatial), global: Array.from(r.global), sym: r.sym ?? 0, optimism: r.optimism ?? 1.0 })),
  }) + '\n');
});
const ortSession = { evalBatch: goldenEval };

/* ---------- aethernn 双版 ---------- */
const load = (f) => {
  const b = readFileSync(join(ROOT, 'models', f));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const gpuQ = await createAewnnSession({ blob: load('b8c96h3tfrs_19.i8.aewn'), calibrate: false, onStatus: () => {} });
const gpuF = await createAewnnSession({ blob: load('b8c96h3tfrs_19.aewn'), calibrate: false, onStatus: () => {} });

/* ---------- 局面 ---------- */
const SEQ_MID = [
  [3, 3], [15, 15], [3, 15], [15, 3], [9, 9], [3, 9], [15, 9], [9, 3],
  [9, 15], [5, 5], [13, 13], [5, 13], [13, 5], [7, 7], [11, 11], [7, 11],
  [11, 7], [2, 8], [16, 8], [8, 2], [8, 16],
];
const SEQ_TAIL = [[0, 0], [0, 1], [18, 18], [18, 17], [1, 1], [1, 2], [17, 17], [17, 16], [0, 18], [0, 17], [18, 0], [18, 1]];
const POSITIONS = [];
{
  POSITIONS.push({ name: '空盘', seq: [] });
  POSITIONS.push({ name: '布局10手', seq: SEQ_MID.slice(0, 10) });
  POSITIONS.push({ name: '中盘21手', seq: SEQ_MID });
  POSITIONS.push({ name: '中后盘33手', seq: [...SEQ_MID, ...SEQ_TAIL] });
}
const coord = (mv) => (mv === E.PASS ? 'pass' : `${'ABCDEFGHJKLMNOPQRST'[mv % E.N]}${E.N - ((mv / E.N) | 0)}`);

const ENGINES = [
  ['ort-golden', ortSession],
  ['aewnn-i8', gpuQ],
  ['aewnn-f32', gpuF],
];

let agreeA = 0, agreeF = 0;
for (let pi = 0; pi < POSITIONS.length; pi++) {
  const { name, seq } = POSITIONS[pi];
  console.log(`\n== ${name}(visits=${VISITS},rngSeed 固定,温度 0)==`);
  const results = [];
  for (const [tag, session] of ENGINES) {
    const bd = E.newBoard();
    const side = E.replayMoves(bd, seq.map(([r, c]) => r * E.N + c));
    S.clearEvalCache();
    const r = await S.nnSearchBest(bd, side, {
      session, visits: VISITS, maxBatch: 8, reuseTree: false,
      rngSeed: 0xA5F30 + pi, recentMoves: seq.map(([r2, c2]) => r2 * E.N + c2),
      debug: true,
    });
    const top = (r.debug?.rootChildMoves ?? []).slice(0, 3)
      .map((mv, i) => `${coord(mv)}/${(r.debug.rootChildStats?.[i]?.visits ?? '?')}v`)
      .join(' ');
    console.log(`  ${tag.padEnd(10)} 选点 ${coord(r.move).padEnd(4)} winRate ${(r.winRate * 100).toFixed(1)}% lead ${(r.scoreLead ?? 0).toFixed(2)} nnCalls ${r.nnCalls} ${top}`);
    results.push({ tag, move: r.move, wr: r.winRate, lead: r.scoreLead });
    void side;
  }
  const [a, q, f32] = results;
  if (a.move === q.move) agreeA++;
  else console.log(`  ⇒ i8 与 golden 分歧:${coord(a.move)} vs ${coord(q.move)}(winRate 差 ${Math.abs(a.wr - q.wr).toFixed(3)})`);
  if (a.move === f32.move) agreeF++;
  else console.log(`  ⇒ f32 与 golden 分歧:${coord(a.move)} vs ${coord(f32.move)}(winRate 差 ${Math.abs(a.wr - f32.wr).toFixed(3)})`);
}
console.log(`\n选点一致:ort vs i8 = ${agreeA}/4,ort vs f32 = ${agreeF}/4`);
gpuQ.dispose(); gpuF.dispose();
server.stdin.end(); server.kill();
