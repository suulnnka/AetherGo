/* aewnn CPU 参考解释器 vs ort(CPU)golden 对拍。
 *
 * 验证链:packer(.aewn 权重映射)× 执行计划(镜像 PyTorch forward 的
 * 数学与融合)—— 全部在纯 JS 里跑(src/nn/webgpu/cpuref.js),不含 WGSL。
 * WGSL 内核另测(test/aewnn-wgsl-test.mjs,以本文件验证过的 cpuref 为参照)。
 *
 * 端侧:
 *   - golden = tools/diff/full_ort_server.py(CUDA_VISIBLE_DEVICES='' 强制
 *     CPU EP,确定性;输出即 evalBatch 契约形状)。
 *   - golden 走「CPU 预置换 + sym=0」(旧 search 语义),cpuref 走「原始特征
 *     + sym」(GPU gather 语义)—— 两侧一致同时验证 gather 表与 stem 置换。
 *
 * 运行:node test/aewnn-cpuref-test.mjs
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const PY = process.env.PYTHON_BIN ?? '/home/a/miniconda3/envs/bleed/bin/python';
const MODEL = join(ROOT, 'models/b8c96h3tfrs_19.onnx');

const { N, N2, BLACK, WHITE, PASS, newBoard, make, replayMoves } = await import(join(ROOT, 'src/engine.js'));
const { encodeFeatures } = await import(join(ROOT, 'src/nn/features.js'));
const { SYM8, permuteSpatial } = await import(join(ROOT, 'src/nn/symmetry.js'));
const { createCpuRefSession } = await import(join(ROOT, 'src/nn/webgpu/cpuref.js'));

let failed = 0;
const check = (name, cond, extra) => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  — ' + (extra ?? '')}`);
  if (!cond) failed++;
};

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
    if (!t) continue;
    /* ort 会往 stdout 打 EP 回退告警(EP Error 横幅),只认 JSON 数组行 */
    if (!t.startsWith('[')) { console.log(`[server stdout] ${t}`); continue; }
    pending.shift().resolve(JSON.parse(t));
  }
});
const golden = (rows) => new Promise((resolve, reject) => {
  pending.push({ resolve, reject });
  server.stdin.write(JSON.stringify({ rows }) + '\n');
});

/* ---------- 局面集:空盘 / 固定对局中盘 / 官子 / 双停 ---------- */
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
  const out = [];
  const bd0 = newBoard();
  out.push({ bd: bd0, side: BLACK, moves: [] });                       // 空盘
  const bd1 = newBoard();
  for (let i = 0; i < SEQ_MID.length; i++) {
    const [r, c] = SEQ_MID[i];
    make(bd1, r * N + c, i % 2 === 0 ? BLACK : WHITE);
  }
  out.push({ bd: bd1, side: WHITE, moves: SEQ_MID.map(([r, c]) => r * N + c) });
  const bd2 = newBoard();
  const toks = [];
  for (let i = 0; i < SEQ_END.length; i++) {
    const mv = SEQ_END[i][0] === undefined ? PASS : SEQ_END[i][0] * N + SEQ_END[i][1];
    toks.push(make(bd2, mv, i % 2 === 0 ? BLACK : WHITE));
  }
  out.push({ bd: bd2, side: WHITE, moves: SEQ_END.map((m) => (m[0] === undefined ? PASS : m[0] * N + m[1])) });
  return out;
}

/* ---------- 主流程 ---------- */
const blob = readFileSync(join(ROOT, 'models/b8c96h3tfrs_19.aewn'));
const bufCopy = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
const session = createCpuRefSession(bufCopy);

const CASES = [];
for (const { bd, side, moves } of positions()) {
  for (const sym of [0, 1, 5]) {                                       // 恒等/旋转/翻转族
    for (const optimism of [1.0, 0.2]) {
      CASES.push({ bd, side, moves: moves.slice(), sym, optimism });
    }
  }
}
console.log(`位置 × 对称 × λ = ${CASES.length} 例(单批 8 行对拍,含混批)`);

/* 批对拍:8 例一组;golden 输入做 CPU 预置换(旧语义),cpuref 走 sym */
const TOL = { policy: 3e-4, policyPass: 3e-4, winLoss: 1e-4, score: 2e-3, ownership: 3e-4 };
let worst = { policy: 0, winLoss: 0, ownership: 0 };
for (let c0 = 0; c0 < CASES.length; c0 += 8) {
  const group = CASES.slice(c0, c0 + 8);
  /* golden 与 cpuref 并发:两侧各自独立 encode(全新缓冲,互不干扰);
   * golden 走 CPU 预置换 + sym=0(旧语义),cpuref 走原始特征 + sym(gather)。 */
  const gRows = group.map(({ bd, side, moves, sym, optimism }) => {
    const f = encodeFeatures(bd, side, { recentMoves: moves, komi: 7.5 });
    return {
      spatial: Array.from(sym ? permuteSpatial(f.spatial, SYM8[sym], new Float32Array(f.spatial.length)) : f.spatial),
      global: Array.from(f.global),
      optimism,
    };
  });
  const [gold, refOuts] = await Promise.all([
    golden(gRows),
    session.evalBatch(group.map(({ bd, side, moves, sym, optimism }) => {
      const f = encodeFeatures(bd, side, { recentMoves: moves, komi: 7.5 });
      return { spatial: f.spatial, global: f.global, sym, optimism };
    })),
  ]);

  group.forEach((cs, i) => {
    const g = gold[i], r = refOuts[i];
    const tag = `case${c0 + i}(sym${cs.sym},λ${cs.optimism})`;
    let dp = 0;
    for (let p = 0; p < N2; p++) dp = Math.max(dp, Math.abs(g.policy[p] - r.policy[p]));
    let dov = 0;
    if (g.ownership && r.ownership) {
      for (let p = 0; p < N2; p++) dov = Math.max(dov, Math.abs(g.ownership[p] - r.ownership[p]));
    }
    const dw = Math.abs(g.winLoss - r.winLoss);
    worst.policy = Math.max(worst.policy, dp);
    worst.winLoss = Math.max(worst.winLoss, dw);
    worst.ownership = Math.max(worst.ownership, dov);
    check(`${tag} policy`, dp < TOL.policy, `max|Δ|=${dp.toExponential(2)}`);
    check(`${tag} policyPass`, Math.abs(g.policyPass - r.policyPass) < TOL.policyPass,
      `${g.policyPass.toFixed(4)} vs ${r.policyPass.toFixed(4)}`);
    check(`${tag} winLoss`, dw < TOL.winLoss, `|Δ|=${dw.toExponential(2)}`);
    check(`${tag} scoreLead`, Math.abs(g.scoreLead - r.scoreLead) < TOL.score,
      `${g.scoreLead.toFixed(3)} vs ${r.scoreLead.toFixed(3)}`);
    check(`${tag} scoreMean`, Math.abs(g.scoreMean - r.scoreMean) < TOL.score);
    check(`${tag} ownership`, dov < TOL.ownership, `max|Δ|=${dov.toExponential(2)}`);
  });
}
console.log(`最差偏差: policy=${worst.policy.toExponential(2)} winLoss=${worst.winLoss.toExponential(2)} ownership=${worst.ownership.toExponential(2)}`);

server.stdin.end();
server.kill();
process.exit(failed ? 1 : 0);
