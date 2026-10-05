/* NN 引擎端到端验证(无浏览器环境):
 *   特征编码(JS)→ ONNX 推理(python ort_server 子进程)→ 异步 PUCT → 完整自对弈。
 * 用法:node test/nn-e2e.mjs [模型路径,默认 models/b8c96h3tfrs_19.onnx] [黑方访问数] [白方访问数]
 * (UCT 引擎已移除;NN vs NN 对战用 test/nn-match.mjs)
 */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync } from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const model = process.argv[2] ?? join(ROOT, 'models', 'b8c96h3tfrs_19.onnx');
const blackVisits = Number(process.argv[3] ?? 24);
const whiteVisits = Number(process.argv[4] ?? 48);

const E = await import(join(ROOT, 'src/engine.js'));
const { nnSearchBest } = await import(join(ROOT, 'src/nn/search.js'));

const PY = '/home/a/miniconda3/envs/bleed/bin/python';
const proc = spawn(PY, [join(ROOT, 'training/ort_server.py'), model], { stdio: ['pipe', 'pipe', 'inherit'] });
let buf = '';
const pending = [];
proc.stdout.on('data', (d) => {
  buf += d.toString();
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
    if (!line.trim()) continue;
    pending.shift().resolve(JSON.parse(line));
  }
});

function evalBatch(rows) {
  return new Promise((resolve, reject) => {
    pending.push({ resolve, reject });
    proc.stdin.write(JSON.stringify({
      rows: rows.map((r) => ({ spatial: Array.from(r.spatial), global: Array.from(r.global) })),
    }) + '\n');
  });
}

const session = { evalBatch };

async function playGame() {
  const bd = E.newBoard();
  const hist = [];
  let side = 0, lastPass = false, plies = 0;
  let t0 = Date.now();
  let nnEvalMs = 0;
  for (let t = 0; t < 760; t++) {
    const s = Date.now();
    const r = await nnSearchBest(bd, side, {
      session, visits: side === 0 ? blackVisits : whiteVisits, batch: 8,
      reuseTree: process.env.NOREUSE !== '1',
      recentMoves: hist.slice(),
    });
    nnEvalMs += Date.now() - s;
    const mv = r.move;
    if (mv !== E.PASS && !E.isLegal(bd, side, mv)) throw new Error('非法着法 ' + mv);
    hist.push(mv);
    const pass = mv === E.PASS;
    if (pass && lastPass) break;
    lastPass = pass;
    E.make(bd, mv, side);
    side ^= 1;
    plies++;
  }
  const s2 = E.finalScore(bd);
  const blackWin = s2.margin > 0;
  return { plies, margin: s2.margin, blackWin, ms: Date.now() - t0, nnEvalMs, score: s2 };
}

/* ---- 主流程 ---- */
console.log(`模型: ${model.split('/').pop()}  自对弈(黑 ${blackVisits} / 白 ${whiteVisits} 访问)`);
// 预热一次推理(加载模型)
await evalBatch([{ spatial: new Float32Array(22 * E.N2), global: new Float32Array(19) }]);
console.log('推理服务就绪');

const GAMES = Number(process.env.GAMES ?? 1);
for (let g = 0; g < GAMES; g++) {
  const r = await playGame();
  console.log(`局${g + 1}: ${r.plies} 手 · 目差 ${r.margin.toFixed(1)}(黑${r.blackWin ? '胜' : '负'})· 全局 ${r.ms}ms(NN 累计 ${r.nnEvalMs}ms)`);
}
console.log('✓ NN 自对弈完整终局');

proc.kill();
process.exit(0);
