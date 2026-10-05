/* NN 对 NN 对战:两个 ONNX 模型各起一个 ort_server 子进程,等 visits 对弈。
 * 用法:node test/nn-match.mjs <modelA.onnx> <modelB.onnx> [局数] [visits]
 */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
/* 19 路对拍:两个模型必须是同一盘径(缺省都指向学生模型 b8c96h3tfrs_19,比较时显式传两个模型) */
const modelA = process.argv[2] ?? join(ROOT, 'models/b8c96h3tfrs_19.onnx');
const modelB = process.argv[3] ?? join(ROOT, 'models/b8c96h3tfrs_19.onnx');
const GAMES = Number(process.argv[4] ?? 4);
const VISITS = Number(process.argv[5] ?? 150);

const E = await import(join(ROOT, 'src/engine.js'));
const S = await import(join(ROOT, 'src/nn/search.js'));
const PY = '/home/a/miniconda3/envs/bleed/bin/python';

function makeSession(model) {
  const proc = spawn(PY, [join(ROOT, 'training/ort_server.py'), model], { stdio: ['pipe', 'pipe', 'inherit'] });
  let buf = '';
  const pending = [];
  proc.stdout.on('data', (d) => {
    buf += d.toString();
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
      if (line.trim()) pending.shift().resolve(JSON.parse(line));
    }
  });
  return {
    evalBatch(rows) {
      return new Promise((resolve, reject) => {
        pending.push({ resolve, reject });
        proc.stdin.write(JSON.stringify({
          rows: rows.map((r) => ({ spatial: Array.from(r.spatial), global: Array.from(r.global) })),
        }) + '\n');
      });
    },
    kill: () => proc.kill(),
  };
}

const sessA = makeSession(modelA);
const sessB = makeSession(modelB);
await sessA.evalBatch([{ spatial: new Float32Array(22 * E.N2), global: new Float32Array(19) }]);
await sessB.evalBatch([{ spatial: new Float32Array(22 * E.N2), global: new Float32Array(19) }]);
const nameA = modelA.split('/').pop(), nameB = modelB.split('/').pop();
console.log(`对战: ${nameA} vs ${nameB},${GAMES} 局,每手 ${VISITS} NN 访问,等算力`);

let winsA = 0, totalMoves = 0;
for (let g = 0; g < GAMES; g++) {
  const aIsBlack = g % 2 === 0;
  const bd = E.newBoard();
  const hist = [];
  let side = 0, lastPass = false, plies = 0, resigned = null;
  /* 开局 2 手随机(避免确定性对局,4 局=2 局的退化样本) */
  for (let t = 0; t < 760; t++) {
    if (t < 2) {
      const legal = E.genLegal(bd, side);
      const mv = legal[(Math.random() * legal.length) | 0];
      hist.push(mv); E.make(bd, mv, side); side ^= 1; plies++;
      lastPass = false;
      continue;
    }
    const isA = (side === 0) === aIsBlack;
    const r = await S.nnSearchBest(bd, side, {
      session: isA ? sessA : sessB, visits: VISITS, batch: 8, recentMoves: hist.slice(),
    });
    if (r.resign) { resigned = isA ? 'A' : 'B'; break; }
    if (r.move !== E.PASS && !E.isLegal(bd, side, r.move)) throw new Error(`非法着法 ${r.move} (${isA ? nameA : nameB})`);
    hist.push(r.move);
    const pass = r.move === E.PASS;
    if (pass && lastPass) break;
    lastPass = pass;
    E.make(bd, r.move, side);
    side ^= 1; plies++;
  }
  totalMoves += plies;
  let aWon;
  if (resigned === 'A') aWon = false;
  else if (resigned === 'B') aWon = true;
  else { const s = E.finalScore(bd); aWon = aIsBlack ? s.margin > 0 : s.margin < 0; }
  if (aWon) winsA++;
  const score = resigned ? (resigned === 'A' ? 'A 认输' : 'B 认输') : `目差 ${E.finalScore(bd).margin.toFixed(1)}`;
  console.log(`局${g + 1}: ${nameA} 执${aIsBlack ? '黑' : '白'} ${aWon ? '胜' : '负'} · ${plies} 手 · ${score}${resigned ? '(中盘)' : ''}`);
}
console.log(`\n结果: ${nameA} ${winsA}:${GAMES - winsA} ${nameB}`);
sessA.kill(); sessB.kill();
process.exit(0);
