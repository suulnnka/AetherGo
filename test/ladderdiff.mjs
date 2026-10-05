/* 征子通道逐链对拍:KataGo 原生探针(/tmp/probe,链的是真 board.cpp)vs JS 特征通道 14。
 * 用法:node test/ladderdiff.mjs /tmp/featdiff.json [最多打印数]
 * 探针输入:N2=361 个盘面值 + ko(探针须按 19 路编译);输出:每颗 1/2 气子一行 "p libs laddered"。 */
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { N, N2, newBoard, make } from '../src/engine.js';
import { encodeFeatures } from '../src/nn/features.js';

const file = process.argv[2] ?? '/tmp/featdiff20.json';
const maxPrint = Number(process.argv[3] ?? 6);
const data = JSON.parse(readFileSync(file, 'utf8'));

let pos = 0, bad = 0, printed = 0;
for (let gi = 0; gi < data.games.length; gi++) {
  const g = data.games[gi];
  const bd = newBoard();
  for (let i = 0; i < g.moves.length; i++) {
    const side = i % 2;
    const ko = (await import('../src/engine.js')).koPoint();
    const input = [...bd].join(' ') + ' ' + ko + '\n';
    const probe = spawnSync('/tmp/probe', { input });
    const refLines = probe.stdout.toString().trim().split('\n').filter(Boolean);
    const ref = new Map(refLines.map((l) => { const [p, , r] = l.split(' '); return [Number(p), Number(r)]; }));

    const r = encodeFeatures(bd, side, { recentMoves: g.moves.slice(0, i), komi: g.komi });
    for (const [p, want] of ref) {
      const got = r.spatial[14 * N2 + p];
      if (got !== want) {
        bad++;
        if (printed < maxPrint) {
          printed++;
          console.log(`分歧:局${gi} 行${i} 点${p}(${Math.floor(p / N)},${p % N}) JS=${got} 原生=${want}`);
          if (printed === 1) {
            let s = '';
            for (let y = 0; y < N; y++) { for (let x = 0; x < N; x++) s += '.XO'[bd[y * N + x]]; s += '\n'; }
            console.log('盘面(ko=' + ko + '):\n' + s);
          }
        }
      }
    }
    pos += ref.size;
    make(bd, g.moves[i], side);
  }
}
console.log(`共 ${pos} 条 1/2 气链判定,分歧 ${bad}`);
process.exit(bad ? 1 : 0);
