/* 特征对拍(全路线质量的锚):JS 编码器 vs katago selfplay 训练行,逐位比对。
 *
 * 用法:
 *   1) 用 KataGo 产出对拍数据(一行=一手的干净配置):
 *      katago selfplay -models-dir ... -config training/selfplay_featdiff.cfg \
 *        -output-dir <data> -max-games-total 2
 *   2) python3 training/extract_rows.py <sgf目录> <npz目录> /tmp/featdiff.json
 *   3) node test/featdiff.mjs /tmp/featdiff.json
 *
 * 通过标准:22×361 空间通道逐位一致;19 全局通道 |差| < 1e-4。
 * (19 路口径;9 路对拍数据需要对位改回 pos-len 9 才能用)
 */
import { N2, newBoard, make, replayMoves, BLACK, PASS, KOMI } from '../src/engine.js';
import { encodeFeatures, SPATIAL_CHANNELS, GLOBAL_CHANNELS } from '../src/nn/features.js';

const file = process.argv[2] ?? '/tmp/featdiff.json';
const data = JSON.parse(await (await import('node:fs')).readFileSync(file, 'utf8'));

let totalRows = 0, spatialBad = 0, globalBad = 0;
const chBad = new Array(SPATIAL_CHANNELS).fill(0);
const gBad = new Array(GLOBAL_CHANNELS).fill(0);
const samples = [];

outer:
for (const game of data.games) {
  const bd = newBoard();
  const moves = game.moves;
  const rows = game.rows;
  if (rows !== moves.length) {
    console.log(`! 行数 ${rows} != 手数 ${moves.length} —— 数据不是「一行一手」口径或 SGF/npz 配对错了`);
    if (Math.abs(rows - moves.length) > 2) continue outer;
  }
  for (let i = 0; i < Math.min(rows, moves.length); i++) {
    const side = i % 2 === 0 ? BLACK : 1 - BLACK;
    const r = encodeFeatures(bd, side, {
      recentMoves: moves.slice(0, i),
      komi: game.komi,
    });
    const wantSp = game.spatial[i], wantGl = game.global[i];
    let bad = false;
    for (let k = 0; k < 22 * N2; k++) {
      if (r.spatial[k] !== wantSp[k]) {
        spatialBad++; bad = true;
        chBad[Math.floor(k / N2)]++;
        if (samples.length < 12) samples.push({ game: game.moves === moves ? 0 : 0, i, k, ch: Math.floor(k / N2), p: k % N2, got: r.spatial[k], want: wantSp[k] });
        break;   // 每行只记首个错,避免刷屏;通道统计以行计
      }
    }
    for (let k = 0; k < 19; k++) {
      if (Math.abs(r.global[k] - wantGl[k]) > 1e-4) {
        globalBad++; bad = true;
        gBad[k]++;
        if (samples.length < 12) samples.push({ i, g: k, got: r.global[k], want: wantGl[k] });
        break;
      }
    }
    if (!bad) totalRows++;
    else if (samples.length < 12) samples[samples.length - 1] && (samples[samples.length - 1].move = moves[i]);
    // 走子(重演由引擎 make 维护 superko 历史,供通道 6 判定)
    make(bd, moves[i], side);
  }
}

const nRows = data.games.reduce((a, g) => a + Math.min(g.rows, g.moves.length), 0);
console.log(`对拍 ${data.games.length} 局 / ${nRows} 行:空间全对 ${totalRows} 行,空间错 ${spatialBad} 行,全局错 ${globalBad} 行`);
if (spatialBad || globalBad) {
  console.log('空间通道错误行分布(通道:行数):', chBad.map((n, c) => n ? `${c}:${n}` : '').filter(Boolean).join(' '));
  console.log('全局通道错误行分布:', gBad.map((n, c) => n ? `${c}:${n}` : '').filter(Boolean).join(' '));
  console.log('样例:', JSON.stringify(samples.slice(0, 8)));
  process.exit(1);
}
console.log('✓ 22+19 通道逐位一致');
