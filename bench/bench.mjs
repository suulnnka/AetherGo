/* 基准:演棋速度 / 固定预算最佳着法回归。
 *
 *   node bench/bench.mjs nps      各档位在初始局面跑满预算,报演棋数与速度
 *   node bench/bench.mjs moves    固定演棋数下的最佳着法与胜率(改搜索后可对拍)
 *
 * 口径与 webos 里的一致:演棋局数为主、墙上时间为兜底,同一台机器上可复现。
 * 围棋没有 perft —— 规则正确性由 test/engine-test.mjs 的模糊测试与
 * make/unmake 往返不变量保证。
 */
import { newBoard, replayMoves, searchBest, LEVELS, moveToText } from '../src/engine.js';

const mode = process.argv[2] || 'nps';
const fmt = (n) => n.toLocaleString('en-US');
const sq = (r, c) => r * 9 + c;

/* 几个开局序列(黑先),用来把引擎拉出初始局面的对称区 */
const OPENINGS = {
  星位对角: [sq(2, 2), sq(6, 6), sq(2, 6), sq(6, 2)],
  小目守角: [sq(2, 2), sq(6, 6), sq(6, 2), sq(4, 4)],
};

function boardAfter(moves) {
  const bd = newBoard();
  replayMoves(bd, moves);
  return bd;
}

if (mode === 'nps') {
  console.log('档位        演棋预算    实际演棋      耗时     速度      胜率  最佳着法');
  for (const lv of LEVELS) {
    const bd = newBoard();
    const t = Date.now();
    const r = searchBest(bd, 0, { playouts: lv.playouts, ms: lv.ms });
    const ms = Math.max(Date.now() - t, 1);
    console.log(
      `${lv.name.padEnd(6)} ${String(fmt(lv.playouts)).padStart(10)} ${String(fmt(r.visits)).padStart(11)} ` +
      `${String(ms + 'ms').padStart(8)} ${String(fmt(Math.round(r.visits / ms * 1000)) + '/s').padStart(9)} ` +
      `${(r.winRate * 100).toFixed(0).padStart(5)}%  ${moveToText(bd, r.move)}`);
  }
} else if (mode === 'moves') {
  const playouts = Number(process.argv[3] || 5000);
  console.log(`固定演棋数 ${fmt(playouts)} 的最佳着法(改搜索/改演棋策略后用来对拍)\n`);
  for (const [name, moves] of [['初始局面', []], ...Object.entries(OPENINGS)]) {
    const bd = boardAfter(moves);
    const side = moves.length % 2;
    const r = searchBest(bd, side, { playouts, ms: 60000 });
    console.log(`  ${name.padEnd(6)} ${moveToText(bd, r.move).padEnd(6)} 胜率 ${(r.winRate * 100).toFixed(1)}%` +
                `  ${fmt(r.visits)} 演棋 / ${r.ms}ms(轮 ${side === 0 ? '黑' : '白'})`);
  }
} else {
  console.error('未知模式:' + mode + '(可用:nps / moves)');
  process.exit(1);
}
