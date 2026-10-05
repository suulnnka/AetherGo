/* 局面特征导出器(纯 PyTorch 蒸馏的数据侧):
 * 重放自对弈 SGF,用已对拍验证的 JS 编码器(src/nn/features.js)逐局面编码,
 * 流式落成裸二进制给 python 侧蒸馏用。
 *
 * 用法:node training/dump_positions.mjs <sgfs目录|文件...> <输出.bin> [最大局面数]
 * 格式:16 字节头(魔数 'AEPOS001' + u32 局面数 + u32 行 f32 数)+
 *       每局面 22*81 个 f32(空间)+ 19 个 f32(全局)+ 1 个 u8(行棋方,4 字节对齐)
 * 数据卫生:SGF 每手注释里 weight=0.00 的行(cheap search)跳过。
 */
import { createWriteStream } from 'node:fs';
import { readFileSync, readdirSync, statSync, openSync, writeSync, closeSync } from 'node:fs';
import { join } from 'node:path';
import { newBoard, make, BLACK, PASS, KOMI } from '../src/engine.js';
import { encodeFeatures } from '../src/nn/features.js';

const args = process.argv.slice(2);
if (args.length < 2) {
  console.error('用法:node training/dump_positions.mjs <sgfs目录|文件...> <输出.bin> [最大局面数]');
  process.exit(1);
}
const outPath = args[args.length - 1];
const maybeMax = args[args.length - 2];
const hasMax = /^\d+$/.test(maybeMax ?? '');
const maxPos = hasMax ? Number(maybeMax) : 1e9;
const inputs = hasMax ? args.slice(0, -2) : args.slice(0, -1);

const games = [];
function pushGames(text) {
  /* SGF 可能跨多行(注释换行)—— 整文件按「(;」分割,不按行 */
  for (const g of text.split(/(?=\(;)/)) {
    if (g.includes('GM[1]')) games.push(g.trim());
  }
}
for (const p of inputs) {
  const st = statSync(p);
  if (st.isDirectory()) {
    for (const f of readdirSync(p).sort()) {
      if (f.endsWith('.sgfs') || f.endsWith('.sgf')) {
        pushGames(readFileSync(join(p, f), 'utf8'));
      }
    }
  } else {
    pushGames(readFileSync(p, 'utf8'));
  }
}
console.log(`读入 ${games.length} 局`);

function parseGame(text) {
  const komiM = text.match(/KM\[([^\]]*)\]/);
  const komi = komiM ? parseFloat(komiM[1]) : KOMI;
  const moves = [];
  const re = /;([BW])\[([a-zA-Z]{0,2})\](?:C\[([^\]]*)\])?/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const coord = m[2];
    const mv = (coord === '' || coord === 'tt')
      ? PASS : (coord.charCodeAt(1) - 97) * 9 + (coord.charCodeAt(0) - 97);
    let weight = 1;
    if (m[3]) {
      const w = m[3].match(/weight=([0-9.]+)/);
      if (w) weight = parseFloat(w[1]);
    }
    moves.push({ mv, weight });
  }
  return { komi, moves };
}

const ROW_F32 = 22 * 81 + 19;            // 每行 f32 数(不含 side)
const fd = openSync(outPath, 'w');
/* 头:占位,收尾回填真实局面数 */
const head = Buffer.alloc(16);
head.write('AEPOS001', 0, 'ascii');
head.writeUInt32LE(0, 8);
head.writeUInt32LE(ROW_F32 + 1, 12);
writeSync(fd, head, 0, 16);

const rowBuf = Buffer.alloc((ROW_F32 + 1) * 4);   // f32×1802 + 尾部 side
let n = 0, skipped = 0;
for (const g of games) {
  if (n >= maxPos) break;
  const { komi, moves } = parseGame(g);
  const bd = newBoard();
  const hist = [];
  for (let i = 0; i < moves.length && n < maxPos; i++) {
    const side = i % 2 === 0 ? BLACK : 1 - BLACK;
    if (moves[i].weight > 0) {
      const r = encodeFeatures(bd, side, { recentMoves: hist.slice(), komi });
      /* 分段写:空间 22*81 + 全局 19 */
      for (let k = 0; k < 22 * 81; k++) rowBuf.writeFloatLE(r.spatial[k], k * 4);
      for (let k = 0; k < 19; k++) rowBuf.writeFloatLE(r.global[k], (22 * 81 + k) * 4);
      rowBuf.writeUInt8(side, ROW_F32 * 4);
      writeSync(fd, rowBuf, 0, rowBuf.length);
      n++;
    } else skipped++;
    make(bd, moves[i].mv, side);
    hist.push(moves[i].mv);
  }
}
closeSync(fd);
/* 回填局面数 */
const fd2 = openSync(outPath, 'r+');
const cnt = Buffer.alloc(4);
cnt.writeUInt32LE(n, 0);
writeSync(fd2, cnt, 0, 4, 8);
closeSync(fd2);
console.log(`导出 ${n} 局面(跳过 weight=0 行 ${skipped})→ ${outPath}(${(16 + n * rowBuf.length) / 1e6 | 0}MB)`);
