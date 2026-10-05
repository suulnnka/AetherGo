/* 特征位级对拍:我方 encodeFeatures vs KataGo 搜索路径实录(KATA_NNLOG)。
 *
 * 用法(工作目录 = 仓库根,WSL):
 *   1) 用 KataGo 抓实录(见 docs/DIFF_WORKFLOW.md §特征对拍):
 *      cd ~/go/trainrun && { printf "boardsize 19\nkomi 7.5\nclear_board\n";
 *        cat /tmp/kbench/plays.txt;   # 逐行 play b/w <gtp> 的着法序列
 *        echo "genmove b"; sleep 6; echo "quit"; } |
 *        KATA_NNLOG=/tmp/kbench/nnlog.bin KATA_NNSERVER=127.0.0.1:9911 \
 *        LD_LIBRARY_PATH=$ORT_CPU_LIB timeout 90 ~/go/KataGo/build-onnx/katago gtp \
 *        -config gtp_b8c96.cfg -config ae_rules.cfg \
 *        -model ~/go/AetherGo/models/b8c96h3tfrs_19.onnx \
 *        -override-config "numSearchThreads=1,maxVisits=8,ponderingEnabled=false,logDir=/tmp/kbench/logs,onnxProvider=cpu" >/dev/null 2>&1
 *   2) node tools/diff/feat_bitcmp.mjs <moves的JS数组文件> /tmp/kbench/nnlog.bin [行号=0]
 *      moves 文件内容形如 [72,288,...,361](361=PASS,行棋方由重演决定)
 *
 * 说明:搜索路径带随机对称变换与 passing hacks —— 本脚本自动在 8 个对称中
 * 匹配棋盘通道后整体反变换比对,并给出逐通道差异;全局通道直接比对。
 * ★ 不要用 kata-raw-nn 抓实录:它不设 enablePassingHacks(selfplay 口径),
 *    带停着的局面会给出错误对照(2026-10-05 实测踩坑)。
 */
import { newBoard, replayMoves, PASS } from "../../src/engine.js";
import { encodeFeatures } from "../../src/nn/features.js";
import { readFileSync } from "node:fs";

const [movesPath, logPath, rowIdxArg] = process.argv.slice(2);
if (!movesPath || !logPath) {
  console.error("用法: node tools/diff/feat_bitcmp.mjs <moves.js数组文件> <nnlog.bin> [行号=0]");
  process.exit(1);
}
const rowIdx = Number(rowIdxArg ?? 0);

const moves = JSON.parse(readFileSync(movesPath, "utf8"));
const bd = newBoard();
const stm = replayMoves(bd, moves);
if (stm < 0) { console.error("着法序列重演失败(非法)"); process.exit(1); }
const r = encodeFeatures(bd, stm, { recentMoves: moves.slice(), komi: 7.5 });

/* ---- 解析 AENNL001 日志行 ---- */
const data = readFileSync(logPath);
if (data.subarray(0, 8).toString("latin1") !== "AENNL001") { console.error("非 AENNL001 日志"); process.exit(1); }
const spN = data.readUInt32LE(8), glN = data.readUInt32LE(12);
const ROW = spN + 4 * glN;
const row = data.subarray(40 + rowIdx * ROW, 40 + (rowIdx + 1) * ROW);
if (row.length < ROW) { console.error(`行号 ${rowIdx} 超界`); process.exit(1); }
const kspat = [];
for (let i = 0; i < spN; i++) kspat.push(row[i]);
const kglob = [];
for (let i = 0; i < glN; i++) kglob.push(row.readFloatLE(spN + 4 * i));

/* ---- 8 对称 ---- */
const N = 19, N2 = 361;
const SYM = [];
for (let s = 0; s < 8; s++) {
  const m = new Int32Array(N2);
  for (let rr = 0; rr < N; rr++) for (let cc = 0; cc < N; cc++) {
    let y = rr, x = cc;
    for (let k = 0; k < (s & 3); k++) { const t = y; y = x; x = N - 1 - t; }
    if (s & 4) x = N - 1 - x;
    m[rr * N + cc] = y * N + x;
  }
  SYM.push(m);
}
const ours = Array.from(r.spatial).map((x) => (x >= 0.5 ? 1 : 0));

let found = -1;
for (let s = 0; s < 8; s++) {
  const m = SYM[s];
  let ok = true;
  for (let p = 0; p < N2 && ok; p++) {
    if (ours[1 * N2 + p] !== kspat[1 * N2 + m[p]] || ours[2 * N2 + p] !== kspat[2 * N2 + m[p]]) ok = false;
  }
  if (ok) { found = s; break; }
}
if (found < 0) { console.error("棋盘通道在任何对称下都不匹配 —— 局面或行号不对"); process.exit(1); }

const m = SYM[found];
let diffs = 0;
const chs = new Set();
for (let ch = 0; ch < 22; ch++) {
  for (let p = 0; p < N2; p++) {
    if (ours[ch * N2 + p] !== kspat[ch * N2 + m[p]]) { diffs++; chs.add(ch); }
  }
}
const gd = [];
for (let i = 0; i < 19; i++) {
  if (Math.abs(r.global[i] - kglob[i]) > 1e-6) gd.push(`${i}:${r.global[i].toFixed(3)}/k${kglob[i].toFixed(3)}`);
}
console.log(`行${rowIdx} 对称=${found} stm=${stm === 0 ? "B" : "W"}`);
console.log(`spatial diffs=${diffs}${diffs ? " ch=" + [...chs].join(",") : "(位级一致)"}`);
console.log(`global diffs=${gd.length ? gd.join(" | ") : "none"}`);
console.log(`gl0(尾停标志)=${r.global[0].toFixed(0)}/k${kglob[0].toFixed(0)}  gl14(passWouldEnd)=${r.global[14].toFixed(0)}/k${kglob[14].toFixed(0)}`);
