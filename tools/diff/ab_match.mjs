/* 旧引擎(/tmp/aether_head3)vs 新引擎(~/go/AetherGo)A/B 对战 */
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";

const VISITS = Number(process.env.VISITS ?? 200), GAMES = Number(process.env.GAMES ?? 6);
const oldE = await import("/tmp/aether_head3/src/engine.js");
const newE = await import("/home/a/go/AetherGo/src/engine.js");
const oldS = await import("/tmp/aether_head3/src/nn/search.js");
const newS = await import("/home/a/go/AetherGo/src/nn/search.js");
const PASS = newE.PASS, BLACK = newE.BLACK;

/* 共享推理服务(串行互斥) */
const srv = spawn("/home/a/miniconda3/envs/bleed/bin/python",
  ["/tmp/kbench/full_ort_server.py", "/home/a/go/AetherGo/models/b8c96h3tfrs_19.onnx"],
  { stdio: ["pipe", "pipe", "ignore"] });
srv._buf = ""; srv._wait = null; srv._lock = Promise.resolve();
srv.stdout.on("data", (ch) => {
  srv._buf += ch;
  let i;
  while ((i = srv._buf.indexOf("\n")) >= 0) {
    const l = srv._buf.slice(0, i); srv._buf = srv._buf.slice(i + 1);
    if (srv._wait) { const w = srv._wait; srv._wait = null; w.res(JSON.parse(l)); }
  }
});
const evalBatchRaw = (rows) => new Promise((res, rej) => {
  srv._wait = { res, rej };
  srv.stdin.write(JSON.stringify({ rows: rows.map((r) => ({
    spatial: Array.from(r.spatial), global: Array.from(r.global), sym: r.sym ?? 0, optimism: r.optimism ?? 1.0 })) }) + "\n");
  setTimeout(() => rej(new Error("inf-timeout")), 240000);
});
const evalBatch = (rows) => srv._lock.then(() => {
  const p = evalBatchRaw(rows);
  srv._lock = p.catch(() => {});
  return p;
});
const session = { maxBatch: 16, evalBatch };

const toTxt = (E, mv) => mv === E.PASS ? "pass" : "ABCDEFGHJKLMNOPQRST"[mv % 19] + (19 - ((mv / 19) | 0));

async function think(side, bd, moves) {
  const r = await (side === "new"
    ? newS.nnSearchBest(bd, moves.length % 2, { session, visits: VISITS, maxBatch: 16, recentMoves: moves.slice() })
    : oldS.nnSearchBest(bd, moves.length % 2, { session, visits: VISITS, maxBatch: 16, recentMoves: moves.slice() }));
  return r;
}

const results = [];
for (let g = 0; g < GAMES; g++) {
  const newBlack = g % 2 === 0;
  const moves = [];
  let resign = { new: [], old: [] }, passes = 0, result = null, nMs = 0, oMs = 0;
  for (let ply = 0; ply < 560 && !result; ply++) {
    const blackToMove = moves.length % 2 === 0;
    const mover = (blackToMove === newBlack) ? "new" : "old";
    const E = mover === "new" ? newE : oldE;
    const bd = E.newBoard();
    if (E.replayMoves(bd, moves) < 0) throw new Error("replay failed");
    const t0 = Date.now();
    const r = await think(mover, bd, moves);
    (mover === "new" ? nMs += Date.now() - t0 : oMs += Date.now() - t0);
    /* 认输(worker 同款;每引擎独立历史) */
    const h = resign[mover];
    if (moves.length < h.lastLen) h.hist = [];
    h.lastLen = moves.length;
    const wlStm = r.winRate * 2 - 1;
    const stmBlack = blackToMove;
    h.hist = (h.hist ?? []).concat(stmBlack ? -wlStm : wlStm).slice(-3);
    const losing = stmBlack ? ((v) => v > 0.90) : ((v) => v < -0.90);
    if (moves.length >= 73 && h.hist.length >= 3 && h.hist.every(losing)) {
      result = { winner: mover === "new" ? "old" : "new", byResign: true }; break;
    }
    moves.push(r.move);
    if (r.move === PASS) { if (++passes >= 2) break; } else passes = 0;
  }
  let scoreText;
  if (result) scoreText = (result.winner === "new" ? "NEW+" : "OLD+") + (result.byResign ? "R" : "");
  else {
    const bd = newE.newBoard();
    newE.replayMoves(bd, moves);
    const f = newE.finalScore(bd);
    scoreText = f.margin > 0 ? "B+" + f.margin.toFixed(1) : "W+" + (-f.margin).toFixed(1);
    result = { winner: ((f.margin > 0) === newBlack) ? "new" : "old" };
  }
  results.push(result.winner === "new");
  console.log(JSON.stringify({ game: g + 1, newColor: newBlack ? "B" : "W", plies: moves.length,
    result: scoreText, newWon: result.winner === "new", nMs, oMs }));
  const sgf = "(;KM[7.5]PB[" + (newBlack ? "NEW" : "OLD") + "]PW[" + (newBlack ? "OLD" : "NEW") + "]RE[" + scoreText + "]" +
    moves.map((mv, i) => ";" + (i % 2 === 0 ? "B" : "W") + "[" + (mv === PASS ? "" : String.fromCharCode(97 + mv % 19) + String.fromCharCode(97 + ((mv / 19) | 0))) + "]").join("") + ")";
  writeFileSync("/tmp/kbench/ab_game" + (g + 1) + ".sgf", sgf);
}
const nw = results.filter(Boolean).length;
console.log("SUMMARY new " + nw + " - " + (GAMES - nw) + " old  (" + VISITS + " visits)");
srv.stdin.write("\n");
process.exit(0);
