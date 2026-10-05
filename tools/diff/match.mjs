/* AetherGo(src/ 同源码,推理=ORT CUDA 全契约服务)vs KataGo GTP 对战驱动 */
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { newBoard, replayMoves, PASS, BLACK } from "/home/a/go/AetherGo/src/engine.js";
import { nnSearchBest } from "/home/a/go/AetherGo/src/nn/search.js";

const VISITS = Number(process.env.VISITS ?? 80), GAMES = Number(process.env.GAMES ?? 8);
const COLS = "ABCDEFGHJKLMNOPQRST";
const toGtp = (mv) => mv === PASS ? "pass" : COLS[mv % 19] + (19 - ((mv / 19) | 0));
const fromGtp = (s0) => {
  const s = s0.trim().toLowerCase();
  if (s === "pass") return PASS;
  const c = "abcdefghjklmnopqrst".indexOf(s[0]);
  return (19 - parseInt(s.slice(1))) * 19 + c;
};

/* ---------- KataGo GTP ---------- */
const kata = spawn("/home/a/go/KataGo/build-cuda/katago", ["gtp",
  "-config", "/home/a/go/trainrun/gtp_b8c96.cfg", "-config", "ae_rules.cfg",
  "-model", "/home/a/go/trainrun/export_bin/b8c96h3tfrs-s68320512.bin.gz",
  "-override-config", "numSearchThreads=1,maxVisits=" + VISITS
    + ",ponderingEnabled=false,logAllGTPCommunication=false,logDir=/tmp/kbench/logs"
    + ",searchFactorWhenWinning=1.0,searchFactorAfterOnePass=1.0,searchFactorAfterTwoPass=1.0",
], { stdio: ["pipe", "pipe", "inherit"] });
kata.stdout._buf = ""; kata.stdout._wait = null;
kata.stdout.on("data", (ch) => {
  kata.stdout._buf += ch;
  let i;
  while ((i = kata.stdout._buf.indexOf("\n")) >= 0) {
    const l = kata.stdout._buf.slice(0, i).replace(/\r$/, "");
    kata.stdout._buf = kata.stdout._buf.slice(i + 1);
    if (kata.stdout._wait) {
      if (l === "") { const w = kata.stdout._wait; kata.stdout._wait = null; w.res(kata.stdout._acc); }
      else kata.stdout._acc += (kata.stdout._acc ? "\n" : "") + l;
    }
  }
});
const kataCmd = (cmd) => new Promise((res, rej) => {
  kata.stdout._acc = ""; kata.stdout._wait = { res, rej };
  kata.stdin.write(cmd + "\n");
  setTimeout(() => rej(new Error("gtp-timeout: " + cmd)), 120000);
});

/* ---------- 推理服务 ---------- */
const srv = spawn("/home/a/miniconda3/envs/bleed/bin/python",
  ["/tmp/kbench/full_ort_server.py", "/home/a/go/AetherGo/models/b8c96h3tfrs_19.onnx"],
  { stdio: ["pipe", "pipe", "inherit"] });
srv.stdout._buf = ""; srv.stdout._wait = null; srv.stdout._lines = [];
srv.stdout.on("data", (ch) => {
  srv.stdout._buf += ch;
  let i;
  while ((i = srv.stdout._buf.indexOf("\n")) >= 0) {
    const l = srv.stdout._buf.slice(0, i);
    srv.stdout._buf = srv.stdout._buf.slice(i + 1);
    if (srv.stdout._wait) { const w = srv.stdout._wait; srv.stdout._wait = null; w.res(JSON.parse(l)); }
  }
});
const evalBatch = async (rows) => {
  const req = { rows: rows.map((r) => ({
    spatial: Array.from(r.spatial), global: Array.from(r.global), optimism: r.optimism ?? 1.0 })) };
  return new Promise((res, rej) => {
    srv.stdout._wait = { res, rej };
    srv.stdin.write(JSON.stringify(req) + "\n");
    setTimeout(() => rej(new Error("inf-timeout")), 120000);
  });
};
const session = { maxBatch: 16, evalBatch };

/* ---------- 对局 ---------- */
await kataCmd("boardsize 19"); await kataCmd("komi 7.5");
const results = [];
for (let g = 0; g < GAMES; g++) {
  const aetherBlack = g % 2 === 0;
  await kataCmd("clear_board");
  const moves = [];
  let resignHist = [], resignLastLen = -1;
  let passes = 0, result = null, aMs = 0, kMs = 0;
  for (let ply = 0; ply < 560 && !result; ply++) {
    const blackToMove = moves.length % 2 === 0;
    const aetherTurn = blackToMove === aetherBlack;
    let mv;
    if (aetherTurn) {
      const bd = newBoard(); const stm = replayMoves(bd, moves);
      if (stm < 0) throw new Error("replay failed at moves=" + JSON.stringify(moves));
      const t0 = Date.now();
      const r = await nnSearchBest(bd, stm, { session, visits: VISITS, maxBatch: session.maxBatch, recentMoves: moves.slice() });
      aMs += Date.now() - t0;
      if (moves.length < resignLastLen) resignHist = [];
      resignLastLen = moves.length;
      const wlStm = r.winRate * 2 - 1;
      resignHist.push(stm === BLACK ? -wlStm : wlStm);
      if (resignHist.length > 3) resignHist.shift();
      const losing = stm === 1 ? ((v) => v < -0.90) : ((v) => v > 0.90);
      if (moves.length >= 73 && resignHist.length >= 3 && resignHist.every(losing)) {
        result = { winner: aetherBlack ? "W" : "B", byResign: true, scoreText: "resign" }; break;
      }
      mv = r.move;
      await kataCmd("play " + (blackToMove ? "b" : "w") + " " + toGtp(mv));
    } else {
      const t0 = Date.now();
      const reply = await kataCmd("genmove " + (blackToMove ? "b" : "w"));
      kMs += Date.now() - t0;
      const s = reply.trim().replace(/^=\s*/, "").toLowerCase();
      if (s === "resign") { result = { winner: aetherBlack ? "B" : "W", byResign: true, scoreText: "resign" }; break; }
      mv = fromGtp(s);
      if (!Number.isInteger(mv) || mv < 0 || mv > 361) throw new Error("bad kata move: " + JSON.stringify(reply));
    }
    moves.push(mv);
    if (mv === PASS) { if (++passes >= 2) break; } else passes = 0;
  }
  if (!result) {
    const fs = (await kataCmd("final_score")).trim().replace(/^=\s*/, "");
    const m = fs.match(/^([BW])\+(R|[0-9.]+)/);
    result = m ? { winner: m[1], byResign: m[2] === "R", margin: m[2] === "R" ? null : parseFloat(m[2]), scoreText: fs }
               : { winner: "D", scoreText: fs };
  }
  const sgf = "(;KM[7.5]PB[" + (aetherBlack ? "AetherGo" : "KataGo") + "]PW[" + (aetherBlack ? "KataGo" : "AetherGo") + "]RE[" + result.scoreText.replace(/^=\s*/, "") + "]" + moves.map((mv, i) =>
    ";" + (i % 2 === 0 ? "B" : "W") + "[" + (mv === PASS ? "" : String.fromCharCode(97 + (mv % 19)) + String.fromCharCode(97 + ((mv / 19) | 0))) + "]").join("") + ")";
  writeFileSync("/tmp/kbench/game" + (g + 1) + ".sgf", sgf);
  const rec = { game: g + 1, aetherColor: aetherBlack ? "B" : "W", plies: moves.length,
    result: result.scoreText.replace(/^=\s*/, ""), aetherWon: (result.winner === "B") === aetherBlack,
    aMs, kMs, opening: moves.slice(0, 24).map(toGtp).join(" ") };
  results.push(rec);
  console.log(JSON.stringify(rec));
}
const aw = results.filter((r) => r.aetherWon).length;
console.log("SUMMARY aether " + aw + " - " + (GAMES - aw) + " kataGo  (" + VISITS + " visits 双方)");
kata.stdin.write("quit\n"); srv.stdin.write("\n");
process.exit(0);
