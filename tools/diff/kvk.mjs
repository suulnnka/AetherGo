/* KataGo fp16 vs KataGo fp32 对打(同引擎同模型,只切推理精度) */
import { spawn } from "node:child_process";

const VISITS = Number(process.env.VISITS ?? 200), GAMES = Number(process.env.GAMES ?? 4);
const COLS = "ABCDEFGHJKLMNOPQRST";
const toGtp = (mv) => mv === 361 ? "pass" : COLS[mv % 19] + (19 - ((mv / 19) | 0));

function mkKata(tag, extra) {
  const k = spawn("/home/a/go/KataGo/build-cuda/katago", ["gtp",
    "-config", "/home/a/go/trainrun/gtp_b8c96.cfg",
    "-model", "/home/a/go/trainrun/export_bin/b8c96h3tfrs-s68320512.bin.gz",
    "-override-config", "numSearchThreads=1,maxVisits=" + VISITS
      + ",ponderingEnabled=false,logAllGTPCommunication=false,logDir=/tmp/kbench/logs"
      + ",searchFactorWhenWinning=1.0,searchFactorAfterOnePass=1.0,searchFactorAfterTwoPass=1.0" + extra,
  ], { stdio: ["pipe", "pipe", "ignore"] });
  k._buf = ""; k._wait = null;
  k.stdout.on("data", (ch) => {
    k._buf += ch;
    let i;
    while ((i = k._buf.indexOf("\n")) >= 0) {
      const l = k._buf.slice(0, i).replace(/\r$/, "");
      k._buf = k._buf.slice(i + 1);
      if (k._wait) {
        if (l === "") { const w = k._wait; k._wait = null; w.res(k._acc); }
        else k._acc += (k._acc ? "\n" : "") + l;
      }
    }
  });
  k.cmd = (c) => new Promise((res, rej) => {
    k._acc = ""; k._wait = { res, rej };
    k.stdin.write(c + "\n");
    setTimeout(() => rej(new Error("timeout " + c)), 180000);
  });
  return k;
}
const A = mkKata("fp16", "");                       // auto → fp16
const B = mkKata("fp32", ",cudaUseFP16=false");
for (const k of [A, B]) { await k.cmd("boardsize 19"); await k.cmd("komi 7.5"); }

let awins = 0;
for (let g = 0; g < GAMES; g++) {
  const aBlack = g % 2 === 0;
  for (const k of [A, B]) await k.cmd("clear_board");
  let passes = 0, result = null, plies = 0;
  for (let ply = 0; ply < 560 && !result; ply++) {
    const blackToMove = plies % 2 === 0;
    const mover = (blackToMove === aBlack) ? A : B;
    const other = mover === A ? B : A;
    const r = (await mover.cmd("genmove " + (blackToMove ? "b" : "w"))).trim().replace(/^=\s*/, "").toLowerCase();
    plies++;
    if (r === "resign") { result = (mover === A) ? "L-fp16" : "W-fp16"; break; }
    // 同步到对方棋盘(genmove 已在己方落子)
    await other.cmd("play " + (blackToMove ? "b" : "w") + " " + r);
    if (r === "pass") { if (++passes >= 2) break; } else passes = 0;
  }
  let scoreText;
  if (result) scoreText = result;
  else {
    scoreText = (await A.cmd("final_score")).trim().replace(/^=\s*/, "");
    result = scoreText;
  }
  const m = scoreText.match(/^([BW])\+(R|[0-9.]+)/);
  const winnerB = m ? m[1] === "B" : null;
  const aWon = winnerB !== null && (winnerB === aBlack);
  if (aWon) awins++;
  console.log(JSON.stringify({ game: g + 1, fp16Color: aBlack ? "B" : "W", plies, result: scoreText, fp16Won: aWon }));
}
console.log("SUMMARY fp16 " + awins + " - " + (GAMES - awins) + " fp32  (" + VISITS + " visits)");
for (const k of [A, B]) k.stdin.write("quit\n");
process.exit(0);
