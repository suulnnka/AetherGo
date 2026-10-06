/* KataGo 1T vs 8T 等visits内战(同fp16只切线程数) —— 标定多线程批税是否引擎普遍。
 * 2026-10-06 实测 64v×12:1T 4-8 8T —— KataGo 多线程在等visits下是增益(vl扩探→更准根值),
 * 「多线程必降访问效率」不成立;我方批税的根源是值动力学(领先子q粘性乐观,
 * runner-up的sv反超窗口不开),见 AetherGo 主仓提交记录。 */
import { spawn } from "node:child_process";
const VISITS = Number(process.env.VISITS ?? 64), GAMES = Number(process.env.GAMES ?? 12);
function mkKata(threads) {
  const k = spawn("/home/a/go/KataGo/build-cuda/katago", ["gtp",
    "-config", "/home/a/go/trainrun/gtp_b8c96.cfg",
    "-model", "/home/a/go/trainrun/export_bin/b8c96h3tfrs-s68320512.bin.gz",
    "-override-config", "numSearchThreads=" + threads + ",maxVisits=" + VISITS
      + ",ponderingEnabled=false,logAllGTPCommunication=false,logDir=/tmp/kbench/logs"
      + ",searchFactorWhenWinning=1.0,searchFactorAfterOnePass=1.0,searchFactorAfterTwoPass=1.0",
  ], { stdio: ["pipe", "pipe", "ignore"] });
  k._buf = ""; k._wait = null;
  k.stdout.on("data", (ch) => {
    k._buf += ch;
    let i;
    while ((i = k._buf.indexOf("\n")) >= 0) {
      const l = k._buf.slice(0, i).replace(/\r$/, "");
      k._buf = k._buf.slice(i + 1);
      if (k._wait) { if (l === "") { const w = k._wait; k._wait = null; w.res(k._acc); } else k._acc += (k._acc ? "\n" : "") + l; }
    }
  });
  k.cmd = (c) => new Promise((res, rej) => {
    k._acc = ""; k._wait = { res, rej };
    k.stdin.write(c + "\n");
    setTimeout(() => rej(new Error("timeout " + c)), 180000);
  });
  return k;
}
const A = mkKata(1), B = mkKata(8);
for (const k of [A, B]) { await k.cmd("boardsize 19"); await k.cmd("komi 7.5"); }
let awins = 0;
for (let g = 0; g < GAMES; g++) {
  const aBlack = g % 2 === 0;
  for (const k of [A, B]) await k.cmd("clear_board");
  let passes = 0, result = null, plies = 0;
  for (let ply = 0; ply < 560 && !result; ply++) {
    const blackToMove = plies % 2 === 0;
    const mover = (blackToMove === aBlack) ? A : B;
    const r = (await mover.cmd("genmove " + (blackToMove ? "b" : "w"))).trim().replace(/^=\s*/, "").toLowerCase();
    plies++;
    const other = mover === A ? B : A;
    await other.cmd("play " + (blackToMove ? "b" : "w") + " " + r);
    if (r === "pass") { if (++passes >= 2) result = "pass2"; } else passes = 0;
    if (r === "resign") result = (mover === A ? "B" : "A") + " resigns";
  }
  if (result === "pass2") {
    const sc = (await A.cmd("final_score")).trim().replace(/^=\s*/, "");
    result = sc;
    if ((sc.includes("W+") && !aBlack) || (sc.includes("B+") && aBlack)) awins++;
  } else if (result === "A resigns") { /* B wins */ }
  else if (result === "B resigns") awins++;
  console.log(JSON.stringify({ game: g + 1, aBlack, plies, result }));
}
console.log("SUMMARY 1T " + awins + " - " + (GAMES - awins) + " 8T  (" + VISITS + " visits)");
for (const k of [A, B]) await k.cmd("quit");
