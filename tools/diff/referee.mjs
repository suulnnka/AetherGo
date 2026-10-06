/* 每手质量裁判:真实局面取样,批1/批4 各选点(64v),oracle 直接评估选点后局面,
 * 比较两口径的即时局面质量(mover 视角 wl/scoreLead)。oracle 客户端 = diff_test 原版。 */
import net from "node:net";
import fs from "node:fs";
import { newBoard, replayMoves, KOMI, make, BLACK, WHITE } from "/home/a/go/AetherGo/src/engine.js";
import { nnSearchBest } from "/home/a/go/AetherGo/src/nn/search.js";
import { encodeFeatures } from "/home/a/go/AetherGo/src/nn/features.js";

const VISITS = 64;
const COLS = "ABCDEFGHJKLMNOPQRST";
const toGtp = (mv) => mv === 361 ? "pass" : COLS[mv % 19] + (19 - ((mv / 19) | 0));

let sock = null, rbuf = Buffer.alloc(0), waiters = [];
function connect() {
  return new Promise((res, rej) => {
    sock = net.connect(9911, "127.0.0.1", res);
    sock.on("error", (e) => rej(e));
    sock.on("data", (ch) => {
      rbuf = Buffer.concat([rbuf, ch]);
      while (true) {
        if (rbuf.length < 24) return;
        const cnt = [];
        for (let k = 0; k < 5; k++) cnt.push(rbuf.readUInt32LE(4 + 4 * k));
        const need = 24 + 4 * cnt.reduce((a, b) => a + b, 0);
        if (rbuf.length < need) return;
        const body = rbuf.subarray(24, need);
        rbuf = rbuf.subarray(need);
        const w = waiters.shift();
        if (w) {
          const arrs = []; let off = 0;
          for (const c of cnt) { arrs.push(new Float32Array(body.buffer, body.byteOffset + off, c)); off += 4 * c; }
          w(arrs);
        }
      }
    });
  });
}
function askOracle(spatial, global) {
  return new Promise((res, rej) => {
    waiters.push(res);
    const head = Buffer.alloc(4);
    head.writeUInt32LE(0x314F4741);
    const parts = [head];
    for (const arr of [spatial, global]) {
      const n = Buffer.alloc(4); n.writeUInt32LE(arr.length); parts.push(n);
      parts.push(Buffer.from(arr.buffer, arr.byteOffset, 4 * arr.length));
    }
    sock.write(Buffer.concat(parts));
    setTimeout(() => rej(new Error("oracle-timeout")), 120000);
  });
}
const softPlus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));
const session = {
  async evalBatch(rows) {
    const out = [];
    for (const r of rows) {
      const [pas, pol, val, sv, own] = await askOracle(r.spatial, r.global);
      const lam = r.optimism ?? 1.0;
      const polC = pol.length / 361;
      const policy = new Float32Array(361);
      for (let p = 0; p < 361; p++) {
        const p0 = pol[p];
        policy[p] = (polC >= 2 && lam !== 1.0) ? p0 + (pol[361 + p] - p0) * lam : p0;
      }
      const l0 = val[0], l1 = val[1];
      const m = Math.max(l0, l1), e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
      out.push({
        policy, policyPass: pas[0],
        winLoss: (e0 - e1) / (e0 + e1),
        scoreMean: sv[0] * 20, scoreStdev: softPlus(sv[1]) * 20, scoreLead: sv[2] * 20,
        shorttermWinlossError: softPlus(sv[4] * 0.5) * 0.5,
        shorttermScoreError: softPlus(sv[5] * 0.5) * Math.sqrt(150),
        ownership: Float32Array.from(own),
      });
    }
    return out;
  },
};

const parseSgf = (t) => {
  const moves = [];
  const re = /;[BW]\[([a-z]{2})\]/g; let m;
  while ((m = re.exec(t))) moves.push((18 - (m[1].charCodeAt(0)-97))*19 + (m[1].charCodeAt(1)-97));
  return moves;
};
const posList = [];
for (const f of process.argv.slice(2)) {
  const moves = parseSgf(fs.readFileSync(f, "utf8"));
  for (let ply = 20; ply < moves.length; ply += 40) posList.push({ tag: f.split("/").pop() + "#" + ply, moves: moves.slice(0, ply) });
}

const sp = new Float32Array(22*361), gl = new Float32Array(19);
async function evalAfter(moves, mv, stm) {
  const bd = newBoard();
  replayMoves(bd, moves);
  make(bd, mv, stm);
  const f = encodeFeatures(bd, stm ^ 1, { recentMoves: moves.concat([mv]), komi: KOMI, outSpatial: sp, outGlobal: gl });
  const [pas, pol, val, sv, own] = await askOracle(f.spatial, f.global);
  const l0 = val[0], l1 = val[1];
  const mm = Math.max(l0, l1), e0 = Math.exp(l0-mm), e1 = Math.exp(l1-mm);
  const wlW = (e0-e1)/(e0+e1);
  return { wl: stm === BLACK ? -wlW : wlW, lead: stm === BLACK ? -sv[2]*20 : sv[2]*20 };
}

await connect();
let nSame = 0, nDiff = 0, b4Better = 0, b1Better = 0, tieQ = 0;
for (const { tag, moves } of posList) {
  const bd = newBoard();
  const stm = replayMoves(bd, moves);
  const r1 = await nnSearchBest(bd, stm, { session, visits: VISITS, maxBatch: 1, reuseTree: false, recentMoves: moves.slice() });
  const r4 = await nnSearchBest(bd, stm, { session, visits: VISITS, maxBatch: 4, reuseTree: false, recentMoves: moves.slice() });
  if (r1.move === r4.move) { nSame++; continue; }
  nDiff++;
  const e1 = await evalAfter(moves, r1.move, stm);
  const e4 = await evalAfter(moves, r4.move, stm);
  const dWl = e4.wl - e1.wl, dLead = e4.lead - e1.lead;
  if (Math.abs(dLead) < 0.15 && Math.abs(dWl) < 0.01) { tieQ++; continue; }
  if (dLead > 0 || (Math.abs(dLead) < 0.15 && dWl > 0)) b4Better++; else b1Better++;
  console.log(tag, "stm", stm === BLACK ? "B" : "W", "b1:", toGtp(r1.move), "b4:", toGtp(r4.move),
    "dLead(b4-b1)", dLead.toFixed(2), "dWl", dWl.toFixed(3));
}
console.log("SUMMARY same=" + nSame + " diff=" + nDiff + " (b4 better " + b4Better + " | b1 better " + b1Better + " | tie " + tieQ + ")");
process.exit(0);
