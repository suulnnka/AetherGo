/* 差分调试 JS 侧:共享 oracle 推理、批 1、64 visits,输出根子访问分布 */
import net from "node:net";
import { newBoard, replayMoves, KOMI } from "/home/a/go/AetherGo/src/engine.js";
import { nnSearchBest } from "/home/a/go/AetherGo/src/nn/search.js";

const VISITS = Number(process.env.VISITS ?? 64);
const COLS = "ABCDEFGHJKLMNOPQRST";
const toGtp = (mv) => mv === 361 ? "pass" : COLS[mv % 19] + (19 - ((mv / 19) | 0));

/* ---- oracle 客户端(与 C++ 侧同协议) ---- */
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

/* ---- session.js 同款后处理 ---- */
const softPlus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));
let oracleCalls = 0;
const session = {
  maxBatch: 1,
  async evalBatch(rows) {
    const out = [];
    for (const r of rows) {
      oracleCalls++;
      const [pas, pol, val, sv, own] = await askOracle(r.spatial, r.global);
      const lam = r.optimism ?? 1.0;
      const polC = pol.length / 361;
      const passC = pas.length;
      const policy = new Float32Array(361);
      for (let p = 0; p < 361; p++) {
        const p0 = pol[p];
        policy[p] = (polC >= 2 && lam !== 1.0) ? p0 + (pol[361 + p] - p0) * lam : p0;
      }
      const pb = pas[0];
      const policyPass = (polC >= 2 && lam !== 1.0) ? pb + (pas[1] - pb) * lam : pb;
      const l0 = val[0], l1 = val[1];
      const m = Math.max(l0, l1), e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
      out.push({
        policy, policyPass,
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

const POSITIONS = {
  P1_empty: [], P2_q16: [72], P3: [72, 288], P6_k10: [180],
  P7: [72, 288, 300, 60, 111, 313, 249, 43],
  P8: [72, 288, 300, 60, 111, 313, 249, 43, 270, 35],
  M1_ply40: [72, 288, 300, 60, 111, 313, 249, 43, 270, 35, 53, 41, 301, 97, 63, 308, 74, 296, 46, 271, 319, 192, 51, 286, 225, 212, 100, 292, 67, 318],
};

await connect();
for (const [tag, moves] of Object.entries(POSITIONS)) {
  const bd = newBoard();
  const stm = replayMoves(bd, moves);
  oracleCalls = 0;
  const t0 = Date.now();
  const r = await nnSearchBest(bd, stm, { session, visits: VISITS, maxBatch: 1, recentMoves: moves.slice(), debug: true });
  const stats = (r.rootChildStats ?? [])
    .map((s) => ({ mv: toGtp(s.m), v: s.v, p: s.p, q: s.q, w: s.w, n: s.n }))
    .sort((a, b) => b.v - a.v);
  console.log(JSON.stringify({
    tag, stm: stm === 0 ? "B" : "W", move: toGtp(r.move), winRate: +r.winRate.toFixed(4),
    scoreLead: r.scoreLead == null ? null : +r.scoreLead.toFixed(2),
    oracleCalls, ms: Date.now() - t0,
    children: stats,
  }));
}
process.exit(0);
