/* 真实棋谱差分 JS 侧:从 SGF 取样局面,oracle 推理 64 visits,输出根分布 */
import net from "node:net";
import { readFileSync } from "node:fs";
import { newBoard, replayMoves, PASS } from "/home/a/go/AetherGo/src/engine.js";
import { nnSearchBest } from "/home/a/go/AetherGo/src/nn/search.js";

const VISITS = Number(process.env.VISITS ?? 64);
const STEP = Number(process.env.STEP ?? 40);
const SGFS = process.argv.slice(2);
const COLS = "ABCDEFGHJKLMNOPQRST";
const toGtp = (mv) => mv === PASS ? "pass" : COLS[mv % 19] + (19 - ((mv / 19) | 0));

let sock = null, rbuf = Buffer.alloc(0), waiters = [];
function connect() {
  return new Promise((res, rej) => {
    sock = net.connect(9911, "127.0.0.1", res);
    sock.on("error", rej);
    sock.on("data", (ch) => {
      rbuf = Buffer.concat([rbuf, ch]);
      while (true) {
        if (rbuf.length < 24) return;
        const cnt = [];
        for (let k = 0; k < 5; k++) cnt.push(rbuf.readUInt32LE(4 + 4 * k));
        const need = 24 + 4 * cnt.reduce((a, b) => a + b, 0);
        if (rbuf.length < need) return;
        const body = rbuf.subarray(24, need); rbuf = rbuf.subarray(need);
        const w = waiters.shift();
        if (w) { const arrs = []; let off = 0;
          for (const c of cnt) { arrs.push(new Float32Array(body.buffer, body.byteOffset + off, c)); off += 4 * c; }
          w(arrs); }
      }
    });
  });
}
function ask(spatial, global) {
  return new Promise((res, rej) => {
    waiters.push(res);
    const head = Buffer.alloc(4); head.writeUInt32LE(0x314F4741);
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
  maxBatch: 1,
  async evalBatch(rows) {
    const out = [];
    for (const r of rows) {
      const [pas, pol, val, sv, own] = await askOracleCache(r);
      const lam = r.optimism ?? 1.0;
      const polC = pol.length / 361;
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
/* 输入哈希缓存:同一局面重复评估直接复用 */
const cache = new Map();
async function askOracleCache(r) {
  const key = Buffer.from(r.spatial.buffer, r.spatial.byteOffset, 4 * r.spatial.length).toString("latin1")
    + "|" + Buffer.from(r.global.buffer, r.global.byteOffset, 4 * r.global.length).toString("latin1");
  if (cache.has(key)) return cache.get(key);
  const res = await ask(r.spatial, r.global);
  cache.set(key, res);
  return res;
}

function loadSgf(path) {
  const t = readFileSync(path, "utf8");
  const re = /;([BW])\[([a-z]{0,2})\]/g;
  const moves = [];
  let m;
  while ((m = re.exec(t)) !== null) {
    moves.push(m[2] === "" ? PASS : (m[2].charCodeAt(1) - 97) * 19 + (m[2].charCodeAt(0) - 97));
  }
  return moves;
}

await connect();
for (const path of SGFS) {
  const moves = loadSgf(path);
  const g = path.replace(/.*\//, "");
  for (let i = 20; i + 30 < moves.length; i += STEP) {
    const prefix = moves.slice(0, i);
    const bd = newBoard();
    const stm = replayMoves(bd, prefix);
    if (stm < 0) { console.error("replay fail " + g + "@" + i); continue; }
    const r = await nnSearchBest(bd, stm, { session, visits: VISITS, maxBatch: 1, recentMoves: prefix.slice(), debug: true });
    const children = (r.rootChildStats ?? [])
      .map((s) => ({ mv: toGtp(s.m), v: s.v, p: s.p, q: s.q, w: s.w, n: s.n }))
      .sort((a, b) => b.v - a.v);
    console.log(JSON.stringify({ game: g, ply: i, stm: stm === 0 ? "B" : "W",
      move: toGtp(r.move), winRate: +r.winRate.toFixed(4), children }));
  }
}
process.exit(0);
