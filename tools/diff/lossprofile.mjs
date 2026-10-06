/* 逐手损失剖析:重放 sgf,对指定方的每手算 oracle 局面评估(lead),按 40 手窗聚合损失 */
import net from "node:net";
import fs from "node:fs";
import { newBoard, replayMoves, KOMI, make, BLACK, WHITE } from "/home/a/go/AetherGo/src/engine.js";
import { encodeFeatures } from "/home/a/go/AetherGo/src/nn/features.js";

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
    const head = Buffer.alloc(4); head.writeUInt32LE(0x314F4741);
    const parts = [head];
    for (const arr of [spatial, global]) {
      const n = Buffer.alloc(4); n.writeUInt32LE(arr.length); parts.push(n);
      parts.push(Buffer.from(arr.buffer, arr.byteOffset, 4 * arr.length));
    }
    sock.write(Buffer.concat(parts));
    setTimeout(() => rej(new Error("timeout")), 60000);
  });
}
const sp = new Float32Array(22*361), gl = new Float32Array(19);
async function leadW(moves) {
  const bd = newBoard();
  const stm = replayMoves(bd, moves);
  const f = encodeFeatures(bd, stm, { recentMoves: moves, komi: KOMI, outSpatial: sp, outGlobal: gl });
  const [, , , sv, ] = await askOracle(f.spatial, f.global);
  return sv[2] * 20;   /* 白视角 lead */
}

const sgf = fs.readFileSync(process.argv[2], "utf8");
const moves = [];
const re = /;([BW])\[([a-z]{2})\]/g; let m;
while ((m = re.exec(sgf))) moves.push(m[2] === "" ? 361 : (18 - (m[2].charCodeAt(0)-97))*19 + (m[2].charCodeAt(1)-97));
const targetBlack = process.argv[3] === "B";   /* 剖析哪一方 */
await connect();
/* 每 20 手采样 lead,看崩坏时段 */
let prev = null;
const marks = [];
for (let ply = 0; ply <= moves.length; ply += 20) {
  const l = await leadW(moves.slice(0, ply));
  marks.push({ ply, leadW: l });
}
console.log("ply: lead(white persp)  — target", targetBlack ? "BLACK" : "WHITE");
for (const mk of marks) console.log(mk.ply, mk.leadW.toFixed(1));
process.exit(0);
