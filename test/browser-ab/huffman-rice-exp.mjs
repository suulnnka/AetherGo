/* 实验 v2:int8 权重 Huffman / Rice 编码(真码流 + 回环校验),编码后再 gzip。
 * 对 i8.aewn 的 int8 打包区(权重值字节流)编码;头部/scale 区不参与。
 * 变体:全局 Huffman / 逐张量 Huffman(各带码长表)/ 全局最优 k Rice / 分块自适应 k Rice。
 * 用法:node test/browser-ab/huffman-rice-exp.mjs */
import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';

const FILE = 'models/b8c96h3tfrs_19.i8.aewn';
const buf = readFileSync(FILE);
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const T = process.env.TEMP.replace(/\\/g, '/');

if (dv.getUint32(0, true) !== 0x4e574541) throw new Error('not AEWN');
if (dv.getUint32(8, true) !== 1) throw new Error('expect dtype=1');
const nTensors = dv.getUint32(16, true);
let off = 20;
const dir = [];
for (let i = 0; i < nTensors; i++) {
  const nameLen = dv.getUint32(off, true); off += 4;
  const name = Buffer.from(buf.subarray(off, off + nameLen)).toString(); off += nameLen;
  const ndim = dv.getUint32(off, true); off += 4;
  off += ndim * 4;
  const tOff = Number(dv.getBigUint64(off, true)); off += 8;
  const nbytes = dv.getUint32(off, true); off += 8;
  dir.push({ name, tOff, nbytes });
}
/* int8 权重张量:scale(.s)与 f32 头/r Rope 表排除(scale 是 f32,量纲不同) */
const wts = dir.filter((t) => !t.name.endsWith('.s') && !t.name.startsWith('rope.'));
const W = Buffer.concat(wts.map((t) => buf.subarray(t.tOff, t.tOff + t.nbytes)));
console.log(`权重区 ${(W.length / 1048576).toFixed(3)}MB(${wts.length} 张量)`);


function histOf(chunk) {
  const f = new Float64Array(256);
  for (const b of chunk) f[b]++;
  return f;
}
/* ---- 熵 ---- */
function entropyOf(chunk) {
  const f = new Float64Array(256);
  for (const b of chunk) f[b]++;
  let H = 0, n = chunk.length;
  for (let i = 0; i < 256; i++) if (f[i]) { const p = f[i] / n; H -= p * Math.log2(p); }
  return H;
}
const H = entropyOf(W);
console.log(`全局熵 ${H.toFixed(3)} bit/值 → 下限 ${(W.length * H / 8 / 1048576).toFixed(3)}MB(${(H / 8 * 100).toFixed(1)}%)`);
let hSum = 0;
for (const t of wts) hSum += entropyOf(buf.subarray(t.tOff, t.tOff + t.nbytes)) * t.nbytes;
console.log(`逐张量熵和 ${(hSum / 8 / 1048576).toFixed(3)}MB(${(hSum / 8 / W.length * 100).toFixed(1)}%)← 分布差异带来的理论空间`);

/* ---- 位流写入器 ---- */
class BitW {
  constructor() { this.bytes = []; this.acc = 0; this.n = 0; this.bits = 0n; }
  put(code, len) {                       // code < 2^len,高位在前
    this.bits += BigInt(len);
    for (let i = len - 1; i >= 0; i--) {
      this.acc = (this.acc << 1) | ((code >> i) & 1);
      if (++this.n === 8) { this.bytes.push(this.acc); this.acc = 0; this.n = 0; }
    }
  }
  flush() { if (this.n) { this.bytes.push(this.acc << (8 - this.n)); this.acc = 0; this.n = 0; } return Buffer.from(this.bytes); }
}

/* ---- 规范 Huffman:码长表(256B)+ 码流;回环校验 ---- */
function huffLens(freq) {
  const nd = [];
  for (let s = 0; s < 256; s++) if (freq[s]) nd.push({ s, f: freq[s] });
  if (nd.length === 1) { const l = new Uint8Array(256); l[nd[0].s] = 1; return l; }
  while (nd.length > 1) {
    nd.sort((a, b) => a.f - b.f);
    const a = nd.shift(), b = nd.shift();
    nd.push({ f: a.f + b.f, l: a, r: b });
  }
  const lens = new Uint8Array(256);
  (function walk(n, d) { n.l ? (walk(n.l, d + 1), walk(n.r, d + 1)) : (lens[n.s] = Math.max(1, d)); })(nd[0], 0);
  return lens;
}
function canonCodes(lens) {               // 规范码:按 (长度,符号) 排序连续分配
  const maxLen = Math.max(...lens);
  const codes = new Uint32Array(256);
  let code = 0;
  for (let l = 1; l <= maxLen; l++) {
    for (let s = 0; s < 256; s++) if (lens[s] === l) codes[s] = code++;
    code <<= 1;
  }
  return codes;
}
function huffEncode(chunk, lens) {
  const codes = canonCodes(lens);
  const w = new BitW();
  for (const b of chunk) w.put(codes[b], lens[b]);
  return w.flush();
}
function huffDecode(enc, lens, nOut) {
  const codes = canonCodes(lens);
  const map = new Map();
  for (let s = 0; s < 256; s++) if (lens[s]) map.set(`${lens[s]}:${codes[s]}`, s);
  const out = Buffer.alloc(nOut);
  let code = 0, len = 0, oi = 0;
  for (const byte of enc) {
    for (let i = 7; i >= 0; i--) {
      code = (code << 1) | ((byte >> i) & 1); len++;
      const s = map.get(`${len}:${code}`);
      if (s !== undefined) { out[oi++] = s; code = 0; len = 0; if (oi === nOut) return out; }
    }
  }
  throw new Error('decode underrun');
}
function huffVariant(name, chunk, lens) {
  const enc = huffEncode(chunk, lens);
  const dec = huffDecode(enc, lens, chunk.length);
  const ok = dec.equals(chunk);
  const size = enc.length + lens.reduce((a, l, s) => a + (l ? 1 : 0), 0);   // 表=非零码长各 1B
  console.log(`[${name}] 表+码流 ${size}B = ${(size * 100 / chunk.length).toFixed(1)}% of raw ${'回环' + (ok ? 'OK' : 'FAIL!')}`);
  return { enc, size, ok };
}

/* ---- Rice:zigzag → (u>>k) 个 0 + 1 + k 位余数 ---- */
const zig = new Uint8Array(256);
for (let v = 0; v < 256; v++) { const s = v >= 128 ? v - 256 : v; zig[v] = s >= 0 ? 2 * s : -2 * s - 1; }
function riceEncodeReal(chunk, k) {
  const w = new BitW();
  for (const b of chunk) {
    const u = zig[b], q = u >> k;
    for (let i = 0; i < q; i++) w.put(0, 1);
    w.put(1, 1);
    if (k) w.put(u & ((1 << k) - 1), k);
  }
  return w.flush();
}
function riceVariant(name, chunk, kFn, kStore) {
  const w = new BitW();
  const ks = [];
  const TILE = 1 << 14;
  for (let t = 0; t < chunk.length; t += TILE) {
    const c = chunk.subarray(t, Math.min(chunk.length, t + TILE));
    const k = kFn(c);
    ks.push(k);
    if (kStore) w.put(k, 4);
    for (const b of c) {
      const u = zig[b], q = u >> k;
      for (let i = 0; i < q; i++) w.put(0, 1);
      w.put(1, 1);
      if (k) w.put(u & ((1 << k) - 1), k);
    }
  }
  const enc = w.flush();
  const size = enc.length + (kStore ? ks.length * 0.5 : 0);
  console.log(`[${name}] 码流 ${Math.round(size)}B = ${(size * 100 / chunk.length).toFixed(1)}% of raw(ks 分布 ${[...new Set(ks)].join(',')})`);
  return { enc, size };
}
const bestK = (c) => { let bk = 0, bb = Infinity; for (let k = 0; k <= 6; k++) { let bits = 0; for (const b of c) bits += (zig[b] >> k) + 1 + k; if (bits < bb) { bb = bits; bk = k; } } return bk; };

/* ---- 逐张量 Huffman:每张量自带 256B 码长表 ---- */
console.log('\n===== Huffman =====');
{
  const lens = huffLens(histOf(W));
  const v = huffVariant('全局表', W, lens);
  writeFileSync(`${T}/w-huf-g.bin`, v.enc);
}
{
  let total = 0, parts = [], ok = true;
  for (const t of wts) {
    const c = buf.subarray(t.tOff, t.tOff + t.nbytes);
    const lens = huffLens(histOf(c));
    const enc = huffEncode(c, lens);
    if (!huffDecode(enc, lens, c.length).equals(c)) ok = false;
    parts.push(enc);
    total += enc.length + 256;
  }
  const all = Buffer.concat(parts);
  writeFileSync(`${T}/w-huf-t.bin`, all);
  console.log(`[逐张量表] ${total}B = ${(total * 100 / W.length).toFixed(1)}% of raw 回环${ok ? 'OK' : 'FAIL!'}`);
}

/* ---- Rice ---- */
console.log('===== Rice =====');
riceVariant('全局最优 k', W, bestK, false);
riceVariant('16KB 分块自适应 k', W, bestK, true);
writeFileSync(`${T}/w-rice.bin`, (() => {
  const w = new BitW();
  for (let t = 0; t < W.length; t += (1 << 14)) {
    const c = W.subarray(t, Math.min(W.length, t + (1 << 14)));
    const k = bestK(c); w.put(k, 4);
    for (const b of c) { const u = zig[b], q = u >> k; for (let i = 0; i < q; i++) w.put(0, 1); w.put(1, 1); if (k) w.put(u & ((1 << k) - 1), k); }
  }
  return w.flush();
})());

/* ---- 基线与编码后再 gzip ---- */
const gz = (f) => +execSync(`gzip -9 -c "${f}" | wc -c`);
console.log('\n===== 汇总(权重区 ' + (W.length / 1048576).toFixed(3) + 'MB)=====');
console.log(`raw 权重区 gzip:            ${((gz('models/b8c96h3tfrs_19.i8.aewn') / 1048576) * (W.length / buf.length)).toFixed(3)}MB(整文件 gz 按占比折算)`);
for (const [n, f] of [['huffman-全局', `${T}/w-huf-g.bin`], ['huffman-逐张量', `${T}/w-huf-t.bin`], ['rice-16K', `${T}/w-rice.bin`]]) {
  console.log(`${n}: 码流 ${(statSync(f).size / 1048576).toFixed(3)}MB → gzip ${(gz(f) / 1048576).toFixed(3)}MB`);
}
