/* ============================================================
 * aethernn WGSL 内核 —— b8c96h3tfrs 专用(v17 transformer)
 *
 * 内核面与执行顺序一一对应 PyTorch forward(见 plan.js 头注释);数学与
 * src/nn/webgpu/cpuref.js 逐算子同构(同一累加序),互为对拍参照。
 * 部分内核结构移植自 katago-webgpu(webgpukernels.cpp,MIT),按本引擎的
 * NHWC 布局与「无掩码」口径(require-exact-nnlen,输入恒满盘)改写:
 *   - tiledGemm 家族 ← tiledGemm/tiledGemmRT(B 版改为 NHWC 连续 K 读法)
 *   - flashAttention ← flashAttention(去掩码分支)
 *   - rmsNorm ← rmsNorm(NCHW→NHWC,索引更简)
 *
 * 两种模式(buildKernels(mode),与 .aewn 的 header.dtype 对应):
 *   'f32'  全 fp32 存储/计算(对拍基准)。
 *   'q'    i8f16(INT8 报告 §4.1 方案 A 的引擎侧实现):
 *          - trunk 大权重 int8(4×int8 打包 u32,LSB 在前)+ per-oc f32 scale,
 *            tile 装载时反量化(shader 移位取字节,免 i8 类型);
 *          - 激活中间量 f16 存储、寄存器/共享内存/累加一律 f32
 *            (katago-webgpu 的「f16 storage + fp32 compute」形态;需要
 *            adapter 的 shader-f16 特性);
 *          - 头部(policy/value)按排除清单留 f32 权重,但输入仍是 f16 激活;
 *          - gp 池化向量与全部输出缓冲(pol/pass/val/misc/own)保持 f32
 *            (免 JS 侧 f16 解码;量小,带宽无关紧要)。
 *
 * 融合清单(相对 ONNX 图的 423 节点):
 *   stem    conv3x3 + 对称 gather + linear_global 广播加  → 1 dispatch
 *   qkv     3 MatMul → 1 GEMM(PyTorch fused_qkv_proj 同构)+ head-major 散排
 *   rope    Gather/Cos/Sin 图内子图 → 打包期预算表 + 1 dispatch(原地配对旋转)
 *   attn    flashAttention 在线 softmax(scores/softmax/sv 3→1,免 S² 物化)
 *   out_proj/ffn2  残差加并入 GEMM epilogue
 *   ffn     linear1+gate → 1 GEMM(fused_gate_proj 同构);swiglu 1 dispatch
 *   heads   BiasMask 的 scale 折进权重(packer);linear_g+gpbias+bias2+relu、
 *           linear_pass 全路、value MLP 全路各 1 dispatch
 * ============================================================ */

/* ---- GEMM 家族:16×16 平铺,A=激活 NHWC (n·361, K),W=[K][O] k 主序 ----
 * 三种权重/IO 形态:
 *   gemmSource     f32 权重,f32 io(f32 模式全体 + 无)
 *   gemm32Source   f32 权重,f16 io(q 模式的头部 GEMM:conv1p/conv1g/conv1)
 *   gemmQSource    int8 权重(u32 打包+scale),f16 io(q 模式 trunk GEMM)
 * 变体以 epilogue 区分:plain / res(残差加,f16)/ biasrelu(f32 bias)。 */

function gemmSource(epi, name) {
  const extra = epi === 'res'
    ? `@group(0) @binding(4) var<storage, read>       geRes : array<f32>;`
    : epi === 'biasrelu'
      ? `@group(0) @binding(4) var<storage, read>       geBias : array<f32>;`
      : ``;
  const epilogue = epi === 'res'
    ? `acc = acc + geRes[(n2 * 361u + m) * geo.o + o];`
    : epi === 'biasrelu'
      ? `acc = max(acc + geBias[o], 0.0);`
      : ``;
  return /* wgsl */ `
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> geo : GParams;
@group(0) @binding(1) var<storage, read>       geIn  : array<f32>;
@group(0) @binding(2) var<storage, read>       geW   : array<f32>;
${extra}
@group(0) @binding(3) var<storage, read_write> geOut : array<f32>;

var<workgroup> geAs : array<f32, 256>;   // W tile [o][k]
var<workgroup> geBs : array<f32, 256>;   // in tile [k][m]

@compute @workgroup_size(16, 16)
fn ${name}(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n2 = wid.z;
  let o  = wid.x * 16u + lid.x;          // 输出通道
  let m  = wid.y * 16u + lid.y;          // 空间位置 [0,361)
  var acc : f32 = 0.0;
  let nTiles = (geo.k + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * 16u + lid.y;
    var av : f32 = 0.0;
    if (o < geo.o && kA < geo.k) { av = geW[kA * geo.o + o]; }
    geAs[lid.x * 16u + lid.y] = av;
    let kB = t * 16u + lid.x;
    var bv : f32 = 0.0;
    if (kB < geo.k && m < 361u) { bv = geIn[(n2 * 361u + m) * geo.k + kB]; }
    geBs[lid.x * 16u + lid.y] = bv;
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < 16u; kk = kk + 1u) {
      acc = acc + geAs[lid.x * 16u + kk] * geBs[kk * 16u + lid.y];
    }
    workgroupBarrier();
  }
  if (o < geo.o && m < 361u) {
    ${epilogue}
    geOut[(n2 * 361u + m) * geo.o + o] = acc;
  }
}
`;
}

function gemm32Source(epi, name) {
  const extra = epi === 'biasrelu'
    ? `@group(0) @binding(4) var<storage, read>       geBias : array<f32>;`
    : ``;
  const epilogue = epi === 'biasrelu' ? `acc = max(acc + geBias[o], 0.0);` : ``;
  return /* wgsl */ `
enable f16;
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> geo : GParams;
@group(0) @binding(1) var<storage, read>       geIn  : array<f16>;
@group(0) @binding(2) var<storage, read>       geW   : array<f32>;
${extra}
@group(0) @binding(3) var<storage, read_write> geOut : array<f16>;

var<workgroup> geAs : array<f32, 256>;
var<workgroup> geBs : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn ${name}(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n2 = wid.z;
  let o  = wid.x * 16u + lid.x;
  let m  = wid.y * 16u + lid.y;
  var acc : f32 = 0.0;
  let nTiles = (geo.k + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * 16u + lid.y;
    var av : f32 = 0.0;
    if (o < geo.o && kA < geo.k) { av = geW[kA * geo.o + o]; }
    geAs[lid.x * 16u + lid.y] = av;
    let kB = t * 16u + lid.x;
    var bv : f32 = 0.0;
    if (kB < geo.k && m < 361u) { bv = f32(geIn[(n2 * 361u + m) * geo.k + kB]); }
    geBs[lid.x * 16u + lid.y] = bv;
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < 16u; kk = kk + 1u) {
      acc = acc + geAs[lid.x * 16u + kk] * geBs[kk * 16u + lid.y];
    }
    workgroupBarrier();
  }
  if (o < geo.o && m < 361u) {
    ${epilogue}
    geOut[(n2 * 361u + m) * geo.o + o] = f16(acc);
  }
}
`;
}

function gemmQSource(epi, name, wkind = 'i8') {
  const i8 = wkind === 'i8';
  const f16w = wkind === 'f16';
  /* 绑定:[0]u [1]in [2]W [3](S=i8) [3|4](res/bias) [末]out。
   * res 输入 f16(残差载体),bias 向量 f32(头部常量)。wkind='f32' 用于
   * q 模式下被排除的张量(f32 权重 + f16 io,逐层敏感度排除用)。 */
  /* 绑定序与 session 一致:f16+biasrelu 是 [u,in,W,out,bias](out@3,bias@4),
   * 其余 res/bias 在 out 之后。 */
  const extraB = 4;
  const extra = epi === 'res'
    ? `@group(0) @binding(${extraB}) var<storage, read>       geRes : array<f16>;`
    : epi === 'biasrelu'
      ? `@group(0) @binding(${extraB}) var<storage, read>       geBias : array<f32>;`
      : ``;
  const epilogue = epi === 'res'
    ? `acc = acc + f32(geRes[(n2 * 361u + m) * geo.o + o]);`
    : epi === 'biasrelu'
      ? `acc = max(acc + f32(geBias[o]), 0.0);`
      : ``;
  /* i8:[0]u [1]in [2]W [3]S [4]res [5]out;非 i8:[0]u [1]in [2]W [3]out [4]res/bias。 */
  const outB = i8 ? (epi === 'plain' ? 4 : 5) : 3;
  return /* wgsl */ `
enable f16;
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> geo : GParams;
@group(0) @binding(1) var<storage, read>       geIn  : array<f16>;
@group(0) @binding(2) var<storage, read>       geW   : array<${i8 ? 'u32' : f16w ? 'f16' : 'f32'}>;
${i8 ? `@group(0) @binding(3) var<storage, read>       geS   : array<f32>;` : ''}
${extra}
@group(0) @binding(${outB}) var<storage, read_write> geOut : array<f16>;

var<workgroup> geAs : array<f32, 256>;
var<workgroup> geBs : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn ${name}(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n2 = wid.z;
  let o  = wid.x * 16u + lid.x;
  let m  = wid.y * 16u + lid.y;
  var acc : f32 = 0.0;
  let nTiles = (geo.k + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * 16u + lid.y;
    var av : f32 = 0.0;
    if (o < geo.o && kA < geo.k) {
      ${i8
    ? `let lin = kA * geo.o + o;
      let byte = (geW[lin >> 2u] >> ((lin & 3u) * 8u)) & 0xFFu;   // 4×int8 LSB 在前
      av = (f32(byte) - select(0.0, 256.0, byte >= 128u)) * geS[o];`
    : f16w
      ? `av = f32(geW[kA * geo.o + o]);`
      : `av = geW[kA * geo.o + o];`}
    }
    geAs[lid.x * 16u + lid.y] = av;
    let kB = t * 16u + lid.x;
    var bv : f32 = 0.0;
    if (kB < geo.k && m < 361u) { bv = f32(geIn[(n2 * 361u + m) * geo.k + kB]); }
    geBs[lid.x * 16u + lid.y] = bv;
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < 16u; kk = kk + 1u) {
      acc = acc + geAs[lid.x * 16u + kk] * geBs[kk * 16u + lid.y];
    }
    workgroupBarrier();
  }
  if (o < geo.o && m < 361u) {
    ${epilogue}
    geOut[(n2 * 361u + m) * geo.o + o] = f16(acc);
  }
}
`;
}

/* qkv 变体生成(f32 与 Q 两形态):输出按 o 散排到 head-major (b,h,361,32) 三缓冲 */
function gemmQkvSource(wkind, ioF16 = true, name = 'gemmQkv') {
  const quant = wkind === 'i8';
  const f16 = ioF16;
  const inT = f16 ? 'f16' : 'f32';
  const outT = f16 ? 'f16' : 'f32';
  return /* wgsl */ `
${f16 ? 'enable f16;' : ''}
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> gq : GParams;
@group(0) @binding(1) var<storage, read>       gqIn  : array<${inT}>;
@group(0) @binding(2) var<storage, read>       gqW   : array<${wkind === 'i8' ? 'u32' : wkind === 'f16' ? 'f16' : 'f32'}>;
${quant ? `@group(0) @binding(6) var<storage, read>       gqS   : array<f32>;` : ''}
@group(0) @binding(3) var<storage, read_write> gqQh  : array<${outT}>;
@group(0) @binding(4) var<storage, read_write> gqKh  : array<${outT}>;
@group(0) @binding(5) var<storage, read_write> gqVh  : array<${outT}>;

var<workgroup> gqAs : array<f32, 256>;
var<workgroup> gqBs : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn ${name}(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n2 = wid.z;
  let o  = wid.x * 16u + lid.x;
  let m  = wid.y * 16u + lid.y;
  var acc : f32 = 0.0;
  let nTiles = (gq.k + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * 16u + lid.y;
    var av : f32 = 0.0;
    if (o < gq.o && kA < gq.k) {
      ${quant
    ? `let lin = kA * gq.o + o;
        let byte = (gqW[lin >> 2u] >> ((lin & 3u) * 8u)) & 0xFFu;
        av = (f32(byte) - select(0.0, 256.0, byte >= 128u)) * gqS[o];`
    : wkind === 'f16'
      ? `av = f32(gqW[kA * gq.o + o]);`
      : `av = gqW[kA * gq.o + o];`}
    }
    gqAs[lid.x * 16u + lid.y] = av;
    let kB = t * 16u + lid.x;
    var bv : f32 = 0.0;
    if (kB < gq.k && m < 361u) { bv = ${f16 ? 'f32(gqIn[(n2 * 361u + m) * gq.k + kB])' : 'gqIn[(n2 * 361u + m) * gq.k + kB]'}; }
    gqBs[lid.x * 16u + lid.y] = bv;
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < 16u; kk = kk + 1u) {
      acc = acc + gqAs[lid.x * 16u + kk] * gqBs[kk * 16u + lid.y];
    }
    workgroupBarrier();
  }
  if (o < gq.o && m < 361u) {
    /* 各段内的局部头号:q 取 o 直接拆,k/v 需先减段基址(o=96/192) */
    if (o < 96u) {
      let dst = ((n2 * 3u + o / 32u) * 361u + m) * 32u + (o % 32u);
      gqQh[dst] = ${f16 ? 'f16(acc)' : 'acc'};
    } else if (o < 192u) {
      let ok = o - 96u;
      let dst = ((n2 * 3u + ok / 32u) * 361u + m) * 32u + (ok % 32u);
      gqKh[dst] = ${f16 ? 'f16(acc)' : 'acc'};
    } else {
      let ov = o - 192u;
      let dst = ((n2 * 3u + ov / 32u) * 361u + m) * 32u + (ov % 32u);
      gqVh[dst] = ${f16 ? 'f16(acc)' : 'acc'};
    }
  }
}
`;
}

export function buildKernels(mode) {
  const q = mode !== 'f32';                     // 'q'(int8 权重)| 'f16w'(f16 权重)
  const T = q ? 'f16' : 'f32';                  // 激活中间量存储类型
  const ld = (expr) => (q ? `f32(${expr})` : expr);   // 装载转换
  const st = (expr) => (q ? `f16(${expr})` : expr);   // 存储转换
  const K = {};

  /* stem:对称 gather + conv3x3(22→96)+ linear_global 广播加 → NHWC trunk
   * out[n,q,oc] = Σ_{ic,d} W[oc,ic,d]·in[n,ic,inv_sym(q+δ)] + Σ_j Wg[oc,j]·g[n,j]
   * 盘外邻居以 zeroSlot 哨兵跳过(等价零填充)。压缩模式两矩阵都压缩(报告 58 层
   * 口径含 linear_global);stem32 = f32 权重 + f16 io(q 模式排除时用)。 */
  const stemSource = (wkind, name = 'stem') => {
    const i8 = wkind === 'i8';
    const f16w = wkind === 'f16';
    const W = i8 ? 'u32' : f16w ? 'f16' : 'f32';
    const wld = (expr) => (wkind === 'f32' ? expr : `f32(${expr})`);
    return /* wgsl */ `
${q ? 'enable f16;' : ''}
struct StemParams { n: u32, zeroSlot: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> st : StemParams;
@group(0) @binding(1) var<storage, read>       stIn    : array<f32>;
@group(0) @binding(2) var<storage, read>       stW     : array<${W}>;
@group(0) @binding(3) var<storage, read>       stGw    : array<${W}>;
@group(0) @binding(4) var<storage, read>       stG     : array<f32>;
@group(0) @binding(5) var<storage, read>       stTbl   : array<u32>;
@group(0) @binding(6) var<storage, read>       stSym   : array<u32>;
@group(0) @binding(7) var<storage, read_write> stOut   : array<${T}>;
${i8 ? `@group(0) @binding(8) var<storage, read>       stS     : array<f32>;
@group(0) @binding(9) var<storage, read>       stGS    : array<f32>;` : ''}

@compute @workgroup_size(64)
fn ${name}(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let per = 361u * 96u;
  if (idx >= st.n * per) { return; }
  let n  = idx / per;
  let q  = (idx % per) / 96u;
  let oc = idx % 96u;
  let sym = stSym[n];
  let tBase = (sym * 361u + q) * 9u;
  var acc : f32 = 0.0;
  let wBase = oc * (22u * 9u);
  for (var ic : u32 = 0u; ic < 22u; ic = ic + 1u) {
    let inBase = (n * 22u + ic) * 361u;
    let kw = wBase + ic * 9u;
    for (var d : u32 = 0u; d < 9u; d = d + 1u) {
      let pos = stTbl[tBase + d];
      if (pos != st.zeroSlot) {
        var wv : f32 = 0.0;
        ${i8
    ? `let lin = kw + d;
        let byte = (stW[lin >> 2u] >> ((lin & 3u) * 8u)) & 0xFFu;
        wv = (f32(byte) - select(0.0, 256.0, byte >= 128u)) * stS[oc];`
    : `wv = ${wld('stW[kw + d]')};`}
        acc = acc + stIn[inBase + pos] * wv;
      }
    }
  }
  var g : f32 = 0.0;
  for (var j : u32 = 0u; j < 19u; j = j + 1u) {
    ${i8
    ? `let glin = oc * 19u + j;
      let gbyte = (stGw[glin >> 2u] >> ((glin & 3u) * 8u)) & 0xFFu;
      g = g + stG[n * 19u + j] * (f32(gbyte) - select(0.0, 256.0, gbyte >= 128u)) * stGS[oc];`
    : `g = g + stG[n * 19u + j] * ${wld('stGw[oc * 19u + j]')};`}
  }
  stOut[idx] = ${st('acc + g')};
}
`;
  };
  K.stem = stemSource(mode === 'q' ? 'i8' : mode === 'f16w' ? 'f16' : 'f32');
  if (mode === 'q') { K.stem32 = stemSource('f32', 'stem32'); }

  /* rmsNorm:逐位置跨通道 RMS(eps=1e-6)× gamma。NHWC:线程 i 直接持 96 连续元素 */
  K.rms = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct RmsParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> rm : RmsParams;
@group(0) @binding(1) var<storage, read>       rmIn    : array<${T}>;
@group(0) @binding(2) var<storage, read>       rmGamma : array<f32>;
@group(0) @binding(3) var<storage, read_write> rmOut   : array<${T}>;

@compute @workgroup_size(64)
fn rms(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;                       // (n*361 + q)
  if (idx >= rm.n * 361u) { return; }
  let base = idx * 96u;
  var ss : f32 = 0.0;
  for (var c : u32 = 0u; c < 96u; c = c + 1u) {
    let v = ${ld('rmIn[base + c]')};
    ss = ss + v * v;
  }
  let r = inverseSqrt(ss / 96.0 + 1.0e-6);
  for (var c : u32 = 0u; c < 96u; c = c + 1u) {
    rmOut[base + c] = ${st(`${ld('rmIn[base + c]')} * r * rmGamma[c]`)};
  }
}
`;

  /* rope:q/k 原地配对旋转(out[j]=x[j]·cos[q,j]+x[j^1]·sin[q,j],swap 已折进
   * sin 符号;cos[2i]=cos[2i+1],sin[2i]=-sin[2i+1] → 每对 (2p,2p+1) 一线程
   * 读两元写两元,免竞态免中间缓冲)。gemmQkv 已把 q/k/v 散排为 head-major,
   * v 无需处理。cos/sin 表保持 f32。 */
  K.rope = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct ElmParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> rp : ElmParams;
@group(0) @binding(1) var<storage, read_write> rpQh  : array<${T}>;
@group(0) @binding(2) var<storage, read_write> rpKh  : array<${T}>;
@group(0) @binding(3) var<storage, read>       rpCos : array<f32>;
@group(0) @binding(4) var<storage, read>       rpSin : array<f32>;

@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;                       // ((n*3 + h)*361 + q)*16 + p
  let total = rp.n * 3u * 361u * 16u;
  if (idx >= total) { return; }
  let p  = idx % 16u;
  let r  = idx / 16u;                    // (n*3 + h)*361 + q
  let q  = r % 361u;
  let base = r * 32u;
  let c0 = q * 32u + 2u * p;
  let c1 = c0 + 1u;
  let cos0 = rpCos[c0]; let sin0 = rpSin[c0];
  let cos1 = rpCos[c1]; let sin1 = rpSin[c1];
  let qx0 = ${ld('rpQh[base + 2u * p]')}; let qx1 = ${ld('rpQh[base + 2u * p + 1u]')};
  rpQh[base + 2u * p]     = ${st('qx0 * cos0 + qx1 * sin0')};
  rpQh[base + 2u * p + 1u] = ${st('qx1 * cos1 + qx0 * sin1')};
  let kx0 = ${ld('rpKh[base + 2u * p]')}; let kx1 = ${ld('rpKh[base + 2u * p + 1u]')};
  rpKh[base + 2u * p]     = ${st('kx0 * cos0 + kx1 * sin0')};
  rpKh[base + 2u * p + 1u] = ${st('kx1 * cos1 + kx0 * sin1')};
}
`;

  /* flashAttention:在线 softmax 融合注意力,免 361×361 scores 物化。
   * q/k/v head-major 入,NHWC (361,96) 出。满盘无掩码分支。累加全程 f32。 */
  K.flash = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct ElmParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> fa : ElmParams;
@group(0) @binding(1) var<storage, read>       faQ   : array<${T}>;
@group(0) @binding(2) var<storage, read>       faK   : array<${T}>;
@group(0) @binding(3) var<storage, read>       faV   : array<${T}>;
@group(0) @binding(4) var<storage, read_write> faOut : array<${T}>;

@compute @workgroup_size(64)
fn flash(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;                       // (n*3 + h)*361 + qi
  let total = fa.n * 3u * 361u;
  if (idx >= total) { return; }
  let qi = idx % 361u;
  let h  = (idx / 361u) % 3u;
  let n2 = idx / (361u * 3u);
  let qBase = ((n2 * 3u + h) * 361u + qi) * 32u;
  var acc : array<f32, 32>;
  for (var e : u32 = 0u; e < 32u; e = e + 1u) { acc[e] = 0.0; }
  var m : f32 = -3.0e38;
  var l : f32 = 0.0;
  let kHead = (n2 * 3u + h) * 361u * 32u;
  for (var ki : u32 = 0u; ki < 361u; ki = ki + 1u) {
    let kBase = kHead + ki * 32u;
    var s : f32 = 0.0;
    for (var d : u32 = 0u; d < 32u; d = d + 1u) {
      s = s + ${ld('faQ[qBase + d]')} * ${ld('faK[kBase + d]')};
    }
    s = s * 0.1767766922712326;
    let newMax = max(m, s);
    let corr = exp(m - newMax);
    let p = exp(s - newMax);
    l = l * corr + p;
    for (var e : u32 = 0u; e < 32u; e = e + 1u) {
      acc[e] = acc[e] * corr + p * ${ld('faV[kBase + e]')};
    }
    m = newMax;
  }
  let inv = 1.0 / l;
  let outBase = (n2 * 361u + qi) * 96u + h * 32u;
  for (var e : u32 = 0u; e < 32u; e = e + 1u) {
    faOut[outBase + e] = ${st('acc[e] * inv')};
  }
}
`;

  /* trunkfinal(fixup):x·scale[c]+bias[c] → relu(无归一化,ONNX norm_trunkfinal 同) */
  K.trunkFinal = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct ElmParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> tf : ElmParams;
@group(0) @binding(1) var<storage, read>       tfIn    : array<${T}>;
@group(0) @binding(2) var<storage, read>       tfScale : array<f32>;
@group(0) @binding(3) var<storage, read>       tfBias  : array<f32>;
@group(0) @binding(4) var<storage, read_write> tfOut   : array<${T}>;

@compute @workgroup_size(64)
fn trunkFinal(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= tf.n * 361u * 96u) { return; }
  let c = idx % 96u;
  tfOut[idx] = ${st(`max(${ld('tfIn[idx]')} * tfScale[c] + tfBias[c], 0.0)`)};
}
`;

  /* poolPolicy:每 (n,c) 池化 → gp[mean, mean·0.5, max](InputMask 常数已折叠)
   * 一个 workgroup(32 线程)处理一行,线程 = 通道。gp 保持 f32(下游小核直接用)。 */
  K.poolPolicy = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct PoolParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> pp : PoolParams;
@group(0) @binding(1) var<storage, read>       ppIn  : array<${T}>;
@group(0) @binding(2) var<storage, read_write> ppGp  : array<f32>;

@compute @workgroup_size(32)
fn poolPolicy(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  if (wid.x >= pp.n) { return; }
  let c = lid.x;
  let base = wid.x * 361u * 32u + c;
  var sum : f32 = 0.0;
  var mx : f32 = -3.0e38;
  for (var q : u32 = 0u; q < 361u; q = q + 1u) {
    let v = ${ld('ppIn[base + q * 32u]')};
    sum = sum + v;
    mx = max(mx, v);
  }
  let mean = sum / 361.0;
  ppGp[wid.x * 96u + c] = mean;
  ppGp[wid.x * 96u + 32u + c] = mean * 0.5;
  ppGp[wid.x * 96u + 64u + c] = mx;
}
`;

  /* poolValue:同上,第三统计 = mean·0.15(quad) */
  K.poolValue = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct PoolParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> pv : PoolParams;
@group(0) @binding(1) var<storage, read>       pvIn  : array<${T}>;
@group(0) @binding(2) var<storage, read_write> pvGp  : array<f32>;

@compute @workgroup_size(32)
fn poolValue(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  if (wid.x >= pv.n) { return; }
  let c = lid.x;
  let base = wid.x * 361u * 32u + c;
  var sum : f32 = 0.0;
  for (var q : u32 = 0u; q < 361u; q = q + 1u) {
    sum = sum + ${ld('pvIn[base + q * 32u]')};
  }
  let mean = sum / 361.0;
  pvGp[wid.x * 96u + c] = mean;
  pvGp[wid.x * 96u + 32u + c] = mean * 0.5;
  pvGp[wid.x * 96u + 64u + c] = mean * 0.15;
}
`;

  /* lingFused:linear_g(gp)→g2[c] + gpbias(+p1)·bias2(scale/bias)→relu → act2
   * 一个 workgroup 一行,线程 = 通道;权重留 f32(头部,排除清单)。 */
  K.ling = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct PoolParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> lg : PoolParams;
@group(0) @binding(1) var<storage, read>       lgP1   : array<${T}>;
@group(0) @binding(2) var<storage, read>       lgGp   : array<f32>;
@group(0) @binding(3) var<storage, read>       lgW    : array<f32>;
@group(0) @binding(4) var<storage, read>       lgS2   : array<f32>;
@group(0) @binding(5) var<storage, read>       lgB2   : array<f32>;
@group(0) @binding(6) var<storage, read_write> lgOut  : array<${T}>;

@compute @workgroup_size(32)
fn ling(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  if (wid.x >= lg.n) { return; }
  let c = lid.x;
  let gpBase = wid.x * 96u;
  var g2 : f32 = 0.0;
  for (var j : u32 = 0u; j < 96u; j = j + 1u) {
    g2 = g2 + lgGp[gpBase + j] * lgW[j * 32u + c];
  }
  let s2 = lgS2[c];
  let b2 = lgB2[c];
  let base = wid.x * 361u * 32u + c;
  for (var q : u32 = 0u; q < 361u; q = q + 1u) {
    let v = (g2 + ${ld('lgP1[base + q * 32u]')}) * s2 + b2;
    lgOut[base + q * 32u] = ${st('max(v, 0.0)')};
  }
}
`;

  /* passFused:gp → linear_pass+b → relu → linear_pass2,单线程整行(MAC 3136)
   * gp/权重/输出全 f32(q 模式也不变:头部权重 + 输出缓冲)。 */
  K.pass = /* wgsl */ `
struct PoolParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> ps : PoolParams;
@group(0) @binding(1) var<storage, read>       psGp   : array<f32>;
@group(0) @binding(2) var<storage, read>       psW    : array<f32>;
@group(0) @binding(3) var<storage, read>       psB    : array<f32>;
@group(0) @binding(4) var<storage, read>       psW2   : array<f32>;
@group(0) @binding(5) var<storage, read_write> psOut  : array<f32>;

@compute @workgroup_size(32)
fn passHead(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n2 = gid.x;
  if (n2 >= ps.n) { return; }
  let gpBase = n2 * 96u;
  var h : array<f32, 32>;
  for (var c : u32 = 0u; c < 32u; c = c + 1u) {
    var acc : f32 = 0.0;
    for (var j : u32 = 0u; j < 96u; j = j + 1u) {
      acc = acc + psGp[gpBase + j] * psW[j * 32u + c];
    }
    h[c] = max(acc + psB[c], 0.0);
  }
  for (var o : u32 = 0u; o < 2u; o = o + 1u) {
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < 32u; k = k + 1u) {
      acc = acc + h[k] * psW2[k * 2u + o];
    }
    psOut[n2 * 2u + o] = acc;
  }
}
`;

  /* valueMlp:gp → v2+b → relu → (vh+b | misc+b),单线程整行。全 f32(头部)。 */
  K.valueMlp = /* wgsl */ `
struct PoolParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> vm : PoolParams;
@group(0) @binding(1) var<storage, read>       vmGp    : array<f32>;
@group(0) @binding(2) var<storage, read>       vmW2    : array<f32>;
@group(0) @binding(3) var<storage, read>       vmB2    : array<f32>;
@group(0) @binding(4) var<storage, read>       vmWv    : array<f32>;
@group(0) @binding(5) var<storage, read>       vmBv    : array<f32>;
@group(0) @binding(6) var<storage, read>       vmWm    : array<f32>;
@group(0) @binding(7) var<storage, read>       vmBm    : array<f32>;
@group(0) @binding(8) var<storage, read_write> vmVal   : array<f32>;
@group(0) @binding(9) var<storage, read_write> vmMisc  : array<f32>;

@compute @workgroup_size(32)
fn valueMlp(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n2 = gid.x;
  if (n2 >= vm.n) { return; }
  let gpBase = n2 * 96u;
  var a : array<f32, 64>;
  for (var j : u32 = 0u; j < 64u; j = j + 1u) {
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < 96u; k = k + 1u) {
      acc = acc + vmGp[gpBase + k] * vmW2[k * 64u + j];
    }
    a[j] = max(acc + vmB2[j], 0.0);
  }
  for (var o : u32 = 0u; o < 3u; o = o + 1u) {
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < 64u; k = k + 1u) {
      acc = acc + a[k] * vmWv[k * 3u + o];
    }
    vmVal[n2 * 3u + o] = acc + vmBv[o];
  }
  for (var o : u32 = 0u; o < 6u; o = o + 1u) {
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < 64u; k = k + 1u) {
      acc = acc + a[k] * vmWm[k * 6u + o];
    }
    vmMisc[n2 * 6u + o] = acc + vmBm[o];
  }
}
`;

  /* swiglu:hidden[i] = silu(gate[i])·gate[i+256](sigmoid 走 tanh 形式,Metal 安全) */
  K.swiglu = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct ElmParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> sw : ElmParams;
@group(0) @binding(1) var<storage, read>       swGate : array<${T}>;
@group(0) @binding(2) var<storage, read_write> swOut  : array<${T}>;

@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = sw.n * 361u * 256u;
  if (idx >= total) { return; }
  let m = idx / 256u;
  let i = idx % 256u;
  let a = ${ld('swGate[m * 512u + i]')};
  let s = 0.5 * (1.0 + tanh(0.5 * a));
  swOut[idx] = ${st(`a * s * ${ld('swGate[m * 512u + 256u + i]')}`)};
}
`;

  /* gemmSmall:小 O(头部 2 通道 policy、1 通道 ownership)的平凡 GEMM。
   * 输入 f16 激活(p1/act2/v1),权重 f32(头部),输出 f32(输出缓冲)。 */
  K.gemmSmall = /* wgsl */ `
${q ? 'enable f16;' : ''}
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> gs : GParams;
@group(0) @binding(1) var<storage, read>       gsIn  : array<${T}>;
@group(0) @binding(2) var<storage, read>       gsW   : array<f32>;
@group(0) @binding(3) var<storage, read_write> gsOut : array<f32>;

@compute @workgroup_size(64)
fn gemmSmall(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = gs.n * 361u * gs.o;
  if (idx >= total) { return; }
  let m = idx / gs.o;
  let o = idx % gs.o;
  var acc : f32 = 0.0;
  for (var k : u32 = 0u; k < gs.k; k = k + 1u) {
    acc = acc + ${ld('gsIn[m * gs.k + k]')} * gsW[k * gs.o + o];
  }
  gsOut[idx] = acc;
}
`;

  /* ---- GEMM 主力(按模式装配) ---- */
  if (mode === 'q') {
    /* trunk 大权重:int8 + scale,f16 io(绑定序:[u, in, W, S, out] / res 时 [u, in, W, S, res, out]) */
    K.gemmPlain = gemmQSource('plain', 'gemmPlain');
    K.gemmRes = gemmQSource('res', 'gemmRes');
    K.gemmQkv = gemmQkvSource('i8');
    /* f32 权重 + f16 io(q 模式下被排除张量与头部用) */
    K.gemmPlain32 = gemm32Source('plain', 'gemmPlain32');
    K.gemmBiasRelu32 = gemm32Source('biasrelu', 'gemmBiasRelu32');
    K.gemmRes32 = gemmQSource('res', 'gemmRes32', 'f32');
    K.gemmQkv32 = gemmQkvSource('f32', true, 'gemmQkv32');
  } else if (mode === 'f16w') {
    /* trunk 权重 f16,f16 io;头部仍 f32 权重(32 变体,f16 io) */
    K.gemmPlain = gemmQSource('plain', 'gemmPlain', 'f16');
    K.gemmRes = gemmQSource('res', 'gemmRes', 'f16');
    K.gemmQkv = gemmQkvSource('f16');
    K.gemmPlain32 = gemm32Source('plain', 'gemmPlain32');
    K.gemmBiasRelu32 = gemm32Source('biasrelu', 'gemmBiasRelu32');
  } else {
    K.gemmPlain = gemmSource('plain', 'gemmPlain');
    K.gemmRes = gemmSource('res', 'gemmRes');
    K.gemmBiasRelu = gemmSource('biasrelu', 'gemmBiasRelu');
    K.gemmQkv = gemmQkvSource('f32', false);
  }

  return K;
}

export default buildKernels;
