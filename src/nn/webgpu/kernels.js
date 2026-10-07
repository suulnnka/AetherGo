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
 * 权重形态唯一:i8f16(2026-10-08 起,全 f32 模式与 f16 权重模式移除)
 *   (INT8 报告 §4.1 方案 A 的引擎侧实现):
 *   - trunk 大权重 int8(4×int8 打包 u32,LSB 在前)+ per-oc f32 scale,
 *     tile 装载时反量化(shader 移位取字节,免 i8 类型);
 *   - 激活中间量 f16 存储、寄存器/共享内存/累加一律 f32
 *     (katago-webgpu 的「f16 storage + fp32 compute」形态;需要
 *     adapter 的 shader-f16 特性);
 *   - 头部(policy/value)按排除清单留 f32 权重,但输入仍是 f16 激活;
 *   - gp 池化向量与全部输出缓冲(pol/pass/val/misc/own)保持 f32
 *     (免 JS 侧 f16 解码;量小,带宽无关紧要)。
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
 * 两种权重形态(i8f16 唯一激活口径:f16 存储、f32 累加):
 *   gemm32Source   f32 权重,f16 io(头部 GEMM + 排除清单张量)
 *   gemmQSource    int8 权重(u32 打包+scale),f16 io(trunk GEMM;
 *                  wkind='f32' 变体用于排除张量的 res 形态)
 * 变体以 epilogue 区分:plain / res(残差加,f16)/ biasrelu(f32 bias)。 */

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
  /* 绑定:[0]u [1]in [2]W [3](S=i8) [3|4](res/bias) [末]out。
   * res 输入 f16(残差载体),bias 向量 f32(头部常量)。wkind='f32' 用于
   * 排除清单张量(f32 权重 + f16 io,逐层敏感度排除用)。 */
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
@group(0) @binding(2) var<storage, read>       geW   : array<${i8 ? 'u32' : 'f32'}>;
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

/* qkv 变体生成(wkind='i8' 量化 | 'f32' 排除清单):f16 io,输出按 o 散排到
 * head-major (b,h,361,32) 三缓冲 */
function gemmQkvSource(wkind, name = 'gemmQkv') {
  const quant = wkind === 'i8';
  return /* wgsl */ `
enable f16;
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> gq : GParams;
@group(0) @binding(1) var<storage, read>       gqIn  : array<f16>;
@group(0) @binding(2) var<storage, read>       gqW   : array<${quant ? 'u32' : 'f32'}>;
${quant ? `@group(0) @binding(6) var<storage, read>       gqS   : array<f32>;` : ''}
@group(0) @binding(3) var<storage, read_write> gqQh  : array<f16>;
@group(0) @binding(4) var<storage, read_write> gqKh  : array<f16>;
@group(0) @binding(5) var<storage, read_write> gqVh  : array<f16>;

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
    : `av = gqW[kA * gq.o + o];`}
    }
    gqAs[lid.x * 16u + lid.y] = av;
    let kB = t * 16u + lid.x;
    var bv : f32 = 0.0;
    if (kB < gq.k && m < 361u) { bv = f32(gqIn[(n2 * 361u + m) * gq.k + kB]); }
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
      gqQh[dst] = f16(acc);
    } else if (o < 192u) {
      let ok = o - 96u;
      let dst = ((n2 * 3u + ok / 32u) * 361u + m) * 32u + (ok % 32u);
      gqKh[dst] = f16(acc);
    } else {
      let ov = o - 192u;
      let dst = ((n2 * 3u + ov / 32u) * 361u + m) * 32u + (ov % 32u);
      gqVh[dst] = f16(acc);
    }
  }
}
`;
}

/* ==================== B 系(高批)GEMM 变体 ====================
 * n ≥ session 的 batchHi(缺省 8)时由 evalBatch 切换:每个 workgroup 仍固定
 * 一个 (O×M) 16×16 瓦片,但串 R 行批(z 网格 = ceil(n/R))—— W 瓦片每工作组建
 * 一次共享内存、跨 R 行复用,W 全局装载与装载栅栏摊薄 R 倍;输入瓦片逐行装
 * (不可免),累加器逐行独立。逐行 FMA 累加序与低批内核完全一致 → 同一行输出
 * 与低批路径逐位一致(wgsl-test 的 B 系对拍钉住)。绑定序与低批模板逐一镜像,
 * session 侧 bindgroup entries 可直接复用。 */
export const HI_R = 4;

function gemmHiSource({ epi, name, wkind = 'f32', ioF16, scatter = false }) {
  const i8 = wkind === 'i8';
  const T = ioF16 ? 'f16' : 'f32';
  const Wt = i8 ? 'u32' : 'f32';
  const ld = (expr) => (ioF16 ? `f32(${expr})` : expr);

  /* 绑定序逐一镜像低批模板(gemmSource/gemm32Source/gemmQSource/gemmQkvSource):
   * 非 scatter:i8 [U,in,W,S,(res|bias),out];非 i8 [U,in,W,out,(res|bias)]
   * scatter:   [U,in,W,qh,kh,vh,(S)] —— qh/kh/vh 即输出 */
  let src = `
${ioF16 || i8 ? 'enable f16;' : ''}
struct GParams { n: u32, k: u32, o: u32, pad: u32 };
@group(0) @binding(0) var<uniform> geo : GParams;
@group(0) @binding(1) var<storage, read>       gIn  : array<${T}>;
@group(0) @binding(2) var<storage, read>       gW   : array<${Wt}>;
`;
  let nb = 3;
  let resB = -1, biasB = -1, outB = -1;
  if (scatter) {
    src += `@group(0) @binding(3) var<storage, read_write> gQh  : array<${T}>;
@group(0) @binding(4) var<storage, read_write> gKh  : array<${T}>;
@group(0) @binding(5) var<storage, read_write> gVh  : array<${T}>;
`;
    nb = 6;
    if (i8) { src += `@group(0) @binding(${nb}) var<storage, read>       gS   : array<f32>;\n`; nb++; }
  } else if (i8) {
    /* i8:[U,in,W,S,(res|bias),out] —— extra 在 out 前(镜像 gemmQSource) */
    src += `@group(0) @binding(${nb}) var<storage, read>       gS   : array<f32>;\n`; nb++;
    if (epi === 'res') { resB = nb; src += `@group(0) @binding(${nb}) var<storage, read>       gRes : array<${T}>;\n`; nb++; }
    if (epi === 'biasrelu') { biasB = nb; src += `@group(0) @binding(${nb}) var<storage, read>       gBias : array<f32>;\n`; nb++; }
    outB = nb;
    src += `@group(0) @binding(${outB}) var<storage, read_write> gOut : array<${T}>;\n`;
  } else {
    /* 非 i8:[U,in,W,out,(res|bias)] —— out 在 extra 前(镜像 gemmSource/gemm32Source) */
    outB = nb;
    src += `@group(0) @binding(${outB}) var<storage, read_write> gOut : array<${T}>;\n`; nb++;
    if (epi === 'res') { resB = nb; src += `@group(0) @binding(${nb}) var<storage, read>       gRes : array<${T}>;\n`; nb++; }
    if (epi === 'biasrelu') { biasB = nb; src += `@group(0) @binding(${nb}) var<storage, read>       gBias : array<f32>;\n`; nb++; }
  }

  /* W 瓦片装载表达式(逐模板镜像) */
  const wLoad = i8
    ? `let lin = kA * geo.o + o;
      let byte = (gW[lin >> 2u] >> ((lin & 3u) * 8u)) & 0xFFu;
      av = (f32(byte) - select(0.0, 256.0, byte >= 128u)) * gS[o];`
    : `av = gW[kA * geo.o + o];`;

  /* R 行展开:输入瓦片装载 / FMA / epilogue(WGSL 无三元,越界行用 select 取 0) */
  const row = (rb) => `(b0 + ${rb}u)`;
  let inStores = '';
  let fmaBlocks = '';
  let epiBlocks = '';
  for (let rb = 0; rb < HI_R; rb++) {
    const base = `(${rb}u * 256u)`;
    inStores += `    gBs[${base} + lid.x * 16u + lid.y] = select(0.0, ${ld(`gIn[(${row(rb)} * 361u + m) * geo.k + kB]`)}, ${row(rb)} < geo.n && kB < geo.k && m < 361u);\n`;
    /* kk 手动展开(常量索引)+ gAs 步长 17(消 8-way bank conflict):
     * 内循环从「2 次 smem 读/FMA 且权重读 8 路冲突」变为「1 次无冲突 smem 读/FMA」 */
    let fma = '';
    for (let kk = 0; kk < 16; kk++) {
      fma += `      acc${rb} = acc${rb} + gAs[lid.x * 17u + ${kk}u] * gBs[${base} + ${kk}u * 16u + lid.y];\n`;
    }
    fmaBlocks += `    {\n${fma}    }\n`;
    const acc = `acc${rb}`;
    if (scatter) {
      epiBlocks += `  if (${row(rb)} < geo.n && o < geo.o && m < 361u) {
    if (o < 96u) {
      let dst = ((${row(rb)} * 3u + o / 32u) * 361u + m) * 32u + (o % 32u);
      gQh[dst] = ${ioF16 ? `f16(${acc})` : acc};
    } else if (o < 192u) {
      let ok = o - 96u;
      let dst = ((${row(rb)} * 3u + ok / 32u) * 361u + m) * 32u + (ok % 32u);
      gKh[dst] = ${ioF16 ? `f16(${acc})` : acc};
    } else {
      let ov = o - 192u;
      let dst = ((${row(rb)} * 3u + ov / 32u) * 361u + m) * 32u + (ov % 32u);
      gVh[dst] = ${ioF16 ? `f16(${acc})` : acc};
    }
  }\n`;
    } else {
      const epiLine = epi === 'res'
        ? `${acc} = ${acc} + ${ld(`gRes[(${row(rb)} * 361u + m) * geo.o + o]`)};`
        : epi === 'biasrelu'
          ? `${acc} = max(${acc} + gBias[o], 0.0);`
          : '';
      epiBlocks += `  if (${row(rb)} < geo.n && o < geo.o && m < 361u) {
    ${epiLine}
    gOut[(${row(rb)} * 361u + m) * geo.o + o] = ${ioF16 ? `f16(${acc})` : acc};
  }\n`;
    }
  }
  const accDecl = Array.from({ length: HI_R }, (_, i) => `  var acc${i} : f32 = 0.0;`).join('\n');

  src += `
var<workgroup> gAs : array<f32, 272>;    /* 步长 17:16×17 消 8-way bank conflict */
var<workgroup> gBs : array<f32, ${256 * HI_R}>;

@compute @workgroup_size(16, 16)
fn ${name}(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let b0 = wid.z * ${HI_R}u;
  let o  = wid.x * 16u + lid.x;
  let m  = wid.y * 16u + lid.y;
${accDecl}
  let nTiles = (geo.k + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * 16u + lid.y;
    var av : f32 = 0.0;
    if (o < geo.o && kA < geo.k) { ${wLoad} }
    gAs[lid.x * 17u + lid.y] = av;
    let kB = t * 16u + lid.x;
${inStores}    workgroupBarrier();
${fmaBlocks}    workgroupBarrier();
  }
${epiBlocks}}
`;
  return src;
}

export function buildKernels() {
  /* i8f16 唯一形态:激活中间量 f16 存储(f32 累加) */
  const T = 'f16';
  const ld = (expr) => `f32(${expr})`;            // 装载转换
  const st = (expr) => `f16(${expr})`;            // 存储转换
  const K = {};

  /* stem:对称 gather + conv3x3(22→96)+ linear_global 广播加 → NHWC trunk
   * out[n,q,oc] = Σ_{ic,d} W[oc,ic,d]·in[n,ic,inv_sym(q+δ)] + Σ_j Wg[oc,j]·g[n,j]
   * 盘外邻居以 zeroSlot 哨兵跳过(等价零填充)。压缩模式两矩阵都压缩(报告 58 层
   * 口径含 linear_global);stem32 = f32 权重 + f16 io(排除清单时用)。 */
  const stemSource = (wkind, name = 'stem') => {
    const i8 = wkind === 'i8';
    const W = i8 ? 'u32' : 'f32';
    const wld = (expr) => (wkind === 'f32' ? expr : `f32(${expr})`);
    return /* wgsl */ `
enable f16;
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
  K.stem = stemSource('i8');
  K.stem32 = stemSource('f32', 'stem32');

  /* rmsNorm:逐位置跨通道 RMS(eps=1e-6)× gamma。NHWC:线程 i 直接持 96 连续元素 */
  K.rms = /* wgsl */ `
enable f16;
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
enable f16;
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


  /* trunkfinal(fixup):x·scale[c]+bias[c] → relu(无归一化,ONNX norm_trunkfinal 同) */
  K.trunkFinal = /* wgsl */ `
enable f16;
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
enable f16;
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
enable f16;
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
enable f16;
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
enable f16;
struct ElmParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> sw : ElmParams;
@group(0) @binding(1) var<storage, read>       swGate : array<${T}>;
@group(0) @binding(2) var<storage, read_write> swOut  : array<${T}>;

/* workgroup 256:批 64 时网格 22,576(< 65535 单维上限;64 时会到 90,304 派发失效) */
@compute @workgroup_size(256)
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
enable f16;
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

  /* flashB:唯一注意力算子(2026-10-07 旧线程级 flash 算子与低/高批双派发移除)。
   * 线程 = query,零 smem;32 维点积拆 4 路部分和,打断「s += ·」的 32 拍
   * 依赖链(实测高批吞吐 ×2~3 的来源)。acc 更新天然 32 路独立。数值仅
   * 点积累加序改变,与旧 flash 同容差(wgsl-test 对拍)。 */
  const flashBSource = (name = 'flashB') => {
    const NA = 32;
    const accDecl = Array.from({ length: NA }, (_, i) => `    var a${i} : f32 = 0.0;`).join('\n');
    const qLoad = Array.from({ length: NA }, (_, i) => `    let q${i} = ${ld(`faQ[qBase + ${i}u]`)};`).join('\n');
    const dotLines = Array.from({ length: NA }, (_, i) => `      s${i % 4} = s${i % 4} + q${i} * ${ld(`faK[kBase + ${i}u]`)};`).join('\n');
    const accLines = Array.from({ length: NA }, (_, i) => `      a${i} = a${i} * corr + p * ${ld(`faV[kBase + ${i}u]`)};`).join('\n');
    const writeAll = Array.from({ length: NA }, (_, i) => `      faOut[outBase + ${i}u] = ${st(`a${i} * inv`)};`).join('\n');
    return /* wgsl */ `
enable f16;
struct ElmParams { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> fa : ElmParams;
@group(0) @binding(1) var<storage, read>       faQ   : array<${T}>;
@group(0) @binding(2) var<storage, read>       faK   : array<${T}>;
@group(0) @binding(3) var<storage, read>       faV   : array<${T}>;
@group(0) @binding(4) var<storage, read_write> faOut : array<${T}>;

@compute @workgroup_size(64)
fn ${name}(@builtin(workgroup_id) wid : vec3<u32>,
           @builtin(local_invocation_id) lid3 : vec3<u32>) {
  let idx = wid.x * 64u + lid3.x;
  let total = fa.n * 3u * 361u;
  if (idx >= total) { return; }
  let qi = idx % 361u;
  let h  = (idx / 361u) % 3u;
  let n2 = idx / (361u * 3u);
  let qBase = ((n2 * 3u + h) * 361u + qi) * 32u;
  let kHead = (n2 * 3u + h) * 361u * 32u;
${qLoad}
${accDecl}
  var m : f32 = -3.0e38;
  var l : f32 = 0.0;
  for (var ki : u32 = 0u; ki < 361u; ki = ki + 1u) {
    let kBase = kHead + ki * 32u;
    var s0 : f32 = 0.0;
    var s1 : f32 = 0.0;
    var s2 : f32 = 0.0;
    var s3 : f32 = 0.0;
${dotLines}
    var s : f32 = (s0 + s1) + (s2 + s3);
    s = s * 0.1767766922712326;
    let newMax = max(m, s);
    let corr = exp(m - newMax);
    let p = exp(s - newMax);
    l = l * corr + p;
${accLines}
    m = newMax;
  }
  let inv = 1.0 / l;
  let outBase = (n2 * 361u + qi) * 96u + h * 32u;
${writeAll}
}
`;
  };

  /* ---- GEMM 主力(i8f16 装配) ---- */
  /* trunk 大权重:int8 + scale,f16 io(绑定序:[u, in, W, S, out] / res 时 [u, in, W, S, res, out]) */
  K.gemmPlain = gemmQSource('plain', 'gemmPlain');
  K.gemmRes = gemmQSource('res', 'gemmRes');
  K.gemmQkv = gemmQkvSource('i8');
  /* f32 权重 + f16 io(被排除张量与头部用) */
  K.gemmPlain32 = gemm32Source('plain', 'gemmPlain32');
  K.gemmBiasRelu32 = gemm32Source('biasrelu', 'gemmBiasRelu32');
  K.gemmRes32 = gemmQSource('res', 'gemmRes32', 'f32');
  K.gemmQkv32 = gemmQkvSource('f32', 'gemmQkv32');
  /* B 系(高批,n ≥ session.batchHi 由 evalBatch 二选一) */
  K.gemmPlainB = gemmHiSource({ epi: 'plain', name: 'gemmPlainB', wkind: 'i8', ioF16: true });
  K.gemmResB = gemmHiSource({ epi: 'res', name: 'gemmResB', wkind: 'i8', ioF16: true });
  K.gemmQkvB = gemmHiSource({ epi: 'plain', name: 'gemmQkvB', wkind: 'i8', ioF16: true, scatter: true });
  K.gemmPlain32B = gemmHiSource({ epi: 'plain', name: 'gemmPlain32B', wkind: 'f32', ioF16: true });
  K.gemmBiasRelu32B = gemmHiSource({ epi: 'biasrelu', name: 'gemmBiasRelu32B', wkind: 'f32', ioF16: true });
  K.gemmRes32B = gemmHiSource({ epi: 'res', name: 'gemmRes32B', wkind: 'f32', ioF16: true });
  K.gemmQkv32B = gemmHiSource({ epi: 'plain', name: 'gemmQkv32B', wkind: 'f32', ioF16: true, scatter: true });

  /* 高批注意力 */
  K.flashB = flashBSource();

  return K;
}

export default buildKernels;
