/* ============================================================
 * aethernn 执行计划 —— b8c96h3tfrs 专用(plan 数据 + .aewn 解析)
 *
 * 自研 WebGPU 引擎的唯一计划表。范围裁决(docs/WEBGPU_ENGINE_RESEARCH.md §4.1):
 * 不是通用 ONNX 解释器,只支持 b8c96h3tfrs(v17 transformer,8×(attn+ffn));
 * 换模型 = 换权重 + (若架构变)改本表 —— 与「仅一个模型」的产品边界对齐。
 *
 * 算子顺序镜像 PyTorch forward(KataGo/python/katago/train/model_pytorch.py,
 * 蒸馏与导出的权威源):stem(conv_spatial + linear_global) →
 * 8 × TransformerAttentionBlock(norm1 → qkv → RoPE → attention → out_proj → 残差)
 * + 8 × TransformerFFNBlock(norm → gate SwiGLU → ffn2 → 残差) →
 * norm_trunkfinal(scale+bias,fixup) → relu → PolicyHead → ValueHead。
 * ONNX 导出图(dumponnx)与该顺序一致,已逐节点核对(423 节点)。
 *
 * 布局:trunk 全程 NHWC(b,361,96);q/k/v 由融合 GEMM 直接写成 head-major
 * (b,h,361,32),attention 输出写回 NHWC;输出 policy/ownership 按 NHWC 读回,
 * 由 session 的 JS 后处理出契约结果(契约口径见 src/nn/session.js)。
 *
 * 融合点(对齐 PyTorch 训练侧的 fused_qkv_proj / fused_gate_proj):
 *   qkv 单 GEMM(96→288)、ffn1+gate 单 GEMM(96→512)、out_proj/ffn2 的
 *   epilogue 内做残差加、头部 conv 的 BiasMask scale 折进权重(packer)、
 *   flash attention(在线 softmax,免 361×361 scores 物化,可开关回三段)。
 * ============================================================ */
import { N2 } from '../../engine.js';
import { buildStemGatherTable } from '../symmetry.js';

export const POS_LEN = 19;
export const HW = N2;                      // 361
export const C_TRUNK = 96;
export const NUM_HEADS = 3;
export const HEAD_DIM = 32;
export const FFN = 256;
export const FFN_FUSED = 512;              // ffn1 + gate 拼接后的出通道
export const QKV_FUSED = 288;              // q+k+v 拼接后的出通道
export const NUM_BLOCKS = 8;
export const SPATIAL_C = 22;
export const GLOBAL_C = 19;
export const HEAD_C = 32;                  // 头部 1×1 conv 出通道(p1/g/v1)
export const V2_C = 64;                    // value linear2 出通道

export const ATTN_SCALE = 0.1767766922712326;   // 1/√32(packer meta 互验)
export const RMS_EPS = 1e-6;                    // f32 下与 ONNX 常量一致

/* .aewn blob 解析。返回 { meta, w: Map(name → Float32Array 视图) }。 */
export function parseAewn(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  if (u8[0] !== 0x41 || u8[1] !== 0x45 || u8[2] !== 0x57 || u8[3] !== 0x4e) {
    throw new Error('.aewn: 魔数不符(不是 AEWN)');
  }
  const version = dv.getUint32(4, true);
  const dtype = dv.getUint32(8, true);          // 唯一形态:1 = i8f16(trunk 权重 int8 打包 u32 + 激活 f16;
                                                // 全 f32=0 与 f16 权重=2 已随 2026-10-08 拍板移除)
  const metaLen = dv.getUint32(12, true);
  const nTensors = dv.getUint32(16, true);
  if (version !== 1) throw new Error(`.aewn: 不支持的版本 ${version}`);
  if (dtype !== 1) throw new Error(`.aewn: 不支持的 dtype ${dtype}(引擎仅支持 i8f16=1)`);
  let off = 20;
  const dir = [];
  for (let i = 0; i < nTensors; i++) {
    const nameLen = dv.getUint32(off, true); off += 4;
    const name = new TextDecoder().decode(u8.subarray(off, off + nameLen)); off += nameLen;
    const ndim = dv.getUint32(off, true); off += 4;
    const dims = [];
    for (let d = 0; d < ndim; d++) { dims.push(dv.getUint32(off, true)); off += 4; }
    const tOff = Number(dv.getBigUint64(off, true)); off += 8;
    const nbytes = dv.getUint32(off, true); off += 8;   // nbytes + pad u32
    dir.push({ name, dims, tOff, nbytes });
  }
  const meta = JSON.parse(new TextDecoder().decode(u8.subarray(off, off + metaLen)));
  off += metaLen;
  const quant = meta.quant ?? {};
  const w = new Map();
  const dims = new Map();
  const range = new Map();                       // name → {byteOffset, byteLength}(GPU 子区绑定用)
  for (const t of dir) {
    if (t.tOff % 4 !== 0) throw new Error(`.aewn: 张量 ${t.name} 偏移未对齐`);
    range.set(t.name, { byteOffset: t.tOff, byteLength: t.nbytes });
    dims.set(t.name, t.dims);
    w.set(t.name, quant[t.name]
      ? new Uint32Array(buffer, t.tOff, t.nbytes >> 2)     // 4×int8 打包 u32
      : new Float32Array(buffer, t.tOff, t.nbytes >> 2));  // 头部/排除清单留 f32
  }
  return { meta, w, range, dims, dtype };
}

/* 计划与 blob 的一致性断言(架构不符启动即报错,RESEARCH §6 风险表最后一条)。 */
export function assertPlanMeta(meta) {
  const expect = {
    model: 'b8c96h3tfrs', posLen: POS_LEN, channels: C_TRUNK, heads: NUM_HEADS,
    headDim: HEAD_DIM, ffn: FFN, blocks: NUM_BLOCKS, spatialC: SPATIAL_C, globalC: GLOBAL_C,
  };
  for (const [k, v] of Object.entries(expect)) {
    if (meta[k] !== v) throw new Error(`.aewn 计划互验失败: ${k}=${meta[k]} 预期 ${v}`);
  }
  if (Math.abs(meta.attnScale - ATTN_SCALE) > 1e-9) throw new Error('.aewn attnScale 不符');
}

/* stem 卷积的对称 gather 表(初始化期一次;sym=0 时也走同一内核)。
 * zeroSlot = 0xFFFFFFFF 哨兵:内核/参考实现遇它按 0 贡献(等价零填充)。 */
export function makeStemTables() {
  const zeroSlot = 0xffffffff;
  return { zeroSlot, table: buildStemGatherTable(zeroSlot) };
}
