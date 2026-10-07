/* 测试辅助:确保 .aewn blob 在场(仅 i8f16 一份;缺失时报错,不再兜底现做
 * —— 重打包需训练管线产出的 onnx,出库后由本地训练侧自行提供)。
 *
 * 2026-10-08 拍板:引擎仅支持 i8 权重 + f16 激活;fp32 golden blob 与
 * onnx 已出库(f16 权重版此前已撤销清出仓库与历史)。 */
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

export function ensureBlob(file) {
  const p = join(ROOT, 'models', file);
  if (!existsSync(p)) {
    throw new Error(`权重 blob 缺失:${p}(用训练管线产出 onnx 后按 training/pack_aewn.py 重打包)`);
  }
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

export function asArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}
