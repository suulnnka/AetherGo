/* 测试辅助:确保 .aewn blob 在场(缺失时调 packer 现做)。
 *
 * 仓库只入库两个发布产物:.i8.aewn(默认)与 .f16.aewn(回退)。
 * f32 blob(3.82MB)只是测试/golden 夹具,不入库,按需本地生成 ——
 * 计算本就是 f16,f32 权重存储没有发布意义(INT8 报告三步走的结论)。
 */
import { existsSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
export const PY = process.env.PYTHON_BIN ?? '/home/a/miniconda3/envs/bleed/bin/python';
export const ONNX = join(ROOT, 'models/b8c96h3tfrs_19.onnx');

export function ensureBlob(file, dtypeArgs = []) {
  const p = join(ROOT, 'models', file);
  if (!existsSync(p)) {
    execFileSync(PY, [join(ROOT, 'training/pack_aewn.py'), ONNX, p, '--dtype', dtypeArgs[0] ?? 'f32'],
      { stdio: 'inherit' });
  }
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

export function asArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}
