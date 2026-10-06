/* ============================================================
 * aethernn WebGPU session —— 自研引擎宿主(b8c96h3tfrs 专用)
 *
 * evalBatch 契约与 ort 路径完全同构(src/nn/session.js),另加两点:
 *   1. rows[i].sym(可省,缺省 0):8 对称改为 **GPU 侧置换** —— stem 内核按
 *      gather 表直接以变换后坐标取输入(search.js 不再做 CPU permuteSpatial,
 *      也不再为跨 await 快照而拷贝特征;本函数在首个 await 前同步完成全部
 *      上传,调用方在 promise settle 前复用特征缓冲是安全的)。
 *   2. 数据面直传:每行 queue.writeBuffer 直写批缓冲的固定槽位,无 CPU 端
 *      拼批拷贝(ort 路径的 sp.set 整批拷贝被消除)。
 *
 * 提交形态(RESEARCH §4.4):每 evalBatch 一个 encoder → 单 compute pass →
 * ~84 dispatch → 一次 submit → 5 段 copyBufferToBuffer 汇入一个 readback →
 * 一次 mapAsync → JS 后处理(与 ort 路径同口径,不改 search 消费侧)。
 * 权重单缓冲常驻一次上传;uniform 每批一次整体重写(仅 n 变化);
 * 全部 pipeline/bindgroup 初始化期预创建,运行期零对象分配(除结果数组)。
 * ============================================================ */
import {
  POS_LEN, HW, C_TRUNK, NUM_HEADS, HEAD_DIM, FFN, FFN_FUSED, QKV_FUSED,
  NUM_BLOCKS, SPATIAL_C, GLOBAL_C, HEAD_C, ATTN_SCALE, RMS_EPS,
  parseAewn, assertPlanMeta, makeStemTables,
} from './plan.js';
import KERNELS from './kernels.js';
import { calibrateMaxBatch } from '../calibrate.js';

const CAP = 32;                       // 缓冲容量(= 校准最大档;搜索 maxBatch ≤ CAP)
const SLOT = 256;                     // uniform 槽步长(minUniformBufferOffsetAlignment)
const WORDS = SLOT / 4;

const align = (x, a = 256) => Math.ceil(x / a) * a;

/* 评估串行锁:worker 里 think / score / estimate 可能异步交错,缓冲与
 * readback 都是共享态,evalBatch 必须逐批执行(ort 路径由 session 内部队列
 * 天然串行,这里显式补上)。 */
let chain = Promise.resolve();

export async function createAewnnSession(opt) {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('当前环境没有 WebGPU(需 Chrome/Edge 113+ 等启用 WebGPU 的浏览器)');
  }
  const status = opt.onStatus ?? (() => {});
  status('请求 WebGPU 设备');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter 不可用');
  /* valueMlp 内核需要 9 个 storage 绑定(默认上限 8);向适配器申请上限内
   * 的更高额度(原生 Dawn 支持 16,主流桌面/手机 GPU 亦普遍 >8)。 */
  const reqLimits = {};
  if (adapter.limits.maxStorageBuffersPerShaderStage > 8) {
    reqLimits.maxStorageBuffersPerShaderStage = Math.min(9, adapter.limits.maxStorageBuffersPerShaderStage);
  }
  const device = await adapter.requestDevice({ requiredLimits: reqLimits });
  device.addEventListener?.('uncapturederror', (e) => {
    console.error('[aewnn] GPU 错误:', e.error?.message || e.error);
  });

  status('加载权重 blob');
  let blob;
  if (opt.blob) {
    blob = opt.blob;                               // 测试直载(Node fetch 不支持 file://)
  } else {
    const url = opt.aewnUrl ?? opt.modelUrl.replace(/\.onnx$/, '.aewn');
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`权重 blob 加载失败(HTTP ${resp.status}): ${url}`);
    blob = await resp.arrayBuffer();
  }
  const { meta, w: weights } = parseAewn(blob);
  assertPlanMeta(meta);
  const wOf = (k) => {
    const v = weights.get(k);
    if (!v) throw new Error(`blob 缺张量 ${k}`);
    return v;
  };

  /* ==================== GPUBuffer 规划 ==================== */
  const u = (usage, size) => device.createBuffer({ usage, size });
  const ST_R = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const ST_RW = ST_R | GPUBufferUsage.COPY_SRC;
  const buf = {
    spatial: u(ST_R, CAP * SPATIAL_C * HW * 4),          // 行直传(NCHW 原始特征)
    global: u(ST_R, CAP * GLOBAL_C * 4),
    syms: u(ST_R, CAP * 4),
    weights: u(ST_R, blob.byteLength),
    trunk: u(ST_RW, CAP * HW * C_TRUNK * 4),
    normed: u(ST_RW, CAP * HW * C_TRUNK * 4),
    proj: u(ST_RW, CAP * HW * C_TRUNK * 4),
    qh: u(ST_RW, CAP * NUM_HEADS * HW * HEAD_DIM * 4),
    kh: u(ST_RW, CAP * NUM_HEADS * HW * HEAD_DIM * 4),
    vh: u(ST_RW, CAP * NUM_HEADS * HW * HEAD_DIM * 4),
    attn: u(ST_RW, CAP * HW * C_TRUNK * 4),
    gate: u(ST_RW, CAP * HW * FFN_FUSED * 4),
    hidden: u(ST_RW, CAP * HW * FFN * 4),
    p1: u(ST_RW, CAP * HW * HEAD_C * 4),
    actg: u(ST_RW, CAP * HW * HEAD_C * 4),
    v1: u(ST_RW, CAP * HW * HEAD_C * 4),
    act2: u(ST_RW, CAP * HW * HEAD_C * 4),
    gp: u(ST_RW, CAP * 3 * HEAD_C * 4),
    pol: u(ST_RW, CAP * HW * 2 * 4),
    pass: u(ST_RW, CAP * 2 * 4),
    val: u(ST_RW, CAP * 3 * 4),
    misc: u(ST_RW, CAP * 6 * 4),
    own: u(ST_RW, CAP * HW * 4),
  };
  const ropeCos = u(ST_R, HW * HEAD_DIM * 4);
  const ropeSin = u(ST_R, HW * HEAD_DIM * 4);
  const { zeroSlot, table } = makeStemTables();
  const stemTbl = u(ST_R, table.byteLength);

  device.queue.writeBuffer(buf.weights, 0, blob);
  device.queue.writeBuffer(ropeCos, 0, wOf('rope.cos'));
  device.queue.writeBuffer(ropeSin, 0, wOf('rope.sin'));
  device.queue.writeBuffer(stemTbl, 0, table);

  /* ==================== pipelines ====================
   * 入口名与 key 缺省一致;WGSL 保留字冲突的(如 pass → passHead)在此映射。 */
  const FN_OF = { pass: 'passHead' };
  const pipes = {};
  for (const [name, code] of Object.entries(KERNELS)) {
    pipes[name] = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: FN_OF[name] ?? name },
    });
  }

  /* 权重子区绑定:{buffer: weights, offset, size}(blob 内 256 对齐) */
  const W = (name) => {
    const v = wOf(name);
    return { buffer: buf.weights, offset: v.byteOffset, size: v.byteLength };
  };
  const B = (name) => ({ buffer: buf[name] });

  /* ==================== 执行计划(镜像 PyTorch forward) ====================
   * 每项 {pipe, params: {...}, bg: bindgroup entries, wg: (n) => [x,y,z], slot}。
   * 顺序即 dispatch 顺序;trunk 残差载体 A/B 两缓冲按半块交替(attn: A→B,ffn: B→A)。
   * entries[0] = 'U' 占位:创建 bindgroup 时替换为该 dispatch 自己的 uniform 槽
   * (256B 对齐偏移;若全体共享 offset 0,所有内核都会读到槽 0 的参数 —— 高危)。 */
  const dispatches = [];
  function add(pipe, params, entries, wg) {
    dispatches.push({
      pipe: pipes[pipe], entries, wg,
      slot: dispatches.length * WORDS,
      params: { n: 0, ...params },
    });
  }
  const elm = { n: 0 };
  const g2 = (k, o) => ({ n: 0, k, o });

  add('stem', { zeroSlot },
    ['U', B('spatial'), W('stem.conv_w'), W('stem.global_w'), B('global'),
     { buffer: stemTbl }, B('syms'), B('trunk')],
    (n) => [Math.ceil(n * HW * C_TRUNK / 64)]);

  for (let b = 0; b < NUM_BLOCKS; b++) {
    /* PyTorch TransformerAttentionBlock(attn{b}):norm1 → qkv → RoPE → attn → out_proj → 残差 */
    add('rms', elm, ['U', B('trunk'), W(`attn${b}.norm`), B('normed')],
      (n) => [Math.ceil(n * HW / 64)]);
    add('gemmQkv', g2(C_TRUNK, QKV_FUSED),
      ['U', B('normed'), W(`attn${b}.qkv`), B('qh'), B('kh'), B('vh')],
      (n) => [Math.ceil(QKV_FUSED / 16), Math.ceil(HW / 16), n]);
    add('rope', elm, ['U', B('qh'), B('kh'), { buffer: ropeCos }, { buffer: ropeSin }],
      (n) => [Math.ceil(n * NUM_HEADS * HW * (HEAD_DIM / 2) / 64)]);
    add('flash', elm, ['U', B('qh'), B('kh'), B('vh'), B('attn')],
      (n) => [Math.ceil(n * NUM_HEADS * HW / 64)]);
    /* gemmRes 绑定序:[uniform, in, W, out, res](out=新残差载体,res=块输入) */
    add('gemmRes', g2(C_TRUNK, C_TRUNK),
      ['U', B('attn'), W(`attn${b}.out`), B('proj'), B('trunk')],
      (n) => [Math.ceil(C_TRUNK / 16), Math.ceil(HW / 16), n]);

    /* PyTorch TransformerFFNBlock(ffn{b}):norm → gate SwiGLU → ffn2 → 残差 */
    add('rms', elm, ['U', B('proj'), W(`ffn${b}.norm`), B('normed')],
      (n) => [Math.ceil(n * HW / 64)]);
    add('gemmPlain', g2(C_TRUNK, FFN_FUSED),
      ['U', B('normed'), W(`ffn${b}.gate`), B('gate')],
      (n) => [Math.ceil(FFN_FUSED / 16), Math.ceil(HW / 16), n]);
    add('swiglu', elm, ['U', B('gate'), B('hidden')],
      (n) => [Math.ceil(n * HW * FFN / 64)]);
    add('gemmRes', g2(FFN, C_TRUNK),
      ['U', B('hidden'), W(`ffn${b}.ffn2`), B('trunk'), B('proj')],
      (n) => [Math.ceil(C_TRUNK / 16), Math.ceil(HW / 16), n]);
  }

  add('trunkFinal', elm, ['U', B('trunk'), W('trunkfinal.scale'), W('trunkfinal.bias'), B('normed')],
    (n) => [Math.ceil(n * HW * C_TRUNK / 64)]);

  /* PolicyHead */
  add('gemmPlain', g2(C_TRUNK, HEAD_C), ['U', B('normed'), W('policy.conv1p'), B('p1')],
    (n) => [Math.ceil(HEAD_C / 16), Math.ceil(HW / 16), n]);
  add('gemmBiasRelu', g2(C_TRUNK, HEAD_C),
    ['U', B('normed'), W('policy.conv1g'), B('actg'), W('policy.conv1g_b')],
    (n) => [Math.ceil(HEAD_C / 16), Math.ceil(HW / 16), n]);
  add('poolPolicy', elm, ['U', B('actg'), B('gp')], (n) => [n]);
  add('ling', elm, ['U', B('p1'), B('gp'), W('policy.gp_ling'),
    W('policy.bias2_scale'), W('policy.bias2_bias'), B('act2')], (n) => [n]);
  add('gemmSmall', g2(HEAD_C, 2), ['U', B('act2'), W('policy.conv2p'), B('pol')],
    (n) => [Math.ceil(n * HW * 2 / 64)]);
  add('pass', elm, ['U', B('gp'), W('policy.pass_w'), W('policy.pass_b'),
    W('policy.pass2'), B('pass')], () => [1]);

  /* ValueHead */
  add('gemmBiasRelu', g2(C_TRUNK, HEAD_C),
    ['U', B('normed'), W('value.conv1'), B('v1'), W('value.conv1_b')],
    (n) => [Math.ceil(HEAD_C / 16), Math.ceil(HW / 16), n]);
  add('poolValue', elm, ['U', B('v1'), B('gp')], (n) => [n]);
  add('valueMlp', elm, ['U', B('gp'), W('value.v2'), W('value.v2_b'),
    W('value.vh'), W('value.vh_b'), W('value.misc'), W('value.misc_b'),
    B('val'), B('misc')], () => [1]);
  add('gemmSmall', g2(HEAD_C, 1), ['U', B('v1'), W('value.own'), B('own')],
    (n) => [Math.ceil(n * HW / 64)]);

  /* ==================== uniform 模板 ====================
   * 字段布局与 WGSL struct 对齐:slot+0 = n(每批补丁);gemm 族 slot+1 = k、
   * slot+2 = o;stem slot+1 = zeroSlot。其余内核只读 n。 */
  const params = new Uint32Array(dispatches.length * WORDS);
  for (const d of dispatches) {
    if (d.pipe === pipes.gemmPlain || d.pipe === pipes.gemmRes
      || d.pipe === pipes.gemmBiasRelu || d.pipe === pipes.gemmQkv
      || d.pipe === pipes.gemmSmall) {
      params[d.slot + 1] = d.params.k;
      params[d.slot + 2] = d.params.o;
    } else if (d.pipe === pipes.stem) {
      params[d.slot + 1] = zeroSlot;
    }
  }

  /* ==================== uniform 缓冲与 bindgroups ====================
   * 每 dispatch 一个 256B 槽;bindgroup 静态偏移指向各自槽位(n 每批补丁,
   * writeBuffer 一次整体重写)。 */
  const uniformBuf = u(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, dispatches.length * SLOT);
  for (const d of dispatches) {
    const layout = d.pipe.getBindGroupLayout(0);
    const entries = d.entries.map((e, bi) => ({
      binding: bi,
      resource: e === 'U' ? { buffer: uniformBuf, offset: d.slot * 4, size: SLOT } : e,
    }));
    d.bg = device.createBindGroup({ layout, entries });
  }

  /* ==================== readback 布局 ==================== */
  const rbOff = { pol: 0 };
  rbOff.pass = align(rbOff.pol + CAP * HW * 2 * 4);
  rbOff.val = align(rbOff.pass + CAP * 2 * 4);
  rbOff.misc = align(rbOff.val + CAP * 3 * 4);
  rbOff.own = align(rbOff.misc + CAP * 6 * 4);
  const rbSize = align(rbOff.own + CAP * HW * 4);
  const readback = device.createBuffer({ usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, size: rbSize });

  const symScratch = new Uint32Array(CAP);

  /* ==================== evalBatch ====================
   * 串行锁 + 空闲快路径:引擎空闲时,上传在本次调用的同步前缀完成(调用方
   * 在返回的 promise settle 后即可复用特征缓冲 —— search 的环形槽位与
   * 「发射前必先收上一批」的批序共同保证这点);引擎忙时(如 think 进行中
   * 又来 score/estimate)调用入链排队,此时 rows 需自行持有有效数据
   * (worker 的单发行走全新 encode 缓冲,天然满足)。 */
  let busy = false;
  function evalBatch(rows) {
    const run = async () => {
      const n = rows.length;
      if (n > CAP) throw new Error(`evalBatch: 批 ${n} 超容量 ${CAP}`);
      if (n === 0) return [];
      /* 首个 await 前同步上传(契约:此后调用方可复用特征缓冲) */
      for (let i = 0; i < n; i++) {
        const r = rows[i];
        if (r.spatial.length !== SPATIAL_C * HW) throw new Error('spatial 尺寸不符');
        device.queue.writeBuffer(buf.spatial, i * SPATIAL_C * HW * 4, r.spatial);
        device.queue.writeBuffer(buf.global, i * GLOBAL_C * 4, r.global);
        symScratch[i] = r.sym ?? 0;
      }
      device.queue.writeBuffer(buf.syms, 0, symScratch, 0, n);
      for (const d of dispatches) params[d.slot] = n;
      device.queue.writeBuffer(uniformBuf, 0, params);

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const d of dispatches) {
        pass.setPipeline(d.pipe);
        pass.setBindGroup(0, d.bg);
        const [x, y, z] = d.wg(n);
        pass.dispatchWorkgroups(x, y ?? 1, z ?? 1);
      }
      pass.end();
      enc.copyBufferToBuffer(buf.pol, 0, readback, rbOff.pol, n * HW * 2 * 4);
      enc.copyBufferToBuffer(buf.pass, 0, readback, rbOff.pass, n * 2 * 4);
      enc.copyBufferToBuffer(buf.val, 0, readback, rbOff.val, n * 3 * 4);
      enc.copyBufferToBuffer(buf.misc, 0, readback, rbOff.misc, n * 6 * 4);
      enc.copyBufferToBuffer(buf.own, 0, readback, rbOff.own, n * HW * 4);
      device.queue.submit([enc.finish()]);

      await readback.mapAsync(GPUMapMode.READ);
      const rb = new Float32Array(readback.getMappedRange());
      const pol = rb.subarray(rbOff.pol / 4, rbOff.pol / 4 + n * HW * 2);
      const passOut = rb.subarray(rbOff.pass / 4, rbOff.pass / 4 + n * 2);
      const val = rb.subarray(rbOff.val / 4, rbOff.val / 4 + n * 3);
      const misc = rb.subarray(rbOff.misc / 4, rbOff.misc / 4 + n * 6);
      const own = rb.subarray(rbOff.own / 4, rbOff.own / 4 + n * HW);

      /* JS 后处理:与 ort 路径(src/nn/session.js)同口径 */
      const softPlus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));
      const res = new Array(n);
      for (let i = 0; i < n; i++) {
        const optimism = rows[i].optimism ?? 1.0;
        const policy = new Float32Array(HW);
        for (let p = 0; p < HW; p++) {
          const p0 = pol[(i * HW + p) * 2], pOpt = pol[(i * HW + p) * 2 + 1];
          policy[p] = (optimism !== 1.0) ? p0 + (pOpt - p0) * optimism : p0;
        }
        const pb = passOut[i * 2], pbOpt = passOut[i * 2 + 1];
        const policyPass = (optimism !== 1.0) ? pb + (pbOpt - pb) * optimism : pb;
        const l0 = val[i * 3], l1 = val[i * 3 + 1];
        const m = Math.max(l0, l1);
        const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
        const b = i * 6;
        res[i] = {
          policy,
          policyPass,
          winLoss: (e0 - e1) / (e0 + e1),
          ownership: Float32Array.from(own.subarray(i * HW, (i + 1) * HW)),
          scoreMean: misc[b] * 20,
          scoreStdev: softPlus(misc[b + 1]) * 20,
          scoreLead: misc[b + 2] * 20,
          shorttermScoreError: softPlus(misc[b + 5] * 0.5) * Math.sqrt(150),
          shorttermWinlossError: softPlus(misc[b + 4] * 0.5) * 0.5,
        };
      }
      readback.unmap();
      return res;
    };
    let p;
    if (!busy) {
      busy = true;
      p = run();                                   // 空闲:同步前缀(全部上传)立即执行
    } else {
      p = chain.then(run, run);                    // 忙:入链排队
    }
    const settle = () => { busy = false; };
    chain = p.then(settle, settle);
    return p;
  }

  /* 调试探针:跑一批,然后把指定内部缓冲拷回 CPU(仅测试用)。
   * specs = [{name, buf, floats}],rows 会先经 evalBatch 落定 GPU 状态。 */
  async function __debugCopy(specs, rows, upto = Infinity) {
    /* upto:只执行前 upto 个 dispatch(分段快照);缺省全量(结果同 evalBatch)。 */
    await (upto === Infinity ? evalBatch(rows) : __runPartial(rows, upto));
    const maxBytes = Math.max(...specs.map((x) => x.bytes ?? x.floats * 4));
    const tmp = device.createBuffer({
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      size: maxBytes,
    });
    const out = {};
    for (const sp of specs) {
      const bytes = sp.bytes ?? sp.floats * 4;
      const enc = device.createCommandEncoder();
      const src = sp.buf === 'uniform' ? uniformBuf : (sp.buf === 'stemTbl' ? stemTbl : buf[sp.buf]);
      enc.copyBufferToBuffer(src, 0, tmp, 0, bytes);
      device.queue.submit([enc.finish()]);
      await tmp.mapAsync(GPUMapMode.READ);
      const raw = tmp.getMappedRange();
      out[sp.name] = sp.u32
        ? Uint32Array.from(new Uint32Array(raw).subarray(0, sp.count ?? 0))
        : Float32Array.from(new Float32Array(raw).subarray(0, sp.floats ?? 0));
      tmp.unmap();
    }
    tmp.destroy();
    return out;
  }

  /* 只跑前 K 个 dispatch(调试用);uniform n 补丁与上传逻辑同 evalBatch。 */
  async function __runPartial(rows, upto) {
    const n = rows.length;
    for (let i = 0; i < n; i++) {
      device.queue.writeBuffer(buf.spatial, i * SPATIAL_C * HW * 4, rows[i].spatial);
      device.queue.writeBuffer(buf.global, i * GLOBAL_C * 4, rows[i].global);
      symScratch[i] = rows[i].sym ?? 0;
    }
    device.queue.writeBuffer(buf.syms, 0, symScratch, 0, n);
    for (const d of dispatches) params[d.slot] = n;
    device.queue.writeBuffer(uniformBuf, 0, params);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const [k, d] of dispatches.entries()) {
      if (k >= upto) break;
      pass.setPipeline(d.pipe);
      pass.setBindGroup(0, d.bg);
      const [x, y, z] = d.wg(n);
      pass.dispatchWorkgroups(x, y ?? 1, z ?? 1);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  /* ==================== 校准与返回 ==================== */
  let maxBatch = null;
  if (opt.calibrate !== false) {
    try {
      status('校准批次大小');
      maxBatch = await calibrateMaxBatch((rows) => evalBatch(rows));
      status(`校准完成(批上限 ${maxBatch})`);
    } catch {
      maxBatch = null;
    }
  }

  return {
    ep: 'webgpu-aewnn',
    evalBatch,
    maxBatch,
    meta,
    dispatchCount: dispatches.length,
    __debugCopy,
    __tbl: stemTbl,                     // 调试探针
    dispose() {
      for (const b of Object.values(buf)) b.destroy();
      uniformBuf.destroy();
      ropeCos.destroy(); ropeSin.destroy(); stemTbl.destroy(); readback.destroy();
      device.destroy?.();
    },
  };
}

export { ATTN_SCALE, RMS_EPS };
