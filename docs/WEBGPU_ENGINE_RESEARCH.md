# 调研报告:放弃 ort-web,自研 WebGPU 推理引擎

> 2026-10-06 立项调研。背景:[NEURAL_PLAN](NEURAL_PLAN.md) §7 现行裁决「不自研 WebGPU 算子
> (onnxruntime-web 不够用时再评估,先假设够用)」。本文评估**现在自研**的动机、方案、成本与风险,
> 为维持或推翻该裁决提供依据;拍板前不动现状。
>
> **2026-10-06 当日落地(M0~M3 的 Node 侧全部完成,见文末「落地记录」)**:
> aethernn 已实现并通过全链路对拍,默认引擎已切换,ort-web 降级为 `?engine=ort` 逃生舱。
>
> 参照物:`/home/a/go/katago-webgpu`(KataGo fork,手写 WGSL WebGPU 后端,v17 全架构覆盖,
> 与 Eigen CPU 基准逐位对拍;其 WEBGPU_STATUS.md 是一份完整的性能工程日志)。
> 关联:[INT8_QUANT_RESEARCH.md](INT8_QUANT_RESEARCH.md) —— 自研引擎落地后,fp16/int8 权重
> 只是 packer 的一个选项,量化报告的 ort-web 算子风险整体消失。

---

## 1. 结论(TL;DR)

1. **值得做,且比想象中便宜**:模型只有一个、架构固定(v17 transformer),需要写的是
   ~14 个 WGSL 内核 + 一个硬编码执行计划,不是通用推理引擎。katago-webgpu 已把全部内核写好、
   验证好(MIT 同源),WGSL 可以近乎逐字移植 —— 主要工程量变成「移植 + 验证链」,
   估计 **1~1.5 周**,不是「重写 onnxruntime」。
2. **动机的量级(实测,jsdelivr 2026-10-06)**:ort-web 每次冷加载 = `ort.all.min.mjs`
   0.82MB(gzip 187KB)+ `ort-wasm-simd-threaded.jsep.wasm` 28.31MB(**gzip 后 5.35MB**)
   ≈ **5.5MB 传输,是模型本体(3.79MB)的 1.45 倍**,还绑死 CDN(离线不可用、单点)。
   自研引擎 = ~30-60KB JS + WGSL 字符串,**运行时依赖归零**。
3. **接口面已经预留好了**:session.js 的 `evalBatch` 契约就是为换后端设计的
   (session.js:67-68 注释原话「换自研后端时本函数不改(只依赖 evalBatch 契约)」);
   `calibrateMaxBatch` 原样复用;features/search/worker 零改动。换芯 = 重写一个文件。
4. **最大风险不是写错内核,而是浏览器侧性能不达预期**:katago-webgpu 的性能结论来自
   原生 Dawn + 软件 Vulkan(GB10),浏览器(Chrome 同用 Dawn,可信度高;Safari/WebKit 另算)
   的 dispatch 开销、管线编译行为必须实测。因此迁移计划(§5)把「ort-web 基线测量」放在
   M0,把「性能 ≥ ort-web」定为切换闸门 —— 不达标就维持现状,ort-web 并不碍事。
5. **两个报告互为条件**:先自研后量化是更优顺序 —— packer 直接产 fp16/int8 权重,
   一步拿到 2~4× 体积收益,不必在 ort-web 上做一轮量化再迁移一遍。

---

## 2. 现状盘点:ort-web 在本项目的真实成本

| 维度 | 事实 | 影响 |
|---|---|---|
| 下载体积 | 187KB(mjs,gzip)+ 5.35MB(jsep.wasm,gzip)= 5.5MB/冷加载 | 占总冷加载(5.5+3.79≈9.3MB)的 59%;N4 体积闸门的最大单项 |
| CDN 依赖 | `cdn.jsdelivr.net` 懒加载(session.js:40,55) | 离线/CDN 故障/审查网络 = 引擎不可用;WebOS 场景(纯前端离线 OS)尤其冲突 |
| 版本耦合 | 锁 1.30.0;`.mjs` 具名导出、wasmPaths、JSEP 算子覆盖(NEURAL_PLAN §9 已核实过一次 Sin/Cos kernel 存在性) | 每次升级都要重核算子覆盖;升级本身无收益驱动,但不升级则 bug/性能修复也拿不到 |
| 性能黑盒 | 通用图解释器:每个 ONNX 节点一次 JSEP 分发;本图 423 节点(融合后 ort 内部仍 ~200+ dispatch);无单 pass/单 submit 控制;批校准只能整图测 | 我们无法做 katago-webgpu 已证明的大头优化(§3.3):单 compute pass、单 submit、单 readback、内核融合 |
| 通用性错配 | 产品边界:仅一个模型、仅 WebGPU、无回退。通用引擎的「任意 ONNX」能力 100% 用不上 | 为用不到的灵活性付 5.5MB + 黑盒税 |

**已知的够用性风险**(NEURAL_PLAN §8 风险表第一条「ort-web WebGPU 对新导出的 ONNX 图兼容性差」):
每代学生出炉都要人工核一轮 ort-web 兼容性 —— 自研引擎把这条风险从「每代重检」变成
「架构固定、一次验证」。

---

## 3. 参照物 katago-webgpu:哪些直接可搬

### 3.1 它是什么

KataGo fork(saigo-online/katago-webgpu),给 KataGo 加了第三个 NN 后端:手写 WGSL,
经 Dawn 跑原生、经 emdawnwebgpu 跑 WASM 浏览器。**v17 全架构**(conv/nbt/transformer、
RMSNorm、RoPE、GQA、SwiGLU、optimism/q-value)与 Eigen CPU 参考**逐位一致**(runnnlayertests
14/14 + evalsgf 全网 byte-identical)。AetherGo 是纯 JS(拍板:不上 WASM),
所以可搬的是 **WGSL 内核 + 架构模式 + 验证方法论**,不是它的 C++ 宿主。

### 3.2 内核清单 × b8c96h3tfrs 需要面

katago-webgpu 的 25 个 WGSL entry point(webgpukernels.cpp)中,本模型需要 ~14 个:

| 内核 | 作用 | b8c96h3tfrs 需要? | 备注 |
|---|---|---|---|
| `conv2dNCHW` | 3×3 卷积 | ✅ 仅 stem(1 层) | 直接卷积即可;Winograd 不值得(只 1 层 3×3,占全网计算量 ~1.3%) |
| `conv1x1NCHW`/`proj1x1` | 1×1 卷积(头) | ✅ | 头部 4 个 1×1 conv |
| `tiledGemm`(+RT 变体) | 共享内存平铺 GEMM | ✅ 核心主力 | qkv/out/ffn 全是 (361,96)×(96,·);权重已 Wnhwc,**零转置** |
| `matMulBiasAct` | GEMM+bias+激活融合 | ✅ | 头部 linear;ffn1 可带 silu 前置 |
| `rmsReduceSpatial`/`rmsApplySpatial` | RMSNorm 归约/应用 | ✅ ×16 | 逐 token 对 C=96 归约;18 个 ReduceMean 里的 16 个 |
| `ropeApply` | 2D RoPE | ✅ ×16 | **改良**:cos/sin 表打包期预算好(θ=100,head_dim32,19×19),不再图内现算 —— make_rope_ongraph.py 的职责被 packer 吸收 |
| `attnScores`/`attnSoftmax`/`attnOutput` | 注意力三段 | ✅(起步用) | O(seq²) 在 361 token、3 头下完全可行(其结论:"fine at 19×19") |
| `flashAttention` | 融合注意力(在线 softmax) | ✅(M2 升级项) | 3 dispatch → 1;其 A/B 有开关 |
| `swigluGate` | SwiGLU | ✅ ×8 | |
| `activate` | relu/silu 等 | ✅ | 头部 6 个 Relu |
| `addInPlace` | 残差加 | ✅ | |
| `globalPoolMeanMax`/`globalPoolValueHead` | 全局池化(value 头 gpmean/gpconcat) | ✅(value 头用 mean 路径) | subgroups 特性门控,可后置 |
| `scaleBiasMaskAct` | BN 融合 | ❌ | 本模型无 BN(RMSNorm 架构) |
| `winograd*` 四件套 | 3×3 Winograd | ❌ | 同上,只有 1 层 3×3 |

### 3.3 性能经验(全部有 A/B 数据,直接继承)

1. **体制判断:小网是延迟受限,不是算力受限**。b6c96 @ batch6:~90 dispatch,
   固定 ~7ms GPU 开销(Dawn 在依赖性 dispatch 间插 storage-hazard barrier),
   batch 3 ≈ batch 15 的墙钟 —— **吞吐靠大批摊薄固定延迟**(这正是 AetherGo 批校准
   已在做的事,方向互证)。
2. **最大单笔收益是提交合并,不是 FLOPs**:每评估一次 `Submit` + 一次合并 readback,
   替代 ~50 次 submit + 5 次阻塞往返 → 616→765 nnEvals/s(+24%);单 compute pass 再 +9%;
   1×1 fast conv +31%。**这三条是纯宿主架构,搬 WGSL 时免费继承**。
3. **资源复用**:权重常驻一次上传;中间量/uniform 走 BufferPool(坑:按「≤2× 大小复用槽」,
   否则不同 batch size 会累积 ~80 个 buffer,浏览器 VRAM 爆炸 —— 它踩过)。
4. **RoPE 表一次算好**(每 handle 一次,别每 eval 重传 —— 它也踩过)。
5. **fp16 的血泪**:fp16 storage + fp32 compute(`alias STO` 单源双精度)是对的形态;
   host 侧 float→half 必须 **clamp 到 ±65504**(KataGo 的 1e9 off-board 哨兵会变 0×Inf=NaN);
   旧 g170 网 trunk 溢出 fp16 需 scale8 重标定,而 **v17 silu/modern 网 fp16-stable** ——
   本学生模型是 v17 silu,但 kata1 冷启动的实际激活范围未测,先跑 INT8 报告 §5.3 的
   激活范围扫描再决定(同一份工具两用)。
6. **selective fp32 heads**:fp16 模式下 norm 归约/池化/头 matmul 留 fp32(学 TensorRT
   路径,实测 0.08% 胜率误差 + 2.4× 吞吐)—— 我们的可选后置项。
7. **subgroups 归约**:feature-gated(探测到才编译带 `enable subgroups;` 的内核),
   无该特性的设备回落共享内存树归约 —— 同一内核双版本并存的范式值得照抄。
8. **cooperative matrix(张量核)**:实验特性,不进依赖;列为未来免费升级。

### 3.4 踩坑记录(逐条可核)

- readback buffer 必须 `CopySrc` 能力,否则 MapAsync 读到全零(症状:策略退化为常数);
- 输入布局:native 侧曾因 NHWC/NCHW 错配把每手评估变成乱码垃圾且**性能正常所以无人发现** ——
  全网对拍(§5 M2)是唯一防线;
- `requireExactNNLen` 口径:先验 mask 全 1 即可断言后**跳过 mask 乘法**(本模型恒喂 1,
  session.js:137,是个白送的小优化,保留断言防呆);
- fp32 与 CPU 参考的浮点累加序差异是「预期内不可消除项」(其 b10c128:Win 73.51c vs 73.52c),
  闸门要按量级定,不追逐位。

---

## 4. 自研引擎设计(暂名 aethernn)

### 4.1 范围裁决(建议)

- **不是通用 ONNX 解释器**:执行计划硬编码为 JS 数据表(层清单由 §4.2 的 ONNX 结构导出,
  人工固化 + 一致性断言:packer 产出的权重 SHA 与计划表里的形状清单互验)。
  换模型 = 换权重 + (若架构变)改计划表 —— 与产品边界「仅一个模型」严格对齐。
- ONNX **保留为训练侧权威格式**:dumponnx 管线照旧;packer(`training/pack_aewn.py`)
  消费 ONNX → 私有权重 blob(`.aewn`)。训练侧零改动,且现有「rope-graph 瘦身」脚本
  的职能(预算 cos/sin 表)自然并入 packer。

### 4.2 执行计划(从 ONNX 实测结构导出)

```
输入:spatial(N,22,19,19) f32 → 上传;global(N,19) f32 → 上传
stem:  conv3x3 22→96(直接卷积,单层不配 Winograd)
       linear_global 1×1 conv 19→96 → broadcast 加 trunk bias      [2 dispatch]
trunk ×8 blocks:
       rmsReduce+rmsApply(norm1)                                   [1~2]
       qkv:三矩阵在 packer 里拼成 (96,288) → 单 tiledGemm          [1]
       ropeApply(q,k;cos/sin 表常量 buffer)                        [1]
       attnScores(q·kᵀ,×1/√32)→ attnSoftmax → attnOutput(·v)     [3]
         —— M2 起换 flashAttention 单内核 [1],省 2 dispatch
       out_proj tiledGemm + 残差 addInPlace                        [2]
       rmsReduce+rmsApply(norm)                                    [1~2]
       ffn1 tiledGemm 96→256;gate GEMM;swigluGate;ffn2 256→96     [4]
       残差 addInPlace                                             [1]
       (每块 ~13~14 dispatch;8 块 ~108)
final: norm_trunkfinal(scale+bias)                                  [1]
heads: policy conv1p 1×1 96→32 + relu + conv2p 32→2 → OutputPolicy(C=2:主/乐观)
       policy 全局路(conv1g/linear_g/gpbias)+ linear_pass → OutputPolicyPass
       value conv1 + gpmean 池化 + concat + linear2 + linear_valuehead → OutputValue(3)
       linear_miscvaluehead → OutputScoreValue(6);conv_ownership → OutputOwnership(1)
                                                                    [~12]
合计 ~125 dispatch(全融合后 ~110);单 compute pass、单 Submit、单 readback。
```

数值口径全部继承 session.js 已验证的后处理(×20/softplus scoreValue、乐观插值在 JS、
winLoss 双 logits 归一)—— **readback 后的 JS 后处理一行不改**,先把 GPU 侧替换干净。

### 4.3 权重格式 `.aewn`(packer 产出)

- 头(魔数+版本+形状清单+SHA)+ 单一大 payload:全部权重按计划表布局预排
  (qkv 拼接、matmul K 主序、通道 padding 到 8/16 对齐 tiledGemm);
- **dtype 即量化选项**:`--dtype f32|f16|int8`(int8 = 权重 int8+per-channel scale,
  WGSL 加载路径反量化或加载时展开 —— INT8 报告 §4.5);
- RoPE cos/sin 表(361×16×2 f32 ≈ 46KB)由 packer 预计算写入 —— 删掉图内 Sin/Cos 子图;
- 体积预期:f32 版 ≈3.8MB(持平),f16 版 ≈1.9MB,int8 版 ≈0.95MB(+计划表+packer 自身 ~0)。

### 4.4 GPU 资源与宿主结构

```
src/nn/webgpu/
  device.js      adapter/device 探测(shader-f16/subgroups 特性位上报)
  kernels.js     WGSL 源字符串(从 katago-webgpu webgpukernels.cpp 移植,MIT 注明)
  plan.js        层计划表(§4.2)+ 权重绑定布局
  session.js     createSession 实现:packer blob → buffers → evalBatch
```

- 权重:单一大 GPUBuffer,加载时一次 writeBuffer 常驻;
- 中间量:BufferPool(≤2× 槽复用,size 按 maxBatch 预留);
- uniform:一块大 uniform buffer + dynamic offset 环(零每 dispatch 分配);
- 提交:每 evalBatch 一个 encoder → 一个 pass → N dispatch → 一次 submit →
  readback buffer(**CopySrc**,§3.4)一次 map → JS 后处理 → 返回同构结果;
- batch:`calibrateMaxBatch`(session.js:80)原样复用 —— 各档吞吐现测选最优 90% 最小批,
  与 ort-web 完全同口径,还天然成为两后端 A/B 的标准测速器;
- 精度:fp32 先行(对拍闸门全绿)→ fp16 storage/fp32 compute 作为特性开关
  (前置条件:激活范围扫描确认无 ±65504 溢出,INT8 报告 §5.3 工具两用)。

### 4.5 接口与切换面

- `src/nn/session.js` 保留为门面:`createSession` 内按构建期常量选 `ort` 实现或 `aethernn`
  实现;`evalBatch` 契约、返回字段、maxBatch 语义一字不动 → search/worker/难度四档零改动;
- 迁移期两实现并存(供 A/B 与回滚),默认切换稳定一版后删除 ort 路径
  (与「不做 WASM 回退」裁决一致:并存是开发期脚手架,不是运行时回退);
- Worker 里跑 WebGPU:Chrome 113+ 支持(device 在 worker 内请求),现有 nn-worker 结构不变。

---

## 5. 验证与迁移计划(M0~M4)

| 里程碑 | 内容 | 闸门 | 工作量 |
|---|---|---|---|
| **M0 基线** | ort-web 吞吐基线:对弈页 console 挂 calibrateMaxBatch 日志,记录各档 rows/ms、加载耗时、内存(目标设备:桌面 Chrome + 一台中端 Android);golden dump:3 个固定局面(空盘/激战/官子)× 5 输出,ort CPU fp32 落盘 | 基线数字入库,后面对比 | 0.5 天 |
| **M1 内核** | packer + ~14 内核移植 + **逐内核对拍**:每内核输出 vs ort CPU 中间张量(golden 切段),fp32 容差 1e-5 | 每内核全绿 | 2~3 天 |
| **M2 全网** | 整网对拍:2,048 局面(复用 INT8 报告 §5.3 的回归集)policy KL / value MAE / ownership MAE vs ort CPU;修到:KL < 1e-4、winLoss MAE < 1e-4、ownership MAE < 1e-3(fp32 GPU vs CPU 累加序差异量级) | 对拍全绿;nn-e2e 完整自对弈通过 | 1~2 天 |
| **M3 集成** | session 换芯(门面开关);现有测试全绿(engine/features/katago-align/worker-smoke/nn-e2e/nn-match);浏览器 A/B:aethernn vs ort-web 各档 rows/ms、加载耗时、maxBatch、内存 | **性能 ≥ ort-web×0.9(任一目标设备);数值闸门同 M2**;中端手机抽查 | 1~2 天 |
| **M4 收尾** | 默认切 aethernn;ort-web 依赖删除(CDN 引用、session ort 路径);README「不做」清单与 NEURAL_PLAN §7 裁决回写;fp16 packer 选项接力 INT8 报告 | 全部测试 + 对弈页人工 smoke | 0.5 天 |

总计 **~1~1.5 周**(全职口径;M1 是大头,但内核是「移植+对拍」不是「发明」)。

## 6. 风险登记表

| 风险 | 评估 | 缓解 |
|---|---|---|
| 浏览器性能不达 ort-web | 中。katago-webgpu 数据来自原生 Dawn;Chrome 亦用 Dawn(内核编译器同源,可信度高),但 dispatch 开销/管线缓存行为有差 | M0 先测基线;M3 性能是**切换闸门**而非愿望;不达标维持现状,零损失退出 |
| WGSL 移植引入数值 bug | 中。内核逐字移植但宿主全换(布局/binding) | M1 逐内核对拍 + M2 全网对拍;katago-webgpu 的坑位清单(§3.4)逐条对照 |
| Safari/WebKit 兼容 | 低(产品边界已限定 WebGPU 浏览器,N4 才圈设备清单) | WGSL 用保守子集(不用 subgroup 特性门控之外的任何扩展);特性探测上报 |
| shader 首用卡顿 | 低-中(管线编译) | 加载期预热全部 pipeline(createComputePipeline 预编译);首推理丢弃已有惯例(calibrateMaxBatch 预热) |
| 单模型硬编码绑架未来 | 低(§2 已裁决升级路径:b14c192 下一代需要改计划表) | 计划表是数据不是代码;packer 产形状清单互验,架构不符启动即报错 |
| 工程分散注意力(N3 训练并行) | 中 | M0/M1 与训练无依赖可并行;M2 起才需要学生模型在线 |

## 7. 被拒备选(与理由)

| 备选 | 结论 | 理由 |
|---|---|---|
| 维持 ort-web | 保留为默认直到 M3 闸门 | 5.5MB + CDN + 黑盒是长期税,但当下能跑;不是非换不可,是换了明显更好 |
| ort-web + fp16 模型(不换引擎) | 便宜的一半 | 拿到 2× 模型体积,拿不到 5.5MB 运行时体积与离线能力;作为 M0 顺带验证项 |
| 直接用 katago-webgpu 的 WASM 产物(kataeval) | **拒** | 违反「纯 JS,不上 WASM」拍板(NEURAL_PLAN §0);且其 wasm 含完整 KataGo 搜索,与本产品「JS 搜索」架构冲突;其价值在于 WGSL 内核与经验,已吸收 |
| WebNN | 观察 | 浏览器覆盖与算子成熟度不足,不进依赖;若未来标准化可再评估 |

## 8. 拍板请求

1. 是否立项 aethernn(§5 M0-M4,~1~1.5 周,退出点 M3);
2. 若立项:顺序建议 **先 M0+M1(不碰线上)**,与 N3 训练并行,M2 起占用发布流程;
3. 若暂不立项:至少落地 M0 基线测量(0.5 天,给未来的决策留数据),
   并把 fp16 模型(半天)作为 ort-web 路线上的独立小改进先做。

---

## 9. 落地记录(2026-10-06,立项当日完成 Node 侧全部里程碑)

### 9.1 交付物

```
training/pack_aewn.py        packer:ONNX → .aewn(qkv/gate 拼接、BiasMask scale 折叠、
                             头部 conv 权重转置 [outC][inC]→[inC][outC]、RoPE 表打包期预算)
models/b8c96h3tfrs_19.aewn   3.82MB(与 onnx 持平;头+目录+meta+256B 对齐 payload)
src/nn/webgpu/plan.js        计划表 + blob 解析 + 计划互验(架构不符启动即报错)
src/nn/webgpu/kernels.js     15 个 WGSL entry point(stem/rms/gemmQkv/rope/flash/
                             gemmPlain|Res|BiasRelu/swiglu/trunkFinal/pool×2/ling/pass/valueMlp/gemmSmall)
src/nn/webgpu/session.js     宿主:单 pass 84 dispatch、单 submit、单 readback 一次 mapAsync、
                             权重单缓冲常驻、uniform 256B 槽/每 dispatch、行直传 writeBuffer
src/nn/webgpu/cpuref.js      CPU 参考解释器(与 WGSL 逐算子同构,测试专用)
src/nn/symmetry.js           SYM8 抽出共用 + stem gather 表(8 对称 GPU 侧置换)
src/nn/session.js            门面:默认 aewnn,?engine=ort 切回(CPU 对称置换,契约同构)
src/nn/calibrate.js          批校准抽出(两后端共用同一测速器)
src/nn/search.js             零拷贝环形槽特征 + sym 随行下发;eval-cache 键并入 sym
```

### 9.2 执行计划与融合(对照 §4.2 的预期)

算子顺序**逐块镜像 PyTorch forward**(`model_pytorch.py` TransformerAttentionBlock/
TransformerFFNBlock,ONNX 423 节点已逐节点核对):stem → 8×(attn: norm→qkv→RoPE→
attention→out_proj+残差 ‖ ffn: norm→gate SwiGLU→ffn2+残差) → trunkfinal → policy/value 头。
423 节点 → **84 dispatch**,全部融合点兑现:qkv 单 GEMM(与训练侧 fused_qkv_proj 同构)、
ffn1+gate 单 GEMM、残差入 GEMM epilogue、BiasMask scale 折权重、RoPE 表打包期预算
(make_rope_ongraph.py 的图内子图职责被 packer 吸收)、flash attention 免 S² 物化、
linear_g/pass/value MLP 各单 dispatch。

### 9.3 对拍数据(M1/M2 闸门,全部通过)

| 对拍 | 闸门 | 实测 |
|---|---|---|
| cpuref vs ort CPU golden(18 例 × 6 字段,含 sym×λ 混批) | fp32 累加序量级 | policy **1.4e-5**,winLoss 1.0e-6,ownership 1.9e-6(M2 要求 <1e-4/<1e-4/<1e-3) |
| WGSL vs cpuref(12 行混批满链) | 同上 | policy 2.7e-5,winLoss 8.3e-7,ownership 2.0e-6 |
| 分段(checkpoint 张量 32 处 × 逐块残差 16 处) | ≤3.7e-5 | 最大 3.7e-5(trunkfinal),16 块单调漂移 ≤5.7e-5 |
| 现有测试(engine/features/katago-align/worker-smoke/nn-e2e) | 全绿 | 全绿(nn-e2e 361 手完整终局) |

对拍链路:`test/aewnn-cpuref-test.mjs`(golden 闸门)→ `test/aewnn-wgsl-test.mjs`(内核闸门,
Dawn node 绑定 `npm i --no-save webgpu`)→ `test/aewnn-stage-diff.mjs` / `aewnn-gpu-probe.mjs`
(分段定位工具,回归时用)。

### 9.4 与原计划的偏差(实测修正)

1. **头部 conv 权重布局**:trunk 的 `Wnhwc` 已是 [inC][outC] k 主序,但头部 1×1 conv 仍是
   Conv 布局 [outC][inC] —— packer 统一转置(§4.2 未预见,ort 黑盒下无从发现)。
2. **RoPE 预算表与 PyTorch 的关系**:ONNX 图内 Sin/Cos 子图与 `precompute_freqs_cos_sin_2d`
   的 repeat_interleave 语义已逐位验证等价(交换 sin 符号折进 swapidx),packer 直接按
   PyTorch 公式重建,`make_rope_ongraph.py` 退役为历史记录。
3. **多行语义**:内核全部按批内行距寻址;CPU 参考解释器首版逐行传参导致越界读 NaN ——
   已修(输入拼批),并成为「内核批布局」的回归锚。
4. **性能数字未出**:开发机(WSL2)Dawn 落在 llvmpipe 软件渲染,`test/aewnn-bench.mjs`
   只有软件下限;**M3 的真机 A/B(≥ort-web×0.9 闸门)仍待浏览器实测**,切换闸门不豁免。

### 9.6 量化落地(同日晚;i8f16 = INT8 报告 §4.1 方案 A 的引擎侧实现)

- **完全对齐 quant_explore 研究**(/home/a/go/trainrun/quant/,第 40 批权重
  s68320512 重跑确认):全部 58 层 trunk(conv_spatial、linear_global、attn q/k/v/out、
  ffn 三矩阵)逐输出通道对称 int8(clip=1.0;研究在 58 层上做了逐层裁剪搜索,
  **全部 minmax 最优,无敏感离群值,不需要数据驱动权重校准**),4×int8 打包 u32;
  头部(policy/value 全部)、RMSNorm、RoPE 按 model_pytorch.py 的 fp32 头部口径留 f32。
  **0.91M 参数量化,blob 1.14MB(f32 的 30%)**。
- **计算**:激活中间量 f16 存储、寄存器/共享内存/f32 累加(katago-webgpu「f16 storage +
  fp32 compute」形态;即研究结论的 W8A16,激活保持 f16 不做 a8)。设备无
  `shader-f16` 与不支持 WebGPU 同款**直接报错,不做降级**。
- **激活范围**(cpuref-Q 扫描):max absmax 54.9(hidden)≪ 65504,f16 存储安全。
- **L2 输出级**(引擎 i8 vs ort CPU fp32 golden,30 例;权威对照 = quant_explore
  第 40 批 8192 盘面 int8w 行:Top1 98.02% / KL 7.9e-4 / winMAE 5.1e-3 / 目差 0.061):
  policy KL 9.7e-4 ✓、winLoss MAE 6.2e-3 ✓、目差 MAE 0.115 ✓、ownership 3.8e-3 ✓、
  top1 含近平局 100%(4 例平局翻转,0 实质翻转)✓、pass 偏差 4.1e-2(绊线 0.1 ✓,
  KL 已覆盖)。与研究报告数字同带,「W8A16 基本无损」在引擎侧复现。
- **教训两则**:①packer 逐张量量化轴曾把 stem.global_w([OC][19])按 'last' 轴
  量化,引擎按 stGS[oc] 读 → GPU 与 cpuref 分歧 0.3~4 logit,靠「WGSL-Q vs
  cpuref-Q」隔离层定位(cpuref 与打包自洽,golden 又与 cpuref 一致,唯 GPU 独错);
  ②WGSL 探针读 f16 中间缓冲必须显式 f16 解码,按 f32 误读会制造大量假差异。
- **产物入库(2026-10-07 拍板,二次修正)**:.onnx(源)+ .i8.aewn(默认)+
  .aewn(f32 golden)三份入库;**f16 权重版按二次拍板撤销**——既不做运行时
  回退(无 shader-f16 即报错),也已从仓库与历史清出(git filter-repo)。
- **默认**:对弈页缺省加载量化版;`?weights=f32` 强制 fp32 golden 版。
  L3 对弈级(300 局等 visits)为遗留验收。

### 9.5 风险表销账情况

- WGSL 移植数值 bug(§6-2):对拍链已建,逐内核可定位(stage-diff/gpu-probe);
- 浏览器性能不达(§6-1):**未销账**,M3 真机 A/B 前不删 ort 路径(`?engine=ort`);
- Safari(§6-3):WGSL 用保守子集;valueMlp 需要 maxStorageBuffersPerShaderStage ≥9
  (适配器普遍 16,session 已按 adapter.limits 收敛申请);
- shader 首用卡顿(§6-4):84 pipeline 加载期一次性预编译(本机 124ms@llvmpipe)。
