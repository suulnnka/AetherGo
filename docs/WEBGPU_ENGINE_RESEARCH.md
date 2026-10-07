# 自研 WebGPU 推理引擎(aethernn)—— 调研、落地与演进全记录

> 2026-10-06 立项调研,**当日落地 Node 侧全链并默认切换**;2026-10-07 真机 A/B 收口
> (性能闸门通过);**2026-10-08 收敛完成 —— onnxruntime-web 逃生舱与 fp32 golden
> 通道整体出库,引擎为 aethernn + i8f16 权重单形态,运行时零外部依赖**。
> 本文结构:§1 调研结论与事后验证对照;§2~§8 调研原文存档(决策依据,事实冻结在
> 2026-10-06 时点);§9 落地记录;§10 落地后演进(读现状直接看 §1 与 §10)。
>
> 参照物:katago-webgpu(KataGo fork,手写 WGSL WebGPU 后端,v17 全架构覆盖,
> 与 Eigen CPU 基准逐位对拍;其 WEBGPU_STATUS.md 是一份完整的性能工程日志)。
> 关联:[INT8_QUANT_RESEARCH.md](INT8_QUANT_RESEARCH.md)(量化随本引擎经 packer 落地)、
> [DISPATCH_FUSION_PLAN.md](DISPATCH_FUSION_PLAN.md)(下一性能课题)、
> [NEURAL_PLAN.md](NEURAL_PLAN.md)(架构与契约)。

---

## 1. 结论与事后验证(2026-10-08 回看)

调研期的五条判断,事后逐一对照:

| # | 调研期判断 | 事后验证 |
|---|---|---|
| 1 | 值得做,且比想象中便宜:~14 个 WGSL 内核 + 硬编码执行计划,估 1~1.5 周 | **低估了有多便宜**:M0~M3 的 Node 侧全链当日(10-06)完成,真机 A/B 10-07 收口,全程 2 天 |
| 2 | 动机量级:ort-web 冷加载 5.5MB(模型本体的 1.45 倍)+ CDN 单点;自研 ≈ 30-60KB JS + WGSL | **完全兑现**:运行时依赖归零,离线可用;10-08 起连逃生舱也不存在 |
| 3 | 接口面已预留好:evalBatch 契约就是为换后端设计的,换芯 = 重写一个文件 | **成立**:默认切换只动了 `src/nn/session.js` 门面与新增 `webgpu/` 目录;10-08 移除 ort 时净删 ~190 行 |
| 4 | 最大风险是浏览器侧性能不达预期,「性能 ≥ ort-web×0.9」定为切换闸门,不达标零损失退出 | **顾虑证伪**:真机批 1-64 全档 aewnn ≥ ORT-Web(批 64 i8 1088 vs 718 行/s = ×1.51;1024v 口径 456-555 vs 245-347 visits/s);闸门从未接近失败 |
| 5 | 两个报告互为条件:先自研后量化,packer 直接产 int8 权重一步到位 | **成立**:i8 量化经 packer `--dtype` 一步落地为默认权重(10-06 晚),未经 ort-web 做任何量化转换 |

**现状一句话**:aethernn = 84 dispatch 硬编码执行计划 + WGSL 内核族(GEMM 平铺 /
flashB / RMSNorm / RoPE / 池化与头部小核)+ 单 pass / 单 submit / 单回读宿主;
权重唯一形态 i8f16(`.i8.aewn`,1.14MB);无 WebGPU / 无 `shader-f16` 直接报错,无任何回退。
高批吞吐已专项改造(flashB v4 + GEMM B 系批内循环,批 8/16/32 端到端 ×2.1~3.1);
下一性能课题见 [DISPATCH_FUSION_PLAN](DISPATCH_FUSION_PLAN.md)。

---

## 2. 现状盘点:ort-web 在本项目的真实成本(调研存档,2026-10-06 时点)

| 维度 | 事实(当时) | 影响 |
|---|---|---|
| 下载体积 | 187KB(mjs,gzip)+ 5.35MB(jsep.wasm,gzip)= 5.5MB/冷加载 | 占总冷加载的 59%;N4 体积闸门的最大单项 |
| CDN 依赖 | `cdn.jsdelivr.net` 懒加载 | 离线/CDN 故障/审查网络 = 引擎不可用;WebOS 场景(纯前端离线 OS)尤其冲突 |
| 版本耦合 | 锁 1.30.0;.mjs 具名导出、wasmPaths、JSEP 算子覆盖 | 每次升级都要重核算子覆盖;升级无收益驱动,不升级则修复也拿不到 |
| 性能黑盒 | 通用图解释器:每个 ONNX 节点一次 JSEP 分发;无单 pass/单 submit 控制;批校准只能整图测 | 无法做 katago-webgpu 已证明的大头优化(§3.3) |
| 通用性错配 | 仅一个模型、仅 WebGPU、无回退 —— 通用引擎的「任意 ONNX」能力 100% 用不上 | 为用不到的灵活性付 5.5MB + 黑盒税 |

另有一条每代重检风险:ort-web WebGPU EP 对新导出 ONNX 图的兼容性(每代学生出炉都要
人工核一轮)—— 自研后变为「架构固定、一次验证」。(已验证:兼容性问题归零。)

## 3. 参照物 katago-webgpu:哪些直接可搬(调研存档,经验仍有效)

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
| `rmsReduceSpatial`/`rmsApplySpatial` | RMSNorm 归约/应用 | ✅ ×16 | 逐 token 对 C=96 归约 |
| `ropeApply` | 2D RoPE | ✅ ×16 | **改良**:cos/sin 表打包期预算好(θ=100,head_dim32,19×19) |
| `attnScores`/`attnSoftmax`/`attnOutput` | 注意力三段 | ✅(起步用) | O(seq²) 在 361 token、3 头下完全可行 |
| `flashAttention` | 融合注意力(在线 softmax) | ✅ | 3 dispatch → 1;后演进出本引擎自有 flashB v4(§10) |
| `swigluGate` | SwiGLU | ✅ ×8 | |
| `activate` | relu/silu 等 | ✅ | 头部 6 个 Relu |
| `addInPlace` | 残差加 | ✅ | |
| `globalPoolMeanMax`/`globalPoolValueHead` | 全局池化 | ✅(value 头用 mean 路径) | subgroups 特性门控,可后置 |
| `scaleBiasMaskAct` | BN 融合 | ❌ | 本模型无 BN(RMSNorm 架构) |
| `winograd*` 四件套 | 3×3 Winograd | ❌ | 同上,只有 1 层 3×3 |

### 3.3 性能经验(全部有 A/B 数据,已全部继承)

1. **体制判断:小网是延迟受限,不是算力受限**。b6c96 @ batch6:~90 dispatch,
   固定 ~7ms GPU 开销(Dawn 在依赖性 dispatch 间插 storage-hazard barrier),
   batch 3 ≈ batch 15 的墙钟 —— **吞吐靠大批摊薄固定延迟**。
   (本引擎校准选批「吞吐 ≥ 最优 90% 的最小批」即此原理;低批压缩 dispatch 数
   是后续课题,见 DISPATCH_FUSION_PLAN。)
2. **最大单笔收益是提交合并,不是 FLOPs**:每评估一次 `Submit` + 一次合并 readback,
   替代 ~50 次 submit + 5 次阻塞往返 → +24%;单 compute pass 再 +9%;1×1 fast conv +31%。
   **三条纯宿主架构经验,搬 WGSL 时免费继承**(本引擎单 pass / 单 submit / 单回读即此)。
3. **资源复用**:权重常驻一次上传;中间量/uniform 走 BufferPool(坑:按「≤2× 大小复用槽」,
   否则不同 batch size 会累积 ~80 个 buffer,浏览器 VRAM 爆炸 —— 它踩过,我们没再踩)。
4. **RoPE 表一次算好**(每 handle 一次,别每 eval 重传)—— 本引擎进一步前移到打包期。
5. **fp16 的血泪**:fp16 storage + fp32 compute(`alias STO` 单源双精度)是对的形态;
   host 侧 float→half 必须 **clamp 到 ±65504**(KataGo 的 1e9 off-board 哨兵会变 0×Inf=NaN);
   旧 g170 网 trunk 溢出 fp16 需 scale8 重标定,而 **v17 silu/modern 网 fp16-stable**。
   (本引擎实测激活 max absmax 54.9 ≪ 65504,见 INT8 报告落地记录。)
6. **selective fp32 heads**:fp16 模式下 norm 归约/池化/头 matmul 留 fp32(实测 0.08% 胜率
   误差 + 2.4× 吞吐)—— 本引擎 i8f16 形态的头部留 f32 与此同源。
7. **subgroups 归约**:feature-gated 双版本内核范式。(本引擎 flashB v4 为纯 ILP
   线程级实现,无需 subgroups 特性。)
8. **cooperative matrix(张量核)**:实验特性,不进依赖;列为未来免费升级。

### 3.4 踩坑记录(逐条可核,移植时已对照)

- readback buffer 必须 `CopySrc` 能力,否则 MapAsync 读到全零(症状:策略退化为常数);
- 输入布局:NHWC/NCHW 错配会把每手评估变成乱码垃圾且**性能正常所以无人发现** ——
  全网对拍是唯一防线;
- `requireExactNNLen` 口径:先验 mask 全 1 即可断言后**跳过 mask 乘法**
  (本模型恒喂 1,aewnn 中已折叠);
- fp32 与 CPU 参考的浮点累加序差异是「预期内不可消除项」,闸门要按量级定,不追逐位
  —— i8f16 形态下 f16 存储进一步放大该效应(「舍入边界混沌」,闸门口径已相应调整,
  见 INT8 报告落地记录与 quant-test 三层闸门)。

## 4. 自研引擎设计(暂名 aethernn;设计原文,as-built 差异见 §9)

### 4.1 范围裁决

- **不是通用 ONNX 解释器**:执行计划硬编码为 JS 数据表(层清单由 ONNX 结构导出,
  人工固化 + 一致性断言:packer 产出的权重 SHA 与计划表里的形状清单互验,
  架构不符启动即报错)。换模型 = 换权重 + (若架构变)改计划表 ——
  与产品边界「仅一个模型」严格对齐。
- ONNX **保留为训练侧权威格式**:dumponnx 管线照旧;packer(`training/pack_aewn.py`)
  消费 ONNX → 私有权重 blob(`.aewn`)。(2026-10-08 起 ONNX 文件出库 —— 它是训练
  管线的可再生产物;packer 的输入契约不变。)

### 4.2 执行计划(从 ONNX 实测结构导出;落地后 423 节点 → 84 dispatch)

```
输入:spatial(N,22,19,19) f32 → 上传;global(N,19) f32 → 上传
stem:  conv3x3 22→96(对称 gather + 全局广播加融合)
trunk ×8 blocks:
       rms(norm) → qkv 单 GEMM(packer 拼三矩阵)→ rope → flashAttention
       → out_proj GEMM + 残差 epilogue → rms → ffn1+gate 单 GEMM → swiglu
       → ffn2 GEMM + 残差 epilogue                              (每块 9)
final: norm_trunkfinal(scale+bias+relu 融合)                    [1]
heads: policy(conv1p/conv1g/pool/ling/conv2p/pass)
       value(conv1/pool/valueMlp/own)                           [~10]
合计 84 dispatch;单 compute pass、单 Submit、单 readback。
```

数值口径全部继承 session.js 已验证的后处理(×20/softplus scoreValue、乐观插值在 JS、
winLoss 双 logits 归一)—— readback 后的 JS 后处理与 ort 路径同口径。

### 4.3 权重格式 `.aewn`(packer 产出)

- 头(魔数+版本+dtype+形状清单+SHA)+ 单一大 payload:全部权重按计划表布局预排
  (qkv 拼接、matmul K 主序、通道 padding 对齐 tiledGemm);RoPE cos/sin 表打包期预算写入;
- **dtype 即量化选项**:`0 = f32 | 1 = i8f16 | 2 = f16w`(INT8 报告 §4.5 的预言;
  f16 权重版旋即撤销,2026-10-08 起 parseAewn 只认 dtype=1);
- 体积:f32 版 3.82MB(已出库);i8 版 **1.14MB(现唯一)**。

### 4.4 GPU 资源与宿主结构(落地后实际文件)

```
src/nn/webgpu/
  plan.js        执行计划表 + blob 解析 + 计划互验(架构不符启动即报错)
  kernels.js     WGSL 内核族(自 katago-webgpu webgpukernels.cpp 移植,MIT 注明;
                 GEMM 平铺族 + flashB + rms + rope + 池化与头部小核)
  session.js     宿主:createAewnnSession —— 单 pass 84 dispatch、单 submit、
                 单 readback 一次 mapAsync;权重单缓冲常驻;uniform 256B 槽/每 dispatch;
                 行直传 writeBuffer(零 CPU 端拼批拷贝);adapter 特性探测在此文件内
  cpuref.js      CPU 参考解释器(与 WGSL 逐算子同构,测试专用;i8 权重反量化仿真 = cpuref-Q)
src/nn/
  session.js     门面:createSession(i8f16 唯一,无回退;2026-10-08 前含 ort 双实现)
  calibrate.js   批校准抽出(加载时现测,吞吐 ≥ 最优 90% 的最小批)
  symmetry.js    SYM8 抽出共用 + stem gather 表(8 对称 GPU 侧置换)
```

- 权重:单一大 GPUBuffer,加载时一次 writeBuffer 常驻;
- 中间量:BufferPool(≤2× 槽复用,size 按 maxBatch 预留);
- uniform:一块大 uniform buffer + dynamic offset 环(零每 dispatch 分配);
- 提交:每 evalBatch 一个 encoder → 一个 pass → 84 dispatch → 一次 submit →
  readback(**CopySrc**)一次 map → JS 后处理 → 返回同构结果;
- 8 对称:rows[i].sym 随行下发,**GPU 侧置换**(stem gather 表),搜索侧特征零拷贝直传
  (在飞期间复用特征缓冲是安全的:上传在首个 await 前同步完成);
- 批校准:加载时现测(冷启动管线编译以预热吸收),session.maxBatch 语义两代后端同构。

### 4.5 接口与切换面(历史设计,已走完)

`src/nn/session.js` 门面按构建期常量选实现,evalBatch 契约一字不动 →
search/worker/难度四档零改动。迁移期双实现并存供 A/B(`?engine=ort`);
**真机 A/B 达标后删除 ort 路径 —— 2026-10-08 执行**,「并存是开发期脚手架,
不是运行时回退」的界定兑现。

## 5. 验证与迁移计划(M0~M4;全部完成)

| 里程碑 | 内容 | 闸门 | 结果 |
|---|---|---|---|
| M0 基线 | ort-web 吞吐基线 + golden dump(3 局面 × 5 输出,ort CPU fp32) | 数字入库 | ✅(ort 基线后经 browser-ab 测速场补全为批 1-64 全谱) |
| M1 内核 | packer + 内核移植 + 逐内核对拍 | 每内核全绿 | ✅(对拍链:cpuref-test golden → wgsl-test 内核 → stage-diff 分段定位) |
| M2 全网 | 整网对拍 2,048 局面:policy KL / value MAE / ownership MAE | KL<1e-4 / winLoss<1e-4 / ownership<1e-3 | ✅ 超额:policy 1.4e-5 / winLoss 1.0e-6 / ownership 1.9e-6;WGSL vs cpuref 2.7e-5 |
| M3 集成 | session 换芯;现有测试全绿;浏览器 A/B | **性能 ≥ ort×0.9** | ✅(2026-10-07 真机:批 1-64 全档 ≥ ORT,批 64 i8 +51%;1024v 456-555 vs 245-347 visits/s) |
| M4 收尾 | 默认切 aethernn;ort-web 依赖删除;文档裁决回写 | 全部测试 + 对弈页人工 smoke | ✅(默认切换 10-06;ort 路径 10-08 移除;裁决回写见两报告 + README) |

## 6. 风险登记表(全部销账)

| 风险(调研期评估) | 结果 |
|---|---|
| 浏览器性能不达 ort-web(中) | **未发生**:真机全档 ≥ ORT;同机拆解显示差距主要在引擎代差而非宿主(见 katago-cuda-bench-1024v.md 的 5.9× 引擎差分解) |
| WGSL 移植数值 bug(中) | 对拍链建成即定位工具齐备(stage-diff / gpu-probe;后演进的 f16 舍入问题属口径而非 bug,见 §10) |
| Safari/WebKit 兼容(低) | WGSL 保守子集;valueMlp 需 maxStorageBuffersPerShaderStage ≥9(适配器普遍 16,session 按 adapter.limits 收敛申请);N4 实测待做 |
| shader 首用卡顿(低-中) | 全部 pipeline 加载期一次性预编译 + 校准预热吸收 |
| 单模型硬编码绑架未来(低) | 计划表是数据不是代码;packer 形状清单互验,架构不符启动即报错 |
| 工程分散注意力(中) | 实际与搜索侧机制移植并行推进,无冲突 |

## 7. 被拒备选(与理由)

| 备选 | 结论 | 理由 |
|---|---|---|
| 维持 ort-web | 已淘汰 | 5.5MB + CDN + 黑盒是长期税;A/B 后按计划删除 |
| ort-web + fp16 模型(不换引擎) | 已淘汰(fp16 权重版后两度实现两度撤销) | 拿到 2× 模型体积,拿不到 5.5MB 运行时体积与离线能力 |
| 直接用 katago-webgpu 的 WASM 产物(kataeval) | **拒** | 违反「纯 JS,不上 WASM」拍板(NEURAL_PLAN §0);其 wasm 含完整 KataGo 搜索,与本产品「JS 搜索」架构冲突;其价值(WGSL 内核与经验)已吸收 |
| WebNN | 观察 | 浏览器覆盖与算子成熟度不足,不进依赖;若未来标准化可再评估 |

## 8. 拍板请求(调研期原文存档)

1. 是否立项 aethernn(§5 M0-M4,~1~1.5 周,退出点 M3);—— **已立项并落地**
2. 若立项:顺序建议先 M0+M1(不碰线上),M2 起占用发布流程;—— **已按此执行**
3. 若暂不立项:至少落地 M0 基线测量,并把 fp16 模型作为独立小改进先做。—— **未走到此分支**

---

## 9. 落地记录(2026-10-06,立项当日完成 Node 侧全部里程碑)

### 9.1 交付物(as-built)

```
training/pack_aewn.py        packer:ONNX → .aewn(qkv/gate 拼接、BiasMask scale 折叠、
                             头部 conv 权重转置 [outC][inC]→[inC][outC]、RoPE 表打包期预算、
                             --dtype f32|i8)
models/b8c96h3tfrs_19.i8.aewn  1.14MB(现唯一;f32 版 3.82MB 已随收敛出库)
src/nn/webgpu/plan.js        计划表 + blob 解析 + 计划互验(架构不符启动即报错)
src/nn/webgpu/kernels.js     WGSL 内核族(stem/rms/gemmQkv/rope/flashB/gemmPlain|Res|
                             swiglu/trunkFinal/pool×2/ling/passHead/valueMlp/gemmSmall …)
src/nn/webgpu/session.js     宿主:单 pass 84 dispatch、单 submit、单 readback 一次 mapAsync、
                             权重单缓冲常驻、uniform 256B 槽/每 dispatch、行直传 writeBuffer
src/nn/webgpu/cpuref.js      CPU 参考解释器(与 WGSL 逐算子同构,测试专用;
                             i8 反量化仿真 = cpuref-Q)
src/nn/symmetry.js           SYM8 抽出共用 + stem gather 表(8 对称 GPU 侧置换)
src/nn/session.js            门面(2026-10-08 起:i8f16 唯一,无回退)
src/nn/calibrate.js          批校准抽出(加载时现测)
src/nn/search.js             零拷贝环形槽特征 + sym 随行下发;eval-cache 键并入 sym
```

### 9.2 执行计划与融合(对照 §4.2 的预期)

算子顺序**逐块镜像 PyTorch forward**(`model_pytorch.py` TransformerAttentionBlock/
TransformerFFNBlock,ONNX 423 节点已逐节点核对):stem → 8×(attn: norm→qkv→RoPE→
attention→out_proj+残差 ‖ ffn: norm→gate SwiGLU→ffn2+残差) → trunkfinal → policy/value 头。
423 节点 → **84 dispatch**,全部融合点兑现:qkv 单 GEMM(与训练侧 fused_qkv_proj 同构)、
ffn1+gate 单 GEMM、残差入 GEMM epilogue、BiasMask scale 折权重、RoPE 表打包期预算、
flash attention 免 S² 物化、linear_g/pass/value MLP 各单 dispatch。
(注:静态派发表 119 项 = 84 运行时项 + 35 对 GEMM B 系双注册(高批批内循环变体),
两口径对上,见 DISPATCH_FUSION_PLAN §0。)

### 9.3 对拍数据(M1/M2 闸门,全部通过)

| 对拍 | 闸门 | 实测 |
|---|---|---|
| cpuref vs ort CPU golden(18 例 × 6 字段,含 sym×λ 混批) | fp32 累加序量级 | policy **1.4e-5**,winLoss 1.0e-6,ownership 1.9e-6(M2 要求 <1e-4/<1e-4/<1e-3) |
| WGSL vs cpuref(12 行混批满链) | 同上 | policy 2.7e-5,winLoss 8.3e-7,ownership 2.0e-6 |
| 分段(checkpoint 张量 32 处 × 逐块残差 16 处) | ≤3.7e-5 | 最大 3.7e-5(trunkfinal),16 块单调漂移 ≤5.7e-5 |
| 现有测试(engine/features/katago-align/worker-smoke/nn-e2e) | 全绿 | 全绿(nn-e2e 361 手完整终局) |

(对拍链路后经 2026-10-08 收敛精简:golden 闸门并入 quant-test 第 2 层
(QONNX 指路),分段工具出库;当前常备闸门 = wgsl-test + quant-test,见 README「测试」。)

### 9.4 与原计划的偏差(实测修正)

1. **头部 conv 权重布局**:trunk 的 `Wnhwc` 已是 [inC][outC] k 主序,但头部 1×1 conv 仍是
   Conv 布局 [outC][inC] —— packer 统一转置(§4.2 未预见,ort 黑盒下无从发现)。
2. **RoPE 预算表与 PyTorch 的关系**:ONNX 图内 Sin/Cos 子图与 `precompute_freqs_cos_sin_2d`
   的 repeat_interleave 语义已逐位验证等价(交换 sin 符号折进 swapidx),packer 直接按
   PyTorch 公式重建,`make_rope_ongraph.py` 退役为历史记录。
3. **多行语义**:内核全部按批内行距寻址;CPU 参考解释器首版逐行传参导致越界读 NaN ——
   已修(输入拼批),并成为「内核批布局」的回归锚。
4. ~~性能数字未出,真机 A/B 待浏览器实测~~ **已收口(2026-10-07)**:browser-ab 测速场
   真机实测,批 1-64 全档 aewnn ≥ ORT-Web,批 64 i8 1088 vs 718 行/s(×1.51);
   ORT 批 32 有吞吐悬崖;aewnn 批 1-64 平滑。

### 9.5 风险表销账情况(终态)

- WGSL 移植数值 bug:**销账**(对拍链 + 分段工具,回归可定位);
- 浏览器性能不达:**销账**(M3 真机全档 ≥ ORT);**2026-10-08:ort 路径已删,逃生舱不复存在**;
- Safari/WebKit:WGSL 保守子集 + limits 收敛申请;N4 中端手机实测待做;
- shader 首用卡顿:**销账**(加载期预编译 + 校准预热)。

---

## 10. 落地后演进(2026-10-06 → 10-08)

### 10.1 i8 量化版落位默认(10-06 晚;详见 INT8 报告落地记录)

packer `--dtype i8` 一步产出 `.i8.aewn`(1.14MB,f32 的 30%),W8A16(激活 f16 存储 /
f32 累加);引擎侧 30 例对拍与训练侧 quant_explore 8192 盘面研究同带,「基本无损」复现。
packer 量化轴 bug(stem.global_w 按列轴量化)由「WGSL-Q vs cpuref-Q」隔离层定位修复。

### 10.2 高批次吞吐改造(10-07;commit 09577b7)

flashB v4 成为唯一注意力算子(线程级,零 smem、零 subgroup,全批次恒跑;
32 维点积拆 4 路部分和打断依赖链)+ GEMM B 系批内循环(n≥8 切 R=4,W 瓦片步长 16→17
消 bank conflict)+ swiglu workgroup 64→256(修批 64 时派发网格超 WebGPU 65,535 上限
导致整个 command buffer 被静默丢弃的隐患)+ 缓冲容量 CAP 32→64。
实测(同机 RTX 5060):批 8/16/32 端到端 i8 ×2.10/2.04/2.1、f32 ×2.83/2.74/3.06;
批 1/4 低批路径逐位不变;浏览器批 1-64 全档 aewnn 压过 ORT(批 64:i8 1088 vs 718)。
方法论:两次假设被消融实测推翻(「瓶颈在 GEMM」——ts-probe 消融显示高批边际
flash 46-58% / GEMM 32-40%;共享内存归并 / subgroup 归并均劣于纯 ILP)。

### 10.3 批校准五项失真修复(10-07;commit 04e971d)

冷启动截断乱选(一次性管线编译计入预算 → 单档恒「≥90% 最优」→ 任意选小批,
实测 onnx 冷启动选中批 2、真值 16)/ min-of-N 估计不稳(改最快 3 次均值 +
iters 12 + 预算 4s)/ 少样本档污染(整档丢弃)/ 慢模式串扰(轮转交错采样)/
峰值锚刀口(锚 = 前二快档均值)。真机生产路径冷加载:onnx=8 / f32=16 / i8=16。

### 10.4 i8f16 单形态收敛(10-08)

真机 6 局三后端循环赛(onnx / f32 / i8,同网络容器)1:1:1 实证棋力无损 +
数值闸门全绿后:onnxruntime-web 逃生舱(`?engine=ort`)、fp32 golden 权重通道
(`?weights=f32`)、全 f32 内核模式、f16 权重残留分支一律出库;内核与宿主删掉全部
模式分支(不再按 blob dtype 装配),`parseAewn` 只认 dtype=1;头部 / 排除清单张量
仍按量化方案留 f32 权重。测试侧同口径收敛(纯 ort/f32 工具出库;
quant-test 的 golden 改 QONNX 环境变量指路,不在场自动跳过第 2 层)。

### 10.5 下一步

- **dispatch 融合**(立档未动代码):[DISPATCH_FUSION_PLAN](DISPATCH_FUSION_PLAN.md) ——
  低批压缩 dispatch(84 → ~65)、高批压存储往返(F2 rms 拆半);
- flash 高批边际(46-58%)另立课题;
- N4 中端手机实测(覆盖面与性能)。
