# 调研报告:模型 INT8 量化(方案 / 校准数据选择 / 损失估算)

> 2026-10-06 立项调研。背景:[NEURAL_PLAN](NEURAL_PLAN.md) §7 与 README「不做」清单
> 现行裁决是「不做 INT8 / fp16 量化(fp32 起步,体积与速度真成瓶颈再上,主仓库有现成量化工具)」。
> 本文为**维持或推翻该裁决**提供依据;拍板之前不动现有裁决,不落任何生产改动。
>
> 回答三个问题:**①怎么量化;②怎么从训练数据里选校准/调优数据;③量化损失怎么估**。
> 最终建议(详见 §1):fp16 先行、int8 权重量化(W8A16)次之、W8A8 全量化缓行 ——
> 在 WebGPU 上 int8 的硬收益只有体积,而体积这件事 fp16 能拿一半、int8 能拿满,
> 但风险和工程量差一个量级。

---

## 1. 结论(TL;DR)

1. **量化收益的真实来源是下载体积,不是速度。** 模型 fp32 3.79MB;int8 权重 ≈0.93MB(−75%),
   fp16 权重 ≈1.86MB(−50%)。而速度:WebGPU/WGSL **没有 int8 类型与点积指令**(§3.1),
   int8 计算在 GPU 上不会比 fp32 快;ort-web 的 WebGPU EP 里量化模型大概率还要逐算子反量化回
   fp32 再算(§3.2)。
2. **推荐三步走**:① fp16 权重模型(脚本 30 行,精度风险最小,2× 体积收益,ort-web 可直接跑);
   ② int8 权重 W8A16(DequantizeLinear 反量化进 fp32 算子,4× 体积收益,精度可测可控);
   ③ W8A8 全量化 / QAT 仅当 ② 的对弈级损失超标才启动(§4.3、§4.4)。
3. **校准/调优数据从训练数据里选,选法 = 分布对齐 + 分层覆盖 + 激活范围覆盖**(§5):
   主池用 N3 自对弈行(部署分布),kata1 行补多样性;按手数阶段 × 特殊局面(劫/征子/提子)分层,
   再按「激活 absmax 分位」强制覆盖尾部;4k 条 + 固定种子;对称 ×8 增强。
4. **损失估算分四级验收**(§6):张量级(SQNR)→ 输出级(policy KL / top1 一致率 / value MAE /
   ownership MAE,带闸门表)→ 对弈级(nn-match 等 visits A/B,Elo 闸门)→ 浏览器 e2e。
   对弈级是唯一有决定权的闸门,其余都是它的先导指标。
5. 与[报告二](WEBGPU_ENGINE_RESEARCH.md)联动:自研引擎落地后,权重格式是我们自己的,
   int8/fp16 只是 packer 的一个选项(§4.5),ort-web 量化算子支持问题整体消失。

---

## 2. 量化对象:模型结构与可量化面

`models/b8c96h3tfrs_19.onnx`(v17 transformer,rope-graph 版,2026-10-05):
**930,790 个 initializer 参数 ≈ 3.72MB fp32**(文件 3.79MB)。onnx 1.23 实测结构:

| 部件 | 形状(×块数) | 参数量 | 占比 |
|---|---|---|---|
| FFN(SwiGLU)linear1/gate 96→256、linear2 256→96 | 24576 × 3 × 8 | 589,824 | 63.4% |
| 注意力 q/k/v/out_proj 96→96 | 9216 × 4 × 8 | 294,912 | 31.7% |
| stem conv_spatial 3×3 22→96 | 19008 × 1 | 19,008 | 2.0% |
| 各头(policy/value/ownership/scoreValue 1×1 conv + linear) | — | ≈25,000 | 2.7% |
| RMSNorm 权重(norm1/norm ×16 + trunkfinal) | 96/192 每个 | ≈2,000 | 0.2% |

图算子清单(节点数):MatMul 72、Conv 13(1 个 3×3 stem + 12 个 1×1 头)、Mul 82、Add 58、
Reshape 34、Transpose 42、ReduceMean 18、Pow/Sqrt/Div 各 16、Softmax 8、Sigmoid 8、Relu 6、
Gather 16(rope swapidx)、Sin/Cos 各 1(rope 表图首子图)、ReduceMax 1、Unsqueeze/Concat/Identity 各 5。

**可量化面**:全部 93 万参数都住在 72 个 MatMul + 13 个 Conv 的权重里;
其余算子(Softmax/Sigmoid/Pow/Sqrt/Sin/Cos/Reshape/Transpose)是逐点或元数据操作,无权重。
即:**权重量化 = 全量覆盖,逐点算子留在 fp32 即可**。头(value/scoreValue/ownership)参数
只有 ~2.5 万,量化它们省不了几百 KB,却直接动用户可见的目差/死子标注 —— 默认排除,见 §4.1。

每条前向的计算量:8 块 ×(qkv+out 投影 4×3.3M + 注意力分数/加权 2×12.5M + FFN 3×8.9M)
+ stem 6.9M ≈ **0.53 GMAC/行(≈1.1 GFLOP)**,batch 8 ≈ 8.4 GFLOP —— 在现代 GPU 上是
微不足道的量,再次印证速度不是量化动机。

---

## 3. WebGPU 上的 int8 现实(为什么不是「量化 = 又小又快」)

### 3.1 WGSL 层面

- WGSL 核心标量类型只有 i32/u32/f32/f16(开 `shader-f16`);**没有 i8 类型**。
  int8 数据只能以 packed u32 形式存 storage buffer,加载后转 fp32 再算 ——
  也就是说 GPU 侧 int8 推理的正确形态是「int8 存储 + fp32/f16 计算」,存储带宽省 4×,
  ALU 不省。
- WebGPU 核心没有 int8 矩阵乘指令。`chromium_experimental_subgroup_matrix`(协作矩阵/张量核)
  是实验特性,katago-webgpu 的结论是「flag-gated、stable 浏览器不可移植」(其 WEBGPU_STATUS.md
  "Tensor cores" 节),1~2 年内不该进我们的依赖。

结论:**W8A8 在 WebGPU 上不存在速度收益**;int8 的收益空间 = 权重存储(下载 + VRAM)。

### 3.2 ort-web 层面(现状约束)

现行栈是 ort-web 1.30.0 WebGPU EP(session.js:39)。两个待验证事实(本机网络受限,
无法直接读 1.30 源码,给出 10 分钟验证法):

- 量化算子(MatMulInteger / QLinearMatMul / QLinearConv / DynamicQuantizeLinear)在
  WebGPU EP 的 `jsep/webgpu/op-resolve-rules.ts` 里是否有 kernel **未核实**;按 ONNX Runtime
  Web 一贯的算子覆盖情况,这几个大概率缺席或仅 WASM EP 支持。产品边界「仅 WebGPU、不可用即报错」
  意味着:**算子缺席 = 会话创建失败 = int8 QOperator 模型直接不可部署**。
  验证方法(浏览器控制台,先把 §4.2 的脚本产出 int8 模型放到 models/ 下):
  ```js
  const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.mjs');
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  await ort.InferenceSession.create('/models/b8c96h3tfrs_19.int8.onnx',
    { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });  // 失败即缺席
  ```
- `DequantizeLinear` 是基础算子,EP 覆盖面远好于 QOperator 系列(且本项目刚为 rope-graph
  验证过 Sin/Cos/Mul/Concat 这类基础 kernel 的核实流程,NEURAL_PLAN §9 2026-10-05 条)。
  **W8A16 路线只需要它**(§4.1),风险低得多;同样用上面的命令验证即可。

### 3.3 三种量化形态对比

| | fp16 权重 | int8 权重(W8A16) | int8 全量化(W8A8) |
|---|---|---|---|
| 体积 | 1.86MB(2×↓) | **≈0.93MB(4×↓)** | ≈0.93MB(4×↓) |
| 速度 | fp16 存储/fp32 计算,带宽 2×(真提速) | 与 fp32 持平(反量化后照旧) | 不会更快(§3.1),可能更慢 |
| 精度风险 | 极小(指数范围 ±65504,katago-webgpu 实证 v17 族网可用) | 小(纯权重误差,可控可测) | 大(激活范围 + 异常值敏感,transformer 注意力尤甚) |
| ort-web 可跑性 | 高(fp16 是常见路径;需 §3.2 同法验证) | 高(只多 DequantizeLinear) | **存疑**(§3.2,可能直接不可用) |
| 工程量 | 半天 | 1 天(自写转换脚本) | 2~3 天 + 校准闭环 + QAT 兜底预案 |
| 真正值得? | **先做** | 体积仍不够再上 | **缓行** |

> 「主仓库有现成量化工具」核查:KataGo 主仓库/Transformer 仓库 grep 无 int8 量化工具链
> (NEURAL_PLAN §7 该表述不成立);现成的是**验证工具**——`katago testgpuerror -reference-file
> ref.bin`(KataGo docs/ONNX_Model_Files.md:195,Eigen 写参考输出、被检后端对比),恰好是 §6
> L0 级验收的现成实现。真正的量化工具在 onnxruntime(python)侧,bleed 环境已具备(§4.2)。

---

## 4. 怎么量化:三档方案

### 4.1 方案 A(主推):W8A16 —— int8 权重存储 + fp32 计算

**思路**:不动图结构,只把 72 个 MatMul + 13 个 Conv 的权重 initializer 换成 int8 + 每输出通道
scale,插入 `DequantizeLinear` 反量化回 fp32 再进原算子。图里除新增 Dequant 节点外与现在逐位同构,
ort-web / ort CPU / 未来自研引擎三边行为一致,无算子覆盖风险。

- 粒度:**per-channel(按输出通道)symmetric int8**,zero-point=0 —— transformer 权重分布
  近似零均值对称,per-tensor 会因通道间幅度差白白丢 1~2 bit;per-channel 是零成本的默认。
- 排除清单(不量化,留 fp32):全部头(policy/value/scoreValue/ownership 的 conv+linear,
  ≈2.5 万参数,量化省 <100KB 却直接动目差/死子标注)、全部 RMSNorm 权重、RoPE 相关。
  可量化面收窄为 **FFN 58.98 万 + 注意力 29.49 万 + stem 1.9 万 ≈ 91.2 万参数**,int8 后
  文件 ≈0.93MB(stem 是 3×3 卷积,若嫌敏感也可排除,只贵 18KB)。
- 转换脚本(新增 `training/quantize_student.py`,onnx 库直接实现,~80 行):
  读 fp32 模型 → 对每个目标 initializer 计算 `scale_c = max(|W[c,:]|)/127` → 写 int8 initializer
  + scale tensor → 插 DequantizeLinear → 输出 `models/b8c96h3tfrs_19.int8w.onnx`。
  **逐权重断言**:反量化回代与原权重相对误差 < 1/127×1.5,防止搞错轴(per-channel 的轴 =
  输出通道;MatMul 的 Wnhwc 是 (in,out) 布局,轴=1 —— 转换脚本必须用 make_rope_ongraph.py
  同款「改前逐表校验」纪律)。
- 失败形态友好:任何一步 assert,不产文件。
- 变体:fp16 版同一脚本 `--dtype float16` 产出 `b8c96h3tfrs_19.fp16.onnx`
  (initializer 全 cast fp16;Sin/Cos/Softmax 等计算节点不动,由运行时决定计算精度)。

### 4.2 方案 B(缓行):W8A8 静态量化(QDQ)

仅当 A 之后速度/内存实测仍成瓶颈(大概率不会,§2 计算量太小时才考虑)。工具链 bleed 环境
现成:`/home/a/miniconda3/envs/bleed/bin/python`,onnx 1.23.1 + onnxruntime 1.23.2,
`onnxruntime.quantization.quantize_static` 可用,`QuantFormat.QDQ` + `QuantType.QInt8`,
校准器 `CalibrationMethod ∈ {MinMax, Entropy, Percentile, Distribution(=MSE/分布损失)}`,
校准数据喂 §5 选出的集合(用 `CalibrationDataReader` 适配 AEPOS001/npz 格式,内存 memmap 流式)。

- 参数起点:`Distribution`(MSE 最小化点积误差,对 transformer 最稳)为主,
  `Percentile 99.9/99.99` 与 `MinMax` 做对照三组;per-channel 权重、per-tensor 激活(EP 限制)。
- 注意力 q/k 投影、FFN down-proj(256→96,直连残差)是历史经验上的敏感层;§6 的逐层敏感度
  扫描排出来后,敏感层从量化节点列表剔除(quantize_static 支持按节点名单指定 op_types_to_quantize
  / nodes_to_quantize,排除即回落 fp32)。
- **前置闸门**:§3.2 的 EP 算子验证通过才有意义。

### 4.3 方案 C(兜底):QAT(量化感知微调)

PTQ(方案 A/B)对弈级损失超标时才启动。基础设施全部现成:

- checkpoint 在 `trainrun/b8c96h3tfrs_run/checkpoint.ckpt`,模型定义
  `KataGo/python/katago/train/model_pytorch.py`(b8c96h3tfrs-fson-silu);
- **教师 logits 已有**:nnlog 旁路(`KATA_NNLOG`,`onnxbackend.cpp` nnEvalLogBatch,
  AENNL001 格式,`training/distill_from_nnlog.py` memmap 直读)——QAT 不需要第二次教师前向,
  伪量化学生直接对教师原始输出做 KL(policy)+ MSE(value/ownership)微调 1~2 epoch
  (9 路同管线实测 35 分钟/2 epoch,19 路估计小时级);
- 伪量化:`torch.ao.quantization.fake_quant.FakeQuantize` 包在目标 Linear/Conv 权重上
  (学 scale 的 LSQ 式或固定 scale 二选一,先固定),导出走 export_model_pytorch.py →
  .bin.gz → dumponnx 照旧,导出后再用 §4.1 脚本把学到的 scale 烧成 QDQ/W8A16 图。
- 工程量 1~2 天(改 model_pytorch.py 加伪量化开关 + 训练脚本参数),**不做预研,
  只在触发条件出现时立项**:对弈级 Elo 损失 > 30(§6 闸门)且逐层排除已到头。

### 4.4 方案排序与触发条件

```
fp16(半天,几乎零风险)─体积够→ 完事
   │体积仍要降
   ▼
W8A16(1 天)─对弈级闸门过→ 完事
   │闸门不过:先逐层排除(半天),还不行
   ▼
W8A8(§4.2)→ 仍不行 → QAT(§4.3)→ 仍不行 → 维持 fp32(推翻失败,裁决维持)
```

### 4.5 与自研引擎的联动(报告二)

自研 WebGPU 引擎落地后,模型载体从 ONNX 换成 packer 产出的私有格式
(WEBGPU_ENGINE_RESEARCH.md §4.3)。那时:

- fp16 / int8 存储 = packer 的 `--dtype` 选项(权重按 int8+scale 或 half 打进权重 buffer,
  WGSL 加载路径里反量化或加载时一次性展开),**ort-web 量化算子支持问题整体消失**;
- 反量化发生在 GPU 加载路径,int8 存储的 VRAM 收益与下载收益同时兑现;
- 本报告的校准数据选择(§5)与损失估算(§6)全部原样适用 —— 它们量化的是「权重精度」,
  与载体格式无关。**建议:§5/§6 的工具链先行(它们对 fp16 同样必需),W8A16 转换在
  ort-web 上先出一版拿体积收益,自研引擎落地后 packer 直接接管。**

---

## 5. 校准与调优数据:从训练数据里怎么选

### 5.1 原则:校准分布必须 = 部署分布

推理引擎见的局面 = **学生网自己下出来的局面 + 用户对人机的实战局面**,不是 kata1 的
教师分布。量化误差只在「激活值分布被压缩」处伤模型,所以:

- **主池:N3 自对弈行**(run_n3_loop.sh 产出,1 万局 ≈ 80 万行,~250MB,§3.3 运行约束);
  N3 未启动前的过渡池:用现行学生模型按 `selfplay_aether9.cfg`(visits 32/16)采 2~3 千局,
  或直接 `KATA_NNLOG=1` 挂 onnxbackend 旁路收「输入特征」(distill_from_nnlog.py 同款格式,
  免任何二次编码);
- **补充池:kata1 行**(`kata1_data/`,17G,单目录 23,482 个 npz × 81 行 ≈ 190 万行)——
  提供自对弈覆盖不到的极端形态(大龙死活、超长劫争),占 ≤25%;
- **实战样本:对弈页真实人机局**(经 dump_positions.mjs 的浏览器编码器回放 SGF 产出;
  该脚本 9 路时代所写,跑 19 路前先用 featdiff 同款对拍确认 22×361 布局)。

### 5.2 数据源与格式对照(全部已验证存在)

| 源 | 格式 | 取「模型输入」的方式 | 备注 |
|---|---|---|---|
| N3 selfplay npz | KataGo 训练行 | `binaryInputNCHWPacked (81,22,46) uint8` 位解包 → (22,361) f32;`globalInputNC (81,19)` 直用 | 位解包 = np.unpackbits 后取前 361 位;与 fillRowV7 逐位对拍是 featdiff 已锚定的事实 |
| kata1 npz | 同上 | 同上 | 抽样下载块,受 30G 磁盘哨兵约束 |
| KATA_NNLOG 旁路 | AENNL001 自定义 | 输入特征本来就是 f32 直存 | **最省事**:免解码、免对拍,采集时顺手就有 |
| SGF 回放(dump_positions.mjs) | AEPOS001 | JS 编码器逐局面编码 | 浏览器真值口径;适合实战局样本 |

**统一原则:所有校准输入必须经过与浏览器完全相同的编码口径**(fillRowV7 JS 版,已逐位对拍),
禁止从 npz 现场发明第二种解码 —— 这是「训练强、浏览器弱」风险的量化版。

### 5.3 选择配方(写入 `training/select_calib_data.py`)

1. **分层(stratify)**:按 ①手数阶段(0-20 开局 / 21-80 前中盘 / 81-160 中盘 / 161+ 官子,
   等比各 ~25%);②特殊局面旗标(盘上存在劫点、征子进行中(特征通道 15/16 非零)、
   上三手内提子、|scoreLead| > 20、双停附近)—— 每个旗标桶至少 200 行。
   依据:这些旗标直接对应特征通道的取值范围,漏了哪类,哪类通道的激活范围就没被校准覆盖。
2. **激活范围覆盖(range coverage)**:先用 fp32 模型(bleed 环境 ort CPU,或 torch
   model_pytorch.py 挂 forward hook)对候选池 ~2 万行扫一遍**逐张量激活 absmax**,
   按「该行激活 absmax 落在全体分布的分位桶」(<50% / 50-90% / 90-99% / ≥99%)分层,
   校准集必须含 ≥99 分位桶的完整覆盖 —— MinMax/Distribution 校准器都靠真实尾部定 scale,
   没有 99 分位的样本,激活量化就是「按未见过的大值裁剪」。
3. **规模与批次**:4,096 行(ORT 官方建议数百~数千;transformer 有注意力异常值,取上限档),
   batch 64 流式喂。回归集(**regression set**)另抽 2,048 行,固定后**永不参与调优**,
   只做验收(§6)—— 防止「调优把校准集调好了」的自欺。
4. **对称增强**:每行随机取 8 对称之一(与训练侧 random symmetry 同分布),白送 ×8 多样性。
5. **固定种子**:seed = 68320512(与模型快照同源),数据清单(calib/回归/hard book 三份
   文件名+行哈希)写进本报告附档,保证任何后续重跑可复现。
6. **hard book(固定难例册)**:从自对弈库按旗标精挑 ~200 局面(长劫争、征子吃不吃、
   双活/ Sebastiano、贴目边界细棋),单独成册 —— 它是 §6 输出级指标里死子标注(ownership
   阈值 0.35)与 pass 判断的专门考卷;`training/ladder_probe.py` 现成可产征子专项。

### 5.4 「量化调优」的闭环

选出的数据同时服务三件事:

- **校准**(方案 B 的 calibrator 输入);
- **调参**:校准器超参扫描(MinMax / Percentile∈{99.9,99.99} / Distribution)×
  排除清单(§4.1 基础上按 §6 敏感度排名追加),在回归集上选指标最优组合 ——
  每组合产出一张 §6 的 L1/L2 指标表,择优上线;
- **QAT 微调**(方案 C 的训练集,叠加 nnlog 教师 logits)。

---

## 6. 量化损失怎么估:四级验收

### 6.1 L0:黄金参考(fp32 基线)

- 固定 eval 集(§5.3 回归集 + hard book)在 **ort CPU EP fp32**(bleed 环境)跑一遍,
  落盘全部 5 输出 = 黄金参考(复用/扩展 `training/ort_server.py`,加 `--dump` 模式);
- C++ 侧等价物现成:`katago testgpuerror -model X.onnx -reference-file ref.bin`
  (Eigen 写 ref.bin),以后每代学生出炉顺手可查。
- 尺度参照:rope-graph 重写的输出相对误差 ~5e-7 = 「无感」基线;int8 权重量化的
  天然量级是 1e-3(8 bit 截断),所以闸门数值都按 1e-3 量级定。

### 6.2 L1:张量级(逐层定位用)

对每个量化层,比较该层输出(整网隔离该层):SQNR(dB)= 10·log10(Σx²/Σ(x−x̂)²)、
余弦相似度。**用途是定位,不是验收**:SQNR 最低的 2~3 层 → 优先追加进排除清单。
工具:`training/eval_quant_loss.py --per-layer`,onnx 图按节点切两段跑(ort CPU)。

### 6.3 L2:输出级(整网,带闸门)

整网 fp32 vs 量化版,同一 eval 集,按输出头出表:

| 指标 | 定义 | 闸门(建议) | 超标直接后果 |
|---|---|---|---|
| policy KL | KL(p_fp32 ‖ p_int8),362 点含 pass,均值 | < 1e-3 nat | 搜索先验失真,访问浪费 |
| top1 一致率 | argmax 相同的行占比 | ≥ 99% | 走子直观偏差 |
| top5 重合 | 前 5 集合 Jaccard | ≥ 98% | 同上 |
| winLoss MAE | |Δwinrate| 均值 | < 1e-3 | 效用/认输(−0.90 线)漂移 |
| scoreLead MAE | |Δlead|(目)均值 | < 0.1 目 | 形势判断/目差效用可见劣化 |
| ownership MAE | 逐点均值 | < 0.01 | 死子标注(阈值 0.35)翻转风险 |
| 死子翻转率 | 两版 `deadStonesWithOwnership` 结论不同的局面占比(hard book) | < 0.5% | **用户可见 bug** |
| pass logits 偏差 | OutputPolicyPass[0] | < 1e-2 | 停着时机劣化 |

统计纪律:所有指标带 95% CI(按行 bootstrap);hard book 单独出表(它就是专挑敏感局面的,
超标不许用「平均达标」糊弄过去)。

### 6.4 L3:对弈级(唯一决定性闸门)

- 工具现成:`test/nn-match.mjs <modelA> <modelB> <局数> <visits>`(双 ort_server 子进程,
  等 visits 对弈)。**量化模型是合法参赛者** —— python ort(CPU)对 QDQ/W8A16/int8 全支持。
- 方案:每档难度(80/200/400/800 visits)打 **≥300 局**(胜率 95% CI ≈ ±5.5%,
  Elo ≈ ±38);汇总 Elo 损失 = 400·log10(p/(1−p)) 按局数加权。
- **闸门:Elo 损失 ≤ 30(任何单档 ≤ 40)**。理由:这是陪人下棋产品,30 Elo 在人类对手
  感知之下;难度四档本身是访问数分级,量化不应改变档位间相对强度。
- 加测一组「访问补偿」:量化版 1.25× 访问 vs fp32 版 1× 访问,若 Elo 反超,
  说明损失可用极小的搜索预算换回 —— 给「要不要上线」多一个谈判筹码。
- 注意:nn-match 跑的是 ort CPU 后端,与浏览器 GPU 的数值不完全同序;GPU 侧差异由
  L4 兜住,L3 关心的是量化本身的系统性棋力损失。

### 6.5 L4:浏览器端到端

- `npm run test:nn-e2e` + 对弈页人工 smoke(量化模型换成 session 加载目标跑通);
- 加载时批校准(calibrateMaxBatch)对量化模型重跑一遍,确认 maxBatch 与吞吐没有意外劣化
  (W8A16 会多一批 Dequant kernel,理论影响 <5%)。

### 6.6 失败路径

L2 超标但 L3 达标 → 仍可上线(L2 是先导指标不是闸门);L3 超标 → 逐层排除扩清单 →
仍超标 → §4.4 路径图右移(W8A8/QAT),全部走完仍超标 → **维持 fp32,本调研结论「量化不可行」,
裁决自动维持** —— 这也是验收的一部分,不是失败。

---

## 7. 落地清单

| 产出 | 内容 | 工作量 |
|---|---|---|
| `training/quantize_student.py` | fp16 / W8A16 转换(§4.1,含逐权重校验断言) | 0.5 天 |
| `training/select_calib_data.py` | §5.3 配方(分层 + 范围覆盖 + 固定种子三份清单) | 1 天 |
| `training/eval_quant_loss.py` | L0 dump + L1 SQNR + L2 指标表(§6) | 1 天 |
| nn-match 对弈批次 | 4 档 × 300 局 × 2~3 个候选(脚本化过夜) | 机时 1 天 |
| 结论回写 | 本报告补实测数据表 + README/NEURAL_PLAN 裁决更新 | 0.5 天 |

第一里程碑(fp16 + W8A16 + L0~L2 表)≈ **3 个工作日 + 一晚机时**,即可拿到「量化是否可行」
的完整数据,支撑 §4.4 的路径决策。环境零新增依赖(bleed 现成)。

## 8. 风险与开放问题

- **ort-web WebGPU EP 对量化图的支持未实测**(§3.2 给了 10 分钟验证法)—— 这是方案 B 的
  前置闸门,fp16/W8A16 不受其害;
- **注意力异常值**:transformer 的 softmax 前激活有重尾,W8A8 静态激活量化历史翻车率高;
  本报告把 W8A8 定为缓行即是为此;
- **kata1 行的位解包正确性**:binaryInputNCHWPacked 解包实现必须过 featdiff 对拍后再用
  (§5.2);nnlog 旁路无此风险,可优先;
- **每代学生都要重量化**:N3 每代出炉 → 量化脚本 + L2/L3 快检(300 局一档)应作为发布流程
  的一环(gatekeeper 之后的浏览器发布闸门),否则「fp32 合格、int8 不合格」的代际会静默上线;
- 开放问题:pass 通道与乐观通道(C=2)是否需要分开评估量化损失 —— 先按整体 KL 考核,
  hard book 的停着专项若超标再拆。
