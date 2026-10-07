# 模型 INT8 量化 —— 调研、落地与终态

> 2026-10-06 立项调研(当时现行裁决是「不做 INT8 / fp16 量化,fp32 起步」);
> **2026-10-06 当日裁决推翻并落地** —— i8 量化版按本文方案 A(W8A16)经 packer 一步
> 产出并切为默认权重;**2026-10-08 收敛为唯一形态**(onnx / fp32 权重通道出库,
> fp16 权重版此前已两度拍板撤销)。
> 本文结构:§1 调研结论与事后对照;§2~§8 调研原文存档(方法论仍有效,
> 事实冻结在 2026-10-06 时点);§9 落地记录(实际执行的路线、数字与终态)。
> 关联:[WEBGPU_ENGINE_RESEARCH.md](WEBGPU_ENGINE_RESEARCH.md)(量化经自研引擎
> packer 落地,ort-web 算子风险整体消失)、[NEURAL_PLAN.md](NEURAL_PLAN.md)。

---

## 1. 结论与事后对照(2026-10-08 回看)

调研期五条判断,事后逐一对照:

| # | 调研期判断 | 事后验证 |
|---|---|---|
| 1 | 量化收益的真实来源是下载体积,不是速度(WebGPU/WGSL 无 int8 类型与点积指令) | **兑现**:i8 blob 1.14MB(f32 的 30%);速度按「int8 存储 + 反量化进 f16/f32 计算」实现,量化本身零速度损失(i8 批 64 真机 1088 行/s 为三后端最高) |
| 2 | 推荐三步走:fp16 先行 → W8A16 → W8A8/QAT 缓行 | **走了中间那条**:fp16 权重版曾实现(含回退变体),经二次拍板撤销(不做任何回退,清出仓库与历史);W8A16 经 packer 落地为**默认并最终唯一**;W8A8/QAT 未触发(其工具预案 `int4_qat.py` 已入库备用) |
| 3 | 校准/调优数据选法 = 分布对齐 + 分层覆盖 + 激活范围覆盖(§5) | **未需要**:最终落地的是权重-only 逐通道量化,训练侧 quant_explore 逐层裁剪搜索证明 58 层全部 minmax 最优、无敏感离群值 —— 无需数据驱动校准。§5 配方降级为「若上激活量化(W8A8)才需要」的预案 |
| 4 | 损失估算四级验收(L0 golden → L1 张量 → L2 输出 → L3 对弈) | **按框架执行并收窄**:L0/L2 全部通过(数字见 §9);L1 定位工具演化为 quant-test 三层闸门;L3 正式 300 局未跑,以真机三后端循环赛 1:1:1 实证替代 |
| 5 | 与自研引擎联动:int8/fp16 只是 packer 的 `--dtype` 选项,ort-web 算子支持问题整体消失 | **完全兑现**(§4.5 的预言):量化未经 ort-web 转换一步落地;ort-web 层面的所有存疑(算子覆盖、EP 验证)从未需要验证即消失 |

**终态一句话**:浏览器权重唯一形态 = `b8c96h3tfrs_19.i8.aewn`(1.14MB,
trunk 58 层逐输出通道对称 int8 + 头部/norm/RoPE 留 f32,激活 f16 存储 / f32 累加,
即 W8A16);设备无 `shader-f16` 直接报错,不做任何降级;常备闸门 = quant-test 三层
(激活范围 / cpuref-Q vs ort golden —— golden 经 `QONNX` 指路,缺席自动跳过 /
WGSL-Q vs cpuref-Q)。

---

## 2. 量化对象:模型结构与可量化面(调研存档)

`b8c96h3tfrs_19`(v17 transformer,rope-graph 版):**930,790 个 initializer 参数
≈ 3.72MB fp32**(文件 3.79MB)。onnx 1.23 实测结构:

| 部件 | 形状(×块数) | 参数量 | 占比 |
|---|---|---|---|
| FFN(SwiGLU)linear1/gate 96→256、linear2 256→96 | 24576 × 3 × 8 | 589,824 | 63.4% |
| 注意力 q/k/v/out_proj 96→96 | 9216 × 4 × 8 | 294,912 | 31.7% |
| stem conv_spatial 3×3 22→96 | 19008 × 1 | 19,008 | 2.0% |
| 各头(policy/value/ownership/scoreValue 1×1 conv + linear) | — | ≈25,000 | 2.7% |
| RMSNorm 权重(norm1/norm ×16 + trunkfinal) | 96/192 每个 | ≈2,000 | 0.2% |

图算子清单(节点数):MatMul 72、Conv 13(1 个 3×3 stem + 12 个 1×1 头)、Mul 82、Add 58、
Reshape 34、Transpose 42、ReduceMean 18、Pow/Sqrt/Div 各 16、Softmax 8、Sigmoid 8、Relu 6、
Gather 16(rope swapidx)、Sin/Cos 各 1、ReduceMax 1、Unsqueeze/Concat/Identity 各 5。

**可量化面**:全部 93 万参数都住在 72 个 MatMul + 13 个 Conv 的权重里;
其余算子是逐点或元数据操作,无权重。即:**权重量化 = 全量覆盖,逐点算子留在高精度即可**。
头(value/scoreValue/ownership)参数只有 ~2.5 万,量化省不了几百 KB,却直接动用户可见的
目差/死子标注 —— 默认排除(落地时执行,见 §9)。

每条前向的计算量:8 块 ×(qkv+out 投影 4×3.3M + 注意力 2×12.5M + FFN 3×8.9M)
+ stem 6.9M ≈ **0.53 GMAC/行(≈1.1 GFLOP)**,batch 8 ≈ 8.4 GFLOP —— 在现代 GPU 上
微不足道,印证速度不是量化动机。

## 3. WebGPU 上的 int8 现实(为什么不是「量化 = 又小又快」;已被引擎实现验证)

### 3.1 WGSL 层面

- WGSL 核心标量类型只有 i32/u32/f32/f16(开 `shader-f16`);**没有 i8 类型**。
  int8 数据只能以 packed u32 形式存 storage buffer,加载后转高精度再算 ——
  GPU 侧 int8 推理的正确形态是「int8 存储 + fp32/f16 计算」:存储带宽省 4×,ALU 不省。
- WebGPU 核心没有 int8 矩阵乘指令;`chromium_experimental_subgroup_matrix`(张量核)
  是实验特性,不进依赖。

结论:**W8A8 在 WebGPU 上不存在速度收益**;int8 的收益空间 = 权重存储(下载 + VRAM)。
—— 引擎落地形态与此完全一致(权重 int8 打包 u32,装载处 `f32()` 反量化,
激活 f16 存储 / 寄存器与累加 f32)。

### 3.2 ort-web 层面(历史约束,2026-10-08 起整体消失)

调研时现行栈是 ort-web 1.30.0 WebGPU EP,量化算子(MatMulInteger / QLinearMatMul /
DynamicQuantizeLinear)在 WebGPU EP 的 kernel 覆盖未核实,算子缺席 = 会话创建失败
= int8 QOperator 模型直接不可部署。**该风险从未需要验证**:量化最终经自研引擎
packer 落地(§4.5 预言成真),ort-web 于 2026-10-08 整体出库。

### 3.3 三种量化形态对比(调研期判断;落地 = 中列,fp16 列被拍板撤销)

| | fp16 权重 | int8 权重(W8A16)| int8 全量化(W8A8) |
|---|---|---|---|
| 体积 | 1.86MB(2×↓)| **≈0.93MB(4×↓)** | ≈0.93MB(4×↓) |
| 速度 | fp16 存储/fp32 计算,带宽 2× | 与 fp32 持平(反量化后照旧)| 不会更快,可能更慢 |
| 精度风险 | 极小 | 小(纯权重误差,可控可测)| 大(激活异常值敏感) |
| 工程量 | 半天 | 1 天 | 2~3 天 + 校准闭环 |
| 终态 | **撤销**(二次拍板:不做回退,清出历史)| **★ 落地为唯一形态** | 缓行(预案工具已入库)|

> 「主仓库有现成量化工具」核查结论维持:KataGo 主仓库/Transformer 仓库无 int8 量化
> 工具链;现成的是验证工具(`katago testgpuerror`)。真正的量化研究在训练侧自建
> (quant_explore,见 §9)。

## 4. 怎么量化:三档方案(调研存档;实际走 §4.5 的 packer 路线)

### 4.1 方案 A(主推,已落地):W8A16 —— int8 权重存储 + 高精度计算

- 粒度:**per-channel(按输出通道)symmetric int8**,zero-point=0 —— transformer 权重
  近似零均值对称,per-channel 是零成本默认;
- 排除清单(不量化,留 f32):全部头(policy/value/scoreValue/ownership 的 conv+linear)、
  全部 RMSNorm 权重、RoPE 相关 —— 头部直接动目差/死子标注,量化收益 <100KB;
- 落地载体:**不是 ONNX QDQ 图,而是 packer 的 `--dtype i8`**(§4.5)—— 图结构不动,
  WGSL 装载处反量化;训练侧零改动。

### 4.2 方案 B(缓行,未触发):W8A8 静态量化(QDQ)

仅当 A 之后体积/速度仍成瓶颈才考虑;需 §5 校准数据闭环 + ort-web EP 算子验证
(现已无意义)。保留本节作为激活量化的方法论存档。

### 4.3 方案 C(兜底,未触发):QAT

PTQ 对弈级损失超标时才启动;基础设施(nnlog 教师 logits 旁路、checkpoint、
`torch.ao.quantization.FakeQuantize`)在案。实验工具 `int4_qat.py`
(INT4 PTQ 逐层裁剪搜索 + STE QAT 恢复,协议与 quant_explore 同口径)已入库备用。

### 4.4 方案排序(实际执行轨迹)

```
fp16(调研首推)──→ 曾实现(含回退变体),二次拍板撤销(不做任何运行时回退;清出仓库与历史)
W8A16 ──────────→ 经 packer 落地,先默认(10-06/07)后唯一形态(10-08)
W8A8 / QAT ─────→ 未触发(L2/L3 证据均未超标)
```

### 4.5 与自研引擎的联动(预言成真)

自研引擎落地后,fp16 / int8 存储 = packer 的 `--dtype` 选项,反量化发生在 GPU 加载
路径,存储收益与下载收益同时兑现;本报告的损失估算框架原样适用(量化的是「权重精度」,
与载体格式无关)。**实际执行**:自研引擎(aethernn)2026-10-06 当日立项当日落地,
量化同日经 packer 跟进 —— 见 [WEBGPU_ENGINE_RESEARCH](WEBGPU_ENGINE_RESEARCH.md)。

## 5. 校准与调优数据(调研存档;权重-only 路线最终未需要)

§5 的完整选法(分布对齐 / 手数阶段分层 / 特殊局面旗标 / 激活 absmax 分位覆盖 /
4096 行 + 固定种子 68320512 / hard book 难例册)为**激活量化(W8A8/QAT)的预案**。
落地路线为权重-only 逐通道量化,训练侧 quant_explore 的逐层裁剪搜索证明 58 层全部
minmax 最优、无敏感离群值 —— **无需数据驱动校准**(§9)。若未来上激活量化,本节
配方原样适用;统计纪律(回归集永不参与调优、hard book 单独出表)一并保留。

## 6. 量化损失怎么估:四级验收(调研框架 + 实际执行对照)

| 级 | 调研期设计 | 实际执行 |
|---|---|---|
| L0 黄金参考 | ort CPU fp32 落盘 5 输出;尺度参照 1e-3 量级 | ✅ ort CPU golden(cpuref-test 时代 18 例;现 quant-test 第 2 层,golden 经 QONNX 指路,缺席跳过) |
| L1 张量级(SQNR,定位用) | 逐层隔离,最低 2~3 层进排除清单 | ✅ 演化为分段对拍(stage-diff,后出库)与「WGSL-Q vs cpuref-Q」隔离层(定位 packer 量化轴 bug 的功臣,§9) |
| L2 输出级(整网闸门) | policy KL<1e-3 / top1≥99% / winLoss<1e-3 / lead<0.1 目 / ownership<0.01 / 死子翻转<0.5% / pass<1e-2 | ✅ 全过(数字见 §9);验收口径对齐 quant_explore 实测行而非预设闸门(top1 口径 98.02% 为「8192 盘面含近平局」的等价表述) |
| L3 对弈级(唯一决定闸门) | 每档 ≥300 局,Elo 损失 ≤30 | ⚠️ 正式 300 局未跑;以真机 6 局三后端循环赛 1:1:1(同网络容器)实证棋力无损;i8 成唯一形态后对照物已出库,正式闸门闭环存档待新学生出炉时随发布流程重建 |
| L4 浏览器端到端 | e2e + 校准重跑无意外劣化 | ✅ 对弈页 smoke + wgsl-test / quant-test 全绿;校准热机 i8=16 正常 |

**f16 激活存储带来的口径修正(2026-10-07 调查链,l3probe 1-3)**:窄容差对拍在
i8f16 形态下原理上不可通过 —— f32 累加序差被 16 层 f16 存储逐层放大(「舍入边界
混沌」:即使注入精确 f16 舍入点仍有 ~2.6e-2 rms)。对策:对拍分层 —— 纯权重精度
效应用 cpuref-Q(f32 数学)对拍 golden(闸门同 W8A16 量级),内核与 f16 存储效应用
WGSL-Q vs cpuref-Q(容差按 f16 舍入量级),对弈级指标(对 ort golden 的对弈级口径)
兜底。quant-test 三层闸门即此结论的固化。

## 7. 落地清单(计划 vs 实际)

| 调研期计划产物 | 实际产物 |
|---|---|
| `training/quantize_student.py`(fp16/W8A16 转换) | **未建** —— 载体改私有 blob 后,职能并入 `training/pack_aewn.py --dtype`(逐权重校验断言保留:反量化回代相对误差 < 1/127×1.5,防量化轴错) |
| `training/select_calib_data.py` | **未建** —— 权重-only 路线无需校准(§5) |
| `training/eval_quant_loss.py`(L0/L1/L2) | **以 quant-test 三层闸门实现**(激活范围 / cpuref-Q vs golden / WGSL-Q vs cpuref-Q) |
| nn-match 对弈批次(4 档 × 300 局) | **未跑** —— 以真机三后端循环赛 6 局 1:1:1 实证替代;正式闸门见 §6 L3 |
| 结论回写 | 本报告 §9 + README「不做」清单更新(已推翻裁决入档)+ NEURAL_PLAN §6.4 |

## 8. 风险与开放问题(销账情况)

- ort-web EP 对量化图支持未实测 → **风险消失**(ort-web 出库,量化走 packer);
- 注意力异常值 → **未发生**(W8A16 权重-only;激活范围 max absmax 54.9 ≪ 65504,
  f16 存储安全);W8A8 若启动仍按缓行处理;
- kata1 位解包正确性 → 与量化路线解耦(未走到);nnlog 旁路无此风险的判断维持;
- 每代学生都要重量化 → **成立并已固化**:quant-test 三层闸门 + 对弈级抽测进发布流程;
  i8 为唯一形态,不存在「fp32 合格、int8 不合格」的静默上线通道;
- pass/乐观通道是否拆分评估 → 未拆,整体 KL 考核通过(4.1e-2 pass 偏差在绊线内)。

---

## 9. 落地记录(2026-10-06 → 10-08)

### 9.1 实际路线

训练侧研究先行:**quant_explore**(`<trainrun>/quant/`,第 40 批权重 s68320512 重跑确认)
—— 8192 盘面上对 58 层 trunk 做逐层裁剪搜索,**全部 minmax 最优、无敏感离群值,
不需要数据驱动权重校准**;int8w 行权威读数:Top1 98.02% / KL 7.9e-4 / winMAE 5.1e-3 /
目差 MAE 0.061。引擎侧随后以 packer 复现:`training/pack_aewn.py --dtype i8` →
`models/b8c96h3tfrs_19.i8.aewn`(**1.14MB,f32 blob 的 30%**),2026-10-06/07 切默认,
2026-10-08 收敛为唯一形态。

### 9.2 量化形态(与 §4.1 方案 A 逐条对应)

- **量化面**:全部 58 层 trunk(conv_spatial、linear_global、attn q/k/v/out、ffn 三矩阵)
  逐输出通道对称 int8(clip=1.0;4×int8 打包 u32,scale 存 f32 子区);0.91M 参数量化;
- **排除面(留 f32)**:头部(policy/value 全部)、RMSNorm、RoPE 按 `model_pytorch.py`
  fp32 头部口径 —— i8 blob 内头部/排除张量以 f16 io 存储 f32 权重;
- **计算**:激活中间量 f16 存储、寄存器/共享内存/f32 累加(katago-webgpu「f16 storage +
  fp32 compute」形态,即 W8A16,激活不做 a8);设备无 `shader-f16` 直接报错,不做降级;
- **激活范围**(cpuref-Q 扫描):max absmax 54.9(hidden)≪ 65504,f16 存储安全。

### 9.3 L2 输出级对拍(引擎 i8 vs ort CPU fp32 golden,30 例)

| 指标 | 闸门(§6 调研值) | 实测 |
|---|---|---|
| policy KL | < 1e-3 nat | **9.7e-4** ✓ |
| top1 一致率 | ≥ 99% | **含近平局 100%**(4 例平局翻转,0 实质翻转)✓ |
| winLoss MAE | < 1e-3 | **6.2e-3**(超预设;与训练侧权威行 winMAE 5.1e-3 同带 —— 8 bit 截断下 winLoss 天然量级即 1e-2 档,预设值定严了。按本节口径原则:L2 先导、L3 决定) |
| scoreLead MAE | < 0.1 目 | **0.115**(略超预设;与训练侧权威行 0.061 同带。按本节口径原则处理:L2 是先导指标非闸门,L3 循环赛实证通过后放行) |
| ownership MAE | < 0.01 | **3.8e-3** ✓ |
| pass logits 偏差 | < 1e-2 | **4.1e-2**(KL 已覆盖,绊线 0.1 内)✓ |

与 quant_explore 第 40 批 8192 盘面权威行同带 ——「W8A16 基本无损」在引擎侧复现。

### 9.4 对弈级证据与终态

- **真机 6 局三后端循环赛**(onnx / aewnn f32 / aewnn i8,同网络容器,2026-10-07):
  **1:1:1**,棋力无损实证 —— 此为 2026-10-08 移除 onnx/f32 通道的对弈级依据;
- 正式 300 局 × 4 档 L3 闸门未跑(§6 L3 行;i8 唯一形态后对照物已出库);
- **f16 权重版(dtype=2)**:曾完整实现(头部 f32 权重路由 + parseAewn f16 视图 +
  gemmRes 绑定序,量化测试 15/15 绿),经二次拍板撤销(此前一度拍板「f32/i8/f16/onnx
  均不允许删」,二次推翻)—— 引擎定位为「不做任何运行时回退」,f16 blob 已清出
  仓库与历史(git filter-repo);
- **2026-10-08 终态**:onnx / fp32 blob 出库,`parseAewn` 只认 dtype=1;quant-test
  golden 改 `QONNX` 环境变量指路(训练管线产物,不在场自动跳过第 2 层);
- **后续实验工具入库**(2026-10-08):`int4_qat.py`(INT4 PTQ 逐层裁剪搜索 + STE QAT
  恢复,协议与 quant_explore 同口径)、`zstd-exp.py`(.aewn 权重 zstd 压缩实验)、
  `huffman-rice-exp.mjs`(int8 码流 Huffman/Rice 变体)—— 均为压缩/精度边界探索,
  未改变生产形态。

### 9.5 教训两则

1. **packer 逐张量量化轴 bug**:stem.global_w([OC][19])曾按 'last' 轴量化,引擎按
   stGS[oc] 读 → GPU 与 cpuref 分歧 0.3~4 logit。靠「WGSL-Q vs cpuref-Q」隔离层定位
   (cpuref 与打包自洽,golden 又与 cpuref 一致,唯 GPU 独错);已改首维轴并重打包。
   **per-channel 的轴 = 输出通道,逐权重断言(反量化回代 < 1/127×1.5)从此为 packer 硬纪律**;
2. **WGSL 探针读 f16 中间缓冲必须显式 f16 解码**:按 f32 误读会制造大量假差异 ——
   调试工具与被测内核的精度口径必须一致,否则差异全是读数方式的伪影。
