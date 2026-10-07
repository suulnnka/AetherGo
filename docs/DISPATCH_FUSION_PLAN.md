# dispatch 压缩融合方案(静态 119 / 运行时 84 → 目标 ~65)

> 2026-10-08 立档。来源:katago-webgpu 的工程记录(WEBGPU_STATUS.md,MIT)证明
> 小网前向在「每 dispatch 固定启动/屏障开销」体制下运行,dispatch 数就是低批延迟的
> 货币;本引擎内核本就移植自它(tiledGemm/flash/rmsNorm,见 kernels.js 头注),
> 单 pass / 单 Submit / 单回读、权重常驻、BufferPool ≤2×、RoPE 打包期预算等宿主
> 经验已全部继承。本档列出**还没搬完的最后一批融合点**,并给出精确的现状账、
> 逐项方案、数值口径影响与验收闸门。
>
> 事实基线(2026-10-08 本机 llvmpipe 实测 + 代码核对):
> 静态派发表 **119** 项,其中 35 对 GEMM B 系双注册(运行时按 n≥batchHi 二选一)
> → **每个 evalBatch 实际发射 84 个 dispatch**(README「84」与 bench「119」由此对上)。
> 真机消融(test/browser-ab/ts-probe.mjs,RTX 5060)已证:高批边际成本
> **flash 46-58% / GEMM 32-40%** —— 高批是计算/带宽主导,低批(对局口径批 1-8)
> 仍是固定开销主导。本方案同时打两条杠杆:**F1/F3/F4 压 dispatch(低批),
> F2 压存储往返(高批)**。

## 0. 现状账(代码核对,session.js buildDispatches)

| 段 | dispatch(运行时) | 内核 |
|---|---|---|
| stem | 1 | stem(conv+对称 gather+全局广播加) |
| 每块 ×8 | **9** | rms(norm1) → qkv → rope → flashB → out_proj(gemmRes) → rms(ffn norm) → gate(gemmPlain) → swiglu → ffn2(gemmRes) |
| trunkFinal | 1 | scale+bias+relu(融合) |
| policy 头 | 6 | conv1p → conv1g → poolPolicy → ling → conv2p(gemmSmall) → pass |
| value 头 | 4 | conv1 → poolValue → valueMlp → own(gemmSmall) |
| 合计 | **84** | 每块 9×8 = 72 + 1 + 1 + 6 + 4 |

heads 的 poolPolicy / ling / pass / poolValue / valueMlp 是 5 个单 workgroup 小核
(绑定表 wg 均为 [n] 或 [1]),数据量小、启动开销占比高 —— 是 F4 的对象。

## 1. 方案(按风险从低到高排序)

### F1 swiglu 折进 ffn2 GEMM 的 A 装载 —— 运行时 84→76

- **改法**:新增 gemmRes 的 A-source 变体(暂名 `gemmSwRes`):A 装载时不读
  `B('hidden')`,改读 `B('gate')` 的两列并现场算 `silu(g[:,c])·u[:,c+256]`;
  epilogue 残差不动。与 PyTorch fused_gate_proj 的下游形态同构。
- **收益**:每块 -1 dispatch(×8);省 hidden(256×361×n)整级 f16 写+读,
  BufferPool 少一个槽。顺带消除 swiglu 的派发网格上限隐患
  (workgroup 64→256 那次修复的历史包袱随内核一起消失)。
- **数值口径**:silu 输入从「f16 舍入后的 gate」变为「GEMM 累加器里的 f32 未舍入
  值」—— 精度持平或略优,f32 模式逐位一致;i8 模式对 f32 golden 的偏差按
  l3probe 调查链的结论(一次性探针已出库,结论存档于 INT8 报告 §6)预期在
  量化噪声内,以实测为准。
- **风险**:低。A 装载每 k-步读 2 列(512 vs 256),tile 内复用后净带宽仍降;
  swiglu 数学(silu 用 f32)必须在装载处保持同一口径。

### F2 rms 拆半:归约写标量,应用折进下游 GEMM 的 A 装载 —— dispatch 持平,打高批带宽

- **改法**:`rms` 内核改为只算每 (b,pos) 的 `rsqrt(mean(x²)+eps)` 写入小标量
  buffer(361×n f32);`gemmQkv` / `gate` 的 A 装载改乘 `r[row]·γ[col]`
  (GEMM 增绑 γ 与 r)。trunkFinal→heads 一处同型可作可选延伸(F2b)。
- **收益**:16 处 norm 每处省一遍 96 通道 normed 的写+读。流量账(估算,以
  A/B 实测为准):每处每评估省 ≈ 96×361×n×2B ≈ 4.4MB@n=64,16 处 ≈ 70MB,
  对比同批 A+C 主流(~80MB)约省一半存储访问 —— 正对 ts-probe 量到的高批
  GEMM 边际。
- **数值口径**:**有变**。现路径 normed 落过一次 f16 舍入,融合后 GEMM 读未舍入
  乘积(少一次舍入)。f32 模式不再逐位等同旧版;i8 模式「舍入边界
  混沌」教训在此适用(调查链结论存档于 INT8 报告 §6)—— 对拍闸门按量级定
  (policy Δ ≤ 5e-2 现行口径),不追位。
  cpuref.js 须同步增融合版分段函数,保持与内核逐算子同构(回归定位时可临时
  复活分段对拍;常备分段工具 stage-diff 已随 2026-10-08 收敛出库)。
- **风险**:中。GEMM 族两条路径(基本/B 系)都要出变体;A 装载多一次 buffer 读。

### F3 RoPE 折进 qkv GEMM 的 epilogue —— 运行时 76→68

- **改法**:qkv GEMM 已输出 head-major;RoPE 配对旋转 (2p,2p+1) 沿 head_dim
  相邻、**永不跨 16 宽 N-tile 边界**(16 为偶),epilogue 线程改为一次产出
  相邻两列(寄存器级配对输出)即可原地旋转,cos/sin 常量表照旧。
- **收益**:每块 -1 dispatch(×8);省 q/k(288 中的 192 通道)整级读写。
- **风险**:**四项中最高** —— 动 GEMM epilogue 结构,每线程输出 1 列改 2 列,
  基本与 B 系两路都要改。放最后做;若 F1/F2 落地后真机数据已达标,可砍。

### F4 heads 小核合并 —— 运行时 ~68→~65

- **改法**:把 poolPolicy+ling(+pass)与 poolValue+valueMlp 各并成每批一个
  内核(workgroup 内先归约出 gp,经共享内存广播后接 MLP 链)。数据量小,
  纯启动开销生意。
- **风险**:低;但单笔收益小(3 个 dispatch),排最后、顺手做。

## 2. 不做(与既有消融/裁决一致)

- **subgroup 归并 / 共享内存归并版 flash**:已被消融否决(共享内存版打崩占用率、
  subgroup 版均劣于 flashB v4 纯 ILP,commit 09577b7),不再回头;
- **Winograd / BN 融合**:本模型仅 stem 一层 3×3、无 BN(RMSNorm 架构),不适用;
- **多线程 / CPU 回退 / flash 重写**:产品边界单线程、仅 WebGPU;flashB v4 刚
  重构完毕,不在本档范围(其 46-58% 高批边际是**相邻**独立课题,另立档)。

## 3. 实施顺序与验收

顺序:**F1 → F2 → F4 → F3**(风险升序;每项独立落地、独立可回滚)。

- **A/B 开关**:仿 katago-webgpu 的 `NO_*` 模式,每项一个
  `KAE_NO_FUSION_SWIGLU` / `KAE_NO_FUSION_RMS` / `KAE_NO_FUSION_ROPE` 开关
  (bench/cpuref 路径读 `process.env`;对弈页经 query 参数进 Worker 需新增一条
  转发 —— 先例 `?engine=ort` 已随 2026-10-08 收敛移除),
  默认开,同一二进制随时关回旧路径对拍归因。
- **数值闸门**(全绿才准合):wgsl-test(新增融合 case)、
  aewnn-quant-test 三层(第 2 层 golden 经 QONNX 指路,不在场自动跳过)、
  `npm test` 全绿。
- **性能验收**:本机 aewnn-bench 先看 dispatch 数下降(119→111→111→103→100
  静态口径,F2 持平)+ 各档 rows/ms;真机走 browser-ab 测速场既有口径
  (?bench 批4 微基准 / ?sweep 批 1-64 / ?vbench 1024v),与落地前基线同机
  A/B。**基线先行**:每项动手前先跑一次并归档 results*.json。

## 4. 预期与止损

- 低批(对局口径批 1-8):F1+F3+F4 合计 -19 dispatch(84→65,≈23%)。按
  katago-webgpu 的体制记录(每 dispatch 固定启动/屏障开销),低批延迟收益与
  dispatch 数近似线性;真机驱动每内核启动开销高于桌面,方向只会更有利。
- 高批:F2 是主菜(存储往返约减半);若 ts-probe 复测显示 GEMM 边际明显下降
  而 flash(46-58%)不动,则本档收官、flash 另立课题。
- **止损线**:64v 对局口径受搜索预算钳制(现行测量:批4 已被钳到无感),若
  真机 ?bench 批4 与 ?vbench 口径连续两项提升 < 3%(采样噪声带),停止继续
  融合 —— 复杂度换不到可见收益时,复杂度就是负资产。
