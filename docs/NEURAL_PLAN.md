# AetherGo 路线计划(神经网络路线)

> 2026-10 立项,**本文是 AetherGo 唯一的路线计划**;2026-10-03 按产品边界拍板整体重写。
> 职责划分:**README = 产品边界 / 待办 / 不做 / 存疑的唯一权威**;
> 本文 = 技术口径:架构、训练管线、模型、ONNX 契约、搜索配置与差距的技术细节。
> 两处不一致时以 README 为准,并回来改本文。
>
> 目标:**浏览器里的 19×19 强棋力人机对弈**。路线:用 KataGo 官方工具链做
> 「监督训练(蒸馏 / kata1 冷启动)→ 自对弈强化」训练一个小型 transformer 模型,
> 导出 ONNX,浏览器端 WebGPU 推理 + JS 侧 PUCT 搜索。

---

## 0. 总体架构:三件套,各司其职

| 角色 | 用什么 | 不做什么 |
|---|---|---|
| **训练引擎** | KataGo 官方 C++ 二进制(selfplay / gatekeeper / dumponnx) | 不自己写训练侧搜索 |
| **模型** | 主仓库 v17 transformer 配置(见 §2) | 不在浏览器里训练 |
| **对弈引擎** | AetherGo 纯 JS:特征编码 → ONNX 推理(WebGPU) → PUCT 搜索 | 不做随机演棋(NN 叶子评估取代) |

关键认知:**自对弈数据生成用官方 C++ 引擎**(快几个数量级),JS 引擎只负责「在浏览器里陪人下棋」。
整条训练闭环全部使用现成工具,我们只写配置和浏览器侧代码。

### 技术选型(已拍板)

- **纯 JS,不上 WASM/Zig/Rust**。NN 推理占 95% 以上算力且全部在 WebGPU(onnxruntime-web)里,
  JS 只做调度与树操作。WASM 的收益场景(CPU 密集演棋)两头都不占,还要跨边界调 WebGPU,不值。
- **搜索必须异步化**:WebGPU 推理是异步的,PUCT 主循环 async 化,待评估叶子攒成 batch 提交
  (batch 8 叶,已实现)。
- **模型 pos-len 跟随盘面(现 19)**:RoPE 的 cos/sin 表在 ONNX 里按盘径烘焙,JS 侧无需实现;
  dumponnx 用 `-nn-x-len 19 -nn-y-len 19 -require-exact-nnlen` 导出无掩码图,浏览器喂恒 1 的 mask。

## 1. 当前状态快照(2026-10-03)

**引擎侧(全部已落地,含测试):**

- 规则:气尽提子 / 禁自杀 / position superko(Zobrist,零分配 make/unmake)、中国规则数子、
  Benson 死子 + NN ownership 辅助标注、双停终局流程;
- 特征编码器:`fillRowV7` 的 JS 版,与 selfplay 训练行逐位对拍通过(1500 行),
  征子通道与 C++ 原生实现对拍 8393 链零分歧;
- NN 搜索:异步动态批 PUCT(校准上限 + 虚拟损失,单槽管线)+ KataGo 机制对齐(§5)+ 跨手树复用 +
  动态认输 + 温度选点;难度四档 = 80/200/400/800 访问;
- 对弈页:人机 / 双人、数子明细窗(死子手改)、形势判断(ownership 热图 + 目差)、悔棋 / 换边。

**模型侧:**

- 浏览器模型(唯一):**`models/b8c96h3tfrs_19.onnx`** —— 19 路学生 b8c96h3tfrs
  **第 40 份(s68320512)**,循环赛 40:41 63:57、40:45 75:45、41:45 69:51 拍板选用
  (训练侧 C++ 128 visits 各 120 局);v17 transformer,929,641 参数,
  dumponnx `-nn-x-len 19 -nn-y-len 19 -require-exact-nnlen` fp32 导出,8.2MB;
  2026-10-05 起 rope-graph 版 3.79MB(RoPE 表改图内现场计算,§9 末 2026-10-05 条)。
  2026-10-03 落位,旧占位模型(b6c96_19 / b6c96_9 / student_9 / student-random-init_9)
  已全部删除 —— 仅支持这一个模型;
- 9 路蒸馏学生 `models/student_9.onnx` 已随旧占位一并退役(方法记录见 §3.1)。

**测试基线:** 模糊测试(40 局 × 420 手,子数守恒 + 禁全同不变量 + 逐步撤销重演)、
特征对拍(featdiff)、征子对拍(ladderdiff)、搜索机制对齐项(katago-align)、
NN 端到端(nn-e2e)、等 visits 对战(match)、温度分布(nn-temp)。命令见 README「测试」。

## 2. 学生模型(已拍板:**`b8c96h3tfrs-fson-silu`**)

> 为什么不用 KataGo_Transformer 仓库的 `b11c96h4tfrs`:它标记 model version **15**,
> 而官方引擎对 transformer 的支持从 **version 17** 开始,直接导出的 .bin.gz 官方 C++ 加载不了。
> 自对弈闭环依赖官方引擎,所以学生网统一用**主仓库自带的 v17 transformer 配置**
> (2D RoPE + SwiGLU + RMSNorm)。

**学生模型:`b8c96h3tfrs-fson-silu`** —— v17 格式、0.94M 参数、主干 8×(attn+ffn) 全宽
transformer(每头 32 维,3 头;stem 3×3 卷积后位置信息全靠 2D RoPE)。N2 蒸馏与 N3 强化
从头到尾用这一个配置,不中途换架构;强度封顶再评估升级 `b14c192h6tflrs-fson-silu`(6.29M)并重走监督训练。

主仓库 1M 档候选(参数量实测,除拍板款外仅作对照):

| 配置 | 参数量 | 结构 | 备注 |
|---|---|---|---|
| `b5c48h3tfr` | 0.13M | 5×(attn+ffn) | 管线冒烟可用 |
| `b7c96h3tfrs` | 0.83M | 7×(attn+ffn) | 同族少一层 |
| `b8c96h3tfrs` | **0.94M** | 8×(attn+ffn) | **★ 学生模型(拍板)** |
| `b2b10c96h3tfrs` | 1.49M | 2 卷积残块 + 10×(attn+ffn) | 备选 |
| `b14c192h6tflrs` | 6.29M | 14×(attn+ffn),可学习 RoPE | 强度上限档 |

训练用主仓库 `python/train.py`(支持 `-use-muon`)。**老师**:官方 b11c768
(用户拍板,原计划 b18c384nbt)。

## 3. 训练管线与运行约束

### 3.1 N2 监督训练(已完成)

9 路走的是**蒸馏**路线,19 路走 **kata1 冷启动**路线:

- **9 路(完成,留作方法记录)**:NN 旁路采集 + 纯 PyTorch 蒸馏 —— 在自建 ONNX 后端
  (`cpp/neuralnet/onnxbackend.cpp`)加旁路记录器(`KATA_NNLOG=` 开启):自对弈(visits 8/4、
  噪声 0.35、贴目抖动 ±0.75、120 手上限)每个推理批次顺手落盘「教师输入特征 → 教师原始输出」
  (C++ 权威 fillRowV7,随机对称白送增强,2874B/行)。`training/distill_from_nnlog.py`
  memmap 直读,GPU 现算 softmax 目标 —— 无 npz、无 train.py、无第二次教师前向。
  实测 ~117 万行/小时(~3.4GB/h),采满 146 万行;蒸馏 2 epochs(~35 分钟,
  loss 3.64→2.34)→ 导出 → `models/student_9.onnx` → e2e 4:0 胜 2000 演棋 UCT(冒烟);
  等 visits=150 对弈 vs b6c96 三批合计 **16:10(61.5%,26 局)**。
- **19 路(在训)**:kata1 公开数据冷启动监督训练(`-pos-len 19`),学生配置同 §2;
  kata1 全量数百 GB,受 30G 磁盘上限约束按需下小块。
- 出炉流程:export_model_pytorch.py → .bin.gz → `dumponnx -nn-x-len 19 -nn-y-len 19
  -require-exact-nnlen` → 替换浏览器模型。**2026-10-03 已执行**:45 个导出快照中
  循环赛选出第 40 份,浏览器侧(nn-match 等 visits)对拍验证后落位(§1)。

### 3.2 N3 自对弈强化(待启动)

学生网自己的 selfplay → shuffle → train → **gatekeeper**(新一代必须赢过上一代才发布),
循环往复。全部是主仓库现成命令,我们只维护配置与调度脚本
(`training/run_n3_loop.sh` + `gatekeeper_aether9.cfg` 已就绪,待 19 路首个学生出炉后启动)。

**验收**:gatekeeper Elo 曲线持续上涨;每隔 N 代抽一个模型在浏览器实测
(防「训练强、浏览器弱」的编码 / 搜索不一致)。

### 3.3 运行约束(用户拍板,2026-10-01)

- **磁盘硬上限 30G**(整条训练链相关目录合计),哨兵 `training/disk_guard.sh`
  (每 10 分钟查,≥25G 先 SIGINT 优雅停、2 分钟未退强杀);
- **自对弈双保险停止**:`-max-games-total 10000` + 磁盘哨兵;手动停
  `pkill -INT -f "katago selfplay"`(会落盘在途数据);
- **selfplay 参数按「纯蒸馏」拍板**:visits 32/cheap 16,关对称采样 / side position /
  lead 估计 / fork / 非对称对弈,噪声权重 0.10,28 线程 + nnMaxBatchSize 256;
  实测 ~30 万手/小时 ≈ 4600 局/小时,1 万局 ≈ 2 小时,数据 ~250MB。

### 3.4 决策记录

- **「零搜索直出教师结果」否决(2026-10-01 实测)**:官方 `writetrainingdata` 强制人类棋谱
  元数据(KGS/OGS 段位),自对弈 SGF 全数被拒("Unknown rating status");真·零搜索需自写
  npz 生成器,有格式写错即静默污染的风险。visits 16-32 的自对弈在效果上已等价教师直出,
  故走旁路采集路线。
- **备选路线 B(仅当需要 fork 的训练增强时)**:用 KataGo_Transformer 仓库训练(它的
  Muon-KI / SWA 更激进),训完经主仓库 `export_model_pytorch.py` 的
  `-export-15-or-16-as-17` 桥接为 v17 再导 ONNX。代价是两仓库配置 / 数据目录手动对齐,
  仅在路线 A 不够快时启用。

## 4. ONNX 接口契约(`dumponnx` 产出的图,IO 名以 `onnxmodelbuilder.cpp` 为权威)

输入(全部 f32):

- `InputSpatial`:(N, 22, 19, 19) —— fillRowV7 空间特征(JS 版 `src/nn/features.js`,
  逐位对拍是全路线质量的锚);
- `InputGlobal`:(N, 19, 1, 1) —— 全局特征;
- `InputMask`:(N, 1, 19, 19) —— `-require-exact-nnlen` 无掩码图,占位喂 1。

输出(**5 个**,与 C++ ONNX 后端同构):

> **★ 视角约定(2026-10-03 双模型实证 + C++ 对证,必读)**:模型原始输出一律是
> **行棋方**视角,+ = 行棋方优 —— value / scoreValue / ownership 全部如此。
> 训练目标即按行棋方写(`trainingwrite.cpp` fillValueTDTargets:
> 「Training rows need things from the perspective of the player to move, so we flip
> as appropriate」,ownership 目标同理 ±1 = 行棋方 / 对手);C++ 后端解出后按
> `nextPlayer` 翻成白方视角存 `NNOutput`(`nneval.cpp`:
> 「the neural net gives us back the value from the perspective of the player」)。
> 实证:同一盘面分别以黑 / 白行棋编码喂图,value 的 ch0↔ch1 对调、ownership 整体变号
> (b6c96 v8 与学生 v17 行为一致)。**本引擎不转白方视角**,消费方各自按 side 换算:
> search 的 winLoss 本来就按行棋方回传(✓ 天然正确);worker 的 estimate / score
> 需翻成黑方视角(2026-10-03 修正 —— 旧实现没翻,白行棋时目差 / 热图 / 死子辅助反号)。

- `OutputPolicy`:(N, C, 19, 19) 策略 **logits**。C 按模型版本(`desc.cpp` PolicyHeadDesc,
  2026-10-03 核实,**修正旧契约的「6 通道」错误**):version < 12 → C=1(只有主策略;
  g170 占位网属此);version ≥ 12 → C=2([0] 主策略,[1] 乐观策略);
  v16 / v17 带 q 值 → C=4([2][3] 是 q 胜负 / q 目度,与策略无关)。
  训练头的 6 个面(主 / 对手回应 / soft / 乐观长短期)只有 [0][1] 进导出图。
  **乐观插值逻辑(权威:`onnxbackend.cpp`,2026-10-03 拍板照抄)**:对每个落点与 pass 做
  `p + (pOpt − p) × λ`,**在 logits 空间、softmax 之前**;λ 逐行随每次评估传入 ——
  根评估用 rootPolicyOptimism、树内用 policyOptimism,GTP/比赛配方为
  **树内 1.0、根 0.2**(selfplay / distributed 为 0)。老网(C=1)无乐观面,
  直接用通道 0 就是 KataGo 对老网的原生行为。本引擎现状:取通道 0、无插值 ——
  切换列入待办(§6 棋力线 ①);
- `OutputPolicyPass`:(N, C) pass 的 logits([0] 主策略;C ≥ 2 时 [1] 乐观,插值同上);
- `OutputValue`:(N, 3) 行棋方视角 胜 / 负 / 无结果 logits → softmax。本引擎压成
  winLoss 标量 —— 中国规则 7.5 无和棋、position superko 无无结果,标量即精确
  (拍板不做 WDL 三元);C++ 对面积计分把无结果 logit 压 −1e5(`nneval.cpp`),
  session.js 已按同口径实现(只按 e0+e1 归一);
- `OutputScoreValue`:(N, C, 1, 1) **按模型版本分通道(2026-10-03 核实,`onnxbackend.cpp`)**:
  version ≥ 9(含 v17 学生)为 6 通道:[0] scoreMean、[1] scoreMeanSq(→ stdev 可推导)、
  [2] scoreLead、[3] varTimeLeft、[4] shorttermWinlossError、[5] shorttermScoreError;
  更老版本通道递减(4/2/1,只有 scoreMean 系)。**该张量已在现有导出图里**,
  消费无需重新导出模型,按输出实际形状判别即可。
  **裸值后处理(乘数存在 .bin.gz 头 v≥13 段;学生模型头实测
  20/20/20/40/0.25/150,outputScale=1;公式权威 `nneval.cpp` v≥14 分支)**:
  scoreMean = raw0×20;scoreStdev = softplus(raw1)×20;scoreLead = raw2×20(已含贴目);
  varTimeLeft = softplus(raw3)×40;stWLerr = softplus(raw4×0.5)×√0.25;
  stScoreErr = softplus(raw5×0.5)×√150。
  **session.js 已消费(2026-10-03,显示层)**:scoreMean / scoreStdev / scoreLead /
  shorttermScoreError 逐行后处理接出,供形势判断目差与信息行根目差;
  树内效用消费 = 待办棋力线 ②;
- `OutputOwnership`:(N, 1, 19, 19) 逐点归属,**行棋方视角裸 pretanh 值**
  (2026-10-03 修正:图构建器与 C++ 各后端都直接输出裸值不过 tanh ——
  `onnxmodelbuilder.cpp` markOutput 直连、CUDA/ONNX 后端直赋 `whiteOwnerMap`;
  实测离分布局面 |值| 可超 1;`nneval.cpp` 训练指标亦名 ownership_pretanh)。
  形势判断热图 / 数子死子辅助消费时按视角约定换算黑方视角;

搜索侧消费:policy(softmax 后做先验)、value、ownership 现已消费;
scoreMean 系已消费到**显示层**(形势判断 / 信息行,2026-10-03);
乐观策略插值(棋力线 ①)与分数输出进树效用(棋力线 ②)待接入。

## 5. 搜索配置

### 5.1 已落地(2026-10-03:GTP 实战配方全套,search.js v3)

| 机制 | 本引擎现值(= GTP 配方) | 状态 |
|---|---|---|
| 效用函数 | winLoss + 0.1·static + 0.3·dynamic(atan 平滑期望,3 点 GH;动态中心 zeroW 0.20 / scale 0.75) | ✅(②) |
| cpuct 调度 | 1.0 + 0.45·ln((W+500)/500)·√(W+0.01)·父效用方差因子(prior 0.40/W 2.0/scale 0.85) | ✅ |
| FPU | 按已访问 policy 质量混父 NN 效用(pow 2);root 0.1 / 树内 0.2 | ✅ |
| 不确定度加权 | w = 0.25/(stWL + \|∂u/∂s\|·stScore + 0.25/8);老网恒 1 | ✅(②) |
| LCB 选点 | 效用方差 ESS 修正半径 5σ + radiusFactor² 权重奖励,资格线 0.15×参照权重 | ✅(②) |
| 选点精修 | noisePruning(share×2,scale 0.15)→ valueWeightExponent 0.25(t₃ CDF) | ✅ |
| 根评估 | 每次思考重算(λ=0.2):根先验刷新 + recentScoreCenter + FPU 混合 | ✅ |
| 乐观策略插值 | session 层,logits 空间 softmax 前;根 0.2 / 树内 1.0;老网退化通道 0 | ✅(①) |
| 根对称剪枝 | 保守式:劫点/禁点即弃;盘面自同构对称的等价着法先验置 0 | ✅ |
| 无用着剪枝 | 填自己真眼不进树;先验 < 1e-4 不建子(pass 除外) | ✅(保留) |
| NN 评估缓存 | 特征双种子 FNV → 2048 条 LRU(λ=1 口径;root 不入缓存) | ✅(保留) |
| 终局停着 | fillDameBeforePass:单官未清 pass 先验 ×1e-3(可关) | ✅(保留) |
| 批量 / 树复用 | 单槽管线动态批(v4.1,2026-10-04):GPU 估值第 N 批时同步攒第 N+1 批,在途结果一到即发射手头半批(waitPopUpToN 单线程投影);批上限 = createSession 现测校准(吞吐 ≥ 最优 90% 的最小批)再按预算压 stale(≤ 预算/16,下限 2)+ 虚拟损失;根沿「己方+对方」下移整体继承 | ✅(2026-10-04) |
| 认输 | 白方视角值连续 3 手越 −0.90,手数 ≥73(play.cpp 口径);悔棋/新局作废历史 | ✅(②) |
| 温度选点 | 权重制 w^(1/T) 对数空间抽样;调用方传入 + 半衰(拍板口径) | ✅(保留) |
| 图搜索 | useGraphSearch(repBound 11):节点表跨手持久;键控按 graphhash 语义(最后一步周边空域 > 11 → 状态键合并,局部战斗 → 链式路径唯一);惰性子节点 + 边访问缩放(getChildWeight)+ 在途跳过(循环守卫)+ 每手 mark-and-sweep GC | ✅(2026-10-03 深夜) |

未对齐(记录在案):retrospective 权重回溯削减(getReducedPlaySelectionWeight)——
LCB 奖励承担同等稳定职能;endingScoreBonus 属停着行为组,拍板不做。
温度选点与 KataGo `getChosenMoveLoc` 同分布(onlyBelowProb=1 档,权重制)。

### 5.2 参数基准(2026-10-03 发现,同日拍板采用 GTP 配方)

上表现值的出处是 `searchparams.h` 构造函数口径;**KataGo 对战实际生效的是
`cpp/program/setup.cpp` 中 SETUP_FOR_GTP 的硬编码配方**,两者差异很大:

| 参数 | 本引擎现值 | setup.cpp GTP 实战配方 |
|---|---|---|
| cpuct | 1.1 + 0.6·ln((N+4096)/4096) | 1.0 + 0.45·ln((N+500)/500) |
| FPU | 0.25 / root 0.25 | 0.2 / root 0.1 |
| policyOptimism | 不消费(取通道 0) | 1.0(树)/ 0.2(根),乐观通道在用 |
| utility | 只有 winLoss | static 0.1 + dynamic 0.3(center 0.2/scale 0.75) |
| cpuctUtilityStdevScale | 无 | 0.85(开) |
| uncertainty | 无 | 开(coeff 0.25,按 shorttermScoreError) |
| valueWeightExponent / noisePruning | 无 | 0.25 / 开 |
| 图搜索 / 根对称剪枝 | 无 | 开 / 开(repBound 11) |
| fpuParentWeightByVisitedPolicy | 无 | 开(pow 2.0) |
| 温度 | 默认 0 | 0.5 → 0.1(半衰 19 手) |
| 认输 | 胜率 < 0.04 且 ≥250 访问,一次即认 | −0.90 连续 3 手(gtp_example.cfg) |

两组都「来自 KataGo」但不一致。**2026-10-03 拍板:采用 GTP 配方,当日已切换落地**
(§5.1;「本引擎现值」列已随之更新为切换后的值,原 1.1+0.6·ln/4096、FPU 0.25 口径废止)。

## 6. 待办(2026-10-03 裁决后)

范围裁决见 README「产品边界」;以下是与 GTP 实战配方的全部残余技术差距,按优先序。

### 棋力线(✅ 全部完成,2026-10-03 深夜)

GTP 配方全套 + 图搜索全部落地(search.js v4 图搜索重写;全量单测含图搜索专项通过;
对弈 A/B 按用户指示暂缓):

- ~~① GTP 常数 + 乐观策略~~;~~② NN 输出进树~~;~~③ 图搜索~~;~~④ FPU 混合~~;
  ~~⑤ valueWeightExponent + noisePruning~~;~~⑥ 根对称剪枝~~ —— 细节见 §5.1;
- **③ 图搜索**(最后落地):节点表跨手持久(Worker 生命期),转置局面共享子图;
  键控按 `graphhash.cpp` 语义 —— 最后一步周边空域 > repBound(11) → 状态键合并
  (大范围着法后历史局部性弱),局部战斗 → 链式路径唯一(杜绝短循环);PASS 链式;
  惰性子节点(展开只建「着法+先验」,节点本体下降首次经过该边时建表/查表);
  边访问缩放(`getChildWeight = rawWeight × edgeVisits / max(nodeVisits,1)`);
  循环守卫 = 在途节点打戳跳过(单线程下等价于 KataGo 的 graphPath 到达终止);
  每手 mark-and-sweep GC。同进程 A/B:空盘转置密集局面单叶 −42%,80 手持平;
  同根连续思考第二次推理调用大减(对齐测试 8a)。
  **与 eval-cache 分层互补不是取代**(已核实 `searchupdatehelpers.cpp:94`):
  图搜索管搜索内节点共享;eval-cache 管跨手评估复用。脚注:KataGo GTP 默认
  useEvalCache=false(仅 analysis 接外部缓存);本引擎保留内存 LRU 是拍板裁决。
  已记录偏差:maybeCatchUpEdgeVisits 未实现(单线程收益小);转置子树着法合法性
  按首访路径生成不重查(KataGo 同);根键 = fnv(整局着法)+状态键(语义等价)。

### 进度线

- ~~19 路学生出炉 → dumponnx 导出 → 替换浏览器模型~~ **完成(2026-10-03)**:
  第 40 份(s68320512)落位 `models/b8c96h3tfrs_19.onnx`(§1),显示层消费
  OutputScoreValue(形势判断网端目差 + 信息行根目差)随同落地;
- N3 循环启动(§3.2);
- N4 收尾:中端手机实测 + 体积闸门核定(fp32 模型 4~25MB 懒加载 + HTTP 缓存,
  引擎 chunk 预算单独放宽,更新 WebOS 侧定位标记)。

### 裁决记录(2026-10-03,原「存疑待 A/B」五项全部落定)

- **搜索参数基准**:采用 GTP 配方(§5.2),切换 = 棋力线 ①;
- **乐观策略通道**:照抄 KataGo 逻辑(§4 spec;通道按版本 1/2/4,logit 空间插值,
  λ 根 0.2 / 树 1.0,老网无此面),接入 = 棋力线 ①;
- **温度**:由调用方传入,引擎不自带任何温度策略;需要随机时必须带衰减
  (半衰已实现)—— 现设计即拍板结果,开局温度给不给、给多少是对弈页调用方的事;
- **eval cache**:类置换表(hash 表存 NN 评估结果),内存 LRU 保留、落盘不做(维持原裁决);
- **参数格式**:内置唯一一套,外部不可配置(维持原裁决)。

### 裁决记录(2026-10-04:批大小自适应)

- **机制对齐 KataGo,不做运行时热调**:批上限 = createSession 现测校准值
  (预热丢 shader 编译 → 各档 rows/ms → 吞吐 ≥ 最优 90% 的最小批,强设备自动
  选小批压 stale、弱设备选大批保吞吐),搜索侧再按预算压 stale(≤ 预算/16)。
  对应物:`katago benchmark`(离线 CLI、人写 config)在浏览器语境改为
  「加载时自动、产物为内部常量」;运行时批构成由涌现机制决定,与
  `waitPopUpToN`(nneval.cpp:839 / threadsafequeue.h:172)同构;
- **不做跨加载缓存**:校准全程 <2.5s(硬性时间预算),设备热态/负载漂移下
  每加载现测比信任旧值稳;换自研 WebGPU 后本函数不改(只依赖 evalBatch 契约);
- **单线程注意**:攒批是同步块,宏任务无法批中打断 —— 「结果一到提前发射」
  只作用于末批/死端结算后的余量批;批构成自适应主要靠校准的上限,这一点
  与 KataGo 多线程(多生产者队列深度涌现)有结构性差异,记录在案。

## 7. 明确不做(技术侧;产品级不做清单在 README)

- 不在浏览器里训练或跑自对弈(训练全在本地 GPU 的 C++ 侧);
- 不自研 WebGPU 算子 —— **2026-10-06 推翻并落地**(aethernn,见 docs/WEBGPU_ENGINE_RESEARCH.md
  §9:对拍全绿后默认切换;ort-web 保留为 `?engine=ort` 逃生舱,真机 A/B 达标后删除);
- 不做 INT8 / fp16 量化(fp32 起步,体积与速度真成瓶颈再上,主仓库有现成量化工具);
- 不为「看起来强」堆未验证的搜索技巧 —— 每一项对拍 / 自对弈数据说话;
- 不自写训练侧搜索 / 数据格式(selfplay / shuffle / train / gatekeeper / dumponnx 全用官方工具链)。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| ort-web WebGPU 对新导出的 ONNX 图兼容性差 | **2026-10-06 随 aethernn 落地整体消除**:架构固定、一次验证(.aewn 计划互验,架构不符启动即报);ort-web 仅存逃生舱 |
| 特征编码不一致导致「训练强、浏览器弱」 | 逐位对拍是硬验收;N3 每代抽测浏览器实局 |
| 训练数据量 / 质量不足 | 学生已用 kata1 冷启动保底;N3 曲线不行先翻量,老师可换更强官方网 |
| WebGPU 移动端覆盖(Safari 旧版等) | 不支持即明确报错(裁决:不做慢速兜底);N4 实测圈定可用范围 |
| 两仓库(KataGo / AetherGo)联调摩擦 | 本文即契约:接口(IO 名、特征表、参数口径)以本文 + README 为准,改动先改文档 |
| 参数基准混乱(构造默认 ≠ GTP 配方 ≠ 现值) | §5.2 对照表为准;引用 KataGo 参数一律注明出处文件;A/B 后回写本表 |

## 9. 里程碑存档

- **2026-10-06**:自研 WebGPU 引擎 aethernn 立项并当日完成 Node 侧全链(packer → 15 WGSL
  内核 → 单 pass/单 submit 宿主 → 8 对称 GPU 侧置换 + 特征零拷贝直传);对拍闸门全绿
  (vs ort CPU golden:policy 1.4e-5 / winLoss 1.0e-6 / ownership 1.9e-6;WGSL vs CPU 参考
  2.7e-5),现有测试与 nn-e2e 全过;默认引擎切 aethernn,ort-web 降为逃生舱,
  运行时依赖归零(5.5MB CDN wasm 不再加载)。真机性能 A/B(≥ort-web×0.9 闸门)待浏览器实测。
- **2026-10-01**:立项;N0~N2 全链打通(规则 / 特征对拍 / ort-web 会话 / 异步 PUCT /
  Worker 接线);蒸馏路线定稿(旁路采集 + PyTorch 蒸馏,否决 npz 与零搜索直出);
  9 路学生 35 分钟蒸馏出炉,等 visits 16:10 胜官方同尺寸老网。
- **2026-10-02**:盘面升级 19×19(全链尺寸参数化,贴目 7.5,难度改按访问数;
  数子明细窗 + 形势判断上线);UCT 随机演棋引擎与 WASM 回退移除(NN 成为唯一引擎);
  搜索机制六项对齐 KataGo(§5.1);死子标注(ownership 辅助)+ 双停终局流程落地;
  19 路学生(kata1 冷启动)开训;差距全面盘点(§5.2 / §6 的来源)。
- **2026-10-03**:产品边界拍板,README 完全重写(待办 / 不做 / 存疑的唯一权威);
  发现并修正参数基准问题(§5.2)与乐观策略通道口径(§4);本文整体重写。
  同日晚五项存疑全部裁决(§6 裁决记录):采用 GTP 配方、乐观策略照抄 KataGo 逻辑
  (核实:导出图通道按版本 C∈{1,2,4},logit 空间插值,λ 树 1.0 / 根 0.2,老网无乐观面 ——
  顺带修正旧契约「6 通道」错误)、温度外部传入 + 衰减、eval cache 维持内存版、
  参数不可外部配置。
  **同日再晚:19 路学生第 40 份(s68320512)出炉落位浏览器**(循环赛 40:41 63:57、
  40:45 75:45 选出;dumponnx 19 路 8.2MB;旧占位全删,仅支持新模型)。模型输出
  **行棋方视角**契约实证入档(§4,双模型 + C++ 对证),并修正 estimate / 数子在
  白行棋时的反号 bug;OutputScoreValue 后处理公式(×20 / softplus×20 / √150)
  与「裸 pretanh ownership」口径核实,session 消费落地(显示层:形势判断网端目差、
  信息行根目差;noResult 抑制与 C++ 对齐)。
  **同日深夜:GTP 实战配方全套落地(search.js v3)** —— 效用函数 / 加权统计 /
  PUCT 全公式对齐 / 真实 LCB(ESS)/ uncertainty 加权 / noisePruning /
  valueWeightExponent / FPU 混合 / 根评估重算(λ=0.2)/ 根对称剪枝(保守式)/
  认输精化(−0.90 连续 3 手,play.cpp 口径);session 补乐观插值与 stWinlossError 通道;
  worker 补 ownership tanh 归一与认输历史;对齐项测试新增 λ 接线与 score 头兼容断言,
  全量单测通过(对弈 A/B 按用户指示暂缓)。
  **同日深夜其三:图搜索落地(search.js v4)** —— 节点表跨手持久 + graphhash 语义键控
  (状态合并/链式路径唯一由「最后一步周边空域 vs repBound 11」裁决)+ 惰性子节点 +
  边访问缩放 + 在途跳过循环守卫 + mark-and-sweep GC;对齐测试新增子图复用 / GC 边界 /
  转置合并 / 清表隔离四项;同进程 A/B:空盘单叶 −42%,80 手持平。**棋力线全部完成。**
  **同日深夜其二:特征编码增量化(KataGo recentBoards 同款)** —— rules.js 落地滚动
  盘面环(ringSnapshot/Restore = 每 playout 拷 rootHistory 的等价物),features.js
  征子历史通道(15/16)改读环,replayBoard 降级为参考实现 `encodeFeaturesReplay`;
  features-test 新增「环 vs 重演」逐位对拍模糊测试(12 局 × 148 局面,含劫争 /
  提子 / pass / 换分支)兜底。编码成本消除 O(手数) 重演:80 手 807→295µs(−63%);
  240 手 2585→2325µs(重演虽消,征子搜索本身随局面复杂度增长,成为新大头 ——
  与 KataGo 每评估现算征子同构;进一步优化方向:iterLadders 结果按局面键缓存)。
  **2026-10-05:模型瘦身——RoPE 查找表改 ONNX 图内现场计算** —— 8 个注意力块 q/k 的
  ropecos/ropesinsigned/ropeswapidx 共 48 份 initializer(4.44MB,占模型 54%)实为
  同一张 theta=100 棋盘 2D RoPE 表的复制(theta=100、head_dim=32、19×19 行主序,
  公式逐值核实最大误差 6e-8);rope-graph 版在图首以 Mul/Concat/Unsqueeze/Reshape/
  Sin/Cos 现场生成共享表,16 个应用点结构不动,48 表全删:8.22MB→3.79MB。
  全 5 输出与原模型相对误差 ~5e-7(batch 1/8),CPU/CUDA 速度持平,nn-e2e 完整
  自对弈通过;ort-web 1.30.0 WebGPU EP 核实含 Sin/Cos/Mul/Concat kernel
  (jsep/webgpu/op-resolve-rules.ts),Reshape 现网已用、Unsqueeze 同为元数据类;
  IO 契约不变,前端零改动。转换脚本 `training/make_rope_ongraph.py`(删表前逐表
  校验 <1e-5,theta/结构不符即 assert,换模型/换 theta 不会静默算错)。

## 10. 参照物(本地路径)

- 引擎与训练主仓库:`/home/a/go/KataGo`(C++ selfplay/gatekeeper/dumponnx;python train.py);
- transformer 训练参照(备选 B):`/home/a/go/KataGo_Transformer`;
- 19 路学生训练现场:`/home/a/go/trainrun`(run_train.sh / b8c96h3tfrs_run);
- 官方网络与 Elo 对照:katagotraining.org、`KataGo/docs/NetworkArchitectures.md`;
- 特征定义权威:`KataGo/cpp/neuralnet/nninputs.cpp` 的 `fillRowV7`;
- v17 transformer 结构权威:`KataGo/python/katago/train/model_pytorch.py` + `modelconfigs.py`;
- **搜索参数实战配方的权威:`KataGo/cpp/program/setup.cpp`(SETUP_FOR_GTP 硬编码默认)**,
  其次 `gtp_example.cfg`;`searchparams.h` 构造默认不是对战配方(§5.2);
- ONNX 图构建权威:`KataGo/cpp/neuralnet/onnxmodelbuilder.cpp`(IO 名与通道数);
  输出解释权威:`onnxbackend.cpp`;
- **策略通道数与乐观插值权威**:`desc.cpp` 的 PolicyHeadDesc(版本 → C∈{1,2,4})+
  `onnxbackend.cpp` / `eigenbackend.cpp` 的插值实现(logit 空间,softmax 前,λ 逐行)。
