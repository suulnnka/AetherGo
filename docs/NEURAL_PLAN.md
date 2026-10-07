# AetherGo 技术口径(架构 / 训练管线 / 模型 / 契约 / 搜索配置)

> 2026-10 立项;2026-10-03 按产品边界拍板整体重写;**2026-10-08 按落地现状再次整体重写**
> (aethernn 唯一引擎 + i8f16 单形态 + 搜索机制移植完成后,旧快照 / 待办 / 裁决大量过期)。
> 职责划分:**README = 产品边界 / 待办 / 不做的唯一权威**;本文 = 技术口径:架构、训练管线、
> 模型、IO 契约、搜索配置与差距的技术细节。两处不一致时以 README 为准,并回来改本文。
>
> 目标:**浏览器里的 19×19 强棋力人机对弈**。路线:用 KataGo 官方工具链做
> 「监督训练(kata1 冷启动)→ 自对弈强化」训练一个小型 transformer 模型,
> 导出 ONNX,打包为私有权重 blob,浏览器端自研 WebGPU 引擎(aethernn)推理 +
> JS 侧 PUCT 搜索。

---

## 0. 总体架构:三件套,各司其职

| 角色 | 用什么 | 不做什么 |
|---|---|---|
| **训练引擎** | KataGo 官方 C++ 二进制(selfplay / gatekeeper / dumponnx) | 不自己写训练侧搜索 |
| **模型** | 主仓库 v17 transformer 配置(见 §2) | 不在浏览器里训练 |
| **对弈引擎** | AetherGo 纯 JS:特征编码 → aethernn 推理(WebGPU,i8f16)→ PUCT 搜索 | 不做随机演棋(NN 叶子评估取代) |

关键认知:**自对弈数据生成用官方 C++ 引擎**(快几个数量级),JS 引擎只负责「在浏览器里陪人下棋」。
整条训练闭环全部使用现成工具,我们只写配置和浏览器侧代码。

### 技术选型(已拍板)

- **纯 JS,不上 WASM/Zig/Rust**。NN 推理占 95% 以上算力且全部在 WebGPU(自研 aethernn)里,
  JS 只做调度与树操作。WASM 的收益场景(CPU 密集演棋)两头都不占,还要跨边界调 WebGPU,不值。
  (katago-webgpu 的 WASM 产物路线曾评估,**拒**,见 [WEBGPU_ENGINE_RESEARCH](WEBGPU_ENGINE_RESEARCH.md) §7。)
- **搜索必须异步化**:WebGPU 推理是异步的,PUCT 主循环 async 化,待评估叶子攒成 batch 提交;
  攒批语义 = KataGo 多线程投影(v4.5,§5.1)。
- **模型 pos-len 跟随盘面(现 19)**:dumponnx 用 `-nn-x-len 19 -nn-y-len 19
  -require-exact-nnlen` 导出无掩码图(掩码恒 1 已折叠);RoPE cos/sin 表由 packer
  打包期预算进 `.aewn`(曾经历「图内 Sin/Cos 子图」形态,2026-10-06 起 packer 吸收该职能)。
- **权重载体 = 私有 blob `.aewn`**(2026-10-06 起):ONNX 只是训练侧产物与 packer 输入,
  浏览器不加载、不解析;2026-10-08 起 blob 形态唯一为 i8f16(`parseAewn` 只认 dtype=1),
  ONNX / fp32 blob 均已出库。

## 1. 当前状态快照(2026-10-08)

**引擎侧(全部已落地,含测试):**

- 规则:气尽提子 / 禁自杀 / position superko(Zobrist,零分配 make/unmake)、中国规则数子、
  Benson 死子 + NN ownership 辅助标注、双停终局流程;
- 特征编码器:`fillRowV7` 的 JS 版,与 selfplay 训练行逐位对拍通过(1500 行),
  征子通道与 C++ 原生实现对拍 8393 链零分歧;征子历史走滚动盘面环(编码不随手数增长);
- NN 搜索:异步 PUCT(图搜索 + 随机对称 + 重算式统计 + subtreeValueBias + 攒批多线程投影,
  全表见 §5.1)+ 跨手树复用 + 动态认输 + 温度选点;难度四档 = 80/200/400/800 访问;
- 推理:**aethernn 唯一引擎**(自研 WebGPU,84 dispatch / 单 pass / 单 submit / 单回读;
  ort-web 已于 2026-10-08 整体移除);权重唯一形态 i8f16
  (`models/b8c96h3tfrs_19.i8.aewn`,1.14MB);设备无 WebGPU 或 `shader-f16` 直接报错;
- 批校准:`src/nn/calibrate.js`(加载时现测,吞吐 ≥ 最优 90% 的最小批;
  2026-10-07 修复五项失真:冷启动截断 / fast-3 均值 / 少样本档丢弃 / 轮转交错采样 /
  峰值锚取前二快档均值);本机真机生产路径冷加载校准结果 i8=16;
- 对弈页:人机 / 双人、数子明细窗(死子手改)、形势判断(ownership 热图 + 目差)、
  悔棋 / 换边 / 主题切换;测试套件 Windows 原生可跑(2026-10-07 起)。

**模型侧:**

- **唯一权重**:`models/b8c96h3tfrs_19.i8.aewn`(1.14MB)—— 19 路学生 b8c96h3tfrs
  **第 40 份(s68320512)**,循环赛 40:41 63:57、40:45 75:45、41:45 69:51 拍板选用
  (训练侧 C++ 128 visits 各 120 局);v17 transformer,929,641 参数,0.91M 参数量化
  (trunk 逐输出通道对称 int8 + 头部/norm/RoPE 留 f32,W8A16);
- ONNX(3.79MB rope-graph 版)与 fp32 blob 已出库(2026-10-08):均为训练管线可再生产物,
  仓库不再携带;quant-test 的 golden 对照经 `QONNX` 环境变量指路,不在场自动跳过;
- 9 路蒸馏学生 `models/student_9.onnx` 已随旧占位时代退役(方法记录见 §3.1)。

**测试基线(全绿;命令见 README「测试」):** 模糊测试(40 局 × 420 手,子数守恒 +
禁全同不变量 + 逐步撤销重演)、特征对拍(featdiff)、征子对拍(ladderdiff)、
搜索机制对齐项(katago-align)、Worker 冒烟(worker-smoke)、温度分布(nn-temp)、
aethernn 对拍(wgsl-test:WGSL vs cpuref-Q,i8f16 口径)、量化闸门(quant-test 三层:
激活范围 / cpuref-Q vs ort golden(QONNX 指路,缺席跳过)/ WGSL-Q vs cpuref-Q)。

**性能基线(同机 RTX 5060,真实浏览器,`test/browser-ab/`):**

- 批 1-64 吞吐扫描:aewnn 全档压过 ORT-Web(批 64:i8 1088 vs 718 行/s,ORT 批 32 有悬崖);
- 1024 visits 端到端:aewnn 456-555 visits/s(随批上限);同网络 KataGo CUDA(8-64 线程)
  2528-4905 visits/s;差距分解 = 环境差 1.5× × 引擎差 5.9×(KataGo 侧 NN 占墙钟 83%、
  批填满率 96%);详见 [katago-cuda-bench-1024v.md](../test/browser-ab/katago-cuda-bench-1024v.md);
- 三后端同网络容器 6 局循环赛(onnx / aewnn f32 / aewnn i8)= 1:1:1(棋力无损实证,
  此为 2026-10-08 移除 onnx/f32 通道的对弈级依据);
- 真机对局口径(批 4):i8 430→615 行/s(flashB 改造,+43%);f32 64v 搜索 242→165ms。

**对战基线(vs KataGo@64v,GPU 双侧):** 批 1 口径累计 **11-13/24**(≈46%,统计平手);
批 4 口径在 sym 序列化 bug 修复后首轮 **3-3**(首次追平批 1 基线;历史 3-21/24 为
bug 期数据)。测强度一律 MBATCH=1(§6 强度线)。

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

9 路走**蒸馏**路线(方法记录),19 路走 **kata1 冷启动**路线(已出炉落位):

- **9 路(完成,留作方法记录)**:NN 旁路采集 + 纯 PyTorch 蒸馏 —— 在自建 ONNX 后端
  (`cpp/neuralnet/onnxbackend.cpp`)加旁路记录器(`KATA_NNLOG=` 开启):自对弈(visits 8/4、
  噪声 0.35、贴目抖动 ±0.75、120 手上限)每个推理批次顺手落盘「教师输入特征 → 教师原始输出」
  (C++ 权威 fillRowV7,随机对称白送增强,2874B/行)。`training/distill_from_nnlog.py`
  memmap 直读,GPU 现算 softmax 目标 —— 无 npz、无 train.py、无第二次教师前向。
  实测 ~117 万行/小时(~3.4GB/h),采满 146 万行;蒸馏 2 epochs(~35 分钟,
  loss 3.64→2.34)→ 导出 → 等 visits=150 对弈 vs b6c96 三批合计 **16:10(61.5%,26 局)**。
- **19 路(完成)**:kata1 公开数据冷启动监督训练(`-pos-len 19`),学生配置同 §2;
  2026-10-03 从 45 个导出快照中经循环赛选出第 40 份(s68320512)落位(§1)。
- 出炉流程:export_model_pytorch.py → .bin.gz → `dumponnx -nn-x-len 19 -nn-y-len 19
  -require-exact-nnlen` → `training/pack_aewn.py` 打包 `.aewn` → 替换浏览器权重。

### 3.2 N3 自对弈强化(待启动)

学生网自己的 selfplay → shuffle → train → **gatekeeper**(新一代必须赢过上一代才发布),
循环往复。全部是主仓库现成命令,我们只维护配置与调度脚本
(`training/run_n3_loop.sh` + `gatekeeper_aether9.cfg` 已就绪,未启动)。

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

## 4. IO 契约(权威:`dumponnx` 产出的图,IO 名以 `onnxmodelbuilder.cpp` 为准)

> ONNX 文件已出库(2026-10-08),但**契约不变**:aewnn 执行计划逐块镜像该图,
> packer(`training/pack_aewn.py`)以其为输入;本节是训练侧与浏览器侧的共同语言。

输入(全部 f32):

- `InputSpatial`:(N, 22, 19, 19) —— fillRowV7 空间特征(JS 版 `src/nn/features.js`,
  逐位对拍是全路线质量的锚);
- `InputGlobal`:(N, 19, 1, 1) —— 全局特征;
- `InputMask`:(N, 1, 19, 19) —— `-require-exact-nnlen` 无掩码图,占位恒 1
  (aewnn 中已折叠,不占 dispatch)。

输出(**5 个**,与 C++ ONNX 后端同构):

> **★ 视角约定(2026-10-03 双模型实证 + C++ 对证,必读)**:模型原始输出一律是
> **行棋方**视角,+ = 行棋方优 —— value / scoreValue / ownership 全部如此。
> 训练目标即按行棋方写(`trainingwrite.cpp` fillValueTDTargets);C++ 后端解出后按
> `nextPlayer` 翻成白方视角存 `NNOutput`(`nneval.cpp`)。**本引擎不转白方视角**,
> 消费方各自按 side 换算:search 的 winLoss 本来就按行棋方回传(✓ 天然正确);
> worker 的 estimate / score 需翻成黑方视角(2026-10-03 修正落地)。

- `OutputPolicy`:(N, C, 19, 19) 策略 **logits**。C 按模型版本(`desc.cpp` PolicyHeadDesc):
  version < 12 → C=1;version ≥ 12 → C=2([0] 主策略,[1] 乐观策略);
  v16 / v17 带 q 值 → C=4([2][3] 是 q 胜负 / q 目度,与策略无关)。本模型 C=2。
  **乐观插值(权威:`onnxbackend.cpp`)已落地(session 层)**:对每个落点与 pass 做
  `p + (pOpt − p) × λ`,**在 logits 空间、softmax 之前**;λ 逐行随每次评估传入
  (rows[i].optimism)—— 根评估 rootPolicyOptimism(0.2)、树内 policyOptimism(1.0);
- `OutputPolicyPass`:(N, C) pass 的 logits([0] 主策略,[1] 乐观,插值同上);
- `OutputValue`:(N, 3) 行棋方视角 胜 / 负 / 无结果 logits → softmax。本引擎压成
  winLoss 标量 —— 中国规则 7.5 无和棋、position superko 无无结果,标量即精确
  (拍板不做 WDL 三元);无结果 logit 压 −1e5 口径(`nneval.cpp`)已按同款实现;
- `OutputScoreValue`:(N, 6)(v≥9 六通道):[0] scoreMean、[1] scoreMeanSq、
  [2] scoreLead、[3] varTimeLeft、[4] shorttermWinlossError、[5] shorttermScoreError。
  **裸值后处理(乘数存在 .bin.gz 头;本模型 20/20/20/40/0.25/150;公式权威
  `nneval.cpp` v≥14 分支)**:scoreMean = raw0×20;scoreStdev = softplus(raw1)×20;
  scoreLead = raw2×20(已含贴目);varTimeLeft = softplus(raw3)×40;
  stWLerr = softplus(raw4×0.5)×√0.25;stScoreErr = softplus(raw5×0.5)×√150。
  **消费状态:全部落地** —— 显示层(形势判断网端目差 / 信息行根目差,2026-10-03)+
  树内效用(不确定度加权消费 stWL/stScore,2026-10-03 随 GTP 配方落地);
- `OutputOwnership`:(N, 1, 19, 19) 逐点归属,**行棋方视角裸 pretanh 值**
  (图构建器与 C++ 各后端都直接输出裸值不过 tanh,实测离分布局面 |值| 可超 1)。
  形势判断热图 / 数子死子辅助消费时按视角约定换算黑方视角。

aewnn 附加随行契约(2026-10-06 起,`src/nn/session.js` 为准):

- `rows[i].sym`(可省,缺省 0):8 对称随行下发,**GPU 侧置换**(stem gather 表),
  policy/ownership 由引擎逆置换回恒等系 —— 搜索侧特征零拷贝直传,不做 CPU permute;
- `rows[i].optimism`(可省,缺省 1.0):乐观插值 λ 逐行传入;
- `evalBatch` 返回同序数组(policy 插值后 logits / policyPass / winLoss / scoreLead /
  scoreMean / scoreStdev / shorttermScoreError / ownership),契约见 session.js 头注。

## 5. 搜索配置

### 5.1 机制全表(2026-10-08;基准 = GTP 实战配方 + 后续机制移植)

| 机制 | 本引擎现值 | 状态 |
|---|---|---|
| 效用函数 | winLoss + 0.1·static + 0.3·dynamic(atan 平滑期望,3 点 GH;动态中心 zeroW 0.20 / scale 0.75) | ✅(2026-10-03) |
| cpuct 调度 | 1.0 + 0.45·ln((W+500)/500)·√(W+0.01)·父效用方差因子(prior 0.40/W 2.0/scale 0.85) | ✅(2026-10-03) |
| FPU | 按已访问 policy 质量混父 NN 效用(pow 2);root 0.1 / 树内 0.2 | ✅(2026-10-03;2026-10-06 修正「已访问」口径) |
| 不确定度加权 | w = 0.25/(stWL + \|∂u/∂s\|·stScore + 0.25/8) | ✅(2026-10-03) |
| LCB 选点 | 效用方差 ESS 修正半径 5σ + radiusFactor² 权重奖励,资格线 0.15×参照权重 | ✅(2026-10-03) |
| 选点精修 | noisePruning(share×2,scale 0.15)→ valueWeightExponent 0.25(t₃ CDF) | ✅(2026-10-03) |
| 根评估 | 每次思考重算(λ=0.2):根先验刷新 + recentScoreCenter + FPU 混合 | ✅(2026-10-03) |
| 乐观策略插值 | session 层,logits 空间 softmax 前;根 0.2 / 树内 1.0 | ✅(2026-10-03) |
| 根对称剪枝 | 保守式:劫点/禁点即弃;自同构等价着法先验置 0 | ✅(2026-10-03) |
| **重算式节点统计** | 每回传自叶向根从「子边统计 + 自身 NN 评估」重算;子权重 = getChildWeight 边分摊,good 子按先验序过 pruneNoiseWeight + valueWeight 降权;回传统计一律加权;vl 移出统计改选点期效用混合 + 分母膨胀(getExploreSelectionValueOfChild 同款) | ✅(2026-10-06,v4.2;同日修复回传加权缺失与 vl 方向符号两个 bug) |
| **搜索随机对称** | 每评估随机取 8 对称之一(nnRandomize 同款):spatial 全通道置换,policy/ownership 逆置换;评估缓存键取变换后特征;`opt.symmetry===false` 关,rngSeed 可复现;**默认开**(对齐 KataGo) | ✅(2026-10-06 移植,当日因交互崩坏临时默认关,2026-10-07 根因修复后恢复默认开) |
| **subtreeValueBias** | GTP 默认 0.45:同「行棋方+上二手+落点 5×5 局部形(8 对称规范)+劫」签名的节点共享表项;重算累计 (子树均值−自身评估)·origTotal^0.85;GC 回退 80% 贡献 | ✅(2026-10-06,v4.4) |
| **攒批语义** | KataGo 多线程投影(v4.5):批上限 = 在途评估总量 T(在飞 + 待发 ≤ T,统计盲区 ≤ T−1),死端/预算边界机会主义发射,批序严格;vl 符号修复后 T4 vs T1 剂量反应恢复中性(6-6) | ✅(2026-10-06,v4.5;**测强度一律 MBATCH=1**,与 KataGo 单线程同语义) |
| 批上限校准 | 加载时现测(calibrate.js):吞吐 ≥ 最优 90% 的最小批,再按预算压 stale(≤ 预算/16);2026-10-07 修复五项采样失真 | ✅(2026-10-04 立,10-07 修) |
| 无用着剪枝 | 填自己真眼不进树;先验 < 1e-4 不建子(pass 除外) | ✅(保留) |
| NN 评估缓存 | 特征双种子 FNV → 2048 条 LRU;键随 sym 取变换后特征(8 子缓存) | ✅(保留 + 2026-10-06 扩展) |
| 终局停着 | fillDameBeforePass(单官未清 pass 先验 ×1e-3)+ **终局 pass 守门**(根评估 ownership 判据,KataGo 终选层同款,常开产品护栏) | ✅(守门 2026-10-05/06) |
| 特征 passing hacks | 停一手会终局且面积数子+贴目非胜 → 全隐历史、不置 passWouldEndGame 位(nninputs.cpp:2046,推理期口径) | ✅(2026-10-06) |
| 认输 | 白方视角值连续 3 手越 −0.90,手数 ≥73(play.cpp 口径);悔棋/新局作废历史 | ✅(2026-10-03) |
| 温度选点 | 权重制 w^(1/T) 对数空间抽样;调用方传入 + 半衰(拍板口径) | ✅(保留) |
| 图搜索 | useGraphSearch(repBound 11):节点表跨手持久;graphhash 语义键控(空域 > 11 → 状态键合并,局部战斗 → 链式路径唯一);惰性子节点 + 边访问缩放 + 在途跳过循环守卫 + mark-and-sweep GC | ✅(2026-10-03,v4) |
| 跨手树复用 | 根沿「己方+对方」下移整体继承;与图搜索 / eval-cache 三层复用互补 | ✅(保留) |

**未对齐(记录在案)**:

- retrospective 权重回溯削减(getReducedPlaySelectionWeight)—— LCB 奖励承担同等稳定职能;
- 转置边访问追平(maybeCatchUpEdgeVisits):两版实现(纯记账 / 合成回传)A/B 均不利,
  **默认关闭保留代码,动它前先读提交历史**(8b06792);
- endingScoreBonus 属停着行为组,拍板不做;wideRootNoise 为 KataGo ANALYSIS 专属参数,
  本引擎不实现(差分对拍时对侧须置 0,§6 差分注记)。

### 5.2 参数基准(2026-10-03 发现并拍板,历史对照存档)

上表值的出处是 `setup.cpp` 中 SETUP_FOR_GTP 的硬编码配方(即 KataGo 对战机器人实际生效值),
**不是** `searchparams.h` 构造函数默认。当日发现的两口径差异(cpuct 1.1+0.6·ln/4096、
FPU 0.25 vs GTP 配方)已随切换废止;引用 KataGo 参数一律注明出处文件。

## 6. 差距收敛记录与待办

### 6.1 棋力线(✅ 2026-10-03 完成)+ 机制推进线(✅ 2026-10-06 完成)

GTP 配方全套 + 图搜索 2026-10-03 当日落地(§5.1);2026-10-06 差分定位驱动再落五项:

1. **回传加权一致 + 重算式节点统计**(c1bc7ea):修复后差分 1 访问 q 与 KataGo 逐位一致
   (P7 O3 -0.223 vs -0.222);vs KataGo@64v 历史首胜(1-5,基线 0-6);
2. **搜索随机对称**(aeff8f7):修正差分基线(kata-analyze 专属 wideRootNoise=0.04 须置 0)
   后,固定组 M1 相关 0.984、真实棋谱 top1 6/9(67%,历史最好);A/B 自对弈 5-1 优于无对称版;
3. **subtreeValueBias**(18c56d4):P8 相关 0.994(31/31 vs KataGo 31/26,历史最好);
   定位方法 = 给 KataGo build-onnx 打 KATA_SELTRACE 门控补丁实录逐手选点值;
4. **攒批改多线程投影**(bebca28)+ **vl 方向符号修复**(bc24b5b):旧管线两处偏差
   (minBatch 空闲凑满 + 在途不计在途量,盲区最深 2T−1)为批税主因;vl sign bug
   (白视角空间取错方向)在批 1 完全不可见、批 4 放血 —— 「单局面分布对拍正常但
   游戏级崩坏」矛盾的总解释。修复后 T4 vs T1 剂量反应 1-11 → 6-6(中性);
5. **sym 端到端契约**(86b7b3b,2026-10-07):对战 harness 序列化丢 `sym` 字段 →
   服务端按恒等特征评估、引擎反置换 → 实战棋力崩坏(H2H 0-6 全 73 手认输)而静态
   对拍全绿;修复后 SYMCHECK 恢复去相关噪声带,H2H sym-on vs sym-off = 4-2,
   vs KataGo 批 4 口径 3-3 追平批 1 基线。**教训入注:引擎契约字段必须端到端透传,
   任何中间层丢字段即静默乱序。**

### 6.2 强度口径(拍板)

- **测强度一律 MBATCH=1**(与 KataGo 单线程同语义;`match.mjs` 缺省即此,环境变量可覆盖);
  批 >1 是吞吐换访问质量的产品选项(浏览器延迟优先,session 校准批);
- 批 1 口径 vs KataGo@64v 累计 11-13/24(≈46%);批 4 口径 bug 修复后 3-3(样本待扩);
- 方法论教训:批次类改动的强度验收以**跨引擎对战**为准,自对弈镜像局放大双方共有弱点
  (v4.5@T4 曾 6 局自对弈 5-1 胜旧管线@批1,被 24 局跨引擎数据推翻)。

### 6.3 进度线(2026-10-08 盘点)

- **N3 自对弈强化循环**:配置就绪未启动(§3.2);
- **N4 收尾**:中端手机实测(shader-f16 覆盖面)+ 体积闸门核定;
- **N5 dispatch 融合**:立档未动代码([DISPATCH_FUSION_PLAN](DISPATCH_FUSION_PLAN.md);
  运行时 84 → 目标 ~65,四项融合 + KAE_NO_* A/B 开关 + 止损线:两项提升 <3% 即停);
- **校准残余**:吞吐高原批 8/16 间距 3-5%、采样噪声 ±2-3%,选档偶摆(容差设计内);
- **对战样本**:批 4 口径修复后样本仅 6 局,继续累积;批 1 口径 24 局为最大样本。

### 6.4 裁决记录(按时间)

- **2026-10-03**(五项存疑落定):采用 GTP 配方;乐观策略照抄 KataGo(通道 C∈{1,2,4},
  logit 空间插值,λ 树 1.0 / 根 0.2);温度外部传入 + 衰减;eval cache 内存 LRU 不落盘;
  参数不可外部配置。
- **2026-10-04**(批大小自适应):机制对齐 KataGo、不做运行时热调 —— 批上限 =
  createSession 现测校准(对应 `katago benchmark` 的加载时自动化),不做跨加载缓存
  (校准 <4s,设备热态漂移下现测比信任旧值稳);单线程攒批是同步块,批构成自适应
  主要靠校准上限,与 KataGo 多生产者队列的结构性差异记录在案。(2026-10-07 校准算法
  五项失真修复,见 §1。)
- **2026-10-06**(推理引擎):**推翻「不自研 WebGPU 算子」裁决** —— aethernn 当日立项
  当日落地 Node 侧全链并对拍全绿,默认切换([WEBGPU_ENGINE_RESEARCH](WEBGPU_ENGINE_RESEARCH.md));
  **推翻「不做量化」裁决** —— i8 量化版按 quant_explore 研究落地为默认权重
  ([INT8_QUANT_RESEARCH](INT8_QUANT_RESEARCH.md));f16 权重版曾实现并一度获准保留
  (「f32/i8/f16/onnx 均不允许删」),二次拍板撤销(无 shader-f16 直接报错,不做降级,
  清出仓库与历史)。
- **2026-10-07**(对称默认):交互崩坏根因定位为 harness 序列化丢字段(非引擎机制),
  恢复「对称默认开」对齐 KataGo nnRandomize=true。
- **2026-10-08**(收敛):onnx / fp32 权重通道与 ort-web 逃生舱整体出库,i8f16 单形态;
  测强度口径 MBATCH=1 固化;dispatch 融合立档。

## 7. 明确不做(技术侧;产品级不做清单在 README)

- 不在浏览器里训练或跑自对弈(训练全在本地 GPU 的 C++ 侧);
- 不为「看起来强」堆未验证的搜索技巧 —— 每一项对拍 / 自对弈数据说话;
- 不自写训练侧搜索 / 数据格式(selfplay / shuffle / train / gatekeeper / dumponnx 全用官方工具链);
- 不做运行时多形态 / 降级(引擎 i8f16 单形态,无 WebGPU / shader-f16 即报错)。

**已推翻的历史裁决(记录)**:「不自研 WebGPU 算子」(2026-10-06 推翻并落地);
「不做 INT8 / fp16 量化」(2026-10-06/07 推翻并落地;fp16 权重版旋即撤销)。
当时理由与决策过程分别见两份调研报告的「落地记录」节。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 特征编码不一致导致「训练强、浏览器弱」 | 逐位对拍是硬验收;N3 每代抽测浏览器实局 |
| 训练数据量 / 质量不足 | 学生已用 kata1 冷启动保底;N3 曲线不行先翻量,老师可换更强官方网 |
| WebGPU 移动端覆盖(Safari 旧版、无 shader-f16 设备) | 不支持即明确报错(裁决:不做慢速兜底);N4 实测圈定可用范围 |
| 量化代际回归(新学生出炉量化后棋力变化) | quant-test 三层闸门 + 对弈级抽测进发布流程;i8 为唯一形态,无「fp32 合格、int8 不合格」静默上线的通道 |
| 中间层丢契约字段(2026-10-07 sym 事故) | 新增字段随 evalBatch 契约端到端核对;对战 harness 与服务端同仓同步改 |
| 校准选档偶摆 | 容差设计内(吞吐差 5-7%),对局口径被预算钳制无感;不做 persistence(用户裁决) |
| 两仓库(KataGo / AetherGo)联调摩擦 | 本文即契约:接口(IO 名、特征表、参数口径)以本文 + README 为准,改动先改文档 |
| 参数基准混乱(构造默认 ≠ GTP 配方) | 引用 KataGo 参数一律注明出处文件(§5.2);A/B 后回写 |

## 9. 里程碑存档(新→旧)

- **2026-10-08**:**i8f16 单形态收敛** —— ort-web 逃生舱 / fp32 golden blob / 全 f32 内核
  模式 / f16 权重残留分支全部出库,`models/` 仅剩 `.i8.aewn`,parseAewn 只认 dtype=1;
  依据 = 三后端循环赛 1:1:1 + 数值闸门全绿(此前真机批 1-64 扫描 aewnn 已全档压过 ORT)。
  测试侧同口径收敛(纯 ort/f32 工具出库,quant-test golden 改 QONNX 指路)。
  同日:**浏览器 A/B 测速场**入库(server.mjs COOP/COEP + harness 五模式 +
  ts-probe 消融 + 1024v 三方基线文档);上游 MIT 许可证逐份核实入库
  (LICENSE_katago / LICENSE_katago-webgpu;saigo.online 第三方调查套件整体移除);
  dispatch 融合方案立档(静态 119 / 运行时 84 → 目标 ~65)。
- **2026-10-07**:**aewnn 高批次吞吐改造**(flashB v4 成为唯一注意力算子 + GEMM B 系
  批内循环 + swiglu workgroup 256 修复派发网格越限 + CAP 64):批 8/16/32 i8 ×2.1、
  f32 ×2.7-3.1,低批逐位不变;两次假设被消融实测推翻后定位(瓶颈在高批是 flash 46-58%
  而非 GEMM;共享内存 / subgroup 归并版均劣于纯 ILP)。**批次校准五项失真修复**
  (冷启动截断乱选 / fast-3 均值 / 少样本档丢弃 / 慢模式串扰 / 峰值锚刀口;
  真机生产路径冷加载 onnx=8/f32=16/i8=16)。**实战崩坏根因修复**(harness 序列化丢
  sym 字段)+ 对称默认恢复开启;vs KataGo 批 4 口径 3-3 追平。测试套件 Windows 原生兼容
  (动态 import file:// URL)。
- **2026-10-06**:**自研 WebGPU 引擎 aethernn 立项并当日完成 Node 侧全链**(packer →
  WGSL 内核 → 单 pass 宿主 → 8 对称 GPU 侧置换 + 特征零拷贝直传);对拍闸门全绿
  (cpuref vs ort golden:policy 1.4e-5;WGSL vs cpuref:2.7e-5),默认切 aethernn,
  ort-web 降逃生舱,运行时依赖归零。**同日晚 i8 量化版落位默认**(1.14MB,W8A16,
  「基本无损」引擎侧复现)。**搜索机制五连**(v4.2 重算式统计 + 回传加权修复 →
  v4.3 随机对称 + 差分基线修正 → v4.4 subtreeValueBias → v4.5 攒批多线程投影 →
  vl sign 修复):差分指标连创历史最好(P8 0.994 / 真实棋谱 top1 6/9),vs KataGo
  历史首胜(1-5);批 1 口径推进至 9-9/18(50%)。
  **同日前后(10-05/06 差分定位期)**:semOf 视角翻转修复(棋力主根,黑行棋叶效用
  半数反号进树)、终局 pass 守门、FPU 已访问子口径、特征 passing hacks 分支、
  totalW 边分摊 —— 全部由「双引擎同输入差分」定位,C++ 出处逐项注明。
- **2026-10-05**:模型瘦身(RoPE 表改图内现场计算,8.2MB→3.79MB,相对误差 ~5e-7,
  IO 契约不变零前端改动;该图内子图职能 2026-10-06 起 packer 吸收)。
- **2026-10-03**:产品边界拍板,README 完全重写;发现并修正参数基准问题
  (构造默认 ≠ GTP 配方)与乐观策略通道口径(导出图 C∈{1,2,4},修正旧「6 通道」错误);
  同日晚五项存疑全部裁决(§6.4);**19 路学生第 40 份出炉落位浏览器**;模型输出
  行棋方视角契约实证入档(白行棋 estimate 反号 bug 修复);OutputScoreValue 后处理
  公式核实;**同日深夜 GTP 配方全套落地(search.js v3)+ 图搜索落地(v4)+
  特征编码增量化(滚动盘面环,80 手编码 −63%)**——棋力线全部完成。
- **2026-10-02**:盘面升级 19×19(全链尺寸参数化);UCT 随机演棋引擎与 WASM 回退移除;
  搜索机制六项对齐 KataGo;死子标注 + 双停终局流程落地;19 路学生开训;差距全面盘点。
- **2026-10-01**:立项;N0~N2 全链打通;蒸馏路线定稿(旁路采集 + PyTorch 蒸馏,
  否决 npz 与零搜索直出);9 路学生 35 分钟蒸馏出炉,等 visits 16:10 胜官方同尺寸老网。

## 10. 参照物(本地路径)

- 引擎与训练主仓库:`/home/a/go/KataGo`(C++ selfplay/gatekeeper/dumponnx;python train.py);
- transformer 训练参照(备选 B):`/home/a/go/KataGo_Transformer`;
- 19 路学生训练现场:`/home/a/go/trainrun`(run_train.sh / b8c96h3tfrs_run;
  量化研究 quant_explore 同在 `<trainrun>/quant/`);
- 官方网络与 Elo 对照:katagotraining.org、`KataGo/docs/NetworkArchitectures.md`;
- 特征定义权威:`KataGo/cpp/neuralnet/nninputs.cpp` 的 `fillRowV7`;
- v17 transformer 结构权威:`KataGo/python/katago/train/model_pytorch.py` + `modelconfigs.py`;
- **搜索参数实战配方的权威:`KataGo/cpp/program/setup.cpp`(SETUP_FOR_GTP 硬编码默认)**,
  其次 `gtp_example.cfg`;`searchparams.h` 构造默认不是对战配方(§5.2);
- ONNX 图构建权威:`KataGo/cpp/neuralnet/onnxmodelbuilder.cpp`(IO 名与通道数);
  输出解释权威:`onnxbackend.cpp`;
- **策略通道数与乐观插值权威**:`desc.cpp` 的 PolicyHeadDesc(版本 → C∈{1,2,4})+
  `onnxbackend.cpp` / `eigenbackend.cpp` 的插值实现(logit 空间,softmax 前,λ 逐行);
- WGSL 内核移植出处:`katago-webgpu` 的 `webgpukernels.cpp`(MIT,入库许可见仓库根
  LICENSE_katago-webgpu;kernels.js 文件头注逐项出处);
- 差分工具链与本机操作手册:`tools/diff/`(手册 DIFF_WORKFLOW.md 为本地文件,不入库)。
