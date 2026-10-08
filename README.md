# AetherGo

19×19 浏览器人机对弈围棋:规则与数子是纯 JavaScript(零依赖、无 DOM、浏览器 / Worker / Node 通用),
对弈是**自研 WebGPU 推理引擎 aethernn**(int8 权重 + f16 激活存储 / f32 累加)+
KataGo GTP 实战配方对齐的 PUCT 搜索(纯 JS)。运行时零外部依赖(无 CDN、离线可用、
无构建无 CI),从 [WebOS](https://github.com/suulnnka/AetherWebOS)(纯前端网页操作系统)
的围棋应用中抽离而来。

## 大事记(细节见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §9 里程碑)

- **2026-10-02/03**:盘面 9×9 → 19 路;UCT 随机演棋引擎移除,NN 成为唯一引擎;
  学生模型 **b8c96h3tfrs 第 40 份(s68320512)**出炉落位(v17 transformer,929,641 参数,
  dumponnx 19 路 fp32);产品边界拍板;KataGo GTP 实战配方全套 + 图搜索当日落地。
- **2026-10-05**:模型瘦身 54% —— RoPE 查找表(4.44MB)改 ONNX 图内现场计算,8.2MB→3.79MB。
- **2026-10-06**:**自研 WebGPU 推理引擎 aethernn 上线,运行时依赖归零** —— 删除 5.5MB 的
  onnxruntime-web CDN 依赖(离线可用、无单点);423 节点 ONNX 图固化为 84 dispatch 的
  硬编码执行计划(单 pass / 单 submit / 单回读;qkv/ffn-gate 融合 GEMM、残差入 epilogue、
  flash attention、8 对称置换下沉 GPU),权重经 `training/pack_aewn.py` 打包为 `.aewn`。
  同日晚:默认权重切 **int8 量化版**(`.i8.aewn`,1.14MB,f32 的 30%;W8A16,
  引擎侧 30 例对拍与训练侧 8192 盘面研究同带,「基本无损」复现)。
  搜索侧同日五连:重算式节点统计、搜索随机对称、subtreeValueBias、攒批改 KataGo
  多线程投影(v4.5)、虚拟损失方向符号修复(批税主因)。
- **2026-10-07**:**aewnn 高批次吞吐改造**(flashB 成为唯一注意力算子 + GEMM B 系批内循环):
  批 8/16/32 端到端 i8 ×2.10/2.04/2.1、f32 ×2.83/2.74/3.06,低批逐位不变;
  批次校准算法五项失真修复(冷启动截断 / fast-3 均值 / 少样本污染 / 慢模式串扰 / 峰值锚);
  实战崩坏根因修复(对战 harness 序列化丢 `sym` 字段)后对称默认恢复开启;
  测试套件 Windows 原生兼容。
- **2026-10-08**:**引擎收敛为 i8f16 单形态,onnx / f32 支持全部移除** —— onnxruntime-web
  逃生舱(`?engine=ort`)、fp32 golden 权重通道(`?weights=f32`/.aewn blob)、全 f32 内核
  模式、f16 权重残留分支一律出库;`models/` 仅剩 `b8c96h3tfrs_19.i8.aewn`;
  `parseAewn` 只认 dtype=1(头部 / 排除清单张量仍按量化方案留 f32 权重)。
  真机 6 局循环赛实证 i8 与 onnx/f32 三后端棋力 1:1:1,数值闸门(quant-test 三层 +
  wgsl-test,i8 口径)全绿后执行;测试侧同口径收敛。上游许可逐份核实入库(KataGo /
  katago-webgpu 均为 MIT,见 License 节)。同日:浏览器 A/B 测速场与三方引擎对比测量入库;
  dispatch 压缩融合方案立档(未动代码)。

**在线体验:** 打开 <https://suulnnka.github.io/AetherWebOS/> 启动「围棋」应用 —— 那里面跑的就是本引擎
(默认高级档,窗口信息行实时显示演棋局数 / 胜率 / 耗时)。

**在线对弈页(GitHub Pages,免 CI):** **<https://suulnnka.github.io/AetherGo/>**
页面即仓库本身(`index.html` + `pages/` + `src/`,全相对路径,无构建无 CI),
仓库根起任意静态服务器即可本地预览(需支持 WebGPU + `shader-f16` 的浏览器):

```bash
python3 -m http.server 8000     # 打开 http://localhost:8000/
```

线上开启只需一次:仓库 **Settings → Pages → Source 选「Deploy from a branch」+ `/(root)`**。

对弈页功能:新对局 / 难度七档(NN 访问数分级)/ 停一手 / 数子(独立明细窗口,死子可手改)/
形势判断(NN ownership 方块热图 + 目差)/ 人机或双人 / 换边 / 悔棋 / 深浅色主题
(停一手双停即终局数子,劫点与禁着点有提示),底栏左侧行棋状态、右侧实时引擎搜索信息。

---

## 现状基线(2026-10-08,同机 RTX 5060)

- **棋力**:vs KataGo@64visits(GPU 双侧)批 1 口径累计 11-13/24(≈46%,统计平手,
  测强度一律 MBATCH=1);批 4 口径在 `sym` 序列化 bug 修复后首轮 3-3,首次追平批 1 基线。
- **吞吐(真实浏览器)**:批 1-64 全档压过 ORT-Web(批 64:i8 1088 vs 718 行/s,
  ORT 批 32 有悬崖);1024 visits 端到端 456-555 visits/s,同网络 KataGo CUDA
  (8-64 线程)2528-4905 visits/s —— 「9.8× 差距」实为环境差 1.5× × 引擎差 5.9×
  ([基线文档](test/browser-ab/katago-cuda-bench-1024v.md))。
- **数值闸门**:wgsl-test(WGSL vs cpuref-Q,i8f16 口径)全绿;quant-test 三层
  (激活范围 / cpuref-Q vs ort golden / WGSL-Q vs cpuref-Q)全绿;模糊测试 +
  特征对拍 + KataGo 对齐项 + Worker 冒烟全绿。
- **下一性能课题**:dispatch 融合已立档(运行时 84 → 目标 ~65,四项融合 + A/B 开关
  + 止损线):[docs/DISPATCH_FUSION_PLAN.md](docs/DISPATCH_FUSION_PLAN.md)。
  真实浏览器测速场 `test/browser-ab/`(?bench 批4 微基准 / ?sweep 批1-64 吞吐扫描 /
  ?calib 校准探针 / ?vbench 1024v 基准;曾有的三后端 6 局循环赛已随 2026-10-08
  逃生舱收敛移除,其 1:1:1 结论留档)。

## 产品边界(2026-10-03 拍板;引擎侧条目随实现演进更新,更新处注明日期)

**唯一场景:浏览器里的人机对弈**(也可双人)。所有工程决策以此为唯一准绳,逐条裁决如下:

| 维度 | 裁决 |
|---|---|
| 盘面 | 仅 19×19(代码内部尺寸参数化是实现细节,不承诺其他盘径) |
| 规则 | 仅中国规则:数子法、position superko、禁自杀、贴 7.5;**无让子** |
| 执行 | 单线程;**无背后思考**(只在己方回合搜索;跨手树复用不属于背后思考) |
| 引擎 | 仅一种(NN 搜索);推理仅自研 aethernn,仅 WebGPU + `shader-f16`,不可用时报错,无任何回退 |
| 模型 | 仅内置一个权重包(`b8c96h3tfrs_19.i8.aewn`,i8 权重 + f16 计算,2026-10-08 起唯一形态);搜索参数仅内置一套(无配置文件系统) |
| 难度 | 难度 = 固定 NN 访问数,设备无关、可复现;无按钟时间管理 |
| 范围外 | 开局库 / 残局库、GTP / SGF / 分析引擎等一切协议、第三方引擎对接、human-SL、行为 / 特征项、落盘 eval cache |

完整「不做」清单与理由见[下文](#不做拍板);技术口径见 [NEURAL_PLAN](docs/NEURAL_PLAN.md)。

## 引擎与分层

```
src/protocol.js   契约常量(盘面尺寸 / 走法编码 / 记谱)—— UI 与引擎的唯一共享层
src/rules.js      规则核心:走子/撤销、合法性、禁全同(Zobrist)、洪泛工作区
src/scoring.js    数子与死子:中国规则数子、Benson + 提子搜索、ownership 辅助标注
src/engine.js     对外门面:再导出 rules + scoring(既有 import 路径不变)
src/nn/           对弈搜索:features(特征)→ session(引擎门面)→ search(PUCT),
                  辅助模块 flood / ladder(征子)/ eval-cache(评估缓存)/ move-select(选点)
                  / calibrate(批校准)/ symmetry(8 对称置换表)
src/nn/webgpu/    自研推理引擎 aethernn:plan(执行计划 + blob 解析互验)+ kernels
                  (WGSL 内核:GEMM 平铺族 / flashB / RMSNorm / RoPE / 池化与头部小核)
                  + session(单 pass 宿主,84 dispatch)+ cpuref(CPU 参考解释器,测试专用);
                  权重经 training/pack_aewn.py 打包为 .aewn
src/nn-worker.js  Worker 门面:UI 唯一入口,一切引擎事实经消息获取
pages/            对弈页 UI:app.js(控制器)+ dom.js(通用 DOM)+ board.js(棋盘绘制)
```

UI 与引擎严格分离:UI 不 import 任何引擎代码,只 import `src/protocol.js`
(契约常量);规则/数子/搜索全在 Worker 线程里,主线程零引擎代码。
规则核心的撤销栈 / 历史键栈全是模块级 typed array,make/unmake 过程**零分配**;
`rules.js` 与 `scoring.js` 共享同一套洪泛工作区(经访问器串行使用)。

### 棋盘与编码

- 361 个交叉点(19 行 × 19 列),`idx = 行×19 + 列`;行 0 是上边,列 0 是左边,黑先白后。
- 棋子:1 = 黑,2 = 白,0 = 空;行棋方 BLACK = 0 / WHITE = 1,棋子 = 行棋方 + 1。
- 走法:交叉点 0..360,`PASS = 361` 表示停一手。UI 与 Worker 之间只传这一种编码,不搞两套。
- 坐标记谱:列 A~T(跳过 I)+ 行 1~19(下边为 1),如天元 = `K10`。

### 规则

气尽提子(整块)、禁自杀(能提子则不算)、**position superko(禁全同)**:
对局历史维护 64 位 Zobrist 局面键栈(只含盘上棋子,与 KataGo `KO_POSITIONAL` 同口径),
落子后键在历史中出现过即非法 —— 单劫是它的特例,三劫 / 双劫循环天然被破。
`koPoint()` 仍报单劫点,但只作 UI「打劫提示」,不再是合法性依据。
停一手永远合法,连续两手停即终局。
着法生成即「逐点 isLegal」:临时落子 → 完整枚举提子 → 查气 → 算落子后键查历史,只读棋盘、无副作用;
没被占过且不提子的落点局面必新,免扫历史(KataGo 同款剪枝)。

### 计分与死子

中国规则数子法:己方子数 + 只被己方贴边的空点区域,黑贴 7.5 目。
双停时盘上常残留死子 —— `deadStones()` 规则侧判定(Benson 绝对活棋 + 「对方先手能否提掉」
的小预算 AND/OR 搜索,预算耗尽按活处理,宁漏勿错),`finalScore()` 移除死子后数子。

**死子标注(NN ownership 辅助)**:NN 引擎加载时,`deadStonesWithOwnership()` 在规则侧
结果之上,把 ownership 与链色强烈相悖的链(链平均归属阈值 0.35,KataGo 口径)
补标为死 —— 补的是规则侧小预算搜不出来的中型死棋。保底三条:Benson 绝对活棋永不死、
规则侧已判死的不翻案、ownership 模糊(|均值| < 阈值)不动作。数子请求带 `deadOverride`
手改列表时以用户为准,不走 NN。

数子不限于双停后:工具栏「数子」随时可点,进入数子模式会打开独立明细窗口
(`scoreBreakdown`:黑 / 白各自的子数 + 空数 + 贴目、死子分布、目差),
棋盘上点击棋子整块切换死 / 活、数子实时重算,「继续对局」随时退回对局;
双停终局后自动进入数子并弹同一窗口(此时兼作结算窗),工具栏「结果」随时重看。

### 形势判断

对弈页「形势」按钮走 Worker 的 `estimate` 消息:单次 NN 推理,黑方目差优先取
`OutputScoreValue` 的 **lead 头**(模型自己的分数预测,已含贴目),缺了回落
归属求和口径(`OutputOwnership` 求和扣贴目);黑方胜率取 `OutputValue`。
归属层采用方块热图:每个交叉点画一个小方块,颜色表归属方(黑 / 白),
深浅与大小都编码归属强度 —— 方块越大、越实,归属越确定;越小越淡,争夺越模糊。
走子后旧形势图自动作废。

> **视角口径(2026-10-03 修正)**:模型原始输出全部是**行棋方**视角(+ = 行棋方优,
> 与 KataGo 训练目标一致,C++ 侧拿到后才翻成白方视角)。Worker 统一把
> ownership / 目差换算成黑方视角再出消息 —— 旧实现没翻,白方行棋时目差、
> 归属热图颜色、数子死子辅助会整体反号,已修。

## NN 搜索(src/nn/)

`src/nn-worker.js`(Worker)→ `src/nn/session.js`(引擎门面,**自研 aethernn** 唯一引擎,
仅 WebGPU;ort 逃生舱已于 2026-10-08 移除)→ `src/nn/search.js`(异步 PUCT 批量搜索,
v4.5)+ `src/nn/features.js`(fillRowV7 特征,与 KataGo C++ 逐位对拍通过,征子通道另与
原生实现对拍 8393 链零分歧)。特征编码的征子历史通道走**滚动盘面环**
(KataGo `recentBoards` 同款,make 增量维护,编码成本不随手数增长);
环实现由「环 vs 重演」逐位对拍模糊测试锚定(features-test)。

已对齐 KataGo(基准 = GTP 实战配方 `setup.cpp SETUP_FOR_GTP`,机制逐项移植、注明 C++ 出处,
完整机制表见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §5.1):

- **GTP 配方常数全套**:效用函数(winLoss + 0.1·静态目差 + 0.3·动态目差)、
  cpuct 1.0 + 0.45·ln((W+500)/500)·√(W+0.01)·父效用方差因子、FPU 按已访问
  policy 质量混合父 NN 效用(root 0.1 / 树内 0.2)、不确定度加权回传、
  终选 noisePruning → valueWeightExponent 0.25 → LCB 5σ(ESS 修正);
- **乐观策略插值**:session 层,logits 空间 softmax 之前,根 0.2 / 树内 1.0;
- **重算式节点统计**(recomputeNodeStats 移植):每回传自叶向根从「子边统计 + 自身
  NN 评估」重算,虚拟损失移出统计改选点期效用混合(2026-10-06);
- **搜索随机对称**(nnRandomize 同款):每评估随机取 8 对称之一,policy/ownership
  逆置换回恒等系,评估缓存键取变换后特征;默认开(对齐 KataGo),
  `opt.symmetry === false` 可关(2026-10-06 移植;当日曾在实战中观察到交互崩坏临时
  默认关,2026-10-07 定位根因为对战 harness 序列化丢 `sym` 字段后恢复默认开);
- **subtreeValueBias**(GTP 默认 0.45):同局部形签名的节点共享在线表项,校正网络
  评估的系统性偏差;GC 删除回退 80% 贡献(2026-10-06);
- **攒批 = KataGo 多线程投影**(v4.5):批上限 = 在途评估总量 T,任何下降的统计盲区
  ≤ T−1,死端 / 预算边界机会主义发射;虚拟损失方向符号已修复(2026-10-06,批税主因);
  批上限由加载时校准现测(吞吐 ≥ 最优 90% 的最小批,再按预算压 stale);
- **图搜索**(`useGraphSearch`,repBound 11):节点表跨手持久,转置局面共享子图,
  graphhash 语义键控 + 惰性子节点 + 边访问缩放 + mark-and-sweep GC;
  与 eval-cache(2048 LRU)分层互补;
- **特征与终局**:fillRowV7 全通道含 passing hacks 分支(推理期口径);终局 pass 守门
  (KataGo 终选层 ownership 判据,常开产品护栏);无用着剪枝 + 低先验 1e-4 不建子 +
  fillDameBeforePass;根对称剪枝(保守式);
- **认输**:白方视角值连续 3 手越 −0.90 且手数 ≥73(play.cpp 口径;悔棋 / 新局作废历史);
  **温度**:由调用方传入(`temperature` / `temperatureHalflife` 按手数半衰),
  引擎不自带温度策略。

未对齐(记录在案):retrospective 权重回溯削减(LCB 奖励承担同等稳定职能);
转置边访问追平(maybeCatchUpEdgeVisits)两版实现实测有害,默认关闭保留代码。
机制细节与 C++ 出处见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §5。

### 难度七档(NN 访问数)

| 档位 | 入门 | 初级 | 中级 | 高级 | 大师 | 宗师 | 棋圣 |
|---|---|---|---|---|---|---|---|
| 访问预算 | 1 | 16 | 64 | 256 | 1024 | 2048 | 4096 |

入门 1 访 = **模型直出**:仅根单次评估,选点 = 原始策略 argmax,胜率 = 网络直出值
(不参与认输判定);其余档走完整搜索。默认高级档。

## 待办(2026-10-08 盘点)

- **N3 自对弈强化循环**:`training/run_n3_loop.sh` + gatekeeper 配置就绪,未启动;
- **N4 收尾**:中端手机实测(`shader-f16` 覆盖面)+ 体积闸门核定;
- **N5 dispatch 融合**:方案已立档(未动代码):[docs/DISPATCH_FUSION_PLAN.md](docs/DISPATCH_FUSION_PLAN.md);
- **对局口径强度**:vs KataGo 批 4 口径样本仍少(修复后首轮 3-3),继续扩大;
  批 1 口径 11-13/24 为当前口径(测强度一律 MBATCH=1,与 KataGo 单线程同语义);
- **校准残余**:aewnn 吞吐高原批 8/16 间距 3-5%、采样噪声 ±2-3%,选档会在相邻档间
  偶摆(容差设计内,吞吐差仅 5-7%;对局口径批 4 被搜索预算钳制,无感);
  彻底消除需 persistence(用户明确不引入 localStorage)或大幅加采样;
- **量化正式 L3 闸门**:300 局等 visits 对弈级验收未跑(现以真机三后端循环赛 1:1:1
  实证 + 输出级对拍为准;i8 已是唯一形态)。

## 不做(拍板)

以下均为**已拍板不做**,不是「还没做」;再评估需先推翻对应裁决。

- **多线程 / 背后思考(pondering)**:单线程到底;只在己方回合搜索(NN batch 化不算多线程);
- **按钟时间管理**:无 lagBuffer / byoyomi / 中残局时间分配 / obviousMoves 等 —— 难度即固定访问数;
- **开局库 / 残局库**:不内置任何着法表,开局与终局一律进搜索;
- **GTP / 引擎侧 SGF / 分析引擎**:不接任何围棋协议与 GUI;对弈页的「形势」是单次前向,
  不做 kata-analyze 式搜索精化;SGF 属于 WebOS 应用侧;
- **第三方引擎对接 / 多引擎**:仅一种引擎(自研 aethernn);
- **非 WebGPU 后端 / 任何降级**:无 WASM / CPU 回退;无 fp32 / f16 权重回退
  (无 `shader-f16` 即报错;fp16 权重版曾实现并一度获准保留,二次拍板撤销并清出历史);
- **多模型 / 模型热切换 / ONNX 运行时解析 / .bin.gz 解析**:仅内置一个 i8f16 权重包;
  ONNX / .bin.gz 只是训练侧产物与 packer 输入,浏览器不加载、不解析;
- **非中国规则 / 让子 / 其他盘径**:数目法、encore、button、simple/situational ko、
  多步自杀、2~25 路等一律不支持(特征编码器相应通道恒 0 占位,保证与训练数据逐位一致);
- **行为 / 特性项**:PDA(playoutDoublingAdvantage)、对手建模(visitCapContempt)、
  antiMirror、avoidRepeatedPatternUtility、wideRootNoise、avoidMYTDaggerHack
  —— 陪人下棋不需要(passing hacks 的**特征编码分支**属推理期对齐口径,已实现;
  作为行为选项不做);
- **human-SL**(按段位下人味棋):依赖模型侧支持,范围外;
- **落盘 eval cache**(KataGo `evalcache` 文件):内存 LRU 已覆盖转置复用,不做持久化
  (浏览器产品也不引入 localStorage);
- **WDL 三元效用**:中国规则 7.5 贴目无和棋、position superko 无无结果,winLoss 标量即精确;
- **根 Dirichlet 噪声**:训练侧机制,浏览器对人不噪声;训练全在 C++ 侧;
- **训练侧任何东西**:selfplay / shuffle / train / gatekeeper / dumponnx 全用 KataGo 官方工具链,
  不自己写训练侧搜索(架构见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §0)。

**已推翻的裁决(记录在案,防止反复)**:不自研 WebGPU 算子(2026-10-06 推翻,aethernn
落地,[调研](docs/WEBGPU_ENGINE_RESEARCH.md));不做量化(2026-10-06/07 推翻,i8 先成
默认、后成唯一形态,[调研](docs/INT8_QUANT_RESEARCH.md));subtreeValueBias 曾列「不做」
(2026-10-06 移植,移出清单);ort-web 逃生舱与 fp32 golden 通道(2026-10-08 移除)。

## 裁决记录(2026-10-03,原「存疑与待确认」五项全部落定)

1. **搜索参数基准** → 采用 KataGo GTP 实战配方(出处 `setup.cpp SETUP_FOR_GTP`,
   非 `searchparams.h` 构造默认),当日切换落地;
2. **乐观策略通道** → 照抄 KataGo 逻辑:通道按模型版本 C ∈ {1,2,4}(v17 为 4),
   logits 空间、softmax 前逐点插值,λ 树内 1.0 / 根 0.2,老网无乐观面用通道 0;
3. **温度** → 由调用方传入,引擎不自带任何温度策略;需要随机时必须带衰减
   (现实现即此设计,开局温度给不给是对弈页调用方的事);
4. **eval cache** → 类置换表(存 NN 评估输出),内存 2048 LRU 保留、不做落盘持久化;
5. **参数格式** → 搜索参数内置唯一一套,外部不可配置;难度七档是产品分级,不属于参数配置。

## 用法

```js
import { BLACK, WHITE, PASS, newBoard, genLegal, isLegal, make, unmake,
         capturedOf, scoreGame, moveToText, deadStones, finalScore } from './src/engine.js';

const bd = newBoard();                        // Int8Array(361)
const moves = genLegal(bd, BLACK);            // 黑方合法落点(停一手不进列表)
console.log(isLegal(bd, BLACK, 180));         // (9,9) 天元是否合法
const tok = make(bd, 180, BLACK);             // 走子:返回撤销令牌(含提子数)
console.log(capturedOf(tok));                 // 这一手提了几颗
unmake(bd, 180, tok);                         // 撤销

console.log(moveToText(bd, 180));             // 'K10'(记谱不依赖盘面)

const s = scoreGame(bd);                      // { black, white, margin } 原始数子
const f = finalScore(bd);                     // 双停终局数子:{ ..., dead: 死子点数组 }
const d = deadStones(bd);                     // 仅死子判定(UI 标示 / 手改后传 deadOverride)
```

对弈 / 数子 / 形势判断经 Worker(`src/nn-worker.js`):

| 消息 | 回包 |
|---|---|
| `{ type:'levels' }` | 难度表(NN 访问数分级) |
| `{ type:'load', id, modelUrl }` | `{ type:'loaded', ep }` / `{ error }`(懒加载 aethernn + i8 权重,仅 WebGPU + shader-f16,不可用报错) |
| `{ type:'state', id, moves }` | 棋盘 / 提子 / 劫点 / 合法着法 / 双停终局与数子 |
| `{ type:'think', id, moves, level, temperature?, temperatureHalflife? }` | 逐步 progress + `{ move, visits, winRate, scoreLead?, ms }` / `{ resign }` |
| `{ type:'score', id, moves, deadOverride? }` | 移除死子后的数子结果(含 `detail` 明细) |
| `{ type:'estimate', id, moves }` | 黑方胜率 / 目差(双口径:`netScoreLead` 网端 lead 优先,`scoreLead` 归属求和回落)/ ownership 图(黑方视角) |

`moves` 是从初始局面起的走法序列(0..360 或 PASS=361),Worker 自己重演棋盘
(结构化克隆最省,且不会有两份规则实现)。

## 测试

```bash
npm test                            # 规则 + 数子 + 记谱 + 模糊测试 + NN 特征 + KataGo 对齐项
                                    # + Worker 冒烟(Windows 原生可跑,2026-10-07 起)
node test/engine-test.mjs fuzz      # 只跑指定小节(--list 看全部)

npm run test:aewnn                  # aethernn 对拍(i8f16 唯一口径):WGSL vs CPU 参考
                                    # (cpuref-Q;Dawn 绑定 npm i --no-save webgpu)
npm run test:quant                  # 量化三层闸门:激活范围 / cpuref-Q vs ort golden
                                    # (golden 模型经 QONNX 环境变量指路,不在场自动跳过)
                                    # / WGSL-Q vs cpuref-Q
npm run bench:aewnn                 # aethernn 吞吐基准(Dawn;WSL2 上为软件渲染下限)

npm run test:featdiff               # 特征对拍(需先用 katago 产数据,见 training/)
node test/nn-temp-test.mjs          # 温度选点分布(LCB / 抽样)

cd test/browser-ab && node server.mjs   # 真实浏览器测速场(COOP/COEP 隔离):harness.html
                                        # ?bench 批4 微基准+64v 搜索 / ?sweep 批1-64 吞吐扫描
                                        # / ?calib 校准算法探针 / ?vbench 1024v 基准
                                        # (三后端循环赛已随 2026-10-08 收敛移除)
```

围棋没有 perft,规则正确性的金标准是**模糊测试**:随机对局 40 局 × 最多 420 手,
每一手验证盘面子数守恒(走前 + 1 − 提子 = 走后)与**禁全同不变量**
(非停一手产生的局面键整局互不重复),然后逐手撤销到空盘、再逐手重演 ——
每一步「走前的合法着法列表」必须逐步复原(劫点、提子、历史键全覆盖)。

NN 路线的金标准是**特征对拍**(`test/featdiff.mjs`):katago selfplay 训练行
(npz 的 `binaryInputNCHWPacked` / `globalInputNC`)与 JS 编码器逐位一致;
征子通道另有 `test/ladderdiff.mjs` 直接对拍 KataGo 原生 C++ 实现。
推理引擎的金标准是**分层对拍链**(cpuref-Q vs ort golden → WGSL vs cpuref-Q),
容差按量级定不追位 —— f16 激活存储使浮点累加序差被逐层放大,窄容差对拍
在原理上不可通过(「舍入边界混沌」,2026-10-07 调查链结论,已并入 quant-test 闸门口径)。

架构、训练管线与差距盘点的技术细节:**[docs/NEURAL_PLAN.md](docs/NEURAL_PLAN.md)**。

## License

本仓库自有代码:MIT(全文见 [LICENSE](LICENSE))。

第三方许可(全文均附于仓库根目录):

- [`LICENSE_katago`](LICENSE_katago) — [KataGo](https://github.com/lightvector/KataGo)(MIT)。
  搜索机制 / GTP 配方 / 特征编码 / 训练管线对齐与对拍的上游;内置模型
  `models/b8c96h3tfrs_19.i8.aewn` 经官方 kata1 公开训练数据冷启动自训(kata1 网络为 MIT 系
  [KataGo Neural Network License](https://katagotraining.org/network_license/),数据可自由下载)。
- [`LICENSE_katago-webgpu`](LICENSE_katago-webgpu) — [katago-webgpu](https://github.com/saigo-online/katago-webgpu)(MIT,继承 KataGo)。
  `src/nn/webgpu/kernels.js` 部分内核结构(tiledGemm 家族 / flashAttention / rmsNorm)移植自其
  `webgpukernels.cpp`,文件头注有逐项出处。
