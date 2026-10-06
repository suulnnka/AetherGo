# AetherGo

19×19 浏览器人机对弈:规则与数子是纯 JavaScript(零依赖、无 DOM、浏览器 / Worker / Node 通用),
对弈搜索是浏览器端 ONNX(WebGPU)推理 + PUCT。
从 [WebOS](https://github.com/suulnnka/AetherWebOS)(纯前端网页操作系统)的围棋应用中抽离而来,全部自研。
2026-10-02:盘面从 9×9 升级为 19 路;随机演棋 UCT 引擎移除,NN(WebGPU)成为唯一对弈引擎。
2026-10-03:**学生模型 b8c96h3tfrs 第 40 份(s68320512)出炉落位 `models/b8c96h3tfrs_19.onnx`**
(v17 transformer,929,641 参数,dumponnx 19 路 fp32 8.2MB);旧占位模型(官方 g170 b6c96、
9 路蒸馏学生)全部退役 —— **仅支持这一个模型**。
2026-10-05:模型瘦身 54%——8 个注意力块 q/k 的 RoPE 查找表(48 份同表复制的 cos/sin,
共 4.44MB)改为 ONNX 图内现场计算(图首 Sin/Cos 子图生成共享表,应用结构不动),
`models/b8c96h3tfrs_19.onnx` 8.2MB→3.79MB;输出等价(相对误差 ~5e-7)、IO 契约不变
零前端改动(转换脚本 `training/make_rope_ongraph.py`,删表前逐表校验)。
2026-10-06:**自研 WebGPU 推理引擎 aethernn 上线,运行时依赖归零** —— 删除 5.5MB 的
onnxruntime-web CDN 依赖(离线可用、无单点);423 节点 ONNX 图固化为 84 个 dispatch 的
硬编码执行计划(顺序逐块镜像 PyTorch forward;qkv/ffn-gate 融合 GEMM、残差入 epilogue、
flash attention、BiasMask 折叠、RoPE 打包期预算),权重经 `training/pack_aewn.py` 打包为
`models/b8c96h3tfrs_19.aewn`(3.82MB);8 对称随机置换下沉 GPU(stem 卷积 gather 表),
搜索侧特征零拷贝直传。对拍全绿(vs ort CPU:policy 1.4e-5;WGSL vs CPU 参考:2.7e-5),
nn-e2e 完整终局。真机性能 A/B 后再删 ort 逃生舱(`?engine=ort`)。

2026-10-06(晚):**默认权重切 int8 量化版**(`models/b8c96h3tfrs_19.i8.aewn`,1.14MB,
f32 的 30%)—— 完全按量化研究(quant_explore,第 40 批权重)执行:全部 58 层 trunk
权重逐输出通道对称 int8(研究的逐层裁剪搜索证明 minmax 即最优,无敏感离群值),
计算 f16 激活存储/f32 累加(W8A16,激活不做 a8);头部/norm/RoPE 按 PyTorch fp32
头部口径留 f32。引擎侧 30 例对拍与研究 8192 盘面 int8w 行同带(KL 9.7e-4 /
winMAE 6.2e-3 / top1 含近平局 100%,0 实质翻转),「W8A16 基本无损」在引擎侧复现。
回退产物为 **f16 权重版**(`models/b8c96h3tfrs_19.f16.aewn`,2.01MB,fp16 纯变体
Top1 99.62% ≈ 无损)。三份权重全部入库:f32 golden(`.aewn`,3.82MB,对拍基准)、
i8 量化版(默认)、f16 回退版(2026-10-07 拍板不允许删)。
`?weights=f32` 切 golden 通道;无 shader-f16 自动回落 f16 版。遗留:L3 对弈级验收
(300 局等 visits)。另修复:packer 曾把 stem.global_w([OC][19])按列轴量化,
GPU 按通道读 → 引擎侧 0.3~4 logit 偏差,已改首维轴并重打包。

**在线体验:** 打开 <https://suulnnka.github.io/AetherWebOS/> 启动「围棋」应用 —— 那里面跑的就是本引擎
(默认高级档,窗口信息行实时显示演棋局数 / 胜率 / 耗时)。

**在线对弈页(GitHub Pages,免 CI):** **<https://suulnnka.github.io/AetherGo/>**
页面即仓库本身(`index.html` + `pages/` + `src/`,全相对路径,无构建无 CI),
仓库根起任意静态服务器即可本地预览(需支持 WebGPU 的浏览器):

```bash
python3 -m http.server 8000     # 打开 http://localhost:8000/
```

线上开启只需一次:仓库 **Settings → Pages → Source 选「Deploy from a branch」+ `/(root)`**。

对弈页功能:新对局 / 难度四档(NN 访问数分级)/ 停一手 / 数子(独立明细窗口,死子可手改)/
形势判断(NN ownership 方块热图 + 目差)/ 人机或双人 / 换边 / 悔棋
(停一手双停即终局数子,劫点与禁着点有提示),底栏左侧行棋状态、右侧实时引擎搜索信息。

---

## 产品边界(2026-10-03 拍板)

**唯一场景:浏览器里的人机对弈**(也可双人)。所有工程决策以此为唯一准绳,逐条裁决如下:

| 维度 | 裁决 |
|---|---|
| 盘面 | 仅 19×19(代码内部尺寸参数化是实现细节,不承诺其他盘径) |
| 规则 | 仅中国规则:数子法、position superko、禁自杀、贴 7.5;**无让子** |
| 执行 | 单线程;**无背后思考**(只在己方回合搜索;跨手树复用不属于背后思考) |
| 引擎 | 仅一种(NN 搜索);推理后端仅 WebGPU,不可用时报错,无任何回退 |
| 模型 | 仅内置一个 ONNX 文件;搜索参数仅内置一套(无配置文件系统) |
| 难度 | 难度 = 固定 NN 访问数,设备无关、可复现;无按钟时间管理 |
| 范围外 | 开局库 / 残局库、GTP / SGF / 分析引擎等一切协议、第三方引擎对接、human-SL、行为 / 特性项、落盘 eval cache |

完整「不做」清单与理由见[下文](#不做拍板);残余技术缺口见[待办](#待办2026-10-03-按边界整理后)。

## 引擎与分层

```
src/protocol.js   契约常量(盘面尺寸 / 走法编码 / 记谱)—— UI 与引擎的唯一共享层
src/rules.js      规则核心:走子/撤销、合法性、禁全同(Zobrist)、洪泛工作区
src/scoring.js    数子与死子:中国规则数子、Benson + 提子搜索、ownership 辅助标注
src/engine.js     对外门面:再导出 rules + scoring(既有 import 路径不变)
src/nn/           对弈搜索:features(特征)→ session(引擎门面)→ search(PUCT),
                  辅助模块 flood / ladder(征子)/ eval-cache(评估缓存)/ move-select(选点)
src/nn/webgpu/    自研推理引擎 aethernn(plan 执行计划 + 15 个 WGSL 内核 + 单 pass 宿主
                  + cpuref 参考解释器);权重经 training/pack_aewn.py 打包为 .aewn
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

`src/nn-worker.js`(Worker)→ `src/nn/session.js`(引擎门面,默认**自研 aethernn**,
仅 WebGPU;`?engine=ort` 切回 onnxruntime-web 逃生舱,两者 evalBatch 契约同构)
→ `src/nn/search.js`(异步 PUCT 批量搜索)+ `src/nn/features.js`(fillRowV7 特征,
与 KataGo C++ 逐位对拍通过,征子通道另与原生实现对拍 8393 链零分歧)。
特征编码的征子历史通道走**滚动盘面环**(KataGo `recentBoards` 同款,make 增量维护,
编码成本不随手数增长);环实现由「环 vs 重演」逐位对拍模糊测试锚定(features-test)。

已对齐 KataGo **GTP 实战配方**（`setup.cpp` SETUP_FOR_GTP，2026-10-03 拍板并当日落地）：

- 效用函数：winLoss + 0.1·静态目差 + 0.3·动态目差（atan 平滑期望，动态中心随局面更新）——
  目差进选点，落后抢目差、领先收敛；
- PUCT：cpuct 1.0 + 0.45·ln((W+500)/500)·√(W+0.01)·父效用方差因子；分母按回传权重；
  FPU 按已访问 policy 质量混合父 NN 效用（pow 2），root 0.1 / 树内 0.2；
- 不确定度加权：每次评估按短期胜负 / 目差误差定回传权重（老网无 score 头恒 1）；
- 最终选点：noisePruning → valueWeightExponent 0.25（t₃ CDF）→ LCB 5σ（效用方差
  ESS 修正 + radiusFactor² 奖励）→ 温度 0 取最优 / >0 按 w^(1/T) 抽样（支持按手数半衰）；
- 根评估每次思考重算（乐观插值 λ=0.2，供根先验刷新 + 动态目差中心）；叶子评估 λ=1；
  乐观通道插值在 logits 空间、softmax 之前（session 层，老网自动退化）；
- 根对称剪枝（保守式：劫点 / 禁点存在即放弃；对称等价着法只留先验最高者）；
- 既有保留：无用着剪枝 + 低先验 1e-4 不建子、fillDameBeforePass、批量推理
  （8 叶 + 虚拟损失）、跨手树复用、评估缓存（2048 LRU）、温度由调用方传入；
- 认输：白方视角值连续 3 手越 −0.90 且手数 ≥73（C++ play.cpp 口径；悔棋 / 新局作废历史）。

未对齐（记录在案）：图搜索转置共享（待办唯一棋力项）；retrospective 权重回溯削减
（LCB 奖励机制承担同等稳定职能）。机制细节与 C++ 出处见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §5。

### 难度四档(NN 访问数)

| 档位 | 初级 | 中级 | 高级 | 大师 |
|---|---|---|---|---|
| 访问预算 | 80 | 200 | 400 | 800 |

## 待办(2026-10-03 按边界整理后)

按「产品边界」裁剪后,与 KataGo 对战配方的**真实**残余差距如下。
判定基准:本地 KataGo checkout 的 `cpp/program/setup.cpp` 中 `SETUP_FOR_GTP` 的
硬编码默认(即 KataGo 对战机器人实际生效的配方),**不是** `searchparams.h` 构造函数默认。
**2026-10-03 拍板:采用该配方**;切换落地前引擎维持现值不动。

### 棋力线(✅ 全部完成,2026-10-03)

GTP 配方全套 + 图搜索均已落地(机制、参数与 C++ 出处见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §5):

1. GTP 推理配方常数 + 乐观策略插值(session 层,logits 空间,根 0.2 / 树内 1.0);
2. NN 输出进树:目差效用 / 真实 LCB(ESS 修正)/ cpuct 方差自适应 /
   uncertainty 加权 / 认输精化(−0.90 连续 3 手,play.cpp 口径);
3. **图搜索**(`useGraphSearch`,repBound 11):节点表跨手持久,转置局面共享子图;
   键控按 KataGo graphhash 语义 —— 最后一手周边空域 > 11 → 状态键合并,
   局部战斗 → 链式路径唯一;惰性子节点 + 边访问缩放(getChildWeight 口径)+
   在途跳过(循环守卫)+ 每手 mark-and-sweep GC。与 eval-cache 分层互补:
   图搜索管搜索内节点共享,eval-cache 管跨手评估复用;
4. FPU 按已访问 policy 混合(pow 2);5. 回传权重精修(valueWeightExponent + noisePruning);
6. 根对称剪枝(保守式);7. 特征编码增量化(滚动盘面环,编码不随手数增长)。

同根连续思考的子图复用实测:第二次思考推理调用大幅下降(对齐测试 8a);
空盘等转置密集局面单叶成本较树搜索降约四成(同进程 A/B)。

### 进度线

- ~~19 路学生模型出炉后经 dumponnx 导出替换浏览器占位模型~~ **完成(2026-10-03)**:
  第 40 份(s68320512)落位 `models/b8c96h3tfrs_19.onnx`,旧占位退役;
- N3 自对弈强化循环启动(`training/run_n3_loop.sh` + gatekeeper,配置就绪待首代学生);
- N4 收尾:中端手机实测 + 体积闸门核定。

## 不做(拍板)

以下均为**已拍板不做**,不是「还没做」;再评估需先推翻对应裁决。

- **多线程 / 背后思考(pondering)**:单线程到底;只在己方回合搜索(NN batch 化不算多线程);
- **按钟时间管理**:无 lagBuffer / byoyomi / 中残局时间分配 / obviousMoves 等 —— 难度即固定访问数;
- **开局库 / 残局库**:不内置任何着法表,开局与终局一律进搜索;
- **GTP / 引擎侧 SGF / 分析引擎**:不接任何围棋协议与 GUI;对弈页的「形势」是单次前向,
  不做 kata-analyze 式搜索精化;SGF 属于 WebOS 应用侧;
- **第三方引擎对接 / 多引擎**:仅一种引擎,棋力只靠自对弈 A/B 定方向;
- **非 WebGPU 后端**:无 WASM / CPU 回退,不可用时明确报错;
- **多模型 / 模型热切换 / .bin.gz 解析 / 量化**:仅内置一个 fp32 ONNX;
- **非中国规则 / 让子 / 其他盘径**:数目法、encore、button、simple/situational ko、
  多步自杀、2~25 路等一律不支持(特征编码器相应通道恒 0 占位,保证与训练数据逐位一致);
- **行为 / 特性项**:PDA(playoutDoublingAdvantage)、对手建模(visitCapContempt)、
  antiMirror、avoidRepeatedPatternUtility、wideRootNoise、avoidMYTDaggerHack、
  passing hacks 系列 —— 陪人下棋不需要;
- **human-SL**(按段位下人味棋):依赖模型侧支持,范围外;
- **落盘 eval cache**(KataGo `evalcache` 文件):内存 LRU 已覆盖转置复用,不做持久化;
- **WDL 三元效用**:中国规则 7.5 贴目无和棋、position superko 无无结果,winLoss 标量即精确;
- **根 Dirichlet 噪声 / subtreeValueBias**:训练侧机制,浏览器对人不噪声;训练全在 C++ 侧;
- **训练侧任何东西**:selfplay / shuffle / train / gatekeeper / dumponnx 全用 KataGo 官方工具链,
  不自己写训练侧搜索(架构见 [NEURAL_PLAN](docs/NEURAL_PLAN.md) §0)。

## 裁决记录(2026-10-03,原「存疑与待确认」五项全部落定)

1. **搜索参数基准** → **采用 KataGo GTP 实战配方**(cpuct 1.0 + 0.45·ln((N+500)/500)、
   FPU 0.2 / root 0.1)。现值(1.1 + 0.6·ln/4096、FPU 0.25)的切换列入待办棋力线 ①;
   切换落地前引擎行为不变。
2. **乐观策略通道** → **改用 KataGo 逻辑**(已核实源码):导出图的策略通道按模型版本为
   **C ∈ {1, 2, 4}**(v < 12 只有主策略;v ≥ 12 为 [0] 主策略 + [1] 乐观策略;
   v16 / v17 带 q 值再加 2 通道 q 值)—— 修正旧文档「6 通道」的错误。
   KataGo 在 NN 后端对每个落点与 pass 做逐点插值 `p + (pOpt − p) × λ`,
   **在 logits 空间、softmax 之前**;λ 逐评估传入(实战配方:树内 1.0、根 0.2)。
   老网没有乐观通道 —— 用通道 0 就是 KataGo 对老网的原生行为,不存在「老网乐观头
   质量」问题。接入列入待办棋力线 ①。
3. **温度** → 拍板:**温度由调用方传入,引擎不自带任何温度策略;需要随机时必须带衰减**。
   现实现即此设计(`think` 消息的 `temperature` / `temperatureHalflife`,按手数半衰),
   维持不动;开局要不要温度、给多少,是对弈页调用方的事。
4. **eval cache 是什么** → 解答:是的,可以理解为其他棋类引擎的**置换表(hash 表)**,
   只是存的东西不同 —— 棋类置换表存「这个局面搜出来的子树 / 结果」,这里存的是
   「NN 对这个局面的评估输出」:相同局面特征再次出现时直接查表拿评估,省一次推理。
   已在内存实现(2048 条 LRU,随 Worker 生命周期)。拍板:**保留内存版,不做落盘持久化**。
5. **参数格式** → 确认:搜索参数内置唯一一套,外部不可配置(无配置文件系统);
   难度四档(访问数)是产品分级,不属于参数配置。

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
| `{ type:'load', id, modelUrl }` | `{ type:'loaded', ep }` / `{ error }`(懒加载,仅 WebGPU) |
| `{ type:'state', id, moves }` | 棋盘 / 提子 / 劫点 / 合法着法 / 双停终局与数子 |
| `{ type:'think', id, moves, level, temperature?, temperatureHalflife? }` | 逐步 progress + `{ move, visits, winRate, scoreLead?, ms }` / `{ resign }` |
| `{ type:'score', id, moves, deadOverride? }` | 移除死子后的数子结果(含 `detail` 明细) |
| `{ type:'estimate', id, moves }` | 黑方胜率 / 目差(双口径:`netScoreLead` 网端 lead 优先,`scoreLead` 归属求和回落)/ ownership 图(黑方视角) |

`moves` 是从初始局面起的走法序列(0..360 或 PASS=361),Worker 自己重演棋盘
(结构化克隆最省,且不会有两份规则实现)。

## 测试

```bash
npm test                            # 规则 + 数子 + 记谱 + 模糊测试 + NN 特征 + KataGo 对齐项
node test/engine-test.mjs fuzz      # 只跑指定小节(--list 看全部)

npm run test:featdiff               # 特征对拍(需先用 katago 产数据,见 training/)
npm run test:nn-e2e                 # NN 自对弈端到端(需 python onnxruntime)
npm run match                       # NN vs NN 对战(等 visits)
node test/nn-temp-test.mjs          # 温度选点分布(LCB / 抽样)

npm run test:aewnn                  # 自研引擎对拍:WGSL vs CPU 参考 + CPU 参考 vs ort
                                    # golden(需 bleed 环境 onnxruntime;Dawn 绑定 npm i --no-save webgpu)
npm run bench:aewnn                 # 自研引擎吞吐基准(Dawn;WSL2 上为软件渲染下限)
npm run diff:aewnn                  # 分段对拍工具(内核回归定位用)
```

围棋没有 perft,规则正确性的金标准是**模糊测试**:随机对局 40 局 × 最多 420 手,
每一手验证盘面子数守恒(走前 + 1 − 提子 = 走后)与**禁全同不变量**
(非停一手产生的局面键整局互不重复),然后逐手撤销到空盘、再逐手重演 ——
每一步「走前的合法着法列表」必须逐步复原(劫点、提子、历史键全覆盖)。

NN 路线的金标准是**特征对拍**(`test/featdiff.mjs`):katago selfplay 训练行
(npz 的 `binaryInputNCHWPacked` / `globalInputNC`)与 JS 编码器逐位一致;
征子通道另有 `test/ladderdiff.mjs` 直接对拍 KataGo 原生 C++ 实现。

路线计划、训练管线与差距盘点的技术细节:**[docs/NEURAL_PLAN.md](docs/NEURAL_PLAN.md)**。

## License

MIT
