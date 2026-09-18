# AetherGo

纯 JavaScript 9×9 围棋引擎:零依赖、无 DOM、浏览器 / Worker / Node 通用。
从 [WebOS](https://github.com/suulnnka/AetherWebOS)(纯前端网页操作系统)的围棋应用中抽离而来,全部自研。

**在线体验:** 打开 <https://suulnnka.github.io/AetherWebOS/> 启动「围棋」应用 —— 那里面跑的就是本引擎
(默认高级档,窗口信息行实时显示演棋局数 / 胜率 / 耗时)。

> v0.1:规则完整(提子 / 禁自杀 / 单劫 / 双停终局),搜索是**第一版 UCT + 纯随机演棋**,
> 刻意做简单 —— 围棋分支因子大、静态评估难写,alpha-beta 在这里跑不动,
> 蒙特卡洛树搜索是正路。后续路线见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

## 棋盘与编码

- 81 个交叉点(9 行 × 9 列),`idx = 行×9 + 列`;行 0 是上边,列 0 是左边,黑先白后。
- 棋子:1 = 黑,2 = 白,0 = 空;行棋方 BLACK = 0 / WHITE = 1,棋子 = 行棋方 + 1。
- 走法:交叉点 0..80,`PASS = 81` 表示停一手。UI 与 Worker 之间只传这一种编码,不搞两套。
- 坐标记谱:列 A~J(跳过 I)+ 行 1~9(下边为 1),如天元 = `E5`。

## 引擎

`src/engine.js` 单文件(规则 + 数子 + 搜索),`src/worker.js` 只是 Worker 薄壳。
树节点池、未试手 arena、撤销栈全是模块级 typed array,搜索过程中**零分配**
(唯一例外:每次演棋复制一份 81 字节棋盘 —— 演棋的随机手不撤销,绝不能污染真盘)。

### 规则

气尽提子(整块)、禁自杀(能提子则不算)、simple ko(单劫:恰提一子且自己
落下的子只有一口气时,对方下一手不得立即回提)、停一手永远合法,
连续两手停即终局。着法生成即「逐点 isLegal」:临时落子 → 查四邻敌组是否气尽
(能提则合法)→ 查己组是否有气,只读棋盘、无副作用。

### 计分

中国规则数子法:己方子数 + 只被己方贴边的空点区域,黑贴 5.5 目。
引擎对局里死子会被自然提掉,数子即结果;「双停时残留死子」的处理见 ROADMAP P0。
演棋终局也用同一套数子 —— 所以引擎天然理解「提子等于加目」。

### 搜索(MCTS / UCT)

| 环节 | 现状 |
|---|---|
| 树结构 | 按列拆 typed array 节点池 + 兄弟链挂子;节点 id 即下标,0 恒为根 |
| 选点 | UCB1(`Q + C·√(ln N / n)`,C = 1.0),只穿过**展开完**的节点 |
| 扩展 | 每次到达弹一个随机未试手(未试手列表存在共享 arena 里,逐个消耗) |
| 演棋 | 纯随机合法点 + 不填自己的真眼(4 邻全己方),找不到就停一手 |
| 终局 | 双停即数子;演棋 162 手(2×81)封顶,到点直接数子 |
| 回传 | 节点存「走进该节点那一方」的胜局数;根按访问次数选着法 |
| 停一手 | 空点 ≤ 25、无点可下或对方刚停时才进未试手列表 —— 防止早早乱停 |
| 弱化档 | 初级在「最优胜率 − 0.25」且访问 ≥ 4 的根子集里随机挑,弱得可控 |

**还没做**(见 ROADMAP):局部响应 / 战术响应的演棋策略、RAVE、树复用、
禁全同、终局死子估计。

### 难度四档

演棋局数为主(设备无关、可复现),墙上时间为兜底。
下表的「实测」是 `bench/bench.mjs nps` 在初始局面跑出来的(Node 24 / 桌面级 CPU):

| 档位 | 演棋预算 | 实测 | 实测速度 |
|---|---|---|---|
| 初级 | 400 | 90ms | ~4.4k/s,jitter 0.25 |
| 中级 | 2,500 | 0.5s | ~5.0k/s |
| 高级 | 9,000 | 1.9s | ~4.8k/s |
| 大师 | 24,000 | 5.4s | ~4.5k/s |

棋力定位:轻演棋 MCTS,约等于休闲棋手 —— 开局能占到角、终局知道停,
但战术上有「立即提子不紧迫」(随机演棋里那颗子横竖会被提)这类已知弱点,
改进项都在 ROADMAP 里。

## 用法

```js
import { BLACK, WHITE, PASS, newBoard, genLegal, isLegal, make, unmake,
         capturedOf, scoreGame, moveToText, searchBest, LEVELS } from './src/engine.js';

const bd = newBoard();                        // Int8Array(81)
const moves = genLegal(bd, BLACK);            // 黑方合法落点(停一手不进列表)
console.log(isLegal(bd, BLACK, 40));          // (4,4) 是否合法
const tok = make(bd, 40, BLACK);              // 走子:返回撤销令牌(含提子数)
console.log(capturedOf(tok));                 // 这一手提了几颗
unmake(bd, 40, tok);                          // 撤销

console.log(moveToText(bd, 40));              // 'E5'(记谱不依赖盘面)

const r = searchBest(bd, BLACK, { ...LEVELS[2], onProgress: (p) => console.log(p.visits, p.winRate) });
// r = { move, winRate, visits, nodes, ms, only };move = PASS 表示停一手
const s = scoreGame(bd);                      // { black, white, margin } 数子结果
```

Worker 侧收 `{ id, moves, playouts, ms, jitter }`,回
`{ id, move, visits, nodes, ms, winRate, only }`;逐次回
`{ id, type:'progress', visits, move, winRate, ms }`。
`moves` 是从初始局面起的走法序列(0..80 或 PASS),Worker 自己重演棋盘
(结构化克隆最省,且不会有两份规则实现)。

## 测试与基准

```bash
npm test                            # 规则用例 + 数子 + 记谱 + 搜索行为 + 随机对局模糊测试
node test/engine-test.mjs fuzz      # 只跑指定小节(--list 看全部)

npm run bench                       # 各档位演棋速度(初始局面)
npm run moves                       # 固定演棋数最佳着法(改搜索/改演棋后对拍)
```

围棋没有 perft,规则正确性的金标准是**模糊测试**:随机对局 40 局 × 最多 220 手,
每一手验证盘面子数守恒(走前 + 1 − 提子 = 走后),然后逐手撤销到空盘、
再逐手重演 —— 每一步「走前的合法着法列表」必须逐步复原(连劫点禁着都覆盖)。

## 已知不做(v0.1 的边界)

- simple ko 而非禁全同 —— 循环劫 / 三劫会来回提,见 ROADMAP P0
- 双停时残留死子会把数子算错(引擎对局不受影响),见 ROADMAP P0
- 演棋不查斜角假眼(把部分假眼当真眼避开),轻微影响终盘精度

## 明确不做(已拍板,不是「还没做」)

- **神经网络 / WASM 推理**:纯 JS + 纯 MCTS,与黑白棋、两个象棋引擎同一风格
- **GTP / 引擎侧 SGF**:不接主流围棋 GUI;SGF 属于 WebOS 应用侧
- **开局库 / 残局库**:不内置任何着法表,开局与终盘一律进搜索
- **多线程**:单线程到底
- **接第三方引擎对打**:棋力只靠自对弈 A/B 定方向

> 体积预算 35 KB gzip(与国际象棋、中国象棋引擎同档),由 WebOS 侧
> `tools/check-size.mjs` 在 `npm run build` 时拦(标记 `go-engine-v1`);当前约 5 KB。

## License

MIT
