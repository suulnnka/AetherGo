/* ============================================================
 * AetherGo NN 版异步 PUCT 搜索 —— v4:图搜索(useGraphSearch,最终选点/效用同 v3)
 *
 * == v4.5(2026-10-06):攒批逻辑改 KataGo 多线程投影(用户指令) ==
 *   - 批上限 = 在途评估总量 T(线程数语义):在飞批 + 待发队列 ≤ T,
 *     任何下降的统计盲区 ≤ T−1 —— KataGo「线程停在叶上等自己的评估」的
 *     异步单线程等价。旧 v4.1 管线 minBatch 空闲凑满 + 在途时不计在途量,
 *     盲区最深 2T−1:同代码 64v 自对弈批 1 对批 4 = 10-2、批 4 口径 vs
 *     KataGo 1-5(批 1 口径 3-3),批税主因即此。死端/预算边界立即发射
 *     手头半批(机会主义不凑批),批序严格。
 *
 * == v4.4(2026-10-06):subtreeValueBias 移植(setup GTP 默认 0.45)==
 *   - 同「行棋方+上二手+落点 5×5 局部形(8 对称规范化)+劫」签名的节点共享
 *     在线表项,重算时累计(子树均值−自身评估)·origTotal^0.85,自身评估
 *     效用往表项均值偏移 0.45 份;叶首评/死端同偏,终局值不偏;GC 删除
 *     节点回退 80% 贡献。校正网络评估的局部系统性偏差,锚定更实。
 *
 * == v4.3(2026-10-06):搜索随机对称(nnEvaluator 同款)==
 *   - 每次评估随机取 8 对称之一:spatial 点映射通道整体置换(等价变换局面
 *     后编码),policy/ownership 逆置换回恒等系;值输出与 global 不变量。
 *     模型对称增广训练下,变换输入评估≈变换输出,残差即去相关评估噪声,
 *     打破确定性访问锁定(P7 集中度:C3:57 vs KataGo 运行带 23-35)。
 *   - 评估缓存键取变换后特征(8 子缓存,同输入同输出不变量保持);
 *     opt.symmetry===false 关闭(位置敏感桩测试),opt.rngSeed 可复现。
 *
 * == v4.2(2026-10-06):重算式节点统计(KataGo recomputeNodeStats 移植)==
 *   - 节点统计不再沿路径累积叶值:每次回传后自叶向根逐节点从「子边统计 +
 *     自身 NN 评估」重算 —— 子权重 = getChildWeight(边分摊),good 子按先验序
 *     过 pruneNoiseWeight + valueWeight t3 降权(终选同款公式,搜索期即生效),
 *     加权混合子均值,再并入自身评估一份(不确定度权重)。
 *     Q 由此成为向网络评估收缩的稳定估计(少访 ≈ 自身评估,多访 ≈ 良序子树
 *     均值),坏子不再稀释父值 —— 消除累积式的「探索噪声沉淀」。
 *   - 虚拟损失移出统计:纯计数器,选点期混合效用向 ±R + 分母膨胀
 *     (getExploreSelectionValueOfChild 同款),回传统计保持纯净可重算。
 *   - 终局值权重 = uncertaintyMaxWeight(KataGo addLeafValue 终局口径)。
 *   - 回传统计一律加权(util/wl/utilSq × weight;v4.1 前分子不加权,Q 幅度
 *     被均权压缩 → 劈分偏均匀,2026-10-06 差分实锤后修正)。
 *
 * == v4.1(2026-10-04):单槽管线 + 批回传叶盘面修复 ==
 *   - 单槽管线(双并行):GPU 估值第 N 批时,主循环同步攒第 N+1 批;估值慢则
 *     攒完在 await 处等。KataGo 多线程「线程停在叶上等 NN」的单线程投影 ——
 *     下降见到的统计 stale 恰一批 = C++ 多线程原生语义(非偏离);在途批的
 *     evalPending 使下一批自动避开同节点(转置去重),apply 严格按批序。
 *   - 批回传沿 path 重 make 到叶再展开:展开时盘面在叶上(C++ runSinglePlayout
 *     全程 board 在叶、playout 末尾才复位,search.cpp:1263/1331)。旧版批路径
 *     把已退根的盘面传给 expand,真眼剪枝/单官压制读的是根盘面(cached 分支
 *     传的是叶盘面,两分支不一致)。
 *   - 终局节点不再挂 evalPending:挂着会被子选点永久跳过,双停终局值只在
 *     创建那次 playout 生效。
 *   - 需求驱动动态批:攒批「在途批结果一到即发射手头半批,否则攒到 maxBatch」
 *     (KataGo waitPopUpToN 语义);批上限 = session 校准值(session.maxBatch,
 *     createSession 现测吞吐 → 最优 90% 的最小批)再按预算压 stale(≤ 预算/16)。
 *
 * == 图搜索核心(KataGo search.cpp + graphhash.cpp 对齐)==
 *   - 节点表:Map<chainKey, node>,Worker 生命期内跨手持久 —— 转置局面共享
 *     同一节点(访问/效用统计合并),树复用升级为「子图复用」。
 *   - 键控(graphhash.cpp 语义):子节点键 = 链式 mix(父键, 状态键) 或 纯状态键 ——
 *     由「最后一手周边空域数」裁决:空域 > graphSearchRepBound(11) → 纯状态键
 *     (大范围着法后历史局部性弱,按局面合并、可转置);局部战斗(≤11)→ 链式
 *     (路径唯一,杜绝短循环)。PASS 走链式(KataGo 同)。状态键 = 盘面 Zobrist
 *     ⊕ 行棋方 ⊕ 劫/禁点 ⊕ 尾随停着数 ⊕ passWouldEndGame。
 *   - 惰性子节点:展开只建「着法+先验」条目;子节点本体在下降首次经过该边时
 *     才建表/查表 —— 转置命中即接入既有子树(KataGo allocateOrFindNode 同款)。
 *   - 边访问缩放(searchnode.h getChildWeight):共享节点的统计按
 *     「该边访问数 / 节点总访问数」占比分摊到各父 —— 选点分母与最终选点权重
 *     均用缩放后权重,效用均值用节点全局值。
 *   - 循环守卫:下降路径节点打戳,选点跳过在途节点(单线程下的干净等价,
 *     KataGo 用 graphPath 集合在到达后终止 playout)。位置超劫禁则物理不可循环:
 *     路径即真实对局历史的延伸,重复局面被 position superko 禁止。
 *   - GC:每次思考后从根 DFS 标记可达节点,清扫节点表(KataGo mark-and-sweep 同款)。
 *
 * == 沿用 v3(2026-10-03 拍板的 GTP 实战配方,详见下节与 NEURAL_PLAN §5.1)==
 *   效用函数(winLoss+static/dynamic 目差)、FPU(按已访问 policy 混合)、
 *   不确定度加权、真实 LCB(ESS)、noisePruning、valueWeightExponent(t₃ CDF)、
 *   cpuct 方差因子、根评估重算(λ=0.2)、根对称剪枝、无用着剪枝、
 *   批量推理(8 叶+虚拟损失)、温度选点(调用方传入+半衰)、
 *   认输(worker 层,−0.90 连续 3 手)。
 *
 * == 与 KataGo 的已记录偏差 ==
 *   - maybeCatchUpEdgeVisits(边访问追平加速)已实现但默认关闭(A/B 存疑,
 *     见 ENABLE_CATCHUP 注);
 *   - 循环到达即终止(KataGo)改为选点跳过在途节点(等价且少浪费一次下降);
 *   - 转置子树的着法合法性按首访路径生成,不按新路径重查(KataGo 同);
 *   - 根键 = fnv(整局着法)+状态键(KataGo 为逐手链式重算,语义等价)。
 *
 * 视角约定:节点 util/wl 累计为「走进该节点那一方」视角;scoreMean/scoreMeanSq
 * 恒白方视角。效用域半径 R = 1.4(LCB 方差先验用)。
 * ============================================================ */
import {
  N, N2, EMPTY, PASS, KOMI, genLegal, make, unmake, scoreGame, BLACK, WHITE,
  superkoBannedPoints, ringSnapshot, ringRestore, positionKey, koPoint,
} from '../engine.js';
import { encodeFeatures } from './features.js';
import { lookupEval, storeEval, fevalKey, clearEvalCache as clearEvalCacheImpl } from './eval-cache.js';
import { SYM8, unpermuteOut } from './symmetry.js';
import { pickBest, pickMove, effectiveTemperature } from './move-select.js';

/* 兼容再导出:测试与外部只认 search.js 一个入口。
 * clearEvalCache 同时清空评估缓存与图节点表(测试隔离用)。 */
export { pickBest, pickMove, effectiveTemperature };
export function clearEvalCache() {
  clearEvalCacheImpl();
  nodeTable = new Map();
  treeKeep = null;
  biasTable = new Map();               // subtreeValueBias 表同随测试隔离清空
}

/* ==================== GTP 实战配方常数(setup.cpp SETUP_FOR_GTP) ==================== */
const CPUCT = 1.0, CPUCT_LOG = 0.45, CPUCT_BASE = 500;
const PUCT_OFFSET = 0.01;               // TOTALCHILDWEIGHT_PUCT_OFFSET
const FPU = 0.2, ROOT_FPU = 0.1;
const FPU_BLEND_POW = 2.0;              // fpuParentWeightByVisitedPolicyPow
const STATIC_F = 0.1, DYNAMIC_F = 0.3;
const CENTER_ZERO_W = 0.20, CENTER_SCALE = 0.75;
const STDEV_PRIOR = 0.40, STDEV_PRIOR_W = 2.0, STDEV_SCALE = 0.85;
const UNCERT_COEFF = 0.25, UNCERT_MAX_W = 8.0;
const VALUE_WEIGHT_EXP = 0.25;
const NOISE_PRUNE_SCALE = 0.15;
const LCB_STDEVS = 5.0, LCB_MIN_PROP = 0.15;
const UTILITY_RADIUS = 1.0 + STATIC_F + DYNAMIC_F;
const PRIOR_FLOOR = 1e-4;
/* 2026-10-05 移除 PASS_SUPPRESS(pass 先验压制):KataGo 的 shouldSuppressPass
 * (fillDameBeforePass)只在数目法(TERRITORY)下生效,面积计分完全不压制 ——
 * 填单官本身涨目,网络值自己学会何时停。旧版用 countDame>0 无条件压制,
 * 且口径过宽(争议区/双活也计入),已定局面永不 pass、填到 400+ 手
 * (真实对局差分:game1 ply380 KataGo pass 拿 26 访,我方 pass 恒 0 访)。 */
const ROOT_OPTIMISM = 0.2;              // rootPolicyOptimism(GTP;树内 λ=1.0)
const REP_BOUND = 11;                   // graphSearchRepBound(GTP)
/* 转置边访问追平开关:合成回传版在 A/B 自对弈中表现存疑(2026-10-05),
 * 视角修复后默认关闭,待单独验证后再启 */
const ENABLE_CATCHUP = false;

const TWO_OVER_PI = 2 / Math.PI;
const SQRT_AREA = Math.sqrt(N2);

/* ==================== 搜索随机对称(KataGo nnEvaluator 同款) ====================
 * 每次评估随机取 8 对称之一,把对称编号随行下发(rows[i].sym)—— 置换在
 * **引擎侧**做(aewnn:stem 卷积按 gather 表直接以变换后坐标取输入,零 CPU
 * 置换)。global 与值输出
 * (winLoss/scoreMean/policyPass)为不变量;policy/ownership 按逆置换
 * (unpermuteOut,dst[p] = src[perm[p]])还原到恒等坐标系。模型经对称增广
 * 训练,对变换输入的评估 ≈ 变换输出,残差即去相关评估噪声 —— 打破确定性
 * 访问锁定(2026-10-06 P7 集中度:C3:57 vs KataGo 运行带 23-35,KataGo 侧
 * 即靠此噪声维持访问分散,恒等对称的引擎则单点锁死)。
 * 评估缓存键取「原始特征 + sym」(同变换同键才命中,「同输入必同输出」
 * 不变量保持,等效 8 个子缓存)。opt.symmetry === false 关闭(位置敏感桩
 * 测试用);opt.rngSeed 固定种子可复现(测试)。SYM8/unpermuteOut 见
 * ./symmetry.js。特征编码进环形槽位(spRing/glRing):在飞批 + 攒批各持
 * 一份槽位,pending 跨 await 引用安全且零拷贝(aewnn 在 evalBatch 调用的
 * 同步前缀里直传 GPU,随后槽位即可复用,环形容量按 2×maxBatch 兜底)。 */
let symRng = ((Date.now() ^ (Math.random() * 0x7fffffff)) >>> 0) || 0x9e3779b9;
function nextSym() {
  let x = symRng;
  x ^= (x << 13) >>> 0; x ^= x >>> 17; x ^= (x << 5) >>> 0;
  symRng = x >>> 0;
  return x & 7;
}

/* ==================== subtreeValueBias(searchupdatehelpers.cpp:287 / subtreevaluebiastable.cpp,setup GTP 默认 0.45) ====================
 * 跨节点在线校正自身评估锚:同「行棋方 + 上二手 + 落点 5×5 局部形(8 对称
 * 规范化,行棋方相对色)+ 劫」签名的节点共享一个表项,重算时累计
 * (子树均值 − 自身评估)·origTotal^0.85,并把自身评估效用往表项均值方向
 * 偏移 factor 份;叶首评/死端再计同样偏移;终局值不偏移。GC 删除节点时按
 * freeProp 回退 80% 贡献(20% 沉淀为历史证据)。表 Worker 生命期内跨手
 * 持久(KataGo Search 对象同生命周期)。 */
const SUBTREE_BIAS_F = 0.45;               // subtreeValueBiasFactor
const SUBTREE_BIAS_WEXP = 0.85;            // subtreeValueBiasWeightExponent
const SUBTREE_BIAS_FREEPROP = 0.8;         // subtreeValueBiasFreeProp
let biasTable = new Map();                 // key → { d, w }
const BIAS_WIN_PERMS = (() => {
  const perms = [];
  for (let s = 0; s < 8; s++) {
    const idx = new Int32Array(25);
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
      let r = i, c = j;
      for (let k = 0; k < (s & 3); k++) { const t = r; r = c; c = 4 - t; }
      if (s & 4) c = 4 - c;
      idx[i * 5 + j] = r * 5 + c;
    }
    perms.push(idx);
  }
  return perms;
})();
/* 签名键:bd 须为落子前盘面(search.cpp:977 getRecentBoard(1) 语义,
 * 窗口中心 = 落点,中心格为空);prevMove = 再上一手(父节点自己的着法)。 */
function biasKeyOf(bd, mover, prevMove, mv, ko) {
  if (mv === PASS || prevMove === PASS || prevMove < 0) return null;
  const r0 = (mv / N) | 0, c0 = mv % N;
  const own = mover + 1;
  const cells = new Array(25);
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const r = r0 + i - 2, c = c0 + j - 2;
    if (r < 0 || r >= N || c < 0 || c >= N) { cells[i * 5 + j] = 3; continue; }
    const s = bd[r * N + c];
    cells[i * 5 + j] = s === EMPTY ? 0 : (s === own ? 1 : 2);
  }
  let best = null;
  for (const perm of BIAS_WIN_PERMS) {
    let k = "";
    for (let t = 0; t < 25; t++) k += String.fromCharCode(48 + cells[perm[t]]);
    if (best === null || k < best) best = k;
  }
  return mover + "|" + prevMove + "|" + mv + "|" + best + "|" + ko;
}
function biasOf(node) {
  if (node.biasKey == null) return 0;
  const e = biasTable.get(node.biasKey);
  return e && e.w > 0.001 ? SUBTREE_BIAS_F * e.d / e.w : 0;
}

/* ==================== ScoreValue JS 版(nninputs.cpp 权威) ==================== */

function expectedScoreValue(m, sd, center, scale) {
  const den = scale * SQRT_AREA;
  if (sd < 1e-9) return Math.atan((m - center) / den) * TWO_OVER_PI;
  const s3 = Math.sqrt(3) * sd;
  const f = (x) => Math.atan((x - center) / den);
  return (2 / 3) * f(m) + (1 / 6) * (f(m + s3) + f(m - s3));
}

function scoreValueDeriv(m, center, scale) {
  const sf = scale * SQRT_AREA, a = m - center;
  return sf / (sf * sf + a * a) * TWO_OVER_PI;
}

/* ==================== 着法筛选助手(无用着 / 单官) ==================== */

export function isOwnTrueEye(bd, side, p) {
  const mine = side + 1;
  const r = (p / N) | 0, c = p % N;
  if (r > 0 && bd[p - N] !== mine) return false;
  if (r < N - 1 && bd[p + N] !== mine) return false;
  if (c > 0 && bd[p - 1] !== mine) return false;
  if (c < N - 1 && bd[p + 1] !== mine) return false;
  let enemyDiag = 0;
  for (let dr = -1; dr <= 1; dr += 2) {
    for (let dc = -1; dc <= 1; dc += 2) {
      const rr = r + dr, cc = c + dc;
      if (rr < 0 || rr >= N || cc < 0 || cc >= N) continue;
      if (bd[rr * N + cc] === 3 - mine) enemyDiag++;
    }
  }
  return enemyDiag === 0;
}

export function countDame(bd) {
  const seen = new Uint8Array(N2);
  const stack = [];
  let dame = 0;
  for (let seed = 0; seed < N2; seed++) {
    if (bd[seed] !== EMPTY || seen[seed]) continue;
    stack.length = 0; stack.push(seed); seen[seed] = 1;
    let touch = 0, size = 0;
    while (stack.length) {
      const q = stack.pop(); size++;
      const r = (q / N) | 0, c = q % N;
      for (let k = 0; k < 4; k++) {
        let nb = -1;
        if (k === 0 && r > 0) nb = q - N;
        else if (k === 1 && r < N - 1) nb = q + N;
        else if (k === 2 && c > 0) nb = q - 1;
        else if (k === 3 && c < N - 1) nb = q + 1;
        if (nb < 0) continue;
        const v = bd[nb];
        if (v === EMPTY) { if (!seen[nb]) { seen[nb] = 1; stack.push(nb); } }
        else touch |= (v === 1 ? 1 : 2);
      }
    }
    if (touch !== 1 && touch !== 2) dame += size;
  }
  return dame;
}

/* ==================== 图搜索:节点表与键控(graphhash.cpp 对齐) ==================== */


let nodeTable = new Map();              // chainKey → node(Worker 生命期内持久)
let gcEpoch = 0;
let treeKeep = null;                    // { root, recentMoves }

/* 双种子 FNV(字符串 → "h1,h2")—— 链式键压缩用 */
function fnv2(str) {
  let h1 = 0x811c9dc5, h2 = 0x01000193;
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 16777619);
    h2 = Math.imul(h2 + c ^ (i * 40503 | 0), 16777619);
  }
  return `${h1},${h2}`;
}

const BAN_SCRATCH = new Uint8Array(N2);
/* 状态键:盘面 Zobrist ⊕ 行棋方 ⊕ 劫/禁点 ⊕ 尾随停着 ⊕ passWouldEndGame。
 * bd/side 必须处于 make 后的活状态(节点创建时机保证)。 */
function stateKeyOf(bd, side, trailingPasses) {
  const bans = superkoBannedPoints(bd, side, BAN_SCRATCH);
  let bh = 0;
  for (let p = 0; p < N2; p++) if (bans[p]) bh = (bh + p + 1) | 0;
  const passEnds = trailingPasses >= 1 ? 1 : 0;
  return `${positionKey()}|${side}|${koPoint()}|${bh}|${trailingPasses}|${passEnds}`;
}

/* 最后一手周边「活动区域」空点数(graphhash.cpp simpleRepetitionBoundGt 的
 * 计数口径:链长 + 相邻连通空域),> REP_BOUND → 状态键合并,否则链式。 */
const REG_SEEN = new Int32Array(N2);
let regStamp = 0;
function regionCount(bd, mv) {
  let total = 0;
  regStamp++;
  const floodEmpty = (start) => {
    const st = [start];
    REG_SEEN[start] = regStamp;
    while (st.length) {
      const q = st.pop(); total++;
      if (total > REP_BOUND) return;
      const r = (q / N) | 0, c = q % N;
      for (let k = 0; k < 4; k++) {
        let nb = -1;
        if (k === 0 && r > 0) nb = q - N;
        else if (k === 1 && r < N - 1) nb = q + N;
        else if (k === 2 && c > 0) nb = q - 1;
        else if (k === 3 && c < N - 1) nb = q + 1;
        if (nb >= 0 && bd[nb] === EMPTY && REG_SEEN[nb] !== regStamp) { REG_SEEN[nb] = regStamp; st.push(nb); }
      }
    }
  };
  if (mv === PASS) return 0;                        // KataGo:PASS → 链式
  if (bd[mv] === EMPTY) { floodEmpty(mv); return total; }   // snapback 空点
  /* 链长 + 每气连通空域 */
  const st = [mv], chainSeen = new Set([mv]);
  const chain = [];
  while (st.length) {
    const q = st.pop(); chain.push(q);
    const r = (q / N) | 0, c = q % N;
    for (let k = 0; k < 4; k++) {
      let nb = -1;
      if (k === 0 && r > 0) nb = q - N;
      else if (k === 1 && r < N - 1) nb = q + N;
      else if (k === 2 && c > 0) nb = q - 1;
      else if (k === 3 && c < N - 1) nb = q + 1;
      if (nb < 0) continue;
      if (bd[nb] === bd[mv] && !chainSeen.has(nb)) { chainSeen.add(nb); st.push(nb); }
    }
  }
  total = chain.length;
  if (total > REP_BOUND) return total;
  for (const q of chain) {
    const r = (q / N) | 0, c = q % N;
    for (let k = 0; k < 4; k++) {
      let nb = -1;
      if (k === 0 && r > 0) nb = q - N;
      else if (k === 1 && r < N - 1) nb = q + N;
      else if (k === 2 && c > 0) nb = q - 1;
      else if (k === 3 && c < N - 1) nb = q + 1;
      if (nb >= 0 && bd[nb] === EMPTY && REG_SEEN[nb] !== regStamp) floodEmpty(nb);
      if (total > REP_BOUND) return total;
    }
  }
  return total;
}

/** 子节点键:局部战斗链式(路径唯一),大范围着法状态键(可转置) */
function childChainKey(parent, mv, bd, childSide, trailingAfter) {
  const sk = stateKeyOf(bd, childSide, trailingAfter);
  if (mv !== PASS && regionCount(bd, mv) > REP_BOUND) return { key: `S${sk}`, state: true };
  return { key: `C${fnv2(parent.chainKey)}|${sk}`, state: false };
}

function computeTrailing(moves) {
  let t = 0;
  for (let i = moves.length - 1; i >= 0 && moves[i] === PASS; i--) t++;
  return t;
}

/** 测试钩子:当前节点表规模 */
export const __nodeTableSize = () => nodeTable.size;
/** 测试钩子:仅清 subtreeValueBias 表(隔离「二次搜索可复现」类断言 ——
 * bias 跨搜索学习是 KataGo 忠实语义,表随引擎生命期持久) */
export function __clearBiasTable() { biasTable = new Map(); }

/* ==================== t 分布(ν=3)CDF 闭式 ==================== */
function t3cdf(x) {
  const y = x / Math.sqrt(3);
  return 0.5 + (y / (1 + y * y) + Math.atan(y)) / Math.PI;
}

/**
 * opt: { session, visits(≤1 = 模型直出:仅根评估,选点 = 原始策略 argmax),
 *        komi = KOMI, recentMoves, onProgress,
 *        temperature = 0, temperatureHalflife = 0,
 *        maxBatch(校准批上限;旧 opt.batch 兜底,缺省 4),
 *        reuseTree = true(默认:节点表跨手持久), debug = false }
 * 返回:{ move, winRate, visits, nodes, ms, only, reused, nnCalls, cacheHits,
 *        temperature, scoreLead, debug: rootChildren/rootChildMoves/rootChildStats }
 */
export async function nnSearchBest(bd, side, opt = {}) {
  const t0 = Date.now();
  const komi = opt.komi ?? KOMI;
  const budget = opt.visits ?? 300;
  const temperature = opt.temperature ?? 0;
  const temperatureHalflife = opt.temperatureHalflife ?? 0;
  /* 批上限:校准值(opt.maxBatch,session 加载时现测)优先,旧 opt.batch 兜底;
   * 再按预算压 stale(批 ≤ 预算/16 → 至少 16 轮回传,下限 2),小预算自动小批 */
  /* 批上限 = 在途评估总量上限 T(虚拟线程数语义):在飞批 + 待发队列 ≤ T,
   * 任何下降的统计盲区 ≤ T−1,与 KataGo numSearchThreads=T 的多线程语义一致
   * (v4.5;旧 minBatch 凑批 + 在途不看量的上限使盲区最深 2T−1,同代码 64v
   * 自对弈批 1 对批 4 = 10-2,见文件头)。沿用 opt.maxBatch(session 校准值)
   * 与预算钳制(≥16 轮回传)。 */
  const maxBatch = Math.max(1, Math.min(opt.maxBatch ?? opt.batch ?? 4, Math.max(2, budget >> 4)));
  const evalBatch = opt.session.evalBatch.bind(opt.session);
  const rootRecent = opt.reuseTree === false ? (opt.recentMoves ?? []).slice() : (opt.recentMoves ?? []);
  const spBuf = new Float32Array(22 * N2), glBuf = new Float32Array(19);
  /* 特征环形槽位(零拷贝发送):叶子特征直接编码进槽,pending 跨 await 引用;
   * 在飞批 + 攒批最多同时占用 2×maxBatch 份,容量再加余量。槽位在
   * applyBatch 后自然回收(按计数取模复用)。 */
  const ringCap = 2 * (maxBatch ?? 8) + 8;
  const spRing = Array.from({ length: ringCap }, () => new Float32Array(22 * N2));
  const glRing = Array.from({ length: ringCap }, () => new Float32Array(19));
  let ringCounter = 0;
  const ringNext = () => (ringCounter++ % ringCap);
  let cacheHits = 0;
  let recentScoreCenter = 0;

  /* reuseTree=false:本搜索用独立节点表(不读不写全局表) */
  const table = opt.reuseTree === false ? new Map() : nodeTable;

  const makeNode = (move, stm, chainKey) => ({
    move, side: stm, children: null,
    visits: 0, weight: 0, weightSq: 0,
    util: 0, utilSq: 0, wl: 0,
    scoreMean: 0, scoreMeanSq: 0,
    nn: null, terminal: false, terminalSem: null,
    chainKey, evalPending: false, vl: 0, _gc: 0,
    biasKey: null, lastBD: 0, lastBW: 0,
  });

  /* ---- 语义值管道(同 v3) ----
   * ★ session 契约的 winLoss/scoreMean 是**行棋方**视角(nneval.cpp 权威:
   * "the neural net gives us back the value from the perspective of the
   * player",C++ 在后处理末尾按 nextPlayer==P_BLACK 整体取反转白方视角)。
   * 本管道的 wlW/mW 一律白方视角 —— 按 stm 翻转。2026-10-05 差分调试实锤:
   * 旧版漏翻,黑行棋叶的效用以反号进树,值信号半数损坏(棋力差距主因)。 */
  const semOf = (out, stm) => {
    const flip = stm === WHITE ? 1 : -1;
    const hasScore = Number.isFinite(out.scoreMean);
    return {
      wlW: flip * out.winLoss,
      mW: flip * (hasScore ? out.scoreMean : 0),
      sdW: hasScore ? (out.scoreStdev ?? 0) : 0,
      stWL: out.shorttermWinlossError ?? 0,
      stScore: out.shorttermScoreError ?? 0,
      hasScore,
    };
  };
  const utilityWhite = (sem) => sem.wlW + (sem.hasScore ? scoreUtility(sem.mW, sem.sdW) : 0);
  function scoreUtility(mW, sdW) {
    return STATIC_F * expectedScoreValue(mW, sdW, 0, 2)
         + DYNAMIC_F * expectedScoreValue(mW, sdW, recentScoreCenter, CENTER_SCALE);
  }
  function scoreUtilityDeriv(mW) {
    return STATIC_F * scoreValueDeriv(mW, 0, 2)
         + DYNAMIC_F * scoreValueDeriv(mW, recentScoreCenter, CENTER_SCALE);
  }
  const uncertaintyWeight = (sem) => {
    if (!sem.hasScore) return 1.0;
    const unc = 1.0 * sem.stWL + scoreUtilityDeriv(sem.mW) * sem.stScore;
    return UNCERT_COEFF / (unc + UNCERT_COEFF / UNCERT_MAX_W);
  };

  /* ---- 根节点:表查/建 + 树复用(reuse 模式沿子边下移) ---- */
  let root, reused = 0;
  const rootStateKey = stateKeyOf(bd, side, computeTrailing(rootRecent));
  const rootKey = `R${fnv2(rootRecent.join(','))}|${rootStateKey}`;
  if (opt.reuseTree !== false && treeKeep
    && treeKeep.recentMoves.length + 2 <= rootRecent.length
    && rootRecent.slice(0, treeKeep.recentMoves.length).every((m, i) => m === treeKeep.recentMoves[i])) {
    /* 沿「己方上一手 + 对方应手」下移(子图复用) */
    let n = treeKeep.root;
    const tail = rootRecent.slice(treeKeep.recentMoves.length);
    let okk = n.side === (treeKeep.recentMoves.length % 2 === 0 ? side : side ^ 1);
    for (const mv of tail) {
      const ch = (n.children ?? []).find((c) => c.move === mv && c.node);
      if (!ch || !ch.node.children) { okk = false; break; }
      n = ch.node;
      reused++;
    }
    if (okk && n.children && n.children.length) root = n;
  }
  if (!root) {
    root = table.get(rootKey) ?? null;
    if (root) reused = -1;                        // 跨局命中(同局面重开)
  }
  if (!root) {
    root = makeNode(-1, side, rootKey);
    table.set(rootKey, root);
    reused = 0;
  }

  /* ---- 虚拟损失(v4.2:纯计数器,不进统计) ----
   * 攒批期间对在途路径子计数;选点期按 KataGo getExploreSelectionValueOfChild
   * (searchexplorehelpers.cpp:140)混合:效用向「对行棋方最坏」的 ±utilityRadius
   * 收缩 + PUCT 分母膨胀。回传时计数归零,统计保持可重算的纯净口径。 */
  const applyVirtualLoss = (path) => {
    for (let i = 1; i < path.length; i++) path[i].node.vl++;
  };
  const removeVirtualLoss = (path) => {
    for (let i = 1; i < path.length; i++) path[i].node.vl--;
  };

  /* ---- 重算式节点统计(KataGo recomputeNodeStats,searchupdatehelpers.cpp:167)----
   * 节点 util/wl 存「走进节点那方」视角(沿用旧约定),scoreMean 恒白视角;
   * 内部按 KataGo 白视角公式计算后统一翻转。u² 与视角无关。
   * 权威语义:统计 = Σ_good wAdj·子均值 + 自身评估×不确定度权重,
   * wAdj 先经 pruneNoiseWeight(先验序)再 valueWeight t3 降权归一。 */
  function initLeafStats(node, sem) {
    /* 叶首评:自身评估一份(searchnnhelpers.cpp:171 addCurrentNNOutputAsLeafValue
     * assumeNoExistingWeight=true,REPLACE 语义)+ subtreeValueBias 偏移
     * (addLeafValue 同款,非终局才偏) */
    const uw = uncertaintyWeight(sem);
    const uW = utilityWhite(sem) + biasOf(node);
    const flip = (node.side ^ 1) === WHITE ? 1 : -1;
    node.visits = 1;
    node.weight = uw; node.weightSq = uw * uw;
    node.util = flip * uW * uw;
    node.utilSq = uW * uW * uw;
    node.wl = flip * sem.wlW * uw;
    node.scoreMean = sem.hasScore ? sem.mW * uw : 0;
    node.scoreMeanSq = sem.hasScore ? sem.mW * sem.mW * uw : 0;
  }
  function accumulateSelfEval(node) {
    /* 死端:自身评估再计一份(search.cpp:1408 addCurrentNNOutputAsLeafValue
     * accumulate 语义 —— 全子被禁时节点困住计访),同样带 bias 偏移 */
    const sem = node.nn;
    if (!sem) return;
    const uw = uncertaintyWeight(sem);
    const uW = utilityWhite(sem) + biasOf(node);
    const flip = (node.side ^ 1) === WHITE ? 1 : -1;
    node.visits++;
    node.weight += uw; node.weightSq += uw * uw;
    node.util += flip * uW * uw;
    node.utilSq += uW * uW * uw;
    node.wl += flip * sem.wlW * uw;
    if (sem.hasScore) { node.scoreMean += sem.mW * uw; node.scoreMeanSq += sem.mW * sem.mW * uw; }
  }
  function accumulateTerminal(node, sem) {
    /* 终局值确定 → 满权重(search.cpp:1273,useUncertainty 时 = uncertaintyMaxWeight) */
    const uw = UNCERT_MAX_W;
    const uW = utilityWhite(sem);
    const flip = (node.side ^ 1) === WHITE ? 1 : -1;
    node.visits++;
    node.weight += uw; node.weightSq += uw * uw;
    node.util += flip * uW * uw;
    node.utilSq += uW * uW * uw;
    node.wl += flip * sem.wlW * uw;
    node.scoreMean += sem.mW * uw; node.scoreMeanSq += sem.mW * sem.mW * uw;
  }
  function recomputeStats(n) {
    if (n.terminal || !n.nn || !n.children) return;
    /* good 子收集 + 边分摊权重(getChildWeight 口径;searchupdatehelpers.cpp:192) */
    const good = [];
    let currentTotal = 0, thisVisits = 1;                 // 1 = 自身访问
    for (const ch of n.children) {
      const c = ch.node;
      if (!c || c.visits <= 0 || c.weight <= 0 || ch.edgeVisits <= 0) continue;
      const wAdj = c.weight * (ch.edgeVisits / c.visits);
      good.push({ c, wAdj, prior: Math.max(ch.prior, 0) });
      currentTotal += wAdj;
      thisVisits += ch.edgeVisits;
    }
    const origTotal = currentTotal;                       // bias 权重用降权前总量(KataGo origTotalChildWeight)
    if (!good.length) {
      /* 无 good 子:统计回落为自身评估一份(KataGo 空子和循环同款) */
      initLeafStats(n, n.nn);
      n.visits = thisVisits;
      return;
    }
    good.sort((a, b) => b.prior - a.prior);               // KataGo 子按先验序(noise 剪枝假设)
    for (const g of good) {
      const c = g.c, flip = (c.side ^ 1) === WHITE ? 1 : -1;
      g.uW = flip * (c.util / c.weight);
      g.wlW = flip * (c.wl / c.weight);
      g.mW = c.scoreMean / c.weight;
      g.mSqW = c.scoreMeanSq / c.weight;
      g.uSqW = c.utilSq / c.weight;
      g.selfU = n.side === WHITE ? g.uW : -g.uW;          // 选子者(本节点行棋方)视角
    }
    /* pruneNoiseWeight(searchupdatehelpers.cpp:521):劣于前缀均值的子,
     * 超出先验份额 2 倍的部分按效用差距指数削权 */
    {
      let uSum = 0, wSum = 0, pSum = 0;
      for (const g of good) {
        const u = g.selfU, oldW = g.wAdj, p = g.prior;
        let newW = oldW;
        if (wSum > 0 && pSum > 0) {
          const gap = uSum / wSum - u;
          if (gap > 0) {
            const share = wSum * p / pSum;
            if (oldW > 2 * share) newW = oldW - (oldW - 2 * share) * (1 - Math.exp(-gap / NOISE_PRUNE_SCALE));
          }
        }
        uSum += u * newW; wSum += newW; pSum += p;
        g.wAdj = newW;
      }
      currentTotal = wSum;
    }
    /* downweightBadChildrenAndNormalizeWeight(searchupdatehelpers.cpp:428;
     * GTP 无根噪声 → subtract/prune=0,仅 valueWeight t3 降权 + 归一到降权前总量) */
    {
      let simpleSum = 0;
      for (const g of good) simpleSum += g.selfU * g.wAdj;
      const simple = simpleSum / currentTotal;
      let newTotal = 0;
      for (const g of good) {
        const prec = 1.5 * Math.sqrt(g.wAdj);
        const stdev = Math.sqrt(1e-8 + 1 / Math.max(prec, 1e-12));
        g.wAdj *= Math.pow(t3cdf((g.selfU - simple) / stdev) + 1e-4, VALUE_WEIGHT_EXP);
        newTotal += g.wAdj;
      }
      if (newTotal > 0) {
        const f = currentTotal / newTotal;
        for (const g of good) g.wAdj *= f;
      }
    }
    /* 加权求和(子均值,白视角)+ 自身评估一份(searchupdatehelpers.cpp:269);
     * 自身评估先过 subtreeValueBias:表项累计 (子树均值−评估)·origTotal^0.85,
     * 再把评估往表项均值偏移 factor 份(searchupdatehelpers.cpp:287 同款) */
    let uSum = 0, wlSum = 0, mSum = 0, mSqSum = 0, uSqSum = 0, wSqSum = 0;
    for (const g of good) {
      const c = g.c, scaling = g.wAdj / c.weight;
      uSum += g.wAdj * g.uW;
      wlSum += g.wAdj * g.wlW;
      mSum += g.wAdj * g.mW;
      mSqSum += g.wAdj * g.mSqW;
      uSqSum += g.wAdj * g.uSqW;
      wSqSum += scaling * scaling * c.weightSq;
    }
    {
      const sem = n.nn;
      const uw = uncertaintyWeight(sem);
      let uW = utilityWhite(sem);
      if (n.biasKey != null) {
        let entry = biasTable.get(n.biasKey);
        if (!entry) { entry = { d: 0, w: 0 }; biasTable.set(n.biasKey, entry); }
        if (currentTotal > 1e-10) {
          const uChildren = uSum / currentTotal;
          const bw = Math.pow(origTotal, SUBTREE_BIAS_WEXP);
          const delta = (uChildren - uW) * bw;
          entry.d += delta - n.lastBD;
          entry.w += bw - n.lastBW;
          n.lastBD = delta; n.lastBW = bw;
        }
        if (entry.w > 0.001) uW += SUBTREE_BIAS_F * entry.d / entry.w;
      }
      uSum += uW * uw; wlSum += sem.wlW * uw;
      if (sem.hasScore) { mSum += sem.mW * uw; mSqSum += sem.mW * sem.mW * uw; }
      uSqSum += uW * uW * uw;
      wSqSum += uw * uw;
      currentTotal += uw;
    }
    /* 写回(翻转回「走进节点那方」视角) */
    const flip = (n.side ^ 1) === WHITE ? 1 : -1;
    n.visits = thisVisits;
    n.weight = currentTotal;
    n.util = flip * uSum;
    n.utilSq = uSqSum;
    n.wl = flip * wlSum;
    n.scoreMean = mSum;
    n.scoreMeanSq = mSqSum;
    n.weightSq = wSqSum;
  }
  const finishNow = (e) => {
    accumulateTerminal(e.node, e.node.terminalSem);
    for (let i = e.path.length - 2; i >= 0; i--) recomputeStats(e.path[i].node);
    for (let i = e.path.length - 1; i >= 1; i--) unmake(bd, e.path[i].move, e.path[i].tok);
  };

  /* ---- 展开:着法+先验条目(节点本体惰性建) ---- */
  function expand(node, policy, policyPass, leafBd, legalMoves) {
    const stm = node.side;
    const moves = [], logits = [];
    for (const mv of legalMoves) {
      if (isOwnTrueEye(leafBd, stm, mv)) continue;
      moves.push(mv); logits.push(policy[mv]);
    }
    const passIdx = moves.length;
    moves.push(PASS); logits.push(policyPass);
    let mx = -Infinity;
    for (const l of logits) if (l > mx) mx = l;
    let sum = 0;
    const probs = new Array(logits.length);
    for (let i = 0; i < logits.length; i++) { const e = Math.exp(logits[i] - mx); probs[i] = e; sum += e; }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    /* pass 先验不压制:面积计分口径(KataGo shouldSuppressPass 仅数目法生效,
     * 见文件头 PASS_SUPPRESS 注);何时停一手交给网络值 + 双停终局值 */
    let keepSum = 0;
    const keep = [];
    for (let i = 0; i < moves.length; i++) {
      if (i !== passIdx && probs[i] < PRIOR_FLOOR) continue;
      keep.push(i); keepSum += probs[i];
    }
    node.children = new Array(keep.length);
    for (let k = 0; k < keep.length; k++) {
      const i = keep[k];
      node.children[k] = { move: moves[i], prior: probs[i] / keepSum, node: null, edgeVisits: 0 };
    }
  }

  /* 树复用后的根先验刷新(同 v3) */
  function refreshRootPriors(policy, policyPass, legalMoves) {
    const chs = root.children;
    const idxOf = new Map(chs.map((ch, i) => [ch.move, i]));
    const mvs = [], logits = [], slot = [];
    for (const mv of legalMoves) {
      const i = idxOf.get(mv);
      if (i === undefined) continue;
      mvs.push(mv); logits.push(policy[mv]); slot.push(i);
    }
    if (idxOf.has(PASS)) { mvs.push(PASS); logits.push(policyPass); slot.push(idxOf.get(PASS)); }
    let mx = -Infinity;
    for (const l of logits) if (l > mx) mx = l;
    let sum = 0;
    const probs = logits.map((l) => { const e = Math.exp(l - mx); sum += e; return e; });
    let keepSum = 0;
    for (let k = 0; k < probs.length; k++) {
      if (mvs[k] !== PASS && probs[k] / sum < PRIOR_FLOOR) probs[k] = 0;
      keepSum += probs[k];
    }
    for (let k = 0; k < probs.length; k++) {
      chs[slot[k]].prior = probs[k] > 0 ? probs[k] / keepSum : 1e-6;
    }
  }

  /* ---- FPU(searchexplorehelpers.cpp 同款) ---- */
  function parentStdev(node) {
    const acc = node.weight > 0 ? node.util / node.weight : 0;
    if (node.visits <= 0 || node.weight <= 1) return STDEV_PRIOR;
    const u2 = acc * acc;
    const uSqAvg = Math.max(node.utilSq / node.weight, u2);
    return Math.sqrt(Math.max(0,
      ((u2 + STDEV_PRIOR * STDEV_PRIOR) * STDEV_PRIOR_W + uSqAvg * node.weight)
      / (STDEV_PRIOR_W + node.weight - 1) - u2));
  }
  function fpuForChildren(node, policyMass, isRoot) {
    const acc = node.weight > 0 ? node.util / node.weight : 0;
    const accW = ((node.side ^ 1) === WHITE) ? acc : -acc;
    let blendW = accW;
    if (node.nn) {
      const avgW = Math.min(1, Math.pow(Math.max(policyMass, 0), FPU_BLEND_POW));
      blendW = avgW * accW + (1 - avgW) * utilityWhite(node.nn);
    }
    const red = (isRoot ? ROOT_FPU : FPU) * Math.sqrt(Math.max(policyMass, 0));
    return (node.side === WHITE ? blendW : -blendW) - red;
  }

  /* ---- PUCT 下降:惰性建子 + 转置接入 + 在途跳过 ---- */
  let rootRing = null;
  let pathStamp = 0;
  function descend() {
    ringRestore(rootRing);
    let cur = root;
    const path = [{ node: root, move: -1, tok: -1 }];
    const recent = rootRecent.slice();
    pathStamp++;
    cur._ps = pathStamp;
    while (!cur.terminal && cur.nn && cur.children) {
      let totalW = 0, pMass = 0;
      for (const ch of cur.children) {
        if (!ch.node) continue;                   // KataGo policyProbMassVisited:仅已访问子
        /* totalChildWeight 同口径:累加边分摊权重(getChildWeight),非子节点全局
         * 权重 —— 转置共享子不得重复计入本父的探索预算(searchexplorehelpers
         * 累加的就是 getChildWeight(edgeVisits)) */
        totalW += ch.node.weight * (ch.edgeVisits / Math.max(ch.node.visits, 1));
        pMass += Math.max(ch.prior, 0);
      }
      const cpuct = CPUCT + CPUCT_LOG * Math.log((totalW + CPUCT_BASE) / CPUCT_BASE);
      const stdevFactor = 1 + STDEV_SCALE * (parentStdev(cur) / STDEV_PRIOR - 1);
      const explore = cpuct * Math.sqrt(totalW + PUCT_OFFSET) * stdevFactor;
      const fpu = fpuForChildren(cur, pMass, cur === root);
      let best = null, bestV = -Infinity;
      for (const ch of cur.children) {
        if (ch.prior <= 0) continue;                          // 对称剪枝置零
        const n = ch.node;
        if (n && (n._ps === pathStamp || n.evalPending)) continue;   // 在途/评估中:跳过(循环守卫)
        let q, cw;
        if (!n) { q = fpu; cw = 0; }                          // 未走过的边:FPU
        else {
          q = (n.visits > 0 && n.weight > 0) ? n.util / n.weight : fpu;
          /* 边访问缩放:该边分摊的权重(getChildWeight 口径) */
          cw = n.weight * (ch.edgeVisits / Math.max(n.visits, 1));
          /* 虚拟损失:效用向「对行棋方最坏」的 −utilityRadius 混合 + 分母膨胀
           * (getExploreSelectionValueOfChild,searchexplorehelpers.cpp:141:
           * KataGo 在白视角空间取 pla==WHITE?−R:+R,即对任何行棋方都是坏向;
           * 本引擎 q 为行棋方视角,坏向恒为 −R,与颜色无关。
           * ★ 2026-10-06 修复:旧版误写 (side===WHITE)?−R:+R —— 黑方行棋
           * 节点 vl 把子拉向好方向,批内同伴挤向同子 → evalPending 死端 →
           * 半批发射 + accumulateSelfEval 统计污染;批 1 时 vl 恒 0,bug 不可
           * 见。这是批 4 游戏级放血(T4 vs T1 自对弈 1-11、vs KataGo 3-33,
           * 而单局面分布对拍正常)的根因) */
          if (n.vl > 0) {
            const vlU = -UTILITY_RADIUS;
            const frac = n.vl / (n.vl + Math.max(0.25, cw));
            q = q + (vlU - q) * frac;
            cw += n.vl;
          }
        }
        const v = q + explore * ch.prior / (1 + cw);
        if (v > bestV) { bestV = v; best = ch; }
      }
      if (!best) break;
      const isFreshEdge = best.node === null;
      let node;
      if (isFreshEdge) {
        /* bias 签名须用落子前盘面(search.cpp:977 getRecentBoard(1) 语义) */
        const biasK = biasKeyOf(bd, cur.side, cur.move, best.move, koPoint());
        const tok = make(bd, best.move, cur.side);            // 活状态 → 子位置
        const trailing = computeTrailing(recent.concat(best.move));
        const { key } = childChainKey(cur, best.move, bd, cur.side ^ 1, trailing);
        node = table.get(key) ?? null;
        if (node && (node.evalPending || !node.nn)) {
          /* 同批已有在途评估:本边暂不可下 —— 回退并当死端(不计边访问,防同节点双评估) */
          unmake(bd, best.move, tok);
          break;
        }
        if (!node) {
          node = makeNode(best.move, cur.side ^ 1, key);
          node.biasKey = biasK;
          /* 双停终局:建表时判(活盘面即终局盘面;节点按位置身份,转置命中同键必然同终局) */
          if (cur.move === PASS && best.move === PASS) {
            node.terminal = true;
            const s = scoreGame(bd, komi);
            node.terminalSem = {
              wlW: s.margin > 0 ? -1 : s.margin < 0 ? 1 : 0,
              mW: -s.margin, sdW: 0, stWL: 0, stScore: 0, hasScore: true,
            };
          }
          /* 终局无评估,不挂在途标:evalPending 的子会被选点跳过,挂着等于
           * 双停终局值只在创建那次 playout 生效,之后永远无法再访问 */
          node.evalPending = !node.terminal;
          table.set(key, node);
        }
        best.node = node;                                     // 转置接入或新建
        /* 转置边访问追平(KataGo maybeCatchUpEdgeVisits,search.cpp:1552;
         * GTP 配置 graphSearchCatchUpLeakProb=0 即总是追平):共享子节点的
         * 总访问超过本边记录时,本次 playout 只补 1 个边访问即折返,
         * 主循环侧随后重算本父节点统计(边访问即信息,重算式统计下语义同源) */
        if (ENABLE_CATCHUP && node.nn && node.visits > best.edgeVisits) {
          unmake(bd, best.move, tok);         // 本层已 make,折返前必须退回
          best.edgeVisits++;
          return { node: cur, path, recent, catchUpChild: node };
        }
        best.edgeVisits++;
        node._ps = pathStamp;
        path.push({ node, move: best.move, tok });
        recent.push(best.move);
        cur = node;
        if (node.terminal) break;                             // 双停终局(转置命中同键必然同终局)
        if (node.nn) continue;                                // 转置命中既有子树:继续下潜
        break;                                                // 新叶:待评估
      } else {
        /* 既有边同理:边访问落后共享子节点总访问时,补 1 折返不下降 */
        if (ENABLE_CATCHUP && best.node.nn && best.node.visits > best.edgeVisits) {
          best.edgeVisits++;
          return { node: cur, path, recent, catchUpChild: best.node };
        }
        const tok = make(bd, best.move, cur.side);
        best.edgeVisits++;
        node = best.node;
        node._ps = pathStamp;
        path.push({ node, move: best.move, tok });
        recent.push(best.move);
        cur = node;
        if (node.terminal) break;
        if (!node.nn) break;                                  // 评估在途(罕见):当叶处理
      }
    }
    return { node: cur, path, recent };
  }

  /* ---- 主循环前置:根评估(λ=0.2)→ recentScoreCenter + 根先验 + 环快照 ---- */
  let rootOwnPre = null;                     // 根评估 ownership(行棋方视角 pretanh),终选 pass 守门用
  /* 对称默认开(2026-10-07 根因修复后恢复,对齐 KataGo nnRandomize=true /
   * nneval.cpp 每评估随机 8 对称之一):此前的「实战崩坏」根因是对战
   * harness 的行序列化丢 sym 字段 —— 服务端收不到 sym 按恒等特征评估,
   * 引擎按契约反置换,策略/ownership 被随机旋转打乱(值输出为标量不受损,
   * 故一切基于胜率的探针全绿;运行时自检实锤:同请求内 sym 行与 sym=0
   * 影子行 wlΔ 精确为 0 = 两行喂了逐位相同的输入,policy L1 高达 244-995、
   * argmax 分歧 63/65)。修 harness 序列化(补 sym 字段)后:L1 降至
   * 137-213(网络真实非等变噪声,即对称的去相关收益),H2H sym-on 由
   * 0-6@73手 变 4-2 正常局长。引擎机制本身自始至终正确。 */
  const useSym = opt.symmetry !== false;
  if (opt.rngSeed !== undefined) symRng = (opt.rngSeed >>> 0) || 1;
  {
    const f = encodeFeatures(bd, side, { recentMoves: rootRecent, komi, outSpatial: spBuf, outGlobal: glBuf });
    const rootSym = useSym ? nextSym() : 0;
    const [out0] = await evalBatch([{ spatial: f.spatial, global: f.global, sym: rootSym, optimism: ROOT_OPTIMISM }]);
    const out = rootSym ? unpermuteOut(out0, SYM8[rootSym]) : out0;
    rootOwnPre = out.ownership ?? null;
    const sem = semOf(out, side);
    const expectedScore = (root.visits > 0 && root.weight > 0)
      ? root.scoreMean / root.weight : sem.mW;
    recentScoreCenter = expectedScore * (1 - CENTER_ZERO_W);
    const cap = SQRT_AREA * CENTER_SCALE;
    if (recentScoreCenter > expectedScore + cap) recentScoreCenter = expectedScore + cap;
    if (recentScoreCenter < expectedScore - cap) recentScoreCenter = expectedScore - cap;
    if (!root.nn) {
      root.nn = sem;
      expand(root, out.policy, out.policyPass, bd, genLegal(bd, side));
      initLeafStats(root, sem);                 // 根统计 = 自身评估一份(KataGo initNodeNNOutput)
    } else {
      refreshRootPriors(out.policy, out.policyPass, genLegal(bd, side));
    }
    rootRing = ringSnapshot();
  }

  /* ---- 根对称剪枝(保守式) ---- */
  {
    const banMask = new Uint8Array(N2);
    superkoBannedPoints(bd, side, banMask);
    let banned = false;
    for (let p = 0; p < N2; p++) if (banMask[p]) { banned = true; break; }
    if (!banned) pruneSymmetricRootMoves(root.children, bd);
  }

  let iters = 0, nnCalls = 0;

  /* 预算 1 = 模型直出:上方根评估就是那 1 访,主循环直接跳过 —— 选点 =
   * 原始策略 argmax(directPolicyMove,树复用带进的旧子树统计一律不看),
   * 胜率 / 目差 = 根直出值(尾段 rootQ / scoreLead 按 root.nn 口径)。 */
  if (budget <= 1) iters = 1;

  /* ---- 批回传:沿 path 重 make 到叶再展开(展开时盘面在叶上 —— C++
   * runSinglePlayout 全程 board 在叶、playout 末尾才复位,search.cpp:1263/1331)。
   * remake 的 token 与攒批侧存的不同,unmake 必须配对新 token。 ---- */
  const remakePath = (path) => {
    const toks = new Array(path.length - 1);
    for (let k = 1; k < path.length; k++) {
      toks[k - 1] = make(bd, path[k].move, path[k - 1].node.side);
    }
    return toks;
  };

  /* ---- 批回传 + 进度上报(单槽管线保证严格按批序调用) ----
   * v4.2:撤虚拟损失计数 → 展开叶 → 叶统计 = 自身评估一份 → 自叶向根
   * 逐节点重算(updateStatsAfterPlayout 的单线程投影)。 ---- */
  const applyBatch = (pending, outs) => {
    nnCalls++;
    for (let i = 0; i < pending.length; i++) {
      const p = pending[i];
      let out = outs[i];
      if (p.sym) out = unpermuteOut(out, SYM8[p.sym]);   // 逆置换回恒等坐标系
      removeVirtualLoss(p.path);
      const toks = remakePath(p.path);
      expand(p.node, out.policy, out.policyPass, bd, p.legalMoves);
      for (let k = p.path.length - 1; k >= 1; k--) unmake(bd, p.path[k].move, toks[k - 1]);
      const sem = semOf(out, p.node.side);
      p.node.nn = sem;
      p.node.evalPending = false;
      initLeafStats(p.node, sem);
      for (let k = p.path.length - 2; k >= 0; k--) recomputeStats(p.path[k].node);
      storeEval(p.key, out);                  // λ=1 口径原样入缓存(root 不走缓存)
      iters++;
    }
    if (opt.onProgress && (iters & 63) === 0) {
      const best = pickBest(root);
      const q = best && best.node.weight > 0 ? best.node.util / best.node.weight : 0;
      opt.onProgress({
        visits: iters, move: best ? best.move : PASS,
        winRate: (q + 1) / 2,
        scoreLead: root.nn && root.nn.hasScore ? (side === WHITE ? root.nn.mW : -root.nn.mW) : null,
        ms: Date.now() - t0,
      });
    }
  };

  /* ---- 虚拟多线程管线(KataGo numSearchThreads=T 的异步单线程投影,v4.5)----
   * T = maxBatch:总在途(在飞批 + 待发队列)≤ T,下降的统计盲区 ≤ T−1 ——
   * KataGo 线程「停在叶上等自己的评估,下降时最多盲于其它 T−1 个在途」的
   * 精确等价;批序严格(先收上一批再发射),evalPending 使同批自动避开同节点;
   * 死端/预算边界立即发射手头半批(机会主义,不凑批)。旧 v4.1 单槽管线的
   * minBatch 空闲凑满 + 在途时 pending<maxBatch 不计在途量,盲区最深 2T−1
   * —— 这正是同代码批 1 对批 4 自对弈 10-2 的差距来源(2026-10-06 实测),
   * 批 4 口径 vs KataGo 1-5、批 1 口径 3-3。 ---- */
  let inflight = null;                        // { promise, pending }
  const settleInflight = async () => {
    if (!inflight) return;
    applyBatch(inflight.pending, await inflight.promise);
    inflight = null;
  };
  while (iters < budget) {
    const pending = [];
    while ((inflight ? inflight.pending.length : 0) + pending.length < maxBatch
      && iters + (inflight ? inflight.pending.length : 0) + pending.length < budget) {
    const e = descend();
    if (e.catchUpChild) {
      /* 追平折返(KataGo maybeCatchUpEdgeVisits → updateStatsAfterPlayout(父),
       * search.cpp:1468):边访问已在下降时计入,此处仅重算父统计,计一次访问,
       * 无推理。旧版「合成回传」在重算式统计下无意义(会被重算覆盖),删除。 */
      recomputeStats(e.path[e.path.length - 1].node);
      for (let i = e.path.length - 1; i >= 1; i--) unmake(bd, e.path[i].move, e.path[i].tok);
      iters++;
      continue;
    }
    if (e.node.terminal) {
        finishNow(e);
        iters++;
        continue;
      }
      if (e.node.nn) {                                        // 死端(全子被跳过:在途/转置回边)
        /* KataGo search.cpp:1408:全子被禁时自身评估再计一份,祖先照常重算 */
        accumulateSelfEval(e.node);
        for (let i = e.path.length - 2; i >= 0; i--) recomputeStats(e.path[i].node);
        for (let i = e.path.length - 1; i >= 1; i--) unmake(bd, e.path[i].move, e.path[i].tok);
        if (pending.length > 0) break;                        // 已攒半批:立即结算
        if (inflight) { await settleInflight(); continue; }   // 在途饱和:收批解饱和,不烧预算
        iters++;                                              // 真死端:计一访(KataGo 循环访问同款)
        continue;
      }
      /* 探测(编码+算键)一律用共享 scratch:缓存命中的编码不进环 —— 否则
       * 热缓存下 collect burst 的命中风暴会烧穿环形容量,回卷覆盖在飞批的
       * 特征缓冲(在飞批收到垃圾特征,评估全毁,2026-10-07 实战 73 手级崩盘
       * 的根因)。未命中才拷入专属环槽,pending 零拷贝引用。 */
      const f = encodeFeatures(bd, e.node.side, {
        recentMoves: e.recent, komi, outSpatial: spBuf, outGlobal: glBuf,
      });
      /* 随机对称:sym 随行下发(引擎侧置换);键取「原始特征 + sym」 */
      const sym = useSym ? nextSym() : 0;
      const key = fevalKey(f.spatial, f.global, sym);
      const cached = lookupEval(key);
      if (cached) {
        cacheHits++;
        const sem = semOf(cached, e.node.side);
        expand(e.node, cached.policy, cached.policyPass, bd, genLegal(bd, e.node.side));
        e.node.nn = sem;
        e.node.evalPending = false;
        initLeafStats(e.node, sem);
        for (let k = e.path.length - 2; k >= 0; k--) recomputeStats(e.path[k].node);
        for (let i = e.path.length - 1; i >= 1; i--) unmake(bd, e.path[i].move, e.path[i].tok);
        iters++;
        continue;
      }
      const slot = ringNext();
      spRing[slot].set(f.spatial); glRing[slot].set(f.global);
      pending.push({
        node: e.node, path: e.path, key, sym,
        legalMoves: genLegal(bd, e.node.side),
        spatial: spRing[slot], global: glRing[slot],  // 未命中:拷入专属环槽,零拷贝引用
      });
      applyVirtualLoss(e.path);
      for (let i = e.path.length - 1; i >= 1; i--) unmake(bd, e.path[i].move, e.path[i].tok);
    }

    await settleInflight();                   // 先收上一批,再发射攒下的(批序严格)
    if (pending.length === 0) continue;       // 预算已尽(或死端结算完):外层条件收口
    inflight = {
      promise: evalBatch(pending.map((p) => ({ spatial: p.spatial, global: p.global, sym: p.sym }))),
      pending,
    };
  }
  await settleInflight();                     // 收尾:末批须在选点前回传

  /* ==================== 最终选点(edge 缩放权重;精修同 v3) ==================== */
  const tEff = effectiveTemperature(temperature, temperatureHalflife, rootRecent.length);
  /* 根终选 pass 守门(KataGo shouldSuppressPass,searchhelpers.cpp:443 语义;
   * KataGo 仅数目法启用,本引擎面积计分下作为产品护栏常开 —— 学生网 pass
   * 价值未标定,2026-10-05 A/B:纯靠网络值会提前停一手,75 手即认输级崩盘)。
   * 判据:存在「非对方铁地深处、访问充分、效用/目差不比 pass 差太多」的
   * 盘上着法 → pass 的选择权重清零。 */
  const suppressPass = rootOwnPre && rootShouldKeepFilling(root, rootOwnPre);
  const move = budget <= 1
    ? directPolicyMove(root, tEff, suppressPass)
    : chooseFinalMove(root, tEff, suppressPass);
  const best = pickBest(root);
  /* 根加权胜率:全部子按边分摊权重的效用均值 —— 比旧口径(最佳子 q)
   * 少一层选点乐观偏差,认输判据与 UI 胜率据此不再系统性虚高
   * (2026-10-05 诊断:落后 35 目时旧口径仍报 ~50%)。
   * 直出:根自身评估即胜率(mover 视角;子树统计可能是树复用的旧账)。 */
  let wSum = 0, uSum = 0;
  if (budget > 1) {
    for (const ch of root.children ?? []) {
      const n = ch.node;
      if (n && n.weight > 0 && n.visits > 0) {
        const w = n.weight * (ch.edgeVisits / n.visits);
        wSum += w; uSum += w * (n.util / n.weight);
      }
    }
  }
  const rootQ = budget <= 1 ? (root.nn ? (side === WHITE ? root.nn.wlW : -root.nn.wlW) : 0)
    : (wSum > 0 ? uSum / wSum
    : (best && best.node.weight > 0 ? best.node.util / best.node.weight : 0));
  const winRate = (rootQ + 1) / 2;

  /* ---- GC:从根标记可达,清扫节点表(KataGo mark-and-sweep) ---- */
  if (opt.reuseTree !== false) {
    gcEpoch++;
    const stack = [root];
    root._gc = gcEpoch;
    while (stack.length) {
      const n = stack.pop();
      for (const ch of n.children ?? []) {
        if (ch.node && ch.node._gc !== gcEpoch) { ch.node._gc = gcEpoch; stack.push(ch.node); }
      }
    }
    for (const [k, n] of table) if (n._gc !== gcEpoch) {
      /* subtreeValueBias:删除节点回退 80% 贡献(removeSubtreeValueBias,
       * freeProp 语义:20% 沉淀为历史证据) */
      if (n.biasKey != null && (n.lastBD !== 0 || n.lastBW !== 0)) {
        const e = biasTable.get(n.biasKey);
        if (e) { e.d -= n.lastBD * SUBTREE_BIAS_FREEPROP; e.w -= n.lastBW * SUBTREE_BIAS_FREEPROP; }
      }
      table.delete(k);
    }
    if (table !== nodeTable) { /* 局部表(reuseTree=false):随搜索结束丢弃 */ }
  }

  treeKeep = { root, recentMoves: rootRecent.slice() };
  return {
    move, winRate, visits: iters, nodes: iters, ms: Date.now() - t0,
    scoreLead: root.nn && root.nn.hasScore ? (side === WHITE ? root.nn.mW : -root.nn.mW) : null,
    only: !root.children || root.children.filter((c) => (c.prior ?? 0) > 0).length <= 1,
    reused: Math.max(reused, 0), nnCalls, cacheHits, temperature: tEff,
    rootChildren: opt.debug ? (root.children?.length ?? 0) : undefined,
    rootChildMoves: opt.debug ? (root.children ?? []).map((c) => c.move) : undefined,
    rootChildStats: opt.debug ? (root.children ?? []).map((c) => ({
      m: c.move, v: c.edgeVisits,
      q: c.node && c.node.weight ? +(c.node.util / c.node.weight).toFixed(3) : 0,
    })) : undefined,
  };
}

/* ==================== 根对称剪枝(保守式) ==================== */
/* SYM8 已抽至 ./symmetry.js(自研 WebGPU 引擎与搜索共用同一份定义) */

function pruneSymmetricRootMoves(children, bd) {
  if (!children || children.length < 2) return;
  for (let s = 1; s < 8; s++) {
    const map = SYM8[s];
    let sym = true;
    for (let p = 0; p < N2; p++) {
      if (bd[p] !== bd[map[p]]) { sym = false; break; }
    }
    if (!sym) continue;
    const order = children.slice().sort((a, b) => (b.prior ?? 0) - (a.prior ?? 0));
    const killed = new Set();
    for (const ch of order) {
      const mv = ch.move;
      if (mv === PASS || killed.has(mv) || (ch.prior ?? 0) <= 0) continue;
      const symMv = map[mv];
      if (symMv !== mv && !killed.has(symMv)) killed.add(symMv);
    }
    if (killed.size) {
      for (const ch of children) {
        if (ch.move !== PASS && killed.has(ch.move)) ch.prior = 0;
      }
    }
  }
}

/* ==================== 最终选点:noisePruning → valueWeight → LCB → 温度 ==================== */
/* shouldSuppressPass 移植:有值得下的盘上着法则压 pass(调用点见 nnSearchBest 终选)。
 * ownPre:根评估 ownership(行棋方视角 pretanh);q 口径为「走进子那方」= 根行棋方,
 * 越大越好不分颜色;scoreMean 恒白视角,按行棋方镜像比较(KataGo 同款双门限)。 */
function rootShouldKeepFilling(root, ownPre) {
  const chs = root.children ?? [];
  let passCh = null;
  for (const ch of chs) if (ch.move === PASS && ch.node) { passCh = ch; break; }
  if (!passCh || passCh.node.weight <= 1e-10) return false;
  const passW = passCh.node.weight * (passCh.edgeVisits / Math.max(passCh.node.visits, 1));
  const passQ = passCh.node.util / passCh.node.weight;
  const passM = passCh.node.scoreMean / passCh.node.weight;
  const EXTREME = 0.95;
  const rOf = (p) => (p / N) | 0, cOf = (p) => p % N;
  for (const ch of chs) {
    if (ch.move === PASS || !ch.node || ch.node.weight <= 1e-10) continue;
    const mv = ch.move;
    const own = Math.tanh(ownPre[mv]);
    if (own < -EXTREME) {
      /* 对方铁地:四邻无一为我方铁地 → 深处死点,不算候选 */
      const r = rOf(mv), c = cOf(mv);
      let adjMine = false;
      if (r > 0 && Math.tanh(ownPre[mv - N]) > EXTREME) adjMine = true;
      if (r < N - 1 && Math.tanh(ownPre[mv + N]) > EXTREME) adjMine = true;
      if (c > 0 && Math.tanh(ownPre[mv - 1]) > EXTREME) adjMine = true;
      if (c < N - 1 && Math.tanh(ownPre[mv + 1]) > EXTREME) adjMine = true;
      if (!adjMine) continue;
    }
    const w = ch.node.weight * (ch.edgeVisits / Math.max(ch.node.visits, 1));
    if (ch.edgeVisits <= 500 && w <= 2 * Math.sqrt(passW)) continue;   // 访问不足:不作为证据
    const q = ch.node.util / ch.node.weight;
    const mW = ch.node.scoreMean / ch.node.weight;
    /* q 为行棋方视角(大=好,不分颜色);mW 白视角按行棋方镜像 */
    if (q > passQ - 0.1
      && (root.side === WHITE ? mW > passM - 0.5 : mW < passM + 0.5)) return true;
  }
  return false;
}

/* 直出(预算 1):选点 = 原始策略 argmax —— 树复用带进的旧子树统计一律不参与,
 * pass 守门照常(产品护栏);温度 > 0 时按 prior^(1/T) 抽样(pickMove 同款对数平移)。 */
function directPolicyMove(root, tEff, suppressPass = false) {
  const live = (root.children ?? []).filter((ch) =>
    (ch.prior ?? 0) > 0 && !(suppressPass && ch.move === PASS));
  if (!live.length) return PASS;
  let arg = 0;
  for (let i = 1; i < live.length; i++) if ((live[i].prior ?? 0) > (live[arg].prior ?? 0)) arg = i;
  if (tEff > 1e-4) {
    const logMax = Math.log(Math.max(live[arg].prior ?? 0, 1e-30));
    let sum = 0;
    const wts = live.map((ch) => {
      const x = Math.exp((Math.log(Math.max(ch.prior ?? 0, 1e-30)) - logMax) / tEff);
      sum += x; return x;
    });
    if (sum > 0) {
      let r = Math.random() * sum;
      for (let i = 0; i < live.length; i++) {
        r -= wts[i];
        if (r < 0) { arg = i; break; }
      }
    }
  }
  return live[arg].move;
}

function chooseFinalMove(root, tEff, suppressPass = false) {
  const live = (root.children ?? []).filter((ch) =>
    (ch.edgeVisits > 0 || (ch.prior ?? 0) > 0) && !(suppressPass && ch.move === PASS));
  if (!live.length) return PASS;
  const scored = !!root.nn && root.nn.hasScore;

  /* 权重起点:边缩放权重(getChildWeight 口径)+ 效用取节点全局均值 */
  const W = live.map((ch) => {
    const n = ch.node;
    if (!n || n.weight <= 0 || n.visits <= 0) return 0;
    return n.weight * (ch.edgeVisits / Math.max(n.visits, 1));
  });
  const U = live.map((ch) => {
    const n = ch.node;
    return n && n.weight > 0 ? n.util / n.weight : 0;
  });

  if (scored && live.length > 1) {
    const order = live.map((_, i) => i).sort((a, b) => (live[b].prior ?? 0) - (live[a].prior ?? 0));
    let uSum = 0, wSum = 0, pSum = 0;
    for (const i of order) {
      const u = U[i], wOld = W[i], p = Math.max(live[i].prior ?? 0, 1e-30);
      if (wSum > 0 && pSum > 0) {
        const gap = uSum / wSum - u;
        if (gap > 0) {
          const share = wSum * p / pSum;
          if (wOld > 2 * share) {
            W[i] = wOld - (wOld - 2 * share) * (1 - Math.exp(-gap / NOISE_PRUNE_SCALE));
          }
        }
      }
      uSum += U[i] * W[i]; wSum += W[i]; pSum += p;
    }
  }

  if (scored && VALUE_WEIGHT_EXP > 0 && live.length > 1) {
    const stdevs = new Array(live.length);
    let total = 0, simpleSum = 0;
    for (let i = 0; i < live.length; i++) {
      total += W[i];
      const prec = 1.5 * Math.sqrt(W[i]);
      simpleSum += U[i] * W[i];
      stdevs[i] = Math.sqrt(1e-8 + 1 / Math.max(prec, 1e-12));
    }
    if (total > 0) {
      const simple = simpleSum / total;
      let newTotal = 0;
      for (let i = 0; i < live.length; i++) {
        if (W[i] <= 0) continue;
        const p = t3cdf((U[i] - simple) / stdevs[i]) + 1e-4;
        W[i] *= Math.pow(p, VALUE_WEIGHT_EXP);
        newTotal += W[i];
      }
      if (newTotal > 0) {
        const f = total / newTotal;
        for (let i = 0; i < live.length; i++) W[i] *= f;
      }
    }
  }

  {
    const radius = new Array(live.length).fill(0);
    const lcb = new Array(live.length).fill(-Infinity);
    let refW = 0, refG = -Infinity, bestIdx = -1;
    for (let i = 0; i < live.length; i++) {
      const n = live[i].node;
      const g = W[i] * Math.max(0, (n ? n.visits : 0) - 1) / Math.max(1, n ? n.visits : 1) + 2 * (live[i].prior ?? 0);
      if (g > refG) { refG = g; refW = W[i]; }
    }
    for (let i = 0; i < live.length; i++) {
      const n = live[i].node;
      if (!n || n.weight <= 0) { radius[i] = 2 * UTILITY_RADIUS * LCB_STDEVS; lcb[i] = -Infinity; continue; }
      const w = n.weight, wsq = n.weightSq;
      const uAvg = n.util / w;
      const ess0 = w * w / Math.max(wsq, 1e-300);
      const priorW = w / Math.max(ess0 * ess0 * ess0, 1e-300);
      let uSqAvg = Math.max(n.utilSq / w, uAvg * uAvg + 1e-8);
      uSqAvg = (uSqAvg * w + (uSqAvg + UTILITY_RADIUS * UTILITY_RADIUS) * priorW) / (w + priorW);
      const wAdj = w + priorW, wsqAdj = wsq + priorW * priorW;
      const ess = wAdj * wAdj / wsqAdj;
      const variance = Math.max(uSqAvg - uAvg * uAvg, 0);
      radius[i] = LCB_STDEVS * Math.sqrt(variance / Math.max(ess, 1e-300));
      lcb[i] = uAvg - radius[i];
      if (W[i] >= LCB_MIN_PROP * refW && (bestIdx < 0 || lcb[i] > lcb[bestIdx])) bestIdx = i;
    }
    if (bestIdx >= 0) {
      let adjusted = W[bestIdx];
      for (let i = 0; i < live.length; i++) {
        if (i === bestIdx) continue;
        const excess = lcb[bestIdx] - lcb[i];
        if (excess <= 0) continue;
        const rf = (radius[i] + excess) / (radius[i] + 0.2 * excess);
        const lb = rf * rf * W[i];
        if (lb > adjusted) adjusted = lb;
      }
      W[bestIdx] = adjusted;
    }
  }

  let arg = 0;
  for (let i = 1; i < live.length; i++) if (W[i] > W[arg]) arg = i;
  if (tEff > 1e-4 && W[arg] > 0) {
    const logMax = Math.log(W[arg]);
    let sum = 0;
    const wts = W.map((w) => {
      if (w <= 0) return 0;
      const x = Math.exp((Math.log(w) - logMax) / tEff);
      sum += x;
      return x;
    });
    if (sum > 0) {
      let r = Math.random() * sum;
      for (let i = 0; i < live.length; i++) {
        r -= wts[i];
        if (r < 0) { arg = i; break; }
      }
    }
  }
  return live[arg].move;
}
