/* ============================================================
 * AetherGo NN 特征编码器 —— KataGo v17 输入特征(fillRowV7)的 JS 对齐版
 *
 * 权威源:KataGo/cpp/neuralnet/nninputs.cpp 的 fillRowV7,及它调用的
 *   Board::calculateArea(board.cpp,Benson 领土)、searchIsLadderCaptured
 *   (征子搜索,已拆至 ./ladder.js)、BoardHistory::passWouldEndPhase。
 * 逐行语义移植,不是「差不多」重写 —— 这是全路线质量的锚:
 * 与 katago selfplay 训练行(npz 的 binaryInputNCHW / globalInputNC)
 * 逐位对拍通过才算数(对拍驱动:test/featdiff.mjs)。
 *
 * 固定口径(与训练数据严格一致;范围外特征恒 0):
 *   中国规则面积计分、position superko(KO_POSITIONAL)、禁自杀、
 *   无让子、无 button、无 encore(通道 7/8/12/13/20/21、全局 8~11 恒 0)、
 *   friendlyPassOk = false —— katago selfplay 的 GameInitializer 用 Rules
 *   默认构造(GameInitializer::createRulesUnsynchronized),该值为 false,
 *   所以「上一手是停着」**不**隐藏历史,只有全局 14 = 1。
 *   (若将来换 friendlyPassOk=true 的数据源,须同步改这里与对拍。)
 *
 * 前两手盘面(通道 15/16,征子):**滚动盘面环**(rules.js recentPrevBd/Ko,
 * KataGo BoardHistory.recentBoards 同款)—— make 增量维护,O(1) 取,编码
 * 成本 O(盘面) 不随手数涨。调用约定:bd 必须处于「经 make/replayMoves 到达」
 * 的状态(引擎一致性约定)。参考实现 encodeFeaturesReplay 从空盘重演,
 * 供 features-test 的逐位对拍锚定 —— 环实现有任何回归它先红。
 *
 * 输出:spatial = Float32Array(22*N2) NCHW(索引 c*N2+p,p = 行*19+列),
 *      global = Float32Array(19);可直接拼进 ONNX 的 (N,22,19,19)/(N,19,1,1)。
 *
 * 注意:recentMoves 必须包含**完整**的尾随停着串(窗口不得切进停着串
 * 中间),否则历史抑制判定会错。
 * ============================================================ */
import { N, N2, EMPTY, BLACK, WHITE, PASS, KOMI, koPoint, superkoBannedPoints,
         recentPrevBd, recentPrevKo, moveCount } from '../engine.js';
import { NB_OFF, adj, chainFlood, F_STONES, F_LIBS } from './flood.js';
import { iterLadders } from './ladder.js';

export const SPATIAL_CHANNELS = 22;
export const GLOBAL_CHANNELS = 19;

const idx = (c, p) => c * N2 + p;

/* ==================== Benson 领土(Board::calculateArea 移植) ==================== */

/* calculateAreaForPla(nonPassAliveStones/safeBig/unsafeBig 全 true、禁自杀) */
function areaForPla(bd, v, result) {
  /* 非 v 点连通区域 */
  const regionIdx = new Int32Array(N2).fill(-1);
  const regions = [];
  const headCache = new Int32Array(N2).fill(-1);          // 点 → 链头(链内最小点)
  const headOf = (q) => {
    chainFlood(bd, q);
    let h = q;
    for (let i = 0; i < F_STONES.len; i++) if (F_STONES[i] < h) h = F_STONES[i];
    for (let i = 0; i < F_STONES.len; i++) headCache[F_STONES[i]] = h;
    return h;
  };
  const adjToHead = (loc, hd) => {
    for (let k = 0; k < 4; k++) {
      const q = NB_OFF[loc * 4 + k];
      if (q >= 0 && bd[q] === v && headCache[q] === hd) return true;
    }
    return false;
  };

  let atLeastOnePla = false;
  for (let p = 0; p < N2; p++) {
    if (bd[p] === v) atLeastOnePla = true;
    if (bd[p] === v || regionIdx[p] >= 0) continue;
    const ri = regions.length;
    const points = [p];
    regionIdx[p] = ri;
    let vital = [];
    for (let k = 0; k < 4; k++) {
      const q = NB_OFF[p * 4 + k];
      if (q >= 0 && bd[q] === v) {
        const h = headOf(q);
        if (!vital.includes(h)) vital.push(h);
      }
    }
    let internal2 = 0, containsOpp = false;
    for (let h = 0; h < points.length; h++) {
      const q = points[h];
      /* vital 过滤:禁自杀口径下只在空点上过滤 */
      if (vital.length > 0 && bd[q] === EMPTY) {
        vital = vital.filter((hd) => adjToHead(q, hd));
      }
      if (internal2 < 2) {
        let touch = false;
        for (let k = 0; k < 4; k++) {
          const r = NB_OFF[q * 4 + k];
          if (r >= 0 && bd[r] === v) { touch = true; break; }
        }
        if (!touch) internal2++;
      }
      if (bd[q] === 3 - v) containsOpp = true;
      for (let k = 0; k < 4; k++) {
        const r = NB_OFF[q * 4 + k];
        if (r >= 0 && bd[r] !== v && regionIdx[r] < 0) { regionIdx[r] = ri; points.push(r); }
      }
    }
    regions.push({ points, vital, internal2, containsOpp, bordersNonPassAlive: false });
  }

  /* v 链头清单(所有链,不论区域构建时是否已缓存 —— 漏掉会导致链永不死、
   * 邻接区域不被污染,领土误标) */
  const heads = [];
  const inHeads = new Set();
  for (let p = 0; p < N2; p++) {
    if (bd[p] !== v) continue;
    const h = headCache[p] >= 0 ? headCache[p] : headOf(p);
    if (!inHeads.has(h)) { inHeads.add(h); heads.push(h); }
  }

  /* Benson 迭代:活链要 ≥2 个 vital 区域;死链污染邻接区域 */
  const vitalCount = new Map();
  for (const h of heads) vitalCount.set(h, 0);
  for (const r of regions) for (const h of r.vital) vitalCount.set(h, vitalCount.get(h) + 1);
  const dead = new Set();
  let changed = true;
  while (changed) {
    changed = false;
    for (const h of heads) {
      if (dead.has(h) || vitalCount.get(h) >= 2) continue;
      dead.add(h);
      changed = true;
      chainFlood(bd, h);
      const stones = Array.from(F_STONES.slice(0, F_STONES.len));
      for (const s of stones) {
        for (let k = 0; k < 4; k++) {
          const q = NB_OFF[s * 4 + k];
          if (q < 0 || bd[q] === v) continue;
          const reg = regions[regionIdx[q]];
          if (reg && !reg.bordersNonPassAlive) {
            reg.bordersNonPassAlive = true;
            for (const hd of reg.vital) vitalCount.set(hd, vitalCount.get(hd) - 1);
          }
        }
      }
    }
  }

  /* 标记:活链全体子点 */
  for (const h of heads) {
    if (dead.has(h)) continue;
    chainFlood(bd, h);
    for (let i = 0; i < F_STONES.len; i++) result[F_STONES[i]] = v;
  }
  /* 标记:领土 */
  for (const r of regions) {
    const mark = (r.internal2 <= 1 || !r.containsOpp) && !r.bordersNonPassAlive && atLeastOnePla;
    if (mark) {
      for (const p of r.points) result[p] = v;
    } else if (!r.containsOpp && atLeastOnePla) {          // unsafeBigTerritories
      for (const p of r.points) if (result[p] === EMPTY) result[p] = v;
    }
  }
}

/** 面积领土图(fillRowV7 通道 18/19 的来源):1/2 = 归黑/白,0 = 无人 */
export function calculateArea(bd) {
  const result = new Int8Array(N2);
  areaForPla(bd, 1, result);
  areaForPla(bd, 2, result);
  for (let p = 0; p < N2; p++) if (result[p] === EMPTY) result[p] = bd[p];  // 非活子按子色
  return result;
}

/* ==================== 前两手盘面(KataGo getRecentBoard 同款) ====================
 * 生产路径:规则核的滚动盘面环(recentPrevBd/Ko)—— make 时增量维护,O(1) 取,
 * 编码成本不随手数涨。参考路径 encodeFeaturesReplay 用 replayBoard 从空盘重演,
 * 只为环 vs 重演的逐位对拍存在(见 features-test 模糊测试)。 */

/** 从空盘重演 moves 的前 n 手(黑先),返回 { board, ko }(simple-ko 跟踪)。参考实现。 */
function replayBoard(moves, n, initialSide) {
  const bd = new Int8Array(N2);
  const his = initialSide ^ 1;
  let ko = -1;
  for (let i = 0; i < n; i++) {
    const mv = moves[i];
    const v = (i % 2 === 0) ? initialSide + 1 : his + 1;
    if (mv === PASS) { ko = -1; continue; }
    bd[mv] = v;
    let capN = 0, capCell = -1;
    for (let k = 0; k < 4; k++) {
      const q = NB_OFF[mv * 4 + k];
      if (q < 0 || bd[q] !== 3 - v) continue;
      chainFlood(bd, q);
      if (F_LIBS.len === 0) {
        for (let j = 0; j < F_STONES.len; j++) { const c = F_STONES[j]; bd[c] = EMPTY; capN++; capCell = c; }
      }
    }
    chainFlood(bd, mv);
    ko = (capN === 1 && F_STONES.len === 1 && F_LIBS.len === 1 && F_LIBS[0] === capCell) ? capCell : -1;
  }
  return { board: bd, ko };
}

/** 前两手盘面解析(生产):环优先,深度不足按原语义退化为当前盘/前一手 */
function resolvePrevRing(bd, numIncluded) {
  const mc = moveCount();
  const prev1 = (numIncluded >= 1 && mc >= 1)
    ? { board: recentPrevBd(1), ko: recentPrevKo(1) } : { board: bd, ko: koPoint() };
  const prev2 = (numIncluded >= 2 && mc >= 2)
    ? { board: recentPrevBd(2), ko: recentPrevKo(2) } : prev1;
  return { prev1, prev2 };
}

/** 前两手盘面解析(参考):从空盘重演,语义与历史版本逐位一致 */
function resolvePrevReplay(bd, moves, numIncluded, initialSide) {
  const prev1 = numIncluded >= 1 ? replayBoard(moves, moves.length - 1, initialSide)
    : { board: bd, ko: koPoint() };
  const prev2 = numIncluded >= 2 ? replayBoard(moves, moves.length - 2, initialSide) : prev1;
  return { prev1, prev2 };
}

/* ==================== 编码入口 ==================== */

/* 通道 6 复用掩码(避免每次编码分配) */
const SKMASK = new Uint8Array(N2);
/* 通道 3/4/5 的每点气数缓存 */
const LIBCACHE = new Int32Array(N2);

/**
 * 编码 fillRowV7 特征(生产入口)。
 * bd:当前棋盘;side:行棋方(BLACK/WHITE);
 * opts.recentMoves:最近着法(最新在**末尾**;须含完整尾随停着串,见文件头);
 * opts.komi(默认 KOMI = 7.5);opts.outSpatial/outGlobal 复用缓冲。
 * 前两手盘面(通道 15/16)从规则核滚动盘面环取 —— 调用方的 bd 必须处于
 * 「经 make/replayMoves 到达」的状态(引擎一致性约定,搜索/Worker 天然满足)。
 */
export function encodeFeatures(bd, side, opts = {}) {
  const sp = opts.outSpatial ?? new Float32Array(22 * N2);
  const gl = opts.outGlobal ?? new Float32Array(19);
  sp.fill(0); gl.fill(0);
  const moves = opts.recentMoves ?? [];

  /* 历史:尾随停着决定收录量与全局 14(friendlyPassOk=false:不抑制) */
  let trailing = 0;
  for (let i = moves.length - 1; i >= 0 && moves[i] === PASS; i--) trailing++;
  const passWouldEndGame = trailing >= 1;
  let maxHistory = 5;
  if (passWouldEndGame && trailing >= 2) maxHistory = 1;   // isGameFinished:只留最后一手
  let numIncluded = 0;
  if (maxHistory > 0 && moves.length > 0) {
    numIncluded = Math.min(maxHistory, moves.length);
  }

  const { prev1, prev2 } = resolvePrevRing(bd, numIncluded);
  return encodeCore(bd, side, opts, moves, numIncluded, passWouldEndGame, sp, gl, prev1, prev2);
}

/**
 * 参考入口:前两手盘面从空盘重演(历史实现)。
 * 与 encodeFeatures 逐位等价 —— features-test 的模糊测试逐位对拍两者;
 * 环实现有任何回归它先红。
 */
export function encodeFeaturesReplay(bd, side, opts = {}) {
  const sp = opts.outSpatial ?? new Float32Array(22 * N2);
  const gl = opts.outGlobal ?? new Float32Array(19);
  sp.fill(0); gl.fill(0);
  const moves = opts.recentMoves ?? [];

  let trailing = 0;
  for (let i = moves.length - 1; i >= 0 && moves[i] === PASS; i--) trailing++;
  const passWouldEndGame = trailing >= 1;
  let maxHistory = 5;
  if (passWouldEndGame && trailing >= 2) maxHistory = 1;
  let numIncluded = 0;
  if (maxHistory > 0 && moves.length > 0) {
    numIncluded = Math.min(maxHistory, moves.length);
  }

  const { prev1, prev2 } = resolvePrevReplay(bd, moves, numIncluded, opts.initialSide ?? BLACK);
  return encodeCore(bd, side, opts, moves, numIncluded, passWouldEndGame, sp, gl, prev1, prev2);
}

function encodeCore(bd, side, opts, moves, numIncluded, passWouldEndGame, sp, gl, prev1, prev2) {
  const mineV = side + 1, oppV = 2 - side;
  const komi = opts.komi ?? KOMI;

  /* 通道 0~5:在盘 / 己敌子 / 1·2·3 气 */
  for (let p = 0; p < N2; p++) {
    sp[idx(0, p)] = 1;
    const v = bd[p];
    if (v === EMPTY) continue;
    chainFlood(bd, p);
    for (let i = 0; i < F_STONES.len; i++) LIBCACHE[F_STONES[i]] = F_LIBS.len;
    sp[idx(v === mineV ? 1 : 2, p)] = 1;
    const libs = F_LIBS.len;
    if (libs <= 3) sp[idx(2 + libs, p)] = 1;              // 1→3,2→4,3→5
  }

  /* 通道 6:单劫点 + 禁全同点(自杀点不算) */
  const ban = superkoBannedPoints(bd, side, SKMASK);
  for (let p = 0; p < N2; p++) if (ban[p]) sp[idx(6, p)] = 1;

  /* 历史:通道 9-13(最近 5 手)由入口按 numIncluded 填好传入逻辑;
   * 此处只填空间/全局通道(见 encodeFeatures 的 numIncluded 计算) */
  if (numIncluded > 0) {
    const h = moves.slice(-numIncluded);
    for (let j = 0; j < h.length; j++) {
      const mv = h[h.length - 1 - j];                      // j=0 最新:spatial 9 / global 0
      if (mv === PASS) gl[j] = 1;
      else sp[idx(9 + j, mv)] = 1;
    }
  }

  /* 通道 14/17:当前盘征子 */
  iterLadders(bd, koPoint(), (loc, work) => {
    sp[idx(14, loc)] = 1;
    if (bd[loc] === oppV && LIBCACHE[loc] > 1) {
      for (const w of work) sp[idx(17, w)] = 1;
    }
  });

  /* 通道 15/16:前一手 / 前二手盘面的征子(来源由入口解析:环或重演参考;
   * 未收录历史时退化为当前盘 / 前一手) */
  iterLadders(prev1.board, prev1.ko, (loc) => { sp[idx(15, loc)] = 1; });
  iterLadders(prev2.board, prev2.ko, (loc) => { sp[idx(16, loc)] = 1; });

  /* 通道 18/19:面积领土 */
  const area = calculateArea(bd);
  for (let p = 0; p < N2; p++) {
    if (area[p] === mineV) sp[idx(18, p)] = 1;
    else if (area[p] === oppV) sp[idx(19, p)] = 1;
  }
  /* 通道 7/8/20/21 恒 0(encore) */

  /* ===== 全局 ===== */
  let selfKomi = side === WHITE ? komi : -komi;
  const clip = N2 + 20;                                    // KOMI_CLIP_RADIUS
  if (selfKomi > clip) selfKomi = clip;
  else if (selfKomi < -clip) selfKomi = -clip;
  gl[5] = selfKomi / 20;
  gl[6] = 1; gl[7] = 0.5;                                  // KO_POSITIONAL
  /* gl[8..13] 恒 0:禁自杀 / 面积计分 / 无税 / 无 encore */
  gl[14] = passWouldEndGame ? 1 : 0;                       // passWouldEndPhase(friendlyPassOk=false 不抑制)
  /* gl[15..17] 恒 0:无让子优势 / 无 button */
  /* gl[18]:贴目奇偶三角波(面积计分) */
  {
    const drawableEven = (N2 % 2) === 0;                   // 361 为奇 → false
    const komiFloor = drawableEven ? Math.floor(selfKomi / 2) * 2
      : Math.floor((selfKomi - 1) / 2) * 2 + 1;
    let delta = selfKomi - komiFloor;
    if (delta < 0) delta = 0;
    if (delta > 2) delta = 2;
    gl[18] = delta < 0.5 ? delta : delta < 1.5 ? 1 - delta : delta - 2;
  }
  return { spatial: sp, global: gl };
}
