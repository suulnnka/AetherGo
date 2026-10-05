/* ============================================================
 * AetherGo NN 特征 — 征子搜索(Board::searchIsLadderCaptured 移植)
 *
 * 从 features.js 拆出:结构与 C++ board.cpp 逐行机械转写保持独立成文,
 * 便于与 KataGo 原生实现对拍(test/ladderdiff.mjs)。对外只暴露
 * iterLadders(bd, initialKo, cb) —— 对盘上每颗 1/2 气的子(按链去重)回调。
 * ============================================================ */
import { N, N2, EMPTY } from '../protocol.js';
import { chainFlood, F_STONES, F_LIBS, F_SEEN, F_LSEEN, nextFStamp, NB_OFF, adj } from './flood.js';

/* ==================== 征子搜索(Board::searchIsLadderCaptured 移植) ====================
 * 在 scratch 棋盘 LBD 上做 AND/OR 搜索;LKO 是局部 simple-ko 点,
 * 与引擎全局状态完全隔离。递归结构与 C++ 显式栈等价:
 *   - 栈深上限 542(= 19*19*3/2+1):照 C++ 视为「被提」(return true)
 *   - 节点预算 25000(每次 lPlay 计数):耗尽则整体返回 false(异常直穿根)
 *   - 守方先行时根节点清劫点(假设守方所有劫都成立)
 *   - 守方遇当前劫点 = 守方胜(依赖劫的征子不算成立)                    */

const LBD = new Int8Array(N2);
/* 每层提子记录:层数 = 征子栈深 + 2,每层最多 N2+1 个提子点 */
const L_STACK = ((N * N * 3) / 2 | 0) + 1;      // C++ 整除:19*19*3/2+1 = 542
const LCAP = new Int32Array((L_STACK + 2) * (N2 + 1));
let LKO = -1;                               // 局部 simple-ko 点
let lNodeCount = 0;
let lLastCapN = 0;                          // lPlay 成功时的提子数(迭代栈记录用)

function lPlay(p, v, lvl) {
  if (p < 0 || p === LKO || LBD[p] !== EMPTY) return -1;
  const his = 3 - v, base = lvl * (N2 + 1);
  LBD[p] = v;
  let capN = 0;
  for (let k = 0; k < 4; k++) {
    const q = NB_OFF[p * 4 + k];
    if (q < 0 || LBD[q] !== his) continue;
    chainFlood(LBD, q);
    if (F_LIBS.len === 0) {
      for (let i = 0; i < F_STONES.len; i++) { const c = F_STONES[i]; LBD[c] = EMPTY; LCAP[base + capN++] = c; }
    }
  }
  chainFlood(LBD, p);
  if (F_LIBS.len === 0) { LBD[p] = EMPTY; return -1; }            // 自杀
  LKO = (capN === 1 && F_STONES.len === 1 && F_LIBS.len === 1 && F_LIBS[0] === LCAP[base]) ? LCAP[base] : -1;
  lLastCapN = capN;
  return capN;
}
function lUndo(p, v, capN, lvl) {
  const his = 3 - v, base = lvl * (N2 + 1);
  LBD[p] = EMPTY;
  for (let i = 0; i < capN; i++) LBD[LCAP[base + i]] = his;
}

/** 空点 p 落 v 子后新链的气数(封顶 max,计入将提子腾出的点;不落盘) */
const CAPM = new Uint8Array(N2);
function lLibsAfterPlay(p, v, max) {
  const his = 3 - v;
  CAPM.fill(0);
  LBD[p] = v;
  for (let k = 0; k < 4; k++) {
    const q = NB_OFF[p * 4 + k];
    if (q < 0 || LBD[q] !== his) continue;
    chainFlood(LBD, q);
    if (F_LIBS.len === 0) for (let i = 0; i < F_STONES.len; i++) CAPM[F_STONES[i]] = 1;
  }
  const st = nextFStamp();
  let ns = 1, libs = 0;
  F_STONES[0] = p; F_SEEN[p] = st;
  for (let h = 0; h < ns; h++) {
    const q = F_STONES[h];
    for (let k = 0; k < 4; k++) {
      const r = NB_OFF[q * 4 + k];
      if (r < 0) continue;
      const c = LBD[r];
      if (c === EMPTY || CAPM[r]) {
        if (F_LSEEN[r] !== st) {
          F_LSEEN[r] = st; libs++;
          if (libs >= max) { LBD[p] = EMPTY; return max; }
        }
      } else if (c === v && F_SEEN[r] !== st) { F_SEEN[r] = st; F_STONES[ns++] = r; }
    }
  }
  LBD[p] = EMPTY;
  return libs;
}

/** 落子后气数上下界(攻守剪枝用) */
function lBoundLibsAfterPlay(p, v) {
  const his = 3 - v;
  let imm = 0, caps = 0, capStones = 0, connLibs = 0, maxConn = 0;
  const seenHeads = [];
  for (let k = 0; k < 4; k++) {
    const q = NB_OFF[p * 4 + k];
    if (q < 0) continue;
    const c = LBD[q];
    if (c === EMPTY) { imm++; continue; }
    chainFlood(LBD, q);
    if (c === his) {
      if (F_LIBS.len === 1 && !seenHeads.includes(F_STONES[0])) {
        seenHeads.push(F_STONES[0]);
        caps++; capStones += F_STONES.len;
      }
    } else {
      const conn = F_LIBS.len - 1;
      connLibs += conn;
      if (conn > maxConn) maxConn = conn;
    }
  }
  return [caps + Math.max(maxConn, imm), imm + capStones + connLibs];
}

/** v 下在空点 p 是否「提单子成劫」(边界外的墙视作已包围,与 C++ C_WALL 一致) */
function lWouldBeKoCapture(p, v) {
  if (p < 0 || LBD[p] !== EMPTY) return false;
  const his = 3 - v;
  let capturable = -1;
  for (let k = 0; k < 4; k++) {
    const q = NB_OFF[p * 4 + k];
    if (q < 0) continue;                     // 墙:不算破坏包围
    if (LBD[q] !== his) return false;        // 空/己子:不是劫口
    chainFlood(LBD, q);
    if (F_LIBS.len === 1) {
      if (capturable >= 0) return false;
      capturable = q;
    }
  }
  if (capturable < 0) return false;
  chainFlood(LBD, capturable);
  return F_STONES.len === 1;
}

const lImmediateLibs = (p) => {
  let n = 0;
  for (let k = 0; k < 4; k++) { const q = NB_OFF[p * 4 + k]; if (q >= 0 && LBD[q] === EMPTY) n++; }
  return n;
};

const lHeuristicConnX2 = (p, v) => {
  let n = 0;
  for (let k = 0; k < 4; k++) {
    const q = NB_OFF[p * 4 + k];
    if (q >= 0 && LBD[q] === v) {
      chainFlood(LBD, q);
      const libs = F_LIBS.len;
      if (libs > 1) n += libs * 2 - 3;
    }
  }
  return n;
};

/** 目标链邻接 1 气敌链的气点(提子获气的着手)—— 写缓冲版(迭代栈用)。
 *  Board::findLibertyGainingCaptures 移植:按链去重后依次写 out[base..]。 */
function lLibertyGainingCapturesInto(seed, out, base) {
  const his = 3 - LBD[seed];
  const checked = [];
  let n = 0;
  chainFlood(LBD, seed);
  const stones = Array.from(F_STONES.slice(0, F_STONES.len));
  for (const s of stones) {
    for (let k = 0; k < 4; k++) {
      const q = NB_OFF[s * 4 + k];
      if (q < 0 || LBD[q] !== his) continue;
      chainFlood(LBD, q);
      const head = F_STONES[0];
      if (F_LIBS.len === 1 && !checked.includes(head)) {
        checked.push(head);
        for (let i = 0; i < F_LIBS.len; i++) out[base + n++] = F_LIBS[i];
      }
    }
  }
  return n;
}

function lHasLibertyGainingCaptures(seed) {
  const his = 3 - LBD[seed];
  chainFlood(LBD, seed);
  for (let h = 0; h < F_STONES.len; h++) {
    const s = F_STONES[h];
    for (let k = 0; k < 4; k++) {
      const q = NB_OFF[s * 4 + k];
      if (q >= 0 && LBD[q] === his) {
        chainFlood(LBD, q);
        if (F_LIBS.len === 1) return true;
      }
    }
  }
  return false;
}

/* ---- searchIsLadderCaptured 主体:C++ 迭代结构的逐行机械转写(显式栈) ----
 * returnValue / returnedFromDeeper / moveListCur 语义与 board.cpp 一一对应。
 * (递归改写在劫点保存时机上出过隐蔽分歧,弃用;机械转写与 calculateArea 同法。)
 * 每层着法表存 BUF 的 lvl*(N2+2) 段;每层的落子记录存 rec*(含落子前 LKO)。 */
const LBUF = new Int32Array((L_STACK + 2) * (N2 + 2));
function lLadderCaptured(seed, defenderFirst) {
  if (LBD[seed] !== 1 && LBD[seed] !== 2) return false;
  chainFlood(LBD, seed);
  if (F_LIBS.len > 2 || (defenderFirst && F_LIBS.len > 1)) return false;

  const v = LBD[seed], his = 3 - v;
  const koSaved = LKO;
  if (defenderFirst) LKO = -1;              // 根节点:假设守方所有劫都成立
  lNodeCount = 0;                           // 预算随每次调用重置(C++ 局部变量)

  const stackSize = L_STACK;                // x*y*3/2+1 = 542(C++ 整除)
  const mlStart = new Int32Array(stackSize), mlLen = new Int32Array(stackSize);
  const mlCur = new Int32Array(stackSize);
  const recP = new Int32Array(stackSize), recV = new Int32Array(stackSize);
  const recCap = new Int32Array(stackSize), recKo = new Int32Array(stackSize);
  let sp = 0, returnValue = false, returnedFromDeeper = false;
  mlCur[0] = -1; mlStart[0] = 0; mlLen[0] = 0;

  while (true) {
    if (sp <= -1) { LKO = koSaved; return returnValue; }

    if (sp >= stackSize - 1) { returnValue = true; returnedFromDeeper = true; sp--; continue; }
    if (lNodeCount >= 25000) {
      sp -= 1;
      while (sp >= 0) { lUndo(recP[sp], recV[sp], recCap[sp], sp); LKO = recKo[sp]; sp -= 1; }
      LKO = koSaved;
      return false;
    }

    const isDefender = (defenderFirst && (sp % 2) === 0) || (!defenderFirst && (sp % 2) === 1);

    if (mlCur[sp] === -1) {
      chainFlood(LBD, seed);
      const libs = F_LIBS.len;
      if (!isDefender && libs <= 1) { returnValue = true; returnedFromDeeper = true; sp--; continue; }
      if (!isDefender && libs >= 3) { returnValue = false; returnedFromDeeper = true; sp--; continue; }
      if (isDefender && libs >= 2) { returnValue = false; returnedFromDeeper = true; sp--; continue; }
      if (isDefender && LKO >= 0) { returnValue = false; returnedFromDeeper = true; sp--; continue; }

      const base = sp * (N2 + 2);
      let len = 0;
      if (isDefender) {
        len = lLibertyGainingCapturesInto(seed, LBUF, base);
        chainFlood(LBD, seed);
        LBUF[base + len++] = F_LIBS[0];                 // 唯一气,恒为表尾
        const [lb, ub] = lBoundLibsAfterPlay(LBUF[base + len - 1], v);
        if (lb >= 3) { returnValue = false; returnedFromDeeper = true; sp--; continue; }
        if (len === 1 && ub <= 1) { returnValue = true; returnedFromDeeper = true; sp--; continue; }
      } else {
        chainFlood(LBD, seed);
        LBUF[base] = F_LIBS[0];
        LBUF[base + 1] = F_LIBS[1];
        len = 2;
        let libs0 = lImmediateLibs(LBUF[base]);
        let libs1 = lImmediateLibs(LBUF[base + 1]);
        if (libs0 === 0 && libs1 === 0
          && lWouldBeKoCapture(LBUF[base], his) && lWouldBeKoCapture(LBUF[base + 1], his)) {
          if (lLibsAfterPlay(LBUF[base], v, 3) <= 2 && lLibsAfterPlay(LBUF[base + 1], v, 3) <= 2
            && !lHasLibertyGainingCaptures(seed)) {
            returnValue = true; returnedFromDeeper = true; sp--; continue;
          }
        }
        if (!adj(LBUF[base], LBUF[base + 1])) {
          if (libs0 >= 3 && libs1 >= 3) { returnValue = false; returnedFromDeeper = true; sp--; continue; }
          else if (libs0 >= 3) len = 1;
          else if (libs1 >= 3) { LBUF[base] = LBUF[base + 1]; len = 1; }
        }
        if (len > 1) {
          libs0 = libs0 * 2 + lHeuristicConnX2(LBUF[base], v);
          libs1 = libs1 * 2 + lHeuristicConnX2(LBUF[base + 1], v);
          if (libs1 > libs0) { const t = LBUF[base]; LBUF[base] = LBUF[base + 1]; LBUF[base + 1] = t; }
        }
      }
      mlLen[sp] = len;
      mlCur[sp] = 0;
    } else {
      if (returnedFromDeeper) {
        lUndo(recP[sp], recV[sp], recCap[sp], sp);
        LKO = recKo[sp];
      }
      if (isDefender && !returnValue) { returnedFromDeeper = true; sp--; continue; }
      if (!isDefender && returnValue) { returnedFromDeeper = true; sp--; continue; }
      mlCur[sp]++;
    }

    if (mlCur[sp] >= mlLen[sp]) {
      returnValue = isDefender;
      returnedFromDeeper = true;
      sp--;
      continue;
    }

    const move = LBUF[sp * (N2 + 2) + mlCur[sp]];
    const p = isDefender ? v : his;
    const koBefore = LKO;
    if (lPlay(move, p, sp) < 0) {
      returnValue = isDefender;             // 非法着 = 该方此着失败,同层换下一手
      returnedFromDeeper = false;
      continue;
    }
    recP[sp] = move; recV[sp] = p; recCap[sp] = lLastCapN; recKo[sp] = koBefore;
    lNodeCount++;
    sp++;
    mlCur[sp] = -1;
    mlStart[sp] = 0; mlLen[sp] = 0;
  }
}

/** 2 气、攻方先行(AttackerFirst2Libs 移植):成立时 workingMoves = 攻方可行着手。
 *  外层落子用保留层 L_STACK-1 —— 内层搜索从 0 层起,同层会互相覆写提子记录(踩过的坑)。 */
const WRAPPER_LVL = L_STACK - 1;
function lLadder2LibsAttackerFirst(seed, workingMoves) {
  if (LBD[seed] !== 1 && LBD[seed] !== 2) return false;
  chainFlood(LBD, seed);
  if (F_LIBS.len !== 2) return false;
  const his = 3 - LBD[seed];
  const work = [];
  for (const mv of [F_LIBS[0], F_LIBS[1]]) {
    const koBefore = LKO;
    const capN = lPlay(mv, his, WRAPPER_LVL);
    if (capN >= 0) {
      if (lLadderCaptured(seed, true)) work.push(mv);    // 全新搜索:守方先行、重置预算
      lUndo(mv, his, capN, WRAPPER_LVL);
    }
    LKO = koBefore;
  }
  if (work.length === 0) return false;
  workingMoves.length = 0;
  for (const m of work) workingMoves.push(m);
  return true;
}

/** iterLadders:对盘上每颗 1/2 气的子(按链去重)回调 cb(loc, workingMoves) */
export function iterLadders(bd, initialKo, cb) {
  const solved = new Map();
  for (let p = 0; p < N2; p++) {
    const stone = bd[p];
    if (stone !== 1 && stone !== 2) continue;
    chainFlood(bd, p);
    const libs = F_LIBS.len;
    if (libs !== 1 && libs !== 2) continue;
    let head = p;
    for (let i = 0; i < F_STONES.len; i++) if (F_STONES[i] < head) head = F_STONES[i];
    if (solved.has(head)) {
      if (solved.get(head)) cb(p, []);
      continue;
    }
    LBD.set(bd);
    LKO = initialKo;
    let laddered;
    const work = [];
    if (libs === 1) laddered = lLadderCaptured(p, true);
    else laddered = lLadder2LibsAttackerFirst(p, work);
    solved.set(head, laddered);
    if (laddered) cb(p, work);
  }
}

