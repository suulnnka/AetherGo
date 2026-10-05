/* ============================================================
 * AetherGo 数子与死子 —— 中国规则数子、双停死子判定、ownership 辅助标注
 *
 * 计分:中国规则数子法(子 + 空,黑贴 7.5 目)。
 * 死子:规则侧 = Benson 绝对活棋 + 「对方先手能否提掉」的小预算 AND/OR 搜索
 *      (预算耗尽按活处理,宁漏勿错);NN ownership 相悖链补标(KataGo 阈值法)。
 *
 * 洪泛工作区(GRP/SEEN/LIBSEEN/stamp/groupInfo)属 src/rules.js,
 * 本文件经 nextStamp()/groupInfo()/gSize()/gLibs()/gLibPt() 访问器使用 ——
 * 与规则核心的合法查询串行使用,互不嵌套。
 * 对外完整 API 由 src/engine.js 统一再导出。
 * ============================================================ */
import { N2, EMPTY, KOMI } from './protocol.js';
import {
  NB, NB_N, GRP, SEEN, LIBSEEN,
  nextStamp, groupInfo, gSize, gLibs, gLibPt,
} from './rules.js';

/* ==================== 数子(中国规则) ==================== */

let EV_BS = 0, EV_WS = 0, EV_BT = 0, EV_WT = 0;

/** 数子:黑 = 黑子 + 黑空,白 = 白子 + 白空(空点区域按边界颜色归属,双方都贴边算公气) */
function areaScore(bd) {
  const st = nextStamp();
  let bs = 0, ws = 0, bt = 0, wt = 0;
  for (let p = 0; p < N2; p++) {
    const v = bd[p];
    if (v === 1) bs++; else if (v === 2) ws++;
  }
  for (let p = 0; p < N2; p++) {
    if (bd[p] !== EMPTY || SEEN[p] === st) continue;
    let head = 0, size = 0, touch = 0;            // touch:1 黑边 / 2 白边
    GRP[size++] = p; SEEN[p] = st;
    while (head < size) {
      const q = GRP[head++];
      for (let k = 0; k < NB_N[q]; k++) {
        const nb = NB[q * 4 + k], v = bd[nb];
        if (v === EMPTY) {
          if (SEEN[nb] !== st) { SEEN[nb] = st; GRP[size++] = nb; }
        } else touch |= (v === 1 ? 1 : 2);
      }
    }
    if (touch === 1) bt += size;
    else if (touch === 2) wt += size;
  }
  EV_BS = bs; EV_WS = ws; EV_BT = bt; EV_WT = wt;
}

/** 静态评估:黑方视角的目差(子 + 空 − 贴目)。 */
export function evaluate(bd, komi = KOMI) {
  areaScore(bd);
  return EV_BS + EV_BT - EV_WS - EV_WT - komi;
}

/** 终局数子结果(UI 显示用):黑/白各自的总目与目差 */
export function scoreGame(bd, komi = KOMI) {
  areaScore(bd);
  const black = EV_BS + EV_BT, white = EV_WS + EV_WT + komi;
  return { black, white, margin: black - white };
}

/* ==================== 双停死子处理 ==================== */

/* 人类对局双停时盘上常残留死子(引擎自对弈会自然提掉),直接数子会算错。
 * 口径:对每条非绝对活棋的链跑「对方先手能否提掉这块」的小预算搜索;
 * 找到的死块全体子点标记为死。守方脱先(pass)分支保留 —— 消极应对
 * 也得活才算活。攻方只下目标链的气(对必胜方收紧着法不产生假必胜);
 * 守方在局部区域(目标气 ∪ 相邻己链的气 ∪ 相邻敌链的气 ∪ 气点所在空域)任意应。
 * 预算耗尽/深度用完 → 按活处理(宁漏勿错:误判活为死会把目数大算错,
 * 反之只是保守)。规则侧漏判由 deadStonesWithOwnership 的 ownership 图补标。 */

/** Benson 绝对活棋(pass-alive):v 色中即使从此不应手也提不掉的链。
 * 返回 Uint8Array(N2),活链的子点标记 1。 */
function bensonAlive(bd, v) {
  /* v 色链 */
  const CHAIN = new Int32Array(N2).fill(-1);
  const chains = [];
  for (let p = 0; p < N2; p++) {
    if (bd[p] !== v || CHAIN[p] >= 0) continue;
    const id = chains.length, mem = [p];
    CHAIN[p] = id;
    for (let h = 0; h < mem.length; h++) {
      for (let k = 0; k < NB_N[mem[h]]; k++) {
        const r = NB[mem[h] * 4 + k];
        if (bd[r] === v && CHAIN[r] < 0) { CHAIN[r] = id; mem.push(r); }
      }
    }
    chains.push(mem);
  }
  /* 非 v 点连通区域(空点 + 敌子),记录空点与邻接链 */
  const RG = new Int32Array(N2).fill(-1);
  const rEmpty = [], rAdj = [];
  for (let p = 0; p < N2; p++) {
    if (bd[p] === v || RG[p] >= 0) continue;
    const id = rEmpty.length, pts = [p];
    RG[p] = id;
    const empt = [], adj = new Set();
    for (let h = 0; h < pts.length; h++) {
      const q = pts[h];
      if (bd[q] === EMPTY) empt.push(q);
      for (let k = 0; k < NB_N[q]; k++) {
        const r = NB[q * 4 + k];
        if (bd[r] !== v) {
          if (RG[r] < 0) { RG[r] = id; pts.push(r); }
        } else adj.add(CHAIN[r]);
      }
    }
    rEmpty.push(empt); rAdj.push(adj);
  }
  /* 区域对链 vital:区域所有空点都是该链的气 */
  const vital = chains.map(() => new Array(rEmpty.length).fill(false));
  for (let r = 0; r < rEmpty.length; r++) {
    if (rEmpty[r].length === 0) continue;         // 无空点的区域不 vital
    for (const c of rAdj[r]) {
      let ok = true;
      for (const e of rEmpty[r]) {
        let isLib = false;
        for (let k = 0; k < NB_N[e]; k++) {
          if (CHAIN[NB[e * 4 + k]] === c) { isLib = true; break; }
        }
        if (!isLib) { ok = false; break; }
      }
      vital[c][r] = ok;
    }
  }
  /* 迭代剪枝:链要 ≥2 个 vital 活区域;区域要所有邻接链都活 */
  const alive = chains.map(() => true), rLive = rEmpty.map(() => true);
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < chains.length; c++) {
      if (!alive[c]) continue;
      let n = 0;
      for (let r = 0; r < rEmpty.length; r++) if (rLive[r] && vital[c][r]) n++;
      if (n < 2) { alive[c] = false; changed = true; }
    }
    for (let r = 0; r < rEmpty.length; r++) {
      if (!rLive[r]) continue;
      for (const c of rAdj[r]) if (!alive[c]) { rLive[r] = false; changed = true; break; }
    }
  }
  const mask = new Uint8Array(N2);
  for (let c = 0; c < chains.length; c++) {
    if (!alive[c]) continue;
    for (const p of chains[c]) mask[p] = 1;
  }
  return mask;
}

/* 提子搜索的暂存:每层一个槽(深度 ≤ 16),存该层提掉的敌子 */
const DS_CAP = new Int32Array(16 * (N2 + 1));
let DS_KO = -1;                                    // 搜索内局部 simple-ko 点
let dsNodes = 0;                                   // 节点预算(每次搜索重置)

/** scratch 落子(含提子/自杀/局部劫),非法 -1,合法返回提子数(提子点在 DS_CAP[lvl*(N2+1)..]) */
function playSc(bd, p, v, lvl) {
  if (p < 0 || p === DS_KO || bd[p] !== EMPTY) return -1;
  const his = 3 - v, base = lvl * (N2 + 1);
  bd[p] = v;
  let capN = 0;
  for (let k = 0; k < NB_N[p]; k++) {
    const q = NB[p * 4 + k];
    if (bd[q] !== his) continue;
    groupInfo(bd, q);
    if (gLibs() === 0) {
      for (let i = 0; i < gSize(); i++) {
        const c = GRP[i];
        bd[c] = EMPTY;
        DS_CAP[base + capN++] = c;
      }
    }
  }
  groupInfo(bd, p);
  if (gLibs() === 0) { bd[p] = EMPTY; return -1; }  // 自杀
  DS_KO = (capN === 1 && gSize() === 1 && gLibs() === 1 && gLibPt() === DS_CAP[base]) ? DS_CAP[base] : -1;
  return capN;
}

/** scratch 撤销(与 playSc 严格对称) */
function undoSc(bd, p, v, capN, lvl) {
  const his = 3 - v, base = lvl * (N2 + 1);
  bd[p] = EMPTY;
  for (let i = 0; i < capN; i++) bd[DS_CAP[base + i]] = his;
}

/** 从 seed 洪泛 v 色链:成员写 out.m,气写 out.l;返回成员数(0 = seed 已被提) */
const DS_OUT = { m: new Int32Array(N2), l: new Int32Array(N2) };
function chainAt(bd, seed, v) {
  if (bd[seed] !== v) return 0;
  const st = nextStamp();
  const seen = SEEN, m = DS_OUT.m, l = DS_OUT.l;
  let ms = 0, ls = 0;
  m[ms++] = seed; seen[seed] = st;
  for (let h = 0; h < ms; h++) {
    const q = m[h];
    for (let k = 0; k < NB_N[q]; k++) {
      const r = NB[q * 4 + k];
      if (bd[r] === EMPTY) {
        if (LIBSEEN[r] !== st) { LIBSEEN[r] = st; l[ls++] = r; }
      } else if (bd[r] === v && seen[r] !== st) { seen[r] = st; m[ms++] = r; }
    }
  }
  DS_OUT.mn = ms; DS_OUT.ln = ls;
  return ms;
}

/** 守方局部应手区域:目标链的气 ∪ 相邻己链的气(连接支援)∪ 相邻敌链的气
 *  (反提)∪ 气点所在空域的空点(做眼空间)。目标的气排最前 —— 找活路优先试延伸 */
function defMoves(bd, seed, v) {
  const n = chainAt(bd, seed, v);
  /* 先快照:后面的 chainAt(邻接链)会覆写 DS_OUT */
  const targetLibs = Array.from(DS_OUT.l.slice(0, DS_OUT.ln));
  const members = Array.from(DS_OUT.m.slice(0, n));
  const mark = new Uint8Array(N2);
  const out = [];
  for (const q of members) {
    for (let k = 0; k < NB_N[q]; k++) {
      const r = NB[q * 4 + k];
      if (bd[r] === EMPTY) continue;
      if (chainAt(bd, r, bd[r]) === 0) continue;
      for (let j = 0; j < DS_OUT.ln; j++) {
        const lp = DS_OUT.l[j];
        if (!mark[lp]) { mark[lp] = 1; out.push(lp); }
      }
    }
  }
  /* 气点所在空域:从每个气点只经空点洪泛 */
  const st = nextStamp();
  for (const lib of targetLibs) {
    if (SEEN[lib] === st) continue;
    const stack = [lib]; SEEN[lib] = st;
    while (stack.length) {
      const p = stack.pop();
      if (!mark[p]) { mark[p] = 1; out.push(p); }
      for (let k = 0; k < NB_N[p]; k++) {
        const r = NB[p * 4 + k];
        if (bd[r] === EMPTY && SEEN[r] !== st) { SEEN[r] = st; stack.push(r); }
      }
    }
  }
  const inHead = new Set(targetLibs);
  return targetLibs.concat(out.filter((p) => !inHead.has(p)));
}

/** v 色目标链(seed)能否被对方**先手**提掉:AND/OR 极小极大。
 *  返回 true = 必死(对方最优、我方最优应对下仍被提)。 */
function dsSearch(bd, seed, v, attacker, depth, lvl) {
  if (bd[seed] !== v) return true;                // 目标链已整体被提
  if (--dsNodes < 0) return false;
  if (chainAt(bd, seed, v) === 0) return true;
  const libs = DS_OUT.ln;
  /* 提完一条链至少要填掉当前全部气(每手攻方至多减一气) */
  if (((depth + 1) >> 1) < libs) return false;
  if (depth <= 0) return false;
  /* 快照气列表:递归里的 chainAt 会覆写 DS_OUT */
  const libList = Array.from(DS_OUT.l.slice(0, libs));
  if (attacker) {
    const av = 3 - v;
    for (const p of libList) {
      const savedKo = DS_KO;
      const capN = playSc(bd, p, av, lvl);
      if (capN < 0) { DS_KO = savedKo; continue; }
      const win = dsSearch(bd, seed, v, false, depth - 1, lvl + 1);
      DS_KO = savedKo;
      undoSc(bd, p, av, capN, lvl);
      if (win) return true;
    }
    return false;
  }
  /* 守方:先试脱先(不应对也得活才算真活) */
  if (!dsSearch(bd, seed, v, true, depth - 1, lvl + 1)) return false;
  const moves = defMoves(bd, seed, v);
  for (const p of moves) {
    const savedKo = DS_KO;
    const capN = playSc(bd, p, v, lvl);
    if (capN < 0) { DS_KO = savedKo; continue; }
    const alive = !dsSearch(bd, seed, v, true, depth - 1, lvl + 1);
    DS_KO = savedKo;
    undoSc(bd, p, v, capN, lvl);
    if (alive) return false;
  }
  return true;
}

/* 19 路块大气长、死活变化比 9 路深,预算与深度都比 9 路放宽;耗尽仍按活处理 */
const DS_DEPTH = 16, DS_BUDGET = 24000;

/** 双停终局时的死子判定:返回应视为死子的全体子点(数组,可能为空) */
export function deadStones(bd) {
  const dead = [];
  for (const v of [1, 2]) {
    const alive = bensonAlive(bd, v);
    const seen = new Uint8Array(N2);
    for (let seed = 0; seed < N2; seed++) {
      if (bd[seed] !== v || seen[seed] || alive[seed]) continue;
      const n = chainAt(bd, seed, v);             // 标记整条链,顺便判重
      const members = Array.from(DS_OUT.m.slice(0, n));
      for (const p of members) seen[p] = 1;
      const scratch = bd.slice();
      DS_KO = -1; dsNodes = DS_BUDGET;
      if (dsSearch(scratch, seed, v, true, DS_DEPTH, 0)) dead.push(...members);
    }
  }
  return dead;
}

/**
 * 数子明细(UI 数子窗口用):先移除死子,再给出黑/白各自的子数、空数与
 * 总目(白含贴目)。deadOverride 语义同 finalScore。 */
export function scoreBreakdown(bd, komi = KOMI, deadOverride) {
  const dead = deadOverride ?? deadStones(bd);
  let b2 = bd;
  if (dead.length) {
    b2 = new Int8Array(N2); b2.set(bd);
    for (const p of dead) b2[p] = EMPTY;
  }
  areaScore(b2);
  const black = EV_BS + EV_BT, white = EV_WS + EV_WT + komi;
  return {
    blackStones: EV_BS, blackTerritory: EV_BT,
    whiteStones: EV_WS, whiteTerritory: EV_WT,
    komi, black, white, margin: black - white, dead: Array.from(dead),
  };
}

/** 双停终局数子:移除死子后的总目与目差(数子明细见 scoreBreakdown)。
 *  deadOverride 可传入手工标注的死子点(UI 允许手改后重算),缺省自动判定。 */
export function finalScore(bd, komi = KOMI, deadOverride) {
  const { black, white, margin, dead } = scoreBreakdown(bd, komi, deadOverride);
  return { black, white, margin, dead };
}

/* ==================== ownership 辅助死子标注 ==================== */

/* 阈值(KataGo 判死同款量级):链平均 ownership 与链色相悖超过它才判死。
 * ownership 逐点是 tanh 归属(黑正白负,-1..1);死子在对方腹地通常 |值| 很大,
 * 公气 / 劫活的链在 0 附近 —— 模糊地带一律不动作。 */
const OWN_DEAD_T = 0.35;

/**
 * ownership 辅助的死子判定(KataGo 阈值法 + 规则侧保底):
 * 在规则侧 deadStones()(Benson 活棋 + 小预算提子搜索,宁漏勿错)的基础上,
 * 把 NN ownership 与链色**强烈相悖**的链补标为死 —— 补的是规则侧小预算
 * 搜不出来的中型死棋(19 路块大气长,预算耗尽按活处理的漏判)。
 * 三条保底:Benson 绝对活棋永不死;规则侧已判死的链不因 ownership 翻案;
 * ownership 模糊(|链均值| < deadT)不动作。
 * ownership 为空(null / undefined)时退化为纯规则判定。
 * opts: { deadT = 0.35 } 阈值可调(测试用)。
 */
export function deadStonesWithOwnership(bd, ownership, opts = {}) {
  const dead = new Set(deadStones(bd));
  if (!ownership) return Array.from(dead);
  const deadT = opts.deadT ?? OWN_DEAD_T;
  for (const v of [1, 2]) {
    const alive = bensonAlive(bd, v);
    const seen = new Uint8Array(N2);
    for (let seed = 0; seed < N2; seed++) {
      if (bd[seed] !== v || seen[seed] || alive[seed]) continue;
      const n = chainAt(bd, seed, v);               // 洪泛整条链(顺便判重)
      const members = Array.from(DS_OUT.m.slice(0, n));
      for (const p of members) seen[p] = 1;
      if (members.some((p) => dead.has(p))) continue;   // 规则侧结论不翻案
      let sum = 0;
      for (const p of members) sum += ownership[p];
      const avg = sum / members.length;             // 链平均(比单点稳,不受邻点稀释)
      if (v === 1 ? avg < -deadT : avg > deadT) {
        for (const p of members) dead.add(p);
      }
    }
  }
  return Array.from(dead);
}
