/* ============================================================
 * AetherGo 规则核心 —— 棋盘状态、走子/撤销、合法性、禁全同
 *
 * 棋盘:N2 = 361 个交叉点,idx = 行×19 + 列;行 0 是上边,列 0 是左边。
 * 棋子:1 = 黑,2 = 白,0 = 空;行棋方 BLACK = 0 / WHITE = 1(黑先)。
 *
 * 规则:气尽提子(含整块)、禁自杀(能提子则不算)、position superko
 *      (禁全同:落子后局面在对局历史中出现过即非法,单劫是它的特例)、
 *      停一手永远合法,连续两手停即终局。
 *
 * 本文件与 src/scoring.js 共享同一套洪泛工作区(GRP/SEEN/LIBSEEN/stamp),
 * 供对方以 nextStamp()/groupInfo()/gSize() 等访问器使用 —— 各函数串行使用,
 * 互不嵌套。对外完整 API 由 src/engine.js 统一再导出。
 * ============================================================ */
import { N, N2, BLACK, EMPTY, PASS } from './protocol.js';

export { N, N2, BLACK, EMPTY, PASS };   // 供 engine.js facade 转出

export const stoneOf = (side) => side + 1;      // 行棋方 → 棋子值
export const sideOf = (st) => st - 1;           // 棋子值 → 行棋方

/* ==================== 邻接表 ==================== */

/* 4 邻居,-1 表示越界;NB_N[p] 是 p 的实际邻居数 */
export const NB = new Int32Array(N2 * 4).fill(-1);
export const NB_N = new Int8Array(N2);
for (let p = 0; p < N2; p++) {
  const r = (p / N) | 0, c = p % N;
  let n = 0;
  if (r > 0) NB[p * 4 + n++] = p - N;
  if (r < N - 1) NB[p * 4 + n++] = p + N;
  if (c > 0) NB[p * 4 + n++] = p - 1;
  if (c < N - 1) NB[p * 4 + n++] = p + 1;
  NB_N[p] = n;
}

/* ==================== 洪泛填充工作区 ==================== */

/* GRP 兼任「组队列」与「组格子输出」;SEEN / LIBSEEN 用递增 stamp 标记,
 * 免去每次清零 —— 这是全文件唯一的「共享暂存」,各函数串行使用互不嵌套。 */
export const GRP = new Int32Array(N2);
export const SEEN = new Int32Array(N2);
export const LIBSEEN = new Int32Array(N2);
let stamp = 0;
/** 取一个新 stamp(本轮标记的起点);scoring.js 的洪泛也走这里 */
export const nextStamp = () => ++stamp;

/* groupInfo 的输出:组大小 / 气数 / 当气数为 1 时那个唯一的气点 */
let GSIZE = 0, GLIBS = 0, GLIB_PT = -1;
export const gSize = () => GSIZE;
export const gLibs = () => GLIBS;
export const gLibPt = () => GLIB_PT;

/** 从 start 洪泛出整块棋:填 GRP[0..GSIZE) 与 GSIZE / GLIBS / GLIB_PT(气为 1 时) */
export function groupInfo(bd, start) {
  const st = ++stamp;
  const mine = bd[start];
  let head = 0, size = 0, libs = 0, libPt = -1;
  GRP[size++] = start; SEEN[start] = st;
  while (head < size) {
    const p = GRP[head++];
    for (let k = 0; k < NB_N[p]; k++) {
      const q = NB[p * 4 + k], v = bd[q];
      if (v === EMPTY) {
        if (LIBSEEN[q] !== st) { LIBSEEN[q] = st; libs++; libPt = q; }
      } else if (v === mine && SEEN[q] !== st) {
        SEEN[q] = st; GRP[size++] = q;
      }
    }
  }
  GSIZE = size; GLIBS = libs; GLIB_PT = libPt;
}

/* ==================== 局面状态 ==================== */

/* 劫点与撤销栈:make 压入旧劫点、登记被提子;unmake 逆序还原。
 * 与象棋引擎的 KSQ 一样是「当前棋盘」的模块级状态,
 * 换局面(newBoard / syncPosition)时必须重置。 */
let KO = -1;
let CAPBUF = new Int32Array(8192), CAPTOP = 0;      // 被提子格登记区
let KOST = new Int32Array(4096), KOTOP = -1;        // 旧劫点栈
const CAP_MAX = N2;                                  // 一手最多提 N2 子

const growCap = (need) => {
  while (CAPBUF.length < need) {
    const t = new Int32Array(CAPBUF.length * 2); t.set(CAPBUF); CAPBUF = t;
  }
};
const growKost = (need) => {
  while (KOST.length <= need) {
    const t = new Int32Array(KOST.length * 2); t.set(KOST); KOST = t;
  }
};

/* ==================== 禁全同(position superko) ==================== */

/* 局面键 = 盘上棋子的 64 位 Zobrist 异或(不含行棋方 —— positional 口径,
 * 与 KataGo KO_POSITIONAL 一致)。历史键栈随 make/unmake 压弹:
 * 键栈顶即当前局面键,syncPosition 时按盘面全量重算。
 * 落子后键在历史中出现过 → 该落子非法;单劫(simple ko)是它的特例。
 * KO 变量仍照算 —— 只作 UI 的「打劫提示」,不再是合法性依据。 */
const ZOB_HI = new Uint32Array(N2 * 2), ZOB_LO = new Uint32Array(N2 * 2);
{
  /* 固定种子的 xorshift32:键表确定,测试可复现 */
  let s = 0x9e3779b9;
  const rnd32 = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s;
  };
  for (let i = 0; i < N2 * 2; i++) { ZOB_LO[i] = rnd32(); ZOB_HI[i] = rnd32(); }
}

/* 历史键栈:HKEY[0] 是初始局面,之后每手(含停一手,positional 下键不变、
 * 原样再压一份,与 KataGo 口径一致)压一个键 */
let HKEY_HI = new Int32Array(1024), HKEY_LO = new Int32Array(1024), HTOP = 0;

/* 各点「曾有过棋子或曾落子」计数(make 里单调递增,unmake 不减 ——
 * 「曾经」不随撤销回退;syncPosition 重置)。没被占过的空点只要不提子,
 * 落子后局面必新 —— isLegal 用它跳过历史扫描(KataGo 同款剪枝) */
const EOC = new Uint8Array(N2);

const growHkey = (need) => {
  while (HKEY_HI.length <= need) {
    const t = new Int32Array(HKEY_HI.length * 2); t.set(HKEY_HI); HKEY_HI = t;
    const u = new Int32Array(HKEY_LO.length * 2); u.set(HKEY_LO); HKEY_LO = u;
  }
};

const pushKey = (hi, lo) => {
  growHkey(HTOP);
  HKEY_HI[HTOP] = hi; HKEY_LO[HTOP] = lo; HTOP++;
};

/** 键是否在对局历史中出现过(顺序扫描;键栈顶就是当前键,含它也无妨 ——
 *  当前键必已入栈,落子键异或了至少一枚子,不会撞上自己) */
function keyInHistory(hi, lo) {
  for (let i = 0; i < HTOP; i++) {
    if (HKEY_HI[i] === hi && HKEY_LO[i] === lo) return true;
  }
  return false;
}

/** 当前局面键(测试/调试用),形如 "hi,lo" 的字符串 */
export function positionKey() {
  return HTOP === 0 ? '0,0' : `${HKEY_HI[HTOP - 1]},${HKEY_LO[HTOP - 1]}`;
}

/* ==================== NN 特征支持(通道 6:禁着点掩码) ==================== */

/* isLegal 的提子探查会复用这块暂存(与 LCAP 分开,避免与合法查询互相踩) */
const SKC = new Int32Array(N2);

/** side 视角的禁全同掩码(含单劫点,不含自杀点 —— KataGo superKoBanned 口径)。
 *  out 为空则内部新开;返回 Uint8Array(N2),1 = 落子非法(局面重复/单劫)。
 *  这是 NN 特征第 6 通道的直接来源(见 src/nn/features.js)。 */
export function superkoBannedPoints(bd, side, out) {
  const mask = out ?? new Uint8Array(N2);
  mask.fill(0);
  if (KO >= 0) mask[KO] = 1;                     // simple-ko 点恒进通道 6
  const mine = side + 1, his = 2 - side;
  for (let p = 0; p < N2; p++) {
    if (bd[p] !== EMPTY || mask[p]) continue;    // 已标(劫点)或非空点
    bd[p] = mine;
    let capN = 0;
    for (let k = 0; k < NB_N[p]; k++) {
      const q = NB[p * 4 + k];
      if (bd[q] !== his) continue;
      groupInfo(bd, q);
      if (GLIBS === 0) {
        for (let i = 0; i < GSIZE; i++) { const c = GRP[i]; bd[c] = EMPTY; SKC[capN++] = c; }
      }
    }
    groupInfo(bd, p);
    let banned = false;
    if (GLIBS === 0) banned = false;             // 自杀不进通道 6
    else if (capN === 0 && EOC[p] === 0) banned = false;   // 处子点不提子:局面必新
    else {
      let hi = HKEY_HI[HTOP - 1] ^ ZOB_HI[p * 2 + side];
      let lo = HKEY_LO[HTOP - 1] ^ ZOB_LO[p * 2 + side];
      const ci = side ^ 1;
      for (let i = 0; i < capN; i++) {
        const c = SKC[i];
        hi ^= ZOB_HI[c * 2 + ci]; lo ^= ZOB_LO[c * 2 + ci];
      }
      banned = keyInHistory(hi, lo);
    }
    bd[p] = EMPTY;
    for (let i = 0; i < capN; i++) bd[SKC[i]] = his;
    if (banned) mask[p] = 1;
  }
  return mask;
}

/** 新棋盘:全空,无劫 */
export function newBoard() {
  const bd = new Int8Array(N2);
  syncPosition(bd);
  return bd;
}

/* ==================== 滚动盘面环(KataGo BoardHistory.recentBoards 同款) ====================
 * slot[head] 恒为当前盘面,slot[head-1]/slot[head-2] 即前一手/前二手 —— make 时写入
 * 新盘面,unmake 只回退指针(槽内旧值不擦,超出 RING_N 的不读)。征子历史通道
 * (features.js 15/16)从这里取前两手盘面,替代整局重演 —— 编码成本 O(盘面) 不随手数涨。
 * 前提:bd 只能经 make/unmake 变更、经 syncPosition 重置(本文件的一致性约定);
 * **读环只在「前进到达」的状态下可靠**(叶子编码都在下降途中 / 紧跟 replayMoves)。
 * 回退后换分支的读法必须先 ringRestore —— 搜索每批下降前做这件事。 */
const RING_K = 4;
const RING_BD = new Int8Array(RING_K * N2);
const RING_KO = new Int32Array(RING_K);
let RING_HEAD = 0, RING_N = 0;                 // RING_N = 已行棋手数(含 pass)

function ringPush(bd) {
  RING_HEAD = (RING_HEAD + 1) % RING_K;
  RING_BD.set(bd, RING_HEAD * N2);
  RING_KO[RING_HEAD] = KO;
  RING_N++;
}

/** k 手前(1=前一手,2=前二手)的盘面(只读视图);不足 k 手返回 null */
export function recentPrevBd(k) {
  if (RING_N < k) return null;
  const slot = (RING_HEAD - k + RING_K) % RING_K;
  return RING_BD.subarray(slot * N2, (slot + 1) * N2);
}

/** k 手前的单劫点(与 recentPrevBd 同一快照);不足 k 手返回 -1 */
export function recentPrevKo(k) {
  if (RING_N < k) return -1;
  return RING_KO[(RING_HEAD - k + RING_K) % RING_K];
}

/** 已行棋手数(含 pass)—— 环历史深度的权威判据 */
export const moveCount = () => RING_N;

/** 环快照(KataGo 每 playout 拷贝 rootHistory 的等价物):搜索在每批下降前恢复,
 * 使「回退后换分支再下降」读到的历史盘面不被上一分支的深层槽位污染。 */
export function ringSnapshot() {
  return { bd: RING_BD.slice(), ko: RING_KO.slice(), head: RING_HEAD, n: RING_N };
}

export function ringRestore(s) {
  RING_BD.set(s.bd); RING_KO.set(s.ko); RING_HEAD = s.head; RING_N = s.n;
}

/** 重置派生状态(劫点 + 撤销栈 + 历史键栈 + 盘面环);摆棋/改盘后必须调用 */
export function syncPosition(bd, ko = -1) {
  KO = ko; CAPTOP = 0; KOTOP = -1;
  let hi = 0, lo = 0;
  for (let p = 0; p < N2; p++) {
    const v = bd[p];
    EOC[p] = v ? 1 : 0;
    if (v) { hi ^= ZOB_HI[p * 2 + v - 1]; lo ^= ZOB_LO[p * 2 + v - 1]; }
  }
  HTOP = 0; pushKey(hi, lo);
  RING_HEAD = 0; RING_N = 0;
  RING_BD.set(bd, 0); RING_KO[0] = ko;
}

/** 从空盘按走法序列重演(Worker 用它还原 UI 的棋盘),返回轮到哪方;非法序列返回 -1 */
export function replayMoves(bd, moves, side = BLACK) {
  let s = side;
  for (let i = 0; i < moves.length; i++) {
    const mv = moves[i];
    if (mv !== PASS && !isLegal(bd, s, mv)) return -1;
    make(bd, mv, s);
    s ^= 1;
  }
  return s;
}

/* ==================== 走子与撤销 ==================== */

/**
 * 走子(假定已合法 —— UI / 搜索都先过 isLegal;Worker 重演也先验)。
 * 返回撤销令牌:PASS 返回 -1;落子返回 CAPTOP 基址×128 + 提子数,
 * unmake 靠它还原。side 是行棋方 —— 围棋棋盘上看不出轮谁走,必须显式传。
 */
export function make(bd, mv, side) {
  growKost(KOTOP + 2);
  KOST[++KOTOP] = KO;                            // 旧劫点入栈
  if (mv === PASS) {
    KO = -1;
    /* positional 禁全同:停一手不改盘面,键原样再压一份(与 KataGo 一致) */
    pushKey(HKEY_HI[HTOP - 1], HKEY_LO[HTOP - 1]);
    ringPush(bd);
    return -1;
  }
  growCap(CAPTOP + CAP_MAX);
  const base = CAPTOP;
  const n = tryPlay(bd, side, mv, true);
  if (n < 0) { KO = KOST[KOTOP--]; return -2; }  // 理论不可达:调用方须先验合法
  EOC[mv] = 1;
  /* 增量更新局面键:落下己子异或进、被提敌子异或出。
   * Zobrist 索引 p*2 + side(棋子值 = side + 1,黑 0 白 1 恰好当低位) */
  let hi = HKEY_HI[HTOP - 1] ^ ZOB_HI[mv * 2 + side];
  let lo = HKEY_LO[HTOP - 1] ^ ZOB_LO[mv * 2 + side];
  const ci = side ^ 1;
  for (let i = 0; i < n; i++) {
    const c = CAPBUF[base + i];
    hi ^= ZOB_HI[c * 2 + ci]; lo ^= ZOB_LO[c * 2 + ci];
    EOC[c] = 1;
  }
  pushKey(hi, lo);
  ringPush(bd);
  return base * 128 + n;
}

/** 撤销:还原棋子、劫点与登记区指针(与 make 严格对称) */
export function unmake(bd, mv, tok) {
  KO = KOST[KOTOP--];
  HTOP--;                                        // 弹出本手压入的局面键
  RING_HEAD = (RING_HEAD - 1 + RING_K) % RING_K; // 盘面环回退(槽内旧值不擦)
  RING_N--;
  if (mv === PASS) return;
  const n = tok % 128, base = (tok - n) / 128;
  const his = 3 - bd[mv];                        // 还原的是对方的子(1↔2)
  bd[mv] = EMPTY;
  for (let i = 0; i < n; i++) bd[CAPBUF[base + i]] = his;
  CAPTOP = base;
}

/** 撤销令牌里的提子数(UI 显示提子用);PASS 的令牌是 -1 */
export const capturedOf = (tok) => (tok < 0 ? 0 : tok % 128);

/* ==================== 落子核心 ==================== */

/**
 * 落子核心:非法返回 -1 且棋盘保持不变;合法则执行(含提子、更新劫点)
 * 返回提子数。record = true 时把被提的格子登记进 CAPBUF,供 unmake 还原。
 */
function tryPlay(bd, side, p, record) {
  if (bd[p] !== EMPTY || p === KO) return -1;
  const mine = side + 1, his = 2 - side;
  bd[p] = mine;
  let capN = 0, capCell = -1;
  for (let k = 0; k < NB_N[p]; k++) {
    const q = NB[p * 4 + k];
    if (bd[q] !== his) continue;
    groupInfo(bd, q);
    if (GLIBS === 0) {
      for (let i = 0; i < GSIZE; i++) {
        const c = GRP[i];
        bd[c] = EMPTY;
        if (record) CAPBUF[CAPTOP++] = c;
        if (capN === 0) capCell = c;             // 恰提一子时,它就是劫点
        capN++;
      }
    }
  }
  /* 自杀:没提到子且自己的新组无气。提到子则被提点必然成了新组的气。 */
  groupInfo(bd, p);
  if (GLIBS === 0) { bd[p] = EMPTY; return -1; }
  /* simple ko:恰提一子、落下的子自成一块且只有一口气、这口气就是被提点
   * —— 此时对方立即回提会复原局面,禁一手。 */
  KO = (capN === 1 && GSIZE === 1 && GLIBS === 1 && GLIB_PT === capCell) ? capCell : -1;
  return capN;
}

/* isLegal 提子探查的暂存:探查中被清掉的敌子(还原用)。
 * 与 CAPBUF 分开 —— isLegal 不推进 CAPTOP,不干扰在途 make 令牌。 */
const LCAP = new Int32Array(N2);

/**
 * p 对 side 是否合法落子点:空点、非劫点、非自杀(能提子则不算自杀)、
 * 且落子后局面不与对局历史重复(禁全同)。
 * isLegal 只读棋盘(临时落子/提子后复原),不产生副作用。
 */
export function isLegal(bd, side, p) {
  if (p === PASS) return true;
  if (bd[p] !== EMPTY || p === KO) return false;
  const mine = side + 1, his = 2 - side;
  bd[p] = mine;
  let capN = 0;
  /* 完整枚举提子(不能见好就收):禁全同要的是落子后的完整局面 */
  for (let k = 0; k < NB_N[p]; k++) {
    const q = NB[p * 4 + k];
    if (bd[q] !== his) continue;
    groupInfo(bd, q);
    if (GLIBS === 0) {
      for (let i = 0; i < GSIZE; i++) {
        const c = GRP[i];
        bd[c] = EMPTY;
        LCAP[capN++] = c;
      }
    }
  }
  groupInfo(bd, p);
  let legal;
  if (GLIBS === 0) legal = false;               // 自杀:无气且提不到子
  else if (capN === 0 && EOC[p] === 0) legal = true;  // 处子点不提子:局面必新,免扫历史
  else {
    /* 禁全同:落子后的局面键不得在历史键栈中出现过 */
    let hi = HKEY_HI[HTOP - 1] ^ ZOB_HI[p * 2 + side];
    let lo = HKEY_LO[HTOP - 1] ^ ZOB_LO[p * 2 + side];
    const ci = side ^ 1;
    for (let i = 0; i < capN; i++) {
      const c = LCAP[i];
      hi ^= ZOB_HI[c * 2 + ci]; lo ^= ZOB_LO[c * 2 + ci];
    }
    legal = !keyInHistory(hi, lo);
  }
  bd[p] = EMPTY;
  for (let i = 0; i < capN; i++) bd[LCAP[i]] = his;
  return legal;
}

/** 合法落子点数组(0..360;停一手永远合法,不进列表,由 UI / 搜索单独处理) */
export function genLegal(bd, side) {
  const out = [];
  for (let p = 0; p < N2; p++) if (isLegal(bd, side, p)) out.push(p);
  return out;
}

/* ==================== 测试钩子 ==================== */

export function boardToArray(bd) { return Array.from(bd); }
export function arrayToBoard(arr, ko = -1) {
  const bd = new Int8Array(N2); bd.set(arr); syncPosition(bd, ko); return bd;
}
/** 当前劫点(测试用;正常走子经 make/unmake 自动维护) */
export function koPoint() { return KO; }
