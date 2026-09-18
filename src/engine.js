/* ============================================================
 * AetherGo —— 9×9 围棋引擎(规则 + 数子 + MCTS)
 *
 * 棋盘:81 个交叉点,idx = 行×9 + 列;行 0 是上边,列 0 是左边。
 * 棋子:1 = 黑,2 = 白,0 = 空;行棋方 BLACK = 0 / WHITE = 1(黑先),
 *      棋子 = 行棋方 + 1,所以 sideOf(子) = 子 − 1。
 * 走法:交叉点 0..80,PASS = 81 表示停一手 —— 就一种编码,UI 与 Worker 共用。
 *
 * 规则:气尽提子(含整块)、禁自杀、simple ko(单劫:提一子后对方下一手
 *      不得立即回提)、停一手永远合法,连续两手停即终局数子。
 * 计分:中国规则数子法(子 + 空,黑贴 5.5 目)—— 引擎对局里死子会被
 *      自然提掉,数子即结果;UI 侧的死子标注见 docs/ROADMAP.md。
 *
 * 搜索:MCTS(UCT)。围棋分支因子大、静态评估难写,alpha-beta 在这里跑不动,
 *      蒙特卡洛树搜索是正路:树内 UCB1 选点 → 每次到达扩展一个未试手 →
 *      随机演棋到终局数子 → 沿路径回传胜负。演棋策略刻意保持很轻
 *      (随机合法点 + 不填自己的真眼),棋力靠访问次数堆。
 *
 * 对外入口:
 *   newBoard() / replayMoves() / genLegal() / isLegal()
 *   make(bd, mv, side) / unmake(bd, mv, tok)
 *   evaluate() / scoreGame() / moveToText()   —— UI 侧规则、数子、记谱
 *   searchBest(bd, side, opt)                 —— Worker 侧搜索
 *   LEVELS                                    —— 难度档(演棋局数为主、墙上时间为辅)
 *
 * 本文件不碰 DOM、不 import 任何库 —— 浏览器、Worker、Node 通用。
 * ============================================================ */

export const BLACK = 0, WHITE = 1;
export const EMPTY = 0;
/** 停一手;走法编码 0..80 是交叉点,81 是停一手 */
export const PASS = 81;
/** 黑贴目(中国规则 9 路常用 5.5) */
export const KOMI = 5.5;

export const stoneOf = (side) => side + 1;      // 行棋方 → 棋子值
export const sideOf = (st) => st - 1;           // 棋子值 → 行棋方

/* ==================== 邻接表 ==================== */

/* 4 邻居,-1 表示越界;NB_N[p] 是 p 的实际邻居数 */
const NB = new Int32Array(81 * 4).fill(-1);
const NB_N = new Int8Array(81);
for (let p = 0; p < 81; p++) {
  const r = (p / 9) | 0, c = p % 9;
  let n = 0;
  if (r > 0) NB[p * 4 + n++] = p - 9;
  if (r < 8) NB[p * 4 + n++] = p + 9;
  if (c > 0) NB[p * 4 + n++] = p - 1;
  if (c < 8) NB[p * 4 + n++] = p + 1;
  NB_N[p] = n;
}

/* ==================== 洪泛填充工作区 ==================== */

/* GRP 兼任「组队列」与「组格子输出」;SEEN / LIBSEEN 用递增 stamp 标记,
 * 免去每次清零 —— 这是全文件唯一的「共享暂存」,各函数串行使用互不嵌套。 */
const GRP = new Int32Array(81);
const SEEN = new Int32Array(81);
const LIBSEEN = new Int32Array(81);
let stamp = 0;

/* groupInfo 的输出:组大小 / 气数 / 当气数为 1 时那个唯一的气点 */
let GSIZE = 0, GLIBS = 0, GLIB_PT = -1;

/** 从 start 洪泛出整块棋:填 GRP[0..GSIZE) 与 GSIZE / GLIBS / GLIB_PT(气为 1 时) */
function groupInfo(bd, start) {
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
const CAP_MAX = 81;                                  // 一手最多提 81 子

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

/** 新棋盘:全空,无劫 */
export function newBoard() {
  const bd = new Int8Array(81);
  syncPosition(bd);
  return bd;
}

/** 重置派生状态(劫点 + 撤销栈);摆棋/改盘后必须调用 */
export function syncPosition(bd, ko = -1) {
  KO = ko; CAPTOP = 0; KOTOP = -1;
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

/* expand() 里 make 的令牌经模块变量带回(避免 expand 返回二元组) */
let N_TOK_PENDING = 0;

/**
 * 走子(假定已合法 —— UI / 搜索都先过 isLegal;Worker 重演也先验)。
 * 返回撤销令牌:PASS 返回 -1;落子返回 CAPTOP 基址×128 + 提子数,
 * unmake 靠它还原。side 是行棋方 —— 围棋棋盘上看不出轮谁走,必须显式传。
 */
export function make(bd, mv, side) {
  growKost(KOTOP + 2);
  KOST[++KOTOP] = KO;                            // 旧劫点入栈
  if (mv === PASS) { KO = -1; return -1; }
  growCap(CAPTOP + CAP_MAX);
  const base = CAPTOP;
  const n = tryPlay(bd, side, mv, true);
  if (n < 0) { KO = KOST[KOTOP--]; return -2; }  // 理论不可达:调用方须先验合法
  return base * 128 + n;
}

/** 撤销:还原棋子、劫点与登记区指针(与 make 严格对称) */
export function unmake(bd, mv, tok) {
  KO = KOST[KOTOP--];
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

/** p 是不是行棋方的「真眼」(4 邻全是己方子)。
 *  只用于演棋时避开自填眼,不是完整的真眼判定(斜角假眼不查)。 */
function isOwnEye(bd, side, p) {
  const mine = side + 1;
  for (let k = 0; k < NB_N[p]; k++) if (bd[NB[p * 4 + k]] !== mine) return false;
  return true;
}

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

/**
 * p 对 side 是否合法落子点:空点、非劫点、非自杀(能提子则不算自杀)。
 * isLegal 只读棋盘(临时落子后复原),不产生副作用。
 */
export function isLegal(bd, side, p) {
  if (p === PASS) return true;
  if (bd[p] !== EMPTY || p === KO) return false;
  const mine = side + 1, his = 2 - side;
  bd[p] = mine;
  let ok = false;
  for (let k = 0; k < NB_N[p]; k++) {
    const q = NB[p * 4 + k];
    if (bd[q] !== his) continue;
    groupInfo(bd, q);
    if (GLIBS === 0) { ok = true; break; }        // 能提子:合法,不必再查
  }
  if (!ok) { groupInfo(bd, p); ok = GLIBS > 0; }  // 不能提子:查自己有无气
  bd[p] = EMPTY;
  return ok;
}

/** 合法落子点数组(0..80;停一手永远合法,不进列表,由 UI / 搜索单独处理) */
export function genLegal(bd, side) {
  const out = [];
  for (let p = 0; p < 81; p++) if (isLegal(bd, side, p)) out.push(p);
  return out;
}

/* ==================== 数子(中国规则) ==================== */

let EV_BS = 0, EV_WS = 0, EV_BT = 0, EV_WT = 0;

/** 数子:黑 = 黑子 + 黑空,白 = 白子 + 白空(空点区域按边界颜色归属,双方都贴边算公气) */
function areaScore(bd) {
  const st = ++stamp;
  let bs = 0, ws = 0, bt = 0, wt = 0;
  for (let p = 0; p < 81; p++) {
    const v = bd[p];
    if (v === 1) bs++; else if (v === 2) ws++;
  }
  for (let p = 0; p < 81; p++) {
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

/** 静态评估:黑方视角的目差(子 + 空 − 贴目)。演棋终盘与终局都用它。 */
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

/* ==================== MCTS ==================== */

const UCT_C = 1.0;                // UCB1 探索系数(胜率尺度 0..1)
const MAXPLY = 160;               // 树内下降深度上限(保底,实际远到不了)
const PLAYOUT_CAP = 162;          // 单次随机演棋手数上限(2×81),到点直接数子
const PASS_MIN_EMPTIES = 25;      // 空点多于此数时树内不考虑停一手,防早早乱停
const ATTEMPTS = 48;              // 演棋选点的随机尝试上限,找不到就停一手

/* 树节点池:按列拆 typed array(与象棋引擎同一套「零分配」思路),
 * 节点 id 即下标,0 恒为根;子用兄弟链挂(省二维数组)。 */
let N_MOVE, N_PARENT, N_CHILD, N_SIB, N_VISITS, N_WINS, N_SIDE, N_TERM, N_USTART, N_UCOUNT;
let nCap = 0, nNodes = 0;

/* 未试手 arena:每个节点首次到达时把合法着法拷进来,逐个弹出扩展 */
let UNTRIED = new Int32Array(1 << 16), uTop = 0;

/* 着法缓冲:树展开时按层分槽复用(展开是瞬时动作,拷进 arena 后即弃) */
const MB = new Int32Array(82 * (MAXPLY + 4));
/* 下降路径:记录 make 过的节点与令牌,迭代结束后统一 unmake */
const DSTACK = new Int32Array(MAXPLY + 8);
const DTOK = new Int32Array(MAXPLY + 8);

function growNodes() {
  const n = nCap ? nCap * 2 : 1024;
  /* 新数组必须按新容量 n 分配(按 old.length 分配等于没扩,越界写入被静默丢弃) */
  const cp = (old, T) => {
    if (!old) return new T(n);
    const t = new T(n); t.set(old); return t;
  };
  N_MOVE = cp(N_MOVE, Int32Array); N_PARENT = cp(N_PARENT, Int32Array);
  N_CHILD = cp(N_CHILD, Int32Array); N_SIB = cp(N_SIB, Int32Array);
  N_VISITS = cp(N_VISITS, Int32Array);
  N_WINS = cp(N_WINS, Float64Array); N_SIDE = cp(N_SIDE, Int8Array);
  N_TERM = cp(N_TERM, Int8Array);
  N_USTART = cp(N_USTART, Int32Array); N_UCOUNT = cp(N_UCOUNT, Int32Array);
  nCap = n;
}

const growUntried = (need) => {
  while (UNTRIED.length < need) {
    const t = new Int32Array(UNTRIED.length * 2); t.set(UNTRIED); UNTRIED = t;
  }
};

function newNode(mv, parent, side) {
  if (nNodes >= nCap) growNodes();
  const i = nNodes++;
  N_MOVE[i] = mv; N_PARENT[i] = parent; N_CHILD[i] = -1; N_SIB[i] = -1;
  N_VISITS[i] = 0; N_WINS[i] = 0; N_SIDE[i] = side; N_TERM[i] = 0;
  N_USTART[i] = -1; N_UCOUNT[i] = 0;
  return i;
}

/** 首次到达节点:生成合法手列表拷进 arena(空点 ≤ 阈值、无点可下或上一手是停时补上停一手) */
function initUntried(cur, bd, ply) {
  const side = N_SIDE[cur], base = ply * 82;
  let n = 0, empties = 0;
  for (let p = 0; p < 81; p++) {
    if (bd[p] === EMPTY) {
      empties++;
      if (isLegal(bd, side, p)) MB[base + n++] = p;
    }
  }
  if (empties <= PASS_MIN_EMPTIES || n === 0 || N_MOVE[cur] === PASS) MB[base + n++] = PASS;
  growUntried(uTop + n);
  const st = uTop; uTop += n;
  for (let i = 0; i < n; i++) UNTRIED[st + i] = MB[base + i];
  N_USTART[cur] = st; N_UCOUNT[cur] = n;
}

/** 弹一个随机未试手,make 后挂成子节点返回;要求调用时 UCOUNT > 0 */
function expand(cur, bd) {
  const uc = N_UCOUNT[cur], st = N_USTART[cur];
  const k = st + ((Math.random() * uc) | 0);
  const mv = UNTRIED[k];
  UNTRIED[k] = UNTRIED[st + uc - 1];
  N_UCOUNT[cur] = uc - 1;
  const side = N_SIDE[cur];
  N_TOK_PENDING = make(bd, mv, side);
  const child = newNode(mv, cur, side ^ 1);
  N_SIB[child] = N_CHILD[cur]; N_CHILD[cur] = child;
  if (mv === PASS && N_MOVE[cur] === PASS) N_TERM[child] = 1;   // 连续两手停 = 终局
  return child;
}

/** 随机演棋:从当前局面(轮 side)随机下到终局,返回黑方胜负(1 / 0 / 0.5) */
function playout(bd, side, komi) {
  let lastPass = false;
  for (let t = 0; t < PLAYOUT_CAP; t++) {
    let played = false;
    for (let a = 0; a < ATTEMPTS; a++) {
      const p = (Math.random() * 81) | 0;
      if (bd[p] !== EMPTY || p === KO || isOwnEye(bd, side, p)) continue;
      if (tryPlay(bd, side, p, false) < 0) continue;
      played = true; break;
    }
    if (!played) {                               // 找不到合法点:停一手
      if (lastPass) break;                       // 连续两手停 → 终局
      lastPass = true; KO = -1; side ^= 1; continue;
    }
    lastPass = false; side ^= 1;
  }
  const m = evaluate(bd, komi);
  return m > 0 ? 1 : m < 0 ? 0 : 0.5;
}

const winFromMargin = (m) => (m > 0 ? 1 : m < 0 ? 0 : 0.5);

/** 演棋入口:在棋盘**副本**上演(随机手不撤销,绝不能污染真盘),劫点状态用完即还 */
function playoutFrom(bd, side, komi) {
  const pb = bd.slice();
  const savedKo = KO;
  const w = playout(pb, side, komi);
  KO = savedKo;
  return w;
}

/** 一次迭代:下降 → 扩展 → 演棋 → 回传 → 撤销路径 */
function iterate(bd, komi) {
  let cur = 0, ply = 0, dTop = 0;
  /* 下降:只穿过已展开完且非终局的节点,UCB1 选最优子 */
  for (;;) {
    if (N_TERM[cur]) break;                              // 终局节点:数子即可
    if (N_USTART[cur] === -1) { initUntried(cur, bd, ply); break; }  // 首次到达
    if (N_UCOUNT[cur] > 0) break;                        // 还有未试手,就地展开
    if (N_CHILD[cur] === -1 || ply >= MAXPLY) break;     // 保底:当叶子演棋
    let best = -1, bestV = -1;
    const logN = Math.log(N_VISITS[cur]);
    for (let c = N_CHILD[cur]; c !== -1; c = N_SIB[c]) {
      const v = N_VISITS[c];
      const ucb = N_WINS[c] / v + UCT_C * Math.sqrt(logN / v);
      if (ucb > bestV) { bestV = ucb; best = c; }
    }
    cur = best;
    DTOK[dTop] = make(bd, N_MOVE[cur], N_SIDE[cur] ^ 1);
    DSTACK[dTop++] = cur;
    ply++;
  }
  /* 扩展:就地弹一个未试手挂成子节点 */
  let leaf = cur;
  if (!N_TERM[cur] && N_UCOUNT[cur] > 0) {
    leaf = expand(cur, bd);
    DTOK[dTop] = N_TOK_PENDING; DSTACK[dTop++] = leaf;
    ply++;
  }
  /* 演棋(终局节点直接数子) */
  const blackWin = N_TERM[leaf] ? winFromMargin(evaluate(bd, komi))
                                 : playoutFrom(bd, N_SIDE[leaf], komi);
  /* 回传:节点记的是「走进该节点那一方」的胜局数 */
  for (let n = leaf; n !== -1; n = N_PARENT[n]) {
    N_VISITS[n]++;
    N_WINS[n] += (N_SIDE[n] ^ 1) === BLACK ? blackWin : 1 - blackWin;
  }
  /* 撤销整条路径 */
  while (dTop > 0) {
    const n = DSTACK[--dTop];
    unmake(bd, N_MOVE[n], DTOK[dTop]);
  }
}

/* ==================== 难度档 ==================== */

/* 演棋局数为主(设备无关、可复现),墙上时间为兜底。
 * jitter:低难度在「最优胜率 − jitter」的根节点子集里随机挑一个。 */
export const LEVELS = [
  { id: 'easy', name: '初级', desc: '400 次演棋', playouts: 400, ms: 500, jitter: 0.25 },
  { id: 'normal', name: '中级', desc: '2.5k 次演棋', playouts: 2500, ms: 1500, jitter: 0 },
  { id: 'hard', name: '高级', desc: '9k 次演棋', playouts: 9000, ms: 5000, jitter: 0 },
  { id: 'master', name: '大师', desc: '24k 次演棋', playouts: 24000, ms: 12000, jitter: 0 },
];
export const DEFAULT_LEVEL = 2;

/**
 * MCTS 搜索最佳着法。
 * opt: { playouts, ms, jitter, komi, lastMove,
 *        onProgress({ visits, move, winRate, ms }) }
 * 返回 { move, winRate, visits, nodes, ms, only }
 * winRate 是行棋方视角的胜率(0..1);move = PASS 表示停一手。
 */
export function searchBest(bd, side, opt = {}) {
  const t0 = Date.now();
  const komi = opt.komi ?? KOMI;
  const budget = opt.playouts ?? 3000;
  const msBudget = opt.ms ?? 3000;
  nNodes = 0; uTop = 0;

  const root = newNode(-1, -1, side);
  initUntried(root, bd, 0);

  /* 唯一选择:不必搜(与象棋引擎同款约定) */
  if (N_UCOUNT[root] === 1) {
    const mv = UNTRIED[N_USTART[root]];
    return { move: mv, winRate: 0.5, visits: 0, nodes: 0, ms: Date.now() - t0, only: true };
  }

  let iters = 0;
  while (iters < budget) {
    if ((iters & 63) === 63 && Date.now() - t0 > msBudget) break;
    iterate(bd, komi);
    iters++;
    if (opt.onProgress && (iters & 255) === 0) {
      const r = rootResult();
      opt.onProgress({ visits: iters, move: r.move, winRate: r.winRate, ms: Date.now() - t0 });
    }
  }

  let r = rootResult();
  /* 低难度:在最优解附近的子集里随机挑,弱得可控(只考虑访问 ≥4 的子,过滤噪声) */
  const jitter = opt.jitter ?? 0;
  if (jitter > 0) {
    const pool = [];
    for (let c = N_CHILD[0]; c !== -1; c = N_SIB[c]) {
      if (N_VISITS[c] >= 4 && N_WINS[c] / N_VISITS[c] >= r.winRate - jitter) pool.push(c);
    }
    if (pool.length > 1) {
      const c = pool[(Math.random() * pool.length) | 0];
      r = { move: N_MOVE[c], winRate: N_WINS[c] / N_VISITS[c] };
    }
  }

  return {
    move: r.move, winRate: r.winRate, visits: iters, nodes: iters,
    ms: Date.now() - t0, only: false,
  };
}

/** 根节点按访问次数选最佳(并列时取胜率高者) */
function rootResult() {
  let best = -1, bv = -1, bw = -1;
  for (let c = N_CHILD[0]; c !== -1; c = N_SIB[c]) {
    const v = N_VISITS[c], w = N_WINS[c] / v;
    if (v > bv || (v === bv && w > bw)) { bv = v; bw = w; best = c; }
  }
  return best < 0 ? { move: PASS, winRate: 0.5 } : { move: N_MOVE[best], winRate: bw };
}

/* ==================== 记谱 ==================== */

/* 坐标:列 A~J(跳过 I)+ 行 1~9(下边为 1)。围棋记谱不依赖盘面,
 * 但与象棋引擎保持同签名(bd, mv)。 */
const COLS = 'ABCDEFGHJ';

export function moveToText(bd, mv) {
  if (mv === PASS) return '停一手';
  return COLS[mv % 9] + String(9 - ((mv / 9) | 0));
}

/* ==================== 测试钩子 ==================== */

export function boardToArray(bd) { return Array.from(bd); }
export function arrayToBoard(arr, ko = -1) {
  const bd = new Int8Array(81); bd.set(arr); syncPosition(bd, ko); return bd;
}
/** 当前劫点(测试用;正常走子经 make/unmake 自动维护) */
export function koPoint() { return KO; }
