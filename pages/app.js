/* ============================================================
 * 围棋在线对弈页 —— UI 移植自 AetherWebOS 的围棋应用
 * (js/apps/go/index.js),布局与交互保持一致:
 * 顶栏「新对局 / 难度 / 停一手 / 数子 / 形势 / 人机 / 换边 / 悔棋」
 * + 中央棋盘 + 底栏左侧行棋状态与提子数、右侧等宽字体引擎搜索信息。
 *
 * UI 与引擎分离:引擎在 src/(rules/scoring/nn/nn-worker),UI 只经
 * Worker 消息问引擎(难度表、合法着法、劫点、数子、形势判断),
 * 契约常量(N/PASS/记谱)取自 src/protocol.js 唯一共享层。
 * 本文件是对局控制器;通用 DOM 工具在 ./dom.js,棋盘绘制在 ./board.js。
 * UI 持有的唯一对局状态是**走法序列**(交叉点 0..360 或 PASS=361)。
 * ============================================================ */
import { $, el, icon, toast, showDialog, countDlg, countMsgEl, countExitBtn } from './dom.js';
import { CS, X, Y, boardSvg } from './board.js';
import { N, PASS, BLACK, WHITE, formatMove } from '../src/protocol.js';

/* ==================== 主题(浅 / 深双主题,记在 localStorage)==================== */
const themeBtn = $('#themeBtn');
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeBtn.replaceChildren(icon(theme === 'dark' ? 'sun' : 'moon'));
  try { localStorage.setItem('aether-pages-theme', theme); } catch {}
}
applyTheme((() => {
  try { return localStorage.getItem('aether-pages-theme') || 'dark'; } catch { return 'dark'; }
})());
themeBtn.addEventListener('click', () =>
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

const setTitle = (t) => { $('#winTitle').textContent = t; };

/* ==================== 数子窗口(行为接线;元素来自 dom.js)====================
 * 独立明细窗(子 / 空 / 贴目 / 死子分布),终局后兼作结算窗。
 * 数子模式下棋盘始终可点(点棋子切死/活),窗口开着时回包落地会实时刷新。 */
let countPanelWanted = false;          // 等数子回包落地后开窗
function countDetailText(s) {
  const dBlack = s.dead.filter((p) => board[p] === 1).length;
  const dWhite = s.dead.length - dBlack;
  const win = s.margin > 0 ? '黑胜' : s.margin < 0 ? '白胜' : '和棋';
  return `黑:子 ${s.blackStones} + 空 ${s.blackTerritory} = ${s.black}\n`
    + `白:子 ${s.whiteStones} + 空 ${s.whiteTerritory} + 贴目 ${s.komi} = ${s.white}\n`
    + `${win === '和棋' ? '和棋' : win + ' ' + Math.abs(s.margin).toFixed(1) + ' 目'}`
    + (s.dead.length ? ` · 死子:黑 ${dBlack} 颗 / 白 ${dWhite} 颗` : '')
    + `\n${gameOver ? '点击棋子可切换死/活,数子实时更新' : '数子随时可做;点击棋子切换死/活'}`;
}
function openCountDlg() {
  countExitBtn.style.display = gameOver ? 'none' : '';
  if (!countScore || countScore.blackStones === undefined) return;   // 回包未到,落地时再开
  countMsgEl.textContent = countDetailText(countScore);
  if (!countDlg.open) countDlg.showModal();
}
function openCountPanel() { countPanelWanted = true; openCountDlg(); }
$('#countOkBtn').addEventListener('click', () => { countDlg.close(); countPanelWanted = false; });
countExitBtn.addEventListener('click', () => {
  countDlg.close(); countPanelWanted = false;
  if (!gameOver) exitCount();
});
countDlg.addEventListener('click', (e) => { if (e.target === countDlg) { countDlg.close(); countPanelWanted = false; } });

/* ==================== 对局(逻辑同 webos 围棋应用)==================== */

const sideName = (s) => (s === BLACK ? '黑方' : '白方');
const fmtRate = (w) => Math.round(w * 100) + '%';
const fmtVisits = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v));
/** 记谱(显示用):交叉点 → 列字母 + 行号;PASS → 短记「停」 */
const moveText = (mv) => (mv === PASS ? '停' : formatMove(mv));
/** 点的 4 邻居(UI 侧数子模式圈同色连通块用) */
function neighborsOf(q) {
  const r = (q / N) | 0, c = q % N, out = [];
  if (r > 0) out.push(q - N);
  if (r < N - 1) out.push(q + N);
  if (c > 0) out.push(q - 1);
  if (c < N - 1) out.push(q + 1);
  return out;
}

const appEl = $('#app');

let board = new Array(N * N).fill(0);   // 引擎棋盘(state 回包驱动;0 空 / 1 黑 / 2 白)
let legal = new Set();               // 行棋方合法着法(state 回包;点击校验以它为准)
let ko = -1;                         // 劫点(回包;提示「先找劫材」用)
let captures = [0, 0];               // 黑提 / 白提(state 回包)
let turn = BLACK;                    // 黑先
let humanSide = BLACK;               // 玩家执子方(换边可改);不翻盘,坐标恒定
let hist = [];                       // 走法序列 —— UI 持有的唯一对局状态
let lastMove = null;
let gameOver = false;
let vsAI = true;
let searching = false;
let levels = [];                     // 难度表由**引擎自报**({type:'levels'})
let levelIdx = 0;
let countMode = false;               // 数子模式:点棋子切换死/活,实时重算
let deadOverride = null;             // 手改死子点列表(null = 引擎自动判定)
let deadShown = new Set();           // 当前标示的死子点(渲染用)
let countScore = null;               // 数子模式最新结果
let endScoreInfo = null;             // 终局时 state 回包的数子(结算弹窗兜底)
let endResigned = false;             // 终局方式(中盘认输 / 双停数子)
let scoreSeq = 0;                    // 数子请求序号(过期回包丢弃)
let estimateOn = false;              // 形势判断开关
let ownership = null;                // 形势判断 ownership 图(渲染用)
let estSeq = 0;                      // 形势请求序号

const aiSide = () => humanSide ^ 1;
const lvName = () => levels[levelIdx]?.name ?? '—';

/* 终局弹窗缓冲(600ms):终局画面先落地,给玩家一点反应时间再弹结算。
 * 缓冲期里的新对局 / 悔棋 / 换边 / 人机切换都调 cancelEndDlg 取消 ——
 * 不然这些操作之后还会蹦出上一局的结算框。 */
const END_DLG_MS = 600;
let endDlgTimer = 0;
const cancelEndDlg = () => { clearTimeout(endDlgTimer); endDlgTimer = 0; };
const popEndDlg = (show) => {
  cancelEndDlg();
  endDlgTimer = setTimeout(() => { endDlgTimer = 0; show(); }, END_DLG_MS);
};

const statusL = el('span', {}, '黑方行棋');
const infoL = el('span', {
  class: 'mono', style: { fontSize: '11px' },
  title: '引擎搜索信息(胜率/目差是 AI 视角;目差来自根局面 NN 网端 lead 头)',
}, '');
const layerEl = el('div', { class: 'go-layer' });
const boardEl = el('div', { class: 'go-board' }, layerEl);
const fitWrap = el('div', { class: 'fit-wrap' }, boardEl);

/** 棋盘按可用空间等比缩放(棋盘内部是固定像素布局) */
function fitBoard() {
  const body = appEl.querySelector('.app-body');
  if (!body) return;
  const w = body.clientWidth - 24, h = body.clientHeight - 24;
  const bw = boardEl.offsetWidth, bh = boardEl.offsetHeight;
  if (!bw || !bh) return;
  fitWrap.style.transform = `scale(${Math.min(1, w / bw, h / bh)})`;
}

/* ---------- 渲染(全部基于最近一次 state 回包的缓存) ---------- */
function render() {
  layerEl.innerHTML = boardSvg();
  /* 虚影与可点光标只在「轮到玩家」时出现 */
  const humanTurn = !gameOver && !countMode && (!vsAI || turn === humanSide);
  boardEl.classList.toggle('turn-b', humanTurn && turn === BLACK);
  boardEl.classList.toggle('turn-w', humanTurn && turn === WHITE);
  const hint = humanTurn ? legal : null;
  for (let p = 0; p < N * N; p++) {
    const btn = el('button', {
      class: 'go-pt' + (hint && !board[p] && hint.has(p) ? ' can' : '')
        + (countMode && board[p] ? ' countable' : ''),
      style: { left: X(p % N) + 'px', top: Y((p / N) | 0) + 'px' },
      dataset: { i: String(p) },
      onClick: () => onPoint(p),
    });
    if (board[p]) {
      const st = el('div', {
        class: `go-stone ${board[p] === 1 ? 'black' : 'white'}${p === lastMove ? ' last' : ''}${countMode && deadShown.has(p) ? ' dead' : ''}`,
      });
      if (p === lastMove) st.classList.add('drop');
      btn.append(st);
    } else if (hint) {
      btn.append(el('div', { class: 'go-ghost' }));
    }
    /* 形势判断覆盖层(Lizzie/KaTrain 同款方块热图):
     * 颜色 = 归属方(黑/白),深浅与大小都编码归属强度 —— 越大越实,越小越虚。 */
    if (ownership && !countMode) {
      const v = ownership[p], a = Math.abs(v);
      if (a > 0.06) {
        const size = Math.round(CS * Math.min(0.62, 0.18 + 0.44 * Math.min(a, 1)));
        btn.append(el('div', {
          class: 'go-own ' + (v > 0 ? 'b' : 'w'),
          style: {
            width: size + 'px', height: size + 'px',
            opacity: String(0.32 + 0.42 * Math.min(a, 1)),
          },
        }));
      }
    }
    layerEl.append(btn);
  }
  passBtn.disabled = !humanTurn;
  estBtn.disabled = countMode || searching;
  countBtn.disabled = searching || (vsAI && !gameOver && turn !== humanSide);
  countBtn.replaceChildren(gameOver ? '结果' : countMode ? '继续对局' : '数子');
}

function updateStatus() {
  if (countMode) {
    if (!countScore) { statusL.textContent = '数子中…'; return; }
    const m = countScore.margin;
    const res = m > 0 ? `黑胜 ${m.toFixed(1)} 目` : m < 0 ? `白胜 ${(-m).toFixed(1)} 目` : '和棋';
    statusL.textContent = `数子:黑 ${countScore.black} · 白 ${countScore.white} —— ${res}(点棋子切换死/活)`;
    return;
  }
  if (gameOver) return;
  const caps = `黑提 ${captures[BLACK]} · 白提 ${captures[WHITE]}`;
  statusL.textContent = `${sideName(turn)}行棋 · ${caps}`;
  const last = hist.length ? ` · 上一手 ${moveText(hist[hist.length - 1])}` : '';
  setTitle(`围棋 — ${sideName(turn)}行棋${last}`);
}

/* ---------- 落子:合法性以缓存 state 为准,走子 = 改序列 + 再问一次引擎 ---------- */
function onPoint(p) {
  if (countMode) { toggleDead(p); return; }
  if (gameOver || board[p] || statePending) return;
  if (vsAI && turn !== humanSide) return;    // AI 回合/思考中不响应点击
  if (!legal.has(p)) {
    toast('围棋:' + (p === ko
      ? '打劫:需先在别处找一手劫材'
      : '禁着点:落子后无气(自杀)'));
    return;
  }
  doMove(p);
}

/** 数子模式:点棋子 = 整块切死/活,重发数子请求(deadOverride 为空 = 引擎自动判定) */
function toggleDead(p) {
  if (!board[p]) return;
  const v = board[p], group = [], seen = new Set([p]), stack = [p];
  while (stack.length) {
    const q = stack.pop(); group.push(q);
    for (const nb of neighborsOf(q)) {
      if (board[nb] === v && !seen.has(nb)) { seen.add(nb); stack.push(nb); }
    }
  }
  const set = new Set(deadOverride ?? deadShown);
  const allDead = group.every((q) => set.has(q));
  for (const q of group) { if (allDead) set.delete(q); else set.add(q); }
  requestScore([...set]);
}

function requestScore(override) {
  deadOverride = override;
  countScore = null;
  render();
  updateStatus();
  if (!ensureWorker()) return;
  worker.postMessage({ type: 'score', id: ++scoreSeq, moves: hist.slice(), deadOverride: override });
}

function enterCount() {
  cancelEndDlg();
  if (searching) abortEngine();
  countMode = true;
  ownership = null; estimateOn = false;
  requestScore(null);                        // 先按引擎自动判定标示(NN 加载时带 ownership 辅助)
  openCountPanel();                          // 独立数子窗口(回包落地时填明细)
}

function exitCount() {
  countMode = false;
  countPanelWanted = false;
  countDlg.close();
  deadOverride = null;
  deadShown = new Set();
  countScore = null;
  render();
  updateStatus();
  if (!gameOver && vsAI && turn === aiSide()) thinkAI();
}

/** 形势判断开关:向引擎要一次 estimate(ownership 覆盖层 + 目差/胜率) */
function toggleEstimate() {
  if (countMode || searching) return;
  estimateOn = !estimateOn;
  ownership = null;
  render();
  if (!estimateOn) { updateStatus(); return; }
  infoL.textContent = '形势判断中…';
  if (!ensureWorker()) return;
  worker.postMessage({ type: 'estimate', id: ++estSeq, moves: hist.slice() });
}

function doMove(mv) {
  hist.push(mv);
  if (mv !== PASS) lastMove = mv;
  turn ^= 1;
  fetchState();
}

/** state 回包落地:重画 + 按回包事实终局(双停自动进数子)/ 调度 AI */
function applyState(d) {
  board = d.board;
  legal = new Set(d.legal);
  ko = d.ko;
  captures = d.captures;
  turn = d.stm;
  ownership = null; estimateOn = false;      // 新局面:旧形势图作废
  if (d.over) {
    /* 双停终局:自动进数子 —— 死子按「规则侧 + NN ownership 辅助」上盘标注,
     * 可点棋子手改,数子实时更新;结算弹窗稍后弹出(读最新数子结果)。 */
    gameOver = true;
    countMode = true;
    deadOverride = null; deadShown = new Set();
    requestScore(null);
    endGame(d.score);
    render();
    return;
  }
  render();
  if (!gameOver && vsAI && turn === aiSide()) setTimeout(thinkAI, 260);
  else updateStatus();
}

/* 结算弹窗(双停):用独立数子窗口展示明细(含死子分布与操作提示) */
function showResultDialog() {
  openCountPanel();
}

function endGame(score, isResign = false) {
  gameOver = true;
  /* 不 terminate worker:数子标注还要用它(自动标注 + 手改重算);过期回包有序号防线 */
  endScoreInfo = score;
  endResigned = isResign;
  const win = score.margin > 0 ? '黑胜' : score.margin < 0 ? '白胜' : '和棋';
  const diff = Math.abs(score.margin).toFixed(1);
  const line = isResign
    ? `终局 · ${win === '和棋' ? win : win}(AI 认输)`
    : `终局 · ${win === '和棋' ? win : win + ' ' + diff + ' 目'}`;
  /* 结算弹窗缓一拍:让玩家看清终局盘面再弹;缓冲期里的操作会取消它。
   * 双停的弹窗文本在读秒时取最新数子 —— NN 标注回包大概率已落地。 */
  popEndDlg(() => {
    if (isResign) {
      showDialog({ title: '终局(中盘)', message: `${win === '和棋' ? '和棋' : win}(AI 认输)` });
    } else {
      showResultDialog();
    }
  });
  statusL.textContent = line;
  setTitle('围棋 — 终局');
  toast('围棋:' + line);
}

/* ---------- Worker:难度表 / 局面事实 / 搜索都经它 ----------
 * 唯一引擎:src/nn-worker.js(aethernn 自研 WebGPU 推理,i8 权重 f16 计算,
 * PUCT 搜索对齐 KataGo;UCT 随机演棋引擎已于 2026-10-02 移除,
 * ?engine=ort 逃生舱已于 2026-10-08 移除)。 */
let worker = null, reqSeq = 0, stateSeq = 0, statePending = null;
let nnLoaded = false, nnWanted = false; // 模型是否就绪 / 是否在等它思考
/* 学生模型 b8c96h3tfrs 第 40 份(s68320512,循环赛 40>41>45 拍板):
 * v17 transformer,dumponnx 19 路导出 —— 旧占位 b6c96 已退役,仅支持此模型 */
const NN_MODEL_URL = new URL('../models/b8c96h3tfrs_19.i8.aewn', import.meta.url).href;

function killWorker() {
  if (worker) { worker.terminate(); worker = null; }
  searching = false;
  nnLoaded = false;
  scoreSeq++;                            // 作废在途数子/形势回包
  estSeq++;
  if (statePending) { const p = statePending; statePending = null; p(null); }
  reqSeq++;    // 作废已进主线程队列的旧结果
}

/** 作废在途请求(局面已变 / 页面关闭),免得过期着法落到新对局上 */
function abortEngine() { killWorker(); infoL.textContent = ''; }

function ensureWorker() {
  if (worker) return worker;
  try {
    /* pages/app.js 的上一级就是仓库根:本地仓库起服与 GitHub Pages 的
     * _site 是同一布局,相对路径在两边走的是同一套 */
    worker = new Worker(new URL('../src/nn-worker.js', import.meta.url), { type: 'module' });
  } catch (err) {
    console.error('[go-pages] 无法创建 AI Worker:', err);
    worker = null; searching = false;
    statusL.textContent = 'AI 不可用(Worker 创建失败)';
    return null;
  }
  worker.onmessage = onEngineMsg;
  worker.onerror = (ev) => {
    console.warn('[go-pages] AI Worker 异常:', ev.message || ev);
    killWorker();
    statusL.textContent = 'AI 出错,已跳过本步';
  };
  worker.postMessage({ type: 'levels' });
  levelSel.disabled = true;
  levelSel.title = 'NN 引擎加载中…';
  infoL.textContent = 'NN:加载运行时与模型…';
  worker.postMessage({ id: ++reqSeq, type: 'load', modelUrl: NN_MODEL_URL });
  return worker;
}

function onEngineMsg(e) {
  const d = e.data;
  if (!d) return;
  if (d.type === 'levels') { applyLevels(d); return; }
  if (d.type === 'status') { infoL.textContent = 'NN:' + d.text; return; }
  if (d.type === 'loaded') {
    nnLoaded = true;
    infoL.textContent = 'NN 就绪(WebGPU)';
    if (nnWanted) { nnWanted = false; thinkAI(); }
    return;
  }
  if (d.type === 'state') {
    if (!statePending || d.id !== stateSeq) return;   // 过期局面直接丢
    const p = statePending; statePending = null;
    p(d.error ? null : d);
    return;
  }
  if (d.type === 'score') {
    if (d.id !== scoreSeq) return;                    // 过期数子直接丢
    countScore = d.detail ?? d.score;                 // detail 带子/空/贴明细(旧回包兜底)
    deadShown = new Set(countScore.dead);
    render();
    updateStatus();
    if (countPanelWanted && countScore.blackStones !== undefined) { countPanelWanted = false; openCountDlg(); }
    else if (countDlg.open) openCountDlg();           // 窗口开着:实时刷新明细
    return;
  }
  if (d.type === 'estimate') {
    if (d.id !== estSeq || !estimateOn) return;       // 过期/已关掉的形势直接丢
    if (d.error) { infoL.textContent = '形势判断失败:' + d.error; return; }
    ownership = d.ownership ? Float32Array.from(d.ownership) : null;
    const wr = fmtRate(d.winRate);
    /* 目差双口径:网端 lead 头(已含贴目,模型自己的分数预测)优先,
     * 缺了回落归属求和口径(旧模型没有 lead 消费) */
    const lead = d.netScoreLead ?? d.scoreLead;
    infoL.textContent = '形势:黑胜率 ' + wr
      + (lead != null ? ` · 黑目差 ${lead > 0 ? '+' : ''}${lead.toFixed(1)}` : '');
    render();
    return;
  }
  /* ---- 以下是搜索回包(progress / 最终结果)---- */
  if (d.id !== reqSeq) return;               // 过期结果(换难度/新对局)直接丢
  if (d.type === 'progress') { showInfo(d); return; }
  searching = false;
  if (d.error) { statusL.textContent = '引擎异常:' + d.error; return; }
  if (d.resign) {                            // NN 引擎动态认输(N4)
    gameOver = true;
    render();
    endGame({ margin: aiSide() === BLACK ? -999 : 999 }, true);
    return;
  }
  if (!d.move && d.move !== 0) { fetchState(); return; }  // AI 无着法 = 判终局,事实以 state 为准
  hist.push(d.move);
  if (d.move !== PASS) lastMove = d.move;
  turn ^= 1;
  showInfo(d);
  fetchState();
}

/** 向 Worker 要当前局面的规则事实(state 契约) */
function fetchState() {
  if (!ensureWorker()) return;
  const id = ++stateSeq;
  statePending = (d) => {
    if (!d) return;                           // 被作废(terminate / 新对局)
    applyState(d);
  };
  worker.postMessage({ type: 'state', id, moves: hist.slice() });
}

/** 开局问一次引擎的难度表,拿到才填下拉 */
function applyLevels(d) {
  const table = Array.isArray(d.levels)
    ? d.levels.filter((lv) => lv && typeof lv.name === 'string' && lv.name) : [];
  if (!table.length) {
    levelSel.title = 'AI 难度不可用(引擎未上报)';
    return;
  }
  levels = table;
  const def = Number.isInteger(d.default) && d.default >= 0 && d.default < table.length ? d.default : 0;
  levelIdx = def;
  levelSel.replaceChildren(...table.map((lv, i) => el('option', { value: String(i) }, lv.name)));
  levelSel.value = String(def);
  levelSel.disabled = false;
  levelSel.title = 'AI 难度:' + table.map((lv) => lv.name).join(' / ');
}

function thinkAI() {
  if (gameOver || searching || countMode) return;
  if (!nnLoaded) { nnWanted = true; statusL.textContent = 'NN 引擎加载中…'; return; }
  searching = true;
  render();
  statusL.textContent = `${sideName(aiSide())}思考中…`;
  setTitle(`围棋 — AI 思考中(${lvName()})`);
  infoL.textContent = '';
  if (typeof Worker === 'undefined') {
    searching = false;
    statusL.textContent = '当前环境不支持 Web Worker,AI 不可用';
    return;
  }
  if (!ensureWorker()) return;
  worker.postMessage({ id: ++reqSeq, moves: hist.slice(), level: levelIdx });
}

/** 底栏右侧的引擎信息行(等宽字体) */
function showInfo(d) {
  const lead = d.scoreLead != null
    ? ` · 目差 ${d.scoreLead > 0 ? '+' : ''}${d.scoreLead.toFixed(1)}` : '';
  infoL.textContent = `NN·${lvName()} · ${fmtVisits(d.visits)} 访问 · ${d.ms}ms · 胜率 ${fmtRate(d.winRate)}${lead}`;
}

/* ---------- 工具栏动作 ---------- */
function resetGame() {
  if (searching) abortEngine(); else reqSeq++;  // 空闲不杀 worker(模型别重载)
  cancelEndDlg();
  countDlg.close(); countPanelWanted = false;
  turn = BLACK; hist = []; lastMove = null;
  gameOver = false;
  countMode = false; deadOverride = null; deadShown = new Set(); countScore = null;
  endScoreInfo = null; endResigned = false;
  ownership = null; estimateOn = false;
  board = new Array(N * N).fill(0);
  legal = new Set(); ko = -1; captures = [0, 0];
  render();
  fetchState();                                // 初始局面事实照问引擎
  if (vsAI && turn === aiSide()) thinkAI();    // 玩家执白时 AI 执黑先行
  else updateStatus();
}

/** 悔棋:撤到「轮到玩家重新决策」为止。人机撤两手,人人撤一手。
 *  空闲时只作废在途回包、不 terminate —— 已加载的模型别白白重载。 */
function doUndo() {
  if (!hist.length) return;
  if (searching) abortEngine(); else reqSeq++;
  cancelEndDlg();
  countDlg.close(); countPanelWanted = false;
  if (countMode) {                            // 数子中悔棋 = 先退出数子
    countMode = false; deadOverride = null; deadShown = new Set(); countScore = null;
  }
  endScoreInfo = null; endResigned = false;
  ownership = null; estimateOn = false;
  let n = 1;
  if (vsAI && turn === humanSide && hist.length >= 2) n = 2;
  while (n-- > 0 && hist.length) hist.pop();
  turn = hist.length % 2 === 0 ? BLACK : WHITE;
  gameOver = false;
  lastMove = null;
  for (let i = hist.length - 1; i >= 0; i--) {
    if (hist[i] !== PASS) { lastMove = hist[i]; break; }
  }
  fetchState();
  if (vsAI && turn === aiSide()) thinkAI();
  else { render(); updateStatus(); }
}

/** 换边:与 AI 互换执子方。围棋不翻盘(坐标恒定)。空闲不杀 worker(理由同悔棋) */
function switchSide() {
  if (searching) abortEngine(); else reqSeq++;
  cancelEndDlg();
  countDlg.close(); countPanelWanted = false;
  if (countMode) {                            // 换边同理:先退出数子
    countMode = false; deadOverride = null; deadShown = new Set(); countScore = null;
  }
  ownership = null; estimateOn = false;
  humanSide ^= 1;
  render();
  if (!gameOver && vsAI && turn === aiSide()) thinkAI();
  else if (!gameOver) updateStatus();
}

/* ---------- 界面 ---------- */
const newBtn = el('button', { class: 'btn primary', onClick: resetGame }, icon('refresh'), '新对局');
const levelSel = el('select', {
  class: 'select go-level',
  title: 'AI 难度(等引擎上报)',
  'aria-label': 'AI 难度',
  disabled: true,
  onChange: (e) => {
    levelIdx = Number(e.currentTarget.value) || 0;
    if (searching) { abortEngine(); thinkAI(); }
  },
});
const aiBtn = el('button', {
  class: 'btn', title: '切换人机 / 双人对弈',
  onClick: (e) => {
    cancelEndDlg();
    vsAI = !vsAI;
    e.currentTarget.replaceChildren(vsAI ? '人机' : '双人');
    sideBtn.disabled = !vsAI;                                 // 换边只对人机模式有意义
    if (!vsAI) { abortEngine(); render(); updateStatus(); }   // 关掉 AI 要把在途搜索停掉
    else if (!gameOver && turn === aiSide()) thinkAI();       // 轮到 AI 就立刻接手
    else updateStatus();
  },
}, '人机');
const sideBtn = el('button', {
  class: 'btn', title: '换边:与 AI 互换执子方(棋盘不翻转)',
  onClick: switchSide,
}, '换边');
const passBtn = el('button', {
  class: 'btn', title: '停一手:双方连续停一手即终局数子',
  onClick: () => { if (!gameOver && (!vsAI || turn === humanSide)) doMove(PASS); },
}, '停一手');
const countBtn = el('button', {
  class: 'btn', title: '数子:打开数子窗口(子/空/贴明细),点棋子切换死/活实时重算;终局后为「结果」',
  onClick: () => {
    if (gameOver) {
      openCountPanel();
      if (!countScore) requestScore(deadOverride);
      return;
    }
    if (countMode) exitCount(); else enterCount();
  },
}, '数子');
const estBtn = el('button', {
  class: 'btn', title: '形势判断:NN 逐点归属覆盖层 + 胜率/目差(再点一次关闭)',
  onClick: toggleEstimate,
}, '形势');
const undoBtn = el('button', {
  class: 'btn', title: '悔棋:人机模式连 AI 的应手一起撤,人人模式撤一手',
  onClick: doUndo,
}, icon('reply'), '悔棋');

appEl.append(el('div', { class: 'app' },
  el('div', { class: 'app-toolbar' },
    newBtn,
    el('label', { class: 'go-level-wrap', title: 'AI 难度' },
      el('span', { class: 'dim', style: { fontSize: '12px' } }, '难度'), levelSel),
    passBtn, countBtn, estBtn, aiBtn, sideBtn, undoBtn),
  el('div', { class: 'app-body' }, fitWrap),
  el('div', { class: 'app-status' }, statusL,
    el('span', { class: 'grow' }),
    infoL)));

render();
updateStatus();
(function fetchLevels() {
  ensureWorker();          // worker 建立时已自发 levels;这里只负责建
})();
fetchState();                                  // 初始局面的合法点等事实也要问引擎
new ResizeObserver(fitBoard).observe(appEl.querySelector('.app-body'));
fitBoard();

/* 页面冒烟探针钩子(验证脚本用) */
window.__pagesStats = () => ({
  plies: hist.length, turn, human: humanSide, gameOver, vsAI,
  captures: captures.slice(), level: lvName(),
  nnLoaded, countMode, estimateOn,
});
window.__pagesHumanMove = () => {
  if (gameOver || (vsAI && turn !== humanSide)) return false;
  const pts = [...legal].sort((a, b) => b - a);   // 挑个靠中心附近的点(下标大的靠下)
  const p = pts[Math.floor(pts.length / 2)];
  if (p == null) return false;
  doMove(p);
  return true;
};
