/* ============================================================
 * 围棋在线对弈页 —— UI 移植自 AetherWebOS 的围棋应用
 * (js/apps/go/index.js),布局与交互保持一致:
 * 顶栏「新对局 / 难度 / 停一手 / 人机 / 换边 / 悔棋」+ 中央棋盘 +
 * 底栏左侧行棋状态与提子数、右侧等宽字体引擎搜索信息。
 *
 * 引擎即本仓库的主角:src/worker.js(纯 JS,MCTS/UCT)。
 * UI 一行引擎代码都不 import:难度表、棋盘事实(合法着法 / 劫点 /
 * 提子数 / 双停终局 / 数子结果)全部经 Worker 消息问引擎。
 *
 * UI 持有的唯一对局状态是**走法序列**(交叉点 0..80 或 PASS=81)。
 * ============================================================ */

/* ==================== 微型工具(替代 webos 的 core)==================== */
const $ = (sel) => document.querySelector(sel);

/** 建 DOM:el('button', {class, onClick, dataset}, ...children) */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** 线性图标(路径数据取自 webos 的 core/icons.js) */
const ICON_PATHS = {
  refresh: '<path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/>',
  reply: '<polyline points="9 17 4 12 9 7"/><path d="M20 18v-2a4 4 0 0 0-4-4H4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
};
const icon = (name) => {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('fill', 'none');
  s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '2');
  s.setAttribute('stroke-linecap', 'round');
  s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = ICON_PATHS[name] || '';
  return s;
};

/** webos dialogs.info 的页内替身 */
const dlg = $('#dlg');
function showDialog({ title, message }) {
  $('#dlgTitle').textContent = title;
  $('#dlgMsg').textContent = message;
  if (!dlg.open) dlg.showModal();
}
$('#dlgOk').addEventListener('click', () => dlg.close());
dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });

/** webos bus.notify 的页内替身:右下角吐司 */
function toast(text) {
  const t = el('div', { class: 'toast' }, text);
  t.addEventListener('click', () => t.remove());
  $('#toasts').append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 220); }, 3200);
}

/** 主题:webos 的浅 / 深双主题,记在 localStorage */
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

/* ==================== 对局(逻辑同 webos 围棋应用)==================== */

/* 协议常量(worker 契约的一部分,不是引擎导出) */
const BLACK = 0, WHITE = 1;
const PASS = 81;

/* 格距(px)。与 style.css 里的 --cs / --pad 必须一致 */
const CS = 52, PAD = 26;
const T = PAD * 2 + CS * 8;
const X = (c) => PAD + c * CS;
const Y = (r) => PAD + r * CS;
const COLS = 'ABCDEFGHJ';                     // 列标(跳过 I)
const sideName = (s) => (s === BLACK ? '黑方' : '白方');
const fmtRate = (w) => Math.round(w * 100) + '%';
const fmtVisits = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v));
/** 记谱(显示用):交叉点 → 列字母 + 行号;PASS → 停 */
const moveText = (mv) => (mv === PASS ? '停' : COLS[mv % 9] + (9 - ((mv / 9) | 0)));

/** 棋盘线(SVG):9×9 线 + 五个星位 + 边缘坐标(下 A~J、左 9~1) */
function boardSvg() {
  const d = [];
  for (let i = 0; i < 9; i++) {
    d.push(`M${X(0)} ${Y(i)}H${X(8)}`);
    d.push(`M${X(i)} ${Y(0)}V${Y(8)}`);
  }
  const stars = [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]]
    .map(([r, c]) => `<circle class="go-star" cx="${X(c)}" cy="${Y(r)}" r="3.2"/>`)
    .join('');
  const coords = [];
  for (let c = 0; c < 9; c++) {
    coords.push(`<text class="go-coord" x="${X(c)}" y="${T - PAD / 2}">${COLS[c]}</text>`);
    coords.push(`<text class="go-coord" x="${PAD / 2}" y="${Y(c)}">${9 - c}</text>`);
  }
  return `<svg class="go-lines" viewBox="0 0 ${T} ${T}" aria-hidden="true">`
    + `<path class="go-line" d="${d.join(' ')}"/>${stars}${coords.join('')}</svg>`;
}

const appEl = $('#app');

let board = new Array(81).fill(0);   // 引擎棋盘(state 回包驱动;0 空 / 1 黑 / 2 白)
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

const aiSide = () => humanSide ^ 1;
const lvName = () => levels[levelIdx]?.name ?? '—';

const statusL = el('span', {}, '黑方行棋');
const infoL = el('span', {
  class: 'mono', style: { fontSize: '11px' },
  title: '引擎搜索信息(胜率是 AI 视角,来自蒙特卡洛演棋)',
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
  const humanTurn = !gameOver && (!vsAI || turn === humanSide);
  boardEl.classList.toggle('turn-b', humanTurn && turn === BLACK);
  boardEl.classList.toggle('turn-w', humanTurn && turn === WHITE);
  const hint = humanTurn ? legal : null;
  for (let p = 0; p < 81; p++) {
    const btn = el('button', {
      class: 'go-pt' + (hint && !board[p] && hint.has(p) ? ' can' : ''),
      style: { left: X(p % 9) + 'px', top: Y((p / 9) | 0) + 'px' },
      dataset: { i: String(p) },
      onClick: () => onPoint(p),
    });
    if (board[p]) {
      const st = el('div', {
        class: `go-stone ${board[p] === 1 ? 'black' : 'white'}${p === lastMove ? ' last' : ''}`,
      });
      if (p === lastMove) st.classList.add('drop');
      btn.append(st);
    } else if (hint) {
      btn.append(el('div', { class: 'go-ghost' }));
    }
    layerEl.append(btn);
  }
  passBtn.disabled = !humanTurn;
}

function updateStatus() {
  if (gameOver) return;
  const caps = `黑提 ${captures[BLACK]} · 白提 ${captures[WHITE]}`;
  statusL.textContent = `${sideName(turn)}行棋 · ${caps}`;
  const last = hist.length ? ` · 上一手 ${moveText(hist[hist.length - 1])}` : '';
  setTitle(`围棋 — ${sideName(turn)}行棋${last}`);
}

/* ---------- 落子:合法性以缓存 state 为准,走子 = 改序列 + 再问一次引擎 ---------- */
function onPoint(p) {
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

function doMove(mv) {
  hist.push(mv);
  if (mv !== PASS) lastMove = mv;
  turn ^= 1;
  fetchState();
}

/** state 回包落地:重画 + 按回包事实终局(双停数子)/ 调度 AI */
function applyState(d) {
  board = d.board;
  legal = new Set(d.legal);
  ko = d.ko;
  captures = d.captures;
  turn = d.stm;
  if (d.over) { endGame(d.score); return; }
  render();
  if (!gameOver && vsAI && turn === aiSide()) setTimeout(thinkAI, 260);
  else updateStatus();
}

function endGame(score) {
  gameOver = true;
  abortEngine();
  const win = score.margin > 0 ? '黑胜' : score.margin < 0 ? '白胜' : '和棋';
  const diff = Math.abs(score.margin).toFixed(1);
  const line = `终局 · ${win === '和棋' ? win : win + ' ' + diff + ' 目'}`;
  showDialog({
    title: '终局(双停)',
    message: `黑 ${score.black} · 白 ${score.white} —— ${win === '和棋' ? '和棋' : win + ' ' + diff + ' 目'}`,
  });
  statusL.textContent = line;
  setTitle('围棋 — 终局');
  toast('围棋:' + line);
}

/* ---------- Worker:难度表 / 局面事实 / 搜索都经它 ---------- */
let worker = null, reqSeq = 0, stateSeq = 0, statePending = null;

function killWorker() {
  if (worker) { worker.terminate(); worker = null; }
  searching = false;
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
    worker = new Worker(new URL('../src/worker.js', import.meta.url), { type: 'module' });
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
  return worker;
}

function onEngineMsg(e) {
  const d = e.data;
  if (!d) return;
  if (d.type === 'levels') { applyLevels(d); return; }
  if (d.type === 'state') {
    if (!statePending || d.id !== stateSeq) return;   // 过期局面直接丢
    const p = statePending; statePending = null;
    p(d.error ? null : d);
    return;
  }
  /* ---- 以下是搜索回包(progress / 最终结果)---- */
  if (d.id !== reqSeq) return;               // 过期结果(换难度/新对局)直接丢
  if (d.type === 'progress') { showInfo(d); return; }
  searching = false;
  if (d.error) { statusL.textContent = '引擎异常:' + d.error; return; }
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
  levelSel.append(...table.map((lv, i) => el('option', { value: String(i) }, lv.name)));
  levelSel.value = String(def);
  levelSel.disabled = false;
  levelSel.title = 'AI 难度:' + table.map((lv) => lv.name).join(' / ');
}

function thinkAI() {
  if (gameOver || searching) return;
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
  infoL.textContent = `${lvName()} · ${fmtVisits(d.visits)} 演棋 · ${d.ms}ms · 胜率 ${fmtRate(d.winRate)}`;
}

/* ---------- 工具栏动作 ---------- */
function resetGame() {
  abortEngine();
  turn = BLACK; hist = []; lastMove = null;
  gameOver = false;
  board = new Array(81).fill(0);
  legal = new Set(); ko = -1; captures = [0, 0];
  render();
  fetchState();                                // 初始局面事实照问引擎
  if (vsAI && turn === aiSide()) thinkAI();    // 玩家执白时 AI 执黑先行
  else updateStatus();
}

/** 悔棋:撤到「轮到玩家重新决策」为止。人机撤两手,人人撤一手 */
function doUndo() {
  if (!hist.length) return;
  abortEngine();
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

/** 换边:与 AI 互换执子方。围棋不翻盘(坐标恒定) */
function switchSide() {
  abortEngine();
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
const undoBtn = el('button', {
  class: 'btn', title: '悔棋:人机模式连 AI 的应手一起撤,人人模式撤一手',
  onClick: doUndo,
}, icon('reply'), '悔棋');

appEl.append(el('div', { class: 'app' },
  el('div', { class: 'app-toolbar' },
    newBtn,
    el('label', { class: 'go-level-wrap', title: 'AI 难度' },
      el('span', { class: 'dim', style: { fontSize: '12px' } }, '难度'), levelSel),
    passBtn, aiBtn, sideBtn, undoBtn),
  el('div', { class: 'app-body' }, fitWrap),
  el('div', { class: 'app-status' }, statusL,
    el('span', { class: 'grow' }),
    infoL)));

render();
updateStatus();
(function fetchLevels() {
  if (!ensureWorker()) return;
  worker.postMessage({ type: 'levels' });      // 回包经 onEngineMsg → applyLevels
})();
fetchState();                                  // 初始局面的合法点等事实也要问引擎
new ResizeObserver(fitBoard).observe(appEl.querySelector('.app-body'));
fitBoard();

/* 页面冒烟探针钩子(验证脚本用) */
window.__pagesStats = () => ({
  plies: hist.length, turn, human: humanSide, gameOver, vsAI,
  captures: captures.slice(), level: lvName(),
});
window.__pagesHumanMove = () => {
  if (gameOver || (vsAI && turn !== humanSide)) return false;
  const pts = [...legal].sort((a, b) => b - a);   // 挑个靠中心附近的点(下标大的靠下)
  const p = pts[Math.floor(pts.length / 2)];
  if (p == null) return false;
  doMove(p);
  return true;
};
