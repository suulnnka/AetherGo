/* ============================================================
 * AI Worker:引擎的门面(UI 不 import 引擎源码,一切经消息)
 *   ping                     → { type:'pong', tag }
 *   { type:'levels' }        → { type:'levels', tag, engine, default, levels }
 *                              纯声明难度表,不触发任何引擎加载;
 *                              UI 只读 name/id,playouts/ms/jitter 是实现细节
 *   { type:'state', id, moves }
 *                            → { type:'state', id, board, stm, captures, ko,
 *                                legal, over, score }
 *                              规则查询的单一入口:重演序列后回报棋盘、提子数、
 *                              劫点、合法着法(不含停一手)、双停终局与数子结果
 *   { type:'think', id, moves, level }
 *                            → 逐步 { id, type:'progress', visits, move, winRate, ms }
 *                            → { id, move, visits, nodes, ms, winRate, only }
 *                              level 是**本引擎难度表的下标**(表由 levels 自报)
 *
 * moves 是走法序列(交叉点 0..80 或 PASS=81)—— 传序列而不是传棋盘:
 * 结构化克隆最省,且编码只有一套,不存在两条解析路径。
 *
 * 搜索是同步的,Worker 收到新消息只会排队;UI 侧用请求序号丢弃过期结果,
 * 需要真正中断时直接 terminate 再造一个(见 webos 应用侧 abortEngine)。
 * ============================================================ */
import {
  newBoard, replayMoves, searchBest, make, capturedOf, koPoint, scoreGame,
  genLegal, boardToArray, LEVELS, DEFAULT_LEVEL, PASS,
} from './engine.js';

/* ENGINE_TAG 让下游 webos 的体积闸门(check-size.mjs)能在 dist 里认出这个 chunk
 * (字符串不会被压缩改名)。 */
const ENGINE_TAG = 'go-engine-v1';
self.__engineTag = ENGINE_TAG;

/** 重演序列并产出「UI 渲染所需的全部规则事实」:棋盘、提子、劫点、合法着法、
 *  双停终局与中国规则数子。这是 state 消息的唯一事实源 —— UI 不复判任何规则。 */
function describeState(d) {
  const bd = newBoard();
  const side = replayMoves(bd, d.moves);
  if (side < 0) return { error: 'illegal-sequence' };

  /* 提子数按行棋方累计:重演时逐手记 make() 的返回令牌(扑子数编码在令牌里) */
  const caps = [0, 0];
  const rb = newBoard();
  let s = 0;
  for (const mv of d.moves) {
    caps[s] += capturedOf(make(rb, mv, s));
    s ^= 1;
  }

  const over = d.moves.length >= 2
    && d.moves[d.moves.length - 1] === PASS && d.moves[d.moves.length - 2] === PASS;

  return {
    board: boardToArray(bd),
    stm: side,
    captures: caps,
    ko: koPoint(),
    legal: over ? [] : genLegal(bd, side),
    over,
    score: over ? scoreGame(bd) : null,       // { black, white, margin }(中国规则,黑贴目)
  };
}

self.onmessage = (e) => {
  const d = e.data;
  if (!d) return;

  if (d.type === 'ping') { self.postMessage({ type: 'pong', tag: ENGINE_TAG }); return; }

  if (d.type === 'levels') {
    /* 纯声明:难度表(含参数)是引擎的实现细节,UI 只拿 name/id 建下拉 */
    self.postMessage({ type: 'levels', tag: ENGINE_TAG, engine: 'js', default: DEFAULT_LEVEL, levels: LEVELS });
    return;
  }

  if (d.type === 'state') {
    const s = describeState(d);
    self.postMessage(s.error
      ? { type: 'state', id: d.id, error: s.error }
      : { type: 'state', id: d.id, tag: ENGINE_TAG, ...s });
    return;
  }

  const t0 = Date.now();
  const bd = newBoard();
  const side = replayMoves(bd, d.moves);
  if (side < 0) { self.postMessage({ id: d.id, error: 'illegal-sequence' }); return; }
  const lv = LEVELS[d.level] ?? LEVELS[DEFAULT_LEVEL] ?? LEVELS[0];
  const r = searchBest(bd, side, {
    playouts: lv.playouts, ms: lv.ms, jitter: lv.jitter ?? 0,
    lastMove: d.moves.length ? d.moves[d.moves.length - 1] : -1,
    onProgress: (p) => self.postMessage({
      id: d.id, type: 'progress', visits: p.visits, move: p.move,
      winRate: p.winRate, ms: p.ms,
    }),
  });
  self.postMessage({
    id: d.id, move: r.move, visits: r.visits, nodes: r.visits,
    ms: Date.now() - t0, winRate: r.winRate, only: !!r.only,
  });
};
