/* ============================================================
 * AI Worker:只是一层薄壳
 *   收 { id, moves, playouts, ms, jitter }
 *   逐次回 { id, type:'progress', visits, move, winRate, ms }
 *   结束回 { id, move, visits, nodes, ms, winRate, only }
 *
 * 逐步回报是给 UI 的搜索信息行用的:搜索在 Worker 里同步跑,但主线程是空的,
 * 所以这些消息能实时送到,画面上就能看到访问次数与胜率在涨。
 *
 * moves 是走法序列(交叉点 0..80 或 PASS=81)—— 传序列而不是传棋盘:
 * 结构化克隆最省,且 UI 与 Worker 共用同一份 engine.js,编码天然一致。
 *
 * 搜索是同步的,Worker 收到新消息只会排队;UI 侧用请求序号丢弃过期结果,
 * 需要真正中断时直接 terminate 再造一个(见 webos 应用侧 abortEngine)。
 * ============================================================ */
import { newBoard, replayMoves, searchBest, PASS } from './engine.js';

/* ENGINE_TAG 让下游 webos 的体积闸门(check-size.mjs)能在 dist 里认出这个 chunk
 * (字符串不会被压缩改名)。 */
const ENGINE_TAG = 'go-engine-v1';
self.__engineTag = ENGINE_TAG;

self.onmessage = (e) => {
  const d = e.data;
  if (d && d.type === 'ping') { self.postMessage({ type: 'pong', tag: ENGINE_TAG }); return; }
  const t0 = Date.now();
  const bd = newBoard();
  const side = replayMoves(bd, d.moves);
  if (side < 0) { self.postMessage({ id: d.id, error: 'illegal-sequence' }); return; }
  const r = searchBest(bd, side, {
    playouts: d.playouts, ms: d.ms, jitter: d.jitter ?? 0,
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
