/* ============================================================
 * NN 引擎 Worker:AetherGo 唯一的对弈引擎(UCT 随机演棋引擎已移除)。
 *
 *   ping / { type:'levels' } / { type:'state', id, moves }
 *     —— 规则查询走同一份 engine.js(规则/数子事实的唯一来源)
 *   { type:'score', id, moves, deadOverride? }   —— 数子(deadOverride 手改死子点)
 *   { type:'estimate', id, moves }               —— 形势判断:单次推理,
 *     回 { engine:'nn', winRate(黑方), scoreLead(黑方目差:归属求和口径,已扣贴目),
 *          netScoreLead(黑方目差:网端 lead 头,已含贴目,新模型可用的备选口径),
 *          ownership(黑方视角归属图) }
 *   { type:'load', id, modelUrl }        → { id, type:'loaded', ep } / { id, error }
 *     懒加载:自研引擎 aethernn(src/nn/webgpu/,权重 .aewn)+ WebGPU;
 *     ?engine=ort 切回 onnxruntime-web 逃生舱。不可用时报错,UI 显示原因。
 *   { type:'think', id, moves, level, temperature?, temperatureHalflife? }
 *     → 逐步 { id, type:'progress', visits, move, winRate, ms } → 最终着法
 *     temperature:开局温度,缺省 0(LCB 选点);>0 按访问数^(1/T) 随机抽
 *     temperatureHalflife:温度半衰期(手),>0 时引擎按 moves 手数将温度减半衰减
 *
 * levels 自报:NN 档按「NN 访问数」分级;未 load 时 think 回 error
 * (UI 侧保证先 load)。
 * ============================================================ */
import {
  newBoard, replayMoves, make, capturedOf, koPoint, genLegal, boardToArray,
  finalScore, scoreBreakdown, deadStonesWithOwnership, PASS, BLACK, WHITE, KOMI,
} from './engine.js';
import { nnSearchBest } from './nn/search.js';
import { createSession } from './nn/session.js';
import { encodeFeatures } from './nn/features.js';

const ENGINE_TAG = 'go-engine-nn-v0';
self.__engineTag = ENGINE_TAG;

export const NN_LEVELS = [
  { id: 'easy', name: '初级', desc: '80 次 NN 访问', visits: 80 },
  { id: 'normal', name: '中级', desc: '200 次 NN 访问', visits: 200 },
  { id: 'hard', name: '高级', desc: '400 次 NN 访问', visits: 400 },
  { id: 'master', name: '大师', desc: '800 次 NN 访问', visits: 800 },
];
export const NN_DEFAULT_LEVEL = 2;

let session = null;
let resignHist = [];                      // 最近 ≤3 手的白方视角 mcts 值(认输判据)
let resignLastLen = -1;                   // 上次思考时的手数(回退即作废历史)

/* 规则查询:重演序列并产出「UI 渲染所需的全部规则事实」(score 含死子处理) */
function describeState(d) {
  const bd = newBoard();
  const side = replayMoves(bd, d.moves);
  if (side < 0) return { error: 'illegal-sequence' };

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
    score: over ? finalScore(bd) : null,
  };
}

self.onmessage = async (e) => {
  const d = e.data;
  if (!d) return;

  if (d.type === 'ping') { self.postMessage({ type: 'pong', tag: ENGINE_TAG }); return; }

  if (d.type === 'levels') {
    self.postMessage({
      type: 'levels', tag: ENGINE_TAG, engine: 'nn', default: NN_DEFAULT_LEVEL,
      levels: NN_LEVELS, loaded: !!session,
    });
    return;
  }

  if (d.type === 'load') {
    try {
      session = await createSession({
        modelUrl: d.modelUrl,
        engine: d.engine,                      // 'aewnn'(缺省)| 'ort'(逃生舱)
        onStatus: (s) => self.postMessage({ id: d.id, type: 'status', text: s }),
      });
      self.postMessage({ id: d.id, type: 'loaded', ep: session.ep });
    } catch (err) {
      self.postMessage({ id: d.id, error: String(err) });
    }
    return;
  }

  if (d.type === 'state') {
    const s = describeState(d);
    self.postMessage(s.error
      ? { type: 'state', id: d.id, error: s.error }
      : { type: 'state', id: d.id, tag: ENGINE_TAG, ...s });
    return;
  }

  /* 数子:规则同源 engine.js;不要求双停。
   * 自动判定时(NN 已加载)先单次推理取 ownership,辅助标注规则侧漏判的
   * 死链(deadStonesWithOwnership:Benson 活棋保底、规则结论不翻案);
   * ownership 模型侧是行棋方视角,这里翻成 scoring.js 约定的黑方视角再传;
   * 用户手改(deadOverride 数组,可为空 = 全活)直接生效,不走 NN。 */
  if (d.type === 'score') {
    const bd = newBoard();
    const side = replayMoves(bd, d.moves);
    if (side < 0) { self.postMessage({ id: d.id, error: 'illegal-sequence' }); return; }
    let ownership = null;
    if (!Array.isArray(d.deadOverride) && session) {
      try {
        const f = encodeFeatures(bd, side, { recentMoves: d.moves.slice(), komi: KOMI });
        const [out] = await session.evalBatch([{ spatial: f.spatial, global: f.global }]);
        if (out.ownership) {
          ownership = new Float32Array(out.ownership.length);
          for (let p = 0; p < ownership.length; p++) {
            /* 会话给的是行棋方视角裸 pretanh:先 tanh 成归属,再翻黑方视角 */
            const t = Math.tanh(out.ownership[p]);
            ownership[p] = side === BLACK ? t : -t;
          }
        }
      } catch { ownership = null; }              // 推理失败:回落纯规则判定
    }
    const auto = deadStonesWithOwnership(bd, ownership);
    const detail = scoreBreakdown(bd, KOMI, Array.isArray(d.deadOverride) ? d.deadOverride : auto);
    self.postMessage({
      id: d.id, type: 'score', tag: ENGINE_TAG,
      /* score 保持旧形状(UI 兼容),detail 带子/空/贴明细给数子窗口 */
      score: { black: detail.black, white: detail.white, margin: detail.margin, dead: detail.dead },
      detail,
    });
    return;
  }

  /* 形势判断:单次 NN 推理 —— 归属求和与网端 lead 两条口径都换成黑方视角
   * (模型输出是行棋方视角,+ = 行棋方优;不翻则白行棋时全盘反号)。 */
  if (d.type === 'estimate') {
    if (!session) { self.postMessage({ id: d.id, type: 'estimate', error: 'nn-engine-not-loaded' }); return; }
    const bd = newBoard();
    const side = replayMoves(bd, d.moves);
    if (side < 0) { self.postMessage({ id: d.id, error: 'illegal-sequence' }); return; }
    const f = encodeFeatures(bd, side, { recentMoves: d.moves.slice(), komi: KOMI });
    const [out] = await session.evalBatch([{ spatial: f.spatial, global: f.global }]);
    const flip = side === BLACK ? 1 : -1;
    let ownership = null, scoreLead = null, netScoreLead = null;
    if (out.ownership) {
      ownership = new Float32Array(out.ownership.length);   // 黑方视角给 UI(正=黑)
      let sum = 0;
      for (let p = 0; p < ownership.length; p++) {
        ownership[p] = flip * Math.tanh(out.ownership[p]);  // pretanh → 归属,翻黑方视角
        sum += ownership[p];
      }
      scoreLead = sum - KOMI;                     // 归属求和口径:黑目差(不含贴目再另扣)
    }
    if (out.scoreLead != null) netScoreLead = flip * out.scoreLead;   // 网端 lead 口径(已含贴目)
    self.postMessage({
      id: d.id, type: 'estimate', tag: ENGINE_TAG, engine: 'nn',
      /* winLoss 是行棋方视角(-1..1)→ 换算黑方胜率 */
      winRate: side === BLACK ? (out.winLoss + 1) / 2 : (1 - out.winLoss) / 2,
      scoreLead,
      netScoreLead,
      ownership,
    });
    return;
  }

  /* think:try/catch 必须有 —— handler 是 async,一旦异常逃逸就是 worker 内
   * 未处理拒绝,不会传到页面 worker.onerror,UI 将永远停在「思考中」 */
  if (!session) {
    self.postMessage({ id: d.id, error: 'nn-engine-not-loaded' });
    return;
  }
  const t0 = Date.now();
  try {
  const bd = newBoard();
  const side = replayMoves(bd, d.moves);
  if (side < 0) { self.postMessage({ id: d.id, error: 'illegal-sequence' }); return; }
  const lv = NN_LEVELS[d.level] ?? NN_LEVELS[NN_DEFAULT_LEVEL];
  const r = await nnSearchBest(bd, side, {
    session,
    visits: lv.visits,
    maxBatch: session.maxBatch ?? 8,         // 加载时校准的批上限(吞吐 ≥ 最优 90% 的最小批)
    temperature: d.temperature,              // 最终选点温度(缺省 0 = argmax)
    temperatureHalflife: d.temperatureHalflife,  // 温度半衰期(手);>0 按手数减半
    recentMoves: d.moves.slice(),
    onProgress: (p) => self.postMessage({
      id: d.id, type: 'progress', visits: p.visits, move: p.move,
      winRate: p.winRate, scoreLead: p.scoreLead, ms: p.ms,
    }),
  });
  /* 认输(GTP 实战配方,C++ play.cpp:白方视角 mcts 值连续 3 手越过 −0.90,
   * 且手数 ≥ 1+361/5 = 73 —— 前手不认输;访问过少(初级档)不认,防噪声误判。
   * 手数回退(悔棋 / 新局)即作废历史。 */
  {
    if (d.moves.length < resignLastLen) resignHist = [];
    resignLastLen = d.moves.length;
    const wlStm = r.winRate * 2 - 1;
    resignHist.push(side === BLACK ? -wlStm : wlStm);       // → 白方视角
    if (resignHist.length > 3) resignHist.shift();
    const MIN_TURN = 73;
    const losing = side === WHITE ? ((v) => v < -0.90) : ((v) => v > 0.90);
    if (d.moves.length >= MIN_TURN && lv.visits >= 80
      && resignHist.length >= 3 && resignHist.every(losing)) {
      self.postMessage({ id: d.id, resign: true, winRate: r.winRate, ms: Date.now() - t0 });
      return;
    }
  }
  self.postMessage({
    id: d.id, move: r.move, visits: r.visits, nodes: r.visits,
    ms: Date.now() - t0, winRate: r.winRate, scoreLead: r.scoreLead, only: !!r.only,
  });
  } catch (err) {
    self.postMessage({ id: d.id, error: `think-failed: ${err && (err.stack || err.message) || err}` });
  }
};

/* 测试钩子(worker-smoke-test 用,同 search.js __nodeTableSize 风格):
 * 直注会话绕过 WebGPU/CDN,驱动完整 think 消息流(含认输判定等搜索后路径) */
export function __setSessionForTest(s) { session = s; }
