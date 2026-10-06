/* ============================================================
 * AetherGo NN 搜索 — 评估缓存(KataGo useEvalCache 精神)
 *
 * 叶子特征双种子 FNV 哈希入 LRU:同一(盘面+行棋方+5 手历史窗口+贴目)
 * 特征再次到达时直接复用评估,零推理。键只含输入特征 —— 纯函数模型下
 * 同特征必同输出,缓存恒可靠;只有测试里「同一局面换不同桩配置」才需要
 * clearEvalCache()。真·转置节点共享(useGraphSearch)需要多父节点与
 * 引用计数 GC,后置 —— 本缓存先拿走大头收益。
 * ============================================================ */

const EVAL_CACHE_CAP = 2048;
const evalCache = new Map();

function evalCacheGet(key) {
  const v = evalCache.get(key);
  if (v !== undefined) { evalCache.delete(key); evalCache.set(key, v); }  // LRU 触碰
  return v;
}

function evalCachePut(key, val) {
  if (evalCache.has(key)) return;
  evalCache.set(key, val);
  if (evalCache.size > EVAL_CACHE_CAP) evalCache.delete(evalCache.keys().next().value);
}

/** 测试钩子:清空评估缓存 */
export function clearEvalCache() { evalCache.clear(); }

export const lookupEval = evalCacheGet;
export const storeEval = evalCachePut;

/* 双种子 FNV-1a:把特征缓冲定点哈希成字符串键。特征每次都按同一算法生成,
 * 位级一致(填 0 后哈希,未写段恒 0);双种子把碰撞概率压到可忽略。
 * sym(8 对称编号,缺省 0)并进键:同盘面不同对称 = 不同条目(等效 8 个
 * 子缓存,「同输入必同输出」不变量保持 —— 置换在引擎侧做,见 symmetry.js)。 */
export function fevalKey(sp, gl, sym = 0) {
  const u32sp = new Uint32Array(sp.buffer, sp.byteOffset, sp.byteLength >> 2);
  const u32gl = new Uint32Array(gl.buffer, gl.byteOffset, gl.byteLength >> 2);
  let h1 = 0x811c9dc5, h2 = 0x01000193;
  for (let i = 0; i < u32sp.length; i++) {
    h1 = Math.imul(h1 ^ u32sp[i], 16777619);
    h2 = Math.imul(h2 + u32sp[i] ^ (i * 2654435761 | 0), 16777619);
  }
  for (let i = 0; i < u32gl.length; i++) {
    h1 = Math.imul(h1 ^ u32gl[i], 16777619);
    h2 = Math.imul(h2 + u32gl[i] ^ (i * 40503 | 0), 16777619);
  }
  return `${h1},${h2},${sym | 0}`;
}
