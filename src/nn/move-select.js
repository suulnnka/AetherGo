/* ============================================================
 * AetherGo NN 搜索 — 最终选点(KataGo useLcbForSelection / 温度抽样)
 *
 * 从 search.js 拆出的纯选点逻辑,不含任何搜索状态:
 *   - pickBestLCB:温度 0 的默认选点 —— 「访问 ≥ 0.15×最大访问」的子里
 *     取置信下界最大者,方差用胜率近似 (1−Q²)/n(树里没有目差项,
 *     KataGo 还含 scoreStdev 项);
 *   - pickMove:温度 > 0 时按访问数^(1/T) 抽样,与 KataGo
 *     chooseIndexWithTemperature(onlyBelowProb=1 默认档)同分布;
 *   - effectiveTemperature:温度按手数半衰(interpolateEarly 的 late=0 特例)。
 * 导出供 nn-temp-test 分布校验。
 * ============================================================ */

const LCB_STDEVS = 5.0;                 // KataGo lcbStdevs
const LCB_MIN_VISIT_PROP = 0.15;        // KataGo minVisitPropForLCB

export function pickBest(root) {
  let best = null;
  for (const ch of root.children ?? []) {
    /* 图搜索:边访问数(该边分摊);旧形状回退 node.visits */
    const v = ch.edgeVisits ?? ch.node.visits;
    if (!best || v > (best.edgeVisits ?? best.node.visits)) best = ch;
  }
  return best;
}

/* LCB 最终选点(KataGo useLcbForSelection):在「访问 ≥ 0.15×最大访问」的
 * 子里取 Q − lcbStdevs·√(方差/n) 最大者,方差用胜率近似 (1−Q²)/n。
 * Q 相近时偏爱访问扎实、波动小的着法,减少「少访问高估」翻盘。 */
export function pickBestLCB(root, stdevs = LCB_STDEVS) {
  const chs = (root.children ?? []).filter((ch) => ch.node.visits > 0);
  if (!chs.length) return null;
  let maxV = 0;
  for (const ch of chs) if (ch.node.visits > maxV) maxV = ch.node.visits;
  let best = null, bestL = -Infinity;
  for (const ch of chs) {
    const n = ch.node.visits;
    if (n < maxV * LCB_MIN_VISIT_PROP) continue;
    const q = ch.node.wv / n;
    const lcb = q - stdevs * Math.sqrt(Math.max(1 - q * q, 1e-4) / n);
    if (lcb > bestL) { bestL = lcb; best = ch; }
  }
  return best ?? pickBest(root);
}

/* 温度半衰:temperatureHalflife > 0 时按手数从传入温度指数衰减
 * T_eff = T0 × 0.5^(numMoves/halflife) —— 即 KataGo interpolateEarly 的 late=0 特例;
 * 未传 halflife(或 ≤0)、温度本就 ≤1e-4 时原样返回。导出供测试。 */
export function effectiveTemperature(temperature, halflife, numMoves) {
  if (!(halflife > 0) || temperature <= 1e-4) return temperature;
  return temperature * Math.pow(0.5, numMoves / halflife);
}

/* 最终选点:temperature ≤ 1e-4 = 权重 argmax(并列取先展开者);
 * > 0 时在 log 空间按 (w/wmax)^(1/T) 加权随机抽,零权重子不参与 ——
 * 对齐 KataGo chooseIndexWithTemperature(onlyBelowProb=1 默认档)。
 * 图搜索:子条目带 edgeVisits(该边访问数),权重取它(缺省回退 node.visits,
 * 旧桩 / 旧调用方兼容)。导出供分布校验测试。 */
export function pickMove(root, temperature) {
  const chs = root.children ?? [];
  if (!chs.length) return null;
  const wOf = (ch) => (ch.edgeVisits ?? ch.node.visits) || 0;
  let best = chs[0];
  for (const ch of chs) if (wOf(ch) > wOf(best)) best = ch;
  if (temperature <= 1e-4 || wOf(best) <= 0) return best;
  const logMax = Math.log(wOf(best));
  let sum = 0;
  const weights = chs.map((ch) => {
    const w = wOf(ch);
    if (w <= 0) return 0;
    const x = Math.exp((Math.log(w) - logMax) / temperature);
    sum += x;
    return x;
  });
  let r = Math.random() * sum;
  for (let i = 0; i < chs.length; i++) {
    r -= weights[i];
    if (r < 0) return chs[i];
  }
  return best;                              // 浮点尾差兜底:回到 argmax
}
