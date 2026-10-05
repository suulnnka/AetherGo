/* pickMove 温度选点单元测试(桩 session,不起 NN)
 * 用法:node test/nn-temp-test.mjs
 * 断言:
 *   A. 缺省/温度 0 = argmax 且完全可复现(两次搜索同一着点)
 *   B. 温度 1 = 按访问数比例抽样(强着点高频、弱着点也会出现)
 *   C. 温度 0.15 = 大概率 argmax,偶尔第二名
 *   D. 抽样分布与 KataGo chooseIndexWithTemperature(onlyBelowProb=1)同分布:
 *      P_i ∝ (v_i/v_max)^(1/T)
 */
import { N2, newBoard, PASS } from '../src/engine.js';
import { nnSearchBest } from '../src/nn/search.js';

/* 桩 evalBatch:policy 固定偏好(40 号点最强、41 次之、42 再次),winLoss 恒 0.3。
 * 尾部着法填 −20:真实网络的 policy 尾部就是这样尖(先验 ~1e-9,低于剪枝地板),
 * 否则 361 个等先验点会把访问摊平,LCB 在噪声里选点 —— 那是桩的病理,不是引擎的。 */
function makeStubSession() {
  return {
    async evalBatch(items) {
      return items.map(() => {
        const policy = new Float32Array(N2).fill(-20);
        policy[40] = 5.0; policy[41] = 4.0; policy[42] = 3.0;
        return { policy, policyPass: 0.0, winLoss: 0.3 };
      });
    },
  };
}

async function search(opts) {
  return nnSearchBest(newBoard(), 0, {
    session: makeStubSession(), visits: 120, batch: 4, reuseTree: false,
    allowResign: false, ...opts,
  });
}

let failed = 0;
function check(name, cond) {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}`);
  if (!cond) failed++;
}

/* A. 缺省与温度 0:LCB 选点(KataGo useLcbForSelection)、可复现 */
const a1 = await search({});
const a2 = await search({ temperature: 0 });
check(`A1 缺省=LCB 得 40(实际 ${a1.move})`, a1.move === 40);
check(`A2 两次温度0结果一致(${a1.move},${a2.move})`, a1.move === a2.move);

/* B. 温度 1:按访问数比例抽样。200 次独立搜索观察抽样多样性 */
const dist = new Map();
for (let i = 0; i < 200; i++) {
  const r = await search({ temperature: 1, visits: 120 });
  dist.set(r.move, (dist.get(r.move) ?? 0) + 1);
}
const seen = [...dist.keys()].sort((x, y) => dist.get(y) - dist.get(x));
console.log('   温度1 抽样分布:', seen.map((m) => `${m}×${dist.get(m)}`).join(' '));
check('B1 温度1 不再恒为 40', dist.size >= 2);
check('B2 40 仍是最高频', dist.get(40) === Math.max(...dist.values()));
check('B3 弱着点(42)也出现过', dist.has(42));

/* C. 温度 0.15:强收敛。200 次里绝大多数为 40 */
const distC = new Map();
for (let i = 0; i < 200; i++) {
  const r = await search({ temperature: 0.15, visits: 120 });
  distC.set(r.move, (distC.get(r.move) ?? 0) + 1);
}
console.log('   温度0.15 分布:', [...distC.entries()].map(([m, c]) => `${m}×${c}`).join(' '));
check('C1 温度0.15 大概率 argmax(≥85%)', (distC.get(40) ?? 0) >= 170);

/* D. 分布数值校验:合成访问数 100:60:40:1,大样本抽样,频率应 ≈ P ∝ (v/vmax)^(1/T) */
import { pickMove } from '../src/nn/search.js';

function fakeRoot(visits) {
  return { children: visits.map((v) => ({ node: { visits: v } })) };
}

async function sampleFreq(visits, T, n) {
  const root = fakeRoot(visits);
  const counts = new Array(visits.length).fill(0);
  for (let i = 0; i < n; i++) {
    const ch = pickMove(root, T);
    if (!ch) return null;
    counts[visits.indexOf(ch.node.visits)]++;
  }
  return counts;
}

{
  const visits = [100, 60, 40, 1];
  /* 20 万样本:最稀桶(理论 0.5%)的相对标准差 ≈ 3.2%,8% 阈值才稳定 */
  const N = 200000;
  const c1 = await sampleFreq(visits, 1, N);
  const theo = visits.map((v) => v / visits.reduce((a, b) => a + b, 0));
  const got = c1.map((c) => c / N);
  const dev = got.map((g, i) => Math.abs(g - theo[i]) / theo[i]);
  console.log(`   D 温度1 实测频率 [${got.map((g) => g.toFixed(3))}] vs 理论 [${theo.map((g) => g.toFixed(3))}]`);
  check('D1 温度1 各点频率与理论偏差 <8%', dev.every((d) => d < 0.08));

  const c15 = await sampleFreq(visits, 0.15, N);
  /* 理论分布:P ∝ (v/vmax)^(1/T),按公式现算而不是拍阈值 */
  const vmax = Math.max(...visits);
  const w = visits.map((v) => Math.exp((Math.log(v) - Math.log(vmax)) / 0.15));
  const sw = w.reduce((a, b) => a + b, 0);
  const theo15 = w.map((x) => x / sw);
  const dev15 = c15.map((c, i) => Math.abs(c / N - theo15[i]) / Math.max(theo15[i], 1e-9));
  console.log(`   D 温度0.15 实测 [${c15.map((c) => (c / N).toFixed(4))}] vs 理论 [${theo15.map((g) => g.toFixed(4))}]`);
  check('D2 温度0.15 频率与理论分布吻合(argmax 偏差<1%,弱点不出现)', dev15[0] < 0.01 && c15[3] === 0);

  const c0 = await sampleFreq(visits, 0, 5);
  check('D3 温度0 恒 argmax', c0[0] === 5 && c0[1] === 0 && c0[2] === 0 && c0[3] === 0);
}

/* E. 温度半衰:按手数从传入温度减半(T_eff = T0 × 0.5^(手数/半衰期)) */
import { effectiveTemperature } from '../src/nn/search.js';

check('E1 未传半衰期:温度原样', effectiveTemperature(0.7, 0, 40) === 0.7 && effectiveTemperature(0.7, undefined, 40) === 0.7);
check('E2 半衰公式:1,19,57 手 → 0.125', Math.abs(effectiveTemperature(1, 19, 57) - 0.125) < 1e-12);
check('E3 手数为 0:不衰减', effectiveTemperature(1, 19, 0) === 1);
check('E4 温度 0:不受半衰影响', effectiveTemperature(0, 19, 100) === 0);

const rOpen = await search({ temperature: 1, temperatureHalflife: 19, visits: 120 });
check('E5 开局(0 手)返回有效温度=1', rOpen.temperature === 1);
const seq57 = Array.from({ length: 57 }, (_, i) => i);
const r57 = await search({ temperature: 1, temperatureHalflife: 19, visits: 120, recentMoves: seq57 });
check('E6 57 手后返回有效温度=0.125', Math.abs(r57.temperature - 0.125) < 1e-12);

let spread0 = 0, spread57 = 0;
for (let i = 0; i < 150; i++) {
  if ((await search({ temperature: 1, temperatureHalflife: 19, visits: 120 })).move !== 40) spread0++;
  if ((await search({ temperature: 1, temperatureHalflife: 19, visits: 120, recentMoves: seq57 })).move !== 40) spread57++;
}
console.log(`   E 非argmax率:0手 ${spread0}/150,57手 ${spread57}/150`);
check('E7 半衰后抽样显著收敛(非argmax率减半以上)', spread57 * 2 <= spread0);

console.log(failed ? `\n${failed} 项失败` : '\n全部通过');
process.exit(failed ? 1 : 0);
