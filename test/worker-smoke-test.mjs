/* Worker 消息层冒烟测试:不起 WebGPU,桩会话直注,驱动完整 think 消息流。
 *
 * 背景(2026-10-05):认输判定块引用了未 import 的 WHITE,think 每次在搜索
 * 完成后、发着法前抛 ReferenceError —— 异步 handler 的异常不进页面
 * worker.onerror,UI 永远「思考中」。搜索层测试全绿也没用,因为这条
 * 「搜索后路径」只有消息层才能走到。本测试钉住这一层:
 *
 *   A. think 全流程:桩会话 + 直出档(1 访问 = 模型直出)→ 必须回着法(不是 error);
 *   B. 错误必达:会话抛错时 think 回 error 消息(UI 能显示,不再无声);
 *   C. levels / state:规则查询路径不受会话注入影响。
 *
 * 运行:node test/worker-smoke-test.mjs
 */
const sent = [];
let resolveReply = null;

/* 假 self:收下 postMessage,think 回包唤醒等待方 */
globalThis.self = {
  __tag: null,
  onmessage: null,
  postMessage(msg) {
    if (msg.type === 'progress' || msg.type === 'status') return;   // 同 id 过程消息:终态才算回包
    sent.push(msg);
    if (resolveReply) {
      const r = resolveReply; resolveReply = null; r(msg);
    }
  },
};

const worker = await import('../src/nn-worker.js');

let failed = 0;
const check = (name, cond, extra) => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  — ' + (extra ?? '')}`);
  if (!cond) failed++;
};

async function send(data) {
  return new Promise((resolve) => {
    resolveReply = resolve;
    self.onmessage({ data });
  });
}

/* 桩会话:policy 集中天元,winLoss 恒定(katago-align-test makeStub 同款口径) */
import { N2, PASS } from '../src/engine.js';
const stubSession = {
  maxBatch: 4,
  calls: 0,
  async evalBatch(items) {
    this.calls += items.length;
    return items.map(() => {
      const policy = new Float32Array(N2).fill(-20);
      policy[180] = 5;                              // 天元
      return { policy, policyPass: -20, winLoss: 0.3 };
    });
  },
};
worker.__setSessionForTest(stubSession);

/* ==================== A. think 全流程(含认输判定等搜索后路径) ==================== */
{
  const r = await send({ id: 1, type: 'think', level: 0, moves: [] });
  check('A1 think 回着法(非 error)', r && r.move !== undefined && !r.error, r?.error ?? JSON.stringify(r));
  check('A2 着法合法(0..360 或 PASS)', r.move >= 0 && (r.move < N2 || r.move === PASS), String(r.move));
  check('A3 直出档访问数 = 1', r.visits === 1, String(r.visits));
  check('A4 胜率域有效', r.winRate > 0 && r.winRate < 1, String(r.winRate));
  /* 初级档(16)走正常搜索路径,访问数同样用满 */
  const r2 = await send({ id: 4, type: 'think', level: 1, moves: [] });
  check('A5 初级档访问数用满(16)', r2.visits === 16 && !r2.error, String(r2.visits) + ' ' + (r2.error ?? ''));
}

/* ==================== B. 会话抛错 → error 消息必达 ==================== */
{
  worker.__setSessionForTest({
    maxBatch: 4,
    async evalBatch() { throw new Error('boom'); },
  });
  const r = await send({ id: 2, type: 'think', level: 0, moves: [] });
  check('B1 会话抛错回 error(UI 可见)', !!r?.error, JSON.stringify(r));
  check('B2 error 带上原因', /boom/.test(r?.error ?? ''), r?.error);
  worker.__setSessionForTest(stubSession);
}

/* ==================== C. 规则查询路径 ==================== */
{
  const lv = await send({ type: 'levels' });
  check('C1 levels 自报七档', lv?.levels?.length === 7 && !!lv.default);
  const st = await send({ id: 3, type: 'state', moves: [180] });
  check('C2 state 重演合法序列', st && !st.error && st.board?.[180] === 1, st?.error);
}

console.log(failed ? `\n✗ ${failed} 项失败` : '\n✓ 全部通过');
process.exit(failed ? 1 : 0);
