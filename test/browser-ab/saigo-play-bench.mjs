/* 用 CDP 驱动 Edge 顶层窗口,在 saigo.online(代理)的 /play 对战页实测 Sharp 档搜索引擎吞吐。
 * 流程:选 19x19 + Sharp → 开始对局 → 点天元(黑) → 等引擎白方应手 → 计时。
 * 用法:node test/browser-ab/saigo-play-bench.mjs */
import WebSocket from 'ws';

const CD_PORT = 9223;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const pages = await (await fetch(`http://127.0.0.1:${CD_PORT}/json`)).json();
const page = pages.find((p) => p.type === 'page' && p.url.includes('/play'));
if (!page) { console.error('play page not found:', pages.map((p) => p.url)); process.exit(1); }
const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
await new Promise((r) => ws.on('open', r));

let seq = 0;
const pending = new Map();
ws.on('message', (d) => {
  const m = JSON.parse(d);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
function send(method, params = {}) {
  const id = ++seq;
  return new Promise((res) => { pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });
}
async function evalJs(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400));
  return r.result?.result?.value;
}

await send('Page.enable');
await send('Runtime.enable');

/* 1. 等引擎就绪(badge 不再是 loading/heuristic) */
console.log('等待引擎就绪…');
let engineName = '';
for (let i = 0; i < 60; i++) {
  engineName = await evalJs(`(document.body.innerText.match(/引擎：[^\\n]{0,40}|engine:[^\\n]{0,40}/)||[''])[0]`);
  if (engineName && !/loading|heuristic|加载/.test(engineName)) break;
  await sleep(2000);
}
console.log('引擎:', engineName);

/* 2. 配置:19x19 + Sharp + 你执黑 */
await evalJs(`(() => {
  const setSel = (label, valueText) => {
    for (const sel of document.querySelectorAll('select')) {
      const opt = [...sel.options].find(o => o.text.includes(valueText));
      if (opt) { sel.value = opt.value; sel.dispatchEvent(new Event('change', { bubbles: true })); return true; }
    }
    return false;
  };
  setSel(null, '普通围棋');
  setSel(null, '19×19');
  setSel(null, '黑');
  setSel(null, 'Sharp');
  return 'ok';
})()`);
await sleep(500);
console.log('配置完成:', await evalJs(`[...document.querySelectorAll('select')].map(s => s.selectedOptions[0]?.text).join(' / ')`));

/* 3. 开始对局 */
const clicked = await evalJs(`(() => {
  for (const b of document.querySelectorAll('button')) {
    if (/开始|Start|对局/.test(b.textContent) && !b.disabled) { b.click(); return b.textContent.trim(); }
  }
  return null;
})()`);
console.log('开始按钮:', clicked);
await sleep(1500);

/* 4. 点棋盘中心(天元)走黑棋;棋盘可能是 canvas 或 DOM 格点 */
const boardInfo = await evalJs(`(() => {
  const c = document.querySelector('canvas');
  if (c) { const r = c.getBoundingClientRect(); return { kind: 'canvas', x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height }; }
  return { kind: 'none' };
})()`);
console.log('棋盘:', JSON.stringify(boardInfo));
if (boardInfo.kind !== 'canvas') { console.error('未找到 canvas 棋盘'); process.exit(1); }

/* 5. 状态探针:轮到谁 / 手数 */
const stateExpr = `(() => {
  const t = document.body.innerText;
  const turn = /轮到你|你的回合|你执黑.*等待|Your turn/i.test(t) ? 'player' : (/思考|thinking|Engine/i.test(t) ? 'engine' : 'unknown');
  return turn;
})()`;

/* 点天元(黑第一手) */
const before = await evalJs(stateExpr);
console.log('落子前状态:', before);
await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: Math.round(boardInfo.x), y: Math.round(boardInfo.y), button: 'left', clickCount: 1 });
await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: Math.round(boardInfo.x), y: Math.round(boardInfo.y), button: 'left', clickCount: 1 });
console.log('已点天元,等待引擎 Sharp(320 visits)应手…');

/* 6. 轮询引擎完成(回到玩家回合)并计时;顺带抓每手耗时 */
const t0 = Date.now();
let elapsed = 0, done = false;
for (let i = 0; i < 240; i++) {
  await sleep(1000);
  elapsed = Date.now() - t0;
  const st = await evalJs(stateExpr).catch(() => 'unknown');
  if (st === 'player') { done = true; break; }
  if (i % 5 === 0) process.stdout.write(`  ${elapsed}s 状态=${st}\r\n`);
}
console.log(done ? `引擎应手完成: ${(elapsed / 1000).toFixed(1)}s` : '超时未完成');
if (done) {
  const info = await evalJs(`(() => {
    const t = document.body.innerText;
    return (t.match(/手数[：:]?\\s*[0-9]+|move\\s*[0-9]+|第\\s*[0-9]+\\s*手/g) || []).join(' ');
  })()`);
  console.log('局面信息:', info);
  console.log(`Sharp=320 visits → ${(320 / (elapsed / 1000)).toFixed(1)} visits/s(若真跑满 visits)`);
}
process.exit(0);
