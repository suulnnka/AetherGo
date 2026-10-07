/* 点天元 → 以「轮到你了」消失/重现计时引擎 Sharp(320v)应手 */
import WebSocket from 'ws';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pages = await (await fetch('http://127.0.0.1:9223/json')).json();
const page = pages.find((p) => p.type === 'page' && p.url.includes('/play'));
const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
await new Promise((r) => ws.on('open', r));
let seq = 0; const pending = new Map();
ws.on('message', (d) => { const m = JSON.parse(d); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => { const id = ++seq; return new Promise((res) => { pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); }); };
const evalJs = async (expr) => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }))?.result?.result?.value;
const MY_TURN = `document.body.innerText.includes('轮到你了')`;
const CAPS = `(() => { const m = document.body.innerText.match(/提子[^\\n]{0,20}/); return m ? m[0] : ''; })()`;

if (await evalJs(MY_TURN) !== true) { console.log('当前不是玩家回合,先处理局面'); }
const b = await evalJs(`(() => { const s = document.querySelector('svg[viewBox]'); const r = s.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; })()`);
const cell = b.w / 20; /* SVG 含半格 padding:20 格划分 */
const cx = Math.round(b.x + cell * 16), cy = Math.round(b.y + cell * 4); /* (3,3) 星位 */
await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: cx, y: cy });
await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: cx, y: cy, button: 'left', clickCount: 1 });
await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: cx, y: cy, button: 'left', clickCount: 1 });

/* 等待「轮到你了」消失(点击被接受,引擎开始) */
let started = false;
for (let i = 0; i < 20; i++) {
  await sleep(250);
  if (await evalJs(MY_TURN) === false) { started = true; break; }
}
if (!started) { console.log('点击未被接受(仍是我的回合)——可能该点已占或热区偏移'); process.exit(1); }
const t0 = Date.now();
console.log('引擎开始应手(Sharp 320 visits)…');

/* 等待「轮到你了」重现 */
for (let i = 0; i < 600; i++) {
  await sleep(500);
  if (await evalJs(MY_TURN) === true) {
    const el = (Date.now() - t0) / 1000;
    const caps = await evalJs(CAPS);
    console.log(`引擎应手完成: ${el.toFixed(1)}s | ${caps}`);
    console.log(`Sharp 320 visits → ${(320 / el).toFixed(1)} visits/s`);
    process.exit(0);
  }
  if (i % 20 === 0) console.log(`  ${((Date.now() - t0) / 1000).toFixed(0)}s…`);
}
console.log('超时');
