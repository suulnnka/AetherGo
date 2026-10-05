/* ============================================================
 * AetherGo 对弈页 — 棋盘绘制(纯视图,无对局状态)
 *
 * 固定像素布局:格距 CS / 边距 PAD(与 style.css 的 --cs / --pad 必须
 * 一致,棋子/热区按这里算出的像素定位);19 路线 + 九星位 + 边缘坐标。
 * 缩放交给 app.js 的 fitBoard(.fit-wrap)。
 * 坐标与记谱常量来自 src/protocol.js(UI 与引擎的唯一共享层)。
 * ============================================================ */
import { N, COLS } from '../src/protocol.js';

/* 格距(px)。与 style.css 里的 --cs / --pad 必须一致 */
export const CS = 30, PAD = 24;
export const T = PAD * 2 + CS * (N - 1);
export const X = (c) => PAD + c * CS;
export const Y = (r) => PAD + r * CS;

/** 棋盘线(SVG):19×19 线 + 九个星位 + 边缘坐标(下 A~T、左 19~1) */
export function boardSvg() {
  const d = [];
  for (let i = 0; i < N; i++) {
    d.push(`M${X(0)} ${Y(i)}H${X(N - 1)}`);
    d.push(`M${X(i)} ${Y(0)}V${Y(N - 1)}`);
  }
  const s = 3;                               // 星位坐标(9 路 3/9/15 同族)
  const stars = [[s, s], [s, 9], [s, N - 1 - s], [9, s], [9, 9], [9, N - 1 - s],
    [N - 1 - s, s], [N - 1 - s, 9], [N - 1 - s, N - 1 - s]]
    .map(([r, c]) => `<circle class="go-star" cx="${X(c)}" cy="${Y(r)}" r="2.6"/>`)
    .join('');
  const coords = [];
  for (let c = 0; c < N; c++) {
    coords.push(`<text class="go-coord" x="${X(c)}" y="${T - PAD / 2}">${COLS[c]}</text>`);
    coords.push(`<text class="go-coord" x="${PAD / 2}" y="${Y(c)}">${N - c}</text>`);
  }
  return `<svg class="go-lines" viewBox="0 0 ${T} ${T}" aria-hidden="true">`
    + `<path class="go-line" d="${d.join(' ')}"/>${stars}${coords.join('')}</svg>`;
}
