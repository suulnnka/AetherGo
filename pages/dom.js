/* ============================================================
 * AetherGo 对弈页 — 通用 DOM 工具(替代 webos 的 core)
 *
 * 只依赖文档结构,不含任何对局逻辑:el 建节点、icon 线性图标、
 * toast 吐司、系统对话框 #dlg 的接线。数子窗口 #countDlg 的元素引用
 * 也在这里(按钮行为属于 app.js 的对局控制器)。
 * ============================================================ */

export const $ = (sel) => document.querySelector(sel);

/** 建 DOM:el('button', {class, onClick, dataset}, ...children) */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** 线性图标(路径数据取自 webos 的 core/icons.js) */
const ICON_PATHS = {
  refresh: '<path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/>',
  reply: '<polyline points="9 17 4 12 9 7"/><path d="M20 18v-2a4 4 0 0 0-4-4H4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
};
export const icon = (name) => {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('fill', 'none');
  s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '2');
  s.setAttribute('stroke-linecap', 'round');
  s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = ICON_PATHS[name] || '';
  return s;
};

/** webos dialogs.info 的页内替身:系统对话框 */
export const dlg = $('#dlg');
export function showDialog({ title, message }) {
  $('#dlgTitle').textContent = title;
  $('#dlgMsg').textContent = message;
  if (!dlg.open) dlg.showModal();
}
$('#dlgOk').addEventListener('click', () => dlg.close());
dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });

/* 数子窗口 #countDlg:元素引用(按钮行为在 app.js 对局控制器里接线) */
export const countDlg = $('#countDlg');
export const countMsgEl = $('#countMsg');
export const countExitBtn = $('#countExitBtn');

/** webos bus.notify 的页内替身:右下角吐司 */
export function toast(text) {
  const t = el('div', { class: 'toast' }, text);
  t.addEventListener('click', () => t.remove());
  $('#toasts').append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 220); }, 3200);
}
