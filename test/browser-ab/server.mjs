/* 浏览器 A/B 测试静态服务器:仓库根目录静态文件 + /report 结果落盘。
 * 用法:node test/browser-ab/server.mjs [port=8137]
 *   GET  /*        静态文件(.mjs/.js → text/javascript,.onnx/.aewn → octet-stream)
 *   GET  /ping     存活探针
 *   POST /report   body=完整状态 JSON → 写 test/browser-ab/results.json
 *                  并在 results.log 追加一行摘要(页面崩溃也不丢中间结果)
 */
import { createServer } from 'node:http';
import { readFile, writeFile, appendFile, mkdir } from 'node:fs/promises';
import { join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)));
const PORT = Number(process.argv[2] ?? 8137);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.aewn': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  if (url.pathname === '/ping') { res.writeHead(200); res.end('ok'); return; }
  if (url.pathname === '/report' && req.method === 'POST') {
    let body = '';
    for await (const ch of req) body += ch;
    try {
      const state = JSON.parse(body);
      await mkdir(OUT_DIR, { recursive: true });
      await writeFile(join(OUT_DIR, 'results.json'), JSON.stringify(state, null, 2));
      const last = state.events?.[state.events.length - 1];
      await appendFile(join(OUT_DIR, 'results.log'),
        `${new Date().toISOString()} ${state.phase ?? '?'} ${last?.text ?? ''}\n`);
      res.writeHead(200); res.end('saved');
    } catch (e) { res.writeHead(400); res.end(String(e)); }
    return;
  }
  try {
    let p = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '');
    const file = join(ROOT, p || 'index.html');
    if (!file.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
    const data = await readFile(file);
    /* COOP/COEP:线程化 WASM 后端(如 ort-web 多线程)需要 cross-origin isolation */
    res.writeHead(200, {
      'Content-Type': MIME[extname(file).toLowerCase()] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    });
    res.end(data);
  } catch {
    res.writeHead(404); res.end('not found');
  }
}).listen(PORT, '127.0.0.1', () => console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}`));
