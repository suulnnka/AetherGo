/* saigo.online 本地代理:转发全部请求,给 HTML/全部响应补 COOP/COEP,
 * 让需要 cross-origin isolation 的线程化 WASM 引擎在 IAB 里可用。
 * 用法:node test/browser-ab/saigo-proxy.mjs [port=8140]  →  http://127.0.0.1:8140/play */
import https from 'node:https';
import { createServer } from 'node:http';

const PORT = Number(process.argv[2] ?? 8140);
const UP = 'saigo.online';

createServer((req, res) => {
  const opts = {
    hostname: UP,
    port: 443,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: UP, 'accept-encoding': req.headers['accept-encoding'] ?? 'gzip' },
  };
  const up = https.request(opts, (ur) => {
    const h = { ...ur.headers };
    h['cross-origin-opener-policy'] = 'same-origin';
    h['cross-origin-embedder-policy'] = 'credentialless';
    h['access-control-allow-origin'] = '*';
    delete h['content-security-policy'];
    res.writeHead(ur.statusCode ?? 502, h);
    ur.pipe(res);
  });
  up.on('error', (e) => { try { res.writeHead(502); res.end('proxy error: ' + e.message); } catch { /* 已响应 */ } });
  req.pipe(up);
}).listen(PORT, '127.0.0.1', () => console.log(`saigo proxy at http://127.0.0.1:${PORT}`));
