/* ============================================================
 * AetherGo NN 特征 — 邻接与链洪泛共享层
 *
 * features.js(Benson 领土 / 前几手重演)与 ladder.js(征子搜索)共用
 * 同一套邻接表和链洪泛工作区 —— 只读棋盘,各函数串行使用互不嵌套。
 * ============================================================ */
import { N, N2, EMPTY } from '../protocol.js';

/* 4 邻居,-1 表示越界 */
export const NB_OFF = new Int32Array(N2 * 4).fill(-1);
for (let p = 0; p < N2; p++) {
  const r = (p / N) | 0, c = p % N;
  let n = 0;
  if (r > 0) NB_OFF[p * 4 + n++] = p - N;
  if (r < N - 1) NB_OFF[p * 4 + n++] = p + N;
  if (c > 0) NB_OFF[p * 4 + n++] = p - 1;
  if (c < N - 1) NB_OFF[p * 4 + n++] = p + 1;
}

export const adj = (a, b) => NB_OFF[a * 4] === b || NB_OFF[a * 4 + 1] === b
  || NB_OFF[a * 4 + 2] === b || NB_OFF[a * 4 + 3] === b;

/* 链洪泛:成员写 F_STONES、气点写 F_LIBS(长度挂 .len)—— 只读棋盘 */
export const F_STONES = new Int32Array(N2), F_LIBS = new Int32Array(N2);
const F_SEEN = new Int32Array(N2), F_LSEEN = new Int32Array(N2);
let fStamp = 0;
/** ladder.js 的 lLibsAfterPlay 需要自管 stamp 与可见性标记 */
export const nextFStamp = () => ++fStamp;
export { F_SEEN, F_LSEEN };
export function chainFlood(bd, seed) {
  const v = bd[seed];
  const st = ++fStamp;
  let ns = 1, nl = 0;
  F_STONES[0] = seed; F_SEEN[seed] = st;
  for (let h = 0; h < ns; h++) {
    const q = F_STONES[h];
    for (let k = 0; k < 4; k++) {
      const r = NB_OFF[q * 4 + k];
      if (r < 0) continue;
      const c = bd[r];
      if (c === EMPTY) {
        if (F_LSEEN[r] !== st) { F_LSEEN[r] = st; F_LIBS[nl++] = r; }
      } else if (c === v && F_SEEN[r] !== st) { F_SEEN[r] = st; F_STONES[ns++] = r; }
    }
  }
  F_STONES.len = ns; F_LIBS.len = nl;
  return ns;
}
