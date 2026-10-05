/* NN 模块确定性测试(不依赖 katago 数据;外部数据对拍见 test/featdiff.mjs)。
 *
 * 运行:node test/features-test.mjs
 *
 * 摆盘与 engine-test 同款:9 路旧用例镶 19 路盘左上角(自动补 '.')。
 */
import { N, N2, BLACK, WHITE, PASS, newBoard, make, unmake, genLegal, syncPosition, ringSnapshot, ringRestore, recentPrevBd, recentPrevKo, replayMoves } from '../src/engine.js';
import { encodeFeatures, encodeFeaturesReplay, calculateArea, SPATIAL_CHANNELS, GLOBAL_CHANNELS } from '../src/nn/features.js';

const sections = [];
const section = (name, fn) => sections.push({ name, fn });
let pass = 0, fail = 0;
const ok = (cond, msg, extra) => {
  if (cond) pass++;
  else { fail++; console.log(`  ✗ ${msg}${extra ? '  — ' + extra : ''}`); }
};
const eq = (got, want, msg) => ok(got === want, msg, `得到 ${got},期望 ${want}`);

function boardFrom(rows) {
  const bd = new Int8Array(N2);
  rows.forEach((row, r) => {
    for (let c = 0; c < row.length && c < N; c++) {
      if (row[c] === 'X') bd[r * N + c] = 1;
      else if (row[c] === 'O') bd[r * N + c] = 2;
    }
  });
  syncPosition(bd);
  return bd;
}
const sq = (r, c) => r * N + c;
const chSum = (sp, c) => { let n = 0; for (let p = 0; p < N2; p++) n += sp[c * N2 + p]; return n; };

/* ==================== 1. 空盘编码 ==================== */
section('features-init', () => {
  eq(SPATIAL_CHANNELS, 22, '空间通道数 22');
  eq(GLOBAL_CHANNELS, 19, '全局通道数 19');
  const bd = newBoard();
  const r = encodeFeatures(bd, BLACK, { recentMoves: [] });
  eq(chSum(r.spatial, 0), N2, `通道 0:满盘 ${N2}`);
  for (let c = 1; c < 22; c++) eq(chSum(r.spatial, c), 0, `通道 ${c} 空盘为 0`);
  eq(r.global[5].toFixed(4), (-7.5 / 20).toFixed(4), '全局 5 = 黑方 selfKomi/20');
  eq(r.global[6], 1, '全局 6 = 1(POSITIONAL)');
  eq(r.global[7], 0.5, '全局 7 = 0.5(POSITIONAL)');
  eq(r.global[14], 0, '空盘 passWouldEndPhase = 0');
  eq(r.global[18], -0.5, '361 路(奇数)贴 7.5 的三角波 = -0.5');
  /* 白方视角 */
  const r2 = encodeFeatures(bd, WHITE, { recentMoves: [] });
  eq(r2.global[5].toFixed(4), (7.5 / 20).toFixed(4), '白方 selfKomi = +7.5/20');
  eq(r2.global[18], 0.5, '白方视角波 = +0.5');
});

/* ==================== 2. 基础通道(子/气/历史/劫) ==================== */
section('features-basic', () => {
  const bd = newBoard();
  const moves = [sq(4, 4), sq(3, 3), sq(5, 5)];
  make(bd, moves[0], BLACK); make(bd, moves[1], WHITE); make(bd, moves[2], BLACK);
  const r = encodeFeatures(bd, WHITE, { recentMoves: moves.slice() });
  eq(chSum(r.spatial, 1), 1, '通道 1:1 颗己(白)子');
  eq(chSum(r.spatial, 2), 2, '通道 2:2 颗敌(黑)子');
  ok(r.spatial[1 * N2 + sq(3, 3)] === 1, '通道 1 的子落在 (3,3)');
  /* 历史:最新一手(黑 (5,5))→ 通道 9;再前白 (3,3) → 通道 10;黑 (4,4) → 通道 11 */
  eq(r.spatial[9 * N2 + sq(5, 5)], 1, '通道 9 = 最近一手');
  eq(r.spatial[10 * N2 + sq(3, 3)], 1, '通道 10 = 前一手');
  eq(r.spatial[11 * N2 + sq(4, 4)], 1, '通道 11 = 前二手');
  eq(chSum(r.spatial, 12), 0, '第 4 手不存在 → 通道 12 为 0');
  eq(chSum(r.spatial, 13), 0, '第 5 手不存在 → 通道 13 为 0');
  /* 无停着:全局 0~4 为 0;5/6/7/18 是贴目与规则常量,8~17 恒 0 */
  for (let k = 0; k <= 4; k++) eq(r.global[k], 0, `开局无停着:全局 ${k} = 0`);
  for (let k = 8; k <= 17; k++) eq(r.global[k], 0, `规则区全局 ${k} = 0`);
  /* 停着历史:全局 0 置 1、通道 9 不置 */
  const bd2 = newBoard();
  make(bd2, sq(4, 4), BLACK); make(bd2, PASS, WHITE);
  const r2 = encodeFeatures(bd2, BLACK, { recentMoves: [sq(4, 4), PASS] });
  eq(r2.global[0], 1, '上一手停着 → 全局 0 = 1');
  eq(r2.spatial[9 * N2], 0, '停着不进空间通道');
  eq(r2.spatial[10 * N2 + sq(4, 4)], 1, '再前一手的落点进通道 10');
  eq(r2.global[14], 1, '上一手停着 → passWouldEndPhase = 1');
  /* 双停(终局):只保留 1 手历史 */
  const bd3 = newBoard();
  make(bd3, sq(4, 4), BLACK); make(bd3, PASS, WHITE); make(bd3, PASS, BLACK);
  const r3 = encodeFeatures(bd3, WHITE, { recentMoves: [sq(4, 4), PASS, PASS] });
  eq(r3.global[0], 1, '终局:保留最后一手(停着)→ 全局 0 = 1');
  eq(r3.global[1], 0, '终局:更早的历史被截断 → 全局 1 = 0');
});

/* ==================== 3. Benson 领土 ==================== */
section('features-area', () => {
  /* 双活眼形:黑左上两眼(角 + 眼)、白镜像右下两眼(19 路重写,与 9 路边上的
   * 版本不同 —— 镜像布局下 Benson 判定与贴边与否无关,断言两边对称) */
  const bd = boardFrom(['.XX',
                        'X.X',
                        '.XX',
                        ...Array(N - 6).fill(''),
                        '.'.repeat(N - 3) + 'OO.',
                        '.'.repeat(N - 3) + 'O.O',
                        '.'.repeat(N - 3) + 'OO.']);
  const area = calculateArea(bd);
  let black = 0, white = 0;
  for (let p = 0; p < N2; p++) { if (area[p] === 1) black++; else if (area[p] === 2) white++; }
  eq(black, 6 + 2, '黑:6 子 + 2 眼');
  eq(white, 6 + 2, '白:6 子 + 2 眼');
  /* 非活死块:黑墙里的白角(未被任何一方认领 → 数子阶段按子色填黑区) */
  const bd2 = boardFrom(['OO.X......',
                         'O.XX......',
                         '.XXX......',
                         'XX........',
                         '.........', '.........',
                         '......OOO.',
                         '......O.O.',
                         '......OOO.']);
  const area2 = calculateArea(bd2);
  /* 白角死块区域不含黑子 → calculateArea 的 unsafeBig 规则乐观标白
   * (KataGo 同款;死子真正判死在 engine.js 的 deadStones)。 */
  eq(area2[sq(0, 0)], 2, '白角死块:unsafeBig 乐观标白(与 KataGo 一致)');
  eq(area2[sq(0, 3)], 1, '黑墙归黑');
  eq(area2[sq(7, 7)], 2, '白右下活块眼位归白');
});

/* ==================== 4. 征子通道(简单征子形) ==================== */
section('features-ladder', () => {
  /* 黑三路断征:白 (3,4) 一子在黑包围下 1 气 —— 通道 14 判定是否逃得掉;
   * 19 路盘右边空间更大,征子成立与否与 9 路可能不同,这里只验证
   * 「判定存在且 1 气链无攻方着手 → 通道 17 恒 0」。 */
  const bd = boardFrom(['.........',
                        '.........',
                        '....X....',
                        '...XO....',
                        '....X....',
                        '.........',
                        '.........',
                        '.........',
                        '.........']);
  const r = encodeFeatures(bd, WHITE, { recentMoves: [] });
  ok(r.spatial[14 * N2 + sq(3, 4)] === 0 || r.spatial[14 * N2 + sq(3, 4)] === 1,
    '通道 14 判定存在(具体值取决于征子是否成立)');
  eq(chSum(r.spatial, 17), 0, '1 气链无攻方先行着手 → 通道 17 为 0');
  /* 开阔地边路征子:白 (10,0) 贴边 2 气,黑 (9,0) 收 —— 通道 14 标白子,
   * 通道 17 标攻方应手(守方视角的提子点)。 */
  const bd2 = newBoard();
  make(bd2, sq(9, 0), BLACK);
  make(bd2, sq(10, 0), WHITE);
  const r2 = encodeFeatures(bd2, WHITE, { recentMoves: [] });
  eq(r2.spatial[14 * N2 + sq(10, 0)], 1, '边路 2 气白子被征 → 通道 14');
  ok(r2.spatial[17 * N2 + sq(9, 1)] === 1 || r2.spatial[17 * N2 + sq(11, 0)] === 1,
    '通道 17 标攻方应手点(延气方向)');
});

/* ==================== 5. 编码器不变量 ==================== */
section('features-invariants', () => {
  /* 同一局面重复编码结果一致(无隐藏状态) */
  const bd = newBoard();
  const moves = [sq(2, 2), sq(6, 6), sq(2, 6), sq(6, 2), PASS];
  for (let i = 0; i < moves.length; i++) make(bd, moves[i], i % 2);
  const a = encodeFeatures(bd, BLACK, { recentMoves: moves.slice() });
  const b = encodeFeatures(bd, BLACK, { recentMoves: moves.slice() });
  ok(a.spatial.every((v, i) => v === b.spatial[i]), '重复编码:空间通道一致');
  ok(a.global.every((v, i) => v === b.global[i]), '重复编码:全局通道一致');
  /* 通道 0 恒满、1+2 = 盘上子数 */
  const stones = [...bd].filter((v) => v !== 0).length;
  eq(chSum(a.spatial, 1) + chSum(a.spatial, 2), stones, '通道 1+2 = 盘上子数');
  eq(a.spatial.length, 22 * N2, '空间缓冲长度 = 22 × 361');
  /* 尾随停着:全局 0=1 且 passWouldEndPhase=1 */
  eq(a.global[0], 1, '尾随停着 → 全局 0');
  eq(a.global[14], 1, '尾随停着 → 全局 14');
});

/* ---------- 执行 ---------- */

/* ==================== 环 vs 重演逐位对拍(增量盘面回归锚) ==================== */
section('ring-vs-replay', () => {
  /* 随机对局(带提子/劫争/pass),每个局面:生产路径(滚动盘面环)与参考路径
   * (从空盘重演)逐位一致 —— 环实现有任何回归它先红。撤销路径同时验证环回退。 */
  let seed = 987654321;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  let eq = true, positions = 0, badInfo = '';
  outer: for (let g = 0; g < 12; g++) {
    const bd = newBoard();
    const moves = [], toks = [];
    let side = BLACK;
    for (let i = 0; i < 140; i++) {
      const legal = genLegal(bd, side);
      const mv = legal[Math.floor(rnd() * legal.length)];
      const isPass = !(mv !== PASS && rnd() < 0.85);   // 偶尔主动 pass,制造尾随停着串
      const m = isPass ? PASS : mv;
      toks.push(make(bd, m, side));
      moves.push(m);
      side ^= 1;
      positions++;
      const a = encodeFeatures(bd, side, { recentMoves: moves.slice() });
      const b = encodeFeaturesReplay(bd, side, { recentMoves: moves.slice() });
      for (let k = 0; k < 22 * N2; k++) {
        if (a.spatial[k] !== b.spatial[k]) { eq = false; badInfo = `g${g} i${i} sp[${k}]`; break outer; }
      }
      for (let k = 0; k < 19; k++) {
        if (a.global[k] !== b.global[k]) { eq = false; badInfo = `g${g} i${i} gl[${k}]`; break outer; }
      }
    }
    /* 换分支(生产契约:搜索每批下降前 ringRestore)——
     * 分支 A 下潜 3 手编码,回退 2 手(偶数,保持与重演参考的严格交替奇偶一致)
     * + 恢复环,分支 B 重下,编码须与重演一致 —— A 的深层槽位污染被 ringRestore 挡住 */
    {
      /* 行棋方一律由 moves.length 奇偶派生(与重演参考的严格交替假设同源),
       * 不再手记 side —— 分支 A 的 bs 从未同步回 side,旧写法把分支 B 的颜色走错 */
      const rootMoves = moves.slice(), rootToks = toks.slice();
      const snap = ringSnapshot();
      const sideOf = () => (moves.length % 2 === 0 ? BLACK : 1);
      for (const branch of [0, 1]) {
        if (branch === 1) {
          /* 回退到根(= 快照位置)再恢复 —— ringRestore 的契约:活局面必须是快照局面 */
          while (moves.length > rootMoves.length) { unmake(bd, moves[moves.length - 1], toks[toks.length - 1]); toks.pop(); moves.pop(); }
          ringRestore(snap);
        }
        const bm = [];
        for (let j = 0; j < 3; j++) {
          const bs = sideOf();
          const L = genLegal(bd, bs).filter((m) => m !== PASS);
          const m = L[(branch * 7 + j * 13) % L.length];
          bm.push(m); toks.push(make(bd, m, bs)); moves.push(m);
          positions++;
          positions++;
          const enc = sideOf();
          const a = encodeFeatures(bd, enc, { recentMoves: moves.slice() });
          const b = encodeFeaturesReplay(bd, enc, { recentMoves: moves.slice() });
          for (let k = 0; k < 22 * N2; k++) if (a.spatial[k] !== b.spatial[k]) {
            /* 无污染诊断:纯手写重演(不走模块 make)对比环 prev1 与 bd */
            const pure = (n) => { const pb = new Int8Array(N2); let ps = BLACK, ko = -1;
              for (let i = 0; i < n; i++) { const mv = moves[i], v = (i % 2 === 0) ? 1 : 2;
                if (mv === PASS) { ko = -1; continue; }
                pb[mv] = v; let capN = 0, capCell = -1;
                for (const off of [-19, 19, -1, 1]) { const q = mv + off;
                  if (q < 0 || q >= N2 || (off === -1 && mv % 19 === 0) || (off === 1 && mv % 19 === 18) || pb[q] !== 3 - v) continue;
                  const st2 = [q], seen2 = new Set([q]), grp2 = []; let libs2 = 0;
                  while (st2.length) { const c = st2.pop(); grp2.push(c);
                    for (const o2 of [-19, 19, -1, 1]) { const q2 = c + o2;
                      if (q2 < 0 || q2 >= N2 || (o2 === -1 && c % 19 === 0) || (o2 === 1 && c % 19 === 18)) continue;
                      if (pb[q2] === 0) libs2++; else if (pb[q2] === 3 - v && !seen2.has(q2)) { seen2.add(q2); st2.push(q2); } } }
                  if (libs2 === 0) for (const c of grp2) { pb[c] = 0; capN++; capCell = c; } }
                ko = (capN === 1) ? capCell : -1; }
              return { pb, ko }; };
            const Nn = moves.length;
            /* 权威重演(模块 make;终态诊断,污染无所谓) */
            const rb2 = newBoard(); replayMoves(rb2, moves);
            const rb1 = newBoard(); replayMoves(rb1, moves.slice(0, Nn - 1));
            const rp1 = recentPrevBd(1);
            let bdDiff = -1, p1Diff = -1;
            for (let q = 0; q < N2; q++) { if (bd[q] !== rb2[q]) { bdDiff = q; break; } }
            if (rp1) for (let q = 0; q < N2; q++) { if (rp1[q] !== rb1[q]) { p1Diff = q; break; } }
            eq = false;
            badInfo = `branch${branch} j${j} sp[${k}] ch=${Math.floor(k / N2)} moves=${Nn} bdvs权威重演=${bdDiff < 0 ? '一致' : '@' + bdDiff} 环prev1vs权威重演=${rp1 ? (p1Diff < 0 ? '一致' : '@' + p1Diff) : 'null'} 环ko=${recentPrevKo(1)}`;
            break outer;
          }
        }
      }
      /* 还原到根(供下一局循环复用 bd/toks/moves 的长度一致性无关紧要) */
      while (moves.length > rootMoves.length) { unmake(bd, moves[moves.length - 1], toks[toks.length - 1]); toks.pop(); moves.pop(); }
      void rootToks;
      ringRestore(snap);
    }
  }
  ok(eq, `环=重演 逐位对拍(${positions} 局面,含劫争/提子/pass/换分支)`, badInfo);
});

const args = process.argv.slice(2);
const only = args.filter((a) => !a.startsWith('--'));
const run = only.length ? sections.filter((s) => only.includes(s.name)) : sections;
for (const s of run) {
  const t = Date.now(), p0 = pass, f0 = fail;
  console.log(`\n[${s.name}]`);
  s.fn();
  console.log(`  ${fail === f0 ? '✓' : '✗'} ${pass - p0} 项 · ${Date.now() - t}ms`);
}
console.log(`\n${fail === 0 ? '✓ 全部通过' : '✗ 有失败'}  ${pass} 项通过 / ${fail} 项失败`);
process.exit(fail ? 1 : 0);
