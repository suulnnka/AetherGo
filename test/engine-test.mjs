/* 9×9 围棋引擎测试:规则用例(提子 / 自杀 / 劫)+ 数子 + 记谱 + 搜索行为
 * + 随机对局模糊测试(make/unmake 往返不变量)。
 *
 * 运行:node test/engine-test.mjs          全部
 *      node test/engine-test.mjs ko       只跑指定节(--list 看全部)
 *
 * 摆盘注意:'X' = 黑,'O' = 白,'.' = 空;坐标 (r, c) = 行×9 + 列,行 0 在上。
 */
import {
  BLACK, WHITE, PASS, KOMI,
  newBoard, replayMoves, genLegal, isLegal, syncPosition,
  make, unmake, capturedOf, koPoint,
  evaluate, scoreGame, moveToText, searchBest, LEVELS, DEFAULT_LEVEL,
  boardToArray, arrayToBoard,
} from '../src/engine.js';

/* ---------- 断言框架 ---------- */
const sections = [];
const section = (name, fn) => sections.push({ name, fn });
let pass = 0, fail = 0;
const ok = (cond, msg, extra) => {
  if (cond) pass++;
  else { fail++; console.log(`  ✗ ${msg}${extra ? '  — ' + extra : ''}`); }
};
const eq = (got, want, msg) => ok(got === want, msg, `得到 ${got},期望 ${want}`);

/* ---------- 摆盘助手 ---------- */
function boardFrom(rows) {
  const bd = new Int8Array(81);
  rows.forEach((row, r) => {
    for (let c = 0; c < 9; c++) {
      const ch = row[c];
      if (ch === 'X') bd[r * 9 + c] = 1;
      else if (ch === 'O') bd[r * 9 + c] = 2;
    }
  });
  syncPosition(bd);
  return bd;
}
const sq = (r, c) => r * 9 + c;
const stoneCount = (bd) => { let n = 0; for (let i = 0; i < 81; i++) if (bd[i]) n++; return n; };

/* ==================== 1. 初始局面 ==================== */
section('init', () => {
  const bd = newBoard();
  eq(stoneCount(bd), 0, '初始局面全空');
  eq(genLegal(bd, BLACK).length, 81, '空盘 81 个落点(停一手不进列表)');
  ok(Math.abs(evaluate(bd) + KOMI) < 1e-9, '空盘评估 = -贴目(黑方视角)');
  eq(koPoint(), -1, '初始无劫');
  const lv = LEVELS[DEFAULT_LEVEL];
  ok(LEVELS.length === 4 && lv && lv.playouts > 0, '难度档 4 级且默认档有效');
});

/* ==================== 2. 提子 ==================== */
section('capture', () => {
  /* 单子:白 (4,4) 只剩一口气 (4,5),黑下 (4,5) 提掉 */
  {
    const bd = boardFrom(['.........', '.........', '.........',
                          '....X....', '...XO....', '....X....',
                          '.........', '.........', '.........']);
    eq(stoneCount(bd), 4, '摆盘 4 子');
    const tok = make(bd, sq(4, 5), BLACK);
    eq(capturedOf(tok), 1, '单子被提:提子数 1');
    eq(bd[sq(4, 4)], 0, '被提点变空');
    eq(koPoint(), -1, '提子后新子 4 口气:不形成劫');
    unmake(bd, sq(4, 5), tok);
    eq(bd[sq(4, 4)], 2, '撤销后白子还原');
    eq(bd[sq(4, 5)], 0, '撤销后落点变空');
    eq(stoneCount(bd), 4, '撤销后回到 4 子');
  }
  /* 整块:两颗白子共享最后一口气 (4,6) */
  {
    const bd = boardFrom(['.........', '.........', '.........',
                          '....XX...', '...XOO...', '....XX...',
                          '.........', '.........', '.........']);
    const tok = make(bd, sq(4, 6), BLACK);
    eq(capturedOf(tok), 2, '整块被提:提子数 2');
    eq(bd[sq(4, 4)] + bd[sq(4, 5)], 0, '两子都没了');
    eq(koPoint(), -1, '提两子不成劫');
    unmake(bd, sq(4, 6), tok);
    eq(bd[sq(4, 4)] + bd[sq(4, 5)], 4, '撤销后两子还原');
  }
  /* 快照与恢复:走 3 步再逐手撤销,棋盘完全复原 */
  {
    const bd = newBoard();
    const seq = [[sq(2, 2), BLACK], [sq(6, 6), WHITE], [sq(2, 6), BLACK]];
    const before = boardToArray(bd);
    const toks = seq.map(([mv, s]) => { const t = make(bd, mv, s); return [mv, s, t]; });
    eq(stoneCount(bd), 3, '走 3 步后 3 子');
    for (let i = toks.length - 1; i >= 0; i--) {
      const [mv, s, t] = toks[i];
      unmake(bd, mv, t);
    }
    eq(JSON.stringify(boardToArray(bd)), JSON.stringify(before), '逐手撤销后回到空盘');
  }
});

/* ==================== 3. 自杀 ==================== */
section('suicide', () => {
  /* 单点自杀:黑包围 (4,4),白下进去提不到子 */
  {
    const bd = boardFrom(['.........', '.........', '.........',
                          '....X....', '...X.X...', '....X....',
                          '.........', '.........', '.........']);
    ok(!isLegal(bd, WHITE, sq(4, 4)), '白下 (4,4) 是自杀:非法');
    ok(!genLegal(bd, WHITE).includes(sq(4, 4)), 'genLegal 不含自杀点');
    ok(isLegal(bd, BLACK, sq(4, 4)), '黑自己下 (4,4) 连回大块:合法');
  }
  /* 多子自杀:先下 (4,4) 合法(还有一口气 (4,5)),再下 (4,5) 整块没气 */
  {
    const bd = boardFrom(['.........', '.........', '.........',
                          '...XXX...', '...X..X..', '...XXX...',
                          '.........', '.........', '.........']);
    ok(isLegal(bd, WHITE, sq(4, 4)), '白下 (4,4) 合法(还有 (4,5) 一口气)');
    make(bd, sq(4, 4), WHITE);
    ok(!isLegal(bd, WHITE, sq(4, 5)), '再下 (4,5) 整块无气且提不到子:自杀');
    ok(isLegal(bd, BLACK, sq(4, 5)), '黑下 (4,5) 是提子:合法');
    const tok = make(bd, sq(4, 5), BLACK);
    eq(capturedOf(tok), 1, '黑提掉白 (4,4)');
  }
});

/* ==================== 4. 劫(simple ko) ==================== */
section('ko', () => {
  /* 经典劫形:白提黑 (1,2) → 打劫;黑不得立即回提 */
  const KO_SHAPE = ['.XO......',
                    'X.XO.....',
                    '.XO......',
                    '.........', '.........', '.........', '.........', '.........', '.........'];
  const bd = boardFrom(KO_SHAPE);
  const tok = make(bd, sq(1, 1), WHITE);        // 白提黑 (1,2)
  eq(capturedOf(tok), 1, '白提一子');
  eq(koPoint(), sq(1, 2), '劫点 = (1,2)');
  eq(bd[sq(1, 1)], 2, '白子落在 (1,1)');
  ok(!isLegal(bd, BLACK, sq(1, 2)), '黑不得立即回提劫点');
  ok(!genLegal(bd, BLACK).includes(sq(1, 2)), 'genLegal 也不含劫点');
  /* 黑寻劫(别处一手)、白应一手,劫点解禁 */
  make(bd, sq(8, 8), BLACK);
  eq(koPoint(), -1, '别处落子后劫点解除');
  make(bd, sq(7, 7), WHITE);
  ok(isLegal(bd, BLACK, sq(1, 2)), '隔一手后可以回提');
  const tok2 = make(bd, sq(1, 2), BLACK);       // 黑回提白 (1,1)
  eq(capturedOf(tok2), 1, '黑回提一子');
  eq(koPoint(), sq(1, 1), '劫点转移 = (1,1)');
  ok(!isLegal(bd, WHITE, sq(1, 1)), '同样不得立即回提');
  /* 劫 + 停:回提被禁时黑停一手,白随便应一手,劫点解禁 */
  {
    const bd2 = boardFrom(KO_SHAPE);
    make(bd2, sq(1, 1), WHITE);
    make(bd2, PASS, BLACK);                     // 黑停一手
    make(bd2, sq(8, 8), WHITE);
    ok(isLegal(bd2, BLACK, sq(1, 2)), '停一手后再下别处,劫点解禁');
  }
});

/* ==================== 5. 数子 ==================== */
section('score', () => {
  eq(evaluate(newBoard()), -KOMI, '空盘 = -贴目');
  {
    /* 黑角地:(0,0) 一块空 + 两颗黑子;白单颗天元 */
    const bd = boardFrom(['.X.......',
                          'X........',
                          '.........', '.........', '.........',
                          '.........', '.........', '.........',
                          '....O....']);
    /* 黑 2 子 + 1 空 = 3;白 1 子;3 − 1 − 5.5 = -3.5 */
    ok(Math.abs(evaluate(bd) + 3.5) < 1e-9, '角地数子正确', String(evaluate(bd)));
  }
  {
    /* 双方各围一块:黑围左上(眼 (1,1) + 角 (0,0)),白对称围右下 */
    /* 黑地 = (1,1) + (0,0);白地 = (7,7) + (8,8);其余空点全是公气 */
    const bd = boardFrom(['.X.......',
                          'X.X......',
                          '.X.......',
                          '.........', '.........', '.........',
                          '.......O.',
                          '......O.O',
                          '.......O.']);
    /* 黑 4 子 + 2 空 = 6;白 4 子 + 2 空 + 5.5 = 11.5;差 -5.5 */
    const s = scoreGame(bd);
    eq(s.black, 6, '黑 4 子 + 2 空');
    eq(s.white, 6 + KOMI, '白 4 子 + 2 空 + 贴目');
    ok(Math.abs(s.margin + KOMI) < 1e-9, '对称局面:白胜贴目', String(s.margin));
  }
  {
    /* 公气:大空盘中央对峙,中间空点双方都贴边 = 公气,不算任何一方 */
    const bd = boardFrom(['.X.......',
                          'X........',
                          '.........', '.........', '.........',
                          '.........', '.........', '.........',
                          '........O']);
    const s = scoreGame(bd);
    eq(s.black, 2 + 1, '黑只算自己围住的 1 点');
  }
});

/* ==================== 6. 记谱与重演 ==================== */
section('notation', () => {
  const bd = newBoard();
  eq(moveToText(bd, sq(2, 0)), 'A7', '左上角 = A7');
  eq(moveToText(bd, sq(0, 8)), 'J9', '右上角 = J9(列跳过 I)');
  eq(moveToText(bd, sq(8, 0)), 'A1', '左下角 = A1');
  eq(moveToText(bd, sq(4, 4)), 'E5', '天元 = E5');
  eq(moveToText(bd, PASS), '停一手', '停一手');

  /* 重演:带提子的序列在 Worker 侧重演出同样的盘面(黑白必须交替) */
  const SEQ = [sq(3, 4), sq(4, 4), sq(5, 4), sq(8, 8), sq(4, 3), sq(7, 7), sq(4, 5)];
  /* 手 7:黑下 (4,5),白 (4,4) 只剩一口气被提 */
  const direct = newBoard();
  SEQ.forEach((mv, i) => make(direct, mv, i % 2));
  eq(direct[sq(4, 4)], 0, '直接对弈:白 (4,4) 被提');
  const replayed = newBoard();
  const end = replayMoves(replayed, SEQ);
  eq(end, WHITE, '重演 7 手(黑先)后轮白');
  eq(JSON.stringify(boardToArray(replayed)), JSON.stringify(boardToArray(direct)), '重演盘面一致');

  /* 非法序列:序列里混入自杀手必须被拒(白往被围死的 (4,4) 里下) */
  const SUICIDE_SEQ = [sq(3, 4), sq(8, 8), sq(4, 3), sq(7, 7), sq(5, 4), sq(6, 6), sq(4, 5), sq(4, 4)];
  eq(replayMoves(newBoard(), SUICIDE_SEQ), -1, '白下被围死的 (4,4)(自杀)→ 整个序列被拒');
  eq(replayMoves(newBoard(), []), BLACK, '空序列从黑开始');
});

/* ==================== 7. 搜索行为 ==================== */
const ATARI = ['.........', '.........', '.........',
               '....X....', '...XO....', '....X....',
               '.........', '.........', '.........'];
section('search', () => {
  /* 白送吃的子:轻演棋下「立即提」与「演棋里反正会被提」价值接近,
   * 不强求树端立刻提 —— 只断言着法合法、胜率不掉出合理区间 */
  {
    const bd = boardFrom(ATARI);
    const r = searchBest(bd, BLACK, { playouts: 3000, ms: 10000 });
    ok(isLegal(bd, BLACK, r.move), '返回的着法合法', `move=${r.move}`);
    ok(r.move !== PASS, '优势时不会停一手');
    ok(r.winRate > 0.3 && r.winRate < 0.95, '胜率在合理区间', String(r.winRate));
    ok(r.visits === 3000, '演棋次数用满');
  }
  /* 开局:不下第一线、不停一手(2000 演棋下偶有一线手,9000 才稳) */
  {
    const bd = newBoard();
    const r = searchBest(bd, BLACK, { playouts: 9000, ms: 20000 });
    ok(r.move !== PASS, '开局不停一手');
    const rr = (r.move / 9) | 0, cc = r.move % 9;
    ok(rr >= 1 && rr <= 7 && cc >= 1 && cc <= 7, '开局避开第一线', `(${rr},${cc})`);
    ok(isLegal(bd, BLACK, r.move), '返回的着法合法');
  }
  /* 逐次回报:visits 递增、winRate 在 [0,1] */
  {
    const bd = newBoard();
    const seen = [];
    searchBest(bd, BLACK, {
      playouts: 1000, ms: 10000,
      onProgress: (p) => seen.push(p),
    });
    ok(seen.length >= 2, '有逐次回报', `${seen.length} 次`);
    ok(seen.every((p, i) => i === 0 || p.visits > seen[i - 1].visits), 'visits 递增');
    ok(seen.every((p) => p.winRate >= 0 && p.winRate <= 1), 'winRate 在 [0,1]');
  }
  /* 白方视角:自己的子在叫吃上,白会应(长气/提子/弃子转换),不会停一手干等 */
  {
    const bd = boardFrom(ATARI);
    const r = searchBest(bd, WHITE, { playouts: 3000, ms: 10000 });
    ok(isLegal(bd, WHITE, r.move), '白方返回合法着法');
    ok(r.move !== PASS, '白方不会坐视被提');
  }
  /* 自对弈到双停:引擎对引擎(低预算)必须能在有限手数内正常终局并数出结果 */
  {
    const bd = newBoard();
    let side = BLACK, lastPass = false, plies = 0, over = false;
    for (let t = 0; t < 400 && !over; t++) {
      const r = searchBest(bd, side, { playouts: 250, ms: 5000 });
      ok(isLegal(bd, side, r.move), `第 ${t} 手着法合法`);
      if (r.move === PASS && lastPass) { over = true; break; }
      lastPass = r.move === PASS;
      make(bd, r.move, side);
      plies++;
      side ^= 1;
    }
    ok(over, '自对弈在 400 手内双停终局', `${plies} 手未终局`);
    const s = scoreGame(bd);
    ok(Number.isFinite(s.margin) && s.black > 0 && s.white > 0, '终局数子结果合理',
      JSON.stringify(s));
  }
});

/* ==================== 8. 随机对局模糊测试 ==================== */
section('fuzz', () => {
  let seed = 20260919;
  const rnd = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  let games = 0, plies = 0, ends = 0, bad = 0;
  for (let g = 0; g < 40; g++) {
    const bd = newBoard();
    let side = BLACK, lastPass = false;
    const trail = [];                            // { mv, side, tok, legal } —— legal 是**走前**的合法列表
    for (let t = 0; t < 220; t++) {
      const legalBefore = genLegal(bd, side);
      const stonesBefore = stoneCount(bd);
      let mv;
      if (legalBefore.length === 0) mv = PASS;
      else if (rnd() < 0.06) mv = PASS;
      else mv = legalBefore[(rnd() * legalBefore.length) | 0];
      if (mv !== PASS && !isLegal(bd, side, mv)) bad++;
      const tok = make(bd, mv, side);
      if (tok === -2) { bad++; break; }          // make 失败:genLegal 给出了非法手
      if (mv !== PASS) {
        /* 盘面子数守恒:走前 + 1 − 提子 = 走后 */
        const capN = capturedOf(tok);
        if (stoneCount(bd) !== stonesBefore + 1 - capN) bad++;
      }
      trail.push({ mv, side, tok, legal: legalBefore });
      plies++;
      const pass = mv === PASS;
      if (pass && lastPass) { ends++; break; }   // 连续两手停 = 终局
      lastPass = pass;
      side ^= 1;
    }
    /* 逐手撤销到空盘,再逐手重演:每一步走前的合法列表必须逐步复原
     * (强不变量 —— 劫点、提子、气全都覆盖) */
    const finalBoard = boardToArray(bd);
    for (let i = trail.length - 1; i >= 0; i--) unmake(bd, trail[i].mv, trail[i].tok);
    eq(JSON.stringify(boardToArray(bd)), JSON.stringify(new Array(81).fill(0)), `撤销到空盘(局 ${g})`);
    for (let i = 0; i < trail.length; i++) {
      const { mv, side, tok, legal } = trail[i];
      const legalNow = genLegal(bd, side);
      if (JSON.stringify(legalNow) !== JSON.stringify(legal)) { bad++; break; }
      make(bd, mv, side);
    }
    if (JSON.stringify(boardToArray(bd)) !== JSON.stringify(finalBoard)) bad++;
    games++;
  }
  eq(bad, 0, '随机对局中不出现非法局面 / 撤销不一致');
  console.log(`    ${games} 局 / ${plies} 手 / 其中 ${ends} 局双停终局`);
});

/* ---------- 执行 ---------- */
const args = process.argv.slice(2);
if (args.includes('--list')) {
  console.log('可用小节:' + sections.map((s) => ' ' + s.name).join(','));
  process.exit(0);
}
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
