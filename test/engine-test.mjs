/* 19×19 围棋引擎测试:规则用例(提子 / 自杀 / 劫)+ 数子 + 记谱
 * + 随机对局模糊测试(make/unmake 往返不变量)。
 * 搜索行为(PUCT / KataGo 对齐项)在 test/katago-align-test.mjs 与
 * test/nn-temp-test.mjs;引擎本体已不含任何搜索。
 *
 * 运行:node test/engine-test.mjs          全部
 *      node test/engine-test.mjs ko       只跑指定节(--list 看全部)
 *
 * 摆盘注意:'X' = 黑,'O' = 白,'.' = 空;坐标 (r, c) = 行×19 + 列,行 0 在上。
 * 历史 9 路用例整体镶在 19 路盘**左上角**(补 '.' 即可)—— 死活 / 劫 /
 * 数子都是局部语义,坐标与断言不变;只有「贴边」语义的用例
 * (对称围空 / 大范围公气 / 记谱)按 19 路重写。
 */
import {
  BLACK, WHITE, PASS, KOMI, N, N2,
  newBoard, replayMoves, genLegal, isLegal, syncPosition,
  make, unmake, capturedOf, koPoint, positionKey,
  evaluate, scoreGame, moveToText,
  boardToArray, arrayToBoard, deadStones, finalScore, deadStonesWithOwnership, scoreBreakdown,
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

/* ---------- 摆盘助手:行/列不足 19 的用例自动右/下补 '.' ---------- */
function boardFrom(rows) {
  const bd = new Int8Array(N2);
  rows.forEach((row, r) => {
    for (let c = 0; c < row.length && c < N; c++) {
      const ch = row[c];
      if (ch === 'X') bd[r * N + c] = 1;
      else if (ch === 'O') bd[r * N + c] = 2;
    }
  });
  syncPosition(bd);
  return bd;
}
const sq = (r, c) => r * N + c;
const stoneCount = (bd) => { let n = 0; for (let i = 0; i < N2; i++) if (bd[i]) n++; return n; };
const EMPTY_ROW = '.'.repeat(N);

/* ==================== 1. 初始局面 ==================== */
section('init', () => {
  const bd = newBoard();
  eq(stoneCount(bd), 0, '初始局面全空');
  eq(genLegal(bd, BLACK).length, N2, `空盘 ${N2} 个落点(停一手不进列表)`);
  ok(Math.abs(evaluate(bd) + KOMI) < 1e-9, '空盘评估 = -贴目(黑方视角)');
  eq(koPoint(), -1, '初始无劫');
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

/* ==================== 4b. 禁全同(position superko) ==================== */
section('superko', () => {
  /* 送二还一:黑提白两子(非单劫,KO 不设),白立即回提一子 —— 净死子数变了,
   * 局面是全新的,禁全同**不得**过度禁 */
  {
    const bd = boardFrom(['.OXX.....',
                          'O.OOX....',
                          '.OXX.....',
                          '.........', '.........', '.........', '.........', '.........', '.........']);
    eq(capturedOf(make(bd, sq(1, 1), BLACK)), 2, '黑提白两子');
    eq(koPoint(), -1, '提两子不是单劫形');
    ok(isLegal(bd, WHITE, sq(1, 2)), '送二还一:白回提合法(局面是新的)');
    eq(capturedOf(make(bd, sq(1, 2), WHITE)), 1, '白回提一子');
  }
  /* 转置路径:不同顺序走到**同一盘面**,两路全程合法、终局键相同 ——
   * 禁全同只认历史里出现过的键,不会过度禁 */
  {
    const a = newBoard(), b = newBoard();
    make(a, sq(2, 2), BLACK); make(a, sq(6, 6), WHITE); make(a, sq(4, 4), BLACK);
    make(b, sq(4, 4), BLACK); make(b, sq(6, 6), WHITE); make(b, sq(2, 2), BLACK);
    eq(JSON.stringify(boardToArray(a)), JSON.stringify(boardToArray(b)), '不同顺序走到同一盘面');
    eq(positionKey(a), positionKey(b), '同盘面 → 同键(键只认盘上棋子)');
    const c = newBoard();
    make(c, sq(2, 2), BLACK); make(c, sq(6, 6), WHITE); make(c, sq(4, 4), BLACK);
    eq(positionKey(c), positionKey(a), '同序列重演 → 同键');
  }
  /* 增量键 == 全量重算:带提子的随机对局走到终盘,positionKey 与
   * syncPosition 重算一致;unmake 后键也逐手还原 */
  {
    const bd = newBoard();
    let seed = 20261001;
    const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const toks = [];
    let side = BLACK, lastPass = false;
    for (let t = 0; t < 120; t++) {
      const legal = genLegal(bd, side);
      const mv = (legal.length === 0 || rnd() < 0.05) ? PASS : legal[(rnd() * legal.length) | 0];
      const kBefore = positionKey();
      toks.push([mv, side, make(bd, mv, side), kBefore]);
      if (mv === PASS && lastPass) break;
      lastPass = mv === PASS;
      side ^= 1;
    }
    const fin = boardToArray(bd);
    const kFin = positionKey();
    /* 逐手撤销:键随栈还原 */
    for (let i = toks.length - 1; i >= 0; i--) {
      const [mv, , tok] = toks[i];
      unmake(bd, mv, tok);
    }
    eq(positionKey(), '0,0', '撤销到空盘 → 空盘键');
    /* 增量键(一路 make 出来)== 全量重算(syncPosition 扫盘) */
    arrayToBoard(fin);
    eq(positionKey(), kFin, '带提子随机对局:增量键 == 全量重算');
  }
  /* 三劫循环:三个互不干扰的单劫。单劫口径下每一手回提时禁点都在别的劫,
   * 可以无限循环;禁全同下第 6 手复原初始局面 —— 必须被禁 */
  {
    const TRIPLE_KO = ['.OX...OXX',
                       'O.OX.OX.X',
                       '.OX...OX.',
                       '.........', '.........', '.........',
                       '.OX......',
                       'O.OX.....',
                       '.OX......'];
    const bd = boardFrom(TRIPLE_KO);
    const k0 = positionKey();
    /* 劫1(左上,黑先提)劫2(右上,白先提)劫3(左下,黑先提),三手各提一子 */
    eq(capturedOf(make(bd, sq(1, 1), BLACK)), 1, '劫1:黑提白');
    eq(capturedOf(make(bd, sq(1, 7), WHITE)), 1, '劫2:白提黑');
    eq(capturedOf(make(bd, sq(7, 1), BLACK)), 1, '劫3:黑提白');
    /* 回提轮转:每次回提时 KO 都指向别的劫,单劫不禁 */
    eq(capturedOf(make(bd, sq(1, 2), WHITE)), 1, '白回提劫1(单劫口径合法)');
    eq(capturedOf(make(bd, sq(1, 6), BLACK)), 1, '黑回提劫2(单劫口径合法)');
    eq(koPoint(), sq(1, 7), '当前劫点在劫2 —— 不在劫3 的回提点上');
    /* 第 6 手:白回提劫3 → 复原初始盘面(键 = k0)→ 禁全同必须禁 */
    ok(!isLegal(bd, WHITE, sq(7, 2)), '三劫循环第 6 手复原局面:禁全同禁止');
    ok(!genLegal(bd, WHITE).includes(sq(7, 2)), 'genLegal 也不含该点');
    /* 白改下别处后,黑再回提劫3 复原的是「白刚下完」之前的局面吗?不是 ——
     * 劫3 回提合法化需要劫3 局面本身出新,这里只验证循环已被打破 */
    make(bd, sq(8, 8), WHITE);
    const legalNow = genLegal(bd, BLACK);
    ok(legalNow.every((p) => isLegal(bd, BLACK, p)), '打破循环后所有合法点自洽');
    ok(k0 !== positionKey(), '局面已离开循环起点');
  }
  /* 摆盘重置:syncPosition 后历史只含当前局面 —— 之前对局的键不再禁着 */
  {
    const bd = newBoard();
    make(bd, sq(4, 4), BLACK);
    make(bd, sq(4, 5), WHITE);
    const arr = boardToArray(bd);
    const bd2 = arrayToBoard(arr);          // 同盘面重摆
    eq(positionKey(bd2), positionKey(bd), '重摆同盘面 → 同键');
    ok(isLegal(bd2, BLACK, sq(3, 3)), '重摆后历史已重置,正常落子不受旧历史影响');
  }
});


section('score', () => {
  eq(evaluate(newBoard()), -KOMI, '空盘 = -贴目');
  {
    /* 数子明细:子 / 空 / 贴目逐项对账 */
    const bd = boardFrom(['.X.......',
                          'X........',
                          '.........', '.........', '.........',
                          '.........', '.........', '.........',
                          '....O....']);
    const d = scoreBreakdown(bd);
    eq(d.blackStones, 2, '黑子 2');
    eq(d.blackTerritory, 1, '黑空 1((0,0))');
    eq(d.whiteStones, 1, '白子 1');
    eq(d.whiteTerritory, 0, '白空 0(孤子)');
    eq(d.black, 3, '黑 = 2 子 + 1 空');
    eq(d.white, 1 + KOMI, '白 = 1 子 + 贴目');
    ok(Math.abs(d.margin - (3 - (1 + KOMI))) < 1e-9, '目差自洽');
    /* 带死子的明细:死子从对账里剔除且回传列表 */
    const d2 = scoreBreakdown(bd, KOMI, [sq(8, 4)]);
    eq(d2.whiteStones, 0, '白子被判死后不计');
    eq(JSON.stringify(d2.dead), JSON.stringify([sq(8, 4)]), 'dead 列表回传');
    /* finalScore 与 breakdown 同源同数 */
    const f = finalScore(bd, KOMI, [sq(8, 4)]);
    ok(f.black === d2.black && f.white === d2.white && f.margin === d2.margin
      && JSON.stringify(f.dead) === JSON.stringify(d2.dead), 'finalScore = scoreBreakdown 的总数投影');
  }
  {
    /* 黑角地:(0,0) 一块空 + 两颗黑子;白单颗近角 */
    const bd = boardFrom(['.X.......',
                          'X........',
                          '.........', '.........', '.........',
                          '.........', '.........', '.........',
                          '....O....']);
    /* 黑 2 子 + 1 空 = 3;白 1 子;3 − 1 − 7.5 = -5.5 */
    ok(Math.abs(evaluate(bd) + 5.5) < 1e-9, '角地数子正确', String(evaluate(bd)));
  }
  {
    /* 双方各围一块:黑围左上(眼 (1,1) + 角 (0,0)),白镜像围右下(19 路重写) */
    const bd = boardFrom([
      '.X' + '.'.repeat(N - 2),
      'X.X' + '.'.repeat(N - 3),
      '.X' + '.'.repeat(N - 2),
      ...Array(N - 6).fill(EMPTY_ROW),
      '.'.repeat(N - 2) + 'O' + '.',
      '.'.repeat(N - 3) + 'O.O',
      '.'.repeat(N - 2) + 'O' + '.',
    ]);
    /* 黑 4 子 + 2 空 = 6;白 4 子 + 2 空 + 贴目;差 -贴目 */
    const s = scoreGame(bd);
    eq(s.black, 6, '黑 4 子 + 2 空');
    eq(s.white, 6 + KOMI, '白 4 子 + 2 空 + 贴目');
    ok(Math.abs(s.margin + KOMI) < 1e-9, '镜像对称局面:白胜贴目', String(s.margin));
  }
  {
    /* 公气:大空盘中央对峙,中间空点双方都贴边 = 公气,不算任何一方 */
    const bd = boardFrom(['.X',
                          'X',
                          ...Array(N - 3).fill(''),
                          '.'.repeat(N - 1) + 'O']);
    const s = scoreGame(bd);
    eq(s.black, 2 + 1, '黑只算自己围住的 1 点');
    eq(s.white, 1 + KOMI, '白孤子不围空(盘面大空全为公气,只剩贴目)');
  }
});

/* ==================== 4c. 双停死子处理 ==================== */
section('deadstones', () => {
  /* 验收局:白角三子死棋(黑墙密封),右下白棋一只真眼活 —— 检出死块、
   * 且估计结果与「实战提子后数子」一致 */
  {
    const bd = boardFrom(['OO.X......',
                          'O.XX......',
                          '.XXX......',
                          'XX........',
                          '.........', '.........',
                          '......OOO.',
                          '......O.O.',
                          '......OOO.']);
    const t0 = Date.now();
    const dead = deadStones(bd);
    ok(Date.now() - t0 < 2000, '死子判定在小预算内完成');
    eq(JSON.stringify([...dead].sort((a, b) => a - b)), JSON.stringify([sq(0, 0), sq(0, 1), sq(1, 0)]),
      '死子 = 白角三子(黑墙与白活块不误判)');
    /* 估计 vs 实战:把死子从盘上真移除后再数子,两者必须一致 */
    const after = boardFrom(['...X......',
                             '..XX......',
                             '.XXX......',
                             'XX........',
                             '.........', '.........',
                             '......OOO.',
                             '......O.O.',
                             '......OOO.']);
    const est = finalScore(bd);
    const real = scoreGame(after);
    eq(est.black, real.black, `估计黑目 ${est.black} == 提子后 ${real.black}`);
    eq(est.white, real.white, `估计白目 ${est.white} == 提子后 ${real.white}`);
    ok(Math.abs(est.margin - real.margin) < 1e-9, '目差一致');
    ok(!est.dead.includes(sq(6, 6)) && !est.dead.includes(sq(0, 3)), '活块/黑墙不在死子列表');
    /* 移除后的角部 6 个空点必须全部归黑 —— 估计里已体现 */
    ok(real.black >= 8 + 6, '提子后黑角空点归黑(8 子 + 6 空)', String(real.black));
  }
  /* 黑方死子对称:白贴着黑一子,黑仅剩 (1,0) 一口气,白先手提 —— 检出 */
  {
    const bd2 = boardFrom(['XO']);
    const dead = deadStones(bd2);
    eq(JSON.stringify(dead), JSON.stringify([sq(0, 0)]), '白先手提黑一子:黑子判死');
  }
  /* 手改死子重算:deadOverride 原样生效 */
  {
    const bd = newBoard();
    make(bd, sq(4, 4), BLACK); make(bd, sq(4, 5), WHITE);
    const s = finalScore(bd, KOMI, [sq(4, 4)]);
    ok(s.dead.length === 1 && s.dead[0] === sq(4, 4), 'deadOverride 透传');
    eq(s.black, 0, '黑子被当死子移除后不计子');
  }
  /* 活棋不误判:黑两眼活棋 + 白两眼活棋,死子列表为空 */
  {
    const bd = boardFrom(['.XX...OO.',
                          'X.X...O.O',
                          '.XX...OO.',
                          '.........', '.........', '.........', '.........', '.........', '.........']);
    const dead = deadStones(bd);
    eq(dead.length, 0, `双方两眼活棋:无死子(得 ${JSON.stringify(dead)})`);
  }
  /* ownership 辅助标注(deadStonesWithOwnership):规则侧漏判补标、Benson 保底、
   * 规则结论不翻案、模糊地带不动作 */
  {
    /* 单颗白子开阔地 2 气:规则侧判活(延伸可逃),NN ownership 强烈相悖 → 补标死 */
    const bd = newBoard();
    make(bd, sq(9, 9), WHITE);
    eq(deadStones(bd).length, 0, '规则侧:开阔地 2 气单子判活(宁漏勿错)');
    const dOwn = deadStonesWithOwnership(bd, new Float32Array(N2).fill(1));   // 全盘黑归属
    eq(JSON.stringify(dOwn), JSON.stringify([sq(9, 9)]), 'ownership 相悖(白子均值 +1)→ 补标死');
    eq(deadStonesWithOwnership(bd, new Float32Array(N2)).length, 0, 'ownership 全 0(模糊)→ 不动作');
  }
  {
    /* Benson 两眼活棋保底:即使 ownership 与链色相悖也不判死 */
    const bd = boardFrom(['.XX',
                          'X.X',
                          '.XX']);
    const d = deadStonesWithOwnership(bd, new Float32Array(N2).fill(1));      // 黑归属:白棋相悖
    eq(d.length, 0, 'Benson 活棋:ownership 相悖仍不死');
  }
  {
    /* 规则侧已判死的链不因 ownership 翻案;全盘白归属下黑墙(NN 认死)被补标 ——
     * 两个方向各自生效 */
    const bd = boardFrom(['OO.X',
                          'O.XX',
                          '.XXX',
                          'XX']);
    const own = new Float32Array(N2).fill(-1);                                // 全盘白归属
    const d = deadStonesWithOwnership(bd, own);
    ok(d.includes(sq(0, 0)) && d.includes(sq(0, 1)) && d.includes(sq(1, 0)), '规则判死:不被翻案');
    ok([sq(0, 3), sq(1, 2), sq(1, 3), sq(2, 1), sq(2, 2), sq(2, 3), sq(3, 0), sq(3, 1)].every((p) => d.includes(p)),
      '黑墙在白归属下(NN 认死)被补标');
  }
  {
    /* ownership 为空:退化为纯规则(与 deadStones 一致) */
    const bd = boardFrom(['OO.X',
                          'O.XX',
                          '.XXX',
                          'XX']);
    const a = deadStones(bd), b = deadStonesWithOwnership(bd, null);
    eq(JSON.stringify([...a].sort((x, y) => x - y)), JSON.stringify([...b].sort((x, y) => x - y)),
      'ownership 为空 = 纯规则');
  }
});

/* ==================== 6. 记谱与重演 ==================== */
section('notation', () => {
  const bd = newBoard();
  eq(moveToText(bd, sq(2, 0)), 'A17', '左上 (2,0) = A17');
  eq(moveToText(bd, sq(0, 8)), 'J19', '(0,8) = J19(列跳过 I)');
  eq(moveToText(bd, sq(18, 0)), 'A1', '左下角 = A1');
  eq(moveToText(bd, sq(9, 8)), 'J10', '(9,8) = J10');
  eq(moveToText(bd, sq(9, 9)), 'K10', '天元 = K10');
  eq(moveToText(bd, sq(0, 18)), 'T19', '右上角 = T19');
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
    /* 禁全同直接不变量:非停一手产生的局面键,整局互不重复
     * (初始键先入集合;停一手键不变,不重复计入) */
    const seenKeys = new Set([positionKey()]);
    let side = BLACK, lastPass = false;
    const trail = [];                            // { mv, side, tok, legal } —— legal 是**走前**的合法列表
    for (let t = 0; t < 420; t++) {
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
        /* 落子后的局面键必须是全新的 —— 这就是禁全同本身 */
        const key = positionKey();
        if (seenKeys.has(key)) bad++;
        seenKeys.add(key);
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
    eq(JSON.stringify(boardToArray(bd)), JSON.stringify(new Array(N2).fill(0)), `撤销到空盘(局 ${g})`);
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
