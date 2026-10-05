/* KataGo 搜索机制六项对齐的专项验证(桩 session,不起 NN)。
 *
 * 覆盖:
 *   1. 无用着剪枝 —— isOwnTrueEye(真眼口径,假眼保留)
 *   2. FPU —— 未访问子 Q = parentQ − 0.25(行为验证:弱先验点不被白送访问)
 *   3. cpuct 调度 —— 随访问数对数增长(行为验证:强先验点主导访问)
 *   4. LCB 选点 —— 温度 0 时取置信下界最大者(nn-temp A 组亦覆盖)
 *   5. NN 评估缓存 —— 同特征叶子零推理复用(cacheHits / nnCalls 断言)
 *   6. fillDameBeforePass —— 还有单官时压制 pass,单官清零后放开
 *
 * 运行:node test/katago-align-test.mjs
 */
import { N, N2, BLACK, WHITE, PASS, newBoard, make, syncPosition } from '../src/engine.js';
import { nnSearchBest, isOwnTrueEye, countDame, clearEvalCache, __nodeTableSize, __clearBiasTable } from '../src/nn/search.js';
import { calibrateMaxBatch, pickBatchSizeFromThroughput } from '../src/nn/session.js';

let failed = 0;
const check = (name, cond, extra) => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  — ' + (extra ?? '')}`);
  if (!cond) failed++;
};
const sq = (r, c) => r * N + c;

function boardFrom(cells) {                  // cells: [[r, c, 'X'|'O'], ...]
  const bd = newBoard();
  for (const [r, c, ch] of cells) make(bd, sq(r, c), ch === 'X' ? BLACK : WHITE);
  return bd;
}

/* 桩:policy 偏好可配,winLoss 恒定。尾部着法 −20(真实网络尾部口径)。 */
function makeStub({ hot = [], passLogit = -20, winLoss = 0.3 } = {}) {
  return {
    calls: 0,
    async evalBatch(items) {
      this.calls += items.length;
      return items.map(() => {
        const policy = new Float32Array(N2).fill(-20);
        for (const [p, v] of hot) policy[p] = v;
        return { policy, policyPass: passLogit, winLoss };
      });
    },
  };
}
const search = (bd, opts) => nnSearchBest(bd, BLACK, {
  session: makeStub(opts), visits: 120, batch: 4, symmetry: false, reuseTree: false, allowResign: false, debug: true, ...opts,
});

/* ==================== 1. 无用着剪枝:真眼 / 假眼 ==================== */
{
  /* 黑棋角部四邻全黑 + 斜角全黑/边界 = 真眼,不该填 */
  const eyeBd = boardFrom([[0, 1, 'X'], [1, 0, 'X'], [1, 1, 'X']]);
  check('1a 角上真眼判定为真', isOwnTrueEye(eyeBd, BLACK, sq(0, 0)));
  /* 斜角有敌子 = 假眼,必须保留(假眼关系死活,不能一刀切禁填):
   * (1,1) 四邻 (0,1)(1,0)(1,2)(2,1) 全黑,斜角 (2,2) 白 → 假眼 */
  const falseBd = boardFrom([[0, 1, 'X'], [1, 0, 'X'], [1, 2, 'X'], [2, 1, 'X'], [2, 2, 'O']]);
  check('1b 斜角有敌子的中心点是假眼', !isOwnTrueEye(falseBd, BLACK, sq(1, 1)));
  /* 四邻不全是己方:根本不是眼 */
  check('1c 四邻有敌子不是眼', !isOwnTrueEye(eyeBd, BLACK, sq(0, 2)));
}

/* ==================== 6. 单官计数与 fillDameBeforePass ==================== */
{
  check('6a 空盘全是单官', countDame(newBoard()) === N2, String(countDame(newBoard())));
  /* 黑在天元:其余 360 点一个连通区域只贴黑 = 黑方领地口径,单官 0
   * (开局 pass 与否由 policy 决定,压制只看单官,这正是 fillDameBeforePass 的口径) */
  const tenji = boardFrom([[9, 9, 'X']]);
  check('6b 只贴一色的空域不算单官', countDame(tenji) === 0, String(countDame(tenji)));
  /* 黑白对峙:大空域同时贴两色 = 单官 */
  const duel = boardFrom([[0, 0, 'X'], [0, 2, 'O']]);
  check('6c 贴两色的空域是单官', countDame(duel) === N2 - 2, String(countDame(duel)));

  /* 行为:2026-10-05 对齐 KataGo 面积计分口径,pass 先验不再压制
   * (KataGo shouldSuppressPass 仅数目法生效)—— 强 pass 先验无论有无单官
   * 都直接当选,何时停一手交给网络值 + 双停终局值。 */
  const hot = [[sq(9, 9), 20], [sq(9, 10), 19]];
  const on = await nnSearchBest(duel, BLACK, {
    session: makeStub({ hot, passLogit: 25 }), visits: 60, batch: 4, symmetry: false,
    reuseTree: false, allowResign: false,
  });
  check('6d 单官在盘时强 pass 先验也当选(面积计分不压制)', on.move === PASS, `move=${on.move}`);
  const clean = await nnSearchBest(tenji, BLACK, {
    session: makeStub({ hot, passLogit: 25 }), visits: 60, batch: 4, symmetry: false,
    reuseTree: false, allowResign: false,
  });
  check('6f 无单官时强 pass 先验当选', clean.move === PASS, `move=${clean.move}`);
  const noPass = await nnSearchBest(duel, BLACK, {
    session: makeStub({ hot }), visits: 60, batch: 4, symmetry: false,
    reuseTree: false, allowResign: false,
  });
  check('6g 无 pass 先验时照常下棋盘点着法', noPass.move !== PASS, `move=${noPass.move}`);

  /* 6h 终选 pass 守门(shouldSuppressPass 移植):强 pass 先验 + 可下的盘上着法
   * (ownership 非对方铁地)→ 守门清零 pass 权重,选盘上着法;
   * 全盘皆对方铁地深处 → 守门不触发,pass 当选。 */
  const ownStub = (ownVal) => ({ async evalBatch(items) {
    return items.map(() => {
      const policy = new Float32Array(N2).fill(-20);
      for (const [p, v] of hot) policy[p] = v;
      const own = new Float32Array(N2).fill(ownVal);
      return { policy, policyPass: 21, winLoss: 0.0, ownership: own };
    });
  } });
  const guarded = await nnSearchBest(duel, BLACK, {
    session: ownStub(0.5), visits: 60, batch: 4, symmetry: false, reuseTree: false, allowResign: false,
  });
  check('6h 可下点存在时守门压掉强 pass 先验', guarded.move !== PASS, `move=${guarded.move}`);
  const freePass = await nnSearchBest(duel, BLACK, {
    session: ownStub(-2.5), visits: 60, batch: 4, symmetry: false, reuseTree: false, allowResign: false,  /* pretanh -2.5 → tanh≈-0.99 */
  });
  check('6h2 全盘对方铁地时守门放行 pass', freePass.move === PASS, `move=${freePass.move}`);
}

/* ==================== 5. NN 评估缓存 ==================== */
{
  const sess = makeStub({ hot: [[sq(9, 9), 5], [sq(9, 10), 4]] });
  const bd = newBoard();
  const r1 = await nnSearchBest(bd, BLACK, {
    session: sess, visits: 120, batch: 4, symmetry: false, reuseTree: false, allowResign: false,
  });
  /* 第二次搜同一局面:模块级缓存命中,推理调用显著变少(根叶子零推理)。
   * v4.4:subtreeValueBias 是跨手在线学习(KataGo 忠实语义),且 r2 走缓存
   * 命中同步路径、r1 走异步批路径,管线交织序不同 + bias 反馈会放大为
   * 近似等值热点对(5/4)内部的换位 —— 断言从「逐位相等」放宽为「热点集
   * 成员」(180/181 本就是 q 差 <0.01 的等值对,换位无决策意义)。 */
  __clearBiasTable();
  const r2 = await nnSearchBest(bd, BLACK, {
    session: sess, visits: 120, batch: 4, symmetry: false, reuseTree: false, allowResign: false,
  });
  check('5a 二次搜索命中评估缓存', r2.cacheHits > 0, `hits=${r2.cacheHits}`);
  check('5b 缓存命中减少推理调用', r2.nnCalls < r1.nnCalls, `${r1.nnCalls} → ${r2.nnCalls}`);
  const hotSet = new Set([sq(9, 9), sq(9, 10)]);
  check('5c 两次结果都在热点集(缓存不破坏决策)', hotSet.has(r1.move) && hotSet.has(r2.move), `${r1.move} vs ${r2.move}`);
}

/* ==================== 2/3/4. FPU / cpuct / LCB 的行为面 ==================== */
{
  /* 桩的 policy 位置无关,但 pass 终局按真实盘面数子 —— 子树价值不同是真实
   * 搜索信号,「最强先验必胜」不是有效断言;这里验证:单强点下根剪到只剩
   * [强点, pass]、LCB 选强点且完全可复现(引擎无随机源)。
   * 缓存键只认特征:组内换桩配置必须先清缓存,否则命中上一配置的桩输出。 */
  clearEvalCache();
  const r = await search(newBoard(), { hot: [[sq(9, 9), 6]] });
  const r2 = await search(newBoard(), { hot: [[sq(9, 9), 6]] });
  check('2/3/4a LCB 选强点且可复现', r.move === sq(9, 9) && r2.move === r.move, `${r.move}/${r2.move}`);
  check('2/3/4b 访问数用满', r.visits === 120, String(r.visits));
  check('2/3/4c 低先验剪枝:根只剩强点 + pass', r.rootChildren === 2, `children=${r.rootChildren}`);
  /* 对照:全 −20(softmax 后均匀,先验 1/361 高于地板)→ 照常全建子 */
  clearEvalCache();
  const r3 = await search(newBoard(), {});
  check('2/3/4d 均匀先验不误剪', r3.rootChildren > 300, `children=${r3.rootChildren}`);
}

/* ==================== 7. GTP 配方接线:乐观插值 λ / score 头兼容 ==================== */
{
  /* 根评估带 rootPolicyOptimism=0.2(GTP 配方),叶子评估 λ=1(树内缺省);
   * 带 score 系输出的桩走效用/不确定度路径不崩,且决策可复现。
   * 缓存键只认特征:本组换桩配置前先清缓存。 */
  clearEvalCache();
  const seen = [];
  const mkPolicy = () => { const p = new Float32Array(N2).fill(-20); p[sq(9, 9)] = 6; return p; };
  const session = {
    calls: 0,
    async evalBatch(items) {
      this.calls += items.length;
      seen.push(items.map((r) => r.optimism));
      return items.map(() => ({
        policy: mkPolicy(), policyPass: -20, winLoss: 0.3,
        scoreMean: 2.5, scoreStdev: 12, scoreLead: 2.5,
        shorttermScoreError: 4, shorttermWinlossError: 0.1,
      }));
    },
  };
  const r = await nnSearchBest(newBoard(), BLACK, {
    session, visits: 40, batch: 4, symmetry: false, reuseTree: false, debug: true,
  });
  check('7a 根评估带 λ=0.2(rootPolicyOptimism)', seen[0] && seen[0][0] === 0.2, JSON.stringify(seen[0]));
  check('7b 叶子评估 λ 缺省 1(树内)', seen.slice(1).every((a) => a.every((v) => v === undefined || v === 1)),
    JSON.stringify(seen.slice(1)));
  check('7c score 头路径返回目差(行棋方视角)', typeof r.scoreLead === 'number', String(r.scoreLead));
  clearEvalCache();
  const session2 = {
    async evalBatch(items) {
      return items.map(() => ({ policy: mkPolicy(), policyPass: -20, winLoss: 0.3 }));
    },
  };
  const r2 = await nnSearchBest(newBoard(), BLACK, { session: session2, visits: 40, batch: 4, symmetry: false, reuseTree: false });
  check('7d 无 score 头退化为纯胜率效用(老网口径)', r2.move === sq(9, 9), String(r2.move));
}

/* ==================== 8. 图搜索:子图复用 / GC / 转置合并 ==================== */
{
  /* 8a 跨手子图复用:同局面第二次思考(reuseTree 默认)大量命中既有节点 */
  clearEvalCache();
  const sessA = makeStub({ hot: [[sq(9, 9), 5], [sq(9, 10), 4]] });
  const bdA = newBoard();
  const t1 = await nnSearchBest(bdA, BLACK, { session: sessA, visits: 120, batch: 4, symmetry: false, allowResign: false });
  __clearBiasTable();   /* v4.4:隔离 bias 跨搜索学习,单验复用树的转置安全 */
  const t2 = await nnSearchBest(bdA, BLACK, { session: sessA, visits: 120, batch: 4, symmetry: false, allowResign: false });
  check('8a 跨手子图复用:第二次思考推理大减', t2.nnCalls < t1.nnCalls, `${t1.nnCalls} → ${t2.nnCalls}`);
  /* 单次搜索(冷表)的可达节点 ≤ 访问数 + 根:每次访问至多建一个新节点 */
  clearEvalCache();
  const t3 = await nnSearchBest(newBoard(), BLACK, { session: makeStub({ hot: [[sq(9, 9), 5], [sq(9, 10), 4]] }), visits: 80, batch: 4, symmetry: false, allowResign: false });
  check('8b GC:节点表以可达子图为界', __nodeTableSize() <= 80 + 2, `size=${__nodeTableSize()} visits=${t3.visits}`);
  check('8c 转置安全:访问用满且落点在热点集', t2.visits === 120
    && [sq(9, 9), sq(9, 10)].includes(t2.move) && [sq(9, 9), sq(9, 10)].includes(t1.move),
    `${t2.visits}/${t2.move}/${t1.move}`);

  /* 8d 转置合并:同一(盘面,行棋方)经不同路径到达 → 同一节点,只评估一次。
   * 桩按盘面选点(缺的星位点优先,两序都会被探到),并按盘面统计「敌方双星、
   * 恰两子」状态的评估行数 —— 该状态有两条可达路径(pass 先行、双星取序互换),
   * 状态键合并则恒 1,不合并则两路各评一次。通道 1/2 是行棋方相对的己/敌子。
   * (管线消掉了批内去重烧访,旧断言「表规模 < 访问数」的代理口径失效。) */
  clearEvalCache();
  const pair = [sq(9, 9), sq(9, 10)];
  let oppBothStars = 0;
  const posAware = { calls: 0, async evalBatch(items) {
    this.calls += items.length;
    return items.map((it) => {
      let n = 0, ownA = 0, ownB = 0;
      for (let p = 0; p < N2; p++) {
        if (it.spatial[1 * N2 + p] > 0.5 || it.spatial[2 * N2 + p] > 0.5) n++;
      }
      ownA = it.spatial[1 * N2 + pair[0]] > 0.5 ? 1 : 0;
      ownB = it.spatial[1 * N2 + pair[1]] > 0.5 ? 1 : 0;
      if (n === 2 && !ownA && !ownB) oppBothStars++;
      const policy = new Float32Array(N2).fill(-20);
      policy[pair[0]] = 6; policy[pair[1]] = 6;
      /* winLoss=0:值中性,路径分布由先验驱动 —— 视角修复(2026-10-05)后
       * 恒 0.3 的桩使黑叶效用反号、搜索改道,转置状态不再被探到 */
      return { policy, policyPass: -20, winLoss: 0.0 };
    });
  } };
  const bdB = newBoard();
  const tr = await nnSearchBest(bdB, BLACK, { session: posAware, visits: 80, batch: 4, symmetry: false, allowResign: false });
  check('8d 转置合并:双星状态跨路径只评估一次', oppBothStars === 1, `evals=${oppBothStars}`);
  check('8d2 节点表规模 ≤ 访问数+根', __nodeTableSize() <= tr.visits + 1 && tr.visits === 80,
    `size=${__nodeTableSize()} visits=${tr.visits}`);
  clearEvalCache();
  check('8e clearEvalCache 清节点表', __nodeTableSize() === 0, `size=${__nodeTableSize()}`);
}

/* ==================== 9. 单槽管线(双并行) ==================== */
{
  /* 异步慢桩:每次推理挂起宏任务模拟 GPU 在途。管线语义面:访问用满、
   * 决策可复现、评估行数不超预算、边界批/预算收敛。
   * 缓存键只认特征:每次运行前清缓存,保证桩配置隔离。 */
  const mkSlowStub = (ms) => {
    let rows = 0;
    return {
      get rows() { return rows; },
      async evalBatch(items) {
        rows += items.length;
        if (ms) await new Promise((r) => setTimeout(r, ms));
        const policy = new Float32Array(N2).fill(-20);
        policy[sq(9, 9)] = 6; policy[sq(9, 10)] = 5;
        return items.map(() => ({ policy, policyPass: -20, winLoss: 0.3 }));
      },
    };
  };
  const run = async (ms, batch, visits) => {
    clearEvalCache();
    const sess = mkSlowStub(ms);
    const r = await nnSearchBest(newBoard(), BLACK, {
      session: sess, visits, batch, symmetry: false, allowResign: false, temperature: 0,
    });
    return { ...r, rows: sess.rows };
  };
  const s1 = await run(1, 8, 80);
  const s2 = await run(1, 8, 80);
  check('9a 慢桩(管线)访问用满', s1.visits === 80 && s2.visits === 80, `${s1.visits}/${s2.visits}`);
  check('9b 管线决策可复现', s1.move === s2.move && s1.winRate === s2.winRate && s1.nnCalls === s2.nnCalls,
    `move ${s1.move}/${s2.move} nnCalls ${s1.nnCalls}/${s2.nnCalls}`);
  check('9c 评估行数不超预算(1 根评估行 + ≤80 叶行)', s1.rows - 1 <= 80 && s1.rows > 1, `rows=${s1.rows}`);
  const s3 = await run(1, 1, 7);
  const s4 = await run(2, 4, 33);
  check('9d 边界批/预算收敛', s3.visits === 7 && s4.visits === 33, `${s3.visits}/${s4.visits}`);

  /* 9e 预算压 stale:maxBatch 与预算取小(≤ 预算/16,下限 2)—— 80 访问传入
   * 上限 8,实际单批不得超过 5 行 */
  {
    clearEvalCache();
    let maxRows = 0;
    const sess = { async evalBatch(items) {
      maxRows = Math.max(maxRows, items.length);
      const p = new Float32Array(N2).fill(-20);
      p[sq(9, 9)] = 6;
      return items.map(() => ({ policy: p, policyPass: -20, winLoss: 0.3 }));
    } };
    const r = await nnSearchBest(newBoard(), BLACK, {
      session: sess, visits: 80, maxBatch: 8, symmetry: false, allowResign: false,
    });
    check('9e 批上限按预算压 stale(≤ 预算/16)', maxRows <= 5 && r.visits === 80,
      `maxRows=${maxRows} visits=${r.visits}`);
  }
}

/* ==================== 10. 批大小校准 ==================== */
{
  /* 纯选择:90% 容差内取最小批(吞吐单调升 → 最大;平坦 → 最小;
   * 饱和曲线(16 边际最优、32 掉头)→ 90% 线落在 8) */
  check('10a 饱和曲线选中边际最优小批',
    pickBatchSizeFromThroughput([[2, 0.1], [4, 0.333], [8, 1.0], [16, 1.067], [32, 0.8]]) === 8);
  check('10b 平坦吞吐选最小批',
    pickBatchSizeFromThroughput([[2, 1], [4, 1], [8, 1]]) === 2);
  check('10c 吞吐随批单调升选最大批',
    pickBatchSizeFromThroughput([[2, 0.5], [4, 1], [8, 2]]) === 8);
  check('10d 空表返回 null(调用方兜底)', pickBatchSizeFromThroughput([]) === null);

  /* 计时集成:固定 20ms/次的开销主导型桩 —— 批越大 rows/ms 越高,应选最大档。
   * margins 巨大(0.1 vs 1.6),对 setTimeout 粒度不敏感。 */
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const flat = async (rows) => {
    await sleep(20);
    return rows.map(() => ({ policy: new Float32Array(N2), policyPass: -20, winLoss: 0.3 }));
  };
  const t = await calibrateMaxBatch(flat, { iters: 2 });
  check('10e 计时校准:固定开销主导选最大批', t === 32, `got=${t}`);
}

/* ==================== 11. 搜索随机对称(nnEvaluator 同款) ====================
 * v4.3:每次评估随机取 8 对称之一(spatial 置换发送,policy/ownership 逆置换
 * 还原)。等变性往返检验:桩从收到的(已变换)spatial 通道 1(己方子)定位
 * 孤子标记 q0,回 policy 热点于 R180(q0);引擎逆置换后恒等系热点 =
 * σ⁻¹·R180·σ(p0) = R180(p0) —— R180 是 D4 群中心,共轭不变 —— 无论采样
 * 到哪个对称,选点必须落在固定空点,以此验证置换/逆置换方向正确。
 * rngSeed 固定时评估序列确定 → 变换后特征键复现 → 二次搜索命中缓存。 */
{
  const p0 = sq(3, 3), pT = 360 - p0;
  const symStub = () => ({ async evalBatch(items) {
    return items.map((r) => {
      const policy = new Float32Array(N2).fill(-20);
      let q0 = -1;
      for (let p = 0; p < N2; p++) if (r.spatial[N2 + p] > 0.5) { q0 = p; break; }
      if (q0 >= 0) policy[360 - q0] = 6;
      return { policy, policyPass: -20, winLoss: 0.3 };
    });
  } });
  const bdS = boardFrom([[3, 3, 'X']]);
  const r1 = await nnSearchBest(bdS, BLACK, {
    session: symStub(), visits: 60, symmetry: true, reuseTree: false, allowResign: false, rngSeed: 20261006,
  });
  const r2 = await nnSearchBest(bdS, BLACK, {
    session: symStub(), visits: 60, symmetry: true, reuseTree: false, allowResign: false, rngSeed: 20261006,
  });
  check('11a 等变往返:任意对称下选点落 R180(标记子)', r1.move === pT && r2.move === pT, `${r1.move}/${r2.move} want ${pT}`);
  check('11b rngSeed 固定可复现(选点/访问一致;缓存命中替代推理属预期)',
    r1.move === r2.move && r1.visits === r2.visits, `${r1.move}/${r2.move} ${r1.visits}/${r2.visits}`);
  check('11c 变换特征键命中缓存', r2.cacheHits > 0, `hits=${r2.cacheHits}`);
}

console.log(failed ? `\n${failed} 项失败` : '\n全部通过');
process.exit(failed ? 1 : 0);
