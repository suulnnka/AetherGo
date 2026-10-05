/* ============================================================
 * AetherGo 引擎对外门面 —— 规则(src/rules.js)+ 数子(src/scoring.js)
 *
 * 本文件只做再导出与记谱:所有既有 import 路径(测试 / Worker / NN 模块)
 * 保持 `from '../src/engine.js'` 或 `from './engine.js'` 不变。
 * 分层:protocol(契约常量)→ rules(走子/合法性/superko)→ scoring(数子/死子)
 *       → nn(PUCT 搜索)→ nn-worker(Worker 门面)。
 * 本文件不碰 DOM、不 import 任何库 —— 浏览器、Worker、Node 通用。
 * ============================================================ */
import { formatMove } from './protocol.js';

// 契约常量(protocol.js —— UI 与引擎的唯一共享层)
export { N, N2, BLACK, WHITE, EMPTY, PASS, KOMI } from './protocol.js';

// 规则核心(rules.js):棋盘状态、走子/撤销、合法性、禁全同
export {
  stoneOf, sideOf, newBoard, syncPosition, replayMoves,
  make, unmake, capturedOf, genLegal, isLegal,
  superkoBannedPoints, koPoint, positionKey, boardToArray, arrayToBoard,
  recentPrevBd, recentPrevKo, moveCount, ringSnapshot, ringRestore,
} from './rules.js';

// 数子与死子(scoring.js):中国规则数子、Benson + 提子搜索、ownership 辅助标注
export {
  evaluate, scoreGame, scoreBreakdown, finalScore,
  deadStones, deadStonesWithOwnership,
} from './scoring.js';

/* ==================== 记谱 ==================== */

/** 记谱(显示用):交叉点 → 列字母 + 行号(下边为 1);PASS → 停一手。
 *  围棋记谱不依赖盘面,但与象棋引擎保持同签名(bd, mv)。 */
export function moveToText(bd, mv) {
  return formatMove(mv);
}
