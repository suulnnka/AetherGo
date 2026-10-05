/* ============================================================
 * AetherGo 对弈契约 —— UI 与引擎之间的唯一共享层
 *
 * 这里只放「走法编码 / 坐标记谱」这类**协议常量**:UI(pages/)与引擎
 * (src/)各自 import 这一份,改盘面尺寸只动这里。
 * 引擎规则(src/rules.js、src/scoring.js)与 NN(src/nn/)都构建在它之上;
 * UI 只 import 本文件,不 import 任何引擎代码(事实全部经 Worker 消息获取)。
 * ============================================================ */

/** 路数(19×19;全仓尺寸参数化的唯一来源) */
export const N = 19;
/** 交叉点数 361 */
export const N2 = N * N;

/** 行棋方:黑先白后;棋子值 = 行棋方 + 1 */
export const BLACK = 0, WHITE = 1;
/** 空点 */
export const EMPTY = 0;
/** 停一手;走法编码 0..360 是交叉点,361 是停一手 */
export const PASS = N2;
/** 黑贴目(中国规则 19 路常用 7.5) */
export const KOMI = 7.5;

/** 列标(跳过 I):与 moveToText / 棋盘坐标共用 */
export const COLS = 'ABCDEFGHJKLMNOPQRST';

/** 记谱(显示用):交叉点 → 列字母 + 行号(下边为 1);PASS → 停一手 */
export const formatMove = (mv) =>
  mv === PASS ? '停一手' : COLS[mv % N] + String(N - ((mv / N) | 0));
