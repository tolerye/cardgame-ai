/**
 * 状态编码器 (JS 版)，必须严格对齐 train/encoder.py 的输出
 * 否则 NeuralAgent 推理出来是垃圾。
 *
 * 输出维度 62: PER_SELF(18) + PER_OPP(8)*3 + GLOBAL(20)
 */

import { PlayerStatus } from './state.js';

const STATUSES = [
  PlayerStatus.ACTIVE, PlayerStatus.FOLDED,
  PlayerStatus.EXILED, PlayerStatus.BUSTED,
];

export const FEATURE_DIM = 62;

/**
 * @param {GameState} state
 * @param {number} myIdx
 * @returns {Float32Array} length 62
 */
export function encodeState(state, myIdx) {
  const n = state.config.numPlayers;
  const target = state.config.targetScore;
  const me = state.players[myIdx];
  const out = new Float32Array(FEATURE_DIM);
  let i = 0;

  // ---- self (18) ----
  const counts = new Array(13).fill(0);
  for (const v of me.handNumbers) counts[v] += 1;
  for (let v = 0; v < 13; v++) out[i++] = counts[v];
  out[i++] = me.bonusFlatTotal / 50.0;
  out[i++] = me.hasInsurance ? 1.0 : 0.0;
  out[i++] = me.totalScore / target;
  const handSum = me.handNumbers.reduce((s, v) => s + v, 0);
  out[i++] = (handSum + me.bonusFlatTotal) / target;
  out[i++] = me.uniqueNumbers.size / 6.0;

  // ---- opponents (8*3=24) ----
  // 按相对座位顺序排（policy 对座位轮换不变）
  const ordered = state.players
    .filter(p => p.index !== myIdx)
    .slice()
    .sort((a, b) => ((a.index - myIdx + n) % n) - ((b.index - myIdx + n) % n));
  for (const p of ordered) {
    out[i++] = p.totalScore / target;
    out[i++] = p.currentRoundScore() / target;
    out[i++] = p.hasInsurance ? 1.0 : 0.0;
    out[i++] = p.uniqueNumbers.size / 6.0;
    for (const s of STATUSES) out[i++] = p.status === s ? 1.0 : 0.0;
  }
  // pad to 3 opponents
  for (let pad = ordered.length; pad < 3; pad++) {
    for (let k = 0; k < 8; k++) out[i++] = 0.0;
  }

  // ---- global (20) ----
  const rem = state.remaining;
  const deckTotal = rem.total();
  const denom = Math.max(deckTotal, 1);
  for (let v = 0; v < 13; v++) out[i++] = rem.numbers[v] / denom;
  out[i++] = rem.bonus_flat / denom;
  out[i++] = rem.bonus_double / denom;
  out[i++] = rem.insurance / denom;
  out[i++] = rem.exile / denom;
  out[i++] = rem.triple / denom;
  out[i++] = deckTotal / 94.0;
  out[i++] = state.roundNumber / 20.0;

  return out;
}
