/**
 * MC 胜率/排位预估器（JS 版）。
 * 给定 4 人当前总分 + agent 风格，模拟若干场 match，统计每人 1/2/3/4 名概率。
 *
 * 因为 engine.playMatch 是 async 的（为了 HumanAgent），但 sim 用的 agents 全部
 * 同步，所以 await 会立即 resolve；MC 整体仍能在主线程跑完。每 N 场 yield 一次让
 * UI 不卡死。
 */

import { GameEngine } from './engine.js';
import { GameConfig } from './state.js';
import { EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent } from './agents.js';

const FACTORY = {
  random: () => new RandomAgent(),
  greedy: () => new GreedyAgent(),
  ev: () => new EVAgent(),
  exmax2: () => new ExpectimaxAgent({ depth: 2 }),
  exmax3: () => new ExpectimaxAgent({ depth: 3 }),
  exmax4: () => new ExpectimaxAgent({ depth: 4 }),
};

/**
 * 跑一场 sim：从 startScores 出发，到比赛结束。
 * @param {number[]} startScores
 * @param {string[]} agentNames
 * @param {number} targetScore
 * @returns {Promise<{rankPos: number[], finalScores: number[]}>}
 */
async function runOne(startScores, agentNames, targetScore) {
  const cfg = new GameConfig({
    numPlayers: startScores.length,
    targetScore,
  });
  const agents = agentNames.map(n => FACTORY[n]());
  const engine = new GameEngine(cfg, agents);
  // 设置起始总分
  for (let i = 0; i < startScores.length; i++) {
    engine.state.players[i].totalScore = startScores[i];
  }
  // 已经达到 target？直接返回
  const maxNow = Math.max(...startScores);
  if (maxNow >= targetScore) {
    const final = startScores.slice();
    const rank = startScores.map((_, i) => i).sort((a, b) => final[b] - final[a]);
    const rp = new Array(startScores.length).fill(0);
    rank.forEach((idx, pos) => rp[idx] = pos);
    return { rankPos: rp, finalScores: final };
  }

  await engine.playMatch();

  const final = engine.state.players.map(p => p.totalScore);
  const n = final.length;
  const rank = final.map((_, i) => i).sort((a, b) => final[b] - final[a]);
  const rp = new Array(n).fill(0);
  rank.forEach((idx, pos) => rp[idx] = pos);
  return { rankPos: rp, finalScores: final };
}

/**
 * 跑 nSims 场 MC，回调 onProgress(done, total) 用于进度条。
 *
 * @param {number[]} startScores
 * @param {string[]} agentNames
 * @param {number} targetScore
 * @param {number} nSims
 * @param {(done: number, total: number) => void} [onProgress]
 * @returns {Promise<{
 *   nSims: number,
 *   elapsedMs: number,
 *   rankProb: number[][],   // rankProb[i] = [P(1名), P(2名), P(3名), P(4名)]
 *   expectedFinal: number[],
 * }>}
 */
export async function estimate(startScores, agentNames, targetScore, nSims, onProgress) {
  const t0 = performance.now();
  const n = startScores.length;
  const rankCounts = Array.from({ length: n }, () => new Array(n).fill(0));
  const sumFinal = new Array(n).fill(0);

  for (let s = 0; s < nSims; s++) {
    const { rankPos, finalScores } = await runOne(startScores, agentNames, targetScore);
    for (let i = 0; i < n; i++) {
      rankCounts[i][rankPos[i]] += 1;
      sumFinal[i] += finalScores[i];
    }
    if (onProgress && (s + 1) % 5 === 0) onProgress(s + 1, nSims);
    // 每 10 场让一下 UI
    if ((s + 1) % 10 === 0) await new Promise(r => setTimeout(r, 0));
  }

  const rankProb = rankCounts.map(row => row.map(c => c / nSims));
  const expectedFinal = sumFinal.map(s => s / nSims);

  return {
    nSims,
    elapsedMs: performance.now() - t0,
    rankProb,
    expectedFinal,
  };
}
