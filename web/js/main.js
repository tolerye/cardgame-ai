/**
 * Entry point: wires up Setup → Game → Result.
 *
 * AI agents are wrapped with a small render+sleep delay so the player can
 * follow opponents' moves visually before the next decision happens.
 */

import { GameEngine } from './engine.js';
import { GameConfig } from './state.js';
import { EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent } from './agents.js';
import { HumanAgent, render, setupUI, showResult, resetFlashTracker, flashDrawnCard } from './ui.js';

const FACTORIES = {
  exmax4: () => new ExpectimaxAgent({ depth: 4 }),
  exmax3: () => new ExpectimaxAgent({ depth: 3 }),
  exmax2: () => new ExpectimaxAgent({ depth: 2 }),
  expectimax: () => new ExpectimaxAgent({ depth: 3 }),  // 别名
  ev: () => new EVAgent(),
  greedy: () => new GreedyAgent(),
  random: () => new RandomAgent(),
};

const NAME_CN = {
  exmax4: 'Exmax depth=4',
  exmax3: 'Exmax depth=3',
  exmax2: 'Exmax depth=2',
  expectimax: 'Expectimax',
  ev: 'EV',
  greedy: 'Greedy',
  random: 'Random',
};

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/** Wrap an AI agent so each turn first re-renders the board so the human can
 *  see what just happened, then pauses briefly. */
function wrapWithDelay(agent, getState, opponentDelayMs = 320) {
  return {
    name: agent.name,
    async chooseAction(state, idx) {
      render(state, 0);
      flashDrawnCard(state, idx); // 显示对手刚抽到的牌（如果上一回合有 DRAW）
      await sleep(opponentDelayMs);
      const a = agent.chooseAction(state, idx);
      return a;
    },
    async chooseSkillTarget(state, idx, kind) {
      await sleep(200);
      return agent.chooseSkillTarget(state, idx, kind);
    },
  };
}

document.getElementById('start-btn').addEventListener('click', startGame);

document.addEventListener('click', (e) => {
  if (e.target.id === 'play-again' || e.target.id === 'quit-btn') {
    location.reload();
  }
});

async function startGame() {
  const target = parseInt(document.getElementById('target-input').value) || 200;

  const oppKeys = ['', '', '', ''];
  document.querySelectorAll('.opp-select').forEach(sel => {
    oppKeys[parseInt(sel.dataset.idx)] = sel.value;
  });

  // 切换界面
  document.getElementById('setup').hidden = true;
  document.getElementById('game').hidden = false;

  const human = new HumanAgent();
  const agents = [
    human,
    wrapWithDelay(FACTORIES[oppKeys[1]]()),
    wrapWithDelay(FACTORIES[oppKeys[2]]()),
    wrapWithDelay(FACTORIES[oppKeys[3]]()),
  ];

  const displayNames = ['你', NAME_CN[oppKeys[1]], NAME_CN[oppKeys[2]], NAME_CN[oppKeys[3]]];
  setupUI(displayNames, 0);
  resetFlashTracker(0);

  const cfg = new GameConfig({ numPlayers: 4, targetScore: target });
  const engine = new GameEngine(cfg, agents);

  try {
    const winner = await engine.playMatch();
    render(engine.state, 0);
    setTimeout(() => showResult(winner, engine.state.players), 500);
  } catch (e) {
    console.error('Match crashed:', e);
    alert('对局出错：' + (e?.message || e));
  }
}

// 默认按钮禁用
document.getElementById('btn-draw').disabled = true;
document.getElementById('btn-fold').disabled = true;
