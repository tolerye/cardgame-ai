/**
 * 手机端游戏页面：可爱风 UI，独立渲染逻辑，复用 engine.js / agents.js。
 */

import { GameEngine } from './engine.js';
import { GameConfig, PlayerStatus } from './state.js';
import { CardKind } from './cards.js';
import { EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent } from './agents.js';
import { NeuralAgent } from './neural-agent.js';
import { estimate as estimateWinrate } from './winrate.js';

// ============================================================ 全局
// 神经网络 lazy 加载
let _neuralPromise = null;
function loadNeural() {
  if (!_neuralPromise) _neuralPromise = NeuralAgent.load('model.json');
  return _neuralPromise;
}

const FACTORY = {
  random: () => new RandomAgent(),
  greedy: () => new GreedyAgent(),
  ev: () => new EVAgent(),
  exmax2: () => new ExpectimaxAgent({ depth: 2 }),
  exmax3: () => new ExpectimaxAgent({ depth: 3 }),
  exmax4: () => new ExpectimaxAgent({ depth: 4 }),
  neural: async () => await loadNeural(),
};
const NAME_CN = {
  random: '随机',
  greedy: '贪心',
  ev: 'EV',
  exmax2: '期望 D2',
  exmax3: '期望 D3',
  exmax4: '期望 D4',
  neural: '神经网',
};
const OPP_EMOJI = {
  random: '🎲',
  greedy: '🤖',
  ev: '📊',
  exmax2: '🧠',
  exmax3: '🧠',
  exmax4: '🧠',
  neural: '🤖✨',
};

let engine;
let myIdx = 0;
let oppNames = [];
let logTrackedLen = 0;
let showReco = true;
let showWinrate = false;
let lastWinrateRound = -1;

// ============================================================ Toggle UI
document.querySelectorAll('.toggle').forEach(t => {
  t.addEventListener('click', () => {
    const on = t.dataset.on === 'true';
    t.dataset.on = (!on).toString();
    t.classList.toggle('on', !on);
  });
});

// ============================================================ Setup → Game
document.getElementById('btn-start').addEventListener('click', startGame);
document.getElementById('btn-quit').addEventListener('click', () => {
  if (confirm('确定退出当前对局？')) location.reload();
});
document.getElementById('btn-again').addEventListener('click', () => location.reload());

async function startGame() {
  const target = parseInt(document.getElementById('target-input').value) || 200;
  const oppKeys = [
    document.getElementById('opp-1').value,
    document.getElementById('opp-2').value,
    document.getElementById('opp-3').value,
  ];
  showReco = document.getElementById('t-reco').dataset.on === 'true';
  showWinrate = document.getElementById('t-winrate').dataset.on === 'true';

  document.getElementById('reco-panel').hidden = !showReco;
  document.getElementById('winrate-panel').hidden = !showWinrate;
  document.getElementById('winrate-run').addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    runWinrate();
  });

  oppNames = oppKeys.map(k => NAME_CN[k] || k);

  document.getElementById('setup').hidden = true;
  document.getElementById('game').hidden = false;

  const cfg = new GameConfig({ numPlayers: 4, targetScore: target });
  const human = new HumanAgent();
  const aiAgents = [];
  for (const k of oppKeys) {
    const a = await FACTORY[k]();
    aiAgents.push(wrapWithDelay(a));
  }
  engine = new GameEngine(cfg, [human, ...aiAgents]);

  try {
    const winner = await engine.playMatch();
    render();
    setTimeout(() => showResult(winner), 500);
  } catch (e) {
    console.error(e);
    alert('对局出错：' + (e?.message || e));
  }
}

// ============================================================ Human agent
class HumanAgent {
  constructor() { this.name = 'human'; }

  chooseAction(state, idx) {
    myIdx = idx;
    render();
    flashDrawnCard();
    if (showReco) renderReco();
    return new Promise(resolve => {
      const drawBtn = document.getElementById('btn-draw');
      const foldBtn = document.getElementById('btn-fold');
      drawBtn.disabled = false;
      foldBtn.disabled = false;
      drawBtn.classList.add('your-turn');
      const finish = (action) => {
        drawBtn.disabled = true;
        foldBtn.disabled = true;
        drawBtn.classList.remove('your-turn');
        drawBtn.removeEventListener('click', onDraw);
        foldBtn.removeEventListener('click', onFold);
        document.removeEventListener('keydown', onKey);
        resolve(action);
      };
      const onDraw = () => finish('draw');
      const onFold = () => finish('fold');
      const onKey = (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'd' || e.key === 'D') { e.preventDefault(); finish('draw'); }
        if (e.key === 'f' || e.key === 'F') { e.preventDefault(); finish('fold'); }
      };
      drawBtn.addEventListener('click', onDraw);
      foldBtn.addEventListener('click', onFold);
      document.addEventListener('keydown', onKey);
    });
  }

  chooseSkillTarget(state, idx, kind) {
    return new Promise(resolve => showSkillSheet(state, idx, kind, resolve));
  }
}

// AI 包装：每次 AI 操作前重渲染 + 短暂延迟
function wrapWithDelay(agent, delayMs = 280) {
  return {
    name: agent.name,
    async chooseAction(state, idx) {
      render();
      flashDrawnCard();
      await sleep(delayMs);
      return agent.chooseAction(state, idx);
    },
    async chooseSkillTarget(state, idx, kind) {
      await sleep(180);
      return agent.chooseSkillTarget(state, idx, kind);
    },
  };
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ============================================================ Render
function render() {
  const state = engine.state;
  document.getElementById('round-num').textContent = state.roundNumber;
  document.getElementById('deck-remaining').textContent = state.remaining.total();

  renderHero(state);
  renderHint(state);
  renderOpps(state);
  renderLog(state);
}

function renderHero(state) {
  const me = state.players[myIdx];
  const target = state.config.targetScore;
  const cur = me.currentRoundScore();
  const pct = Math.min(100, me.totalScore / target * 100);
  const statusLabel = {
    [PlayerStatus.ACTIVE]: '🎯 进行中',
    [PlayerStatus.FOLDED]: '🔒 已跑路',
    [PlayerStatus.BUSTED]: '💥 爆牌',
    [PlayerStatus.EXILED]: '🚷 被放逐',
  }[me.status] || '';
  const handHtml = me.handNumbers.length
    ? me.handNumbers.slice().sort((a, b) => a - b).map(v => `<div class="hand-card">${v}</div>`).join('')
    : '<div style="color:rgba(255,255,255,.7);font-size:13px;padding:6px">还没摸牌</div>';
  const bonusHtml = me.bonusFlatTotal > 0
    ? `<div class="hand-card tag-bonus">+${me.bonusFlatTotal}</div>` : '';
  const insTag = me.hasInsurance ? '<span class="hero-tag">🛡 保险</span>' : '';
  document.getElementById('hero').innerHTML = `
    <div class="hero-top">
      <div class="hero-name">★ 你</div>
      <div class="hero-status">${statusLabel}</div>
    </div>
    <div class="hero-score">${me.totalScore}<span class="target"> / ${target}</span></div>
    <div class="hero-bar"><div class="hero-bar-fill" style="width:${pct}%"></div></div>
    <div class="hero-row">
      <span class="hero-tag">本局 +${cur}</span>
      ${insTag}
    </div>
    <div class="hero-hand">${handHtml}${bonusHtml}</div>
  `;
}

function renderHint(state) {
  const me = state.players[myIdx];
  const hintEl = document.getElementById('hint');
  const target = state.config.targetScore;
  const cur = me.currentRoundScore();
  const handSet = me.uniqueNumbers;
  const total = Math.max(state.remaining.total(), 1);
  let bustC = 0; for (const v of handSet) bustC += state.remaining.numbers[v];
  const bustP = bustC / total * 100;

  if (!me.isActive) {
    hintEl.className = 'hint-box';
    hintEl.innerHTML = `<b>${{
      [PlayerStatus.FOLDED]: '🔒 你已跑路',
      [PlayerStatus.BUSTED]: '💥 你已爆牌',
      [PlayerStatus.EXILED]: '🚷 你已被放逐',
    }[me.status] || '观战中'}</b> · 等其他玩家结束本局`;
    return;
  }
  if (me.totalScore + cur >= target) {
    hintEl.className = 'hint-box success';
    hintEl.innerHTML = `🏆 <b>立即跑路即可获胜！</b>锁 ${cur} 分赢比赛！`;
    return;
  }
  let html = `爆牌概率 <b>${bustP.toFixed(0)}%</b>`;
  if (handSet.size === 5) {
    let safeC = 0;
    for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeC += state.remaining.numbers[v];
    const sixP = safeC / total * 100;
    html += ` · 🚀 <b>6翻成功率 ${sixP.toFixed(0)}%</b>`;
  }
  html += ` · 跑路锁 <b>${cur}</b> 分`;
  if (bustP > 35) hintEl.className = 'hint-box danger';
  else if (handSet.size === 5) hintEl.className = 'hint-box success';
  else hintEl.className = 'hint-box';
  hintEl.innerHTML = html;
}

function renderOpps(state) {
  const oppsEl = document.getElementById('opps');
  oppsEl.innerHTML = '';
  state.players.forEach(p => {
    if (p.index === myIdx) return;
    const oppKey = oppNames[p.index - 1];
    const target = state.config.targetScore;
    const card = document.createElement('div');
    card.className = `opp-card status-${p.status}`;
    if (state.currentPlayer === p.index && p.isActive && !state.gameOver) {
      card.classList.add('is-turn');
    }
    const pct = Math.min(100, p.totalScore / target * 100);
    const statusBadge = {
      [PlayerStatus.FOLDED]: '<span class="opp-status-badge folded">跑路</span>',
      [PlayerStatus.BUSTED]: '<span class="opp-status-badge busted">爆</span>',
      [PlayerStatus.EXILED]: '<span class="opp-status-badge exiled">放逐</span>',
    }[p.status] || '';
    const ins = p.hasInsurance ? ' 🛡' : '';
    const cur = p.currentRoundScore();
    card.innerHTML = `
      <div class="opp-emoji">${OPP_EMOJI[oppKeyOf(p.index)] || '🤖'}</div>
      <div class="opp-name">P${p.index} ${oppKey || ''}</div>
      <div class="opp-score">${p.totalScore}</div>
      <div class="opp-meta">本局 +${cur}${ins}</div>
      ${statusBadge}
      <div class="opp-bar"><div class="opp-bar-fill" style="width:${pct}%"></div></div>
    `;
    oppsEl.appendChild(card);
  });
}
function oppKeyOf(idx) {
  if (idx === 0) return null;
  // can't easily recover the FACTORY key; just look at NAME_CN by reverse
  // (we stored opp name in oppNames[i-1] but it's CN; for emoji lookup we use position)
  return Object.keys(NAME_CN).find(k => NAME_CN[k] === oppNames[idx - 1]);
}

function renderLog(state) {
  const ul = document.getElementById('log');
  ul.innerHTML = '';
  // 最后 25 条事件
  const lines = state.log.slice(-25);
  lines.forEach(line => {
    const li = document.createElement('li');
    const m = line.match(/R(\d+)\|P(\d+):\s*(.*)/);
    let txt = line;
    if (m) {
      const round = m[1];
      const pid = parseInt(m[2]);
      const event = m[3];
      const who = pid === myIdx ? '★ 你' : `P${pid}`;
      txt = `R${round} ${who} ${translateEvent(event)}`;
      if (pid === myIdx) li.classList.add('is-me');
    }
    if (line.includes('BUST') && !line.includes('AVOIDED')) li.classList.add('event-bust');
    else if (line.includes('SIX-BURST')) li.classList.add('event-six');
    else if (line.includes('FOLD')) li.classList.add('event-fold');
    else if (line.includes('EXILE') || line.includes('TRIPLE') || line.includes('INSURANCE')) li.classList.add('event-skill');
    li.textContent = txt;
    ul.appendChild(li);
  });
  ul.scrollTop = ul.scrollHeight;
}

function translateEvent(s) {
  return s
    .replace(/SIX-BURST! lock=(\d+)/, '🚀 6翻了！锁 $1')
    .replace(/FOLD lock=(\d+)/, '🔒 跑路锁 $1')
    .replace(/DRAW (\d+)/, '抽到 $1')
    .replace(/BUST_AVOIDED on (\d+)/, '重复 $1（保险免）')
    .replace(/BUST on (\d+)/, '💥 爆牌！重复 $1')
    .replace(/BONUS\+(\d+)/, '加分牌 +$1')
    .replace(/DOUBLE \+(\d+)/, '翻倍 +$1')
    .replace(/INSURANCE\+/, '🛡 拿到保险')
    .replace(/INSURANCE -> P(\d+)/, '🛡 送给 P$1')
    .replace(/EXILE -> P(\d+) lock=(\d+)/, '🚷 放逐 P$1（锁 $2）')
    .replace(/TRIPLE -> P(\d+)/, '⚡ 三连 → P$1');
}

// ============================================================ Drawn card flash
function flashDrawnCard() {
  const state = engine.state;
  const newLogs = state.log.slice(logTrackedLen);
  logTrackedLen = state.log.length;
  // 找最后一个新 DRAW/BONUS/skill 事件
  const last = [...newLogs].reverse().find(l => /DRAW|BONUS|DOUBLE|INSURANCE\+|EXILE|TRIPLE|BUST/.test(l));
  if (!last) return;
  const m = last.match(/R\d+\|P(\d+):/);
  if (!m) return;
  const pid = parseInt(m[1]);
  let label = '', emoji = '🃏', color = '#FF6B9D';
  let mm;
  if ((mm = last.match(/DRAW (\d+)/))) { label = mm[1]; emoji = '🃏'; color = '#FF6B9D'; }
  else if (/BONUS\+/.test(last)) { label = '+10'; emoji = '⭐'; color = '#FFD93D'; }
  else if (/DOUBLE/.test(last)) { label = '×2'; emoji = '✨'; color = '#FFD93D'; }
  else if (/INSURANCE\+/.test(last)) { label = '保险'; emoji = '🛡'; color = '#6BB6FF'; }
  else if (/EXILE -> /.test(last)) { label = '放逐'; emoji = '🚷'; color = '#B197FC'; }
  else if (/TRIPLE -> /.test(last)) { label = '三连'; emoji = '⚡'; color = '#B197FC'; }
  else if (/BUST on /.test(last)) { label = '爆牌'; emoji = '💥'; color = '#FF6B6B'; }
  else return;
  const flash = document.getElementById('drawn-flash');
  flash.style.color = color;
  flash.innerHTML = `${emoji} <span style="font-size:20px;color:#6B5B7B"> P${pid}</span><br>${label}`;
  flash.classList.remove('show');
  void flash.offsetWidth; // restart animation
  flash.classList.add('show');
}

// ============================================================ Skill modal
function showSkillSheet(state, idx, kind, onPick) {
  const overlay = document.getElementById('skill-overlay');
  const KIND_LABEL = {
    [CardKind.EXILE]: '🚷 放逐',
    [CardKind.TRIPLE]: '⚡ 三连',
    [CardKind.INSURANCE]: '🛡 保险',
  };
  document.getElementById('skill-title').textContent = `★ ${KIND_LABEL[kind] || kind}`;
  document.getElementById('skill-desc').textContent = {
    [CardKind.EXILE]: '选一个对象强制跑路（可选自己）',
    [CardKind.TRIPLE]: '选一个对象连摸 3 张（手牌 5 张时可选自己冲 6 翻）',
    [CardKind.INSURANCE]: '你已有保险，必须把这张转送给一名其他玩家',
  }[kind] || '';
  // 推荐目标（用 EVAgent 计算）
  let recommended = -1;
  try { recommended = new EVAgent().chooseSkillTarget(state, idx, kind); } catch {}

  const targets = document.getElementById('skill-targets');
  targets.innerHTML = '';
  state.players.filter(p => {
    if (!p.isActive) return false;
    if (kind === CardKind.INSURANCE && p.index === idx) return false;
    return true;
  }).forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'skill-target' + (p.index === recommended ? ' recommended' : '');
    const hand = p.handNumbers.slice().sort((a, b) => a - b).join(',') || '空';
    const ins = p.hasInsurance ? ' 🛡' : '';
    const youMark = p.index === idx ? ' ← 你' : '';
    const recMark = p.index === recommended ? ' ⭐ 推荐' : '';
    btn.innerHTML = `
      <div>P${p.index}${youMark}${recMark}</div>
      <small>总分 ${p.totalScore} · 本局 +${p.currentRoundScore()} · [${hand}]${ins}</small>
    `;
    btn.addEventListener('click', () => {
      overlay.hidden = true;
      onPick(p.index);
    });
    targets.appendChild(btn);
  });
  overlay.hidden = false;
}

// ============================================================ Reco panel
function renderReco() {
  const state = engine.state;
  const me = state.players[myIdx];
  const el = document.getElementById('reco-content');
  if (!me.isActive) {
    el.innerHTML = '<div style="color:var(--text-dim);font-size:12px">不是你的回合</div>';
    return;
  }
  const target = state.config.targetScore;
  const cur = me.currentRoundScore();
  if (me.totalScore + cur >= target) {
    el.innerHTML = `<div class="reco-verdict draw" style="background:var(--green-soft)">🏆 跑路即获胜！锁 ${cur} 分</div>`;
    return;
  }

  // 调 expectimax + 概率
  let decision = 'fold';
  try { decision = new ExpectimaxAgent({ depth: 3 }).chooseAction(state, myIdx); } catch {}

  const total = Math.max(state.remaining.total(), 1);
  const handSet = me.uniqueNumbers;
  let bustC = 0; for (const v of handSet) bustC += state.remaining.numbers[v];
  const pBust = bustC / total * 100;
  let safeC = 0;
  for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeC += state.remaining.numbers[v];
  const pSafe = safeC / total * 100;
  const pSix = handSet.size === 5 ? pSafe : 0;

  const pBonus = state.remaining.bonus_flat / total * 100;
  const pDouble = state.remaining.bonus_double / total * 100;
  const pSkill = (state.remaining.insurance + state.remaining.exile + state.remaining.triple) / total * 100;

  let html = '';
  // 主建议
  if (decision === 'draw') {
    html += `<div class="reco-verdict draw">🎲 建议：摸牌</div>`;
  } else {
    html += `<div class="reco-verdict fold">🔒 建议：跑路（锁 ${cur} 分）</div>`;
  }
  const bustCls = pBust > 30 ? 'danger' : pBust > 15 ? 'warn' : 'good';
  html += `
    <div class="reco-row ${bustCls}"><span>💥 爆牌</span><b>${pBust.toFixed(1)}%</b></div>
  `;
  if (handSet.size < 5) {
    html += `<div class="reco-row good"><span>✅ 安全数字</span><b>${pSafe.toFixed(1)}%</b></div>`;
  } else {
    const sixCls = pSix > 50 ? 'good' : 'warn';
    html += `<div class="reco-row ${sixCls}"><span>🚀 6 翻终结</span><b>${pSix.toFixed(1)}%</b></div>`;
  }
  html += `
    <div class="reco-row"><span>⭐ 加分牌</span><b>${pBonus.toFixed(1)}%</b></div>
    <div class="reco-row"><span>✨ 翻倍</span><b>${pDouble.toFixed(1)}%</b></div>
    <div class="reco-row"><span>🃏 技能</span><b>${pSkill.toFixed(1)}%</b></div>
  `;
  el.innerHTML = html;
}

// ============================================================ Winrate panel
async function runWinrate() {
  const state = engine.state;
  const btn = document.getElementById('winrate-run');
  const el = document.getElementById('winrate-content');
  btn.disabled = true;
  btn.textContent = '...';
  try {
    const scores = state.players.map(p => p.totalScore);
    const target = state.config.targetScore;
    const result = await estimateWinrate(scores, ['ev', 'ev', 'ev', 'ev'], target, 200);
    let topRank1 = 0;
    for (let i = 0; i < 4; i++) if (result.rankProb[i][0] > result.rankProb[topRank1][0]) topRank1 = i;
    let html = `<div style="font-size:11px;color:var(--text-dim);margin-bottom:6px">200 局，${(result.elapsedMs/1000).toFixed(1)}s · 最可能夺冠：<b style="color:var(--pink-deep)">P${topRank1}</b> ${(result.rankProb[topRank1][0]*100).toFixed(0)}%</div>`;
    html += '<table style="width:100%;font-size:12px;border-collapse:collapse">';
    html += '<thead style="color:var(--text-dim);font-size:10px"><tr><th style="text-align:left;padding:3px">玩家</th><th style="text-align:right;padding:3px">分</th><th style="text-align:right;padding:3px">1名</th><th style="text-align:right;padding:3px">2</th><th style="text-align:right;padding:3px">3</th><th style="text-align:right;padding:3px">4</th></tr></thead><tbody>';
    for (let i = 0; i < 4; i++) {
      const isMe = i === myIdx;
      const tag = isMe ? '★' : '';
      const [p1, p2, p3, p4] = result.rankProb[i].map(x => x * 100);
      const c1 = p1 >= 50 ? 'color:var(--green);font-weight:700' : '';
      const c4 = p4 >= 50 ? 'color:var(--red)' : '';
      html += `<tr style="${isMe?'background:rgba(255,107,157,.08)':''}">`
        + `<td style="padding:3px">${tag}P${i}</td>`
        + `<td style="text-align:right;padding:3px">${state.players[i].totalScore}</td>`
        + `<td style="text-align:right;padding:3px;${c1}">${p1.toFixed(0)}%</td>`
        + `<td style="text-align:right;padding:3px">${p2.toFixed(0)}%</td>`
        + `<td style="text-align:right;padding:3px">${p3.toFixed(0)}%</td>`
        + `<td style="text-align:right;padding:3px;${c4}">${p4.toFixed(0)}%</td>`
        + `</tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = `<div style="color:var(--red);font-size:12px">出错：${e.message}</div>`;
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ 跑';
  }
}

// ============================================================ Result
function showResult(winnerIdx) {
  const state = engine.state;
  const overlay = document.getElementById('result-overlay');
  const myWon = winnerIdx === myIdx;
  document.getElementById('result-emoji').textContent = myWon ? '🎉' : '🏁';
  document.getElementById('result-title').textContent =
    myWon ? '恭喜，你赢了！' : `P${winnerIdx} ${oppNames[winnerIdx-1] || ''} 获胜`;
  const rank = document.getElementById('result-rank');
  rank.innerHTML = '';
  const sorted = state.players.slice().sort((a, b) => b.totalScore - a.totalScore);
  const medals = ['🥇', '🥈', '🥉', '4️⃣'];
  sorted.forEach((p, i) => {
    const li = document.createElement('li');
    li.className = `rank-${i+1}` + (p.index === myIdx ? ' is-me' : '');
    const name = p.index === myIdx ? '你' : `P${p.index}（${oppNames[p.index-1] || ''}）`;
    li.innerHTML = `<span class="pos">${medals[i]}</span><span class="who">${name}</span><span class="pts">${p.totalScore}</span>`;
    rank.appendChild(li);
  });
  overlay.hidden = false;
  if (myWon) launchConfetti();
}

function launchConfetti() {
  const colors = ['#FF6B9D', '#FFD93D', '#6BCB77', '#B197FC', '#FF9F6B', '#6BB6FF'];
  for (let i = 0; i < 60; i++) {
    const c = document.createElement('div');
    c.className = 'confetti';
    c.style.left = `${Math.random() * 100}%`;
    c.style.background = colors[i % colors.length];
    c.style.animationDelay = `${Math.random() * 1.5}s`;
    c.style.animationDuration = `${2.5 + Math.random() * 1.5}s`;
    document.body.appendChild(c);
    setTimeout(() => c.remove(), 4500);
  }
}
