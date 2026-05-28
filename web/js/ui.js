/**
 * UI module: HumanAgent + render functions + skill-target picker + result modal.
 *
 * The HumanAgent's chooseAction returns a Promise that resolves when the user
 * clicks "摸牌" or "跑路" (or presses D / F). chooseSkillTarget pops a modal
 * with active candidates.
 */

import { CardKind } from './cards.js';
import { PlayerStatus } from './state.js';

const STATUS_CN = {
  [PlayerStatus.ACTIVE]: '进行中',
  [PlayerStatus.FOLDED]: '已跑路',
  [PlayerStatus.BUSTED]: '已爆牌',
  [PlayerStatus.EXILED]: '被放逐',
};

const STATUS_EMOJI = {
  [PlayerStatus.ACTIVE]: '🎲',
  [PlayerStatus.FOLDED]: '🔒',
  [PlayerStatus.BUSTED]: '💥',
  [PlayerStatus.EXILED]: '🚷',
};

const KIND_CN = {
  [CardKind.EXILE]: '放逐',
  [CardKind.TRIPLE]: '三连',
  [CardKind.INSURANCE]: '保险',
};

let _agentNames = ['', '', '', ''];
let _myIdx = 0;

export function setupUI(agentNames, myIdx = 0) {
  _agentNames = agentNames;
  _myIdx = myIdx;
}

// =========================================================== HumanAgent
export class HumanAgent {
  constructor() { this.name = 'human'; }

  chooseAction(state, myIdx) {
    _myIdx = myIdx;
    render(state, myIdx);
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

  chooseSkillTarget(state, myIdx, kind) {
    return new Promise(resolve => {
      showSkillModal(state, myIdx, kind, (target) => {
        hideSkillModal();
        resolve(target);
      });
    });
  }
}

// =========================================================== render
export function render(state, myIdx) {
  document.getElementById('round-num').textContent = state.roundNumber;
  document.getElementById('deck-remaining').textContent = state.remaining.total();
  document.getElementById('target-display').textContent = state.config.targetScore;

  const grid = document.getElementById('players');
  grid.innerHTML = '';
  state.players.forEach(p => grid.appendChild(renderPlayerCard(state, p, myIdx)));

  const me = state.players[myIdx];
  renderMyHand(me);
  document.getElementById('my-cur-score').textContent = me.currentRoundScore();
  document.getElementById('my-insurance').textContent = me.hasInsurance ? '✓ 有' : '无';
  renderHint(state, me);
  renderLog(state, myIdx);
}

function renderPlayerCard(state, p, myIdx) {
  const isMe = p.index === myIdx;
  const card = document.createElement('div');
  card.className = `player-card status-${p.status} ${isMe ? 'is-me' : ''}`;

  const displayName = isMe ? '★ 你' : `P${p.index} · ${_agentNames[p.index] || 'AI'}`;
  const target = state.config.targetScore;
  const pct = Math.min(100, p.totalScore / target * 100);
  const handStr = p.handNumbers.length
    ? p.handNumbers.slice().sort((a, b) => a - b).join(', ')
    : '-';
  const ins = p.hasInsurance ? ' 🛡' : '';
  const cur = p.currentRoundScore();

  card.innerHTML = `
    <div class="name">
      <span>${displayName}</span>
      <span class="status-emoji" title="${STATUS_CN[p.status]}">${STATUS_EMOJI[p.status]}</span>
    </div>
    <div class="total">${p.totalScore}</div>
    <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
    <div class="info">本局 +${cur}${ins}</div>
    <div class="hand">[${handStr}]</div>
  `;
  return card;
}

function renderMyHand(me) {
  const handDiv = document.getElementById('my-hand');
  handDiv.innerHTML = '';
  const sorted = me.handNumbers.slice().sort((a, b) => a - b);
  sorted.forEach(n => {
    const tile = document.createElement('div');
    tile.className = 'card-tile';
    tile.textContent = n;
    handDiv.appendChild(tile);
  });
  if (me.bonusFlatTotal > 0) {
    const tile = document.createElement('div');
    tile.className = 'card-tile bonus';
    tile.textContent = `+${me.bonusFlatTotal}`;
    handDiv.appendChild(tile);
  }
  if (sorted.length === 0 && me.bonusFlatTotal === 0) {
    handDiv.innerHTML = '<span style="color:var(--text-dim);align-self:center">手牌空</span>';
  }
}

function renderHint(state, me) {
  const hintDiv = document.getElementById('hint');
  const rem = state.remaining;
  const total = Math.max(rem.total(), 1);
  const handSet = me.uniqueNumbers;
  let bustCount = 0;
  for (const v of handSet) bustCount += rem.numbers[v];
  const bustP = bustCount / total * 100;
  const cur = me.handNumbers.reduce((s, v) => s + v, 0) + me.bonusFlatTotal;
  const target = state.config.targetScore;

  let html = '';
  if (me.totalScore + cur >= target) {
    html += '<span class="winning">🏆 现在跑路即可获胜！</span>';
  } else {
    const cls = bustP < 15 ? 'safe' : bustP < 30 ? 'warn' : 'danger';
    html += `爆牌概率 <span class="${cls}">${bustP.toFixed(0)}%</span>`;

    if (handSet.size === 5) {
      let safeCount = 0;
      for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeCount += rem.numbers[v];
      const sixP = safeCount / total * 100;
      html += ` · <span class="warn">🚀 6 翻成功率 ${sixP.toFixed(0)}%</span>`;
    }
    html += ` · 跑路锁 <b>${cur}</b> 分`;
  }
  hintDiv.innerHTML = html;
}

function renderLog(state, myIdx) {
  const ul = document.getElementById('event-log');
  ul.innerHTML = '';
  const curPrefix = `R${state.roundNumber}|`;
  const prevPrefix = `R${state.roundNumber - 1}|`;

  // include any tail events from previous round (e.g. forced triple draws after my turn)
  const tail = state.log.filter(l => l.startsWith(prevPrefix));
  // only show last few of previous round
  const tailToShow = tail.slice(-4);
  tailToShow.forEach(line => ul.appendChild(makeLogLi(line, myIdx, true)));
  if (tailToShow.length > 0) {
    const sep = document.createElement('li');
    sep.style.color = 'var(--text-dim)';
    sep.style.borderTop = '1px solid var(--border)';
    sep.style.marginTop = '4px';
    sep.style.paddingTop = '4px';
    sep.textContent = `── 第 ${state.roundNumber} 局 ──`;
    ul.appendChild(sep);
  }

  state.log
    .filter(l => l.startsWith(curPrefix))
    .forEach(line => ul.appendChild(makeLogLi(line, myIdx, false)));

  ul.scrollTop = ul.scrollHeight;
}

function makeLogLi(line, myIdx, fromPrev) {
  const li = document.createElement('li');
  let cn = translateLog(line);
  if (fromPrev) cn = '↳ ' + cn;
  li.textContent = cn;

  if (line.includes('BUST') && !line.includes('AVOIDED')) li.classList.add('bust');
  else if (line.includes('SIX-BURST')) li.classList.add('six-burst');
  else if (line.includes('EXILE') || line.includes('TRIPLE')) li.classList.add('skill');
  else if (line.includes('INSURANCE')) li.classList.add('insurance-event');
  else if (line.includes('BONUS') || line.includes('DOUBLE')) li.classList.add('bonus');
  else if (line.includes('FOLD')) li.classList.add('fold');

  const m = line.match(/R\d+\|P(\d+):/);
  if (m && parseInt(m[1]) === myIdx) li.classList.add('is-me');
  return li;
}

function translateLog(line) {
  let cn = line;
  cn = cn.replace(/SIX-BURST! lock=(\d+)/, '🚀 6翻了！锁分 $1');
  cn = cn.replace(/FOLD lock=(\d+)/, '跑路 锁 $1 分');
  cn = cn.replace(/DRAW (\d+)/, '抽到 $1');
  cn = cn.replace(/BUST_AVOIDED on (\d+)/, '重复 $1（保险消耗）');
  cn = cn.replace(/BUST on (\d+)/, '💥 爆牌！重复 $1');
  cn = cn.replace(/BONUS\+(\d+)/, '加分牌 +$1');
  cn = cn.replace(/DOUBLE \+(\d+)/, '翻倍牌 +$1');
  cn = cn.replace(/INSURANCE\+/, '获得保险');
  cn = cn.replace(/INSURANCE -> P(\d+)/, '送保险给 P$1');
  cn = cn.replace(/INSURANCE wasted.*/, '保险作废');
  cn = cn.replace(/EXILE -> P(\d+) lock=(\d+)/, '放逐 P$1（锁 $2 分）');
  cn = cn.replace(/EXILE wasted.*/, '放逐作废');
  cn = cn.replace(/TRIPLE -> P(\d+)/, '三连 → P$1（连摸 3 张）');
  cn = cn.replace(/TRIPLE wasted.*/, '三连作废');
  return cn;
}

// =========================================================== drawn-card popup
let _lastLogLen = 0;
export function flashDrawnCard(state, myIdx) {
  // Detect new draws since last call and pop the most recent one for `myIdx`
  const newLogs = state.log.slice(_lastLogLen);
  _lastLogLen = state.log.length;
  const myPrefix = `|P${myIdx}:`;
  const recent = [...newLogs].reverse().find(l => l.includes(myPrefix));
  if (!recent) return;
  let label = '', cls = '';
  let m;
  if ((m = recent.match(/DRAW (\d+)/))) { label = m[1]; cls = ''; }
  else if (/BONUS\+/.test(recent)) { label = '+10'; cls = 'bonus'; }
  else if (/DOUBLE/.test(recent)) { label = 'x2'; cls = 'bonus'; }
  else if (/INSURANCE/.test(recent)) { label = '🛡'; cls = 'skill'; }
  else if (/EXILE/.test(recent)) { label = '🚷'; cls = 'skill'; }
  else if (/TRIPLE/.test(recent)) { label = '⚡', cls = 'skill'; }
  else if (/BUST/.test(recent) && !/AVOIDED/.test(recent)) { label = '💥', cls = 'bust'; }
  else return;
  const el = document.getElementById('drawn-card');
  el.className = 'drawn-card ' + cls;
  el.textContent = label;
  el.hidden = false;
  setTimeout(() => { el.hidden = true; }, 750);
}

export function resetFlashTracker(len = 0) { _lastLogLen = len; }

// =========================================================== skill modal
function showSkillModal(state, myIdx, kind, onPick) {
  const modal = document.getElementById('skill-modal');
  document.getElementById('skill-title').textContent = `★ ${KIND_CN[kind] || kind}`;
  const desc = {
    [CardKind.EXILE]: '选择强制跑路的对象（可选自己 → 自我锁分）',
    [CardKind.TRIPLE]: '选择强制连摸 3 张的对象（5 张时可选自己冲 6 翻）',
    [CardKind.INSURANCE]: '你已有保险，必须把这张转送给一名其他玩家',
  }[kind] || '选择目标';
  document.getElementById('skill-desc').textContent = desc;

  const targets = document.getElementById('skill-targets');
  targets.innerHTML = '';
  const candidates = state.players.filter(p => {
    if (!p.isActive) return false;
    if (kind === CardKind.INSURANCE && p.index === myIdx) return false;
    return true;
  });
  candidates.forEach(p => {
    const btn = document.createElement('div');
    btn.className = 'skill-target' + (p.index === myIdx ? ' is-self' : '');
    const ins = p.hasInsurance ? ' 🛡' : '';
    const hand = p.handNumbers.slice().sort((a, b) => a - b).join(',');
    const youMark = p.index === myIdx ? ' ← 你' : '';
    btn.innerHTML = `
      <span><b>P${p.index}${youMark}</b></span>
      <span style="color:var(--text-dim);font-size:13px">总分 ${p.totalScore} · 本局 +${p.currentRoundScore()} · [${hand}]${ins}</span>
    `;
    btn.addEventListener('click', () => onPick(p.index));
    targets.appendChild(btn);
  });
  modal.hidden = false;
}

function hideSkillModal() {
  document.getElementById('skill-modal').hidden = true;
}

// =========================================================== result modal
export function showResult(winnerIdx, players) {
  const modal = document.getElementById('result-modal');
  const myWon = winnerIdx === _myIdx;
  document.getElementById('result-title').textContent =
    myWon ? '🎉 恭喜，你赢了！' : `🏁 ${_agentNames[winnerIdx]} 获胜`;

  const rank = document.getElementById('result-rank');
  rank.innerHTML = '';
  const sorted = players.slice().sort((a, b) => b.totalScore - a.totalScore);
  const medals = ['🥇', '🥈', '🥉', '  '];
  sorted.forEach((p, i) => {
    const li = document.createElement('li');
    if (p.index === _myIdx) li.classList.add('is-me');
    const name = p.index === _myIdx ? '你' : (_agentNames[p.index] || 'AI');
    li.innerHTML = `<span>${medals[i]} P${p.index}（${name}）</span><span><b>${p.totalScore}</b></span>`;
    rank.appendChild(li);
  });
  modal.hidden = false;
}
