/**
 * 实时决策助手：用户手动驱动每张牌的抽取，状态实时更新，每一步都给推荐。
 *
 * 核心逻辑：维护 GameState（复用现有 PlayerState/DeckCounts），用户每次
 * 点一张牌就模拟一次"活跃玩家抽到这张牌"的事件 —— 数字加入手牌、
 * 加分入账、技能弹目标选择、爆牌/六翻自动判断。每次状态变化重新调
 * ExpectimaxAgent 给推荐。
 */

import {
  BONUS_FLAT_VALUES, BONUS_FLAT_AVG, Card, CardKind, DeckCounts,
} from './cards.js';
import {
  GameConfig, GameState, PlayerState, PlayerStatus,
} from './state.js';
import { ExpectimaxAgent, EVAgent } from './agents.js';

// ============================================================ 全局状态
let state;            // GameState
let activeIdx = 0;    // 当前活跃玩家
let history = [];     // [{ snapshot: state-deep-copy, log: '...' }]
let logEntries = [];  // 历史日志条目
const NUM_PLAYERS = 4;

// ============================================================ 初始化
function newState(targetScore = 200) {
  const cfg = new GameConfig({ numPlayers: NUM_PLAYERS, targetScore });
  const players = [];
  for (let i = 0; i < NUM_PLAYERS; i++) {
    const p = new PlayerState(i);
    p.status = PlayerStatus.ACTIVE;
    players.push(p);
  }
  return new GameState(cfg, players, DeckCounts.full());
}

function reset(keepTotals = false) {
  const target = parseInt(document.getElementById('target-input').value) || 200;
  const oldTotals = keepTotals ? state.players.map(p => p.totalScore) : null;
  const oldRound = keepTotals ? state.roundNumber : 0;
  state = newState(target);
  if (oldTotals) {
    state.players.forEach((p, i) => { p.totalScore = oldTotals[i]; });
    state.roundNumber = oldRound + 1;
  }
  activeIdx = 0;
  history = [];
  logEntries = [];
  render();
}

function snapshot() {
  return {
    deck: { ...state.remaining.numbers },
    deckBonus: {
      bonus_flat: state.remaining.bonus_flat,
      bonus_double: state.remaining.bonus_double,
      insurance: state.remaining.insurance,
      exile: state.remaining.exile,
      triple: state.remaining.triple,
    },
    players: state.players.map(p => ({
      hand: p.handNumbers.slice(),
      bonus: p.bonusFlatTotal,
      ins: p.hasInsurance,
      total: p.totalScore,
      lock: p.lockedRoundScore,
      status: p.status,
    })),
    activeIdx,
    roundNumber: state.roundNumber,
    logs: logEntries.slice(),
  };
}

function restore(snap) {
  state.remaining.numbers = { ...snap.deck };
  Object.assign(state.remaining, snap.deckBonus);
  snap.players.forEach((s, i) => {
    const p = state.players[i];
    p.handNumbers = s.hand.slice();
    p.bonusFlatTotal = s.bonus;
    p.hasInsurance = s.ins;
    p.totalScore = s.total;
    p.lockedRoundScore = s.lock;
    p.status = s.status;
  });
  activeIdx = snap.activeIdx;
  state.roundNumber = snap.roundNumber;
  logEntries = snap.logs.slice();
}

function pushHistory() {
  history.push(snapshot());
  if (history.length > 100) history.shift();
}

function undo() {
  if (history.length === 0) return;
  const snap = history.pop();
  restore(snap);
  render();
}

// ============================================================ 应用抽牌
function applyDraw(playerIdx, cardKind, cardValue = null) {
  const p = state.players[playerIdx];
  if (!p.isActive) {
    addLog(`⚠ P${playerIdx} 已不活跃，无法抽牌`, 'warn');
    return;
  }
  pushHistory();

  // 减牌库
  if (cardKind === CardKind.NUMBER) {
    if (state.remaining.numbers[cardValue] <= 0) {
      addLog(`⚠ 数字 ${cardValue} 已抽光（输入与牌库矛盾）`, 'warn');
      history.pop();  // 撤销 push
      return;
    }
    state.remaining.numbers[cardValue] -= 1;
    handleNumberDraw(playerIdx, cardValue);
  } else if (cardKind === CardKind.BONUS_FLAT) {
    if (state.remaining.bonus_flat <= 0) { warn('加分牌已抽光'); history.pop(); return; }
    state.remaining.bonus_flat -= 1;
    p.bonusFlatTotal += cardValue;
    addLog(`P${playerIdx}${meTag(playerIdx)} 抽到加分牌 +${cardValue}`, 'bonus', playerIdx);
  } else if (cardKind === CardKind.BONUS_DOUBLE) {
    if (state.remaining.bonus_double <= 0) { warn('翻倍牌已抽光'); history.pop(); return; }
    state.remaining.bonus_double -= 1;
    const handSum = p.handNumbers.reduce((s, v) => s + v, 0);
    p.bonusFlatTotal += handSum;
    addLog(`P${playerIdx}${meTag(playerIdx)} 抽到翻倍牌（数字总和翻倍 +${handSum}）`, 'bonus', playerIdx);
  } else if (cardKind === CardKind.INSURANCE) {
    if (state.remaining.insurance <= 0) { warn('保险牌已抽光'); history.pop(); return; }
    state.remaining.insurance -= 1;
    if (!p.hasInsurance) {
      p.hasInsurance = true;
      addLog(`P${playerIdx}${meTag(playerIdx)} 获得保险 🛡`, 'skill', playerIdx);
    } else {
      // 强制送给其他活跃玩家
      const candidates = state.players.filter(o => o.isActive && o.index !== playerIdx);
      if (candidates.length === 0) {
        addLog(`P${playerIdx} 重复保险且无可送对象，作废`, 'skill', playerIdx);
      } else {
        promptTarget(candidates, CardKind.INSURANCE, '保险已被你拿过，必须转送给：', (target) => {
          state.players[target].hasInsurance = true;
          addLog(`P${playerIdx}${meTag(playerIdx)} 强制送保险 → P${target}`, 'skill', playerIdx);
          afterAction();
        });
        return; // promptTarget 是异步的，先不调 afterAction
      }
    }
  } else if (cardKind === CardKind.EXILE) {
    if (state.remaining.exile <= 0) { warn('放逐牌已抽光'); history.pop(); return; }
    state.remaining.exile -= 1;
    const candidates = state.players.filter(o => o.isActive);
    if (candidates.length === 0) {
      addLog(`放逐作废（无目标）`, 'skill', playerIdx);
    } else {
      promptTarget(candidates, CardKind.EXILE, `${meTag(playerIdx)} 抽到放逐牌，目标是：`, (target) => {
        const victim = state.players[target];
        victim.lockedRoundScore = victim.handNumbers.reduce((s, v) => s + v, 0) + victim.bonusFlatTotal;
        victim.status = PlayerStatus.EXILED;
        addLog(`P${playerIdx}${meTag(playerIdx)} 放逐 P${target}（锁 ${victim.lockedRoundScore} 分）`, 'skill', playerIdx);
        afterAction();
      });
      return;
    }
  } else if (cardKind === CardKind.TRIPLE) {
    if (state.remaining.triple <= 0) { warn('三连牌已抽光'); history.pop(); return; }
    state.remaining.triple -= 1;
    const candidates = state.players.filter(o => o.isActive);
    if (candidates.length === 0) {
      addLog(`三连作废（无目标）`, 'skill', playerIdx);
    } else {
      promptTarget(candidates, CardKind.TRIPLE, `${meTag(playerIdx)} 抽到三连牌，目标是（连摸 3 张，请逐张点）：`, (target) => {
        addLog(`P${playerIdx}${meTag(playerIdx)} 三连 → P${target}（接下来手动给 P${target} 点 3 张）`, 'skill', playerIdx);
        // 切换 active 到 target，让用户手动点 3 张
        activeIdx = target;
        afterAction();
      });
      return;
    }
  }
  afterAction();
}

function handleNumberDraw(playerIdx, value) {
  const p = state.players[playerIdx];
  if (p.handNumbers.includes(value)) {
    // 重复 → 爆牌或保险消耗
    if (p.hasInsurance) {
      p.hasInsurance = false;
      addLog(`P${playerIdx}${meTag(playerIdx)} 抽到重复 ${value}，消耗保险免爆`, 'skill', playerIdx);
    } else {
      p.handNumbers = [];
      p.bonusFlatTotal = 0;
      p.lockedRoundScore = 0;
      p.status = PlayerStatus.BUSTED;
      addLog(`P${playerIdx}${meTag(playerIdx)} 💥 爆牌！抽到重复 ${value}，本局得分归零`, 'bust', playerIdx);
    }
    return;
  }
  p.handNumbers.push(value);
  addLog(`P${playerIdx}${meTag(playerIdx)} 抽到数字 ${value}`, '', playerIdx);
  // 6 翻
  if (new Set(p.handNumbers).size >= 6) {
    const cur = p.handNumbers.reduce((s, v) => s + v, 0) + p.bonusFlatTotal;
    p.lockedRoundScore = cur + 15;
    p.status = PlayerStatus.FOLDED;
    p.totalScore += p.lockedRoundScore;
    // 其他活跃玩家强制锁分
    state.players.forEach(o => {
      if (o.index !== playerIdx && o.status === PlayerStatus.ACTIVE) {
        o.lockedRoundScore = o.handNumbers.reduce((s, v) => s + v, 0) + o.bonusFlatTotal;
        o.totalScore += o.lockedRoundScore;
        o.status = PlayerStatus.FOLDED;
      }
    });
    addLog(`🚀 P${playerIdx}${meTag(playerIdx)} 6翻了！锁 ${p.lockedRoundScore} 分，全场强制结算`, 'six', playerIdx);
  }
}

function fold(playerIdx) {
  const p = state.players[playerIdx];
  if (!p.isActive) return;
  pushHistory();
  p.lockedRoundScore = p.handNumbers.reduce((s, v) => s + v, 0) + p.bonusFlatTotal;
  p.status = PlayerStatus.FOLDED;
  p.totalScore += p.lockedRoundScore;
  addLog(`P${playerIdx}${meTag(playerIdx)} 🔒 跑路，锁 ${p.lockedRoundScore} 分`, 'skill', playerIdx);
  afterAction();
}

function bust(playerIdx) {
  const p = state.players[playerIdx];
  if (!p.isActive) return;
  pushHistory();
  p.handNumbers = [];
  p.bonusFlatTotal = 0;
  p.lockedRoundScore = 0;
  p.status = PlayerStatus.BUSTED;
  addLog(`P${playerIdx}${meTag(playerIdx)} 💥 标记爆牌出局`, 'bust', playerIdx);
  afterAction();
}

function nextRound() {
  // 把 round-only 字段清空，保留 totalScore
  pushHistory();
  state.roundNumber += 1;
  state.players.forEach(p => {
    p.handNumbers = [];
    p.bonusFlatTotal = 0;
    p.hasInsurance = false;
    p.lockedRoundScore = 0;
    p.status = PlayerStatus.ACTIVE;
  });
  state.remaining = DeckCounts.full();
  activeIdx = 0;
  addLog(`━━━ 进入第 ${state.roundNumber} 局 ━━━`, '');
  render();
}

function afterAction() {
  // 自动切换 active：如果当前 active 已不活跃，找下一个
  if (!state.players[activeIdx].isActive) {
    for (let off = 1; off < NUM_PLAYERS; off++) {
      const nxt = (activeIdx + off) % NUM_PLAYERS;
      if (state.players[nxt].isActive) { activeIdx = nxt; break; }
    }
  }
  render();
}

// ============================================================ 推荐
function computeReco() {
  const me = state.players[activeIdx];
  if (!me.isActive) {
    return { html: `<div class="verdict">P${activeIdx} 已不活跃</div><div class="reason">点击其他玩家切换活跃</div>`,
             cls: '', emoji: '⏸' };
  }
  const target = state.config.targetScore;
  const cur = me.handNumbers.reduce((s, v) => s + v, 0) + me.bonusFlatTotal;

  // 跑路即赢
  if (me.totalScore + cur >= target) {
    return {
      html: `<div class="verdict">🏆 跑路立刻获胜</div><div class="reason">总分 ${me.totalScore} + 本局 ${cur} = ${me.totalScore + cur} ≥ 目标 ${target}</div>`,
      cls: '', emoji: '🏆',
    };
  }

  // 调 expectimax
  let decision;
  try {
    const ex = new ExpectimaxAgent(3);
    decision = ex.chooseAction(state, activeIdx);
  } catch (e) {
    console.error(e);
    decision = 'fold';
  }

  // 概率细节
  const total = Math.max(state.remaining.total(), 1);
  const handSet = me.uniqueNumbers;
  let bustC = 0; for (const v of handSet) bustC += state.remaining.numbers[v];
  const pBust = bustC / total * 100;

  let safeC = 0;
  for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeC += state.remaining.numbers[v];
  const pSafe = safeC / total * 100;
  const pSix = handSet.size === 5 ? pSafe : 0;

  // EV 估算
  const evFold = cur;
  let evDraw = 0;
  evDraw += (bustC / total) * (me.hasInsurance ? cur : 0);
  if (handSet.size < 5) {
    let safeSum = 0;
    for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeSum += v * state.remaining.numbers[v];
    const avgSafe = safeC > 0 ? safeSum / safeC : 0;
    evDraw += (safeC / total) * (cur + avgSafe);
  } else if (handSet.size === 5) {
    let safeSum = 0;
    for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeSum += v * state.remaining.numbers[v];
    const avgSafe = safeC > 0 ? safeSum / safeC : 0;
    evDraw += (safeC / total) * (cur + avgSafe + 15);
  }
  evDraw += (state.remaining.bonus_flat / total) * (cur + BONUS_FLAT_AVG);
  evDraw += (state.remaining.bonus_double / total) * (cur + me.handNumbers.reduce((s, v) => s + v, 0));
  evDraw += ((state.remaining.insurance + state.remaining.exile + state.remaining.triple) / total) * (cur + 5);

  let cls = '';
  let emoji = '🎲';
  let verdict;
  let reason;
  if (decision === 'draw') {
    verdict = '🎲 继续摸牌';
    reason = `爆牌 ${pBust.toFixed(0)}%`;
    if (handSet.size === 5) reason += ` · 6翻概率 ${pSix.toFixed(0)}% 🚀`;
    reason += ` · 跑路只能锁 ${cur} 分`;
  } else {
    verdict = '🔒 跑路（锁分）';
    cls = 'fold';
    emoji = '🔒';
    reason = `继续摸 EV=${evDraw.toFixed(1)}, 跑路 EV=${evFold.toFixed(1)}`;
    if (pBust > 35) { cls = 'danger'; emoji = '⚠'; reason = `⚠ 高风险 爆牌 ${pBust.toFixed(0)}%，` + reason; }
  }

  return {
    html: `<div class="verdict">${verdict}</div><div class="reason">${reason}</div>`,
    cls, emoji,
    evHTML: `<div class="ev">EV摸 ${evDraw.toFixed(1)}<br>EV跑 ${evFold.toFixed(1)}</div>`,
    pBust, pSix, pSafe,
  };
}

// ============================================================ 渲染
function render() {
  const target = state.config.targetScore;
  document.getElementById('round-num').textContent = state.roundNumber;
  document.getElementById('target-display').textContent = target;
  document.getElementById('deck-remaining').textContent = state.remaining.total();
  document.getElementById('active-name').textContent = activeIdx === 0 ? `P0（你）` : `P${activeIdx}`;

  // 玩家网格
  const grid = document.getElementById('players');
  grid.innerHTML = '';
  state.players.forEach(p => grid.appendChild(renderPlayer(p)));

  // 推荐
  const reco = computeReco();
  const recoEl = document.getElementById('reco');
  recoEl.className = 'reco ' + (reco.cls || '');
  recoEl.innerHTML = `
    <div style="font-size:32px">${reco.emoji}</div>
    <div>${reco.html}</div>
    <div>${reco.evHTML || ''}</div>
  `;

  // 牌按钮
  renderCardGrid();

  // 概率细节
  renderProbs(reco);

  // 历史日志
  renderLog();
}

function renderPlayer(p) {
  const el = document.createElement('div');
  el.className = `player status-${p.status}`;
  if (p.index === activeIdx) el.classList.add('active');
  if (p.index === 0) el.classList.add('is-me');

  const emoji = { active: '🎲', folded: '🔒', busted: '💥', exiled: '🚷' }[p.status] || '';
  const name = p.index === 0 ? '★ 你' : `P${p.index}`;
  const cur = p.currentRoundScore();
  const handStr = p.handNumbers.length
    ? p.handNumbers.slice().sort((a, b) => a - b).join(',')
    : '-';

  el.innerHTML = `
    <div class="p-head">
      <span class="p-name">${name}</span>
      <span class="p-emoji">${emoji}</span>
    </div>
    <div class="p-total">${p.totalScore}</div>
    <div class="p-line">本局 +${cur}${p.bonusFlatTotal ? `<span class="p-bonus">+${p.bonusFlatTotal}</span>` : ''}${p.hasInsurance ? '<span class="p-ins">🛡</span>' : ''}</div>
    <div class="p-hand">[${handStr}]</div>
  `;

  el.addEventListener('click', () => {
    if (!p.isActive) return;
    activeIdx = p.index;
    render();
  });

  return el;
}

function renderCardGrid() {
  const me = state.players[activeIdx];
  const handSet = me.isActive ? new Set(me.handNumbers) : new Set();

  // 数字 0-12
  const numGrid = document.getElementById('num-grid');
  numGrid.innerHTML = '';
  for (let v = 0; v <= 12; v++) {
    const left = state.remaining.numbers[v];
    const btn = document.createElement('button');
    btn.className = 'draw-btn';
    if (handSet.has(v)) btn.classList.add('in-hand');
    if (left <= 0) btn.classList.add('depleted');
    btn.disabled = left <= 0 || !me.isActive;
    btn.innerHTML = `${v}<span class="left">${left}</span>`;
    btn.addEventListener('click', () => applyDraw(activeIdx, CardKind.NUMBER, v));
    numGrid.appendChild(btn);
  }

  // 特殊牌
  const specials = [
    { kind: CardKind.BONUS_FLAT, label: '+2', val: 2, count: () => state.remaining.bonus_flat },
    { kind: CardKind.BONUS_FLAT, label: '+4', val: 4, count: () => state.remaining.bonus_flat },
    { kind: CardKind.BONUS_FLAT, label: '+6', val: 6, count: () => state.remaining.bonus_flat },
    { kind: CardKind.BONUS_FLAT, label: '+8', val: 8, count: () => state.remaining.bonus_flat },
    { kind: CardKind.BONUS_FLAT, label: '+10', val: 10, count: () => state.remaining.bonus_flat },
    { kind: CardKind.BONUS_DOUBLE, label: '×2', val: null, count: () => state.remaining.bonus_double },
    { kind: CardKind.INSURANCE, label: '🛡保险', val: null, count: () => state.remaining.insurance },
    { kind: CardKind.EXILE, label: '🚷放逐', val: null, count: () => state.remaining.exile },
    { kind: CardKind.TRIPLE, label: '⚡三连', val: null, count: () => state.remaining.triple },
  ];
  const sg = document.getElementById('special-grid');
  sg.innerHTML = '';
  specials.forEach(s => {
    const left = s.count();
    const btn = document.createElement('button');
    btn.className = 'draw-btn special-tile';
    if (left <= 0) btn.classList.add('depleted');
    btn.disabled = left <= 0 || !me.isActive;
    btn.innerHTML = `${s.label}<span class="left">${left}</span>`;
    btn.addEventListener('click', () => applyDraw(activeIdx, s.kind, s.val));
    sg.appendChild(btn);
  });
}

function renderProbs(reco) {
  const panel = document.getElementById('probs-panel');
  const me = state.players[activeIdx];
  if (!me.isActive || !reco.pBust) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const total = Math.max(state.remaining.total(), 1);
  const rem = state.remaining;

  const probs = document.getElementById('probs');
  const pFlat = (rem.bonus_flat / total * 100).toFixed(1);
  const pDouble = (rem.bonus_double / total * 100).toFixed(1);
  const pIns = (rem.insurance / total * 100).toFixed(1);
  const pExile = (rem.exile / total * 100).toFixed(1);
  const pTriple = (rem.triple / total * 100).toFixed(1);

  const bustCls = reco.pBust > 30 ? 'danger' : reco.pBust > 15 ? 'warn' : '';
  probs.innerHTML = `
    <div class="row ${bustCls}"><span>💥 爆牌</span><b>${reco.pBust.toFixed(1)}%</b></div>
    <div class="row"><span>${me.uniqueNumbers.size === 5 ? '🚀 6翻终结' : '✅ 安全数字'}</span><b>${(me.uniqueNumbers.size === 5 ? reco.pSix : reco.pSafe).toFixed(1)}%</b></div>
    <div class="row"><span>🟢 加分牌（平均 +6）</span><b>${pFlat}%</b></div>
    <div class="row"><span>🟢 翻倍</span><b>${pDouble}%</b></div>
    <div class="row"><span>🛡 保险</span><b>${pIns}%</b></div>
    <div class="row"><span>🚷 放逐</span><b>${pExile}%</b></div>
    <div class="row"><span>⚡ 三连</span><b>${pTriple}%</b></div>
  `;
}

function renderLog() {
  const ul = document.getElementById('event-log');
  ul.innerHTML = '';
  logEntries.slice(-30).forEach(e => {
    const div = document.createElement('div');
    div.className = 'entry ' + (e.cls || '') + (e.isMe ? ' is-me' : '');
    div.textContent = `R${state.roundNumber}  ${e.text}`;
    ul.appendChild(div);
  });
  ul.scrollTop = ul.scrollHeight;
}

// ============================================================ 工具
function meTag(idx) { return idx === 0 ? '（你）' : ''; }

function addLog(text, cls = '', playerIdx = -1) {
  logEntries.push({ text, cls, isMe: playerIdx === 0 });
}

function warn(msg) {
  addLog(`⚠ ${msg}`, 'warn');
  console.warn(msg);
}

// ============================================================ 模态选目标
function promptTarget(candidates, kind, msg, onPick) {
  // 用 EVAgent 算推荐目标
  let recommended = -1;
  try {
    recommended = new EVAgent().chooseSkillTarget(state, activeIdx, kind);
  } catch (e) { console.error(e); }

  const reasons = computeTargetReasons(candidates, kind, recommended);

  const modal = document.getElementById('skill-modal');
  document.getElementById('skill-modal-title').textContent = msg;

  const KIND_DESC = {
    [CardKind.EXILE]: '推荐选择本局已得最高的对手（夺其锁分）',
    [CardKind.TRIPLE]: '推荐选手牌多+无保险的对手（爆牌概率高）；自己 5 张时优先冲 6 翻',
    [CardKind.INSURANCE]: '保险只能送其他玩家。推荐送当前总分最低者（不喂强者）',
  };
  document.getElementById('skill-modal-desc').textContent = KIND_DESC[kind] || '';

  const targetsDiv = document.getElementById('skill-modal-targets');
  targetsDiv.innerHTML = '';
  candidates.forEach(p => {
    const isMe = p.index === activeIdx;
    const isRec = p.index === recommended;
    const ins = p.hasInsurance ? ' 🛡' : '';
    const handStr = p.handNumbers.slice().sort((a, b) => a - b).join(',');
    const cur = p.currentRoundScore();

    const btn = document.createElement('div');
    btn.className = 'skill-target' + (isMe ? ' is-self' : '');
    btn.style.cssText = 'display:flex;justify-content:space-between;align-items:center;background:var(--panel-2);border:1px solid var(--border);border-radius:6px;padding:12px 14px;margin:6px 0;cursor:pointer;transition:all .12s';
    if (isRec) {
      btn.style.borderColor = 'var(--green)';
      btn.style.background = 'rgba(63,185,80,.08)';
    }
    btn.onmouseover = () => { btn.style.transform = 'translateX(2px)'; };
    btn.onmouseout = () => { btn.style.transform = ''; };

    const recBadge = isRec
      ? '<span style="background:var(--green);color:#000;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;margin-left:8px">★ 推荐</span>'
      : '';
    const youMark = isMe ? '<span style="color:var(--green);margin-left:4px">(你)</span>' : '';

    btn.innerHTML = `
      <div>
        <div><b>P${p.index}</b>${youMark}${recBadge}</div>
        <div style="color:var(--text-dim);font-size:12px;margin-top:2px">总分 ${p.totalScore} · 本局 +${cur} · [${handStr || '-'}]${ins}</div>
        <div style="color:var(--text-dim);font-size:11px;margin-top:2px;font-style:italic">${reasons[p.index] || ''}</div>
      </div>
      <span style="color:var(--text-dim);font-size:18px">›</span>
    `;
    btn.addEventListener('click', () => {
      modal.hidden = true;
      onPick(p.index);
    });
    targetsDiv.appendChild(btn);
  });

  modal.hidden = false;
  document.getElementById('skill-cancel').onclick = () => {
    modal.hidden = true;
    // 取消 = 撤销刚才的抽牌（恢复牌库 + 状态）
    undo();
  };
}

function computeTargetReasons(candidates, kind, recommendedIdx) {
  const reasons = {};
  const total = Math.max(state.remaining.total(), 1);

  candidates.forEach(p => {
    const isMe = p.index === activeIdx;
    const cur = p.currentRoundScore();
    const handSet = p.uniqueNumbers;

    if (kind === CardKind.EXILE) {
      if (isMe) {
        reasons[p.index] = cur > 0 ? `自我锁分 ${cur}（极少用）` : '自我锁分 0（无意义）';
      } else {
        reasons[p.index] = cur > 0 ? `夺走 ${cur} 分锁分` : '本局还没得分，浪费';
      }
    } else if (kind === CardKind.TRIPLE) {
      if (isMe) {
        if (handSet.size === 5) {
          let safeC = 0;
          for (let v = 0; v <= 12; v++) if (!handSet.has(v)) safeC += state.remaining.numbers[v];
          const pSafe = (safeC / total * 100).toFixed(0);
          reasons[p.index] = `5 张冲 6翻 (单张安全率 ${pSafe}%)`;
        } else {
          reasons[p.index] = `自己只 ${handSet.size} 张，不该选自己`;
        }
      } else {
        let bustC = 0;
        for (const v of handSet) bustC += state.remaining.numbers[v];
        const p3Bust = 1 - Math.pow(1 - bustC / total, 3);
        const note = p.hasInsurance ? ' (有保险)' : '';
        reasons[p.index] = `3 抽爆牌 ${(p3Bust * 100).toFixed(0)}%${note}`;
      }
    } else if (kind === CardKind.INSURANCE) {
      reasons[p.index] = `总分 ${p.totalScore}（送越弱越好）`;
    }
  });
  return reasons;
}

// ============================================================ 快捷键
function setupKeyboard() {
  // 数字 0-9 + qwe = 10/11/12
  const KEY_TO_NUM = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'q': 10, 'w': 11, 'e': 12,
  };

  document.addEventListener('keydown', (ev) => {
    // 输入框聚焦时不拦截
    if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA') return;
    if (ev.metaKey || ev.altKey) {
      // Cmd+Z / Ctrl+Z 撤销
      if (ev.key === 'z' || ev.key === 'Z') { ev.preventDefault(); undo(); }
      return;
    }
    const k = ev.key.toLowerCase();

    if (k in KEY_TO_NUM) {
      ev.preventDefault();
      const v = KEY_TO_NUM[k];
      if (state.remaining.numbers[v] > 0 && state.players[activeIdx].isActive) {
        applyDraw(activeIdx, CardKind.NUMBER, v);
      } else {
        flashErr();
      }
      return;
    }

    if (k === 'tab') {
      // tab 已被默认行为占用，用其他键
    }

    switch (k) {
      case 'f': ev.preventDefault(); fold(activeIdx); break;
      case 'z': ev.preventDefault(); undo(); break;
      case 'b': ev.preventDefault(); bust(activeIdx); break;
      case 't': ev.preventDefault(); nextActive(); break;  // t = next player
      case 'i': ev.preventDefault();
        if (state.remaining.insurance > 0) applyDraw(activeIdx, CardKind.INSURANCE);
        break;
      case 'x': ev.preventDefault();
        if (state.remaining.exile > 0) applyDraw(activeIdx, CardKind.EXILE);
        break;
      case 'r': ev.preventDefault();
        if (state.remaining.triple > 0) applyDraw(activeIdx, CardKind.TRIPLE);
        break;
      case 'd': ev.preventDefault();
        if (state.remaining.bonus_double > 0) applyDraw(activeIdx, CardKind.BONUS_DOUBLE);
        break;
    }
  });

  // Tab 单独处理（preventDefault 避免焦点跳走）
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Tab' && ev.target.tagName !== 'INPUT') {
      ev.preventDefault();
      nextActive();
    }
  });
}

function nextActive() {
  for (let off = 1; off <= NUM_PLAYERS; off++) {
    const nxt = (activeIdx + off) % NUM_PLAYERS;
    if (state.players[nxt].isActive) {
      activeIdx = nxt;
      render();
      return;
    }
  }
}

function flashErr() {
  // 简单视觉提示
  const reco = document.getElementById('reco');
  reco.style.transition = 'none';
  reco.style.borderColor = 'var(--red)';
  setTimeout(() => { reco.style.transition = ''; reco.style.borderColor = ''; }, 200);
}

// ============================================================ OCR 起步版
function setupOCR() {
  const drop = document.getElementById('ocr-drop');
  const fileInput = document.getElementById('ocr-file');
  const status = document.getElementById('ocr-status');
  const canvas = document.getElementById('ocr-canvas');
  const result = document.getElementById('ocr-result');
  if (!drop) return;

  drop.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) processImage(e.target.files[0]);
  });
  document.addEventListener('paste', (e) => {
    const ocrPanel = document.getElementById('ocr-panel');
    if (!ocrPanel || ocrPanel.hidden) return;  // OCR 面板没展开就不拦截 paste
    const items = e.clipboardData?.items || [];
    for (const item of items) {
      if (item.type && item.type.startsWith('image/')) {
        e.preventDefault();
        processImage(item.getAsFile());
        return;
      }
    }
  });

  async function processImage(file) {
    status.textContent = '加载图片...';
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      canvas.style.display = 'block';

      const T = window.Tesseract;
      if (!T) {
        status.textContent = '❌ Tesseract.js 未加载，刷新页面重试';
        return;
      }
      status.textContent = '🔍 OCR 识别中...（首次需下载语言包约 3MB，请稍等）';
      try {
        const t0 = Date.now();
        const { data } = await T.recognize(canvas, 'eng');
        const dt = ((Date.now() - t0) / 1000).toFixed(1);
        const numbers = (data.text.match(/\d+/g) || []).map(Number);
        status.textContent = `✓ 完成（${dt}s，识别 ${numbers.length} 个数字 token）`;
        result.innerHTML = `
          <div style="color:var(--text-dim);margin-bottom:6px">识别到的数字：</div>
          <div style="background:var(--panel-2);padding:8px;border-radius:4px;font-family:ui-monospace,monospace">
            ${numbers.length ? numbers.join(', ') : '(无)'}
          </div>
          <details style="margin-top:8px">
            <summary style="color:var(--text-dim);cursor:pointer;font-size:12px">原始 OCR 文本</summary>
            <pre style="background:var(--panel-2);padding:8px;border-radius:4px;white-space:pre-wrap;font-size:11px;margin-top:6px">${data.text.trim()}</pre>
          </details>
          <div style="color:var(--text-dim);margin-top:10px;font-size:11px;line-height:1.5">
            💡 这只是 OCR 起步版本，输出原始数字。<br>
            想自动填入状态？请提供一张游戏截图给我看 UI 布局，我可以加上"手牌区/分数区"等区域标定，识别后自动填。
          </div>
        `;
      } catch (e) {
        status.textContent = '❌ ' + e.message;
        console.error(e);
      }
    };
    img.src = url;
  }
}

// ============================================================ 入口
document.addEventListener('DOMContentLoaded', () => {
  reset();
  setupKeyboard();
  setupOCR();
  document.getElementById('action-fold').addEventListener('click', () => fold(activeIdx));
  document.getElementById('action-bust').addEventListener('click', () => bust(activeIdx));
  document.getElementById('undo-btn').addEventListener('click', undo);
  document.getElementById('reset-btn').addEventListener('click', () => {
    if (confirm('确定重置所有状态（包括总分）？')) reset(false);
  });
  document.getElementById('next-round-btn').addEventListener('click', nextRound);
  const hotkeyBtn = document.getElementById('hotkey-toggle');
  if (hotkeyBtn) hotkeyBtn.addEventListener('click', () => {
    const panel = document.getElementById('hotkey-panel');
    panel.hidden = !panel.hidden;
  });
  const ocrBtn = document.getElementById('ocr-toggle');
  if (ocrBtn) ocrBtn.addEventListener('click', () => {
    const panel = document.getElementById('ocr-panel');
    panel.hidden = !panel.hidden;
  });
  document.getElementById('target-input').addEventListener('change', () => {
    state.config.targetScore = parseInt(document.getElementById('target-input').value) || 200;
    render();
  });
});
