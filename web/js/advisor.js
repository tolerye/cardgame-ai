/**
 * Web 版决策助手：用户输入当前局面 → 调 ExpectimaxAgent 给建议 + 概率分解。
 * 镜像 advisor.py 的逻辑。
 */

import {
  BONUS_FLAT_AVG, BONUS_FLAT_VALUES, DeckCounts,
} from './cards.js';
import {
  GameConfig, GameState, PlayerState, PlayerStatus,
} from './state.js';
import { ExpectimaxAgent } from './agents.js';

// ============ 状态 ============
const myHand = []; // [3, 5, 8] 手牌数字（可能重复出现，但 unique 才算手牌）
const seenNums = {}; // {3: 2, 5: 1, ...} 已见过的数字（含其他玩家手牌+弃牌堆）
const seenSpecials = { bonus_flat: 0, bonus_double: 0, insurance: 0, exile: 0, triple: 0 };

// ============ 初始化按钮 ============
function buildPickers() {
  // 我的手牌：13 个数字，点击切换是否在手中
  const myDiv = document.getElementById('my-hand-picker');
  for (let n = 0; n <= 12; n++) {
    const btn = document.createElement('div');
    btn.className = 'pick-btn';
    btn.textContent = n;
    btn.dataset.num = n;
    btn.addEventListener('click', () => toggleMyHand(n));
    myDiv.appendChild(btn);
  }

  // 已见数字：点 +1，右键 / 长按 -1
  const numDiv = document.getElementById('seen-num-picker');
  for (let n = 0; n <= 12; n++) {
    const btn = makeCounter('seen-num-' + n, n.toString(), () => addSeenNum(n), () => subSeenNum(n));
    btn.dataset.num = n;
    numDiv.appendChild(btn);
  }

  // 已见加分牌
  const bonusDiv = document.getElementById('seen-bonus-picker');
  for (const v of BONUS_FLAT_VALUES) {
    const btn = makeCounter(`seen-flat-${v}`, `+${v}`,
      () => addSeenSpecial('bonus_flat'), () => subSeenSpecial('bonus_flat'));
    bonusDiv.appendChild(btn);
  }
  bonusDiv.appendChild(makeCounter('seen-double', '×2',
    () => addSeenSpecial('bonus_double'), () => subSeenSpecial('bonus_double')));

  // 已见技能牌
  const skillDiv = document.getElementById('seen-skill-picker');
  skillDiv.appendChild(makeCounter('seen-ins', '🛡 保险',
    () => addSeenSpecial('insurance'), () => subSeenSpecial('insurance')));
  skillDiv.appendChild(makeCounter('seen-exile', '🚷 放逐',
    () => addSeenSpecial('exile'), () => subSeenSpecial('exile')));
  skillDiv.appendChild(makeCounter('seen-triple', '⚡ 三连',
    () => addSeenSpecial('triple'), () => subSeenSpecial('triple')));
}

function makeCounter(id, label, onAdd, onSub) {
  const btn = document.createElement('div');
  btn.className = 'pick-btn';
  btn.id = id;
  btn.innerHTML = `<span>${label}</span><span class="count" data-count="0"></span>`;
  btn.addEventListener('click', onAdd);
  btn.addEventListener('contextmenu', e => { e.preventDefault(); onSub(); });
  // 长按 -1（移动端）
  let timer = null;
  btn.addEventListener('touchstart', () => {
    timer = setTimeout(() => { onSub(); timer = null; }, 500);
  });
  btn.addEventListener('touchend', () => {
    if (timer) clearTimeout(timer);
  });
  return btn;
}

function toggleMyHand(n) {
  const idx = myHand.indexOf(n);
  if (idx >= 0) myHand.splice(idx, 1);
  else myHand.push(n);
  refreshMyHand();
}

function refreshMyHand() {
  document.querySelectorAll('#my-hand-picker .pick-btn').forEach(btn => {
    const n = parseInt(btn.dataset.num);
    btn.classList.toggle('active', myHand.includes(n));
  });
  const display = document.getElementById('my-hand-display');
  if (myHand.length === 0) {
    display.textContent = '点下面数字加入手牌：';
  } else {
    const sorted = myHand.slice().sort((a, b) => a - b);
    display.innerHTML = `当前手牌: <b style="color:var(--accent);font-family:ui-monospace,monospace">[${sorted.join(', ')}]</b>`;
  }
}

function addSeenNum(n) {
  seenNums[n] = (seenNums[n] || 0) + 1;
  if (seenNums[n] > (n === 0 || n === 1 ? 1 : n)) {
    seenNums[n] = (n === 0 || n === 1 ? 1 : n);  // cap to deck
  }
  refreshSeenNum(n);
}
function subSeenNum(n) {
  if (seenNums[n] > 0) seenNums[n] -= 1;
  refreshSeenNum(n);
}
function refreshSeenNum(n) {
  const btn = document.getElementById('seen-num-' + n);
  const c = seenNums[n] || 0;
  btn.classList.toggle('active', c > 0);
  btn.querySelector('.count').textContent = c > 0 ? c : '';
}

function addSeenSpecial(key) {
  seenSpecials[key] = (seenSpecials[key] || 0) + 1;
  // cap
  const max = { bonus_flat: 5, bonus_double: 3, insurance: 3, exile: 3, triple: 3 }[key];
  if (seenSpecials[key] > max) seenSpecials[key] = max;
  refreshSeenSpecial(key);
}
function subSeenSpecial(key) {
  if (seenSpecials[key] > 0) seenSpecials[key] -= 1;
  refreshSeenSpecial(key);
}
function refreshSeenSpecial(key) {
  const ids = {
    bonus_flat: BONUS_FLAT_VALUES.map(v => 'seen-flat-' + v),
    bonus_double: ['seen-double'],
    insurance: ['seen-ins'],
    exile: ['seen-exile'],
    triple: ['seen-triple'],
  };
  // 加分牌 5 张面值，count 平均显示在每个按钮上
  ids[key].forEach((id, i) => {
    const btn = document.getElementById(id);
    if (!btn) return;
    const total = seenSpecials[key] || 0;
    btn.classList.toggle('active', total > 0);
    if (key === 'bonus_flat') {
      // 不知道用户具体见过哪个面值，全部按 active 显示，count 显示总数在第一个
      btn.querySelector('.count').textContent = (i === 0 && total > 0) ? total : '';
    } else {
      btn.querySelector('.count').textContent = total > 0 ? total : '';
    }
  });
}

// ============ 构建 state ============
function buildRemaining() {
  const rem = DeckCounts.full();
  // 减自己手牌的数字
  for (const v of myHand) rem.numbers[v] -= 1;
  // 减自己的加分（按数量贪心推算）
  const myBonus = parseInt(document.getElementById('my-bonus').value) || 0;
  let needed = myBonus;
  let myBonusFlatCount = 0;
  for (const v of [10, 8, 6, 4, 2]) {
    if (needed >= v) { myBonusFlatCount += 1; needed -= v; }
  }
  rem.bonus_flat -= myBonusFlatCount;
  if (document.getElementById('my-double').checked) rem.bonus_double -= 1;
  if (document.getElementById('my-insurance').checked) rem.insurance -= 1;
  // 减已见
  for (const [v, c] of Object.entries(seenNums)) rem.numbers[v] -= c;
  rem.bonus_flat -= seenSpecials.bonus_flat;
  rem.bonus_double -= seenSpecials.bonus_double;
  rem.insurance -= seenSpecials.insurance;
  rem.exile -= seenSpecials.exile;
  rem.triple -= seenSpecials.triple;

  // 防负
  const issues = [];
  for (let v = 0; v <= 12; v++) {
    if (rem.numbers[v] < 0) {
      issues.push(`数字 ${v} 多算了 ${-rem.numbers[v]} 张`);
      rem.numbers[v] = 0;
    }
  }
  for (const k of ['bonus_flat', 'bonus_double', 'insurance', 'exile', 'triple']) {
    if (rem[k] < 0) {
      issues.push(`${k} 多算了 ${-rem[k]} 张`);
      rem[k] = 0;
    }
  }
  return { rem, issues, myBonusFlatCount };
}

// ============ 分析 ============
function analyze() {
  const { rem, issues } = buildRemaining();
  const myBonus = parseInt(document.getElementById('my-bonus').value) || 0;
  const myTotal = parseInt(document.getElementById('my-total').value) || 0;
  const target = parseInt(document.getElementById('target').value) || 200;
  const numPlayers = parseInt(document.getElementById('num-players').value) || 4;
  const hasInsurance = document.getElementById('my-insurance').checked;

  // 构 state（其他玩家粗暴当 active 总分 0；advisor 主要看自己决策）
  const cfg = new GameConfig({ numPlayers, targetScore: target });
  const me = new PlayerState(0);
  me.totalScore = myTotal;
  me.handNumbers = myHand.slice();
  me.bonusFlatTotal = myBonus;
  me.hasInsurance = hasInsurance;
  me.status = PlayerStatus.ACTIVE;
  const others = [];
  for (let i = 1; i < numPlayers; i++) {
    const p = new PlayerState(i);
    p.status = PlayerStatus.ACTIVE;
    others.push(p);
  }
  const state = new GameState(cfg, [me, ...others], rem);

  // 概率分解
  const total = Math.max(rem.total(), 1);
  const handSet = me.uniqueNumbers;
  const nUnique = handSet.size;
  const curScore = myHand.reduce((s, v) => s + v, 0) + myBonus;

  let bustCount = 0;
  for (const v of handSet) bustCount += rem.numbers[v];
  let safeCount = 0;
  let safeSum = 0;
  for (let v = 0; v <= 12; v++) {
    if (!handSet.has(v)) { safeCount += rem.numbers[v]; safeSum += v * rem.numbers[v]; }
  }
  const pBust = bustCount / total;
  const pSafe = safeCount / total;
  const pSix = nUnique === 5 ? pSafe : 0;
  const pContSafe = nUnique < 5 ? pSafe : 0;
  const avgSafeValue = safeCount > 0 ? safeSum / safeCount : 0;

  const pFlat = rem.bonus_flat / total;
  const pDouble = rem.bonus_double / total;
  const pIns = rem.insurance / total;
  const pExile = rem.exile / total;
  const pTriple = rem.triple / total;

  // EV 估算
  const evFold = curScore;
  let evDraw = 0;
  evDraw += pBust * (hasInsurance ? curScore : 0);
  evDraw += pContSafe * (curScore + avgSafeValue);
  evDraw += pSix * (curScore + avgSafeValue + 15);
  evDraw += pFlat * (curScore + BONUS_FLAT_AVG);
  evDraw += pDouble * (curScore + myHand.reduce((s, v) => s + v, 0));
  evDraw += (pIns + pExile + pTriple) * (curScore + 5);

  // 调真正的 expectimax
  const expectimax = new ExpectimaxAgent(3);
  let exDecision;
  try {
    exDecision = expectimax.chooseAction(state, 0);
  } catch (e) {
    console.error(e);
    exDecision = evDraw > evFold ? 'draw' : 'fold';
  }

  return {
    issues, totalRemaining: total, nUnique, curScore, hasInsurance,
    myTotal, target,
    pBust: pBust * 100, pSafe: pSafe * 100, pSix: pSix * 100, avgSafeValue,
    pFlat: pFlat * 100, pDouble: pDouble * 100,
    pIns: pIns * 100, pExile: pExile * 100, pTriple: pTriple * 100,
    evDraw, evFold, exDecision,
    handSum: myHand.reduce((s, v) => s + v, 0),
  };
}

// ============ 渲染结果 ============
function render(a) {
  document.getElementById('result').hidden = false;
  // 状态摘要
  const summaryDiv = document.getElementById('state-summary');
  const handStr = myHand.length ? myHand.slice().sort((x,y)=>x-y).join(', ') : '空';
  summaryDiv.innerHTML = `
    <div>剩余牌库 <b>${a.totalRemaining}</b> 张</div>
    <div>手牌 <b>[${handStr}]</b> · 加分 <b>+${a.curScore - a.handSum}</b> · 不同数字 <b>${a.nUnique}</b> 个</div>
    <div>本局已得 <b>${a.curScore}</b> · 总分 <b>${a.myTotal} / ${a.target}</b> · 保险 <b>${a.hasInsurance ? '有 🛡' : '无'}</b></div>
  `;

  // banners
  const bannerDiv = document.getElementById('banners');
  bannerDiv.innerHTML = '';
  if (a.issues.length > 0) {
    bannerDiv.innerHTML += `<div class="warning-banner">⚠ 输入与牌库矛盾：${a.issues.join('；')}</div>`;
  }
  if (a.myTotal + a.curScore >= a.target) {
    bannerDiv.innerHTML += `<div class="winning-banner">🏆 现在跑路 锁 ${a.curScore} 分即可获胜！</div>`;
  }
  if (a.pBust > 35) {
    bannerDiv.innerHTML += `<div class="warning-banner">⚠ 高风险：爆牌概率 ${a.pBust.toFixed(0)}%，强烈建议跑路</div>`;
  }

  // 概率分解
  const probDiv = document.getElementById('prob-detail');
  probDiv.innerHTML = '';
  const cls = (p, low, mid) => p < low ? 'safe' : p < mid ? 'warn' : 'danger';
  const rows = [];
  rows.push({
    emoji: '💥', label: '爆牌', pct: a.pBust, cls: cls(a.pBust, 15, 30),
    note: a.hasInsurance ? '有保险，免一次' : `本局 ${a.curScore} 分归零`,
  });
  if (a.nUnique < 5) {
    rows.push({
      emoji: '✅', label: '安全数字', pct: a.pSafe, cls: 'safe',
      note: `平均 +${a.avgSafeValue.toFixed(1)} 分`,
    });
  } else if (a.nUnique === 5) {
    rows.push({
      emoji: '🚀', label: '6 翻终结', pct: a.pSix, cls: a.pSix > 50 ? 'safe' : 'warn',
      note: '+15 奖励 + 全场强制结算',
    });
  }
  rows.push({ emoji: '🟢', label: '加分牌（平均 +6）', pct: a.pFlat, cls: 'safe', note: '+2/+4/+6/+8/+10' });
  rows.push({ emoji: '🟢', label: '翻倍', pct: a.pDouble, cls: 'safe', note: `+${a.handSum} 分` });
  rows.push({ emoji: '🛡', label: '保险', pct: a.pIns, cls: 'safe', note: '免一次爆牌' });
  rows.push({ emoji: '🚷', label: '放逐', pct: a.pExile, cls: 'safe', note: '强制目标跑路' });
  rows.push({ emoji: '⚡', label: '三连', pct: a.pTriple, cls: 'safe', note: '强制连摸 3 张' });

  rows.forEach(r => {
    const el = document.createElement('div');
    el.className = 'prob-row ' + r.cls;
    el.innerHTML = `
      <span>${r.emoji}</span>
      <span>${r.label}</span>
      <span class="pct">${r.pct.toFixed(1)}%</span>
      <span class="note">${r.note}</span>
    `;
    probDiv.appendChild(el);
  });

  // EV 对比
  const evDiv = document.getElementById('ev-compare');
  const drawWins = a.evDraw > a.evFold;
  evDiv.innerHTML = `
    <div class="ev-card ${drawWins ? 'win' : 'lose'}">
      <div class="label">EV(继续摸)</div>
      <div class="value ${drawWins ? 'win' : ''}">${a.evDraw.toFixed(1)}</div>
    </div>
    <div class="ev-card ${!drawWins ? 'win' : 'lose'}">
      <div class="label">EV(跑路)</div>
      <div class="value ${!drawWins ? 'win' : ''}">${a.evFold.toFixed(1)}</div>
    </div>
  `;

  // 推荐
  const recDiv = document.getElementById('recommend');
  if (a.exDecision === 'draw') {
    recDiv.innerHTML = `
      <div class="recommend">
        <div class="verdict">🎲 继续摸牌</div>
        <div class="reason">Expectimax depth=3 计算后建议</div>
      </div>
    `;
  } else {
    recDiv.innerHTML = `
      <div class="recommend fold">
        <div class="verdict">🔒 跑路锁分</div>
        <div class="reason">Expectimax depth=3 计算后建议（锁定 ${a.curScore} 分）</div>
      </div>
    `;
  }
}

// ============ 入口 ============
document.addEventListener('DOMContentLoaded', () => {
  buildPickers();
  refreshMyHand();
  document.getElementById('analyze-btn').addEventListener('click', () => {
    try {
      const a = analyze();
      render(a);
      document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      console.error(e);
      alert('分析失败：' + (e.message || e));
    }
  });
});
