import { estimate } from './winrate.js';

document.getElementById('run-btn').addEventListener('click', runEstimate);

async function runEstimate() {
  const btn = document.getElementById('run-btn');
  const progressEl = document.getElementById('progress');
  const progressBar = document.getElementById('progress-bar');
  const resultEl = document.getElementById('result');

  const targetScore = parseInt(document.getElementById('target').value) || 200;
  const nSims = parseInt(document.getElementById('n-sims').value) || 200;

  const names = [];
  const scores = [];
  const agents = [];
  for (let i = 0; i < 4; i++) {
    names.push(document.getElementById(`name-${i}`).value || `P${i}`);
    scores.push(parseInt(document.getElementById(`score-${i}`).value) || 0);
    agents.push(document.getElementById(`agent-${i}`).value);
  }

  // 校验
  if (scores.every(s => s >= targetScore)) {
    alert('所有人都已达到目标分？');
    return;
  }

  btn.disabled = true;
  btn.textContent = '模拟中…';
  progressEl.hidden = false;
  progressBar.style.width = '0%';
  resultEl.hidden = true;

  try {
    const result = await estimate(scores, agents, targetScore, nSims, (done, total) => {
      progressBar.style.width = `${done / total * 100}%`;
    });
    renderResult(result, names, scores, targetScore);
  } catch (e) {
    alert('模拟失败：' + (e?.message || e));
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = '开始模拟';
    progressEl.hidden = true;
  }
}

function renderResult(result, names, scores, targetScore) {
  const resultEl = document.getElementById('result');
  // 找谁概率第一名最高
  let topRank1 = 0;
  for (let i = 0; i < 4; i++) {
    if (result.rankProb[i][0] > result.rankProb[topRank1][0]) topRank1 = i;
  }

  let html = `
    <div style="margin: 16px 0 8px; font-size: 13px; color: var(--text-dim);">
      ${result.nSims} 局模拟，${(result.elapsedMs / 1000).toFixed(1)}s ·
      最有可能夺冠：<b style="color: var(--green)">${names[topRank1]}</b>
      （${(result.rankProb[topRank1][0] * 100).toFixed(1)}%）
    </div>
    <table class="result-table">
      <thead>
        <tr>
          <th>玩家</th>
          <th>当前分</th>
          <th>1名</th>
          <th>2名</th>
          <th>3名</th>
          <th>4名</th>
          <th>E[终分]</th>
        </tr>
      </thead>
      <tbody>
  `;
  for (let i = 0; i < 4; i++) {
    const me = i === 0 ? ' class="me"' : '';
    const star = i === 0 ? '★ ' : '';
    const [p1, p2, p3, p4] = result.rankProb[i].map(x => x * 100);
    const ef = result.expectedFinal[i];
    const bar = (pct, cls = '') => `<span class="pct-bar ${cls}" style="width:${pct * 0.6}px"></span>`;
    html += `
      <tr${me}>
        <td>${star}${names[i]}</td>
        <td>${scores[i]}</td>
        <td>${p1.toFixed(1)}% ${bar(p1, 'pct-1st')}</td>
        <td>${p2.toFixed(1)}% ${bar(p2)}</td>
        <td>${p3.toFixed(1)}% ${bar(p3)}</td>
        <td>${p4.toFixed(1)}% ${bar(p4, 'pct-4th')}</td>
        <td>${ef.toFixed(0)}</td>
      </tr>
    `;
  }
  html += '</tbody></table>';
  resultEl.innerHTML = html;
  resultEl.hidden = false;
}
