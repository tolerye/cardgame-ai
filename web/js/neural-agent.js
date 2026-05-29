/**
 * NeuralAgent (JS 版): 加载导出的 model.json，对当前局面做 forward pass，
 * 给出 draw / fold 决策。
 *
 * 网络结构：MLP 62 → 128 → 128 → 128 → (policy 2 + value 1)
 * 参数量 ~30k，全部权重存为 JSON。
 *
 * 用法：
 *     const agent = await NeuralAgent.load('model.json');
 *     agent.chooseAction(state, idx);   // 'draw' | 'fold'
 */

import { encodeState } from './neural-encoder.js';
import { EVAgent } from './agents.js';

export class NeuralAgent {
  /**
   * @param {Object} weights 由 model.json 反序列化得到的对象
   */
  constructor(weights) {
    this.name = 'neural';
    this.w = weights;
    this._evHelper = new EVAgent();
  }

  /**
   * 异步加载 — fetch JSON 然后构造实例。
   * @param {string} url - model.json 路径，相对页面
   * @returns {Promise<NeuralAgent>}
   */
  static async load(url = 'model.json') {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`无法加载 ${url}: ${resp.status}`);
    const w = await resp.json();
    return new NeuralAgent(w);
  }

  /**
   * 前向计算：返回 [logits[0..1], value]
   * @param {Float32Array} x  长度 62
   * @returns {{policy: Float32Array, value: number}}
   */
  forward(x) {
    let z = x;
    for (const layer of this.w.trunk) {
      z = relu(linear(z, layer.W, layer.b));
    }
    const policy = linear(z, this.w.policy_head.W, this.w.policy_head.b);
    const valueRaw = linear(z, this.w.value_head.W, this.w.value_head.b);
    const value = Math.tanh(valueRaw[0]);
    return { policy, value };
  }

  chooseAction(state, myIdx) {
    const me = state.players[myIdx];
    const target = state.config.targetScore;
    const cur = me.handNumbers.reduce((s, v) => s + v, 0) + me.bonusFlatTotal;
    // 已经能赢就直接跑
    if (me.totalScore + cur >= target) return 'fold';

    const x = encodeState(state, myIdx);
    const { policy } = this.forward(x);
    // policy[0] = draw logit, policy[1] = fold logit
    return policy[0] >= policy[1] ? 'draw' : 'fold';
  }

  // 技能目标用 EV 启发式（跟 Python 端一致）
  chooseSkillTarget(state, myIdx, kind) {
    return this._evHelper.chooseSkillTarget(state, myIdx, kind);
  }
}

// =============================================== 内部矩阵运算
// W: 形状 [out, in]，b: 形状 [out]，x: 形状 [in]
function linear(x, W, b) {
  const out = new Float32Array(b.length);
  for (let i = 0; i < b.length; i++) {
    let s = b[i];
    const row = W[i];
    for (let j = 0; j < x.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function relu(x) {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}
