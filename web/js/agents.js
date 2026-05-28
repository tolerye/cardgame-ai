/**
 * Agents: RandomAgent, GreedyAgent, EVAgent, ExpectimaxAgent.
 *
 * Mirrors agents/baselines.py, agents/ev_agent.py, agents/expectimax_agent.py.
 * NN agents (mcts/neural) are intentionally omitted — won't run in the browser.
 */

import { BONUS_FLAT_AVG, CardKind, RNG } from './cards.js';

// ============================================================================
// RandomAgent
// ============================================================================
export class RandomAgent {
  /** @param {number | null} [seed] */
  constructor(seed = null) {
    this.name = 'random';
    this.rng = new RNG(seed);
  }

  chooseAction(_state, _myIdx) {
    return this.rng.choice(['draw', 'fold']);
  }

  chooseSkillTarget(state, myIdx, _kind) {
    const candidates = state.players.filter(p => p.isActive && p.index !== myIdx).map(p => p.index);
    if (candidates.length === 0) return myIdx;
    return this.rng.choice(candidates);
  }
}

// ============================================================================
// GreedyAgent — folds at fixed score threshold.
// ============================================================================
export class GreedyAgent {
  /** @param {number} [foldAt=28] */
  constructor(foldAt = 28) {
    this.name = 'greedy';
    this.foldAt = foldAt;
  }

  chooseAction(state, myIdx) {
    const me = state.players[myIdx];
    let score = me.bonusFlatTotal;
    for (const v of me.handNumbers) score += v;
    if (score >= this.foldAt) return 'fold';
    return 'draw';
  }

  chooseSkillTarget(state, myIdx, _kind) {
    // target the leader
    const candidates = state.players.filter(p => p.isActive && p.index !== myIdx);
    if (candidates.length === 0) return myIdx;
    let leader = candidates[0];
    let leaderScore = leader.totalScore + leader.currentRoundScore();
    for (const p of candidates) {
      const s = p.totalScore + p.currentRoundScore();
      if (s > leaderScore) { leader = p; leaderScore = s; }
    }
    return leader.index;
  }
}

// ============================================================================
// EVAgent helpers
// ============================================================================

// --- skill heuristic values (used as their EV contribution when drawn) -------
const INSURANCE_SELF_VALUE = 8.0;   // one extra safe draw, roughly
const EXILE_VALUE = 6.0;            // locking an opponent ≈ steal their next decision
const TRIPLE_VALUE = 5.0;

/**
 * Number of distinct-number cards left that won't bust me.
 * @param {import('./cards.js').DeckCounts} counts
 * @param {Set<number>} handSet
 */
function _safeDrawCount(counts, handSet) {
  let s = 0;
  for (let v = 0; v < 13; v++) {
    if (!handSet.has(v)) s += counts.numbers[v];
  }
  return s;
}

/**
 * @param {import('./cards.js').DeckCounts} counts
 * @param {Set<number>} handSet
 */
function _bustCount(counts, handSet) {
  let s = 0;
  for (const v of handSet) s += counts.numbers[v];
  return s;
}

// ============================================================================
// EVAgent — exact-probability single-step EV plus dynamic fold threshold,
// six-burst sprint mode, and skill-target heuristics.
// ============================================================================
export class EVAgent {
  /**
   * @param {Object} [opts]
   * @param {number} [opts.riskCurve=1.0]      >1 → more aggressive draws
   * @param {number} [opts.skillValueScale=1.0] tune skill EV contribution
   */
  constructor({ riskCurve = 1.0, skillValueScale = 1.0 } = {}) {
    this.name = 'ev';
    this.riskCurve = riskCurve;
    this.skillValueScale = skillValueScale;
  }

  // --------------------------------------------------------------- main API
  chooseAction(state, myIdx) {
    const me = state.players[myIdx];
    let curScore = me.bonusFlatTotal;
    for (const v of me.handNumbers) curScore += v;
    const target = state.config.targetScore;

    // Endgame guard: folding now ends the match in our favour.
    if (me.totalScore + curScore >= target) return 'fold';

    let evDraw = this._evDraw(state, myIdx);
    const evFold = curScore;

    // Risk modifier: scale draw EV by deficit / lead.
    evDraw *= this._riskMultiplier(state, myIdx);

    // Sprint mode: with 5 distinct numbers, six-burst is huge — bias towards draw.
    if (me.uniqueNumbers.size === 5 && this._sprintAttractive(state, me)) {
      return 'draw';
    }

    return evDraw > evFold ? 'draw' : 'fold';
  }

  // --------------------------------------------------------------- EV core
  _evDraw(state, myIdx) {
    const me = state.players[myIdx];
    const counts = state.remaining;
    const total = counts.total();
    if (total <= 0) return 0.0;

    let curScore = me.bonusFlatTotal;
    for (const v of me.handNumbers) curScore += v;
    let handSumNumeric = 0;
    for (const v of me.handNumbers) handSumNumeric += v;
    const handSet = me.uniqueNumbers;
    const hasIns = me.hasInsurance;
    const nUnique = handSet.size;
    let ev = 0.0;

    // numeric outcomes
    for (let v = 0; v < 13; v++) {
      const c = counts.numbers[v];
      if (c === 0) continue;
      const p = c / total;
      if (handSet.has(v)) {
        // bust path
        if (hasIns) {
          // insurance consumed, score unchanged, can still act later
          ev += p * curScore;
        } else {
          ev += p * 0.0;
        }
      } else {
        const newUnique = nUnique + 1;
        if (newUnique >= 6) {
          const lock = curScore + v + state.config.sixBurstBonus;
          ev += p * lock;
        } else {
          // post-draw value: assume we make a follow-up optimal choice next turn
          // Approximation: use new_score as a conservative lower bound; correction
          // comes from risk_curve. Cheap and effective.
          ev += p * (curScore + v);
        }
      }
    }

    // +10 flat bonus
    if (counts.bonus_flat > 0) {
      const p = counts.bonus_flat / total;
      ev += p * (curScore + BONUS_FLAT_AVG);
    }

    // double current numeric hand sum (adds another copy of handSumNumeric)
    if (counts.bonus_double > 0) {
      const p = counts.bonus_double / total;
      ev += p * (curScore + handSumNumeric);
    }

    // skills: numeric score unchanged, plus heuristic positional value
    const scale = this.skillValueScale;
    if (counts.insurance > 0) {
      const p = counts.insurance / total;
      const insVal = !hasIns ? INSURANCE_SELF_VALUE : 2.0;
      ev += p * (curScore + insVal * scale);
    }
    if (counts.exile > 0) {
      const p = counts.exile / total;
      ev += p * (curScore + EXILE_VALUE * scale);
    }
    if (counts.triple > 0) {
      const p = counts.triple / total;
      ev += p * (curScore + TRIPLE_VALUE * scale);
    }

    return ev;
  }

  // ---------------------------------------------------------- modifiers
  _riskMultiplier(state, myIdx) {
    const me = state.players[myIdx];
    let leader = 0;
    let any = false;
    for (const p of state.players) {
      if (p.index === myIdx) continue;
      if (!any || p.totalScore > leader) { leader = p.totalScore; any = true; }
    }
    if (!any) leader = 0;
    const deficit = leader - me.totalScore; // positive means we're behind

    const target = state.config.targetScore;
    // If we're far behind near endgame, push harder.
    const rel = deficit / Math.max(target, 1);
    let m = 1.0 + 0.6 * rel; // behind by 50% of target → +30%
    // Clamp
    m = Math.max(0.7, Math.min(1.6, m));
    return m * this.riskCurve;
  }

  _sprintAttractive(state, me) {
    const counts = state.remaining;
    const total = counts.total();
    if (total <= 0) return false;
    const handSet = me.uniqueNumbers;
    const good = _safeDrawCount(counts, handSet);
    const bad = _bustCount(counts, handSet);
    // Six-burst attempt: if P(good) > P(bad), almost always worth it
    let curScore = me.bonusFlatTotal;
    for (const v of me.handNumbers) curScore += v;
    if (me.totalScore + curScore >= state.config.targetScore) return false;
    if (me.hasInsurance) return true; // net free shot
    return good >= bad;
  }

  // ------------------------------------------------------------ skill targeting
  chooseSkillTarget(state, myIdx, kind) {
    const candidates = state.players.filter(p => p.isActive && p.index !== myIdx);
    if (candidates.length === 0) return myIdx;

    if (kind === CardKind.EXILE) {
      // pick the player whose current round score is highest (deny upside).
      let best = candidates[0];
      let bestKey = [best.currentRoundScore(), best.totalScore];
      for (const p of candidates) {
        const k = [p.currentRoundScore(), p.totalScore];
        if (k[0] > bestKey[0] || (k[0] === bestKey[0] && k[1] > bestKey[1])) {
          best = p; bestKey = k;
        }
      }
      return best.index;
    }

    if (kind === CardKind.TRIPLE) {
      // Self-triple option: with 5 distinct numbers, three forced draws have a
      // real shot at the 6th. Only worth it when the upside outweighs bust risk.
      const me = state.players[myIdx];
      if (me.uniqueNumbers.size === 5) {
        const counts = state.remaining;
        const total = counts.total();
        if (total > 0) {
          let safeN = 0;
          for (let v = 0; v < 13; v++) {
            if (!me.uniqueNumbers.has(v)) safeN += counts.numbers[v];
          }
          let badN = 0;
          for (const v of me.uniqueNumbers) badN += counts.numbers[v];
          // crude expected value: prob ≥1 of 3 picks reaches 6-burst, vs bust prob.
          const pSix = 1.0 - Math.pow(1.0 - safeN / total, 3);
          const pBust = 1.0 - Math.pow(1.0 - badN / total, 3);
          if (pSix > 0.55 && (me.hasInsurance || pBust < 0.4)) {
            return myIdx;
          }
        }
      }

      // Otherwise pick the most likely-to-bust opponent.
      const counts = state.remaining;
      const total = Math.max(counts.total(), 1);

      const bustProb3 = (p) => {
        const handSet = p.uniqueNumbers;
        const bad = _bustCount(counts, handSet);
        const pBad = bad / total;
        const pSafeOne = 1.0 - pBad;
        return 1.0 - Math.pow(pSafeOne, 3);
      };

      let best = candidates[0];
      const keyOf = (p) => [
        p.hasInsurance ? 0 : 1, // prefer no-insurance victims
        bustProb3(p),
        p.currentRoundScore(),
      ];
      let bestK = keyOf(best);
      for (const p of candidates) {
        const k = keyOf(p);
        // lexicographic compare
        if (k[0] > bestK[0]
            || (k[0] === bestK[0] && k[1] > bestK[1])
            || (k[0] === bestK[0] && k[1] === bestK[1] && k[2] > bestK[2])) {
          best = p; bestK = k;
        }
      }
      return best.index;
    }

    if (kind === CardKind.INSURANCE) {
      // forced gift — give to weakest active opponent (least threatening).
      let best = candidates[0];
      let bestKey = [best.totalScore + best.currentRoundScore(), best.index];
      for (const p of candidates) {
        const k = [p.totalScore + p.currentRoundScore(), p.index];
        if (k[0] < bestKey[0] || (k[0] === bestKey[0] && k[1] < bestKey[1])) {
          best = p; bestKey = k;
        }
      }
      return best.index;
    }

    return candidates[0].index;
  }
}

// ============================================================================
// ExpectimaxAgent — depth-limited recursive EV with optimal fold/draw choice
// at each lookahead step.
// ============================================================================

// heuristic skill bonuses applied at draw time (positional value)
const INSURANCE_GAIN_VALUE = 6.0;
const EXILE_DRAW_VALUE = 5.0;
const TRIPLE_DRAW_VALUE = 4.0;

export class ExpectimaxAgent {
  /**
   * @param {Object} [opts]
   * @param {number} [opts.depth=3]
   * @param {number} [opts.riskCurve=1.0]
   */
  constructor({ depth = 3, riskCurve = 1.0 } = {}) {
    this.name = 'expectimax';
    this.depth = depth;
    this.riskCurve = riskCurve;
    // Reused EV agent for skill targeting heuristic (Python does same).
    this._evHelper = new EVAgent();
  }

  // ------------------------------------------------------------ public API
  chooseAction(state, myIdx) {
    const me = state.players[myIdx];
    let curScore = me.bonusFlatTotal;
    for (const v of me.handNumbers) curScore += v;
    const target = state.config.targetScore;
    if (me.totalScore + curScore >= target) return 'fold';

    const handSet = new Set(me.handNumbers);
    let handSum = 0;
    for (const v of me.handNumbers) handSum += v;
    const counts = state.remaining;

    let evDraw = this._drawEv(
      handSet, handSum, me.bonusFlatTotal, me.hasInsurance,
      counts, this.depth, state.config.sixBurstBonus,
    );
    const evFold = curScore;

    evDraw *= this._riskMultiplier(state, myIdx);

    return evDraw > evFold ? 'draw' : 'fold';
  }

  chooseSkillTarget(state, myIdx, kind) {
    // Reuse the EV heuristic — these aren't bottleneck decisions.
    return this._evHelper.chooseSkillTarget(state, myIdx, kind);
  }

  // ----------------------------------------------------------- recursion
  /**
   * @param {Set<number>} handSet
   * @param {number} handSum
   * @param {number} bonus
   * @param {boolean} insurance
   * @param {import('./cards.js').DeckCounts} counts
   * @param {number} depth
   * @param {number} sixBurstBonus
   */
  _bestValue(handSet, handSum, bonus, insurance, counts, depth, sixBurstBonus) {
    const cur = handSum + bonus;
    if (depth === 0) return cur;
    const evDraw = this._drawEv(handSet, handSum, bonus, insurance, counts, depth, sixBurstBonus);
    return Math.max(cur, evDraw);
  }

  /**
   * @param {Set<number>} handSet
   * @param {number} handSum
   * @param {number} bonus
   * @param {boolean} insurance
   * @param {import('./cards.js').DeckCounts} counts
   * @param {number} depth
   * @param {number} sixBurstBonus
   */
  _drawEv(handSet, handSum, bonus, insurance, counts, depth, sixBurstBonus) {
    const total = counts.total();
    if (total <= 0) return handSum + bonus;

    let ev = 0.0;
    // number cards
    for (let v = 0; v < 13; v++) {
      const c = counts.numbers[v];
      if (c === 0) continue;
      const p = c / total;
      if (handSet.has(v)) {
        if (insurance) {
          const newCounts = counts.copy();
          newCounts.numbers[v] -= 1;
          ev += p * this._bestValue(
            handSet, handSum, bonus, false, newCounts, depth - 1, sixBurstBonus,
          );
        } else {
          ev += 0.0; // bust → 0 score
        }
      } else {
        const newCounts = counts.copy();
        newCounts.numbers[v] -= 1;
        const newSet = new Set(handSet);
        newSet.add(v);
        if (newSet.size >= 6) {
          // round ends here with the bonus
          ev += p * (handSum + v + bonus + 15);
        } else {
          ev += p * this._bestValue(
            newSet, handSum + v, bonus, insurance, newCounts, depth - 1, sixBurstBonus,
          );
        }
      }
    }

    // +10 flat bonus
    if (counts.bonus_flat > 0) {
      const p = counts.bonus_flat / total;
      const newCounts = counts.copy();
      newCounts.bonus_flat -= 1;
      ev += p * this._bestValue(
        handSet, handSum, bonus + BONUS_FLAT_AVG, insurance, newCounts, depth - 1, sixBurstBonus,
      );
    }

    // bonus double — adds another copy of hand_sum (numeric only)
    if (counts.bonus_double > 0) {
      const p = counts.bonus_double / total;
      const newCounts = counts.copy();
      newCounts.bonus_double -= 1;
      ev += p * this._bestValue(
        handSet, handSum, bonus + handSum, insurance, newCounts, depth - 1, sixBurstBonus,
      );
    }

    // insurance — gain insurance if not already, else heuristic positional value
    if (counts.insurance > 0) {
      const p = counts.insurance / total;
      const newCounts = counts.copy();
      newCounts.insurance -= 1;
      const newIns = !insurance ? true : insurance;
      const base = this._bestValue(handSet, handSum, bonus, newIns, newCounts, depth - 1, sixBurstBonus);
      const extra = !insurance ? INSURANCE_GAIN_VALUE : 0.0;
      ev += p * (base + extra);
    }

    // exile/triple — modeled as "skill drawn, score unchanged, gain positional value"
    if (counts.exile > 0) {
      const p = counts.exile / total;
      const newCounts = counts.copy();
      newCounts.exile -= 1;
      const base = this._bestValue(handSet, handSum, bonus, insurance, newCounts, depth - 1, sixBurstBonus);
      ev += p * (base + EXILE_DRAW_VALUE);
    }

    if (counts.triple > 0) {
      const p = counts.triple / total;
      const newCounts = counts.copy();
      newCounts.triple -= 1;
      const base = this._bestValue(handSet, handSum, bonus, insurance, newCounts, depth - 1, sixBurstBonus);
      ev += p * (base + TRIPLE_DRAW_VALUE);
    }

    return ev;
  }

  // ----------------------------------------------------------- modifiers
  _riskMultiplier(state, myIdx) {
    const me = state.players[myIdx];
    let leader = 0;
    let any = false;
    for (const p of state.players) {
      if (p.index === myIdx) continue;
      if (!any || p.totalScore > leader) { leader = p.totalScore; any = true; }
    }
    if (!any) leader = 0;
    const deficit = leader - me.totalScore;
    const rel = deficit / Math.max(state.config.targetScore, 1);
    let m = 1.0 + 0.5 * rel;
    m = Math.max(0.7, Math.min(1.5, m));
    return m * this.riskCurve;
  }
}
