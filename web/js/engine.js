/**
 * Game engine. Drives turns, resolves cards, settles rounds.
 *
 * Mirrors game/engine.py — but agent methods may return Promises (human UI),
 * so all agent invocations are awaited.
 */

import {
  BONUS_FLAT_AMOUNT, Card, CardKind, Deck, DeckCounts, RNG,
} from './cards.js';
import {
  GameConfig, GameState, PlayerState, PlayerStatus,
} from './state.js';

/**
 * @typedef {Object} Agent
 * @property {(state: GameState, myIdx: number) => ('draw'|'fold') | Promise<'draw'|'fold'>} chooseAction
 * @property {(state: GameState, myIdx: number, kind: string) => number | Promise<number>} chooseSkillTarget
 */

export class GameEngine {
  /**
   * @param {GameConfig} config
   * @param {Agent[]} agents
   */
  constructor(config, agents) {
    if (agents.length !== config.numPlayers) {
      throw new Error(`agents length (${agents.length}) != numPlayers (${config.numPlayers})`);
    }
    this.config = config;
    /** @type {Agent[]} */
    this.agents = agents.slice();
    this.rng = new RNG(config.seed ?? null);
    const players = [];
    for (let i = 0; i < config.numPlayers; i++) players.push(new PlayerState(i));
    this.state = new GameState(
      config,
      players,
      Deck.shuffled(this.rng).counts(), // filled in per-round
    );
    /** @type {Deck} */
    this._deck = new Deck(); // reshuffled at the start of each round
  }

  // --------------------------------------------------------------- public api
  /**
   * Run rounds until someone hits target_score after a round closes.
   * @returns {Promise<number>} winner index
   */
  async playMatch() {
    while (!this.state.gameOver) {
      await this.playRound();
    }
    if (this.state.winner === null) throw new Error('match ended without a winner');
    return this.state.winner;
  }

  async playRound() {
    this._beginRound();
    while (!this.state.roundOver) {
      await this._takeTurn(this.state.currentPlayer);
      if (this.state.roundOver) break;
      this._advanceTurn();
    }
    this._settleRound();
  }

  // ----------------------------------------------------------------- rounds
  _beginRound() {
    const st = this.state;
    st.roundNumber += 1;
    st.roundOver = false;
    for (const p of st.players) p.resetRound();
    this._deck = Deck.shuffled(this.rng);
    st.remaining = this._deck.counts();
    st.currentPlayer = st.starter;
    st.lastActor = null;
  }

  _settleRound() {
    const st = this.state;
    for (const p of st.players) {
      if (p.status === PlayerStatus.ACTIVE) {
        // round ended by force (six-burst); active means not yet locked
        let s = p.bonusFlatTotal;
        for (const v of p.handNumbers) s += v;
        p.lockedRoundScore = s;
      }
      p.totalScore += p.lockedRoundScore;
    }
    // next round's starter is the last actor
    if (st.lastActor !== null) {
      st.starter = st.lastActor;
    }
    // game-over check
    let maxScore = -Infinity;
    for (const p of st.players) if (p.totalScore > maxScore) maxScore = p.totalScore;
    if (maxScore >= this.config.targetScore) {
      st.gameOver = true;
      // tiebreak: highest score; if tied, lower index wins
      let best = st.players[0];
      for (const p of st.players) {
        if (p.totalScore > best.totalScore
            || (p.totalScore === best.totalScore && p.index < best.index)) {
          best = p;
        }
      }
      st.winner = best.index;
    }
  }

  // --------------------------------------------------------------- turn flow
  _advanceTurn() {
    const st = this.state;
    const n = this.config.numPlayers;
    let nxt = (st.currentPlayer + 1) % n;
    // skip non-active players
    for (let i = 0; i < n; i++) {
      if (st.players[nxt].isActive) {
        st.currentPlayer = nxt;
        return;
      }
      nxt = (nxt + 1) % n;
    }
    // no active players left → round over
    st.roundOver = true;
  }

  /** @param {number} idx */
  async _takeTurn(idx) {
    const st = this.state;
    const player = st.players[idx];
    if (!player.isActive) return;
    const action = await this.agents[idx].chooseAction(st, idx);
    st.lastActor = idx;
    if (action === 'fold') {
      let s = player.bonusFlatTotal;
      for (const v of player.handNumbers) s += v;
      player.lockedRoundScore = s;
      player.status = PlayerStatus.FOLDED;
      st.addLog(`FOLD lock=${player.lockedRoundScore}`);
    } else if (action === 'draw') {
      await this._drawFor(idx);
    } else {
      throw new Error(`unknown action: ${action}`);
    }
    // round may end if six-burst or no actives remain
    let anyActive = false;
    for (const p of st.players) if (p.isActive) { anyActive = true; break; }
    if (!anyActive) st.roundOver = true;
  }

  // ----------------------------------------------------------- card handling
  /**
   * Draw and resolve one card for player idx. `force` skips active-check
   * (used by triple-draw chain).
   * @param {number} idx
   * @param {boolean} [force=false]
   */
  async _drawFor(idx, force = false) {
    const st = this.state;
    const player = st.players[idx];
    if (!force && !player.isActive) return;
    if (this._deck.length === 0) {
      // extreme edge: deck empty mid-round; treat as forced fold
      let s = player.bonusFlatTotal;
      for (const v of player.handNumbers) s += v;
      player.lockedRoundScore = s;
      player.status = PlayerStatus.FOLDED;
      return;
    }
    const card = this._deck.draw();
    st.remaining.remove(card);
    await this._resolve(idx, card);
  }

  /**
   * @param {number} idx
   * @param {Card} card
   */
  async _resolve(idx, card) {
    const st = this.state;
    const player = st.players[idx];
    const kind = card.kind;
    if (kind === CardKind.NUMBER) {
      this._resolveNumber(idx, card.value);
    } else if (kind === CardKind.BONUS_FLAT) {
      player.bonusFlatTotal += BONUS_FLAT_AMOUNT;
      st.addLog(`BONUS+${BONUS_FLAT_AMOUNT}`);
    } else if (kind === CardKind.BONUS_DOUBLE) {
      // double the current numeric hand sum (additive: x → 2x)
      let curSum = 0;
      for (const v of player.handNumbers) curSum += v;
      player.bonusFlatTotal += curSum; // adds another copy
      st.addLog(`DOUBLE +${curSum}`);
    } else if (kind === CardKind.INSURANCE) {
      await this._resolveInsurance(idx);
    } else if (kind === CardKind.EXILE) {
      await this._resolveExile(idx);
    } else if (kind === CardKind.TRIPLE) {
      await this._resolveTriple(idx);
    }
  }

  /**
   * @param {number} idx
   * @param {number} value
   */
  _resolveNumber(idx, value) {
    const st = this.state;
    const player = st.players[idx];
    const unique = player.uniqueNumbers;
    if (unique.has(value)) {
      // bust check
      if (player.hasInsurance) {
        player.hasInsurance = false;
        st.addLog(`BUST_AVOIDED on ${value}`);
        return;
      }
      player.handNumbers = [];
      player.bonusFlatTotal = 0;
      player.lockedRoundScore = 0;
      player.status = PlayerStatus.BUSTED;
      st.addLog(`BUST on ${value}`);
      return;
    }
    player.handNumbers.push(value);
    st.addLog(`DRAW ${value}`);
    if (player.uniqueNumbers.size >= 6) {
      this._triggerSixBurst(idx);
    }
  }

  /** @param {number} idx */
  _triggerSixBurst(idx) {
    const st = this.state;
    const player = st.players[idx];
    const bonus = this.config.sixBurstBonus;
    let s = player.bonusFlatTotal;
    for (const v of player.handNumbers) s += v;
    player.lockedRoundScore = s + bonus;
    player.status = PlayerStatus.FOLDED; // treat as locked
    st.addLog(`SIX-BURST! lock=${player.lockedRoundScore}`);
    // everyone else still active gets force-locked at current score
    for (const other of st.players) {
      if (other.index !== idx && other.status === PlayerStatus.ACTIVE) {
        let s2 = other.bonusFlatTotal;
        for (const v of other.handNumbers) s2 += v;
        other.lockedRoundScore = s2;
        other.status = PlayerStatus.FOLDED;
      }
    }
    st.roundOver = true;
    st.lastActor = idx;
  }

  /** @param {number} idx */
  async _resolveInsurance(idx) {
    const st = this.state;
    const player = st.players[idx];
    if (!player.hasInsurance) {
      player.hasInsurance = true;
      st.addLog('INSURANCE+');
      return;
    }
    // forced gift to another active player
    let candidates = st.othersActive(idx).filter(p => !p.hasInsurance).map(p => p.index);
    if (candidates.length === 0) {
      candidates = st.othersActive(idx).map(p => p.index);
    }
    if (candidates.length === 0) {
      st.addLog('INSURANCE wasted (no targets)');
      return;
    }
    let target = await this.agents[idx].chooseSkillTarget(st, idx, CardKind.INSURANCE);
    if (!candidates.includes(target)) target = candidates[0];
    st.players[target].hasInsurance = true;
    st.addLog(`INSURANCE -> P${target}`);
  }

  /** @param {number} idx */
  async _resolveExile(idx) {
    const st = this.state;
    // spec says "any player" — self-exile is allowed (acts like a forced fold)
    const candidates = st.players.filter(p => p.isActive).map(p => p.index);
    if (candidates.length === 0) {
      st.addLog('EXILE wasted (no targets)');
      return;
    }
    let target = await this.agents[idx].chooseSkillTarget(st, idx, CardKind.EXILE);
    if (!candidates.includes(target)) target = candidates[0];
    const victim = st.players[target];
    let s = victim.bonusFlatTotal;
    for (const v of victim.handNumbers) s += v;
    victim.lockedRoundScore = s;
    victim.status = PlayerStatus.EXILED;
    st.addLog(`EXILE -> P${target} lock=${victim.lockedRoundScore}`);
  }

  /** @param {number} idx */
  async _resolveTriple(idx) {
    const st = this.state;
    // spec says "any player" — self-targeting is allowed (e.g. force a six-burst attempt)
    const candidates = st.players.filter(p => p.isActive).map(p => p.index);
    if (candidates.length === 0) {
      st.addLog('TRIPLE wasted (no targets)');
      return;
    }
    let target = await this.agents[idx].chooseSkillTarget(st, idx, CardKind.TRIPLE);
    if (!candidates.includes(target)) target = candidates[0];
    st.addLog(`TRIPLE -> P${target}`);
    for (let i = 0; i < 3; i++) {
      if (!st.players[target].isActive) break;
      await this._drawFor(target, true);
      if (st.roundOver) break;
    }
  }
}
