/**
 * Game state structures. Mirrors game/state.py.
 */

/** @typedef {'active'|'folded'|'exiled'|'busted'} PlayerStatusValue */

export const PlayerStatus = Object.freeze({
  ACTIVE: 'active',
  FOLDED: 'folded',
  EXILED: 'exiled',
  BUSTED: 'busted',
});

export class PlayerState {
  /** @param {number} index */
  constructor(index) {
    this.index = index;
    this.totalScore = 0; // carries across rounds

    // round-local fields, reset every round
    /** @type {number[]} */
    this.handNumbers = [];
    this.bonusFlatTotal = 0;
    this.hasInsurance = false;
    this.lockedRoundScore = 0;
    /** @type {PlayerStatusValue} */
    this.status = PlayerStatus.ACTIVE;
  }

  get isActive() { return this.status === PlayerStatus.ACTIVE; }

  /** @returns {Set<number>} */
  get uniqueNumbers() { return new Set(this.handNumbers); }

  /** Live score if locked right now (for active players). */
  currentRoundScore() {
    if (this.status === PlayerStatus.BUSTED) return 0;
    if (this.status === PlayerStatus.FOLDED || this.status === PlayerStatus.EXILED) {
      return this.lockedRoundScore;
    }
    let s = this.bonusFlatTotal;
    for (const n of this.handNumbers) s += n;
    return s;
  }

  resetRound() {
    this.handNumbers = [];
    this.bonusFlatTotal = 0;
    this.hasInsurance = false;
    this.lockedRoundScore = 0;
    this.status = PlayerStatus.ACTIVE;
  }
}

export class GameConfig {
  /**
   * @param {Object} [opts]
   * @param {number} [opts.numPlayers=4]
   * @param {number} [opts.targetScore=200]
   * @param {number} [opts.sixBurstBonus=15]
   * @param {number | null} [opts.seed=null]
   */
  constructor({ numPlayers = 4, targetScore = 200, sixBurstBonus = 15, seed = null } = {}) {
    this.numPlayers = numPlayers;
    this.targetScore = targetScore;
    this.sixBurstBonus = sixBurstBonus;
    this.seed = seed;
  }
}

/**
 * Full mutable state of an in-progress match. Engine mutates this; agents
 * read it (must not mutate).
 */
export class GameState {
  /**
   * @param {GameConfig} config
   * @param {PlayerState[]} players
   * @param {import('./cards.js').DeckCounts} remaining
   */
  constructor(config, players, remaining) {
    this.config = config;
    /** @type {PlayerState[]} */
    this.players = players;
    /** multiset of cards still in the deck */
    this.remaining = remaining;
    this.currentPlayer = 0;
    this.starter = 0; // who starts the current round
    /** @type {number | null} */
    this.lastActor = null; // most recent active actor (for next round's starter)
    this.roundNumber = 0;
    this.roundOver = false;
    this.gameOver = false;
    /** @type {number | null} */
    this.winner = null;
    /** @type {string[]} */
    this.log = [];
  }

  get n() { return this.config.numPlayers; }

  /** @returns {PlayerState[]} */
  activePlayers() { return this.players.filter(p => p.isActive); }

  /**
   * @param {number} idx
   * @returns {PlayerState[]}
   */
  othersActive(idx) {
    return this.players.filter(p => p.isActive && p.index !== idx);
  }

  /** @param {string} msg */
  addLog(msg) {
    this.log.push(`R${this.roundNumber}|P${this.currentPlayer}: ${msg}`);
  }
}
