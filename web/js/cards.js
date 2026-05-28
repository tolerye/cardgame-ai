/**
 * Card / Deck definitions. 94-card deck.
 *
 * Mirrors game/cards.py line-for-line.
 */

/** @typedef {'number'|'bonus_flat'|'bonus_double'|'insurance'|'exile'|'triple'} CardKindValue */

export const CardKind = Object.freeze({
  NUMBER: 'number',
  BONUS_FLAT: 'bonus_flat',       // +10 to round score
  BONUS_DOUBLE: 'bonus_double',   // double sum of current numeric hand
  INSURANCE: 'insurance',
  EXILE: 'exile',
  TRIPLE: 'triple',
});

export class Card {
  /**
   * @param {CardKindValue} kind
   * @param {number} [value]  only meaningful for NUMBER
   */
  constructor(kind, value = 0) {
    this.kind = kind;
    this.value = value;
    Object.freeze(this);
  }

  toString() {
    if (this.kind === CardKind.NUMBER) return `N${this.value}`;
    return this.kind;
  }
}

export const BONUS_FLAT_VALUES = [2, 4, 6, 8, 10];  // 加分牌：5 张不同面值，各 1 张
export const BONUS_FLAT_COUNT = BONUS_FLAT_VALUES.length;  // 5
export const BONUS_FLAT_AVG = BONUS_FLAT_VALUES.reduce((s, v) => s + v, 0) / BONUS_FLAT_VALUES.length;  // 6.0

// Counts per the spec
// {0:1, 1:1, 2:2, 3:3, ..., 12:12} = 79
export const NUMBER_COUNTS = (() => {
  const m = { 0: 1, 1: 1 };
  for (let n = 2; n <= 12; n++) m[n] = n;
  return Object.freeze(m);
})();
export const BONUS_DOUBLE_COUNT = 3;
export const SKILL_PER_KIND = 3; // insurance / exile / triple each

/** @returns {Card[]} */
export function buildFullDeck() {
  /** @type {Card[]} */
  const deck = [];
  for (const [n, count] of Object.entries(NUMBER_COUNTS)) {
    const v = Number(n);
    for (let i = 0; i < count; i++) deck.push(new Card(CardKind.NUMBER, v));
  }
  for (const v of BONUS_FLAT_VALUES) deck.push(new Card(CardKind.BONUS_FLAT, v));
  for (let i = 0; i < BONUS_DOUBLE_COUNT; i++) deck.push(new Card(CardKind.BONUS_DOUBLE));
  for (const kind of [CardKind.INSURANCE, CardKind.EXILE, CardKind.TRIPLE]) {
    for (let i = 0; i < SKILL_PER_KIND; i++) deck.push(new Card(kind));
  }
  if (deck.length !== 96) {
    throw new Error(`expected 96 cards, got ${deck.length}`);
  }
  return deck;
}

/**
 * Mulberry32 deterministic PRNG (used when caller provides a seed).
 * @param {number} seed
 */
function makeMulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Tiny RNG wrapper, mimics Python random.Random API surface used here.
 */
export class RNG {
  /** @param {number | null | undefined} [seed] */
  constructor(seed) {
    this._fn = (seed === undefined || seed === null)
      ? Math.random
      : makeMulberry32(seed);
  }
  /** @returns {number} */
  random() { return this._fn(); }
  /**
   * Fisher-Yates in place.
   * @template T
   * @param {T[]} arr
   */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this._fn() * (i + 1));
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
  }
  /**
   * @template T
   * @param {T[]} arr
   * @returns {T}
   */
  choice(arr) {
    return arr[Math.floor(this._fn() * arr.length)];
  }
  /**
   * inclusive both ends.
   * @param {number} a
   * @param {number} b
   */
  randint(a, b) {
    return a + Math.floor(this._fn() * (b - a + 1));
  }
}

/**
 * Stack of remaining cards. Maintains both an order (for actual draws) and
 * a multiset view for agents to compute exact probabilities.
 */
export class Deck {
  /** @param {Card[]} [cards] */
  constructor(cards = []) {
    /** @type {Card[]} */
    this.cards = cards;
  }

  /**
   * @param {RNG} [rng]
   * @returns {Deck}
   */
  static shuffled(rng) {
    const r = rng ?? new RNG();
    const cards = buildFullDeck();
    r.shuffle(cards);
    return new Deck(cards);
  }

  /** @returns {Card} */
  draw() {
    const c = this.cards.pop();
    if (c === undefined) throw new Error('Deck.draw() on empty deck');
    return c;
  }

  get length() { return this.cards.length; }

  // --- multiset view for agents ------------------------------------------------
  /** @returns {DeckCounts} */
  counts() {
    const c = new DeckCounts();
    for (const card of this.cards) c.add(card);
    return c;
  }
}

/**
 * Remaining-card multiset; agents use this to compute exact probabilities.
 */
export class DeckCounts {
  constructor() {
    /** @type {Record<number, number>} */
    this.numbers = {};
    for (let n = 0; n < 13; n++) this.numbers[n] = 0;
    this.bonus_flat = 0;
    this.bonus_double = 0;
    this.insurance = 0;
    this.exile = 0;
    this.triple = 0;
  }

  /** @returns {DeckCounts} */
  static full() {
    const c = new DeckCounts();
    for (const [n, count] of Object.entries(NUMBER_COUNTS)) {
      c.numbers[Number(n)] = count;
    }
    c.bonus_flat = BONUS_FLAT_COUNT;
    c.bonus_double = BONUS_DOUBLE_COUNT;
    c.insurance = SKILL_PER_KIND;
    c.exile = SKILL_PER_KIND;
    c.triple = SKILL_PER_KIND;
    return c;
  }

  /** @param {Card} card */
  add(card) {
    switch (card.kind) {
      case CardKind.NUMBER: this.numbers[card.value] += 1; break;
      case CardKind.BONUS_FLAT: this.bonus_flat += 1; break;
      case CardKind.BONUS_DOUBLE: this.bonus_double += 1; break;
      case CardKind.INSURANCE: this.insurance += 1; break;
      case CardKind.EXILE: this.exile += 1; break;
      case CardKind.TRIPLE: this.triple += 1; break;
    }
  }

  /** @param {Card} card */
  remove(card) {
    switch (card.kind) {
      case CardKind.NUMBER: this.numbers[card.value] -= 1; break;
      case CardKind.BONUS_FLAT: this.bonus_flat -= 1; break;
      case CardKind.BONUS_DOUBLE: this.bonus_double -= 1; break;
      case CardKind.INSURANCE: this.insurance -= 1; break;
      case CardKind.EXILE: this.exile -= 1; break;
      case CardKind.TRIPLE: this.triple -= 1; break;
    }
  }

  /** @returns {number} */
  total() {
    let s = 0;
    for (let n = 0; n < 13; n++) s += this.numbers[n];
    return s + this.bonus_flat + this.bonus_double + this.insurance + this.exile + this.triple;
  }

  /** @returns {DeckCounts} */
  copy() {
    const c = new DeckCounts();
    c.numbers = { ...this.numbers };
    c.bonus_flat = this.bonus_flat;
    c.bonus_double = this.bonus_double;
    c.insurance = this.insurance;
    c.exile = this.exile;
    c.triple = this.triple;
    return c;
  }
}
