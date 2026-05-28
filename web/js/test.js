/**
 * Sanity tests: deck composition, six-burst, bust + insurance, exile, triple.
 * Mirrors tests/test_engine.py.
 *
 * Run: cd web/js && node test.js
 */

import {
  BONUS_FLAT_AMOUNT, Card, CardKind, Deck, buildFullDeck,
} from './cards.js';
import { GameEngine } from './engine.js';
import { GameConfig, PlayerStatus } from './state.js';
import { EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent } from './agents.js';

let _passed = 0;
let _failed = 0;

function assert(cond, msg) {
  if (!cond) {
    _failed += 1;
    throw new Error(`assertion failed: ${msg}`);
  }
}
function assertEq(a, b, msg) {
  if (a !== b) {
    _failed += 1;
    throw new Error(`assertion failed: ${msg ?? ''} — got ${JSON.stringify(a)}, expected ${JSON.stringify(b)}`);
  }
}
function arrEq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ------------------------------ ScriptedAgent --------------------------------
class ScriptedAgent {
  /**
   * @param {Array<'draw'|'fold'>} actions
   * @param {number[]} [targets]
   */
  constructor(actions = [], targets = []) {
    this.actions = actions.slice();
    this.targets = targets.slice();
  }
  chooseAction(_state, _myIdx) {
    if (this.actions.length > 0) return this.actions.shift();
    return 'fold';
  }
  chooseSkillTarget(state, myIdx, _kind) {
    if (this.targets.length > 0) return this.targets.shift();
    for (const p of state.players) {
      if (p.isActive && p.index !== myIdx) return p.index;
    }
    return myIdx;
  }
}

/**
 * Build an engine whose deck top (last element) is `topCards[topCards.length-1]`
 * — same convention as Python (pop from end).
 * @param {Card[]} topCards
 * @param {number} [numPlayers=4]
 */
function engineWithStackedTop(topCards, numPlayers = 4) {
  const cfg = new GameConfig({ numPlayers, seed: 42 });
  const agents = [];
  for (let i = 0; i < numPlayers; i++) agents.push(new ScriptedAgent([]));
  const engine = new GameEngine(cfg, agents);
  // Begin a round to reset deck etc.
  engine.state.roundNumber += 1;
  for (const p of engine.state.players) p.resetRound();
  const deck = new Deck(topCards.slice());
  engine._deck = deck;
  engine.state.remaining = deck.counts();
  engine.state.currentPlayer = 0;
  engine.state.starter = 0;
  return { engine, agents };
}

// ----------------------------- deck composition ------------------------------
function testDeckSize() {
  const deck = buildFullDeck();
  assertEq(deck.length, 94, 'deck length');
  /** @type {Record<number, number>} */
  const counts = {};
  for (const c of deck) {
    if (c.kind === CardKind.NUMBER) {
      counts[c.value] = (counts[c.value] ?? 0) + 1;
    }
  }
  assertEq(counts[0], 1);
  assertEq(counts[1], 1);
  for (let n = 2; n <= 12; n++) assertEq(counts[n], n, `count[${n}]`);
  console.log('OK deck composition');
  _passed += 1;
}

async function testSixBurst() {
  let cards = [
    new Card(CardKind.NUMBER, 1),
    new Card(CardKind.NUMBER, 2),
    new Card(CardKind.NUMBER, 3),
    new Card(CardKind.NUMBER, 4),
    new Card(CardKind.NUMBER, 5),
    new Card(CardKind.NUMBER, 6),
  ];
  cards = cards.slice().reverse(); // pop() yields 1, 2, 3...
  const { engine, agents } = engineWithStackedTop(cards);
  agents[0].actions = ['draw', 'draw', 'draw', 'draw', 'draw', 'draw'];
  for (let i = 0; i < 6; i++) await engine._takeTurn(0);
  const p0 = engine.state.players[0];
  assert(engine.state.roundOver, 'six-burst should end round');
  assertEq(p0.lockedRoundScore, 1 + 2 + 3 + 4 + 5 + 6 + 15, 'six-burst lock');
  assertEq(engine.state.lastActor, 0, 'last_actor');
  console.log('OK six-burst lock + lastActor');
  _passed += 1;
}

async function testBustNoInsurance() {
  let cards = [new Card(CardKind.NUMBER, 5), new Card(CardKind.NUMBER, 5)];
  cards = cards.slice().reverse();
  const { engine, agents } = engineWithStackedTop(cards, 2);
  agents[0].actions = ['draw', 'draw'];
  await engine._takeTurn(0);
  await engine._takeTurn(0);
  const p0 = engine.state.players[0];
  assertEq(p0.status, PlayerStatus.BUSTED, 'busted');
  assertEq(p0.lockedRoundScore, 0, 'lock=0');
  console.log('OK bust without insurance');
  _passed += 1;
}

async function testBustWithInsurance() {
  let cards = [
    new Card(CardKind.NUMBER, 5),
    new Card(CardKind.INSURANCE),
    new Card(CardKind.NUMBER, 5),
  ];
  cards = cards.slice().reverse();
  const { engine, agents } = engineWithStackedTop(cards, 2);
  agents[0].actions = ['draw', 'draw', 'draw'];
  for (let i = 0; i < 3; i++) await engine._takeTurn(0);
  const p0 = engine.state.players[0];
  assertEq(p0.status, PlayerStatus.ACTIVE, 'still active'); // insurance saved us
  assert(!p0.hasInsurance, 'insurance consumed');
  assert(arrEq(p0.handNumbers, [5]), 'hand still [5]');
  console.log('OK bust with insurance consumed');
  _passed += 1;
}

async function testExile() {
  const cards = [new Card(CardKind.EXILE)];
  const { engine, agents } = engineWithStackedTop(cards, 3);
  // P1 has some round score
  engine.state.players[1].handNumbers = [3, 4];
  agents[0].actions = ['draw'];
  agents[0].targets = [1];
  await engine._takeTurn(0);
  const p1 = engine.state.players[1];
  assertEq(p1.status, PlayerStatus.EXILED, 'exiled');
  assertEq(p1.lockedRoundScore, 7, 'lock=7');
  console.log('OK exile locks target');
  _passed += 1;
}

async function testTripleChainBusts() {
  const cards = [
    new Card(CardKind.NUMBER, 7),  // 3rd forced draw → bust
    new Card(CardKind.NUMBER, 4),  // 2nd
    new Card(CardKind.NUMBER, 3),  // 1st (safe)
    new Card(CardKind.TRIPLE),     // P0 draws this
  ];
  const { engine, agents } = engineWithStackedTop(cards, 2);
  engine.state.players[1].handNumbers = [7]; // second triple-card 7 will bust
  agents[0].actions = ['draw'];
  agents[0].targets = [1];
  await engine._takeTurn(0);
  const p1 = engine.state.players[1];
  assertEq(p1.status, PlayerStatus.BUSTED, `triple-bust: got ${p1.status}`);
  console.log('OK triple chain busts target');
  _passed += 1;
}

// Smoke test: full match with mixed agents — make sure nothing crashes and a winner emerges.
async function testFullMatchSmoke() {
  const cfg = new GameConfig({ numPlayers: 4, targetScore: 200, seed: 1234 });
  const agents = [
    new ExpectimaxAgent({ depth: 2 }),
    new EVAgent(),
    new GreedyAgent(28),
    new RandomAgent(7),
  ];
  const engine = new GameEngine(cfg, agents);
  const winner = await engine.playMatch();
  assert(winner >= 0 && winner < 4, `winner index in range: ${winner}`);
  assert(engine.state.gameOver, 'game over');
  // someone hit target
  let max = -Infinity;
  for (const p of engine.state.players) if (p.totalScore > max) max = p.totalScore;
  assert(max >= 200, `max score >= 200, got ${max}`);
  console.log(`OK full match (winner=P${winner}, top=${max})`);
  _passed += 1;
}

async function runAll() {
  const tests = [
    ['deck size', testDeckSize],
    ['six burst', testSixBurst],
    ['bust no insurance', testBustNoInsurance],
    ['bust with insurance', testBustWithInsurance],
    ['exile', testExile],
    ['triple chain busts', testTripleChainBusts],
    ['full match smoke', testFullMatchSmoke],
  ];
  for (const [name, fn] of tests) {
    try {
      await fn();
    } catch (e) {
      console.error(`FAIL ${name}: ${e.message}`);
    }
  }
  console.log(`\n${_passed}/${_passed + _failed} passed`);
  if (_failed > 0) process.exit(1);
}

runAll();
