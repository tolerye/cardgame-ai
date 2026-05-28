"""MCTS agent (Information-Set MCTS, single-observer).

Each simulation re-determinizes the deck according to GameState.remaining,
applies a candidate root action, then rolls out using EVAgent as policy for
all players. The reward is signed margin in total score after the round
ends (or ±1 if the match ends), normalized by target_score.

This is intentionally a flat (root-only) MCTS — fast, simple, and good
enough given EV's quality as rollout policy. Tree expansion past the root
adds complexity without big gains here because draws inject randomness
that re-distributes across simulations naturally."""

from __future__ import annotations

import copy
import math
import random
from typing import List, Optional

from agents.base import BaseAgent
from agents.ev_agent import EVAgent
from agents.expectimax_agent import ExpectimaxAgent
from game.cards import CardKind
from game.engine import GameEngine
from game.state import GameState


class _ActionStat:
    __slots__ = ("n", "w")

    def __init__(self) -> None:
        self.n = 0
        self.w = 0.0

    def mean(self) -> float:
        return self.w / self.n if self.n else 0.0


class MCTSAgent(BaseAgent):
    name = "mcts"

    def __init__(
        self,
        n_simulations: int = 400,
        c_ucb: float = 1.4,
        rollout_factory=None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_sims = n_simulations
        self.c = c_ucb
        # default rollout: lightweight EV (single-step). Expectimax-based rollout is
        # available via rollout_factory but typically too slow to be worth the cost
        # when the search tree is shallow.
        self.rollout_factory = rollout_factory or (lambda: EVAgent())
        self.rng = random.Random(seed)

    # ------------------------------------------------------------ main API
    def choose_action(self, state: GameState, my_idx: int) -> str:
        me = state.players[my_idx]
        cur = sum(me.hand_numbers) + me.bonus_flat_total
        if me.total_score + cur >= state.config.target_score:
            return "fold"  # endgame guard

        actions = ["draw", "fold"]
        stats = {a: _ActionStat() for a in actions}

        for _ in range(self.n_sims):
            a = self._ucb_select(stats)
            r = self._simulate(state, my_idx, a)
            s = stats[a]
            s.n += 1
            s.w += r

        best = max(actions, key=lambda a: stats[a].mean())
        return best

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        # Delegate to EV heuristic — running MCTS for every skill target lookup is too slow.
        return EVAgent().choose_skill_target(state, my_idx, kind)

    # ------------------------------------------------------------ internals
    def _ucb_select(self, stats: dict[str, _ActionStat]) -> str:
        total = sum(s.n for s in stats.values())
        # always try each at least once
        for a, s in stats.items():
            if s.n == 0:
                return a
        log_total = math.log(total)
        return max(stats, key=lambda a: stats[a].mean()
                   + self.c * math.sqrt(log_total / stats[a].n))

    def _simulate(self, state: GameState, my_idx: int, root_action: str) -> float:
        # Build a sim engine. Opponents use the (cheap) rollout policy; my future
        # turns inside the sim use a stronger strategy (expectimax) so that the
        # value of root_action isn't blurred by my rollouts being suboptimal.
        rollout_agents: list = []
        for i in range(state.config.num_players):
            if i == my_idx:
                rollout_agents.append(ExpectimaxAgent(depth=2))
            else:
                rollout_agents.append(self.rollout_factory())
        host = _hollow_engine(state)
        sim = host.clone_for_simulation(
            agents=rollout_agents, rng_seed=self.rng.randint(0, 1 << 30)
        )
        # Apply the root action manually (skipping rollout policy at root).
        sim.state.current_player = my_idx
        if root_action == "fold":
            me = sim.state.players[my_idx]
            me.locked_round_score = sum(me.hand_numbers) + me.bonus_flat_total
            from game.state import PlayerStatus
            me.status = PlayerStatus.FOLDED
            sim.state.last_actor = my_idx
        else:
            sim._draw_for(my_idx)
            sim.state.last_actor = my_idx
        # Continue the round (and possibly more rounds until match ends or
        # we've seen a full round resolve).
        if not sim.state.round_over:
            sim._advance_turn()
            sim.play_round_to_completion()
        else:
            sim._settle_round()
        # If match still going, run one more round to capture next-starter dynamics
        # without making rollouts unbounded.
        if not sim.state.game_over:
            sim.play_round()
        return self._reward(sim.state, my_idx)

    @staticmethod
    def _reward(state: GameState, my_idx: int) -> float:
        target = state.config.target_score
        if state.game_over:
            return 1.0 if state.winner == my_idx else -1.0
        me = state.players[my_idx].total_score
        others = max(p.total_score for p in state.players if p.index != my_idx)
        return max(-1.0, min(1.0, (me - others) / target))


def _hollow_engine(state: GameState) -> GameEngine:
    """Make a GameEngine wrapper around an existing GameState, sharing the
    state object. Only used as a starting point for clone_for_simulation."""
    # Use object.__new__ to bypass __init__ which would build fresh state.
    e = GameEngine.__new__(GameEngine)
    e.config = state.config
    e.agents = []  # not used until clone
    e.rng = random.Random()
    e.state = state
    # _deck for the original engine isn't accessible from state alone; that's
    # fine — clone_for_simulation rebuilds it from state.remaining.
    from game.cards import Deck
    e._deck = Deck()
    return e
