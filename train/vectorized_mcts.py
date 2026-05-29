"""Vectorized parallel self-play.

N 个独立 self-play game 同步推进，每次 NN 推理把所有 game 的 leaf state 拼成
一个大 batch 一起送 GPU。这样 GPU 利用率从 10% 提到 50%+，吞吐 5-10x 提升。

设计要点：
- 单进程（绕开 multiprocessing + CUDA 死锁问题）
- 每个 game 独立的 MCTS 上下文（树/visit/virtual loss）
- "phase 锁步"：所有 active context 都收集完 leaves → 统一推理 → 各自 backup

每个决策周期：
  1. 所有 context 收集 batch_size 个 leaves（virtual loss 散开）
  2. 全部 leaves 拼 batch 推理一次
  3. 各 context backup
  4. 重复直到每个 context 跑完 n_sims
  5. 各 game 应用 best action 进 next decision
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from agents.ev_agent import EVAgent
from agents.mcts_agent import _hollow_engine
from game.cards import CardKind
from game.engine import GameEngine
from game.state import GameState, PlayerStatus
from train.encoder import encode_state


VIRTUAL_LOSS = 1.0
ACTIONS = ['draw', 'fold']


@dataclass
class MCTSContext:
    """MCTS state for one decision in one game."""
    state: GameState
    my_idx: int
    n_sims: int
    batch_size: int
    c_puct: float = 1.5
    dirichlet_eps: float = 0.25
    dirichlet_alpha: float = 0.5
    temperature: float = 1.0
    hybrid_alpha: float = 0.5

    # MCTS state
    N: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    W: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    VL: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    prior: Optional[np.ndarray] = None

    n_done: int = 0
    pending_picks: List[int] = field(default_factory=list)
    pending_leaves: List[Tuple[GameState, int]] = field(default_factory=list)
    rng: Optional[random.Random] = None
    np_rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random()
        if self.np_rng is None:
            self.np_rng = np.random.default_rng()

    def needs_prior(self) -> bool:
        return self.prior is None

    def set_prior(self, p: np.ndarray) -> None:
        if self.dirichlet_eps > 0:
            noise = self.np_rng.dirichlet([self.dirichlet_alpha] * 2)
            p = (1 - self.dirichlet_eps) * p + self.dirichlet_eps * noise
        self.prior = p

    def is_done(self) -> bool:
        return self.n_done >= self.n_sims

    def _puct_select(self) -> int:
        Q_eff = (self.W - self.VL) / np.maximum(self.N + self.VL, 1)
        total = float(self.N.sum() + self.VL.sum() + 1)
        ucb = Q_eff + self.c_puct * self.prior * math.sqrt(total) / (1 + self.N + self.VL)
        return int(np.argmax(ucb))

    def collect_leaves(self) -> List[Tuple[GameState, int]]:
        """Return up to batch_size (state, my_idx) pairs to evaluate. Empty if done."""
        if self.is_done():
            return []
        this_batch = min(self.batch_size, self.n_sims - self.n_done)
        picks: List[int] = []
        leaves: List[Tuple[GameState, int]] = []
        for _ in range(this_batch):
            a_idx = self._puct_select()
            self.VL[a_idx] += VIRTUAL_LOSS
            action = ACTIONS[a_idx]
            leaf_state = _simulate_to_frontier(self.state, self.my_idx, action, self.rng)
            picks.append(a_idx)
            leaves.append((leaf_state, self.my_idx))
        self.pending_picks = picks
        self.pending_leaves = leaves
        return leaves

    def apply_values(self, nn_values: np.ndarray) -> None:
        # hybrid leaf eval: blend NN value with handcrafted EV signal
        for a_idx, leaf, v_nn in zip(self.pending_picks, self.pending_leaves, nn_values):
            v = self.hybrid_alpha * v_nn + (1 - self.hybrid_alpha) * _ev_signal(*leaf)
            self.VL[a_idx] -= VIRTUAL_LOSS
            self.N[a_idx] += 1
            self.W[a_idx] += v
        self.pending_picks = []
        self.pending_leaves = []
        self.n_done = int(self.N.sum())

    def best_action(self) -> Tuple[str, np.ndarray]:
        """Return (action, visit_distribution)."""
        visits = self.N.copy()
        pi = visits / max(visits.sum(), 1)
        if self.temperature > 0 and visits.sum() > 0:
            scaled = visits ** (1.0 / self.temperature)
            probs = scaled / scaled.sum()
            choice = int(self.np_rng.choice(2, p=probs))
            return ACTIONS[choice], pi
        return ACTIONS[int(np.argmax(visits))], pi


def _ev_signal(state: GameState, my_idx: int) -> float:
    me = state.players[my_idx]
    target = state.config.target_score
    my_total = me.total_score + me.current_round_score()
    others_max = max(p.total_score + p.current_round_score()
                     for p in state.players if p.index != my_idx)
    margin = (my_total - others_max) / target
    return max(-1.0, min(1.0, margin * 2))


def _simulate_to_frontier(state: GameState, my_idx: int, action: str,
                          rng: random.Random) -> GameState:
    """Apply action then play out with EV agents until round end."""
    rollout_agents = [EVAgent() for _ in range(state.config.num_players)]
    host = _hollow_engine(state)
    sim = host.clone_for_simulation(rollout_agents, rng_seed=rng.randint(0, 1 << 30))
    sim.state.current_player = my_idx
    if action == 'fold':
        me = sim.state.players[my_idx]
        me.locked_round_score = sum(me.hand_numbers) + me.bonus_flat_total
        me.status = PlayerStatus.FOLDED
        sim.state.last_actor = my_idx
    else:
        sim._draw_for(my_idx)
        sim.state.last_actor = my_idx
    if not sim.state.round_over:
        sim._advance_turn()
        sim.play_round_to_completion()
    else:
        sim._settle_round()
    return sim.state
