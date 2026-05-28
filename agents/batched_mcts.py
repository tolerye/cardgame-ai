"""Batched MCTS with virtual loss.

Difference from MCTSAgent:
- Each decision launches `n_simulations` rollouts, but they are NOT executed
  one by one. Instead we collect `batch_size` leaf states first (using
  virtual loss to force different paths), evaluate them in a single
  GPU/CPU forward pass, then back up. This makes NN inference O(n_sims/batch)
  forward passes instead of O(n_sims), unlocking GPU throughput.

- Leaves are evaluated by the network's value head (no expectimax rollout) —
  this is the AlphaZero design and lets GPU be the bottleneck instead of
  Python rollout.

Falls back to a constant prior + zero value if no model is supplied (useful
for unit tests)."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

from agents.base import BaseAgent
from agents.ev_agent import EVAgent
from agents.mcts_agent import _hollow_engine
from game.cards import CardKind
from game.engine import GameEngine
from game.state import GameState, PlayerStatus
from train.encoder import encode_state


VIRTUAL_LOSS = 1.0  # discourages other parallel sims from re-picking same action


class BatchedNeuralMCTSAgent(BaseAgent):
    name = "batched_mcts"

    def __init__(
        self,
        model=None,
        n_simulations: int = 200,
        batch_size: int = 32,
        c_puct: float = 1.5,
        device: str = "cpu",
        seed: Optional[int] = None,
        dirichlet_eps: float = 0.0,
        dirichlet_alpha: float = 0.5,
        temperature: float = 0.0,
        hybrid_alpha: float = 1.0,
    ) -> None:
        """hybrid_alpha: weight on NN value head when computing leaf value.
            1.0 = pure NN (default), 0.0 = pure handcrafted EV signal.
            Use < 1 early in training when NN value head is unreliable."""
        self.model = model
        self.n_sims = n_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.device = device
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
        self.hybrid_alpha = hybrid_alpha
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._last_visits: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- public
    def choose_action(self, state: GameState, my_idx: int) -> str:
        me = state.players[my_idx]
        cur = sum(me.hand_numbers) + me.bonus_flat_total
        if me.total_score + cur >= state.config.target_score:
            self._last_visits = np.array([0.0, 1.0], dtype=np.float32)
            return "fold"

        # Get root prior in one inference call
        prior = self._infer_prior_batched([(state, my_idx)])[0]
        # If no model, use uniform prior (still benefits from search)
        if self.model is None:
            prior = np.array([0.5, 0.5], dtype=np.float32)
        # Dirichlet exploration noise at root (AlphaZero recipe; only for self-play)
        if self.dirichlet_eps > 0:
            noise = self._np_rng.dirichlet([self.dirichlet_alpha, self.dirichlet_alpha])
            prior = (1 - self.dirichlet_eps) * prior + self.dirichlet_eps * noise

        N = np.zeros(2, dtype=np.float32)
        W = np.zeros(2, dtype=np.float32)
        VL = np.zeros(2, dtype=np.float32)  # virtual loss counters

        n_done = 0
        while n_done < self.n_sims:
            this_batch = min(self.batch_size, self.n_sims - n_done)
            picks: List[int] = []
            leaf_states: List[Tuple[GameState, int]] = []

            # 1. Pick `this_batch` actions via PUCT+VL, simulate each to a frontier state
            for _ in range(this_batch):
                a_idx = self._puct_select(N, W, VL, prior)
                VL[a_idx] += VIRTUAL_LOSS
                action = ["draw", "fold"][a_idx]
                leaf_state = self._simulate_to_frontier(state, my_idx, action)
                picks.append(a_idx)
                leaf_states.append((leaf_state, my_idx))

            # 2. Batched leaf evaluation (this is the GPU win)
            values = self._infer_values_batched(leaf_states)

            # 3. Back up + remove virtual loss
            for a_idx, v in zip(picks, values):
                VL[a_idx] -= VIRTUAL_LOSS
                N[a_idx] += 1
                W[a_idx] += v

            n_done += this_batch

        # Visit counts as policy (for training); pick best for play.
        # In self-play with temperature>0, sample by visit distribution to add diversity.
        self._last_visits = N / max(N.sum(), 1)
        if self.temperature > 0 and N.sum() > 0:
            # T=1: sample proportional to visits; T<1: sharpened; T>1: flatter
            scaled = N ** (1.0 / self.temperature)
            probs = scaled / scaled.sum()
            choice = int(self._np_rng.choice(2, p=probs))
            return ["draw", "fold"][choice]
        return ["draw", "fold"][int(np.argmax(N))]

    def choose_skill_target(self, state, my_idx, kind):
        return EVAgent().choose_skill_target(state, my_idx, kind)

    # --------------------------------------------------------------- PUCT
    def _puct_select(self, N: np.ndarray, W: np.ndarray, VL: np.ndarray,
                     prior: np.ndarray) -> int:
        Q_eff = (W - VL) / np.maximum(N + VL, 1)  # virtual-loss-adjusted mean
        total = N.sum() + VL.sum() + 1
        ucb = Q_eff + self.c_puct * prior * math.sqrt(total) / (1 + N + VL)
        return int(np.argmax(ucb))

    # --------------------------------------------------------------- batched NN
    def _infer_prior_batched(self, states: List[Tuple[GameState, int]]) -> np.ndarray:
        if self.model is None:
            return np.full((len(states), 2), 0.5, dtype=np.float32)
        import torch
        x = np.stack([encode_state(s, idx) for s, idx in states])
        with torch.no_grad():
            X = torch.from_numpy(x).to(self.device)
            logits, _ = self.model(X)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
        return p

    def _infer_values_batched(self, states: List[Tuple[GameState, int]]) -> np.ndarray:
        # Handcrafted score-margin signal — same shape as a placement value
        # (range [-1, 1]). Cheap, no network needed. Used as a teacher signal.
        ev_signal = np.zeros(len(states), dtype=np.float32)
        target = states[0][0].config.target_score if states else 200
        for i, (s, idx) in enumerate(states):
            me = s.players[idx]
            my_total = me.total_score + me.current_round_score()
            others_max = max(p.total_score + p.current_round_score()
                             for p in s.players if p.index != idx)
            margin = (my_total - others_max) / target
            ev_signal[i] = max(-1.0, min(1.0, margin * 2))  # scale so margin=0.5 → 1.0

        if self.model is None or self.hybrid_alpha == 0.0:
            return ev_signal

        import torch
        x = np.stack([encode_state(s, idx) for s, idx in states])
        with torch.no_grad():
            X = torch.from_numpy(x).to(self.device)
            _, v = self.model(X)
            nn_values = v.cpu().numpy()

        if self.hybrid_alpha >= 1.0:
            return nn_values
        return self.hybrid_alpha * nn_values + (1 - self.hybrid_alpha) * ev_signal

    # ------------------------------------------------------------ simulator
    def _simulate_to_frontier(self, state: GameState, my_idx: int, action: str) -> GameState:
        """Apply `action` from current state, then play out using EV policy
        until either: (a) round ends, (b) it's my_idx's turn again, or
        (c) game ends. Returns the final state observed."""
        rollout_agents = [EVAgent() for _ in range(state.config.num_players)]
        host = _hollow_engine(state)
        sim = host.clone_for_simulation(rollout_agents,
                                          rng_seed=self.rng.randint(0, 1 << 30))
        sim.state.current_player = my_idx
        if action == "fold":
            me = sim.state.players[my_idx]
            me.locked_round_score = sum(me.hand_numbers) + me.bonus_flat_total
            me.status = PlayerStatus.FOLDED
            sim.state.last_actor = my_idx
        else:
            sim._draw_for(my_idx)
            sim.state.last_actor = my_idx

        # Continue current round
        if not sim.state.round_over:
            sim._advance_turn()
            sim.play_round_to_completion()
        else:
            sim._settle_round()
        return sim.state
