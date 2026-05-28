"""Inference-only wrapper for a trained PolicyValue model. Stateless agent."""

from __future__ import annotations

from typing import Optional

import numpy as np

from agents.base import BaseAgent
from agents.ev_agent import EVAgent
from game.cards import CardKind
from game.state import GameState
from train.encoder import encode_state


class NeuralAgent(BaseAgent):
    name = "neural"

    def __init__(self, model_path: Optional[str] = None, model=None,
                 use_mcts: bool = False, n_simulations: int = 100) -> None:
        if model is not None:
            self.model = model
        elif model_path is not None:
            from train.network import load
            self.model = load(model_path)
        else:
            raise ValueError("must provide model or model_path")
        self.use_mcts = use_mcts
        self.n_sims = n_simulations
        self._mcts_agent = None
        if use_mcts:
            from agents.batched_mcts import BatchedNeuralMCTSAgent
            # Inference: blend 70% NN + 30% handcrafted EV in leaf evaluation
            # to steady the search until the value head is well-trained.
            self._mcts_agent = BatchedNeuralMCTSAgent(
                model=self.model, n_simulations=n_simulations, batch_size=32,
                hybrid_alpha=0.7)

    def choose_action(self, state: GameState, my_idx: int) -> str:
        me = state.players[my_idx]
        cur = sum(me.hand_numbers) + me.bonus_flat_total
        if me.total_score + cur >= state.config.target_score:
            return "fold"
        if self.use_mcts:
            return self._mcts_agent.choose_action(state, my_idx)
        # raw policy head
        import torch
        x = torch.from_numpy(encode_state(state, my_idx)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(x)
        idx = int(torch.argmax(logits, dim=-1).item())
        return ["draw", "fold"][idx]

    def choose_skill_target(self, state, my_idx, kind):
        return EVAgent().choose_skill_target(state, my_idx, kind)
