"""Vectorized self-play driver.

Runs N concurrent self-play games and batches all NN inference requests into
single forward passes — so GPU is actually utilized.

Usage:
    from train.vectorized_selfplay import vectorized_selfplay
    examples = vectorized_selfplay(model, n_concurrent=32, n_sims=80,
                                    n_games_total=120, device='cuda')
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from agents.ev_agent import EVAgent
from agents.mcts_agent import _hollow_engine
from game.cards import CardKind
from game.engine import GameEngine
from game.state import GameConfig, GameState, PlayerStatus
from train.encoder import encode_state
from train.selfplay import TrainingExample
from train.vectorized_mcts import ACTIONS, MCTSContext


@dataclass
class GameSession:
    """One self-play game in progress. Holds the engine + records training data."""
    engine: GameEngine
    examples: List[Tuple[np.ndarray, np.ndarray, int]]  # (features, π, my_idx)
    finished: bool = False

    def current_idx(self) -> int:
        return self.engine.state.current_player


def _ensure_round(engine: GameEngine):
    """If round just ended, settle and start a new round (or stop if game over)."""
    st = engine.state
    if st.round_over and not st.game_over:
        engine._settle_round()
        if not st.game_over:
            engine._begin_round()


def _step_until_decision(engine: GameEngine) -> bool:
    """Advance the engine until the active player needs a decision (chooseAction).
    Returns True if we stopped at a decision point, False if game is over."""
    st = engine.state
    while not st.game_over:
        if st.round_over:
            engine._settle_round()
            if st.game_over:
                return False
            engine._begin_round()
            continue
        # find next active player
        for _ in range(st.config.num_players):
            if st.players[st.current_player].is_active:
                return True
            engine._advance_turn()
            if st.round_over:
                break
        else:
            st.round_over = True
    return not st.game_over


def _build_engine(num_players: int = 4, target: int = 200, seed: Optional[int] = None) -> GameEngine:
    """Build a fresh engine with placeholder agents (we won't use them — driver controls actions)."""
    cfg = GameConfig(num_players=num_players, target_score=target, seed=seed)
    # placeholder agents (only used for skill targeting via EV heuristic)
    agents = [EVAgent() for _ in range(num_players)]
    engine = GameEngine(cfg, agents)
    engine.state.round_number = 0
    engine._begin_round()
    return engine


def vectorized_selfplay(model, n_concurrent: int = 32, n_sims: int = 80,
                         n_games_total: int = 120, num_players: int = 4,
                         target_score: int = 200,
                         batch_size: int = 32,
                         device: str = 'cpu',
                         seed: Optional[int] = None,
                         verbose: bool = False) -> List[TrainingExample]:
    """Run vectorized self-play, return TrainingExamples for all finished games."""
    import torch

    rng = np.random.default_rng(seed)

    # 池化：active sessions + 等待开始的 pending count
    sessions: List[GameSession] = []
    contexts: List[Optional[MCTSContext]] = []
    games_remaining = n_games_total

    def spawn_new_session():
        nonlocal games_remaining
        if games_remaining <= 0:
            return None
        games_remaining -= 1
        engine = _build_engine(num_players, target_score,
                                seed=int(rng.integers(0, 1 << 30)))
        if not _step_until_decision(engine):
            return None  # impossible but safe
        return GameSession(engine=engine, examples=[])

    # 初始填满 N 个 session
    for _ in range(n_concurrent):
        s = spawn_new_session()
        if s is None:
            break
        sessions.append(s)
        contexts.append(_make_context(s, n_sims, batch_size, rng))

    finished_examples: List[TrainingExample] = []
    decisions_done = 0
    t0 = time.time()

    while sessions:
        # ============ Phase 1: 给所有需要 prior 的 context 算 prior（一次 batched 推理）============
        need_prior_idx = [i for i, c in enumerate(contexts) if c.needs_prior()]
        if need_prior_idx:
            states = [(contexts[i].state, contexts[i].my_idx) for i in need_prior_idx]
            priors = _infer_priors(model, states, device)
            for i, p in zip(need_prior_idx, priors):
                contexts[i].set_prior(p)

        # ============ Phase 2: 同步 MCTS 推进，每轮 collect → batched value → backup ============
        while True:
            active_idx = [i for i, c in enumerate(contexts) if not c.is_done()]
            if not active_idx:
                break
            all_leaves: List[Tuple[GameState, int]] = []
            split_points: List[int] = [0]
            for i in active_idx:
                leaves = contexts[i].collect_leaves()
                all_leaves.extend(leaves)
                split_points.append(len(all_leaves))
            if not all_leaves:
                break
            values = _infer_values(model, all_leaves, device)
            for k, i in enumerate(active_idx):
                seg = values[split_points[k]:split_points[k + 1]]
                contexts[i].apply_values(seg)

        # ============ Phase 3: 每个 context 选 best action，应用，进入下一决策 ============
        for i in range(len(sessions)):
            ctx = contexts[i]
            sess = sessions[i]
            features = encode_state(ctx.state, ctx.my_idx)
            action, pi = ctx.best_action()
            sess.examples.append((features, pi, ctx.my_idx))
            decisions_done += 1

            # 应用 action 到 engine
            engine = sess.engine
            engine.state.current_player = ctx.my_idx
            if action == 'fold':
                p = engine.state.players[ctx.my_idx]
                p.locked_round_score = sum(p.hand_numbers) + p.bonus_flat_total
                p.status = PlayerStatus.FOLDED
            else:
                engine._draw_for(ctx.my_idx)
            engine.state.last_actor = ctx.my_idx

            # 推进到下一个决策点
            if not engine.state.round_over:
                engine._advance_turn()
            if not _step_until_decision(engine):
                # 这局结束了
                sess.finished = True

        # ============ 清理 finished sessions、补新 game ============
        for i in range(len(sessions) - 1, -1, -1):
            if sessions[i].finished:
                # 把 examples 转成 TrainingExample（带最终 placement-z）
                _finalize_session(sessions[i], finished_examples)
                # 替换为新 session 或移除
                new_sess = spawn_new_session()
                if new_sess is None:
                    sessions.pop(i)
                    contexts.pop(i)
                else:
                    sessions[i] = new_sess
                    contexts[i] = _make_context(new_sess, n_sims, batch_size, rng)
            else:
                # 重置 context 给下一个决策
                contexts[i] = _make_context(sessions[i], n_sims, batch_size, rng)

        if verbose and decisions_done % 50 == 0:
            elapsed = time.time() - t0
            done_games = n_games_total - games_remaining - len(sessions)
            print(f"  decisions={decisions_done}  games_done={done_games}/{n_games_total}  "
                  f"active={len(sessions)}  elapsed={elapsed:.1f}s")

    return finished_examples


def _make_context(sess: GameSession, n_sims: int, batch_size: int,
                   np_rng: np.random.Generator) -> MCTSContext:
    import random as _random
    return MCTSContext(
        state=sess.engine.state,
        my_idx=sess.current_idx(),
        n_sims=n_sims,
        batch_size=batch_size,
        rng=_random.Random(int(np_rng.integers(0, 1 << 30))),
        np_rng=np.random.default_rng(int(np_rng.integers(0, 1 << 30))),
    )


def _finalize_session(sess: GameSession, out: List[TrainingExample]) -> None:
    """Match ended — assign placement-z to every example based on final standings."""
    final_totals = [p.total_score for p in sess.engine.state.players]
    n = len(final_totals)
    rank = sorted(range(n), key=lambda i: -final_totals[i])
    placement_values = [1.0, -1/3, -2/3, -1.0]  # asymmetric (winner-biased)
    z = [0.0] * n
    for pos, idx in enumerate(rank):
        z[idx] = placement_values[pos] if pos < 4 else -1.0
    for features, pi, my_idx in sess.examples:
        out.append(TrainingExample(features=features, policy=pi, value=z[my_idx]))


def _infer_priors(model, states: List[Tuple[GameState, int]], device: str) -> np.ndarray:
    if model is None:
        return np.full((len(states), 2), 0.5, dtype=np.float32)
    import torch
    x = np.stack([encode_state(s, idx) for s, idx in states])
    with torch.no_grad():
        X = torch.from_numpy(x).to(device)
        logits, _ = model(X)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _infer_values(model, states: List[Tuple[GameState, int]], device: str) -> np.ndarray:
    if model is None:
        return np.zeros(len(states), dtype=np.float32)
    import torch
    x = np.stack([encode_state(s, idx) for s, idx in states])
    with torch.no_grad():
        X = torch.from_numpy(x).to(device)
        _, v = model(X)
        return v.cpu().numpy()
