"""Monte Carlo 胜率/排位预估器。

给定当前 GameState 和 4 个 agent 的"风格"，返回每位玩家：
  - P(rank=1 / 2 / 3 / 4)：本场比赛最终排名概率
  - P(本局头名)：当前一局结束后谁本局得分最高的概率
  - 本局期望加分

核心思路：clone 当前 state，换上 rollout 用的快速 agent，play_match() 多次取统计。

为了实时用得起，rollout 默认用 EV agent (~10× 比 expectimax depth=3 快)；
若想要更准可传 ExpectimaxAgent(depth=2)。"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from agents import EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent
from agents.base import BaseAgent
from game import GameConfig, GameEngine
from game.state import GameState


AgentFactory = Callable[[], BaseAgent]


# ---------------------------------------------------------- agent factories
def _factory(name: str) -> AgentFactory:
    if name == 'random': return lambda: RandomAgent()
    if name == 'greedy': return lambda: GreedyAgent()
    if name == 'ev': return lambda: EVAgent()
    if name == 'exmax2': return lambda: ExpectimaxAgent(depth=2)
    if name == 'exmax3': return lambda: ExpectimaxAgent(depth=3)
    if name == 'exmax4': return lambda: ExpectimaxAgent(depth=4)
    raise ValueError(f"unknown agent: {name}")


# ---------------------------------------------------------- result struct
@dataclass
class PlayerEstimate:
    idx: int
    p_rank: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # P[rank=1..4]
    p_round_win: float = 0.0          # P(本局得分最高)
    expected_round_gain: float = 0.0  # E[本局得分增量]
    expected_final_score: float = 0.0 # E[比赛结束时总分]


@dataclass
class Estimate:
    n_sims: int
    elapsed: float
    players: List[PlayerEstimate]
    has_active_round: bool = False


# ---------------------------------------------------------- core sim
def _one_sim(args):
    state_pickle, agent_names, seed = args
    # 子进程里重新构建 state（避免 deepcopy 的开销 / pickle 隐式 size）
    import pickle
    state: GameState = pickle.loads(state_pickle)
    agents = [_factory(n)() for n in agent_names]

    # 用 clone_for_simulation 强制重洗 deck（保留 remaining counts）
    host = GameEngine.__new__(GameEngine)
    host.config = state.config
    host.agents = []
    host._deck = None
    host.state = state
    sim = host.clone_for_simulation(agents, rng_seed=seed)

    n = state.config.num_players
    pre_round_scores = [p.total_score for p in state.players]
    # 只有"当前正在进行某局且未结束"时，才有"本局头名"概念
    has_active_round = (sim.state.round_number > 0
                        and not sim.state.round_over
                        and not sim.state.game_over)

    if has_active_round:
        sim.play_round_to_completion()
        round_gains = [
            sim.state.players[i].total_score - pre_round_scores[i] for i in range(n)
        ]
        round_win_idx = max(range(n), key=lambda i: round_gains[i])
    else:
        round_gains = [0.0] * n
        round_win_idx = -1   # 哨兵：没有"当前局"

    if not sim.state.game_over:
        sim.play_match()

    final = [p.total_score for p in sim.state.players]
    rank = sorted(range(n), key=lambda i: -final[i])
    rank_pos = [0] * n
    for pos, idx in enumerate(rank):
        rank_pos[idx] = pos

    return rank_pos, round_gains, round_win_idx, final, has_active_round


def estimate(
    state: GameState,
    agent_names: List[str],
    n_sims: int = 80,
    n_workers: int = 4,
    base_seed: int = 0,
) -> Estimate:
    """跑 n_sims 次模拟，返回每位玩家的概率统计。

    agent_names: 长度等于玩家数，第 i 个是模拟里第 i 位玩家用什么 agent rollout。
    """
    assert len(agent_names) == state.config.num_players
    n = state.config.num_players

    import pickle
    state_pickle = pickle.dumps(state)

    args = [(state_pickle, agent_names, base_seed + i) for i in range(n_sims)]

    t0 = time.time()
    rank_counts = [[0] * n for _ in range(n)]   # rank_counts[i][k] = #(玩家 i 排名 k)
    round_win_counts = [0] * n
    sum_round_gain = [0.0] * n
    sum_final = [0.0] * n
    n_with_round = 0   # 有"当前局"的模拟数

    if n_workers <= 1:
        for a in args:
            rp, rg, rw, final, has_round = _one_sim(a)
            for i in range(n):
                rank_counts[i][rp[i]] += 1
                sum_round_gain[i] += rg[i]
                sum_final[i] += final[i]
            if has_round and rw >= 0:
                round_win_counts[rw] += 1
                n_with_round += 1
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            for rp, rg, rw, final, has_round in pool.imap_unordered(_one_sim, args, chunksize=4):
                for i in range(n):
                    rank_counts[i][rp[i]] += 1
                    sum_round_gain[i] += rg[i]
                    sum_final[i] += final[i]
                if has_round and rw >= 0:
                    round_win_counts[rw] += 1
                    n_with_round += 1

    elapsed = time.time() - t0

    players = []
    for i in range(n):
        round_w = round_win_counts[i] / n_with_round if n_with_round > 0 else 0.0
        round_g = sum_round_gain[i] / n_with_round if n_with_round > 0 else 0.0
        players.append(PlayerEstimate(
            idx=i,
            p_rank=[rank_counts[i][k] / n_sims for k in range(n)],
            p_round_win=round_w,
            expected_round_gain=round_g,
            expected_final_score=sum_final[i] / n_sims,
        ))
    return Estimate(n_sims=n_sims, elapsed=elapsed, players=players,
                    has_active_round=(n_with_round > 0))


# ---------------------------------------------------------- 终端打印
def print_estimate(est: Estimate, names: Optional[List[str]] = None) -> None:
    n = len(est.players)
    if names is None:
        names = [f"P{i}" for i in range(n)]
    print(f"\n=== 胜率预估（{est.n_sims} 局 MC，{est.elapsed:.1f}s）===")
    if est.has_active_round:
        header = f"{'玩家':<10} {'1名':>6} {'2名':>6} {'3名':>6} {'4名':>6}  {'本局头名':>8}  {'E[本局+]':>9}  {'E[终分]':>8}"
        print(header)
        print('─' * len(header))
        for p in est.players:
            nm = names[p.idx]
            p1, p2, p3, p4 = (x * 100 for x in p.p_rank)
            rw = p.p_round_win * 100
            print(f"{nm:<10} {p1:>5.1f}% {p2:>5.1f}% {p3:>5.1f}% {p4:>5.1f}%  "
                  f"{rw:>7.1f}%  {p.expected_round_gain:>+9.1f}  {p.expected_final_score:>8.1f}")
    else:
        header = f"{'玩家':<10} {'1名':>6} {'2名':>6} {'3名':>6} {'4名':>6}  {'E[终分]':>8}"
        print(header)
        print('─' * len(header))
        for p in est.players:
            nm = names[p.idx]
            p1, p2, p3, p4 = (x * 100 for x in p.p_rank)
            print(f"{nm:<10} {p1:>5.1f}% {p2:>5.1f}% {p3:>5.1f}% {p4:>5.1f}%  "
                  f"{p.expected_final_score:>8.1f}")


# ---------------------------------------------------------- CLI
def _build_state_from_cli(args) -> tuple[GameState, list[str]]:
    """命令行构造一个起始 state（用于纯输入分数推演的场景）。"""
    cfg = GameConfig(num_players=4, target_score=args.target)
    # 创建一个空的开局 state，再手动设置 total_score
    engine = GameEngine(cfg, [_factory(n)() for n in args.agents])
    st = engine.state
    scores = [int(x) for x in args.scores.split(',')]
    if len(scores) != 4:
        raise ValueError("--scores 必须是 4 个用逗号分隔的整数")
    for i, s in enumerate(scores):
        st.players[i].total_score = s
    # round_number=0 表示比赛还没开始第一局
    return st, args.agents


def main():
    parser = argparse.ArgumentParser(description="胜率/排位 MC 预估")
    parser.add_argument('--scores', required=True,
                        help='4 人当前总分，逗号分隔。例：80,120,40,60')
    parser.add_argument('--target', type=int, default=200, help='比赛目标分')
    parser.add_argument('--agents', nargs=4, default=['ev'] * 4,
                        help='4 人 agent 类型，空格分隔。可选 random/greedy/ev/exmax2/exmax3/exmax4')
    parser.add_argument('-n', '--n-sims', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--names', default=None,
                        help='4 人显示名，逗号分隔。例：你,小明,小红,小李')
    args = parser.parse_args()

    state, agent_names = _build_state_from_cli(args)
    est = estimate(state, agent_names, n_sims=args.n_sims, n_workers=args.workers)

    names = args.names.split(',') if args.names else None
    print(f"目标分: {args.target}  /  agents: {agent_names}")
    cur = [p.total_score for p in state.players]
    print(f"当前总分: {cur}")
    print_estimate(est, names)


if __name__ == '__main__':
    main()
