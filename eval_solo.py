"""测试每个 agent 对战 3 个 random 的"绝对实力"。
剥离零和博弈干扰，看真实赢率。
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import time

from agents import EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent
from game import GameConfig, GameEngine


def _make_agent(name, weights_bytes=b''):
    if name == 'random': return RandomAgent()
    if name == 'greedy': return GreedyAgent()
    if name == 'ev': return EVAgent()
    if name == 'expectimax' or name == 'exmax3': return ExpectimaxAgent(depth=3)
    if name == 'exmax2': return ExpectimaxAgent(depth=2)
    if name == 'exmax4': return ExpectimaxAgent(depth=4)
    if name == 'neural':
        import torch
        torch.set_num_threads(1)
        from train.network import build_model, infer_arch
        from agents.neural_agent import NeuralAgent
        state = torch.load(io.BytesIO(weights_bytes), map_location='cpu')
        h, n = infer_arch(state)
        m = build_model(hidden=h, n_layers=n)
        m.load_state_dict(state)
        m.eval()
        return NeuralAgent(model=m)
    raise ValueError(name)


def _worker(args):
    name, seed, weights_bytes, opponent = args
    target = _make_agent(name, weights_bytes)
    def make_opp(): return _make_agent(opponent, weights_bytes)

    agents = [target, make_opp(), make_opp(), make_opp()]
    cfg = GameConfig(num_players=4, seed=seed)
    e = GameEngine(cfg, agents)
    winner = e.play_match()
    return 1 if winner == 0 else 0


def run(name: str, n: int, weights_bytes: bytes, opponent: str, workers: int = 4):
    args = [(name, s, weights_bytes, opponent) for s in range(n)]
    wins = 0
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as pool:
        for w in pool.imap_unordered(_worker, args, chunksize=4):
            wins += w
    dt = time.time() - t0
    return wins, n, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-matches', type=int, default=300)
    parser.add_argument('--model', default='checkpoints/model_best.pt')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--opponent', default='random',
                        help='对手 agent: random / greedy / ev / exmax2 / exmax3 / exmax4 / expectimax')
    parser.add_argument('--agents', nargs='+',
                        default=['random', 'greedy', 'ev', 'exmax2', 'exmax3', 'exmax4', 'neural'])
    args = parser.parse_args()

    print(f"每个 agent vs 3×{args.opponent}，每组 {args.n_matches} 局")
    print(f"workers={args.workers}\n")

    weights_bytes = b''
    if 'neural' in args.agents:
        import torch
        try:
            sd = torch.load(args.model, map_location='cpu')
            buf = io.BytesIO()
            torch.save(sd, buf)
            weights_bytes = buf.getvalue()
            print(f"loaded model: {args.model}\n")
        except FileNotFoundError:
            print(f"⚠ model not found: {args.model}, skipping neural\n")
            args.agents = [a for a in args.agents if a != 'neural']

    print(f"{'agent':<14} {'wins':>6} {'胜率':>8}  {'25% 基线':>10}  {'耗时':>8}")
    print('─' * 60)
    for name in args.agents:
        wins, n, dt = run(name, args.n_matches, weights_bytes, args.opponent, args.workers)
        pct = wins / n * 100
        print(f"{name:<14} {wins:>6} {pct:>7.1f}%  {'25.0%':>10}  {dt:>6.1f}s")


if __name__ == '__main__':
    main()
