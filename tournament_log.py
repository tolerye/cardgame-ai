"""锦标赛 + 详细日志工具。

跑 N 局给定 4 agent 的对局，把每一步事件写到桌面的日志文件，最后给汇总统计。

用法：
    python3 tournament_log.py -n 50 --agents exmax3 ev neural greedy
    python3 tournament_log.py -n 200 --agents ev ev ev ev --target 200
    python3 tournament_log.py -n 100 --out ~/Desktop/my_tournament.log

agent 可选：random / greedy / greedy20 / ev / exmax2 / exmax3 / exmax4 /
            neural / neural_mcts
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

from agents import EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent
from agents.base import BaseAgent
from game import GameConfig, GameEngine
from game.state import GameState


AgentFactory = Callable[[], BaseAgent]


REGISTRY: dict[str, AgentFactory] = {
    "random": lambda: RandomAgent(),
    "greedy": lambda: GreedyAgent(),
    "greedy20": lambda: GreedyAgent(fold_at=20),
    "ev": lambda: EVAgent(),
    "exmax2": lambda: ExpectimaxAgent(depth=2),
    "exmax3": lambda: ExpectimaxAgent(depth=3),
    "exmax4": lambda: ExpectimaxAgent(depth=4),
}


def _maybe_register_neural() -> None:
    try:
        from agents.neural_agent import NeuralAgent
        REGISTRY["neural"] = lambda: NeuralAgent("checkpoints/model_best.pt")
        REGISTRY["neural_mcts"] = lambda: NeuralAgent(
            "checkpoints/model_best.pt", use_mcts=True, n_simulations=80)
    except Exception:
        pass


# ---------------------------------------------------- log translation
def translate(line: str) -> str:
    """把 R3|P1: DRAW 7 之类的事件翻译成中文。"""
    out = (line
           .replace("SIX-BURST! lock=", "🚀 6翻了！锁分 ")
           .replace("FOLD lock=", "🔒 跑路 锁 ")
           .replace("DRAW ", "🃏 抽到 ")
           .replace("BUST_AVOIDED on ", "重复 ")
           .replace("BUST on ", "💥 爆牌！重复 ")
           .replace("BONUS+", "⭐ 加分牌 +")
           .replace("DOUBLE +", "✨ 翻倍牌 +")
           .replace("INSURANCE+", "🛡 拿到保险")
           .replace("INSURANCE -> P", "🛡 送保险给 P")
           .replace("INSURANCE wasted", "保险作废")
           .replace("EXILE -> P", "🚷 放逐 P")
           .replace("EXILE wasted", "放逐作废")
           .replace("TRIPLE -> P", "⚡ 三连 → P")
           .replace("TRIPLE wasted", "三连作废"))
    return out


# ---------------------------------------------------- desktop path
def desktop_path() -> Path:
    """跨平台找桌面目录。"""
    home = Path.home()
    candidates = [home / "Desktop", home / "桌面", home / "OneDrive" / "Desktop"]
    for c in candidates:
        if c.is_dir():
            return c
    # fallback: home
    return home


def default_log_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return desktop_path() / f"cardgame_tournament_{ts}.log"


# ---------------------------------------------------- run one match
def run_match(names: List[str], seed: int) -> Tuple[GameState, int]:
    cfg = GameConfig(num_players=len(names), target_score=200, seed=seed)
    agents = [REGISTRY[n]() for n in names]
    engine = GameEngine(cfg, agents)
    winner = engine.play_match()
    return engine.state, winner


def write_match(f, match_idx: int, names: List[str], state: GameState,
                winner: int) -> None:
    """把一局对局完整写入日志文件。"""
    f.write(f"\n{'='*70}\n")
    f.write(f"第 {match_idx + 1} 局对决\n")
    f.write(f"{'='*70}\n")
    f.write(f"参赛方:  ")
    for i, n in enumerate(names):
        tag = '🥇' if i == winner else '  '
        f.write(f"{tag}P{i}={n}  ")
    f.write("\n")
    final = [p.total_score for p in state.players]
    f.write(f"最终分:  " + "  ".join(f"P{i}={s}" for i, s in enumerate(final)) + "\n")
    f.write(f"总轮数:  {state.round_number} 局\n\n")

    # 按 round 分组事件
    events_by_round: dict[int, list[str]] = {}
    for line in state.log:
        # R<N>|P<i>: <msg>
        try:
            r_part, rest = line.split("|", 1)
            r = int(r_part[1:])
        except Exception:
            r = 0
        events_by_round.setdefault(r, []).append(line)

    for r in sorted(events_by_round.keys()):
        f.write(f"  ── 第 {r} 局 ──\n")
        for line in events_by_round[r]:
            try:
                _, rest = line.split("|", 1)
                pid_str, msg = rest.split(":", 1)
                pid = int(pid_str[1:])
                cn = translate(msg.strip())
                f.write(f"    P{pid} ({names[pid]:<10})  {cn}\n")
            except Exception:
                f.write(f"    {line}\n")
        # 局末快照
        f.write(f"    [本局结束] " +
                "  ".join(f"P{i}={p.total_score}" for i, p in enumerate(state.players))
                + "\n\n")


# ---------------------------------------------------- main
def main() -> None:
    _maybe_register_neural()

    parser = argparse.ArgumentParser(description="自定义对局锦标赛 + 日志保存")
    parser.add_argument("-n", "--n-matches", type=int, default=20,
                        help="跑多少局（默认 20）")
    parser.add_argument("--agents", nargs=4, default=["exmax3", "ev", "greedy", "random"],
                        help="4 个 agent，空格分隔（默认 exmax3 ev greedy random）")
    parser.add_argument("--target", type=int, default=200,
                        help="目标分（默认 200）")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子起点（默认 1）")
    parser.add_argument("--out", default=None,
                        help="日志输出路径（默认桌面 cardgame_tournament_<时间>.log）")
    parser.add_argument("--brief", action="store_true",
                        help="不写每局详细事件，只写汇总（日志小很多）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用 agent")
    args = parser.parse_args()

    if args.list:
        print("可用 agent：")
        for name in REGISTRY:
            print(f"  {name}")
        return

    # 验证 agent 名
    for n in args.agents:
        if n not in REGISTRY:
            print(f"❌ 未知 agent: '{n}'。用 --list 看可用列表")
            sys.exit(1)

    out_path = Path(args.out).expanduser() if args.out else default_log_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"开始：{args.n_matches} 局对决")
    print(f"参赛: P0={args.agents[0]}  P1={args.agents[1]}  P2={args.agents[2]}  P3={args.agents[3]}")
    print(f"目标: {args.target} 分    日志: {out_path}")
    print()

    wins = [0] * 4
    sum_scores = [0] * 4
    sum_rounds = 0
    placement_counts = [Counter() for _ in range(4)]  # placement_counts[i][rank]
    busts = [0] * 4
    six_bursts = [0] * 4
    folds = [0] * 4

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        # 标题
        f.write(f"6翻了 · 锦标赛日志\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"机器:     {platform.system()} {platform.release()}\n")
        f.write(f"对局数:   {args.n_matches}\n")
        f.write(f"目标分:   {args.target}\n")
        f.write(f"种子起点: {args.seed}\n\n")
        f.write(f"参赛 agent:\n")
        for i, n in enumerate(args.agents):
            f.write(f"  P{i}: {n}\n")
        f.write("\n")

        for i in range(args.n_matches):
            state, winner = run_match(args.agents, seed=args.seed + i)
            wins[winner] += 1
            sum_rounds += state.round_number
            final = [p.total_score for p in state.players]
            for j, s in enumerate(final):
                sum_scores[j] += s
            # 排名
            rank = sorted(range(4), key=lambda j: -final[j])
            for pos, idx in enumerate(rank):
                placement_counts[idx][pos] += 1
            # 计数事件
            for line in state.log:
                if "BUST on" in line:
                    m = line.split("|P")[1].split(":")[0]
                    busts[int(m)] += 1
                elif "SIX-BURST" in line:
                    m = line.split("|P")[1].split(":")[0]
                    six_bursts[int(m)] += 1
                elif "FOLD" in line:
                    m = line.split("|P")[1].split(":")[0]
                    folds[int(m)] += 1

            if not args.brief:
                write_match(f, i, args.agents, state, winner)

            # 进度条（终端）
            if (i + 1) % max(1, args.n_matches // 20) == 0 or i + 1 == args.n_matches:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (args.n_matches - i - 1)
                print(f"  [{i+1:>4}/{args.n_matches}]  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

        # 汇总
        elapsed = time.time() - t0
        f.write(f"\n{'='*70}\n")
        f.write(f"汇总统计\n")
        f.write(f"{'='*70}\n")
        f.write(f"耗时: {elapsed:.1f}s ({args.n_matches/elapsed:.1f} 局/秒)\n")
        f.write(f"平均每场轮数: {sum_rounds/args.n_matches:.1f}\n\n")

        f.write(f"{'idx':<5}{'agent':<14}{'胜场':>6}{'胜率':>8}"
                f"{'1名':>6}{'2名':>6}{'3名':>6}{'4名':>6}"
                f"{'总分':>10}{'均分':>9}"
                f"{'爆牌':>6}{'6翻':>6}{'跑路':>6}\n")
        f.write("-" * 90 + "\n")
        for i in range(4):
            wp = wins[i] / args.n_matches * 100
            avg = sum_scores[i] / args.n_matches
            r1 = placement_counts[i][0]
            r2 = placement_counts[i][1]
            r3 = placement_counts[i][2]
            r4 = placement_counts[i][3]
            f.write(f"P{i:<4}{args.agents[i]:<14}{wins[i]:>6}{wp:>7.1f}%"
                    f"{r1:>6}{r2:>6}{r3:>6}{r4:>6}"
                    f"{sum_scores[i]:>10}{avg:>9.1f}"
                    f"{busts[i]:>6}{six_bursts[i]:>6}{folds[i]:>6}\n")

    # 终端也打印一份汇总
    print()
    print("=" * 70)
    print(f"{'idx':<5}{'agent':<14}{'胜场':>6}{'胜率':>8}{'1名':>6}{'2名':>6}"
          f"{'3名':>6}{'4名':>6}{'总分':>10}{'均分':>9}")
    print("-" * 70)
    for i in range(4):
        wp = wins[i] / args.n_matches * 100
        avg = sum_scores[i] / args.n_matches
        r1 = placement_counts[i][0]
        r2 = placement_counts[i][1]
        r3 = placement_counts[i][2]
        r4 = placement_counts[i][3]
        print(f"P{i:<4}{args.agents[i]:<14}{wins[i]:>6}{wp:>7.1f}%"
              f"{r1:>6}{r2:>6}{r3:>6}{r4:>6}"
              f"{sum_scores[i]:>10}{avg:>9.1f}")
    print("=" * 70)
    print(f"耗时 {elapsed:.1f}s · 详细日志已保存到 {out_path}")


if __name__ == "__main__":
    main()
