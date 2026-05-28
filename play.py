"""Interactive terminal frontend for the card game.

You play as P0; the other three seats are AI agents (default: expectimax, ev,
greedy). Run:

    python3 play.py
    python3 play.py --opponents neural,expectimax,ev   # custom lineup
    python3 play.py --target 100                       # shorter match"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

from agents import EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent
from agents.base import BaseAgent
from game import GameConfig, GameEngine
from game.cards import CardKind
from game.state import GameState, PlayerStatus


# ANSI colors — disable on Windows or when piped to a non-tty.
USE_COLOR = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def C(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD = lambda t: C("1", t)
DIM = lambda t: C("2", t)
GREEN = lambda t: C("32", t)
YELLOW = lambda t: C("33", t)
RED = lambda t: C("31", t)
CYAN = lambda t: C("36", t)
BLUE = lambda t: C("34", t)
MAGENTA = lambda t: C("35", t)


# --------------------------------------------------------------- HumanAgent
class HumanAgent(BaseAgent):
    name = "human"

    def __init__(self) -> None:
        self._last_log_len = 0
        self._round_seen = 0

    # ----- main decision -----
    def choose_action(self, state: GameState, my_idx: int) -> str:
        self._catch_up_logs(state)
        self._render(state, my_idx)
        while True:
            try:
                raw = input(BOLD(f"\n  P{my_idx} (YOU) → action [d=draw, f=fold, q=quit]: ")).strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n  exiting...")
                sys.exit(0)
            if raw in ("d", "draw", ""):
                return "draw"
            if raw in ("f", "fold"):
                return "fold"
            if raw in ("q", "quit", "exit"):
                print("  bye.")
                sys.exit(0)
            print(RED("  ?  use 'd' or 'f'"))

    # ----- skill targeting -----
    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        self._catch_up_logs(state)
        candidates = [p for p in state.players if p.is_active and p.index != my_idx]
        if not candidates:
            return my_idx
        prompt = {
            CardKind.EXILE: "force this player to fold (lock their round score)",
            CardKind.TRIPLE: "force this player to draw 3 cards",
            CardKind.INSURANCE: "gift insurance to (you already have one)",
        }.get(kind, "target")
        print(MAGENTA(f"\n  ★ {kind.value.upper()}: {prompt}"))
        for i, p in enumerate(candidates):
            ins = " [INS]" if p.has_insurance else ""
            hand = " ".join(str(n) for n in sorted(p.hand_numbers))
            print(f"    {i}: P{p.index}  total={p.total_score:>3}  round={p.current_round_score():>3}  "
                  f"hand=[{hand}]{ins}")
        while True:
            try:
                raw = input("    pick [number]: ").strip()
                idx = int(raw)
                if 0 <= idx < len(candidates):
                    return candidates[idx].index
            except (ValueError, KeyboardInterrupt, EOFError):
                pass
            print(RED("    invalid index"))

    # ----- log replay -----
    def _catch_up_logs(self, state: GameState) -> None:
        new = state.log[self._last_log_len:]
        self._last_log_len = len(state.log)
        if state.round_number != self._round_seen:
            self._round_seen = state.round_number
            print()
            print(BOLD(CYAN(f"━━━━━━━━━━━━━━━━ Round {state.round_number} ━━━━━━━━━━━━━━━━")))
        for line in new:
            self._color_log(line)

    @staticmethod
    def _color_log(line: str) -> None:
        if "BUST" in line and "AVOIDED" not in line:
            print(f"  {RED(line)}")
        elif "SIX-BURST" in line:
            print(f"  {BOLD(YELLOW(line))}")
        elif "EXILE" in line or "TRIPLE" in line:
            print(f"  {MAGENTA(line)}")
        elif "INSURANCE" in line:
            print(f"  {BLUE(line)}")
        elif "BONUS" in line or "DOUBLE" in line:
            print(f"  {GREEN(line)}")
        elif "FOLD" in line:
            print(f"  {DIM(line)}")
        else:
            print(f"  {line}")

    # ----- table render -----
    def _render(self, state: GameState, my_idx: int) -> None:
        target = state.config.target_score
        print()
        print(DIM(f"  Target {target}.  Cards left in deck: {state.remaining.total()}"))
        print(DIM("  ─" * 32))
        for p in state.players:
            you = BOLD(GREEN(" ★ YOU")) if p.index == my_idx else "      "
            st = self._status_str(p.status)
            score_bar = self._bar(p.total_score, target, 20)
            line = (f"   P{p.index}{you}  {st}  "
                    f"total={p.total_score:>3} {score_bar}  round=+{p.current_round_score():>3}")
            print(line)
            if p.index == my_idx:
                hand_str = ", ".join(str(n) for n in sorted(p.hand_numbers)) or DIM("(empty)")
                ins = GREEN("YES") if p.has_insurance else DIM("no")
                bonus = f"+{p.bonus_flat_total}" if p.bonus_flat_total else "0"
                print(f"        hand: [{hand_str}]   bonus: {bonus}   insurance: {ins}")
                self._render_my_hint(state, p)
        print(DIM("  ─" * 32))

    @staticmethod
    def _status_str(status: PlayerStatus) -> str:
        s = status.value
        if status == PlayerStatus.ACTIVE:
            return GREEN(f"{s:<7}")
        if status == PlayerStatus.FOLDED:
            return DIM(f"{s:<7}")
        if status == PlayerStatus.BUSTED:
            return RED(f"{s:<7}")
        return MAGENTA(f"{s:<7}")

    @staticmethod
    def _bar(score: int, target: int, width: int) -> str:
        filled = min(width, int(score / target * width))
        return "[" + GREEN("█" * filled) + DIM("░" * (width - filled)) + "]"

    @staticmethod
    def _render_my_hint(state: GameState, me) -> None:
        """Tell the human what's risky."""
        rem = state.remaining
        total = max(rem.total(), 1)
        bust_count = sum(rem.numbers[v] for v in me.unique_numbers)
        bust_p = bust_count / total * 100
        cur = sum(me.hand_numbers) + me.bonus_flat_total
        unique = len(me.unique_numbers)

        hint_parts = []
        if me.total_score + cur >= state.config.target_score:
            hint_parts.append(BOLD(GREEN("Folding now WINS the match.")))
        else:
            color = GREEN if bust_p < 15 else YELLOW if bust_p < 30 else RED
            hint_parts.append(f"bust prob: {color(f'{bust_p:.0f}%')}")

            if unique == 5:
                safe_count = sum(c for v, c in rem.numbers.items() if v not in me.unique_numbers)
                safe_p = safe_count / total * 100
                hint_parts.append(BOLD(YELLOW(f"6-burst chance: {safe_p:.0f}%")))
            hint_parts.append(f"fold locks {cur}")

        print(f"        {DIM('hint:')} " + "   ".join(hint_parts))


# --------------------------------------------------------------- main
OPP_REGISTRY: Dict[str, callable] = {
    "expectimax": lambda: ExpectimaxAgent(),
    "ev": lambda: EVAgent(),
    "greedy": lambda: GreedyAgent(),
    "greedy20": lambda: GreedyAgent(fold_at=20),
    "random": lambda: RandomAgent(),
}


def _maybe_register_neural() -> None:
    try:
        from agents.neural_agent import NeuralAgent
        OPP_REGISTRY["neural"] = lambda: NeuralAgent("model.pt")
        OPP_REGISTRY["neural_mcts"] = lambda: NeuralAgent("model.pt", use_mcts=True, n_simulations=80)
    except Exception:
        pass


def main() -> None:
    _maybe_register_neural()
    parser = argparse.ArgumentParser(description="Play the card game in your terminal.")
    parser.add_argument("--opponents", default="expectimax,ev,greedy",
                        help="comma-separated AI opponents (3 total).")
    parser.add_argument("--target", type=int, default=200, help="match target score")
    parser.add_argument("--list", action="store_true", help="list available AI types and exit")
    args = parser.parse_args()

    if args.list:
        print("Available opponents:")
        for name in OPP_REGISTRY:
            print(f"  {name}")
        return

    names = [n.strip() for n in args.opponents.split(",")]
    if len(names) != 3:
        print(f"need exactly 3 opponents, got {len(names)}")
        sys.exit(1)
    for n in names:
        if n not in OPP_REGISTRY:
            print(f"unknown opponent '{n}' — try --list")
            sys.exit(1)

    print(BOLD(CYAN("━" * 60)))
    print(BOLD(CYAN("  Card Game — terminal edition")))
    print(BOLD(CYAN("━" * 60)))
    print(f"  Target score: {BOLD(str(args.target))}")
    print(f"  You         : {GREEN('P0 (HUMAN)')}")
    for i, n in enumerate(names, start=1):
        print(f"  Opponent P{i}: {n}")
    print()
    print(DIM("  Rules: draw to build hand of distinct numbers (max 6 = SIX BURST +15)."))
    print(DIM("         duplicate number = BUST (lose round score) unless you have insurance."))
    print(DIM("         fold any time to lock your round score into total."))
    print(DIM("         skill cards (insurance/exile/triple) trigger immediately."))
    print()

    cfg = GameConfig(num_players=4, target_score=args.target)
    agents = [HumanAgent()] + [OPP_REGISTRY[n]() for n in names]
    engine = GameEngine(cfg, agents)
    winner = engine.play_match()

    # final summary
    print()
    print(BOLD(CYAN("━━━━━━━━━━━━━━━━ MATCH OVER ━━━━━━━━━━━━━━━━")))
    rank = sorted(engine.state.players, key=lambda p: -p.total_score)
    medals = ["🥇", "🥈", "🥉", "  "]
    for medal, p in zip(medals, rank):
        you = BOLD(GREEN(" ★ YOU")) if p.index == 0 else ""
        agent_name = "human" if p.index == 0 else names[p.index - 1]
        print(f"  {medal}  P{p.index} ({agent_name}){you}: {BOLD(str(p.total_score))}")
    print()
    if winner == 0:
        print(BOLD(GREEN("  YOU WIN! 🎉")))
    else:
        print(BOLD(RED(f"  {names[winner - 1]} wins.")))


if __name__ == "__main__":
    main()
