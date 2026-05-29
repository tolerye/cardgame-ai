"""隔夜训练编排：
- 每 EVAL_EVERY iter 跑一次 eval（200 局 raw policy）
- 每次自动保存 checkpoint
- 维护 model_best.pt（raw 胜率最高的版本）
- 连续 2 次 raw 胜率低于 baseline-3pp 时自动早停 + 回滚到 best

CLI 示例:
    python3 train_overnight.py                          # 默认 300 iter × 120 game
    python3 train_overnight.py --iters 800 --games 200  # 更多游戏
    python3 train_overnight.py --no-resume              # 从随机权重起步
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# 默认值（被 argparse 覆盖）
DEFAULTS = {
    "iters": 300,
    "eval_every": 30,
    "games": 120,
    "n_sims": 80,
    "workers": 6,
    "eval_matches": 200,
    "baseline_raw": 21.5,
    "early_stop_drop": 3.0,
}

REPO = Path(__file__).resolve().parent
CKPT_DIR = REPO / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Windows: 强制 stdout/stderr 用 UTF-8，否则 emoji 报 GBK 编码错
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        # 兜底：剥掉 non-ASCII（emoji 等）
        safe = msg.encode('ascii', 'replace').decode('ascii')
        print(f"[{ts}] {safe}", flush=True)


def run(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    if capture:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    r = subprocess.run(cmd, cwd=REPO)
    return r.returncode, ""


def extract_neural_pct(eval_output: str) -> float | None:
    m = re.search(r"\bneural\s+\d+\s+(\d+\.\d+)%", eval_output)
    return float(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight training orchestrator")
    parser.add_argument("--iters", type=int, default=DEFAULTS["iters"], help="总 iter 数")
    parser.add_argument("--eval-every", type=int, default=DEFAULTS["eval_every"], help="每 N iter 评测一次")
    parser.add_argument("--games", type=int, default=DEFAULTS["games"], help="每 iter 的 self-play game 数")
    parser.add_argument("--n-sims", type=int, default=DEFAULTS["n_sims"], help="每次决策的 MCTS 模拟数")
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"], help="并行 worker 数")
    parser.add_argument("--eval-matches", type=int, default=DEFAULTS["eval_matches"], help="每次 eval 跑多少局")
    parser.add_argument("--baseline-raw", type=float, default=DEFAULTS["baseline_raw"],
                        help="raw 胜率基线 %（早停参考）")
    parser.add_argument("--early-stop-drop", type=float, default=DEFAULTS["early_stop_drop"],
                        help="低于 baseline 多少 pp 算退化")
    parser.add_argument("--no-resume", action="store_true", help="从随机权重起步")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--vectorized", type=int, default=0,
                        help="vectorized 模式：N 个并发 game。GPU 训练推荐 16/32/64")
    parser.add_argument("--out", default="model.pt", help="模型输出路径")
    parser.add_argument("--ckpt-prefix", default="model_iter", help="checkpoint 前缀")
    parser.add_argument("--hidden", type=int, default=128, help="网络隐藏层维度")
    parser.add_argument("--n-layers", type=int, default=3, help="trunk 层数")
    parser.add_argument("--adversarial", action="store_true",
                        help="对抗微调：1 NN + 3 exmax3")
    parser.add_argument("--reward-shape", default='asym',
                        choices=['asym', 'sym', 'binary', 'margin'],
                        help="value target 形状")
    args = parser.parse_args()

    log(f"start: {args.iters} iter @ {args.games} game/iter, {args.n_sims} sims, {args.workers} workers")
    log(f"eval every {args.eval_every} iter, baseline={args.baseline_raw}%")

    # 自动判断是否 resume：no-resume 显式传 OR model.pt 不存在 → 从头起步
    if args.no_resume or not (REPO / args.out).exists():
        args.no_resume = True
        log(f"starting from scratch (no {args.out} found, or --no-resume)")
    else:
        shutil.copy(REPO / args.out, CKPT_DIR / "model_overnight_start.pt")
        shutil.copy(REPO / args.out, CKPT_DIR / "model_best.pt")

    best_raw = args.baseline_raw
    low_streak = 0
    history: list[tuple[int, float]] = []

    n_batches = args.iters // args.eval_every
    for batch in range(n_batches):
        iter_done = (batch + 1) * args.eval_every
        iter_from = batch * args.eval_every + 1
        log(f"━━━━━━━━ Batch {batch+1}/{n_batches}: training iter {iter_from} → {iter_done} ━━━━━━━━")

        cmd = [
            "python3", "-m", "train.selfplay",
            "--iters", str(args.eval_every),
            "--games-per-iter", str(args.games),
            "--workers", str(args.workers),
            "--n-sims", str(args.n_sims),
            "--device", args.device,
            "--vectorized", str(args.vectorized),
            "--hidden", str(args.hidden),
            "--n-layers", str(args.n_layers),
            "--reward-shape", args.reward_shape,
            "--out", args.out,
        ]
        if args.adversarial:
            cmd.append("--adversarial")
        # 第一批 + no-resume 时不传 --resume；之后每批都从上次的 out 续训
        if not (batch == 0 and args.no_resume):
            cmd += ["--resume", args.out]

        rc, _ = run(cmd)
        if rc != 0:
            log(f"⚠ training subprocess failed (rc={rc})")
            break

        shutil.copy(REPO / args.out, CKPT_DIR / f"{args.ckpt_prefix}_{iter_done}.pt")
        log(f"saved checkpoint: checkpoints/{args.ckpt_prefix}_{iter_done}.pt")

        rc, output = run([
            "python3", "eval_neural.py",
            "--model", args.out,
            "-n", str(args.eval_matches),
            "--workers", str(args.workers),
        ], capture=True)
        if rc != 0:
            log(f"⚠ eval failed (rc={rc}): {output[:500]}")
            continue

        raw_pct = extract_neural_pct(output)
        if raw_pct is None:
            log(f"⚠ failed to parse eval output:\n{output[:500]}")
            continue
        history.append((iter_done, raw_pct))
        log(f"📊 iter {iter_done}: raw policy = {raw_pct:.1f}%  (baseline {args.baseline_raw}%)")

        if raw_pct > best_raw:
            best_raw = raw_pct
            shutil.copy(REPO / args.out, CKPT_DIR / "model_best.pt")
            log(f"  ✓ new best: {raw_pct:.1f}% → checkpoints/model_best.pt")

        if raw_pct < args.baseline_raw - args.early_stop_drop:
            low_streak += 1
            log(f"  ⚠ retrograde streak {low_streak}/2")
            if low_streak >= 2:
                log(f"🛑 early stop. rolling back to best ({best_raw:.1f}%)")
                shutil.copy(CKPT_DIR / "model_best.pt", REPO / args.out)
                break
        else:
            low_streak = 0

    log("")
    log("━━━━━━━━━━━━━━━━ DONE ━━━━━━━━━━━━━━━━")
    log(f"best raw policy: {best_raw:.1f}%")
    log("history:")
    for it, pct in history:
        marker = " ←" if pct == best_raw else ""
        log(f"  iter {it:>4}  raw={pct:>5.1f}%{marker}")
    log("checkpoints saved in checkpoints/")


if __name__ == "__main__":
    main()

