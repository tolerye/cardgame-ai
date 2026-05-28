"""隔夜训练编排：
- 每 EVAL_EVERY iter 跑一次 eval（200 局 raw policy）
- 每次自动保存 checkpoint
- 维护 model_best.pt（raw 胜率最高的版本）
- 连续 2 次 raw 胜率低于 baseline-3pp 时自动早停 + 回滚到 best
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


MAX_ITERS = 300
EVAL_EVERY = 30
GAMES_PER_ITER = 120
N_SIMS = 80
WORKERS = 6
EVAL_MATCHES = 200
BASELINE_RAW = 21.5      # v5 model raw policy 胜率
EARLY_STOP_DROP = 3.0    # raw < baseline - 3pp 算退化

REPO = Path("/Users/tolerye/projects/cardgame-ai")
CKPT_DIR = REPO / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: list[str], capture: bool = False) -> tuple[int, str]:
    """Run subprocess; if capture, return stdout; else stream to current stdout."""
    if capture:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    r = subprocess.run(cmd, cwd=REPO)
    return r.returncode, ""


def extract_neural_pct(eval_output: str) -> float | None:
    m = re.search(r"\bneural\s+\d+\s+(\d+\.\d+)%", eval_output)
    return float(m.group(1)) if m else None


def main() -> None:
    log(f"start. baseline raw = {BASELINE_RAW}%, total {MAX_ITERS} iter, eval every {EVAL_EVERY}")

    # 备份起点
    shutil.copy(REPO / "model.pt", CKPT_DIR / "model_overnight_start.pt")
    shutil.copy(REPO / "model.pt", CKPT_DIR / "model_best.pt")

    best_raw = BASELINE_RAW
    low_streak = 0
    history: list[tuple[int, float]] = []

    n_batches = MAX_ITERS // EVAL_EVERY
    for batch in range(n_batches):
        iter_done = (batch + 1) * EVAL_EVERY
        iter_from = batch * EVAL_EVERY + 1
        log(f"━━━━━━━━ Batch {batch+1}/{n_batches}: training iter {iter_from} → {iter_done} ━━━━━━━━")

        rc, _ = run([
            "python3", "-m", "train.selfplay",
            "--resume", "model.pt",
            "--iters", str(EVAL_EVERY),
            "--games-per-iter", str(GAMES_PER_ITER),
            "--workers", str(WORKERS),
            "--n-sims", str(N_SIMS),
            "--out", "model.pt",
        ])
        if rc != 0:
            log(f"⚠ training subprocess failed (rc={rc})")
            break

        # 保存 checkpoint
        shutil.copy(REPO / "model.pt", CKPT_DIR / f"model_iter_{iter_done}.pt")
        log(f"saved checkpoint: checkpoints/model_iter_{iter_done}.pt")

        # Eval
        rc, output = run([
            "python3", "eval_neural.py",
            "--model", "model.pt",
            "-n", str(EVAL_MATCHES),
            "--workers", str(WORKERS),
        ], capture=True)
        if rc != 0:
            log(f"⚠ eval failed (rc={rc}): {output[:500]}")
            continue

        # 解析 neural raw 胜率
        raw_pct = extract_neural_pct(output)
        if raw_pct is None:
            log(f"⚠ failed to parse eval output:\n{output[:500]}")
            continue
        history.append((iter_done, raw_pct))
        log(f"📊 iter {iter_done}: raw policy = {raw_pct:.1f}%  (baseline {BASELINE_RAW}%)")

        # 维护 best
        if raw_pct > best_raw:
            best_raw = raw_pct
            shutil.copy(REPO / "model.pt", CKPT_DIR / "model_best.pt")
            log(f"  ✓ new best: {raw_pct:.1f}% → checkpoints/model_best.pt")

        # 早停判断
        if raw_pct < BASELINE_RAW - EARLY_STOP_DROP:
            low_streak += 1
            log(f"  ⚠ retrograde streak {low_streak}/2")
            if low_streak >= 2:
                log(f"🛑 early stop after retrograde streak. rolling back to best ({best_raw:.1f}%)")
                shutil.copy(CKPT_DIR / "model_best.pt", REPO / "model.pt")
                break
        else:
            low_streak = 0

    # 最终汇报
    log("")
    log("━━━━━━━━━━━━━━━━ DONE ━━━━━━━━━━━━━━━━")
    log(f"best raw policy: {best_raw:.1f}%")
    log("history:")
    for it, pct in history:
        marker = " ←" if pct == best_raw else ""
        log(f"  iter {it:>3}  raw={pct:>5.1f}%{marker}")
    log(f"final model: {'已回滚到 best' if low_streak >= 2 else '完整训练完成'}")
    log("checkpoints saved in checkpoints/")
    log("启动器：python3 train_overnight.py")


if __name__ == "__main__":
    main()
