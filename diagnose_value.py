"""Value-head health check.

Plays N matches, recording the network's value prediction at every decision
point alongside (a) the binary winner outcome z=±1 and (b) the placement
outcome (1st/2nd/3rd/4th → +1/+1/3/−1/3/−1).

Reports:
  - mean / std of predictions
  - correlation with binary z and with placement z
  - directional accuracy (does sign of pred match sign of outcome?)
  - leader-state sanity: when player is clearly ahead, does pred > 0?

If correlations are near zero and pred std is tiny, the value head has not
learned and is dominating the constant-baseline regime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from agents import EVAgent, ExpectimaxAgent, GreedyAgent
from agents.batched_mcts import BatchedNeuralMCTSAgent
from game import GameConfig, GameEngine
from train.encoder import encode_state
from train.network import load


@dataclass
class Sample:
    pred: float
    z_binary: float       # +1 if I won match, else -1
    z_placement: float    # 1st: +1, 2nd: +1/3, 3rd: -1/3, 4th: -1
    my_score_now: int     # for leader-state analysis
    others_max_now: int


def _placement_z(my_idx: int, totals: List[int]) -> float:
    rank = sorted(range(len(totals)), key=lambda i: -totals[i])  # desc
    pos = rank.index(my_idx)
    return [1.0, 1/3, -1/3, -1.0][pos]


def collect(model, n_matches: int = 30) -> List[Sample]:
    samples: List[Sample] = []
    for s in range(n_matches):
        cfg = GameConfig(num_players=4, seed=s)
        # use 4 EVAgents to drive the games; we just probe NN at each state
        agents = [EVAgent() for _ in range(4)]
        engine = GameEngine(cfg, agents)

        # Hook take_turn to record state/pred for active player
        snapshots = []  # (state_features per player, my_idx)
        orig_take = engine._take_turn

        def hook(idx: int):
            if engine.state.players[idx].is_active:
                me = engine.state.players[idx]
                others_max = max(p.total_score + p.current_round_score()
                                 for p in engine.state.players if p.index != idx)
                feats = encode_state(engine.state, idx)
                snapshots.append((feats, idx,
                                  me.total_score + me.current_round_score(),
                                  others_max))
            orig_take(idx)

        engine._take_turn = hook
        engine.play_match()
        winner = engine.state.winner
        totals = [p.total_score for p in engine.state.players]

        # Predict in batch
        if not snapshots:
            continue
        X = np.stack([f for f, _, _, _ in snapshots])
        with torch.no_grad():
            _, vs = model(torch.from_numpy(X))
        vs = vs.numpy()

        for (_, idx, mine, others), v in zip(snapshots, vs):
            samples.append(Sample(
                pred=float(v),
                z_binary=1.0 if idx == winner else -1.0,
                z_placement=_placement_z(idx, totals),
                my_score_now=mine,
                others_max_now=others,
            ))
    return samples


def report(samples: List[Sample]) -> None:
    preds = np.array([s.pred for s in samples])
    zb = np.array([s.z_binary for s in samples])
    zp = np.array([s.z_placement for s in samples])
    ahead = np.array([s.my_score_now - s.others_max_now for s in samples])

    print(f"\n=== Value-head diagnosis on {len(samples)} samples ===")
    print(f"\n[Prediction distribution]")
    print(f"  pred  mean={preds.mean():+.3f}  std={preds.std():.3f}  "
          f"min={preds.min():+.3f}  max={preds.max():+.3f}")
    print(f"  z_binary    mean={zb.mean():+.3f}  std={zb.std():.3f}")
    print(f"  z_placement mean={zp.mean():+.3f}  std={zp.std():.3f}")

    print(f"\n[Correlations]")
    if preds.std() > 1e-6:
        c_b = np.corrcoef(preds, zb)[0, 1]
        c_p = np.corrcoef(preds, zp)[0, 1]
        print(f"  corr(pred, z_binary)    = {c_b:+.3f}")
        print(f"  corr(pred, z_placement) = {c_p:+.3f}")
    else:
        print("  pred is constant — no correlation defined")

    # MSE comparison vs constant baselines
    const_b = np.mean(zb)
    const_p = np.mean(zp)
    mse_b = ((preds - zb) ** 2).mean()
    mse_p = ((preds - zp) ** 2).mean()
    mse_const_b = ((const_b - zb) ** 2).mean()
    mse_const_p = ((const_p - zp) ** 2).mean()
    print(f"\n[MSE vs targets]")
    print(f"  pred vs z_binary    : {mse_b:.3f}   (constant = {mse_const_b:.3f})")
    print(f"  pred vs z_placement : {mse_p:.3f}   (constant = {mse_const_p:.3f})")
    if mse_b > mse_const_b * 0.95:
        print("  ⚠ value head essentially predicts the constant — no signal learned")
    elif mse_b < mse_const_b * 0.7:
        print("  ✓ value head meaningfully better than constant baseline")

    print(f"\n[Directional accuracy (sign match)]")
    nz = preds != 0
    if nz.any():
        sign_b = (np.sign(preds[nz]) == np.sign(zb[nz])).mean()
        sign_p = (np.sign(preds[nz]) == np.sign(zp[nz])).mean()
        print(f"  sign(pred)==sign(z_binary)    : {sign_b*100:.1f}%  (random = 50%)")
        print(f"  sign(pred)==sign(z_placement) : {sign_p*100:.1f}%  (random ~ 50%)")

    print(f"\n[Leader-state sanity: when score margin > +30, is pred > 0?]")
    leading = ahead > 30
    losing = ahead < -30
    if leading.any():
        pl = preds[leading]
        print(f"  ahead by 30+ ({leading.sum()} samples): pred mean = {pl.mean():+.3f},  "
              f"% pred > 0 = {(pl > 0).mean()*100:.1f}%   (healthy: high)")
    if losing.any():
        pl = preds[losing]
        print(f"  behind by 30+ ({losing.sum()} samples): pred mean = {pl.mean():+.3f}, "
              f"% pred < 0 = {(pl < 0).mean()*100:.1f}%   (healthy: high)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--n-matches", type=int, default=30)
    args = parser.parse_args()
    model = load(args.model)
    samples = collect(model, n_matches=args.n_matches)
    report(samples)


if __name__ == "__main__":
    main()
