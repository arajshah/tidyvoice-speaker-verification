#!/usr/bin/env python3
# fuse_scores.py
#
# Score-level fusion with Tune-S calibration:
# - Loads labeled Tune-S trials (utt1 utt2 label)
# - Loads S1/S2 Tune-S scores (utt1 utt2 score)
# - Z-norms each system using Tune-S score mean/std
# - Tunes alpha on Tune-S via grid search (default 0..1 step 0.01) to minimize EER (or minDCF)
# - Applies frozen (mu/sigma + alpha) to eval score files (A/U) and writes fused scores
# - Writes fusion params JSON for reproducibility

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


Pair = Tuple[str, str]


def read_trials_with_labels(path: Path) -> List[Tuple[str, str, int]]:
    """
    Expected format (3 cols):
      utt1 utt2 target|nontarget
    Also supports:
      utt1 utt2 1|0|true|false
    """
    trials = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{ln}: expected 3 columns, got: {line}")
            u1, u2, lab = parts[0], parts[1], parts[2].lower()
            if lab in ("target", "1", "true", "t", "yes", "y"):
                y = 1
            elif lab in ("nontarget", "non-target", "0", "false", "f", "no", "n"):
                y = 0
            else:
                # try numeric
                try:
                    y = 1 if float(lab) > 0 else 0
                except Exception:
                    raise ValueError(f"{path}:{ln}: unrecognized label '{parts[2]}'")
            trials.append((u1, u2, y))
    if not trials:
        raise ValueError(f"No trials read from {path}")
    return trials


def read_scores(path: Path) -> Dict[Pair, float]:
    """
    Expected format (3 cols):
      utt1 utt2 score
    """
    scores: Dict[Pair, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{ln}: expected 3 columns, got: {line}")
            u1, u2 = parts[0], parts[1]
            try:
                s = float(parts[2])
            except Exception:
                raise ValueError(f"{path}:{ln}: bad score '{parts[2]}'")
            scores[(u1, u2)] = s
    if not scores:
        raise ValueError(f"No scores read from {path}")
    return scores


def join_tune(trials: List[Tuple[str, str, int]],
              s1: Dict[Pair, float],
              s2: Dict[Pair, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.empty(len(trials), dtype=np.int32)
    a = np.empty(len(trials), dtype=np.float64)
    b = np.empty(len(trials), dtype=np.float64)

    missing = 0
    for i, (u1, u2, lab) in enumerate(trials):
        key = (u1, u2)
        if key not in s1 or key not in s2:
            missing += 1
            if missing <= 10:
                print(f"[missing] {key} in {'S1' if key not in s1 else ''} {'S2' if key not in s2 else ''}")
            continue
        y[i] = lab
        a[i] = s1[key]
        b[i] = s2[key]

    if missing:
        raise RuntimeError(f"Missing {missing} Tune-S pairs in one/both score files (see first few above).")
    return y, a, b


def z_norm(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= 0.0 or not np.isfinite(sigma):
        sigma = 1.0
    return (x - mu) / sigma, mu, sigma


def compute_eer(y: np.ndarray, scores: np.ndarray) -> float:
    """
    EER computed from scores where higher = more target.
    Returns EER in [0,1].
    """
    # sort by score descending
    order = np.argsort(-scores)
    y = y[order]
    scores = scores[order]

    # counts
    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        raise ValueError("Need both target and nontarget trials to compute EER")

    # sweep threshold from +inf downwards
    # When we include more items as "accept", FA increases, FR decreases.
    # At each k, accept top-k.
    tar_accept = np.cumsum(y == 1)
    non_accept = np.cumsum(y == 0)

    FRR = (P - tar_accept) / P  # miss rate
    FAR = non_accept / N        # false alarm rate

    # find crossing point
    diff = FAR - FRR
    idx = np.where(diff >= 0)[0]
    if len(idx) == 0:
        return float(FAR[-1])  # never crosses; degenerate
    i = int(idx[0])
    if i == 0:
        return float((FAR[0] + FRR[0]) / 2.0)

    # linear interpolate between i-1 and i
    x0, y0 = diff[i - 1], (FAR[i - 1] + FRR[i - 1]) / 2.0
    x1, y1v = diff[i], (FAR[i] + FRR[i]) / 2.0
    if x1 == x0:
        return float(y1v)
    t = (0.0 - x0) / (x1 - x0)
    eer = y0 + t * (y1v - y0)
    return float(eer)


def compute_min_dcf(y: np.ndarray, scores: np.ndarray,
                    p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    """
    Standard minDCF sweep over thresholds.
    """
    order = np.argsort(-scores)
    y = y[order]
    scores = scores[order]

    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        raise ValueError("Need both target and nontarget trials to compute minDCF")

    tar_accept = np.cumsum(y == 1)
    non_accept = np.cumsum(y == 0)

    pmiss = (P - tar_accept) / P
    pfa = non_accept / N

    dcf = c_miss * p_target * pmiss + c_fa * (1.0 - p_target) * pfa
    return float(np.min(dcf))


def tune_alpha(y: np.ndarray, s1n: np.ndarray, s2n: np.ndarray,
               alpha_min: float, alpha_max: float, alpha_step: float,
               objective: str, p_target: float) -> Tuple[float, float]:
    best_a = None
    best_val = None
    alphas = np.arange(alpha_min, alpha_max + 1e-12, alpha_step, dtype=np.float64)
    for a in alphas:
        fused = a * s1n + (1.0 - a) * s2n
        if objective == "eer":
            val = compute_eer(y, fused)
        else:
            val = compute_min_dcf(y, fused, p_target=p_target)
        if best_val is None or val < best_val:
            best_val = val
            best_a = float(a)
    assert best_a is not None and best_val is not None
    return best_a, float(best_val)


def fuse_eval_file(s1_path: Path, s2_path: Path, out_path: Path,
                   mu1: float, sig1: float, mu2: float, sig2: float,
                   alpha: float) -> None:
    s2 = read_scores(s2_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with s1_path.open("r", encoding="utf-8") as f1, out_path.open("w", encoding="utf-8") as fo:
        for ln, line in enumerate(f1, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{s1_path}:{ln}: expected 3 columns, got: {line}")
            u1, u2 = parts[0], parts[1]
            s1_raw = float(parts[2])
            key = (u1, u2)
            if key not in s2:
                raise RuntimeError(f"Missing pair {key} in S2 score file: {s2_path}")
            s2_raw = float(s2[key])

            s1n = (s1_raw - mu1) / (sig1 if sig1 != 0 else 1.0)
            s2n = (s2_raw - mu2) / (sig2 if sig2 != 0 else 1.0)
            sf = alpha * s1n + (1.0 - alpha) * s2n
            fo.write(f"{u1} {u2} {sf:.10f}\n")
            wrote += 1
    print(f"Wrote {wrote} fused scores -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune_trials", type=Path, required=True,
                    help="Tune-S labeled trials, e.g. data/tune_s/trials/trials.kaldi")
    ap.add_argument("--tune_s1", type=Path, required=True, help="Tune-S S1 scores (utt1 utt2 score)")
    ap.add_argument("--tune_s2", type=Path, required=True, help="Tune-S S2 scores (utt1 utt2 score)")

    ap.add_argument("--evalA_s1", type=Path, required=True, help="Eval-A S1 scores")
    ap.add_argument("--evalA_s2", type=Path, required=True, help="Eval-A S2 scores")
    ap.add_argument("--evalU_s1", type=Path, required=True, help="Eval-U S1 scores")
    ap.add_argument("--evalU_s2", type=Path, required=True, help="Eval-U S2 scores")

    ap.add_argument("--outA", type=Path, required=True, help="Output fused Eval-A score file")
    ap.add_argument("--outU", type=Path, required=True, help="Output fused Eval-U score file")
    ap.add_argument("--out_params", type=Path, required=True, help="Output JSON with frozen fusion params")

    ap.add_argument("--alpha_min", type=float, default=0.0)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--alpha_step", type=float, default=0.01)

    ap.add_argument("--objective", type=str, default="eer", choices=["eer", "mindcf"])
    ap.add_argument("--p_target", type=float, default=0.01, help="Used only for minDCF objective")

    args = ap.parse_args()

    trials = read_trials_with_labels(args.tune_trials)
    s1_tune = read_scores(args.tune_s1)
    s2_tune = read_scores(args.tune_s2)

    y, s1_raw, s2_raw = join_tune(trials, s1_tune, s2_tune)

    s1n, mu1, sig1 = z_norm(s1_raw)
    s2n, mu2, sig2 = z_norm(s2_raw)

    alpha, best = tune_alpha(
        y=y,
        s1n=s1n,
        s2n=s2n,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        objective=args.objective,
        p_target=args.p_target,
    )

    if args.objective == "eer":
        print(f"Best alpha={alpha:.4f} with Tune-S EER={best*100:.3f}% (z-normed)")
    else:
        print(f"Best alpha={alpha:.4f} with Tune-S minDCF={best:.6f} (p_target={args.p_target}) (z-normed)")

    params = {
        "objective": args.objective,
        "alpha": alpha,
        "alpha_grid": {"min": args.alpha_min, "max": args.alpha_max, "step": args.alpha_step},
        "z_norm": {
            "s1": {"mu": mu1, "sigma": sig1, "source": str(args.tune_s1)},
            "s2": {"mu": mu2, "sigma": sig2, "source": str(args.tune_s2)},
        },
        "tune_trials": str(args.tune_trials),
    }
    args.out_params.parent.mkdir(parents=True, exist_ok=True)
    args.out_params.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote frozen fusion params -> {args.out_params}")

    # Apply frozen params to eval
    fuse_eval_file(args.evalA_s1, args.evalA_s2, args.outA, mu1, sig1, mu2, sig2, alpha)
    fuse_eval_file(args.evalU_s1, args.evalU_s2, args.outU, mu1, sig1, mu2, sig2, alpha)


if __name__ == "__main__":
    main()
