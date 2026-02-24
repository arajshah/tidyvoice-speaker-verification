# File: examples/tidyvocie/tools/score_fast.py
#!/usr/bin/env python3
"""
Fast cosine scoring + EER/minDCF for Kaldi-style trials using pre-extracted embeddings.

- Loads embeddings from a Kaldi .scp (xvector.scp) into memory.
- Optional mean-subtraction (cal_mean) using a cohort scp (e.g., embeddings/vox2_dev/xvector.scp).
- L2-normalizes embeddings, then scores trials in vectorized chunks.
- If trials are labeled (3rd column target/nontarget), computes EER + minDCF in-memory.
- Optionally writes the full .score file (slow for 12M trials); default is no write.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import kaldiio


def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "t", "yes", "y")


def load_mean_vec_from_scp(scp_path: Path) -> np.ndarray:
    s = np.zeros((1,), dtype=np.float64)
    n = 0
    mean = None
    for _, v in kaldiio.load_scp_sequential(str(scp_path)):
        v = np.asarray(v, dtype=np.float64)
        if mean is None:
            mean = np.zeros_like(v, dtype=np.float64)
        mean += v
        n += 1
    if mean is None or n == 0:
        raise RuntimeError(f"Mean calc failed: no vectors in {scp_path}")
    mean /= float(n)
    return mean.astype(np.float32)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # mat: (N,D)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def load_embeddings(
    scp_path: Path,
    cal_mean: bool,
    cal_mean_scp: Optional[Path],
) -> Tuple[np.ndarray, Dict[str, int]]:
    # Load all vectors into a matrix for fast indexed access
    keys: List[str] = []
    vecs: List[np.ndarray] = []
    for k, v in kaldiio.load_scp_sequential(str(scp_path)):
        keys.append(k)
        vecs.append(np.asarray(v, dtype=np.float32))

    if not vecs:
        raise RuntimeError(f"No embeddings found in {scp_path}")

    X = np.stack(vecs, axis=0)  # (N,D)
    key2i = {k: i for i, k in enumerate(keys)}

    if cal_mean:
        if cal_mean_scp is None:
            raise ValueError("cal_mean=True but no cal_mean_scp provided")
        mu = load_mean_vec_from_scp(cal_mean_scp)
        if mu.shape[0] != X.shape[1]:
            raise RuntimeError(f"Mean dim mismatch: mean {mu.shape} vs emb dim {X.shape}")
        X = X - mu[None, :]

    X = l2_normalize(X)
    return X, key2i


def parse_label(tok: str) -> int:
    t = tok.strip().lower()
    if t in ("target", "1", "true", "t", "yes", "y"):
        return 1
    if t in ("nontarget", "non-target", "0", "false", "f", "no", "n"):
        return 0
    # numeric fallback
    try:
        return 1 if float(t) > 0 else 0
    except Exception as e:
        raise ValueError(f"Unrecognized label token: {tok}") from e


def compute_eer(y: np.ndarray, scores: np.ndarray) -> float:
    # y in {0,1}, higher score = more target
    order = np.argsort(-scores)
    y = y[order]
    scores = scores[order]

    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        raise ValueError("Need both target and nontarget to compute EER")

    tar_accept = np.cumsum(y == 1)
    non_accept = np.cumsum(y == 0)

    frr = (P - tar_accept) / P
    far = non_accept / N

    diff = far - frr
    idx = np.where(diff >= 0)[0]
    if len(idx) == 0:
        return float(far[-1])

    i = int(idx[0])
    if i == 0:
        return float((far[0] + frr[0]) / 2.0)

    x0 = diff[i - 1]
    x1 = diff[i]
    y0 = (far[i - 1] + frr[i - 1]) / 2.0
    y1 = (far[i] + frr[i]) / 2.0
    if x1 == x0:
        return float(y1)
    t = (0.0 - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def compute_min_dcf(
    y: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    order = np.argsort(-scores)
    y = y[order]

    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        raise ValueError("Need both target and nontarget to compute minDCF")

    tar_accept = np.cumsum(y == 1)
    non_accept = np.cumsum(y == 0)

    pmiss = (P - tar_accept) / P
    pfa = non_accept / N

    dcf = c_miss * p_target * pmiss + c_fa * (1.0 - p_target) * pfa
    return float(np.min(dcf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_scp_path", type=Path, required=True,
                    help="Path to xvector.scp for eval set (e.g., exp/.../embeddings/tidyvoice_dev/xvector.scp)")
    ap.add_argument("--trials_path", type=Path, required=True,
                    help="Trials file. Supports 2-col (utt1 utt2) or 3-col (utt1 utt2 label)")
    ap.add_argument("--out_score_path", type=Path, default=None,
                    help="Optional output score file path (utt1 utt2 score). If omitted, no score file unless --write_scores True.")
    ap.add_argument("--write_scores", type=str, default="False",
                    help="If True, write full score file (can be large and slow).")
    ap.add_argument("--chunk_size", type=int, default=200000,
                    help="Trials processed per chunk for vectorized scoring.")
    ap.add_argument("--missing_policy", type=str, default="error", choices=["error", "skip"],
                    help="What to do if an utterance id is missing in embeddings.")
    ap.add_argument("--cal_mean", type=str, default="False",
                    help="Mean-subtract embeddings using --cal_mean_scp before L2 norm.")
    ap.add_argument("--cal_mean_scp", type=Path, default=None,
                    help="Cohort xvector.scp used to compute mean vector (e.g., exp/.../embeddings/vox2_dev/xvector.scp)")

    ap.add_argument("--p_target", type=float, default=0.01)
    ap.add_argument("--c_miss", type=float, default=1.0)
    ap.add_argument("--c_fa", type=float, default=1.0)

    args = ap.parse_args()

    write_scores = str2bool(args.write_scores)
    cal_mean = str2bool(args.cal_mean)

    X, key2i = load_embeddings(args.eval_scp_path, cal_mean=cal_mean, cal_mean_scp=args.cal_mean_scp)

    # Buffers
    scores_chunks: List[np.ndarray] = []
    labels_chunks: List[np.ndarray] = []
    labeled = False

    # Optional writer
    fout = None
    if write_scores:
        if args.out_score_path is None:
            # default: trialsname.score in same dir as trials
            args.out_score_path = args.trials_path.with_suffix(args.trials_path.suffix + ".score")
        args.out_score_path.parent.mkdir(parents=True, exist_ok=True)
        fout = args.out_score_path.open("w", encoding="utf-8")

    buf_i1: List[int] = []
    buf_i2: List[int] = []
    buf_y: List[int] = []
    buf_u1: List[str] = []
    buf_u2: List[str] = []

    missing = 0
    total = 0

    def flush():
        nonlocal buf_i1, buf_i2, buf_y, buf_u1, buf_u2, labeled
        if not buf_i1:
            return
        i1 = np.asarray(buf_i1, dtype=np.int32)
        i2 = np.asarray(buf_i2, dtype=np.int32)
        # cosine = dot since vectors are L2-normalized
        s = np.sum(X[i1] * X[i2], axis=1).astype(np.float32)

        scores_chunks.append(s)

        if buf_y:
            labeled = True
            labels_chunks.append(np.asarray(buf_y, dtype=np.int8))

        if fout is not None:
            # write utt1 utt2 score
            for a, b, sc in zip(buf_u1, buf_u2, s.tolist()):
                fout.write(f"{a} {b} {sc:.10f}\n")

        buf_i1, buf_i2, buf_y, buf_u1, buf_u2 = [], [], [], [], []

    with args.trials_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split()
            if len(parts) < 2:
                continue
            u1, u2 = parts[0], parts[1]
            y = None
            if len(parts) >= 3:
                y = parse_label(parts[2])

            k1 = key2i.get(u1)
            k2 = key2i.get(u2)
            if k1 is None or k2 is None:
                missing += 1
                if args.missing_policy == "error":
                    raise RuntimeError(f"Missing embedding for trial ({u1}, {u2}). Missing={missing}")
                else:
                    continue

            buf_i1.append(k1)
            buf_i2.append(k2)
            if y is not None:
                buf_y.append(y)
            if fout is not None:
                buf_u1.append(u1)
                buf_u2.append(u2)

            if len(buf_i1) >= args.chunk_size:
                flush()

    flush()
    if fout is not None:
        fout.close()

    scores = np.concatenate(scores_chunks, axis=0) if scores_chunks else np.zeros((0,), dtype=np.float32)

    # Print in the same style your logs expect
    if args.out_score_path is not None:
        print(f"----- {args.out_score_path.name} -----")
    else:
        print("----- trials.score (in-memory) -----")

    if labeled:
        y = np.concatenate(labels_chunks, axis=0).astype(np.int32)
        eer = compute_eer(y, scores) * 100.0
        mindcf = compute_min_dcf(y, scores, p_target=args.p_target, c_miss=args.c_miss, c_fa=args.c_fa)
        print(f"EER = {eer:.3f}")
        print(f"minDCF (p_target:{args.p_target} c_miss:{args.c_miss} c_fa:{args.c_fa}) = {mindcf:.3f}")
    else:
        print(f"Scored {scores.shape[0]} trials (unlabeled).")

    if missing:
        print(f"NOTE: missing trials skipped={missing} (policy={args.missing_policy})")


if __name__ == "__main__":
    main()