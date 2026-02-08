#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os


DL_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
SPLIT_RE = re.compile(r"split_(?P<d>\d{8})_(?P<t>\d{6})")
TUNE_RUN_RE = re.compile(r"(?P<d>\d{8})_(?P<t>\d{6})")

def to_portable_wav_path(p: str, anchor: str) -> str:
    """
    Convert absolute wav paths to repo-portable relative paths by keeping the suffix
    starting at '/<anchor>/'.

    Example:
      /Users/araj/.../repo/data/raw_data/TidyVoiceX_ASV/... -> data/raw_data/TidyVoiceX_ASV/...
    """
    s = str(p).replace("\\", "/")
    if not s.startswith("/"):
        return s  # already relative

    anchor_norm = anchor.strip("/").replace("\\", "/")
    token = f"/{anchor_norm}/"
    idx = s.find(token)
    if idx == -1:
        raise RuntimeError(
            f"Cannot rewrite absolute path to portable form: '{p}'. "
            f"Expected to find '{token}' in the path. "
            f"Pass a different --wav_path_anchor (e.g., 'data', 'data/raw_data')."
        )
    # drop leading '/', keep "<anchor>/..."
    return s[idx + 1 :]



def _parse_dt_from_name(p: Path) -> Optional[datetime]:
    name = p.name
    m = DL_RE.search(name)
    if m:
        return datetime.strptime(m.group("ts"), "%Y-%m-%d_%H-%M-%S")
    m = SPLIT_RE.search(name)
    if m:
        return datetime.strptime(m.group("d") + m.group("t"), "%Y%m%d%H%M%S")
    m = TUNE_RUN_RE.search(name)
    if m:
        return datetime.strptime(m.group("d") + m.group("t"), "%Y%m%d%H%M%S")
    return None


def pick_latest_dir(parent: Path, must_contain: Optional[str] = None) -> Path:
    if not parent.exists():
        raise FileNotFoundError(f"Missing directory: {parent}")

    candidates = [d for d in parent.iterdir() if d.is_dir()]
    if must_contain:
        candidates = [d for d in candidates if must_contain in d.name]
    if not candidates:
        raise FileNotFoundError(f"No candidate directories found under: {parent}")

    def key_fn(d: Path) -> Tuple[int, float]:
        dt = _parse_dt_from_name(d)
        if dt:
            return (1, dt.timestamp())
        return (0, d.stat().st_mtime)

    return max(candidates, key=key_fn)

def read_lines(p: Path) -> List[str]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def read_kv_2col(p: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in read_lines(p):
        k, v = ln.split(maxsplit=1)
        out[k] = v
    return out


def read_utt2spk(p: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in read_lines(p):
        u, s = ln.split()
        out[u] = s
    return out


def write_wav_scp(p: Path, utts: List[str], wav_map: Dict[str, str]) -> None:
    with p.open("w") as f:
        for u in utts:
            f.write(f"{u} {wav_map[u]}\n")


def write_utt2x(p: Path, utts: List[str], m: Dict[str, str]) -> None:
    with p.open("w") as f:
        for u in utts:
            f.write(f"{u} {m[u]}\n")


def write_spk2utt(p: Path, utt2spk: Dict[str, str], utts: List[str]) -> None:
    spk2utts: Dict[str, List[str]] = {}
    for u in utts:
        s = utt2spk[u]
        spk2utts.setdefault(s, []).append(u)

    with p.open("w") as f:
        for spk in sorted(spk2utts.keys()):
            f.write(spk + " " + " ".join(spk2utts[spk]) + "\n")


def ensure_clean_dir(d: Path, force: bool) -> None:
    if d.exists():
        if any(d.iterdir()) and not force:
            raise RuntimeError(f"Refusing to overwrite non-empty dir: {d} (use --force)")
        if force:
            shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def find_trials_file(split_dir: Path) -> Path:
    tune_s = split_dir / "tune_s"
    if not tune_s.exists():
        raise FileNotFoundError(f"Missing tune_s dir in split: {tune_s}")

    hits = list(tune_s.rglob("trials.kaldi"))
    if not hits:
        raise FileNotFoundError(f"No trials.kaldi found under: {tune_s}")

    # If multiple, pick the newest by timestamp-in-name or mtime.
    def key_fn(p: Path) -> Tuple[int, float]:
        dt = _parse_dt_from_name(p.parent)
        if dt:
            return (1, dt.timestamp())
        return (0, p.stat().st_mtime)

    return max(hits, key=key_fn)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default=None, help="Defaults to <repo_root>/artifacts")
    ap.add_argument("--out_data_dir", default=None, help="Defaults to <repo_root>/data")
    ap.add_argument(
        "--wav_path_anchor",
        default="data",
        help=(
            "When source wav.scp uses absolute paths, rewrite them to relative by keeping the "
            "suffix starting at '/<anchor>/'. Default='data' yields paths like 'data/raw_data/...'."
        ),
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing data/train_s and data/tune_s")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    artifacts = Path(args.artifacts_dir) if args.artifacts_dir else (repo_root / "artifacts")
    out_data = Path(args.out_data_dir) if args.out_data_dir else (repo_root / "data")

    # Latest snapshot + latest split (by folder timestamp; fallback to mtime)
    data_lock_root = artifacts / "data_lock"
    splits_root = artifacts / "splits"
    data_lock_dir = pick_latest_dir(data_lock_root)  # e.g. 2026-02-07_...
    split_dir = pick_latest_dir(splits_root, must_contain="split_")  # e.g. split_20260207_...

    base_train = data_lock_dir / "tidyvoice_train"
    train_utts = read_lines(split_dir / "train_utts.txt")
    tune_utts = read_lines(split_dir / "tune_utts.txt")

    # Base maps (from data_lock tidyvoice_train)
    wav_map = read_kv_2col(base_train / "wav.scp")
    utt2spk_all = read_utt2spk(base_train / "utt2spk")
    utt2lang_all = read_kv_2col(base_train / "utt2lang")

    # Rewrite absolute wav paths -> portable relative paths (for Drive/Colab portability)
    wav_map = {u: to_portable_wav_path(p, anchor=args.wav_path_anchor) for u, p in wav_map.items()}


    # Validate existence
    def require_all(utts: List[str], name: str, mp: Dict[str, str]) -> None:
        missing = [u for u in utts if u not in mp]
        if missing:
            raise RuntimeError(f"{name}: missing {len(missing)} utts (showing up to 10): {missing[:10]}")

    require_all(train_utts, "wav.scp", wav_map)
    require_all(tune_utts, "wav.scp", wav_map)
    require_all(train_utts, "utt2spk", utt2spk_all)
    require_all(tune_utts, "utt2spk", utt2spk_all)
    require_all(train_utts, "utt2lang", utt2lang_all)
    require_all(tune_utts, "utt2lang", utt2lang_all)

    # Speaker-disjoint check (leakage safety)
    train_spks = {utt2spk_all[u] for u in train_utts}
    tune_spks = {utt2spk_all[u] for u in tune_utts}
    overlap = train_spks.intersection(tune_spks)
    if overlap:
        raise RuntimeError(f"Leakage: train_s and tune_s share {len(overlap)} speakers (example: {next(iter(overlap))})")

    # Materialize outputs (real files, not symlinks)
    train_out = out_data / "train_s"
    tune_out = out_data / "tune_s"
    ensure_clean_dir(train_out, force=args.force)
    ensure_clean_dir(tune_out, force=args.force)
    (tune_out / "trials").mkdir(parents=True, exist_ok=True)

    # Filtered mappings
    train_utt2spk = {u: utt2spk_all[u] for u in train_utts}
    tune_utt2spk = {u: utt2spk_all[u] for u in tune_utts}
    train_utt2lang = {u: utt2lang_all[u] for u in train_utts}
    tune_utt2lang = {u: utt2lang_all[u] for u in tune_utts}

    write_wav_scp(train_out / "wav.scp", train_utts, wav_map)
    write_utt2x(train_out / "utt2spk", train_utts, train_utt2spk)
    write_utt2x(train_out / "utt2lang", train_utts, train_utt2lang)
    write_spk2utt(train_out / "spk2utt", train_utt2spk, train_utts)

    write_wav_scp(tune_out / "wav.scp", tune_utts, wav_map)
    write_utt2x(tune_out / "utt2spk", tune_utts, tune_utt2spk)
    write_utt2x(tune_out / "utt2lang", tune_utts, tune_utt2lang)
    write_spk2utt(tune_out / "spk2utt", tune_utt2spk, tune_utts)

    # Trials (copy in, don’t symlink, to survive Drive uploads cleanly)
    trials_src = find_trials_file(split_dir)
    trials_dst = tune_out / "trials" / "trials.kaldi"
    shutil.copy2(trials_src, trials_dst)

    base_dev = data_lock_dir / "tidyvoice_dev"
    dev_out = out_data / "dev_s"
    ensure_clean_dir(dev_out, force=args.force)
    (dev_out / "trials").mkdir(parents=True, exist_ok=True)

    dev_wav_map = read_kv_2col(base_dev / "wav.scp")
    dev_wav_map = {u: to_portable_wav_path(p, anchor=args.wav_path_anchor) for u, p in dev_wav_map.items()}
    dev_utt2spk = read_utt2spk(base_dev / "utt2spk")
    dev_utt2lang = read_kv_2col(base_dev / "utt2lang")
    dev_utts = sorted(dev_wav_map.keys())

    # Basic integrity checks (no leakage concept here, just consistency)
    if set(dev_utt2spk.keys()) != set(dev_wav_map.keys()):
        raise RuntimeError("dev_s: utt2spk keys mismatch with wav.scp")
    if set(dev_utt2lang.keys()) != set(dev_wav_map.keys()):
        raise RuntimeError("dev_s: utt2lang keys mismatch with wav.scp")

    write_wav_scp(dev_out / "wav.scp", dev_utts, dev_wav_map)
    write_utt2x(dev_out / "utt2spk", dev_utts, dev_utt2spk)
    write_utt2x(dev_out / "utt2lang", dev_utts, dev_utt2lang)
    write_spk2utt(dev_out / "spk2utt", dev_utt2spk, dev_utts)

    # Copy official dev trials from the locked dev snapshot (NOT from split)
    dev_trials_src = base_dev / "trials" / "trials.kaldi"
    if not dev_trials_src.exists():
        raise FileNotFoundError(f"Missing dev trials: {dev_trials_src}")
    dev_trials_dst = dev_out / "trials" / "trials.kaldi"
    shutil.copy2(dev_trials_src, dev_trials_dst)

    print("OK")
    print(f"Using data_lock: {data_lock_dir}")
    print(f"Using split:     {split_dir}")
    print(f"Wrote: {train_out}  (utts={len(train_utts):,} spks={len(train_spks):,})")
    print(f"Wrote: {tune_out}   (utts={len(tune_utts):,} spks={len(tune_spks):,})")
    print(f"Tune trials: {trials_dst}  (from {trials_src})")
    print(f"Wrote: {dev_out}    (utts={len(dev_utts):,})")
    print(f"Dev trials:  {dev_trials_dst} (from {dev_trials_src})")

if __name__ == "__main__":
    main()
