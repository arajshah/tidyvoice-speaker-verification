#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception as e:
    raise RuntimeError("zoneinfo is required (Python 3.9+).") from e


@dataclass(frozen=True)
class SplitPaths:
    name: str
    split_dir: Path
    wav_scp: Path
    utt2spk: Path
    spk2utt: Path


def read_kaldi_kv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"{path}: line {ln} must have at least 2 columns.")
            k = parts[0]
            v = " ".join(parts[1:])
            if k in out:
                raise ValueError(f"{path}: duplicate key '{k}' at line {ln}.")
            out[k] = v
    return out


def read_spk2utt(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"{path}: line {ln} must have at least 2 columns.")
            spk = parts[0]
            utts = parts[1:]
            if spk in out:
                raise ValueError(f"{path}: duplicate speaker '{spk}' at line {ln}.")
            out[spk] = utts
    return out


def build_spk2utt(utt2spk: Dict[str, str]) -> Dict[str, List[str]]:
    spk_map: Dict[str, List[str]] = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk_map[spk].append(utt)
    for spk in spk_map:
        spk_map[spk].sort()
    return dict(sorted(spk_map.items(), key=lambda kv: kv[0]))


def write_kaldi_kv(path: Path, items: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in items:
            f.write(f"{k} {v}\n")


def write_spk2utt(path: Path, spk2utt: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for spk, utts in spk2utt.items():
            f.write(spk)
            for u in utts:
                f.write(f" {u}")
            f.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_timestamp(tz: str) -> str:
    dt = datetime.now(ZoneInfo(tz))
    return dt.strftime("%Y-%m-%d_%H-%M-%S_%Z")


def safe_mkdir_versioned(root: Path, version: str) -> Path:
    base = root / version
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    i = 1
    while True:
        cand = root / f"{version}__{i}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        i += 1


def derive_utt2lang_from_utt_ids(utt_ids: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for utt in utt_ids:
        parts = utt.split("/")
        if len(parts) < 2:
            raise ValueError(f"Cannot derive language from utt_id '{utt}'. Expected 'spk/lang/...'.")
        out[utt] = parts[1]
    return out


def validate_split(split: SplitPaths, check_wavs: bool, wav_check_limit: int) -> Dict[str, object]:
    if not split.wav_scp.exists():
        raise FileNotFoundError(f"Missing {split.wav_scp}")
    if not split.utt2spk.exists():
        raise FileNotFoundError(f"Missing {split.utt2spk}")

    wav = read_kaldi_kv(split.wav_scp)
    u2s = read_kaldi_kv(split.utt2spk)

    missing_u2s = [u for u in wav.keys() if u not in u2s]
    if missing_u2s:
        raise ValueError(f"{split.name}: {len(missing_u2s)} utts in wav.scp missing from utt2spk (e.g., {missing_u2s[0]}).")

    if check_wavs:
        items = list(wav.items())
        if wav_check_limit > 0:
            items = items[: min(wav_check_limit, len(items))]
        bad = [u for u, p in items if not Path(p).exists()]
        if bad:
            raise FileNotFoundError(f"{split.name}: {len(bad)} wav paths do not exist (e.g., {bad[0]} -> {wav[bad[0]]}).")

    spk2utt_expected = build_spk2utt({u: u2s[u] for u in wav.keys()})
    spk_count = len(spk2utt_expected)
    utt_count = len(wav)

    stats = {
        "split": split.name,
        "utt_count": utt_count,
        "spk_count": spk_count,
    }
    return {
        "wav": wav,
        "utt2spk": u2s,
        "spk2utt_expected": spk2utt_expected,
        "stats": stats,
    }


def validate_trials(trials_path: Path, dev_utts: Sequence[str]) -> Dict[str, object]:
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing {trials_path}")

    dev_set = set(dev_utts)
    n = 0
    bad_format = 0
    bad_label = 0
    missing = 0
    first_missing: str | None = None

    with trials_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 3:
                bad_format += 1
                continue
            u1, u2, lab = parts
            if lab not in {"target", "nontarget"}:
                bad_label += 1
            if u1 not in dev_set or u2 not in dev_set:
                missing += 1
                if first_missing is None:
                    first_missing = f"line {ln}: {u1} {u2}"
            n += 1

    if bad_format:
        raise ValueError(f"{trials_path}: {bad_format} lines have wrong format (expected 3 columns).")
    if bad_label:
        raise ValueError(f"{trials_path}: {bad_label} lines have invalid labels (expected target/nontarget).")
    if missing:
        raise ValueError(f"{trials_path}: {missing} lines reference utts not in dev wav.scp (e.g., {first_missing}).")

    return {"trial_count": n}


def copy_or_write_locked_files(
    lock_dir: Path,
    split_name: str,
    wav: Dict[str, str],
    utt2spk: Dict[str, str],
    spk2utt_expected: Dict[str, List[str]],
    include_utt2lang: bool,
) -> Dict[str, Path]:
    out_split = lock_dir / split_name
    out_split.mkdir(parents=True, exist_ok=False)

    wav_out = out_split / "wav.scp"
    utt2spk_out = out_split / "utt2spk"
    spk2utt_out = out_split / "spk2utt"
    utt2lang_out = out_split / "utt2lang"

    write_kaldi_kv(wav_out, sorted(wav.items(), key=lambda kv: kv[0]))
    write_kaldi_kv(utt2spk_out, sorted(((u, utt2spk[u]) for u in wav.keys()), key=lambda kv: kv[0]))
    write_spk2utt(spk2utt_out, spk2utt_expected)

    paths: Dict[str, Path] = {
        f"{split_name}/wav.scp": wav_out,
        f"{split_name}/utt2spk": utt2spk_out,
        f"{split_name}/spk2utt": spk2utt_out,
    }

    if include_utt2lang:
        utt2lang = derive_utt2lang_from_utt_ids(wav.keys())
        write_kaldi_kv(utt2lang_out, sorted(utt2lang.items(), key=lambda kv: kv[0]))
        paths[f"{split_name}/utt2lang"] = utt2lang_out

    return paths


def copy_trials(lock_dir: Path, src_trials: Path) -> Path:
    out_trials_dir = lock_dir / "tidyvoice_dev" / "trials"
    out_trials_dir.mkdir(parents=True, exist_ok=True)
    dst = out_trials_dir / "trials.kaldi"
    shutil.copy2(src_trials, dst)
    return dst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lock_data.py")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--artifacts_dir", type=str, default="artifacts/data_lock")
    p.add_argument("--timezone", type=str, default="America/New_York")
    p.add_argument("--include_utt2lang", action="store_true", default=True)
    p.add_argument("--no_utt2lang", dest="include_utt2lang", action="store_false")
    p.add_argument("--check_wavs", action="store_true", default=False)
    p.add_argument("--wav_check_limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tidy_root = Path(__file__).resolve().parents[1]
    data_dir = (tidy_root / args.data_dir).resolve()
    artifacts_root = (tidy_root / args.artifacts_dir).resolve()

    train_dir = data_dir / "tidyvoice_train"
    dev_dir = data_dir / "tidyvoice_dev"
    trials_src = dev_dir / "trials" / "trials.kaldi"

    train = SplitPaths(
        name="tidyvoice_train",
        split_dir=train_dir,
        wav_scp=train_dir / "wav.scp",
        utt2spk=train_dir / "utt2spk",
        spk2utt=train_dir / "spk2utt",
    )
    dev = SplitPaths(
        name="tidyvoice_dev",
        split_dir=dev_dir,
        wav_scp=dev_dir / "wav.scp",
        utt2spk=dev_dir / "utt2spk",
        spk2utt=dev_dir / "spk2utt",
    )

    artifacts_root.mkdir(parents=True, exist_ok=True)
    version = now_timestamp(args.timezone)
    lock_dir = safe_mkdir_versioned(artifacts_root, version)

    train_info = validate_split(train, check_wavs=args.check_wavs, wav_check_limit=args.wav_check_limit)
    dev_info = validate_split(dev, check_wavs=args.check_wavs, wav_check_limit=args.wav_check_limit)
    trials_info = validate_trials(trials_src, list(dev_info["wav"].keys()))

    locked_paths: Dict[str, Path] = {}
    locked_paths.update(
        copy_or_write_locked_files(
            lock_dir=lock_dir,
            split_name="tidyvoice_train",
            wav=train_info["wav"],
            utt2spk=train_info["utt2spk"],
            spk2utt_expected=train_info["spk2utt_expected"],
            include_utt2lang=args.include_utt2lang,
        )
    )
    locked_paths.update(
        copy_or_write_locked_files(
            lock_dir=lock_dir,
            split_name="tidyvoice_dev",
            wav=dev_info["wav"],
            utt2spk=dev_info["utt2spk"],
            spk2utt_expected=dev_info["spk2utt_expected"],
            include_utt2lang=args.include_utt2lang,
        )
    )
    trials_dst = copy_trials(lock_dir, trials_src)
    locked_paths["tidyvoice_dev/trials/trials.kaldi"] = trials_dst

    file_hashes = {k: sha256_file(v) for k, v in sorted(locked_paths.items(), key=lambda kv: kv[0])}

    manifest = {
        "created_at": version,
        "timezone": args.timezone,
        "tidy_root": str(tidy_root),
        "data_dir": str(data_dir),
        "lock_dir": str(lock_dir),
        "splits": {
            "tidyvoice_train": train_info["stats"],
            "tidyvoice_dev": dev_info["stats"],
        },
        "trials": trials_info,
        "include_utt2lang": bool(args.include_utt2lang),
        "check_wavs": bool(args.check_wavs),
        "wav_check_limit": int(args.wav_check_limit),
        "files": {k: str(v) for k, v in sorted(locked_paths.items(), key=lambda kv: kv[0])},
        "sha256": file_hashes,
    }

    with (lock_dir / "lock_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(str(lock_dir))


if __name__ == "__main__":
    main()
