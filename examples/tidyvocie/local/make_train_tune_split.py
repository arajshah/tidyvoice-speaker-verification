import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


@dataclass(frozen=True)
class SplitSpec:
    name: str
    speakers: Set[str]


def _now_us_eastern_tag() -> str:
    if ZoneInfo is not None:
        tz = ZoneInfo("America/New_York")
        dt = datetime.now(tz)
        return dt.strftime("%Y%m%d_%H%M%S_ET")
    dt = datetime.utcnow()
    return dt.strftime("%Y%m%d_%H%M%S_UTC")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_kv(path: Path, value_cols: int = 1) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 1 + value_cols:
                raise ValueError(f"Bad line in {path} at {ln}: expected key + {value_cols} value(s), got {len(parts)}")
            k = parts[0]
            v = tuple(parts[1 : 1 + value_cols])
            if k in out:
                raise ValueError(f"Duplicate key in {path}: {k}")
            out[k] = v
    return out


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")


def _utt_lang_from_utt_id(utt_id: str) -> str:
    parts = utt_id.split("/")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    raise ValueError(f"Cannot parse language from utt_id: {utt_id}")


def _make_spk2utt(utt2spk: Dict[str, str]) -> Dict[str, List[str]]:
    spk2utts: Dict[str, List[str]] = {}
    for u, s in utt2spk.items():
        spk2utts.setdefault(s, []).append(u)
    for s in spk2utts:
        spk2utts[s].sort()
    return dict(sorted(spk2utts.items(), key=lambda x: x[0]))


def _assert_integrity(
    wav_scp: Dict[str, str],
    utt2spk: Dict[str, str],
) -> None:
    wav_utts = set(wav_scp.keys())
    spk_utts = set(utt2spk.keys())
    missing_in_utt2spk = sorted(wav_utts - spk_utts)
    extra_in_utt2spk = sorted(spk_utts - wav_utts)
    if missing_in_utt2spk:
        raise ValueError(f"utt2spk missing {len(missing_in_utt2spk)} utts present in wav.scp (e.g. {missing_in_utt2spk[:5]})")
    if extra_in_utt2spk:
        raise ValueError(f"utt2spk has {len(extra_in_utt2spk)} utts not present in wav.scp (e.g. {extra_in_utt2spk[:5]})")


def _subset_by_speakers(
    wav_scp: Dict[str, str],
    utt2spk: Dict[str, str],
    speakers: Set[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    sub_utts = [u for u, s in utt2spk.items() if s in speakers]
    sub_utts.sort()
    sub_wav = {u: wav_scp[u] for u in sub_utts}
    sub_u2s = {u: utt2spk[u] for u in sub_utts}
    return sub_wav, sub_u2s


def _write_split(
    out_dir: Path,
    wav_scp: Dict[str, str],
    utt2spk: Dict[str, str],
) -> Dict[str, str]:
    wav_lines = [f"{u} {wav_scp[u]}" for u in sorted(wav_scp.keys())]
    utt2spk_lines = [f"{u} {utt2spk[u]}" for u in sorted(utt2spk.keys())]
    spk2utt = _make_spk2utt(utt2spk)
    spk2utt_lines = [f"{s} {' '.join(utts)}" for s, utts in spk2utt.items()]
    utt2lang_lines = [f"{u} {_utt_lang_from_utt_id(u)}" for u in sorted(wav_scp.keys())]

    wav_path = out_dir / "wav.scp"
    utt2spk_path = out_dir / "utt2spk"
    spk2utt_path = out_dir / "spk2utt"
    utt2lang_path = out_dir / "utt2lang"

    _write_lines(wav_path, wav_lines)
    _write_lines(utt2spk_path, utt2spk_lines)
    _write_lines(spk2utt_path, spk2utt_lines)
    _write_lines(utt2lang_path, utt2lang_lines)

    return {
        "wav.scp": str(wav_path),
        "utt2spk": str(utt2spk_path),
        "spk2utt": str(spk2utt_path),
        "utt2lang": str(utt2lang_path),
    }


def _write_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(p): _sha256_file(p) for p in paths}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, default="data/tidyvoice_train")
    p.add_argument("--output_root", type=str, default="artifacts/splits")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--tune_speaker_frac", type=float, default=0.20)
    p.add_argument("--write_utt_lists", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_dir = Path(args.train_dir).resolve()
    wav_scp_path = train_dir / "wav.scp"
    utt2spk_path = train_dir / "utt2spk"

    if not wav_scp_path.exists():
        raise FileNotFoundError(f"Missing {wav_scp_path}")
    if not utt2spk_path.exists():
        raise FileNotFoundError(f"Missing {utt2spk_path}")

    wav_scp_kv = _read_kv(wav_scp_path, value_cols=1)
    wav_scp: Dict[str, str] = {k: v[0] for k, v in wav_scp_kv.items()}

    utt2spk_kv = _read_kv(utt2spk_path, value_cols=1)
    utt2spk: Dict[str, str] = {k: v[0] for k, v in utt2spk_kv.items()}

    _assert_integrity(wav_scp, utt2spk)

    speakers = sorted(set(utt2spk.values()))
    if not speakers:
        raise ValueError("No speakers found in utt2spk")

    frac = float(args.tune_speaker_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("--tune_speaker_frac must be in (0, 1)")
    tune_n = int(math.floor(len(speakers) * frac))
    tune_n = max(1, min(len(speakers) - 1, tune_n))

    rng = random.Random(int(args.seed))
    shuffled = speakers[:]
    rng.shuffle(shuffled)

    tune_speakers = set(shuffled[:tune_n])
    train_speakers = set(shuffled[tune_n:])

    if tune_speakers & train_speakers:
        raise ValueError("Speaker overlap detected between Train-S and Tune-S")
    if len(tune_speakers) + len(train_speakers) != len(set(speakers)):
        raise ValueError("Speaker coverage mismatch after splitting")

    tag = _now_us_eastern_tag()
    out_dir = Path(args.output_root).resolve() / f"split_{tag}_seed{args.seed}_tune{frac:.2f}"
    (out_dir / "train_s").mkdir(parents=True, exist_ok=True)
    (out_dir / "tune_s").mkdir(parents=True, exist_ok=True)

    _write_lines(out_dir / "train_speakers.txt", sorted(train_speakers))
    _write_lines(out_dir / "tune_speakers.txt", sorted(tune_speakers))

    train_wav, train_u2s = _subset_by_speakers(wav_scp, utt2spk, train_speakers)
    tune_wav, tune_u2s = _subset_by_speakers(wav_scp, utt2spk, tune_speakers)

    if set(train_wav.keys()) & set(tune_wav.keys()):
        raise ValueError("Utterance overlap detected between Train-S and Tune-S")
    if len(train_wav) + len(tune_wav) != len(wav_scp):
        raise ValueError("Utterance coverage mismatch after splitting")

    if args.write_utt_lists:
        _write_lines(out_dir / "train_utts.txt", sorted(train_wav.keys()))
        _write_lines(out_dir / "tune_utts.txt", sorted(tune_wav.keys()))

    train_files = _write_split(out_dir / "train_s", train_wav, train_u2s)
    tune_files = _write_split(out_dir / "tune_s", tune_wav, tune_u2s)

    produced: List[Path] = []
    produced += [out_dir / "train_speakers.txt", out_dir / "tune_speakers.txt"]
    if args.write_utt_lists:
        produced += [out_dir / "train_utts.txt", out_dir / "tune_utts.txt"]
    produced += [Path(p) for p in train_files.values()]
    produced += [Path(p) for p in tune_files.values()]

    hashes = _write_hashes(produced)

    manifest = {
        "created_at_tag": tag,
        "timezone": "America/New_York",
        "seed": int(args.seed),
        "tune_speaker_frac": float(frac),
        "input_train_dir": str(train_dir),
        "counts": {
            "total_speakers": len(speakers),
            "train_speakers": len(train_speakers),
            "tune_speakers": len(tune_speakers),
            "total_utts": len(wav_scp),
            "train_utts": len(train_wav),
            "tune_utts": len(tune_wav),
        },
        "outputs": {
            "train_speakers_txt": str(out_dir / "train_speakers.txt"),
            "tune_speakers_txt": str(out_dir / "tune_speakers.txt"),
            "train_s": train_files,
            "tune_s": tune_files,
        },
        "hashes_sha256": hashes,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote split to: {out_dir}")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
