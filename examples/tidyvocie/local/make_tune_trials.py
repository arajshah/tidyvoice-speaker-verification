#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def now_et_stamp() -> str:
    if ZoneInfo is None:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_ET")
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_ET")


def read_wav_scp(path: Path) -> dict:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            out[k] = v
    return out


def read_kv(path: Path) -> dict:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            out[k] = v
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_exists(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")


def build_indices(utt2spk: dict, utt2lang: dict, utts: set) -> dict:
    spk2utts = defaultdict(list)
    lang2utts = defaultdict(list)
    spk_lang2utts = defaultdict(list)
    spk2langs = defaultdict(set)

    for u in utts:
        s = utt2spk[u]
        l = utt2lang[u]
        spk2utts[s].append(u)
        lang2utts[l].append(u)
        spk_lang2utts[(s, l)].append(u)
        spk2langs[s].add(l)

    eligible_spk_lang_pairs = [(s, l) for (s, l), xs in spk_lang2utts.items() if len(xs) >= 2]
    eligible_spk_multi_lang = [s for s, ls in spk2langs.items() if len(ls) >= 2]

    lang2speakers = defaultdict(list)
    for (s, l) in spk_lang2utts.keys():
        lang2speakers[l].append(s)
    eligible_lang_multi_speaker = [l for l, ss in lang2speakers.items() if len(set(ss)) >= 2]
    lang2speakers = {l: sorted(list(set(ss))) for l, ss in lang2speakers.items()}

    all_utts_list = sorted(list(utts))

    return {
        "spk2utts": spk2utts,
        "lang2utts": lang2utts,
        "spk_lang2utts": spk_lang2utts,
        "spk2langs": {s: sorted(list(ls)) for s, ls in spk2langs.items()},
        "lang2speakers": lang2speakers,
        "eligible_spk_lang_pairs": eligible_spk_lang_pairs,
        "eligible_spk_multi_lang": eligible_spk_multi_lang,
        "eligible_lang_multi_speaker": eligible_lang_multi_speaker,
        "all_utts_list": all_utts_list,
    }


def pair_key(u1: str, u2: str) -> tuple:
    return (u1, u2) if u1 <= u2 else (u2, u1)


def sample_target_same_lang(rng: random.Random, idx: dict) -> tuple[str, str]:
    s, l = rng.choice(idx["eligible_spk_lang_pairs"])
    xs = idx["spk_lang2utts"][(s, l)]
    u1, u2 = rng.sample(xs, 2)
    return u1, u2


def sample_target_diff_lang(rng: random.Random, idx: dict) -> tuple[str, str]:
    s = rng.choice(idx["eligible_spk_multi_lang"])
    langs = idx["spk2langs"][s]
    l1, l2 = rng.sample(langs, 2)
    u1 = rng.choice(idx["spk_lang2utts"][(s, l1)])
    u2 = rng.choice(idx["spk_lang2utts"][(s, l2)])
    return u1, u2


def sample_nontarget_same_lang(rng: random.Random, idx: dict) -> tuple[str, str]:
    l = rng.choice(idx["eligible_lang_multi_speaker"])
    speakers = idx["lang2speakers"][l]
    s1, s2 = rng.sample(speakers, 2)
    u1 = rng.choice(idx["spk_lang2utts"][(s1, l)])
    u2 = rng.choice(idx["spk_lang2utts"][(s2, l)])
    return u1, u2


def sample_nontarget_diff_lang(
    rng: random.Random,
    idx: dict,
    utt2spk: dict,
    utt2lang: dict,
    max_tries: int,
) -> tuple[str, str]:
    all_utts = idx["all_utts_list"]
    for _ in range(max_tries):
        u1 = rng.choice(all_utts)
        s1 = utt2spk[u1]
        l1 = utt2lang[u1]
        u2 = rng.choice(all_utts)
        if u2 == u1:
            continue
        if utt2spk[u2] == s1:
            continue
        if utt2lang[u2] == l1:
            continue
        return u1, u2
    raise RuntimeError("Failed to sample non-target diff-lang pair (increase --max_pair_tries or reduce caps)")


def generate_bucket(
    name: str,
    n: int,
    sampler,
    rng: random.Random,
    seen: set,
    utt2spk: dict,
    utt2lang: dict,
    idx: dict,
    max_pair_tries: int,
    label: str,
) -> list[tuple[str, str, str, str]]:
    out = []
    attempts = 0
    hard_limit = max(n * 50, 10_000)
    while len(out) < n:
        if attempts > hard_limit:
            raise RuntimeError(f"Too many collisions while building bucket={name} (built {len(out)}/{n}).")
        attempts += 1
        if name == "non_diff_lang":
            u1, u2 = sampler(rng, idx, utt2spk, utt2lang, max_pair_tries)
        else:
            u1, u2 = sampler(rng, idx)
        if u1 == u2:
            continue
        k = pair_key(u1, u2)
        if k in seen:
            continue
        seen.add(k)
        out.append((u1, u2, label, name))
    return out


def validate_trials(trials: list[tuple[str, str, str, str]], utts: set) -> None:
    for u1, u2, lab, _ in trials:
        if u1 not in utts or u2 not in utts:
            raise ValueError(f"Trial references missing utt: ({u1}, {u2})")
        if lab not in {"target", "nontarget"}:
            raise ValueError(f"Invalid label: {lab}")


def compute_coverage(trials: list[tuple[str, str, str, str]], utt2spk: dict, utt2lang: dict) -> dict:
    spks = set()
    langs = set()
    for u1, u2, _, _ in trials:
        spks.add(utt2spk[u1])
        spks.add(utt2spk[u2])
        langs.add(utt2lang[u1])
        langs.add(utt2lang[u2])
    return {"unique_speakers": len(spks), "unique_langs": len(langs)}


def bucket_counts(trials: list[tuple[str, str, str, str]]) -> dict:
    out = defaultdict(int)
    for _, _, _, b in trials:
        out[b] += 1
    return dict(sorted(out.items(), key=lambda x: x[0]))


def write_outputs(out_dir: Path, trials: list[tuple[str, str, str, str]], stats: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.kaldi"
    stats_path = out_dir / "stats.json"
    hashes_path = out_dir / "hashes.json"

    with trials_path.open("w", encoding="utf-8") as f:
        for u1, u2, lab, _ in trials:
            f.write(f"{u1} {u2} {lab}\n")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    hashes = {
        "trials.kaldi": sha256_file(trials_path),
        "stats.json": sha256_file(stats_path),
    }
    with hashes_path.open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    hashes["hashes.json"] = sha256_file(hashes_path)
    with hashes_path.open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    return {"trials_path": str(trials_path), "stats_path": str(stats_path), "hashes_path": str(hashes_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune_dir", required=True, type=str)
    ap.add_argument("--output_dir", default="", type=str)
    ap.add_argument("--seed", default=43, type=int)
    ap.add_argument("--num_per_bucket", default=200000, type=int)
    ap.add_argument("--max_pair_tries", default=2000, type=int)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    tune_dir = Path(args.tune_dir).resolve()
    wav_scp = tune_dir / "wav.scp"
    utt2spk_p = tune_dir / "utt2spk"
    utt2lang_p = tune_dir / "utt2lang"

    ensure_exists(wav_scp, "Tune-S wav.scp")
    ensure_exists(utt2spk_p, "Tune-S utt2spk")
    ensure_exists(utt2lang_p, "Tune-S utt2lang")

    wav_map = read_wav_scp(wav_scp)
    utts = set(wav_map.keys())
    utt2spk = read_kv(utt2spk_p)
    utt2lang = read_kv(utt2lang_p)

    if set(utt2spk.keys()) != utts:
        missing = sorted(list(utts - set(utt2spk.keys())))[:10]
        extra = sorted(list(set(utt2spk.keys()) - utts))[:10]
        raise ValueError(f"utt2spk keys mismatch with wav.scp. missing={missing} extra={extra}")
    if set(utt2lang.keys()) != utts:
        missing = sorted(list(utts - set(utt2lang.keys())))[:10]
        extra = sorted(list(set(utt2lang.keys()) - utts))[:10]
        raise ValueError(f"utt2lang keys mismatch with wav.scp. missing={missing} extra={extra}")

    idx = build_indices(utt2spk, utt2lang, utts)

    if not idx["eligible_spk_lang_pairs"]:
        raise ValueError("No eligible (speaker,lang) with >=2 utterances for target_same_lang.")
    if not idx["eligible_spk_multi_lang"]:
        raise ValueError("No eligible speakers with >=2 languages for target_diff_lang.")
    if not idx["eligible_lang_multi_speaker"]:
        raise ValueError("No eligible languages with >=2 speakers for non_same_lang.")

    rng = random.Random(args.seed)
    seen = set()

    n = int(args.num_per_bucket)
    buckets = [
        ("tar_same_lang", n, sample_target_same_lang, "target"),
        ("tar_diff_lang", n, sample_target_diff_lang, "target"),
        ("non_same_lang", n, sample_nontarget_same_lang, "nontarget"),
        ("non_diff_lang", n, sample_nontarget_diff_lang, "nontarget"),
    ]

    all_trials = []
    for bname, bn, sampler, lab in buckets:
        trials_b = generate_bucket(
            name=bname,
            n=bn,
            sampler=sampler,
            rng=rng,
            seen=seen,
            utt2spk=utt2spk,
            utt2lang=utt2lang,
            idx=idx,
            max_pair_tries=args.max_pair_tries,
            label=lab,
        )
        all_trials.extend(trials_b)

    if args.shuffle:
        rng.shuffle(all_trials)

    validate_trials(all_trials, utts)

    run_dir = tune_dir.parent
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "tune_trials")
    out_dir = out_dir / now_et_stamp()

    stats = {
        "created_et": now_et_stamp(),
        "tune_dir": str(tune_dir),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "num_per_bucket": n,
        "total_trials": len(all_trials),
        "bucket_counts": bucket_counts(all_trials),
        "coverage": compute_coverage(all_trials, utt2spk, utt2lang),
        "unique_utts_in_tune": len(utts),
        "shuffle": bool(args.shuffle),
    }

    paths = write_outputs(out_dir, all_trials, stats)
    print(json.dumps({"ok": True, **paths, "total_trials": len(all_trials), "bucket_counts": stats["bucket_counts"]}, indent=2))


if __name__ == "__main__":
    main()
