#!/usr/bin/env python3
"""
Submission File Preparation Tool for TidyVoice 2026 challenge

Packages your eval score files into the official submission zip format.

You can control inputs upstream via CLI:
  python tools/prepare_submission.py --exp_dir exp/fused
or:
  python tools/prepare_submission.py --scores_dir exp/fused/scores_custom --output_dir exp/fused/submission_out
or explicit files:
  python tools/prepare_submission.py --score_a ... --score_u ... --ref_a ... --ref_u ...
"""

import sys
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import argparse

# Expected row counts (without headers)
TV26_EVAL_U_EXPECTED_ROWS = 1_280_000
TV26_EVAL_A_EXPECTED_ROWS = 4_000_000


def detect_separator(line: str) -> Optional[str]:
    """Detect if the line uses space or tab separator."""
    if "\t" in line:
        return "\t"
    # whitespace split
    parts = line.strip().split()
    if len(parts) >= 2:
        return " "
    return None


def parse_line(line: str, separator: Optional[str] = None) -> List[str]:
    """Parse a line with the given separator."""
    line = line.strip()
    if not line:
        return []
    if separator == "\t":
        return line.split("\t")
    # separator == " " or unknown -> split on any whitespace
    return line.split()


def detect_format(parts: List[str]) -> str:
    """
    Detect the format of the score file.
    Returns: 'enroll_test_score' or 'score_enroll_test'
    """
    if len(parts) < 3:
        raise ValueError(f"Expected 3 columns, got {len(parts)}")

    # First column numeric => score first
    try:
        float(parts[0])
        return "score_enroll_test"
    except ValueError:
        pass

    # Last column numeric => score last
    try:
        float(parts[-1])
        return "enroll_test_score"
    except ValueError:
        raise ValueError("Cannot detect format: score column not found")


def is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def normalize_filename(filename: str) -> str:
    """Remove .wav extension (case-insensitive) if present."""
    if filename.lower().endswith(".wav"):
        return filename[:-4]
    return filename


def is_header_line(parts: List[str]) -> bool:
    """Conservative header detection (only flags clear header patterns)."""
    if len(parts) < 2:
        return False

    header_keywords = {
        "enroll", "enrollment", "test", "utterance", "pair", "id",
        "filename", "file", "speaker", "segment", "trial", "score",
        "similarity", "distance", "label"
    }
    parts_lower = [p.lower().strip() for p in parts]

    for part in parts_lower:
        if part in header_keywords:
            return True
        for kw in header_keywords:
            if part.startswith(kw + " ") or part.startswith(kw + "\t"):
                return True

    return False


def load_reference_file(ref_path: Path, expected_rows: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Load reference file and return list of (enroll, test) tuples (normalized).
    Skips a detected header only on the first non-empty line.
    """
    pairs: List[Tuple[str, str]] = []
    first_line_checked = False

    with ref_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = parse_line(line)
            if len(parts) < 2:
                raise ValueError(f"{ref_path} line {line_num}: expected 2 cols, got {len(parts)}")

            if not first_line_checked:
                first_line_checked = True
                if is_header_line(parts):
                    print(f"  Detected header in {ref_path.name}, skipping line {line_num}: {line[:80]}")
                    continue

            enroll = normalize_filename(parts[0])
            test = normalize_filename(parts[1])
            pairs.append((enroll, test))

    if expected_rows is not None and len(pairs) != expected_rows:
        raise ValueError(
            f"{ref_path}: expected {expected_rows} data rows, found {len(pairs)}"
        )

    return pairs


def load_score_file(score_path: Path) -> Dict[Tuple[str, str], str]:
    """
    Load score file and return dict (enroll, test) -> score (as string).
    Accepts:
      enroll test score
      score enroll test
    Skips a detected header only on the first non-empty line.
    """
    scores: Dict[Tuple[str, str], str] = {}

    separator: Optional[str] = None
    format_type: Optional[str] = None
    first_line_processed = False

    with score_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if separator is None:
                separator = detect_separator(line)
                if separator is None:
                    raise ValueError(f"Cannot detect separator in {score_path} at line {line_num}")

            parts = parse_line(line, separator)
            if len(parts) < 3:
                raise ValueError(f"{score_path} line {line_num}: expected 3 cols, got {len(parts)}")

            if not first_line_processed:
                first_line_processed = True
                try:
                    format_type = detect_format(parts)
                except ValueError:
                    # If format detection fails, likely header
                    if is_header_line(parts[:2]):
                        print(f"  Detected header in {score_path.name}, skipping line {line_num}: {line[:80]}")
                        continue
                    raise

                # If detected score column isn't numeric, treat as header if obvious
                score_col = parts[2] if format_type == "enroll_test_score" else parts[0]
                if not is_numeric(score_col):
                    if is_header_line(parts[:2]) or score_col.lower() in {"score", "scores", "similarity", "distance", "label"}:
                        print(f"  Detected header in {score_path.name}, skipping line {line_num}: {line[:80]}")
                        continue
                    raise ValueError(f"{score_path} line {line_num}: score '{score_col}' is not numeric")

            assert format_type is not None

            if format_type == "enroll_test_score":
                enroll, test, score = parts[0], parts[1], parts[2]
            else:
                score, enroll, test = parts[0], parts[1], parts[2]

            enroll = normalize_filename(enroll)
            test = normalize_filename(test)

            if not is_numeric(score):
                raise ValueError(f"{score_path} line {line_num}: score '{score}' is not numeric")

            key = (enroll, test)
            if key in scores:
                print(f"Warning: duplicate pair {key} in {score_path} at line {line_num} (keeping last)")
            scores[key] = score

    return scores


def process_submission(
    exclusive_score_file: Path,
    exclusive_ref_file: Path,
    semi_score_file: Path,
    semi_ref_file: Path,
    output_dir: Path,
) -> Path:
    """
    Create timestamped submission folder under output_dir and write:
      tv26_eval-U.txt (one score per line in ref order)
      tv26_eval-A.txt (one score per line in ref order)
      submission_<timestamp>.zip containing both
    Returns the created zip path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"submission_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating unique output directory: {run_dir}")

    print("Loading reference files...")
    exclusive_pairs = load_reference_file(exclusive_ref_file, expected_rows=TV26_EVAL_U_EXPECTED_ROWS)
    semi_pairs = load_reference_file(semi_ref_file, expected_rows=TV26_EVAL_A_EXPECTED_ROWS)
    print(f"  tv26_eval-U pairs: {len(exclusive_pairs)} (expected: {TV26_EVAL_U_EXPECTED_ROWS})")
    print(f"  tv26_eval-A pairs: {len(semi_pairs)} (expected: {TV26_EVAL_A_EXPECTED_ROWS})")

    print("\nLoading score files...")
    exclusive_scores = load_score_file(exclusive_score_file)
    semi_scores = load_score_file(semi_score_file)
    print(f"  Exclusive scores: {len(exclusive_scores)}")
    print(f"  Semi scores: {len(semi_scores)}")

    if len(exclusive_scores) != len(exclusive_pairs):
        raise ValueError(
            f"Exclusive count mismatch: ref={len(exclusive_pairs)} score={len(exclusive_scores)}"
        )
    if len(semi_scores) != len(semi_pairs):
        raise ValueError(
            f"Semi count mismatch: ref={len(semi_pairs)} score={len(semi_scores)}"
        )

    print("\nGenerating output files...")
    out_u = run_dir / "tv26_eval-U.txt"
    with out_u.open("w", encoding="utf-8") as f:
        for enroll, test in exclusive_pairs:
            key = (enroll, test)
            if key not in exclusive_scores:
                raise ValueError(f"Missing exclusive score for pair: {key}")
            f.write(f"{exclusive_scores[key]}\n")
    print(f"  Created: {out_u} ({len(exclusive_pairs)} scores)")

    out_a = run_dir / "tv26_eval-A.txt"
    with out_a.open("w", encoding="utf-8") as f:
        for enroll, test in semi_pairs:
            key = (enroll, test)
            if key not in semi_scores:
                raise ValueError(f"Missing semi score for pair: {key}")
            f.write(f"{semi_scores[key]}\n")
    print(f"  Created: {out_a} ({len(semi_pairs)} scores)")

    print("\nValidating output files...")
    for p in (out_u, out_a):
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if not is_numeric(s):
                    raise ValueError(f"{p} line {i}: non-numeric score '{s}'")
    print("  All scores validated as numeric")

    zip_path = run_dir / f"submission_{timestamp}.zip"
    print(f"\nCreating zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_u, arcname="tv26_eval-U.txt")
        zf.write(out_a, arcname="tv26_eval-A.txt")

    print("\nSubmission package created successfully!")
    print(f"  Zip file: {zip_path}")
    print(f"  Contains: tv26_eval-U.txt ({len(exclusive_pairs)} scores)")
    print(f"            tv26_eval-A.txt ({len(semi_pairs)} scores)")
    return zip_path


def parse_args(project_root: Path) -> argparse.Namespace:
    """
    project_root is your examples/tidyvocie directory (where data/ and exp/ live).
    """
    default_exp_name = "samresnet34_voxblink_ft_tidy_langgrl"
    default_exp_dir = project_root / "exp" / default_exp_name
    default_eval_trials_dir = project_root / "data" / "eval_trials" / "TidyVoiceX_Eval_pairs"
    default_scores_dir = default_exp_dir / "scores_custom"
    default_output_dir = default_exp_dir / "submission_out"

    p = argparse.ArgumentParser(
        description="Package TidyVoiceX eval score files into the submission zip."
    )

    # Convenience: exp_dir implies exp_dir/scores_custom and exp_dir/submission_out
    p.add_argument(
        "--exp_dir",
        type=Path,
        default=None,
        help="Experiment directory. If set, defaults scores_dir=exp_dir/scores_custom and output_dir=exp_dir/submission_out",
    )
    p.add_argument(
        "--scores_dir",
        type=Path,
        default=None,
        help="Directory containing tv26_eval-A_score.txt and tv26_eval-U_score.txt",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write submission_out/* (will create unique timestamped subdir)",
    )
    p.add_argument(
        "--eval_trials_dir",
        type=Path,
        default=default_eval_trials_dir,
        help="Directory containing tv26_eval-A.txt and tv26_eval-U.txt",
    )

    # Optional explicit file overrides
    p.add_argument("--ref_a", type=Path, default=None)
    p.add_argument("--ref_u", type=Path, default=None)
    p.add_argument("--score_a", type=Path, default=None)
    p.add_argument("--score_u", type=Path, default=None)

    args = p.parse_args()

    # Resolve dirs
    if args.scores_dir is None:
        args.scores_dir = (args.exp_dir / "scores_custom") if args.exp_dir else default_scores_dir
    if args.output_dir is None:
        args.output_dir = (args.exp_dir / "submission_out") if args.exp_dir else default_output_dir

    return args


def main() -> None:
    # This script lives in examples/tidyvocie/tools/, so parent[1] is examples/tidyvocie
    project_root = Path(__file__).resolve().parents[1]

    args = parse_args(project_root)

    eval_trials_dir = args.eval_trials_dir
    scores_dir = args.scores_dir
    output_dir = args.output_dir

    ref_a = args.ref_a or (eval_trials_dir / "tv26_eval-A.txt")
    ref_u = args.ref_u or (eval_trials_dir / "tv26_eval-U.txt")
    score_a = args.score_a or (scores_dir / "tv26_eval-A_score.txt")
    score_u = args.score_u or (scores_dir / "tv26_eval-U_score.txt")

    print("Checking input files...")
    for file_path, name in [
        (score_u, "tv26_eval_U-scores"),
        (ref_u, "tv26_eval_U-ref"),
        (score_a, "tv26_eval_A-scores"),
        (ref_a, "tv26_eval_A-ref"),
    ]:
        if not file_path.exists():
            print(f"Error: {name} file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        print(f"   {name}: {file_path}")
    print()

    try:
        zip_path = process_submission(
            exclusive_score_file=score_u,
            exclusive_ref_file=ref_u,
            semi_score_file=score_a,
            semi_ref_file=ref_a,
            output_dir=output_dir,
        )
        print(f"\nSuccess! Submission file: {zip_path}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
