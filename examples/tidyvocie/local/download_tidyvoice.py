#!/usr/bin/env python3

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge


import os
import sys
import json
import time
import requests

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

DATASET_ID = "cmihtsewu023so207xot1iqqw"
HF_MODEL_ID = "areffarhadi/Resnet34-tidyvoiceX-ASV"

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{x:.1f}PB"


def _print_bar(done: int, total: int, start_t: float, width: int = 30) -> None:
    if total <= 0:
        print(f"\rDownloaded {_human_bytes(done)}", end="", flush=True)
        return
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_t, 1e-6)
    speed = done / elapsed
    print(
        f"\r[{bar}] {frac*100:6.2f}%  {_human_bytes(done)}/{_human_bytes(total)}  ({_human_bytes(int(speed))}/s)",
        end="",
        flush=True,
    )


def download_with_resume(sess: requests.Session, url: str, out_path: str, expected_size: int = 0) -> None:
    existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    # Already complete?
    if expected_size > 0 and existing >= expected_size:
        print(f"Dataset already present: {out_path} ({_human_bytes(existing)})")
        return

    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"Resuming at {_human_bytes(existing)} ...")
    else:
        print("Starting download ...")

    with sess.get(url, stream=True, headers=headers, timeout=60) as resp:
        # If server says "range not satisfiable", treat as complete
        if resp.status_code == 416:
            print("Server reports file already complete (416).")
            return

        # If we asked for Range but got 200, server ignored Range -> restart cleanly
        if existing > 0 and resp.status_code == 200:
            existing = 0
            mode = "wb"

        resp.raise_for_status()

        # Total for progress bar
        total = expected_size
        if total <= 0:
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                total = existing + int(cl) if resp.status_code == 206 else int(cl)

        start_t = time.time()
        done = existing

        with open(out_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                _print_bar(done, total, start_t)
        print()  # newline


def main():
    if len(sys.argv) < 3:
        print("Usage: python download_tidyvoice.py <output_directory> <api_key>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    api_key = sys.argv[2]
    
    if not api_key or api_key.strip() == "":
        print("ERROR: API key is required")
        print("Please provide your DataCollective API key")
        sys.exit(1)
    
    print("TidyVoice 2026 Challenge Auto-Downloader")
    print("==========================================")
    os.makedirs(output_dir, exist_ok=True)
    
    os.environ["MDC_API_KEY"] = api_key
    os.environ["MDC_DOWNLOAD_PATH"] = output_dir
    
    print(f"Saving to: {output_dir}")
    
    # 1) Download TidyVoiceX dataset (robust: POST -> presigned URL -> stream download)
    try:
        sess = requests.Session()
        r = sess.post(
            f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={},
            timeout=60,
        )
        r.raise_for_status()
        info = r.json()
        url = info["downloadUrl"]
        filename = info.get("filename", "tidyvoicex-asv.tar.gz")
        size_bytes = int(info.get("sizeBytes", "0") or 0)

        out_path = os.path.join(output_dir, filename)

        print(f"Downloading dataset to: {out_path}")
        download_with_resume(sess, url, out_path, expected_size=size_bytes)


        print("\nTidyVoiceX dataset download completed successfully!")
        print(f"Dataset tar saved in: {out_path}\n")

    except Exception as e:
        print("\nERROR while downloading TidyVoiceX dataset:")
        print(str(e))
        sys.exit(1)


    # 2) (Optional) Download pretrained baseline model from Hugging Face
    print("Downloading pretrained baseline model from Hugging Face (optional step)...")
    exp_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "exp",
        "samresnet34_voxblink_ft_tidy",
    )
    os.makedirs(exp_model_dir, exist_ok=True)

    if not HF_AVAILABLE:
        print(
            "\nWARNING: Could not import 'huggingface_hub'. "
            "To auto-download the baseline model, install it with:\n"
            "  pip install huggingface_hub\n"
            "Then re-run this script, or manually download the model from:\n"
            "  https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV\n"
            "and place 'avg_model.pt' and 'config.yaml' into:\n"
            f"  {exp_model_dir}\n"
        )
        return

    for filename in ["models/avg_model.pt", "config.yaml"]:
        try:
            print(f"  - Downloading {filename} from {HF_MODEL_ID} ...")
            local_path = hf_hub_download(
                repo_id=HF_MODEL_ID,
                filename=filename,
                local_dir=exp_model_dir,
                local_dir_use_symlinks=False,
            )
            print(f"    Saved to: {local_path}")
        except Exception as e:
            print(f"\nWARNING: Failed to download {filename} from Hugging Face:")
            print(str(e))
            print(
                "You can still manually download the files from:\n"
                "  https://huggingface.co/areffarhadi/Resnet34-tidyvoiceX-ASV\n"
                "and place 'avg_model.pt' and 'config.yaml' into:\n"
                f"  {exp_model_dir}\n"
            )
            break

    print(
        "\nIf the model files were downloaded successfully, you can run inference with:\n"
        "  ./run.sh --stage 4 --stop_stage 5\n"
    )

if __name__ == "__main__":
    main()

