"""Download and cache open-weight models within free-tier disk limits.

Checks available disk space before downloading to prevent hitting Kaggle's
19.5 GB /kaggle/working ceiling.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download


def get_free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def load_models_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_ltx_video(cache_dir: Path, cfg: dict) -> Path:
    repo_id = cfg["video"]["primary"]["repo_id"]
    filename = cfg["video"]["primary"]["checkpoint_file"]
    print(f"[download] Checking/downloading {repo_id}/{filename}...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return Path(local_path)


def download_sd15(cache_dir: Path, cfg: dict) -> Path:
    repo_id = cfg["image"]["primary"]["repo_id"]
    print(f"[download] Checking/downloading {repo_id} (fp16)...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=["*.json", "*.txt", "*fp16*", "scheduler/*", "tokenizer/*", "feature_extractor/*"],
        ignore_patterns=["*.ckpt", "*.bin", "*non_ema*"],
    )
    return Path(local_dir)


def download_qwen(cache_dir: Path, cfg: dict) -> Path:
    repo_id = cfg["llm"]["primary"]["repo_id"]
    print(f"[download] Checking/downloading {repo_id}...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
    )
    return Path(local_dir)


def download_kokoro(cache_dir: Path, cfg: dict) -> Path:
    repo_id = cfg["tts"]["primary"]["repo_id"]
    print(f"[download] Checking/downloading {repo_id}...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
    )
    return Path(local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned open-weight models.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/tmp/hf_cache")),
        help="Use /tmp/hf_cache on Kaggle to avoid filling /kaggle/working 19.5GB output quota.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["ltx", "sd15", "qwen", "kokoro"],
        choices=["ltx", "sd15", "qwen", "kokoro"],
    )
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    args = parser.parse_args()

    free_gb = get_free_disk_gb(args.cache_dir)
    print(f"[disk] Free space at {args.cache_dir}: {free_gb:.2f} GB")
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Insufficient disk space ({free_gb:.2f} GB < {args.min_free_gb} GB). "
            "Clear scratch files or use /tmp/hf_cache on Kaggle."
        )

    cfg = load_models_config(args.config)
    if "ltx" in args.components:
        print("LTX-Video:", download_ltx_video(args.cache_dir, cfg))
    if "sd15" in args.components:
        print("SD 1.5:", download_sd15(args.cache_dir, cfg))
    if "qwen" in args.components:
        print("Qwen2.5-1.5B:", download_qwen(args.cache_dir, cfg))
    if "kokoro" in args.components:
        print("Kokoro-82M:", download_kokoro(args.cache_dir, cfg))


if __name__ == "__main__":
    main()
