"""Milestone 1 Benchmark: Generate one short AI video clip on a 16 GB free cloud GPU.

Measures wall-clock latency, peak VRAM, and verifies the resulting MP4 via ffprobe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from inference.video.ltx_pipeline import LTXVideoGenerator


def probe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 1 LTX-Video 16GB GPU benchmark.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic drone shot flying over misty emerald mountains at sunrise, soft golden light, photorealistic",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/milestone1_clip.mp4"))
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Milestone 1 video generation benchmark. "
            "Run this script inside Kaggle (T4 x2) or Google Colab Free (T4)."
        )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    generator = LTXVideoGenerator()
    out_path = generator.generate_clip(
        prompt=args.prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
    )
    generator.unload()

    elapsed = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    probe = probe_video(out_path)

    report = {
        "status": "SUCCESS",
        "output_path": str(out_path),
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "gpu_name": torch.cuda.get_device_name(0),
        "ffprobe": probe,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
