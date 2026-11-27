"""FFmpeg-based audio normalization and integrity checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional


def normalize_audio(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -14.0,
    max_true_peak_db: float = -1.5,
    sample_rate: int = 44100,
) -> Path:
    """Normalize speech loudness using FFmpeg loudnorm filter."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={max_true_peak_db}:LRA=11",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def mix_narration_and_music(
    narration_path: Path,
    music_path: Optional[Path],
    output_path: Path,
    music_volume: float = 0.12,
) -> Path:
    """Duck background music under narration, or copy/normalize if no music."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if music_path is None or not Path(music_path).exists():
        return normalize_audio(narration_path, output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(narration_path),
        "-stream_loop",
        "-1",
        "-i",
        str(music_path),
        "-filter_complex",
        f"[1:a]volume={music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]",
        "-map",
        "[out]",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def check_audio_clipping(audio_path: Path) -> dict:
    """Use ffprobe/ffmpeg volumedetect to inspect max_volume for clipping."""
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    max_vol = None
    mean_vol = None
    for line in proc.stderr.splitlines():
        if "max_volume:" in line:
            max_vol = line.split("max_volume:")[1].strip()
        if "mean_volume:" in line:
            mean_vol = line.split("mean_volume:")[1].strip()
    return {"max_volume": max_vol, "mean_volume": mean_vol}
