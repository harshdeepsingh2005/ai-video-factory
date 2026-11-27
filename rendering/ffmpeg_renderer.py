"""Core FFmpeg rendering primitives and Quality Control (QC) checks."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def render_image_with_ken_burns(
    image_path: Path,
    output_path: Path,
    duration_sec: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
) -> Path:
    """Turn a still image into a smooth Ken Burns zoompan MP4 segment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(24, int(round(duration_sec * fps)))

    # Smooth slow zoom-in centered on frame
    vf = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan=z='min(zoom+0.0008,1.15)':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-vf",
        vf,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def normalize_video_clip(
    clip_path: Path,
    output_path: Path,
    duration_sec: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
) -> Path:
    """Loop/trim and scale an AI-generated clip to match target scene duration."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},fps={fps},format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(clip_path),
        "-t",
        f"{duration_sec:.3f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def concat_video_segments(segment_paths: List[Path], output_path: Path) -> Path:
    """Concatenate normalized video-only segments using FFmpeg concat demuxer."""
    if not segment_paths:
        raise ValueError("No video segments provided for concatenation.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = output_path.parent / f"{output_path.stem}_concat.txt"
    lines = [f"file '{Path(p).resolve()}'" for p in segment_paths]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def mux_audio_and_subtitles(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    srt_path: Optional[Path] = None,
    burn_subtitles: bool = False,
) -> Path:
    """Mux normalized audio and optional SRT subtitles into final H.264/AAC MP4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if burn_subtitles and srt_path and Path(srt_path).exists():
        # Escape colons/backslashes for FFmpeg subtitles filter
        escaped_srt = str(Path(srt_path).resolve()).replace("\\", "/").replace(":", "\\:")
        vf = f"subtitles='{escaped_srt}':force_style='FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def run_quality_control(
    video_path: Path,
    expected_aspect: Optional[str] = None,
    min_duration_sec: float = 1.0,
    max_black_ratio: float = 0.25,
) -> Dict[str, object]:
    """Verify file integrity, streams, resolution, duration, and black-frame ratio."""
    video_path = Path(video_path)
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"Render missing or empty: {video_path}")

    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    probe = json.loads(subprocess.check_output(probe_cmd, text=True))
    streams = probe.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not v_streams:
        raise RuntimeError(f"No video stream in {video_path}")
    if not a_streams:
        raise RuntimeError(f"No audio stream in {video_path}")

    duration = float(probe.get("format", {}).get("duration", 0.0))
    if duration < min_duration_sec:
        raise RuntimeError(f"Duration {duration:.2f}s below minimum {min_duration_sec:.2f}s")

    w = int(v_streams[0]["width"])
    h = int(v_streams[0]["height"])
    if expected_aspect == "16:9" and w <= h:
        raise RuntimeError(f"Expected 16:9 landscape, got {w}x{h}")
    if expected_aspect == "9:16" and h <= w:
        raise RuntimeError(f"Expected 9:16 vertical, got {w}x{h}")

    # Black frame detection
    bd_cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        "blackdetect=d=0.5:pix_th=0.10",
        "-an",
        "-f",
        "null",
        "-",
    ]
    bd_proc = subprocess.run(bd_cmd, capture_output=True, text=True, check=True)
    black_dur = sum(
        float(m.group(1))
        for m in re.finditer(r"black_duration:([0-9.]+)", bd_proc.stderr)
    )
    black_ratio = black_dur / max(duration, 0.001)
    if black_ratio > max_black_ratio:
        raise RuntimeError(
            f"Excessive black frames ({black_ratio:.1%} > {max_black_ratio:.1%}) in {video_path}"
        )

    return {
        "path": str(video_path),
        "width": w,
        "height": h,
        "duration_sec": round(duration, 2),
        "has_audio": True,
        "black_ratio": round(black_ratio, 4),
        "passed": True,
    }
