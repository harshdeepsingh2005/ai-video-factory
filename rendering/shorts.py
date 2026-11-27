"""Short-form 9:16 vertical video renderer with reframing (blurred BG + centered foreground)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from rendering.ffmpeg_renderer import (
    concat_video_segments,
    mux_audio_and_subtitles,
    normalize_video_clip,
    render_image_with_ken_burns,
)


def reframe_16_9_to_9_16(
    input_video: Path,
    output_video: Path,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Reframe landscape footage into 9:16 using blurred backdrop + sharp centered subject."""
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"split[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=25:5[bg_blur];"
        f"[fg]scale={width}:-2:force_original_aspect_ratio=decrease[fg_scaled];"
        f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2,format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-an",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_video


def render_short_video(
    short_spec: Dict[str, object],
    storyboard: List[Dict[str, object]],
    assets_dir: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
    srt_path: Optional[Path] = None,
    target_duration_sec: float = 45.0,
) -> Path:
    """Render one 9:16 vertical short from selected storyboard assets + short narration."""
    short_id = str(short_spec["short_id"])
    work_dir = Path(work_dir) / short_id
    work_dir.mkdir(parents=True, exist_ok=True)

    source_ids = set(short_spec.get("source_scene_ids", []))
    matched = [s for s in storyboard if s["scene_id"] in source_ids] or storyboard[:3]
    per_scene_dur = max(4.0, target_duration_sec / len(matched))

    vertical_segments: List[Path] = []
    for idx, scene in enumerate(matched, start=1):
        asset_path = Path(assets_dir) / str(scene["asset_filename"])
        raw_seg = work_dir / f"raw_{idx}.mp4"
        vert_seg = work_dir / f"vert_{idx}.mp4"

        if asset_path.suffix.lower() == ".mp4" and asset_path.exists():
            normalize_video_clip(asset_path, raw_seg, per_scene_dur, width=1080, height=1920)
            vertical_segments.append(raw_seg)
        elif asset_path.exists():
            # Render 16:9 Ken Burns then reframe to 9:16 blurred-backdrop composition
            render_image_with_ken_burns(asset_path, raw_seg, per_scene_dur, width=1280, height=720)
            reframe_16_9_to_9_16(raw_seg, vert_seg, width=1080, height=1920)
            vertical_segments.append(vert_seg)
        else:
            raise FileNotFoundError(f"Missing asset for short {short_id}: {asset_path}")

    silent_short = work_dir / f"{short_id}_silent.mp4"
    concat_video_segments(vertical_segments, silent_short)
    return mux_audio_and_subtitles(
        video_path=silent_short,
        audio_path=audio_path,
        output_path=output_path,
        srt_path=srt_path,
        burn_subtitles=bool(srt_path),
    )
