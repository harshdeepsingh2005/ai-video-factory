"""Long-form 16:9 video assembler."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from rendering.ffmpeg_renderer import (
    concat_video_segments,
    mux_audio_and_subtitles,
    normalize_video_clip,
    render_image_with_ken_burns,
)


def render_longform_video(
    storyboard: List[Dict[str, object]],
    assets_dir: Path,
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
    srt_path: Optional[Path] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    burn_subtitles: bool = True,
) -> Path:
    """Assemble all storyboard scenes into a 16:9 H.264/AAC MP4."""
    work_dir = Path(work_dir)
    seg_dir = work_dir / "longform_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    segments: List[Path] = []
    for scene in storyboard:
        scene_id = str(scene["scene_id"])
        dur = float(scene.get("duration", 10.0))
        asset_path = Path(assets_dir) / str(scene["asset_filename"])
        seg_out = seg_dir / f"{scene_id}.mp4"

        if seg_out.exists() and seg_out.stat().st_size > 0:
            segments.append(seg_out)
            continue

        if asset_path.suffix.lower() == ".mp4" and asset_path.exists():
            normalize_video_clip(asset_path, seg_out, dur, width=width, height=height, fps=fps)
        elif asset_path.exists():
            render_image_with_ken_burns(asset_path, seg_out, dur, width=width, height=height, fps=fps)
        else:
            raise FileNotFoundError(f"Missing visual asset for {scene_id}: {asset_path}")

        segments.append(seg_out)

    silent_concat = work_dir / "longform_silent.mp4"
    concat_video_segments(segments, silent_concat)
    return mux_audio_and_subtitles(
        video_path=silent_concat,
        audio_path=audio_path,
        output_path=output_path,
        srt_path=srt_path,
        burn_subtitles=burn_subtitles,
    )
