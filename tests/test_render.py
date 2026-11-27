"""End-to-end FFmpeg rendering, shorts reframing, thumbnails, and QC tests."""

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rendering.ffmpeg_renderer import run_quality_control
from rendering.longform import render_longform_video
from rendering.shorts import render_short_video
from rendering.thumbnails import create_thumbnail


def _write_silent_wav(path: Path, duration_sec: float = 4.0, sr: int = 24000) -> Path:
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, int(sr * duration_sec))) * 0.1 * 32767).astype(
        np.int16
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_longform_shorts_thumbnails_and_qc(tmp_path: Path):
    assets_dir = tmp_path / "Assets"
    assets_dir.mkdir()
    for i in (1, 2):
        img = Image.new("RGB", (640, 360), color=(30 * i, 100, 180))
        img.save(assets_dir / f"scene_00{i}.png")

    audio_path = _write_silent_wav(tmp_path / "narration.wav", duration_sec=4.0)
    storyboard = [
        {"scene_id": "scene_001", "duration": 2.0, "asset_filename": "scene_001.png"},
        {"scene_id": "scene_002", "duration": 2.0, "asset_filename": "scene_002.png"},
    ]

    long_out = render_longform_video(
        storyboard=storyboard,
        assets_dir=assets_dir,
        audio_path=audio_path,
        output_path=tmp_path / "longform.mp4",
        work_dir=tmp_path / "work",
        width=640,
        height=360,
        burn_subtitles=False,
    )
    qc_long = run_quality_control(long_out, expected_aspect="16:9", min_duration_sec=3.0)
    assert qc_long["passed"] is True

    short_out = render_short_video(
        short_spec={"short_id": "short_1", "source_scene_ids": ["scene_001"]},
        storyboard=storyboard,
        assets_dir=assets_dir,
        audio_path=audio_path,
        output_path=tmp_path / "short_1.mp4",
        work_dir=tmp_path / "work",
        target_duration_sec=4.0,
    )
    qc_short = run_quality_control(short_out, expected_aspect="9:16", min_duration_sec=3.0)
    assert qc_short["passed"] is True

    thumb = create_thumbnail("Test Video", tmp_path / "thumb.jpg", background_image=assets_dir / "scene_001.png")
    assert thumb.exists() and thumb.stat().st_size > 1000
