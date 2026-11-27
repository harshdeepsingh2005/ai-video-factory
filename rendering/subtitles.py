"""SRT subtitle generation from audio timing metadata or storyboard."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List


def format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms_total, 3600_000)
    mins, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def generate_srt_from_chunks(
    chunks: List[Dict[str, object]],
    output_path: Path,
    max_line_width: int = 42,
) -> Path:
    """Write a valid SRT file from Kokoro audio_metadata chunks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    idx = 1
    for ch in chunks:
        text = str(ch.get("text", "")).strip()
        if not text:
            continue
        start = float(ch.get("start_sec", 0.0))
        dur = max(0.5, float(ch.get("duration_sec", 2.0)))
        end = start + dur

        wrapped = textwrap.fill(text, width=max_line_width)
        lines.append(f"{idx}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{wrapped}\n")
        idx += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
