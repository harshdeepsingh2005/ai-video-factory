"""Local job directory layout mirroring the Google Drive Production tree."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


SUBFOLDERS = [
    "Long-Form",
    "Shorts",
    "Thumbnails",
    "Narration",
    "Subtitles",
    "Scripts",
    "Assets",
    "Manifest",
]


class AssetManager:
    """Manages per-job directory paths cleanly."""

    def __init__(self, base_dir: Path, job_id: str) -> None:
        self.job_root = Path(base_dir) / job_id
        self.dirs: Dict[str, Path] = {}
        for sub in SUBFOLDERS:
            p = self.job_root / sub
            p.mkdir(parents=True, exist_ok=True)
            self.dirs[sub] = p
        self.work_dir = self.job_root / "_work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def path(self, folder: str, filename: str) -> Path:
        return self.dirs[folder] / filename
