"""Stage checkpointing so interrupted Kaggle/Colab sessions resume cleanly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class CheckpointManager:
    """Atomic JSON manifest persistence for crash/timeout recovery."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save(self, manifest: Dict[str, Any]) -> Path:
        tmp_path = self.manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp_path.replace(self.manifest_path)
        return self.manifest_path
