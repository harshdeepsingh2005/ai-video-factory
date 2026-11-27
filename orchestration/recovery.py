"""Session timeout & stage recovery helpers for Kaggle/Colab free GPUs."""

from __future__ import annotations

import time
from typing import Any, Dict

from orchestration.state_manager import JobState


class SessionBudgetGuard:
    """Pauses pipeline cleanly before hitting Kaggle's 12-hour hard termination."""

    def __init__(self, max_runtime_hours: float = 10.5) -> None:
        self.start_time = time.monotonic()
        self.max_runtime_sec = max_runtime_hours * 3600.0

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_runtime_sec - (time.monotonic() - self.start_time))

    def should_pause(self, buffer_sec: float = 300.0) -> bool:
        return self.remaining_seconds() < buffer_sec


def is_stage_done(manifest: Dict[str, Any], stage: JobState) -> bool:
    return stage.value in manifest.get("completed_stages", [])
