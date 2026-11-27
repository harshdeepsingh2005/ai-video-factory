"""Explicit 15-state job lifecycle and production manifest management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


class JobState(str, Enum):
    QUEUED = "QUEUED"
    WAITING_FOR_GPU = "WAITING_FOR_GPU"
    INITIALISING = "INITIALISING"
    SCRIPT_GENERATION = "SCRIPT_GENERATION"
    STORYBOARD_GENERATION = "STORYBOARD_GENERATION"
    NARRATION_GENERATION = "NARRATION_GENERATION"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    VIDEO_GENERATION = "VIDEO_GENERATION"
    LONGFORM_RENDER = "LONGFORM_RENDER"
    SHORTS_RENDER = "SHORTS_RENDER"
    QUALITY_CHECK = "QUALITY_CHECK"
    DRIVE_UPLOAD = "DRIVE_UPLOAD"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


ORDERED_STAGES: List[JobState] = [
    JobState.INITIALISING,
    JobState.SCRIPT_GENERATION,
    JobState.STORYBOARD_GENERATION,
    JobState.NARRATION_GENERATION,
    JobState.IMAGE_GENERATION,
    JobState.VIDEO_GENERATION,
    JobState.LONGFORM_RENDER,
    JobState.SHORTS_RENDER,
    JobState.QUALITY_CHECK,
    JobState.DRIVE_UPLOAD,
]


def create_initial_manifest(
    job_id: str,
    topic: str,
    models: Dict[str, str],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "state": JobState.QUEUED.value,
        "models": models,
        "settings": settings,
        "completed_stages": [],
        "asset_paths": {},
        "output_file_ids": {},
        "errors": [],
        "retry_counts": {},
    }


def mark_stage_complete(manifest: Dict[str, Any], stage: JobState) -> Dict[str, Any]:
    completed = list(manifest.get("completed_stages", []))
    if stage.value not in completed:
        completed.append(stage.value)
    return {
        **manifest,
        "state": stage.value,
        "completed_stages": completed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def record_error(manifest: Dict[str, Any], stage: JobState, error: str) -> Dict[str, Any]:
    retries = dict(manifest.get("retry_counts", {}))
    retries[stage.value] = retries.get(stage.value, 0) + 1
    errors = list(manifest.get("errors", []))
    errors.append(
        {
            "stage": stage.value,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {
        **manifest,
        "state": JobState.FAILED.value,
        "retry_counts": retries,
        "errors": errors,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
