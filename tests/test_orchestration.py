"""Orchestration state machine, checkpointing, and recovery tests."""

from pathlib import Path

from orchestration.recovery import SessionBudgetGuard, is_stage_done
from orchestration.state_manager import (
    JobState,
    create_initial_manifest,
    mark_stage_complete,
    record_error,
)
from storage.asset_manager import AssetManager
from storage.checkpoint_manager import CheckpointManager


def test_manifest_lifecycle_and_checkpoint_recovery(tmp_path: Path):
    assets = AssetManager(tmp_path, "job_001")
    ckpt = CheckpointManager(assets.path("Manifest", "manifest.json"))

    manifest = create_initial_manifest(
        job_id="job_001",
        topic="Test Topic",
        models={"video": "Lightricks/LTX-Video"},
        settings={"target_duration_sec": 600},
    )
    manifest = mark_stage_complete(manifest, JobState.INITIALISING)
    manifest = mark_stage_complete(manifest, JobState.SCRIPT_GENERATION)
    ckpt.save(manifest)

    loaded = ckpt.load()
    assert is_stage_done(loaded, JobState.INITIALISING)
    assert is_stage_done(loaded, JobState.SCRIPT_GENERATION)
    assert not is_stage_done(loaded, JobState.VIDEO_GENERATION)

    failed = record_error(loaded, JobState.VIDEO_GENERATION, "CUDA OOM")
    assert failed["state"] == JobState.FAILED.value
    assert failed["retry_counts"][JobState.VIDEO_GENERATION.value] == 1


def test_session_budget_guard():
    guard = SessionBudgetGuard(max_runtime_hours=1.0)
    assert guard.remaining_seconds() > 3500
    assert guard.should_pause(buffer_sec=10.0) is False
