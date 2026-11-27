"""End-to-end sequential, resumable AI Video Factory pipeline runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from inference.audio.audio_processor import normalize_audio
from inference.audio.kokoro_tts import KokoroTTSGenerator
from inference.llm.script_generator import ScriptGenerator
from inference.video.image_pipeline import HybridImageGenerator
from inference.video.ltx_pipeline import LTXVideoGenerator
from orchestration.job_manager import CloudflareJobClient
from orchestration.recovery import SessionBudgetGuard, is_stage_done
from orchestration.state_manager import (
    JobState,
    create_initial_manifest,
    mark_stage_complete,
    record_error,
)
from rendering.ffmpeg_renderer import run_quality_control
from rendering.longform import render_longform_video
from rendering.shorts import render_short_video
from rendering.subtitles import generate_srt_from_chunks
from rendering.thumbnails import create_thumbnail
from storage.asset_manager import AssetManager
from storage.checkpoint_manager import CheckpointManager
from storage.google_drive import GoogleDriveClient


def run_pipeline(
    job_id: str,
    topic: str,
    workspace_dir: Path = Path("outputs"),
    target_duration_sec: int = 600,
    num_shorts: int = 3,
    audience: str = "curious general audience",
    tone: str = "engaging, clear, documentary",
    style: str = "cinematic, clean editorial visuals",
    upload_to_drive: bool = False,
) -> Dict[str, Any]:
    """Execute all 11 production stages sequentially with checkpoint recovery."""
    models_cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    prod_cfg = yaml.safe_load(Path("configs/production.yaml").read_text(encoding="utf-8"))["production"]

    assets = AssetManager(workspace_dir, job_id)
    ckpt = CheckpointManager(assets.path("Manifest", "manifest.json"))
    guard = SessionBudgetGuard(max_runtime_hours=10.5)
    cf_client = CloudflareJobClient()

    manifest = ckpt.load()
    if not manifest:
        manifest = create_initial_manifest(
            job_id=job_id,
            topic=topic,
            models={
                "video": models_cfg["video"]["primary"]["repo_id"],
                "image": models_cfg["image"]["primary"]["repo_id"],
                "llm": models_cfg["llm"]["primary"]["repo_id"],
                "tts": models_cfg["tts"]["primary"]["repo_id"],
            },
            settings={
                "target_duration_sec": target_duration_sec,
                "num_shorts": num_shorts,
                "audience": audience,
                "tone": tone,
                "style": style,
            },
        )
        ckpt.save(manifest)

    def checkpoint(stage: JobState) -> None:
        nonlocal manifest
        manifest = mark_stage_complete(manifest, stage)
        ckpt.save(manifest)
        cf_client.update_job(job_id, {"state": stage.value, "manifest": manifest})

    try:
        # Stage 1: Initialising
        if not is_stage_done(manifest, JobState.INITIALISING):
            checkpoint(JobState.INITIALISING)

        # Stage 2 & 3: Script & Storyboard Generation (Qwen2.5-1.5B-Instruct)
        package_path = assets.path("Scripts", "production_package.json")
        if not is_stage_done(manifest, JobState.STORYBOARD_GENERATION):
            llm = ScriptGenerator(repo_id=models_cfg["llm"]["primary"]["repo_id"])
            package = llm.generate_production_package(
                topic=topic,
                audience=audience,
                tone=tone,
                style=style,
                target_duration_sec=target_duration_sec,
                num_shorts=num_shorts,
                max_video_clips=prod_cfg["max_ai_video_clips_per_job"],
            )
            llm.unload()
            package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
            assets.path("Scripts", "narration.txt").write_text(
                package["full_narration"], encoding="utf-8"
            )
            checkpoint(JobState.SCRIPT_GENERATION)
            checkpoint(JobState.STORYBOARD_GENERATION)
        else:
            package = json.loads(package_path.read_text(encoding="utf-8"))

        storyboard: List[Dict[str, Any]] = package["storyboard"]
        shorts_specs: List[Dict[str, Any]] = package.get("shorts", [])

        if guard.should_pause():
            manifest["state"] = JobState.PAUSED.value
            ckpt.save(manifest)
            return manifest

        # Stage 4: Narration (Kokoro-82M on CPU)
        norm_audio_path = assets.path("Narration", "narration_normalized.wav")
        audio_meta_path = assets.path("Narration", "audio_metadata.json")
        if not is_stage_done(manifest, JobState.NARRATION_GENERATION):
            tts = KokoroTTSGenerator(
                lang_code=models_cfg["tts"]["primary"]["lang_code"],
                voice=models_cfg["tts"]["primary"]["default_voice"],
                sample_rate=models_cfg["tts"]["primary"]["sample_rate"],
            )
            segments = [
                {"scene_id": s["scene_id"], "text": s["narration_segment"]}
                for s in storyboard
            ]
            audio_meta = tts.synthesize_narration(segments, assets.dirs["Narration"])
            normalize_audio(Path(audio_meta["combined_path"]), norm_audio_path)

            # Generate SRT subtitles from exact Kokoro chunk timings
            srt_path = assets.path("Subtitles", "longform.srt")
            generate_srt_from_chunks(audio_meta["chunks"], srt_path)
            checkpoint(JobState.NARRATION_GENERATION)
        else:
            audio_meta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
            srt_path = assets.path("Subtitles", "longform.srt")

        # Stage 5a: Image Generation (SD 1.5 + Title Cards)
        style_prefix = package.get("visual_continuity", {}).get("style_prefix", "")
        if not is_stage_done(manifest, JobState.IMAGE_GENERATION):
            img_gen = HybridImageGenerator(repo_id=models_cfg["image"]["primary"]["repo_id"])
            for scene in storyboard:
                if scene.get("requires_image_gen") and not scene.get("requires_video_gen"):
                    out_img = assets.path("Assets", scene["asset_filename"])
                    if out_img.exists():
                        continue
                    if scene.get("visual_type") == "title_card":
                        img_gen.generate_title_card(
                            heading=package["title"],
                            subheading=scene["narration_segment"][:90],
                            output_path=out_img,
                        )
                    else:
                        img_gen.generate_ai_image(
                            prompt=scene["visual_prompt"],
                            output_path=out_img,
                            style_prefix=style_prefix,
                        )
            img_gen.unload()
            checkpoint(JobState.IMAGE_GENERATION)

        if guard.should_pause():
            manifest["state"] = JobState.PAUSED.value
            ckpt.save(manifest)
            return manifest

        # Stage 5b: Video Generation (LTX-Video 2B Distilled, hero scenes only)
        if not is_stage_done(manifest, JobState.VIDEO_GENERATION):
            vid_cfg = models_cfg["video"]["primary"]
            vid_gen = LTXVideoGenerator(
                repo_id=vid_cfg["repo_id"],
                enable_cpu_offload=vid_cfg["enable_cpu_offload"],
            )
            for scene in storyboard:
                if scene.get("requires_video_gen"):
                    out_vid = assets.path("Assets", scene["asset_filename"])
                    if out_vid.exists():
                        continue
                    vid_gen.generate_clip(
                        prompt=f"{style_prefix}{scene['visual_prompt']}",
                        output_path=out_vid,
                        width=vid_cfg["width"],
                        height=vid_cfg["height"],
                        num_frames=vid_cfg["num_frames"],
                        num_inference_steps=vid_cfg["num_inference_steps"],
                        guidance_scale=vid_cfg["guidance_scale"],
                        fps=vid_cfg["fps"],
                    )
            vid_gen.unload()
            checkpoint(JobState.VIDEO_GENERATION)

        # Stage 6 & 7: Long-form Render
        longform_mp4 = assets.path("Long-Form", f"{job_id}_longform.mp4")
        if not is_stage_done(manifest, JobState.LONGFORM_RENDER):
            render_longform_video(
                storyboard=storyboard,
                assets_dir=assets.dirs["Assets"],
                audio_path=norm_audio_path,
                output_path=longform_mp4,
                work_dir=assets.work_dir,
                srt_path=srt_path,
                burn_subtitles=True,
            )
            # Long-form thumbnail
            first_img = next(
                (assets.path("Assets", s["asset_filename"]) for s in storyboard if str(s["asset_filename"]).endswith(".png")),
                None,
            )
            create_thumbnail(
                title=package["title"],
                output_path=assets.path("Thumbnails", f"{job_id}_longform_thumb.jpg"),
                background_image=first_img,
            )
            checkpoint(JobState.LONGFORM_RENDER)

        # Stage 8: Short-form Creation (3 vertical 9:16 shorts)
        short_outputs: List[Path] = []
        if not is_stage_done(manifest, JobState.SHORTS_RENDER):
            tts = KokoroTTSGenerator(
                lang_code=models_cfg["tts"]["primary"]["lang_code"],
                voice=models_cfg["tts"]["primary"]["default_voice"],
            )
            for s_spec in shorts_specs:
                sid = s_spec["short_id"]
                s_dir = assets.work_dir / f"narration_{sid}"
                s_meta = tts.synthesize_narration(
                    [{"scene_id": sid, "text": s_spec["narration_text"]}],
                    s_dir,
                )
                s_norm = s_dir / f"{sid}_norm.wav"
                normalize_audio(Path(s_meta["combined_path"]), s_norm)
                s_srt = assets.path("Subtitles", f"{sid}.srt")
                generate_srt_from_chunks(s_meta["chunks"], s_srt, max_line_width=24)

                s_mp4 = assets.path("Shorts", f"{job_id}_{sid}.mp4")
                render_short_video(
                    short_spec=s_spec,
                    storyboard=storyboard,
                    assets_dir=assets.dirs["Assets"],
                    audio_path=s_norm,
                    output_path=s_mp4,
                    work_dir=assets.work_dir,
                    srt_path=s_srt,
                    target_duration_sec=float(prod_cfg["target_short_duration_sec"]),
                )
                create_thumbnail(
                    title=s_spec["title"],
                    output_path=assets.path("Thumbnails", f"{job_id}_{sid}_thumb.jpg"),
                    size=(720, 1280),
                    badge_text="SHORTS",
                )
                short_outputs.append(s_mp4)
            checkpoint(JobState.SHORTS_RENDER)
        else:
            short_outputs = sorted(assets.dirs["Shorts"].glob("*.mp4"))

        # Stage 9: Quality Control
        if not is_stage_done(manifest, JobState.QUALITY_CHECK):
            qc_long = run_quality_control(longform_mp4, expected_aspect="16:9", min_duration_sec=5.0)
            qc_shorts = [
                run_quality_control(sp, expected_aspect="9:16", min_duration_sec=3.0)
                for sp in short_outputs
            ]
            manifest["qc"] = {"longform": qc_long, "shorts": qc_shorts}
            checkpoint(JobState.QUALITY_CHECK)

        # Stage 10: Google Drive Upload
        if upload_to_drive and not is_stage_done(manifest, JobState.DRIVE_UPLOAD):
            drive = GoogleDriveClient()
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            folders = drive.ensure_production_tree(date_str)
            uploaded: Dict[str, Any] = {}
            for sub in ["Long-Form", "Shorts", "Thumbnails", "Narration", "Subtitles", "Scripts", "Manifest"]:
                for file_path in assets.dirs[sub].glob("*"):
                    if file_path.is_file():
                        res = drive.upload_file(file_path, folders[sub])
                        uploaded[str(file_path.name)] = res["id"]
            manifest["output_file_ids"] = uploaded
            checkpoint(JobState.DRIVE_UPLOAD)

        manifest["state"] = JobState.COMPLETED.value
        ckpt.save(manifest)
        cf_client.update_job(job_id, {"state": JobState.COMPLETED.value, "manifest": manifest})
        return manifest

    except Exception as exc:
        manifest = record_error(manifest, JobState(manifest.get("state", "FAILED")), str(exc))
        ckpt.save(manifest)
        cf_client.update_job(job_id, {"state": JobState.FAILED.value, "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Video Factory production pipeline.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("outputs"))
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--shorts", type=int, default=3)
    parser.add_argument("--upload-drive", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        job_id=args.job_id,
        topic=args.topic,
        workspace_dir=args.workspace,
        target_duration_sec=args.duration,
        num_shorts=args.shorts,
        upload_to_drive=args.upload_drive,
    )
    print(json.dumps({"job_id": result["job_id"], "state": result["state"]}, indent=2))


if __name__ == "__main__":
    main()
