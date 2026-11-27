"""Self-hosted Qwen2.5-1.5B-Instruct script, storyboard, and shorts generator.

Runs sequentially (and unloads cleanly) so GPU VRAM is free for video/image
models. Also supports CPU inference if GPU is reserved or unavailable.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.video.ltx_pipeline import free_gpu_memory

REQUIRED_SCENE_KEYS = {
    "scene_id",
    "timestamp",
    "narration_segment",
    "visual_prompt",
    "visual_type",
    "requires_video_gen",
    "requires_image_gen",
    "duration",
    "aspect_ratio",
    "subtitle_text",
    "transition",
    "asset_filename",
}


class ScriptGenerator:
    """Generates long-form script, structured storyboard, and 3 short-form scripts."""

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", "/tmp/hf_cache")
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id, cache_dir=self.cache_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            cache_dir=self.cache_dir,
        )
        if not use_cuda:
            self._model.to("cpu")

    def _chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 2048) -> str:
        self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        gen_ids = outputs[0][inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json(raw: str) -> Any:
        match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object or array found in LLM output.")
        return json.loads(match.group(1))

    @staticmethod
    def normalize_storyboard(
        scenes: List[Dict[str, Any]],
        max_video_clips: int = 6,
    ) -> List[Dict[str, Any]]:
        """Validate and normalize scene dicts to match Stage 3 schema."""
        normalized: List[Dict[str, Any]] = []
        current_sec = 0.0
        video_count = 0

        for idx, scene in enumerate(scenes, start=1):
            dur = float(scene.get("duration", 10.0))
            mins, secs = divmod(int(current_sec), 60)
            ts = scene.get("timestamp") or f"{mins:02d}:{secs:02d}"

            req_video = bool(scene.get("requires_video_gen", False))
            if req_video and video_count >= max_video_clips:
                req_video = False
            if req_video:
                video_count += 1

            req_image = bool(scene.get("requires_image_gen", not req_video))
            ext = "mp4" if req_video else "png"
            scene_id = str(scene.get("scene_id") or f"scene_{idx:03d}")
            narration = str(scene.get("narration_segment", "")).strip()

            normalized.append(
                {
                    "scene_id": scene_id,
                    "timestamp": ts,
                    "narration_segment": narration,
                    "visual_prompt": str(scene.get("visual_prompt", narration[:120])).strip(),
                    "visual_type": "video_clip" if req_video else str(scene.get("visual_type", "ai_image")),
                    "requires_video_gen": req_video,
                    "requires_image_gen": req_image,
                    "duration": dur,
                    "aspect_ratio": str(scene.get("aspect_ratio", "16:9")),
                    "subtitle_text": str(scene.get("subtitle_text", narration)).strip(),
                    "transition": str(scene.get("transition", "crossfade")),
                    "asset_filename": str(scene.get("asset_filename") or f"{scene_id}.{ext}"),
                }
            )
            current_sec += dur

        return normalized

    def generate_production_package(
        self,
        topic: str,
        audience: str = "curious general audience",
        tone: str = "engaging, clear, documentary",
        style: str = "cinematic, clean editorial visuals",
        target_duration_sec: int = 600,
        num_shorts: int = 3,
        max_video_clips: int = 6,
    ) -> Dict[str, Any]:
        """Generate full script, storyboard, metadata, and 3 short-form scripts."""
        num_scenes = max(6, min(40, target_duration_sec // 15))
        system_prompt = (
            "You are a senior documentary scriptwriter and video producer. "
            "Always respond with valid JSON only, no markdown commentary."
        )
        user_prompt = (
            f"Create a video production plan for topic: '{topic}'.\n"
            f"Target audience: {audience}. Tone: {tone}. Visual style: {style}.\n"
            f"Target duration: {target_duration_sec} seconds across {num_scenes} scenes.\n"
            f"Return a JSON object with keys:\n"
            f"- title (string)\n"
            f"- description (string)\n"
            f"- hashtags (array of 5 strings)\n"
            f"- visual_continuity (object with style_prefix and palette)\n"
            f"- scenes (array of {num_scenes} objects, each with: scene_id, narration_segment, "
            f"visual_prompt, visual_type ('video_clip'|'ai_image'|'title_card'), "
            f"requires_video_gen (boolean, true for at most {max_video_clips} hero scenes), "
            f"requires_image_gen (boolean), duration (number in seconds), transition ('crossfade'|'cut'))\n"
            f"- shorts (array of {num_shorts} objects, each with: short_id, title, hook, "
            f"narration_text, source_scene_ids (array), hashtags (array))."
        )

        raw = self._chat(system_prompt, user_prompt, max_new_tokens=3072)
        try:
            data = self._extract_json(raw)
        except Exception:
            # Deterministic fallback structure if 1.5B output had syntax noise
            data = self._fallback_package(topic, style, target_duration_sec, num_shorts)

        scenes = self.normalize_storyboard(
            data.get("scenes", []), max_video_clips=max_video_clips
        )
        if not scenes:
            data = self._fallback_package(topic, style, target_duration_sec, num_shorts)
            scenes = self.normalize_storyboard(data["scenes"], max_video_clips=max_video_clips)

        full_narration = "\n\n".join(s["narration_segment"] for s in scenes if s["narration_segment"])

        return {
            "topic": topic,
            "title": data.get("title", topic),
            "description": data.get("description", f"Deep dive into {topic}."),
            "hashtags": data.get("hashtags", ["#ai", "#documentary", "#science"]),
            "visual_continuity": data.get(
                "visual_continuity",
                {"style_prefix": f"{style}, consistent lighting, ", "palette": "deep slate and cyan"},
            ),
            "full_narration": full_narration,
            "storyboard": scenes,
            "shorts": data.get("shorts", [])[:num_shorts],
        }

    @staticmethod
    def _fallback_package(
        topic: str, style: str, target_duration_sec: int, num_shorts: int
    ) -> Dict[str, Any]:
        scene_dur = max(10.0, target_duration_sec / 6.0)
        scenes = []
        for i in range(1, 7):
            is_hero = i in (1, 4)
            scenes.append(
                {
                    "scene_id": f"scene_{i:03d}",
                    "narration_segment": f"Part {i} exploring {topic} in detail.",
                    "visual_prompt": f"{style}, visual representation of {topic}, part {i}",
                    "visual_type": "video_clip" if is_hero else "ai_image",
                    "requires_video_gen": is_hero,
                    "requires_image_gen": not is_hero,
                    "duration": scene_dur,
                    "transition": "crossfade",
                }
            )
        shorts = [
            {
                "short_id": f"short_{j}",
                "title": f"{topic} — Key Insight #{j}",
                "hook": f"Here is one surprising fact about {topic}.",
                "narration_text": f"Here is one surprising fact about {topic}. Part {j} explains why it matters.",
                "source_scene_ids": [f"scene_{j:03d}"],
                "hashtags": ["#shorts", "#explained"],
            }
            for j in range(1, num_shorts + 1)
        ]
        return {
            "title": topic,
            "description": f"An overview of {topic}.",
            "hashtags": ["#explained", "#deepdive"],
            "scenes": scenes,
            "shorts": shorts,
        }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        free_gpu_memory()
