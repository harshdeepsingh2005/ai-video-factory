"""Fallback HunyuanVideo-1.5 480p pipeline with aggressive CPU offloading.

WARNING: Tencent Hunyuan Community License is NOT licensed for use in the
European Union (EU), United Kingdom (UK), or South Korea. Verify compliance
before enabling this fallback in any public or commercial workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from inference.video.ltx_pipeline import free_gpu_memory


class HunyuanVideoGenerator:
    """Experimental 480p HunyuanVideo-1.5 wrapper for 16 GB VRAM."""

    def __init__(
        self,
        repo_id: str = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.repo_id = repo_id
        self.dtype = dtype
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", "/tmp/hf_cache")
        self._pipe: Optional[DiffusionPipeline] = None

    def _load(self) -> DiffusionPipeline:
        if self._pipe is None:
            self._pipe = DiffusionPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
            )
            if hasattr(self._pipe, "vae"):
                if hasattr(self._pipe.vae, "enable_tiling"):
                    self._pipe.vae.enable_tiling()
                if hasattr(self._pipe.vae, "enable_slicing"):
                    self._pipe.vae.enable_slicing()
            if torch.cuda.is_available():
                # Sequential offload is slower but essential for 8.3B DiT on 16GB T4
                self._pipe.enable_sequential_cpu_offload()
        return self._pipe

    def generate_clip(
        self,
        prompt: str,
        output_path: Path,
        width: int = 832,
        height: int = 480,
        num_frames: int = 49,
        num_inference_steps: int = 12,
        fps: int = 24,
        seed: Optional[int] = 42,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load()
        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        )
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        export_to_video(result.frames[0], str(output_path), fps=fps)
        return output_path

    def unload(self) -> None:
        self._pipe = None
        free_gpu_memory()
