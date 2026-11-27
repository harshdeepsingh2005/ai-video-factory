"""LTX-Video 2B Distilled pipeline for 16 GB VRAM (NVIDIA T4) GPUs.

Uses Lightricks/LTX-Video with VAE tiling, VAE slicing, and sequential CPU
offloading so peak VRAM stays safely within 16 GB.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Optional


def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class LTXVideoGenerator:
    """Sequential, memory-bounded LTX-Video generator."""

    def __init__(
        self,
        repo_id: str = "Lightricks/LTX-Video",
        dtype: Optional[Any] = None,
        enable_cpu_offload: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        if dtype is None:
            import torch
            dtype = torch.float16
        self.repo_id = repo_id
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", "/tmp/hf_cache")
        self._t2v_pipe: Optional[Any] = None
        self._i2v_pipe: Optional[Any] = None

    def _configure_memory(self, pipe: Any) -> None:
        import torch

        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
        if self.enable_cpu_offload and torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        elif torch.cuda.is_available():
            pipe.to("cuda")

    def _load_t2v(self) -> Any:
        from diffusers import LTXPipeline

        if self._i2v_pipe is not None:
            del self._i2v_pipe
            self._i2v_pipe = None
            free_gpu_memory()
        if self._t2v_pipe is None:
            self._t2v_pipe = LTXPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
            )
            self._configure_memory(self._t2v_pipe)
        return self._t2v_pipe

    def _load_i2v(self) -> Any:
        from diffusers import LTXImageToVideoPipeline

        if self._t2v_pipe is not None:
            del self._t2v_pipe
            self._t2v_pipe = None
            free_gpu_memory()
        if self._i2v_pipe is None:
            self._i2v_pipe = LTXImageToVideoPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
            )
            self._configure_memory(self._i2v_pipe)
        return self._i2v_pipe

    def generate_clip(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        width: int = 704,
        height: int = 480,
        num_frames: int = 49,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        fps: int = 24,
        seed: Optional[int] = 42,
        init_image_path: Optional[Path] = None,
    ) -> Path:
        """Generate a short MP4 clip via T2V or I2V and return its path."""
        import torch
        from diffusers.utils import export_to_video, load_image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        )

        if init_image_path is not None:
            pipe = self._load_i2v()
            image = load_image(str(init_image_path)).resize((width, height))
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            pipe = self._load_t2v()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        frames = result.frames[0]
        export_to_video(frames, str(output_path), fps=fps)
        return output_path

    def unload(self) -> None:
        """Release model weights from VRAM/RAM before running the next stage."""
        self._t2v_pipe = None
        self._i2v_pipe = None
        free_gpu_memory()
