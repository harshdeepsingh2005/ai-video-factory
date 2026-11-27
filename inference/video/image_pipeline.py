"""Hybrid visual asset generator: Stable Diffusion 1.5 + programmatic charts/cards.

Most scenes in a 10-minute video are rendered from high-res stills with Ken Burns
motion or clean data graphics rather than expensive video diffusion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from inference.video.ltx_pipeline import free_gpu_memory


class HybridImageGenerator:
    """Generates AI still images (SD 1.5) and programmatic charts/title cards."""

    def __init__(
        self,
        repo_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", "/tmp/hf_cache")
        self._pipe = None

    def _load_sd15(self):
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionPipeline

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.repo_id,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                safety_checker=None,
            )
            if torch.cuda.is_available():
                self._pipe.to("cuda")
                self._pipe.enable_attention_slicing()
        return self._pipe

    def generate_ai_image(
        self,
        prompt: str,
        output_path: Path,
        style_prefix: str = "cinematic, high detail, consistent lighting, 4k, ",
        negative_prompt: str = "text, watermark, blurry, deformed, low quality",
        width: int = 768,
        height: int = 432,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = 42,
    ) -> Path:
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load_sd15()
        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        )
        image = pipe(
            prompt=f"{style_prefix}{prompt}",
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.save(output_path)
        return output_path

    def generate_chart(
        self,
        title: str,
        labels: List[str],
        values: List[float],
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
    ) -> Path:
        """Render a dark-mode data chart without needing a GPU."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        bars = ax.bar(labels, values, color="#38bdf8", width=0.55)
        ax.set_title(title, color="#f8fafc", fontsize=28, pad=24, fontweight="bold")
        ax.tick_params(colors="#cbd5e1", labelsize=18)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:g}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color="#f8fafc",
                fontsize=16,
            )

        plt.tight_layout(pad=3.0)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path

    def generate_title_card(
        self,
        heading: str,
        subheading: str,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
    ) -> Path:
        """Render a clean, GPU-free title/section card using Pillow."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

        # Accent bar
        draw.rectangle([120, height // 2 - 90, 136, height // 2 + 90], fill=(56, 189, 248))
        draw.text((170, height // 2 - 60), heading[:80], fill=(248, 250, 252), font=font_lg)
        if subheading:
            draw.text((170, height // 2 + 10), subheading[:120], fill=(148, 163, 184), font=font_sm)

        img.save(output_path)
        return output_path

    def unload(self) -> None:
        self._pipe = None
        free_gpu_memory()
