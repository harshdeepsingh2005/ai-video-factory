"""Thumbnail generation for 16:9 long-form and 9:16 short-form videos."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


def create_thumbnail(
    title: str,
    output_path: Path,
    background_image: Optional[Path] = None,
    size: Tuple[int, int] = (1280, 720),
    badge_text: str = "AI VIDEO FACTORY",
) -> Path:
    """Create a high-contrast thumbnail from a scene still or dark gradient."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = size
    if background_image and Path(background_image).exists():
        img = Image.open(background_image).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        img = ImageEnhance.Brightness(img).enhance(0.55)
    else:
        img = Image.new("RGB", (w, h), color=(15, 23, 42))

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Bottom gradient-like dark band for legibility
    band_top = int(h * 0.55)
    draw.rectangle([0, band_top, w, h], fill=(10, 15, 28))

    if badge_text:
        draw.rectangle([48, 48, 48 + len(badge_text) * 8 + 24, 84], fill=(56, 189, 248))
        draw.text((60, 58), badge_text, fill=(15, 23, 42), font=font)

    wrapped = textwrap.fill(title.upper(), width=28 if w > h else 18)
    draw.multiline_text((48, band_top + 36), wrapped, fill=(248, 250, 252), font=font, spacing=8)

    img.save(output_path, quality=92)
    return output_path
