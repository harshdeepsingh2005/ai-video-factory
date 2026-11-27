"""Config, storyboard schema, and GPU-free visual asset tests."""

from pathlib import Path
import yaml

from inference.llm.script_generator import REQUIRED_SCENE_KEYS, ScriptGenerator
from inference.video.image_pipeline import HybridImageGenerator


def test_configs_valid_and_zero_cost():
    models = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    limits = yaml.safe_load(Path("configs/free_tier_limits.yaml").read_text(encoding="utf-8"))
    prod = yaml.safe_load(Path("configs/production.yaml").read_text(encoding="utf-8"))

    assert models["video"]["primary"]["repo_id"] == "Lightricks/LTX-Video"
    assert models["llm"]["primary"]["repo_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert models["tts"]["primary"]["repo_id"] == "hexgrad/Kokoro-82M"
    assert limits["cloudflare_free"]["r2_excluded"] is True
    assert limits["kaggle"]["requires_credit_card"] is False
    assert prod["production"]["num_shorts"] == 3


def test_storyboard_normalization_caps_video_clips():
    raw_scenes = [
        {"narration_segment": f"Narration {i}.", "requires_video_gen": True, "duration": 5}
        for i in range(10)
    ]
    norm = ScriptGenerator.normalize_storyboard(raw_scenes, max_video_clips=3)
    assert len(norm) == 10
    assert sum(1 for s in norm if s["requires_video_gen"]) == 3
    for s in norm:
        assert REQUIRED_SCENE_KEYS.issubset(s.keys())


def test_title_card_and_chart_generation(tmp_path: Path):
    gen = HybridImageGenerator()
    card = gen.generate_title_card("Test Heading", "Subtitle", tmp_path / "card.png")
    chart = gen.generate_chart("Test Chart", ["A", "B"], [10, 20], tmp_path / "chart.png")
    assert card.exists() and card.stat().st_size > 1000
    assert chart.exists() and chart.stat().st_size > 1000
