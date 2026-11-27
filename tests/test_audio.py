"""Audio chunking and SRT subtitle generation tests."""

from pathlib import Path

from inference.audio.kokoro_tts import split_into_chunks
from rendering.subtitles import format_srt_time, generate_srt_from_chunks


def test_split_into_chunks_respects_sentence_boundaries():
    text = "First sentence here. Second sentence is slightly longer! Third sentence?"
    chunks = split_into_chunks(text, max_chars=45)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)


def test_srt_formatting_and_generation(tmp_path: Path):
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(65.25) == "00:01:05,250"

    srt_path = generate_srt_from_chunks(
        [
            {"start_sec": 0.0, "duration_sec": 2.5, "text": "Hello world from AI Video Factory."},
            {"start_sec": 2.75, "duration_sec": 3.0, "text": "Second subtitle line."},
        ],
        tmp_path / "test.srt",
    )
    content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,500" in content
    assert "Second subtitle line." in content
