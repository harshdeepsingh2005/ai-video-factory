"""Kokoro-82M CPU Text-to-Speech generator (Apache 2.0).

Runs on CPU (~0.46-0.50 RTF) so GPU VRAM remains 100% free. Splits long
narration into chunks, synthesizes at 24 kHz, and concatenates cleanly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf


def split_into_chunks(text: str, max_chars: int = 350) -> List[str]:
    """Split narration on sentence boundaries for stable Kokoro synthesis."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


class KokoroTTSGenerator:
    """Chunked Kokoro-82M CPU narration synthesizer."""

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        sample_rate: int = 24000,
    ) -> None:
        self.lang_code = lang_code
        self.voice = voice
        self.sample_rate = sample_rate
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self.lang_code, device="cpu")
        return self._pipeline

    def synthesize_narration(
        self,
        segments: List[Dict[str, str]],
        output_dir: Path,
        pause_sec: float = 0.25,
    ) -> Dict[str, object]:
        """Synthesize per-scene chunks and concatenate into combined_narration.wav."""
        output_dir = Path(output_dir)
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        pipe = self._load()
        pause_samples = np.zeros(int(self.sample_rate * pause_sec), dtype=np.float32)

        all_audio: List[np.ndarray] = []
        chunk_records: List[Dict[str, object]] = []
        cursor_sec = 0.0

        for seg in segments:
            scene_id = seg["scene_id"]
            text = seg["text"].strip()
            if not text:
                continue

            seg_audio_parts: List[np.ndarray] = []
            for sub_idx, chunk_text in enumerate(split_into_chunks(text)):
                for _, _, audio in pipe(chunk_text, voice=self.voice, speed=1.0):
                    arr = np.asarray(audio, dtype=np.float32)
                    seg_audio_parts.append(arr)

            if not seg_audio_parts:
                continue

            seg_wav = np.concatenate(seg_audio_parts)
            chunk_path = chunks_dir / f"{scene_id}.wav"
            sf.write(str(chunk_path), seg_wav, self.sample_rate)

            dur = len(seg_wav) / float(self.sample_rate)
            chunk_records.append(
                {
                    "scene_id": scene_id,
                    "chunk_path": str(chunk_path),
                    "start_sec": round(cursor_sec, 3),
                    "duration_sec": round(dur, 3),
                    "text": text,
                }
            )
            all_audio.append(seg_wav)
            all_audio.append(pause_samples)
            cursor_sec += dur + pause_sec

        combined = np.concatenate(all_audio) if all_audio else np.zeros(self.sample_rate, dtype=np.float32)
        combined_path = output_dir / "combined_narration.wav"
        sf.write(str(combined_path), combined, self.sample_rate)

        meta = {
            "sample_rate": self.sample_rate,
            "voice": self.voice,
            "total_duration_sec": round(len(combined) / float(self.sample_rate), 3),
            "combined_path": str(combined_path),
            "chunks": chunk_records,
        }
        meta_path = output_dir / "audio_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
