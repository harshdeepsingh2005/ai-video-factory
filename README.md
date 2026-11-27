# AI Video Factory (₹0 / $0 Self-Hosted Open-Weight Pipeline)

Experimental, cloud-only, zero-budget AI video production pipeline that generates:
- **1× ~10-minute long-form video** (16:9 H.264/AAC MP4)
- **3× short-form vertical videos** (9:16 reframed with independent hooks & burned-in subtitles)
- **Narration, subtitles, thumbnails, scripts, storyboards, and production manifests**
- **Automated organization & upload to Google Drive**

## Strict ₹0 / $0 Constraint
- **Compute:** Kaggle Notebooks (Free tier: 2× NVIDIA T4 16 GB VRAM, 30 hrs/week, 12h max session) as primary; Google Colab Free (1× T4 16 GB) as secondary.
- **Orchestration & State:** Cloudflare Workers + D1 + Cron Triggers (Free tier, no credit card required). **Cloudflare R2 is intentionally excluded** because it requires a credit card on file.
- **Storage:** Google Drive (15 GB free tier via OAuth2 Standard Tier, no GCP billing account required), Kaggle Datasets/Working (20 GB), GitHub.
- **Models (100% self-hosted open-weight):**
  - **Video (Primary):** `Lightricks/LTX-Video` (`ltxv-2b-0.9.8-distilled`)
  - **Video (Fallback/Experimental):** `tencent/HunyuanVideo-1.5` (`480p` distilled, subject to Tencent Hunyuan Community License territory restrictions)
  - **Image (Hybrid stills):** `stable-diffusion-v1-5/stable-diffusion-v1-5`
  - **LLM:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0)
  - **TTS:** `hexgrad/Kokoro-82M` (Apache 2.0, CPU inference)
  - **Media Engine:** FFmpeg

## Milestone 1: Single Clip Verification on Kaggle
1. Upload `notebooks/kaggle_production.ipynb` to Kaggle.
2. Enable Accelerator: **GPU T4 x2** and **Internet: On**.
3. Run the Milestone 1 cells to execute `scripts/setup.sh` and `scripts/benchmark.py` — generating a single short AI video clip using `ltxv-2b-0.9.8-distilled` with zero paid APIs.

## Repository Layout
- `configs/` — Pinned model IDs, production defaults, and free-tier quota guardrails.
- `scripts/` — Kaggle `setup.sh`, `download_models.py`, and `benchmark.py`.
- `inference/` — Sequential, memory-safe wrappers for LTX-Video, HunyuanVideo 1.5, SD 1.5, Qwen2.5-1.5B, and Kokoro-82M.
- `rendering/` — FFmpeg scene assembly, Ken Burns motion, subtitles, thumbnails, 16:9 long-form, and 9:16 shorts reframing.
- `storage/` — Google Drive OAuth2 resumable uploader and local stage checkpoint manager.
- `orchestration/` — 15-state resumable pipeline runner, job manager, and crash recovery.
- `cloudflare/` — Free-tier Worker API, D1 schema, and Cron Trigger configuration.
- `notebooks/` — `kaggle_production.ipynb` for cloud GPU execution.
- `tests/` — Unit & integration tests (non-GPU tests runnable locally; GPU tests marked for Kaggle).
