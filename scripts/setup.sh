#!/usr/bin/env bash
set -euo pipefail

echo "=== AI Video Factory: Environment Setup ==="

# 1. System packages (espeak-ng for Kokoro-82M phonemizer, ffmpeg for media processing)
if command -v apt-get >/dev/null 2>&1; then
    echo "[1/4] Installing system dependencies (ffmpeg, espeak-ng, libsndfile1)..."
    sudo apt-get update -qq || apt-get update -qq
    sudo apt-get install -y -qq ffmpeg espeak-ng libsndfile1 || apt-get install -y -qq ffmpeg espeak-ng libsndfile1
else
    echo "[1/4] apt-get not found; assuming ffmpeg and espeak-ng are pre-installed."
fi

# 2. Verify NVIDIA GPU & Driver (if present)
echo "[2/4] Checking GPU & NVIDIA driver..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Running in CPU-only mode."
fi

# 3. Install pinned Python dependencies
echo "[3/4] Installing pinned Python dependencies from requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -q --no-cache-dir -r "$REPO_ROOT/requirements.txt"

# 4. Verify PyTorch CUDA & FFmpeg
echo "[4/4] Verifying PyTorch CUDA and FFmpeg..."
python3 - << 'EOF'
import shutil
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
print(f"FFmpeg path: {shutil.which('ffmpeg')}")
print(f"FFprobe path: {shutil.which('ffprobe')}")
EOF

echo "=== Setup complete ==="
