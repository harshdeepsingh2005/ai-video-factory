# Portable CUDA-compatible Linux GPU image for future self-hosted deployment.
# Note: Kaggle Notebooks do NOT support privileged Docker execution; use scripts/setup.sh on Kaggle.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip==24.3.1 && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /app/requirements.txt

COPY . /app

CMD ["python", "-m", "orchestration.pipeline_runner", "--help"]
