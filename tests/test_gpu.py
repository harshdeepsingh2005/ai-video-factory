"""GPU verification tests (intended for Kaggle T4 x2 / Colab Free T4)."""

import pytest


def test_cuda_available_and_vram():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host (expected on MacBook dev machine).")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    assert vram_gb >= 14.0, f"Expected >= 14 GB VRAM (T4), got {vram_gb:.2f} GB"
