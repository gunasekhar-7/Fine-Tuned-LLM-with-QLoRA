"""Unit tests for model architecture, LoRA configuration, and hardware utils."""

import torch
import torch.nn as nn
from domainadapt.src.models.base_model import get_optimal_device_and_dtype
from domainadapt.src.models.lora_config import build_lora_config, DEFAULT_TARGET_MODULES
from domainadapt.src.models.adapter import count_parameters


def test_hardware_detection():
    device, dtype = get_optimal_device_and_dtype()
    assert device in ["cpu", "cuda"]
    assert dtype in [torch.float32, torch.float16, torch.bfloat16]


def test_lora_config_builder():
    config = build_lora_config(r=16, lora_alpha=64, lora_dropout=0.1)
    assert config.r == 16
    assert config.lora_alpha == 64
    assert config.lora_dropout == 0.1
    assert set(config.target_modules) == set(DEFAULT_TARGET_MODULES)


def test_count_parameters():
    model = nn.Sequential(
        nn.Linear(10, 20),  # 220 params
        nn.Linear(20, 5),   # 105 params
    )
    # Freeze first layer
    for p in model[0].parameters():
        p.requires_grad = False

    stats = count_parameters(model)
    assert stats["total_params"] == 325
    assert stats["trainable_params"] == 105
    assert 0 < stats["trainable_percentage"] < 100
