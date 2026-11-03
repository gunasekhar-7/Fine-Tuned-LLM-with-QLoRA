"""Integration tests for DomainAdapt model components and training abstractions."""

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from domainadapt.src.models.base_model import load_base_model
from domainadapt.src.models.lora_config import build_lora_config
from domainadapt.src.models.adapter import attach_lora_adapter, count_parameters
from domainadapt.src.data.loader import load_jsonl, check_split_leakage

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_lora_adapter_attachment_and_freezing():
    """Verify that attaching a LoRA adapter properly freezes base weights and trains adapter only."""
    base_model = load_base_model("Qwen/Qwen2.5-0.5B-Instruct", quantization="none", device="cpu")
    lora_cfg = build_lora_config(r=16, lora_alpha=32)

    peft_model = attach_lora_adapter(base_model, lora_cfg)
    peft_model.eval()

    stats = count_parameters(peft_model)
    assert stats["trainable_params"] > 0, "Adapter must have trainable parameters"
    assert stats["trainable_params"] < stats["total_params"], "Base model weights must remain frozen"
    assert stats["trainable_percentage"] < 5.0, "LoRA adapter should train < 5% of total parameters"

    # Explicit check on base vs adapter parameters
    for name, param in peft_model.named_parameters():
        if "lora" in name.lower():
            assert param.requires_grad is True, f"LoRA parameter {name} should be trainable"
        else:
            assert param.requires_grad is False, f"Base parameter {name} should be frozen"


def test_clean_split_leakage_invariance():
    """Verify that train, val, and test partitions have zero overlap."""
    train_data = load_jsonl(PROJECT_ROOT / "data" / "processed" / "train.jsonl")
    val_data = load_jsonl(PROJECT_ROOT / "data" / "processed" / "val.jsonl")
    test_data = load_jsonl(PROJECT_ROOT / "data" / "processed" / "test.jsonl")

    leakage = check_split_leakage(train_data, val_data, test_data)
    assert len(leakage["train_val"]) == 0
    assert len(leakage["train_test"]) == 0
    assert len(leakage["val_test"]) == 0
