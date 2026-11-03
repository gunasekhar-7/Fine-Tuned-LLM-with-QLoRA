import logging
import os
from pathlib import Path
from typing import Dict, Tuple
import torch
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import PreTrainedModel

logger = logging.getLogger("domainadapt.models")


def count_parameters(model: torch.nn.Module) -> Dict[str, int | float]:
    """Counts total and trainable parameters in a PyTorch model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percentage": round(pct, 4),
    }


def attach_lora_adapter(base_model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Wraps base model with PEFT LoRA adapter for training."""
    logger.info("Applying LoRA adapter to base model...")
    model = get_peft_model(base_model, lora_config)
    stats = count_parameters(model)
    logger.info(
        f"Adapter initialized. Trainable params: {stats['trainable_params']:,} / "
        f"{stats['total_params']:,} ({stats['trainable_percentage']}%)"
    )
    return model


def load_finetuned_adapter(
    base_model: PreTrainedModel,
    adapter_path: str | Path,
    is_trainable: bool = False,
) -> PeftModel:
    """Loads a previously trained LoRA adapter from disk onto the base model."""
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter path not found: {path}")

    config_file = path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing adapter_config.json at {path}")

    logger.info(f"Loading fine-tuned LoRA adapter weights from {path}...")
    model = PeftModel.from_pretrained(base_model, str(path), is_trainable=is_trainable)
    model.eval()
    return model
