from typing import List, Optional
from peft import LoraConfig, TaskType


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """Constructs a validated PEFT LoraConfig for causal language modeling."""
    modules = target_modules or DEFAULT_TARGET_MODULES
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
