import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    family: str = Field(default="qwen2.5")
    parameters: str = Field(default="0.5B")
    context_length: int = Field(default=32768)
    quantization: str = Field(default="none")  # "none", "4bit", "8bit"
    torch_dtype: str = Field(default="float32")
    device: str = Field(default="cpu")


class LoRAConfigParams(BaseModel):
    r: int = Field(default=16)
    lora_alpha: int = Field(default=64)
    lora_dropout: float = Field(default=0.05)
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")


class TrainingConfig(BaseModel):
    output_dir: str = Field(default="artifacts/trained_adapter")
    epochs: int = Field(default=3)
    per_device_train_batch_size: int = Field(default=1)
    per_device_eval_batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=8)
    learning_rate: float = Field(default=2e-4)
    weight_decay: float = Field(default=0.01)
    optim: str = Field(default="adamw_torch")
    max_grad_norm: float = Field(default=1.0)
    max_seq_length: int = Field(default=512)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=50)
    seed: int = Field(default=42)
    use_gradient_checkpointing: bool = Field(default=True)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=False)


class EvaluationConfig(BaseModel):
    metric: str = Field(default="exact_match")
    task: str = Field(default="domain_instruction_following")
    batch_size: int = Field(default=4)
    max_new_tokens: int = Field(default=128)
    temperature: float = Field(default=0.0)
    fine_tuned_em_target: float = Field(default=0.70)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML file safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
