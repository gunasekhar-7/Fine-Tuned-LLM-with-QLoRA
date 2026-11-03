import logging
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger("domainadapt.models")


def get_optimal_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Detects available hardware and chooses optimal device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def load_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> PreTrainedTokenizerBase:
    """Loads tokenizer and ensures pad_token is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quantization: str = "auto",  # "auto", "4bit", "none"
    device: Optional[str] = None,
) -> PreTrainedModel:
    """Loads the base causal language model.

    If quantization is requested (or auto with CUDA available), configures 4-bit NormalFloat (NF4)
    quantization for true QLoRA. Otherwise, loads in native precision on CPU/device.
    """
    detected_device, default_dtype = get_optimal_device_and_dtype()
    target_device = device or detected_device

    quant_config = None
    if quantization == "4bit" or (quantization == "auto" and target_device == "cuda"):
        if not torch.cuda.is_available():
            logger.warning("4-bit quantization requested but CUDA is not available. Falling back to CPU float32.")
            target_device = "cpu"
            torch_dtype = torch.float32
        else:
            logger.info("Initializing 4-bit NormalFloat (NF4) quantization for QLoRA.")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=default_dtype,
            )
            torch_dtype = default_dtype
    else:
        torch_dtype = default_dtype

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = target_device

    logger.info(f"Loading base model {model_name} on {target_device} ({torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model
