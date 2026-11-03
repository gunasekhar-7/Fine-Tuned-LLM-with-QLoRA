import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from domainadapt.src.data.tokenizer import SFTChatDataProcessor
from domainadapt.src.inference.validator import extract_json_payload, validate_domain_json, TroubleshootingPayload

logger = logging.getLogger("domainadapt.inference")


class GenerationResult(BaseModel):
    instruction: str
    generated_text: str
    parsed_json: Optional[Dict[str, Any]] = None
    is_schema_valid: bool = False
    validation_error: Optional[str] = None
    latency_seconds: float
    tokens_generated: int
    adapter_enabled: bool = True


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    device: Optional[str] = None,
) -> str:
    """Convenience function for generating response text from a single instruction."""
    target_device = device or next(model.parameters()).device
    processor = SFTChatDataProcessor(tokenizer)
    inputs = processor.process_inference_input(instruction)

    input_ids = inputs["input_ids"].to(target_device)
    attention_mask = inputs["attention_mask"].to(target_device)

    do_sample = temperature > 0.0
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(**gen_kwargs)

    prompt_length = input_ids.shape[1]
    generated_ids = output[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class DomainInferenceEngine:
    """Production inference engine for domain troubleshooting generation and validation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[str] = None,
        adapter_enabled: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.adapter_enabled = adapter_enabled
        self.processor = SFTChatDataProcessor(tokenizer)

    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> GenerationResult:
        start_time = time.perf_counter()
        inputs = self.processor.process_inference_input(instruction)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        do_sample = temperature > 0.0
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)

        prompt_length = input_ids.shape[1]
        new_tokens = output[0][prompt_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        latency = time.perf_counter() - start_time

        parsed_json = extract_json_payload(generated_text)
        is_valid, payload, err = validate_domain_json(parsed_json)

        return GenerationResult(
            instruction=instruction,
            generated_text=generated_text,
            parsed_json=payload.model_dump() if payload else parsed_json,
            is_schema_valid=is_valid,
            validation_error=err if not is_valid else None,
            latency_seconds=round(latency, 4),
            tokens_generated=len(new_tokens),
            adapter_enabled=self.adapter_enabled,
        )

    def batch_generate(
        self,
        instructions: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> List[GenerationResult]:
        return [
            self.generate(inst, max_new_tokens=max_new_tokens, temperature=temperature)
            for inst in instructions
        ]
