from typing import Any, Dict, List, Optional
import torch
from transformers import PreTrainedTokenizerBase


class SFTChatDataProcessor:
    """Tokenizes and masks instruction-response pairs for Supervised Fine-Tuning (SFT).

    Ensures that loss is calculated strictly on the assistant's domain response,
    masking the user instruction with -100.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def format_conversation(self, instruction: str, response: Optional[str] = None) -> List[Dict[str, str]]:
        messages = [{"role": "user", "content": instruction.strip()}]
        if response is not None:
            messages.append({"role": "assistant", "content": response.strip()})
        return messages

    def _extract_tensor(self, encoded: Any) -> torch.Tensor:
        """Safely extracts 1D input_ids tensor from BatchEncoding or Tensor."""
        if hasattr(encoded, "get") and "input_ids" in encoded:
            tensor = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            tensor = encoded.input_ids
        elif isinstance(encoded, torch.Tensor):
            tensor = encoded
        else:
            raise TypeError(f"Unexpected tokenization output type: {type(encoded)}")

        if tensor.dim() > 1:
            tensor = tensor[0]
        return tensor

    def process_train_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        instruction = example["instruction"]
        response = example["expected_output"]
        if isinstance(response, dict):
            import json
            response = json.dumps(response, ensure_ascii=False)

        prompt_messages = [{"role": "user", "content": instruction.strip()}]
        full_messages = [
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": response.strip()},
        ]

        raw_prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_tokens = self._extract_tensor(raw_prompt)

        raw_full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        full_tokens = self._extract_tensor(raw_full)

        if len(full_tokens) > self.max_seq_length:
            full_tokens = full_tokens[: self.max_seq_length]

        prompt_len = min(len(prompt_tokens), len(full_tokens))

        labels = full_tokens.clone()
        labels[:prompt_len] = -100

        attention_mask = torch.ones_like(full_tokens)

        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def process_inference_input(self, instruction: str) -> Dict[str, torch.Tensor]:
        messages = [{"role": "user", "content": instruction.strip()}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = self._extract_tensor(encoded).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
