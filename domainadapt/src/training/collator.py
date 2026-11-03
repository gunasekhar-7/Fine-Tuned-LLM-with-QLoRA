from typing import Any, Dict, List
import torch


class SFTDataCollator:
    """Dynamically pads micro-batches of tokenized SFT samples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            attn_mask = item["attention_mask"]
            labels = item["labels"]
            pad_len = max_len - input_ids.shape[0]

            if pad_len > 0:
                padded_input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)]
                )
                padded_attn_mask = torch.cat(
                    [attn_mask, torch.zeros(pad_len, dtype=attn_mask.dtype)]
                )
                padded_labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=labels.dtype)]
                )
            else:
                padded_input_ids = input_ids
                padded_attn_mask = attn_mask
                padded_labels = labels

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attn_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
