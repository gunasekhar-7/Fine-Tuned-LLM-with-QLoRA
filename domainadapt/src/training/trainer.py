import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import PreTrainedTokenizerBase

from domainadapt.src.training.collator import SFTDataCollator
from domainadapt.src.models.adapter import count_parameters

logger = logging.getLogger("domainadapt.training")


class SFTTrainer:
    """Supervised Fine-Tuning trainer with gradient accumulation and validation."""

    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: List[Dict[str, torch.Tensor]],
        eval_dataset: Optional[List[Dict[str, torch.Tensor]]] = None,
        output_dir: str | Path = "artifacts/trained_adapter",
        epochs: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.05,
        logging_steps: int = 5,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulation_steps = max(1, gradient_accumulation_steps)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.seed = seed
        self.device = device

        self.collator = SFTDataCollator(pad_token_id=self.tokenizer.pad_token_id)
        self._set_seed(self.seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self) -> Dict[str, Any]:
        """Runs the full SFT training loop."""
        logger.info("Initializing optimizer and scheduler...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found in model!")

        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_batches = len(self.train_dataset) // self.batch_size
        total_update_steps = (total_batches // self.accumulation_steps) * self.epochs
        total_update_steps = max(1, total_update_steps)
        warmup_steps = int(total_update_steps * self.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        global_step = 0
        update_step = 0
        start_time = time.time()

        param_stats = count_parameters(self.model)
        logger.info(
            f"Beginning training for {self.epochs} epochs. Total update steps: {total_update_steps}"
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            self.model.train()
            indices = list(range(len(self.train_dataset)))
            random.Random(self.seed + epoch).shuffle(indices)

            running_loss = 0.0
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[step_idx : step_idx + self.batch_size]
                raw_batch = [self.train_dataset[i] for i in batch_indices]
                batch = self.collator(raw_batch)

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()

                running_loss += loss.item()
                epoch_loss += loss.item()
                global_step += 1

                if global_step % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update_step += 1

                    if update_step % self.logging_steps == 0 or update_step == total_update_steps:
                        avg_step_loss = running_loss / self.accumulation_steps
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"Epoch {epoch}/{self.epochs} | Step {update_step}/{total_update_steps} | "
                            f"Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e}"
                        )
                        running_loss = 0.0

            # Remaining gradients at the end of epoch
            if global_step % self.accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

            avg_train_loss = epoch_loss / max(1, len(indices) // self.batch_size)

            # Evaluate on validation set
            val_loss = self.evaluate()
            epoch_duration = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch} finished in {epoch_duration:.1f}s | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "duration_seconds": round(epoch_duration, 2),
            }
            history.append(epoch_record)

            # Checkpoint per epoch
            ckpt_dir = self.output_dir / f"checkpoint-epoch{epoch}"
            self.model.save_pretrained(str(ckpt_dir))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = self.output_dir / "best_model"
                self.model.save_pretrained(str(best_dir))

        # Save final model
        logger.info(f"Saving final trained adapter to {self.output_dir}...")
        self.model.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))

        total_duration = time.time() - start_time
        summary = {
            "trainable_parameters": param_stats["trainable_params"],
            "total_parameters": param_stats["total_params"],
            "trainable_percentage": param_stats["trainable_percentage"],
            "epochs": self.epochs,
            "total_update_steps": update_step,
            "best_val_loss": round(best_val_loss, 4),
            "total_duration_seconds": round(total_duration, 2),
            "history": history,
        }

        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def evaluate(self) -> float:
        """Computes average loss on the validation dataset."""
        if not self.eval_dataset:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for step_idx in range(0, len(self.eval_dataset), self.batch_size):
                raw_batch = self.eval_dataset[step_idx : step_idx + self.batch_size]
                batch = self.collator(raw_batch)

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                batches += 1

        self.model.train()
        return total_loss / max(1, batches)
