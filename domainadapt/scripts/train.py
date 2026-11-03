#!/usr/bin/env python3
"""Training entrypoint for DomainAdapt QLoRA/LoRA domain adaptation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from domainadapt.src.utils.logger import setup_logger
from domainadapt.src.data.loader import load_jsonl
from domainadapt.src.data.tokenizer import SFTChatDataProcessor
from domainadapt.src.models.base_model import load_base_model, load_tokenizer, get_optimal_device_and_dtype
from domainadapt.src.models.lora_config import build_lora_config
from domainadapt.src.models.adapter import attach_lora_adapter, count_parameters
from domainadapt.src.training.trainer import SFTTrainer

logger = setup_logger("domainadapt.train", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Train DomainAdapt LoRA/QLoRA adapter")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "artifacts" / "trained_adapter"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit", "auto"])
    args = parser.parse_args()

    device, dtype = get_optimal_device_and_dtype()
    logger.info(f"Target hardware: device={device}, dtype={dtype}")

    # 1. Load Tokenizer & Base Model
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(
        model_name=args.model_name,
        quantization=args.quantization,
        device=device,
    )

    # 2. Attach LoRA Adapter
    lora_cfg = build_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha)
    model = attach_lora_adapter(base_model, lora_cfg)

    # 3. Load & Preprocess Datasets
    data_dir = Path(args.data_dir)
    train_raw = load_jsonl(data_dir / "train.jsonl")
    val_raw = load_jsonl(data_dir / "val.jsonl")

    logger.info(f"Loaded {len(train_raw)} train samples and {len(val_raw)} validation samples")

    processor = SFTChatDataProcessor(tokenizer, max_seq_length=args.max_seq_len)
    train_data = [processor.process_train_example(ex) for ex in train_raw]
    val_data = [processor.process_train_example(ex) for ex in val_raw]

    # 4. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        device=device,
    )

    summary = trainer.train()
    logger.info(f"Training completed successfully! Summary saved to {args.output_dir}/training_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
