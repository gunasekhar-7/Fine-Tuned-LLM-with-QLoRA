"""Unit tests for DomainAdapt data loading, preprocessing, and SFT tokenization."""

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

import domainadapt
from domainadapt.src.data.loader import load_jsonl, check_split_leakage
from domainadapt.src.data.preprocessor import validate_example_schema, REQUIRED_KEYS
from domainadapt.src.data.tokenizer import SFTChatDataProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_domainadapt_import():
    assert domainadapt.__version__ is not None


def test_dataset_files_exist():
    data_dir = PROJECT_ROOT / "data"
    assert (data_dir / "raw" / "domain_instructions.jsonl").exists()
    assert (data_dir / "processed" / "train.jsonl").exists()
    assert (data_dir / "processed" / "val.jsonl").exists()
    assert (data_dir / "processed" / "test.jsonl").exists()


def test_data_integrity():
    for split in ["train", "val", "test"]:
        file_path = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
        records = load_jsonl(file_path)
        assert len(records) > 0, f"Split {split} is empty"

        for idx, rec in enumerate(records):
            is_valid, msg = validate_example_schema(rec)
            assert is_valid, f"Error in {split} sample {idx}: {msg}"


def test_no_split_leakage_strict():
    train_data = load_jsonl(PROJECT_ROOT / "data" / "processed" / "train.jsonl")
    val_data = load_jsonl(PROJECT_ROOT / "data" / "processed" / "val.jsonl")
    test_data = load_jsonl(PROJECT_ROOT / "data" / "processed" / "test.jsonl")

    leaks = check_split_leakage(train_data, val_data, test_data)

    assert len(leaks["train_val"]) == 0, f"Train/Val leakage detected: {leaks['train_val']}"
    assert len(leaks["train_test"]) == 0, f"Train/Test leakage detected: {leaks['train_test']}"
    assert len(leaks["val_test"]) == 0, f"Val/Test leakage detected: {leaks['val_test']}"


def test_split_info_consistency():
    split_info_file = PROJECT_ROOT / "data" / "splits" / "split_info.json"
    assert split_info_file.exists()

    with open(split_info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    assert info["train_samples"] + info["val_samples"] + info["test_samples"] == info["total_samples"]
    assert info["total_samples"] >= 50


def test_sft_tokenizer_loss_masking():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    processor = SFTChatDataProcessor(tokenizer, max_seq_length=256)

    sample = {
        "instruction": "Wi-Fi won't connect",
        "expected_output": json.dumps({
            "category": "connectivity",
            "action": "reset_wifi",
            "parameters": {},
            "estimated_time_minutes": 5,
        })
    }

    processed = processor.process_train_example(sample)
    labels = processed["labels"]
    input_ids = processed["input_ids"]

    # Verify that the initial prompt tokens are masked with -100
    assert (labels == -100).any(), "Loss masking must mask prompt tokens with -100"
    # Verify that the target assistant tokens are NOT masked with -100
    assert (labels != -100).any(), "Loss masking must preserve assistant target tokens"
    # Verify shape consistency
    assert input_ids.shape == labels.shape
