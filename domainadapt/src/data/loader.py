import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_jsonl(file_path: str | Path) -> List[Dict[str, Any]]:
    """Loads a list of JSON records from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {idx + 1} in {path}: {err}") from err
    return records


def save_jsonl(records: List[Dict[str, Any]], file_path: str | Path) -> None:
    """Saves records to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def check_split_leakage(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    key: str = "instruction",
) -> Dict[str, Set[str]]:
    """Checks for data leakage across dataset partitions based on a key."""
    train_keys = {r[key].strip().lower() for r in train if key in r}
    val_keys = {r[key].strip().lower() for r in val if key in r}
    test_keys = {r[key].strip().lower() for r in test if key in r}

    return {
        "train_val": train_keys & val_keys,
        "train_test": train_keys & test_keys,
        "val_test": val_keys & test_keys,
    }


def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically splits records into train, validation, and test sets."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    return train_set, val_set, test_set
