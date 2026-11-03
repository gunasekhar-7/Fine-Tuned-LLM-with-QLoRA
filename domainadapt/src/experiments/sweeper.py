"""Hyperparameter sweep and configuration experiments."""
from typing import Any, Dict, List


class HyperparameterSweeper:
    """Manages hyperparameter search sweeps across rank, alpha, and learning rates."""

    @staticmethod
    def generate_configurations() -> List[Dict[str, Any]]:
        return [
            {"name": "r4_alpha16", "r": 4, "lora_alpha": 16, "learning_rate": 1e-4},
            {"name": "r8_alpha32", "r": 8, "lora_alpha": 32, "learning_rate": 2e-4},
            {"name": "r16_alpha64", "r": 16, "lora_alpha": 64, "learning_rate": 3e-4},
        ]
