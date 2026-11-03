"""End-to-end integration tests for DomainAdapt."""

from pathlib import Path
import pytest
from transformers import AutoTokenizer

from domainadapt.src.models.base_model import load_base_model
from domainadapt.src.inference.generator import generate_text, DomainInferenceEngine
from domainadapt.src.inference.validator import extract_json_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_e2e_generator_interface():
    """Tests that generate_text interface works smoothly."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    base_model = load_base_model("Qwen/Qwen2.5-0.5B-Instruct", quantization="none", device="cpu")

    output = generate_text(
        model=base_model,
        tokenizer=tokenizer,
        instruction="Wi-Fi will not connect",
        max_new_tokens=32,
        device="cpu",
    )
    assert isinstance(output, str)
    assert len(output.strip()) > 0


def test_e2e_inference_engine():
    """Tests the DomainInferenceEngine pipeline end-to-end."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    base_model = load_base_model("Qwen/Qwen2.5-0.5B-Instruct", quantization="none", device="cpu")

    engine = DomainInferenceEngine(
        model=base_model,
        tokenizer=tokenizer,
        device="cpu",
        adapter_enabled=False,
    )

    result = engine.generate("Laptop battery draining quickly", max_new_tokens=32)
    assert result.instruction == "Laptop battery draining quickly"
    assert len(result.generated_text) > 0
    assert result.latency_seconds > 0.0
