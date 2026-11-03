"""Unit tests for inference generator and schema validator."""

import json
from domainadapt.src.inference.validator import (
    extract_json_payload,
    validate_domain_json,
    TroubleshootingPayload,
)
from domainadapt.src.inference.generator import GenerationResult


def test_extract_json_from_markdown():
    text = """Here is the diagnostic result:
```json
{
  "category": "connectivity",
  "action": "restart_router",
  "parameters": {"wait_seconds": 30},
  "estimated_time_minutes": 5
}
```
Please let me know if you need more help."""
    extracted = extract_json_payload(text)
    assert extracted is not None
    assert extracted["category"] == "connectivity"
    assert extracted["action"] == "restart_router"
    assert extracted["estimated_time_minutes"] == 5


def test_extract_json_from_raw_braces():
    text = 'Diagnostic: {"category": "battery", "action": "calibrate", "parameters": {}, "estimated_time_minutes": 15}.'
    extracted = extract_json_payload(text)
    assert extracted is not None
    assert extracted["category"] == "battery"
    assert extracted["action"] == "calibrate"


def test_extract_json_invalid():
    assert extract_json_payload("No json here") is None
    assert extract_json_payload("{unclosed json") is None
    assert extract_json_payload("") is None


def test_validate_domain_json_success():
    data = {
        "category": "performance",
        "action": "clean_temp_files",
        "parameters": {"drive": "C"},
        "estimated_time_minutes": 10,
    }
    is_valid, payload, err = validate_domain_json(data)
    assert is_valid
    assert payload is not None
    assert payload.category == "performance"
    assert payload.estimated_time_minutes == 10.0
    assert err == ""


def test_validate_domain_json_missing_keys():
    data = {"category": "setup"}  # missing action, parameters, estimated_time_minutes
    is_valid, payload, err = validate_domain_json(data)
    assert not is_valid
    assert payload is None
    assert len(err) > 0


def test_generation_result_model():
    result = GenerationResult(
        instruction="Fix slow PC",
        generated_text='{"category": "performance", "action": "reboot", "parameters": {}, "estimated_time_minutes": 3}',
        parsed_json={"category": "performance", "action": "reboot", "parameters": {}, "estimated_time_minutes": 3},
        is_schema_valid=True,
        validation_error=None,
        latency_seconds=0.45,
        tokens_generated=32,
        adapter_enabled=True,
    )
    d = result.model_dump()
    assert d["instruction"] == "Fix slow PC"
    assert d["is_schema_valid"] is True
    assert d["tokens_generated"] == 32
