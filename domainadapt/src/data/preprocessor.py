import json
from typing import Any, Dict, List, Set, Tuple


REQUIRED_KEYS: Set[str] = {"category", "action", "parameters", "estimated_time_minutes"}
VALID_CATEGORIES: Set[str] = {"connectivity", "performance", "setup", "audio_video", "battery"}


def validate_example_schema(example: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that an example contains a valid instruction and compliant expected_output JSON."""
    if not isinstance(example, dict):
        return False, "Example must be a dictionary"
    if "instruction" not in example or not isinstance(example["instruction"], str):
        return False, "Missing or non-string 'instruction'"
    if not example["instruction"].strip():
        return False, "'instruction' cannot be empty"

    if "expected_output" not in example:
        return False, "Missing 'expected_output'"

    raw_output = example["expected_output"]
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as err:
            return False, f"Invalid JSON in expected_output: {err}"
    elif isinstance(raw_output, dict):
        parsed = raw_output
    else:
        return False, "'expected_output' must be stringified JSON or dict"

    if not isinstance(parsed, dict):
        return False, "'expected_output' JSON must be an object"

    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return False, f"Missing required keys in expected_output: {missing}"

    if parsed["category"] not in VALID_CATEGORIES:
        return False, f"Category '{parsed['category']}' not in valid set {VALID_CATEGORIES}"

    if not isinstance(parsed["parameters"], dict):
        return False, "'parameters' must be a dictionary"

    if not isinstance(parsed["estimated_time_minutes"], (int, float)):
        return False, "'estimated_time_minutes' must be a number"

    return True, ""


def deduplicate_by_instruction(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates records based on instruction text to prevent split contamination."""
    seen: Set[str] = set()
    unique_records: List[Dict[str, Any]] = []
    for rec in records:
        inst = rec.get("instruction", "").strip().lower()
        if inst and inst not in seen:
            seen.add(inst)
            unique_records.append(rec)
    return unique_records
