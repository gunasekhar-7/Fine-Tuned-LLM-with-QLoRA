import json
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError


class TroubleshootingPayload(BaseModel):
    category: str = Field(..., description="Troubleshooting category (connectivity, setup, battery, etc.)")
    action: str = Field(..., description="Action identifier or remediation command")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters dictionary")
    estimated_time_minutes: float = Field(..., description="Estimated time in minutes to execute action")


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """Extracts a JSON object from raw generated text.

    Handles markdown fences (```json ... ```), raw curly braces, and escaped quotes.
    """
    if not text:
        return None

    # 1. Try markdown code blocks
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try first '{' to last '}'
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Direct JSON load attempt
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def validate_domain_json(data: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[TroubleshootingPayload], str]:
    """Validates that extracted data strictly satisfies the TroubleshootingPayload schema."""
    if data is None or not isinstance(data, dict):
        return False, None, "No valid JSON object found in output"

    try:
        payload = TroubleshootingPayload(**data)
        return True, payload, ""
    except ValidationError as err:
        return False, None, str(err)
