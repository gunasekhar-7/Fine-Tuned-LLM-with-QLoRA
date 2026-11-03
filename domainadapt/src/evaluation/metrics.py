import json
from typing import Any, Dict, List, Optional
import numpy as np


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Checks whether the prediction text matches ground truth exactly or by parsed JSON."""
    if prediction.strip() == ground_truth.strip():
        return True
    try:
        pred_obj = json.loads(prediction)
        gt_obj = json.loads(ground_truth)
        return pred_obj == gt_obj
    except Exception:
        return False


def compute_schema_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes rigorous evaluation metrics across generated outputs."""
    total = len(predictions)
    if total == 0:
        return {
            "total_samples": 0,
            "exact_match_rate": 0.0,
            "schema_validity_rate": 0.0,
            "category_accuracy": 0.0,
            "avg_latency_ms": 0.0,
        }

    exact_matches = 0
    schema_valid = 0
    category_matches = 0
    latencies: List[float] = []

    for pred, gt in zip(predictions, ground_truths):
        latencies.append(pred.get("latency_seconds", 0.0) * 1000.0)

        # Schema validity
        if pred.get("is_schema_valid", False):
            schema_valid += 1

        # Compare with ground truth
        gt_raw = gt.get("expected_output", "{}")
        if isinstance(gt_raw, str):
            try:
                gt_obj = json.loads(gt_raw)
            except Exception:
                gt_obj = {}
        else:
            gt_obj = gt_raw

        pred_obj = pred.get("parsed_json") or {}

        if pred_obj and gt_obj:
            if pred_obj == gt_obj:
                exact_matches += 1
            if pred_obj.get("category") == gt_obj.get("category"):
                category_matches += 1

    return {
        "total_samples": total,
        "exact_match_rate": round(exact_matches / total, 4),
        "schema_validity_rate": round(schema_valid / total, 4),
        "category_accuracy": round(category_matches / total, 4),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)) if latencies else 0.0, 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
    }
