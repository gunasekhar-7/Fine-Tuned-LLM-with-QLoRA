import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from domainadapt.src.inference.generator import DomainInferenceEngine
from domainadapt.src.evaluation.metrics import compute_schema_metrics

logger = logging.getLogger("domainadapt.evaluation")


class ModelEvaluator:
    """Evaluates base model vs fine-tuned model and produces comparison reports."""

    def __init__(self, test_data: List[Dict[str, Any]]):
        self.test_data = test_data

    def evaluate_engine(
        self,
        engine: DomainInferenceEngine,
        name: str = "model",
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        eval_samples = self.test_data[:limit] if limit else self.test_data
        logger.info(f"Evaluating {name} on {len(eval_samples)} test samples...")

        results = []
        for sample in eval_samples:
            res = engine.generate(sample["instruction"])
            results.append(res.model_dump())

        metrics = compute_schema_metrics(results, eval_samples)
        return {
            "model_name": name,
            "metrics": metrics,
            "sample_predictions": results[:5],
        }

    def run_comparison(
        self,
        base_engine: DomainInferenceEngine,
        finetuned_engine: DomainInferenceEngine,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        base_eval = self.evaluate_engine(base_engine, name="Base Model", limit=limit)
        ft_eval = self.evaluate_engine(finetuned_engine, name="Fine-Tuned Adapter", limit=limit)

        base_metrics = base_eval["metrics"]
        ft_metrics = ft_eval["metrics"]

        comparison = {
            "base_metrics": base_metrics,
            "finetuned_metrics": ft_metrics,
            "delta": {
                "schema_validity_gain": round(
                    ft_metrics["schema_validity_rate"] - base_metrics["schema_validity_rate"], 4
                ),
                "category_accuracy_gain": round(
                    ft_metrics["category_accuracy"] - base_metrics["category_accuracy"], 4
                ),
                "exact_match_gain": round(
                    ft_metrics["exact_match_rate"] - base_metrics["exact_match_rate"], 4
                ),
            },
        }
        return comparison
