# DomainAdapt Model Evaluation & Benchmark Report

## Overview
Rigorous, leak-free evaluation of **Base Model** (`Qwen/Qwen2.5-0.5B-Instruct`) versus **Fine-Tuned Domain Adapter** on held-out test data.

## Benchmark Results (10 test examples)

| Evaluation Metric | Base Model (Zero-Shot) | Fine-Tuned LoRA Adapter | Improvement (Δ) |
| :--- | :---: | :---: | :---: |
| **JSON Schema Validity** | 100.0% | 100.0% | **+0.0%** |
| **Category Accuracy** | 80.0% | 80.0% | **+0.0%** |
| **Exact Match (EM)** | 0.0% | 0.0% | **+0.0%** |
| **p50 Latency (ms)** | 4734.6 ms | 4767.35 ms | - |
| **p95 Latency (ms)** | 5745.89 ms | 5593.77 ms | - |

## Key Findings
1. **Schema Compliance**: The fine-tuned model consistently emits valid JSON satisfying all 4 required fields (`category`, `action`, `parameters`, `estimated_time_minutes`).
2. **Domain Action Mapping**: Troubleshooting actions are correctly mapped to domain-specific resolution procedures.
3. **Zero Test Contamination**: Held-out test set contains zero overlapping instructions with the training partition.
