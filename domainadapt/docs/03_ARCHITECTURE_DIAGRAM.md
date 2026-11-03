## System Architecture Design

**Architecture Overview:** Modular, stage-driven LLM fine-tuning pipeline designed for CPU-based training with GPU-optional QLoRA extension.

```
domainadapt/
├── configs/                  # Centralized YAML configurations
│   ├── model.yaml           # Model selection and dtype
│   ├── training.yaml        # LoRA training hyperparameters
│   └── evaluation.yaml      # Metrics and success criteria
├── data/                     # Dataset pipeline (raw, processed, splits)
│   ├── raw/                 # Original dataset before preprocessing
│   ├── processed/           # Validated, formatted training examples
│   └── splits/              # train/validation/test partitions
├── src/
│   ├── data/                # Data loading, preprocessing, tokenization
│   │   ├── loader.py        # JSONL dataset loader
│   │   ├── preprocessor.py  # Filtering, normalization, schema conversion
│   │   └── tokenizer.py     # Chat template formatting, tokenization
│   ├── models/              # Model loading, LoRA configuration, adapter management
│   │   ├── base_model.py    # Load Qwen2.5-0.5B-Instruct on CPU
│   │   ├── lora_config.py   # LoRA rank, alpha, target modules setup
│   │   └── adapter.py       # Save/load PEFT adapters
│   ├── training/            # Training pipeline (SFTTrainer or custom loop)
│   │   ├── trainer.py       # Training loop with validation
│   │   └── checkpoint.py    # Checkpoint save/load logic
│   ├── evaluation/          # Model evaluation on test set
│   │   ├── evaluator.py     # Exact-match, per-category metrics
│   │   └── comparison.py    # Base vs fine-tuned comparison
│   ├── inference/           # Text generation / structured output
│   │   ├── generator.py     # Base model + adapter inference
│   │   └── validator.py     # JSON schema validation of outputs
│   ├── experiments/         # Hyperparameter sweep orchestration
│   │   └── sweeper.py       # Controlled experiment runner
│   └── utils/               # Logging, exception handling, environment setup
├── api/                      # FastAPI inference server
│   ├── main.py              # FastAPI application with endpoints
│   ├── schemas.py           # Pydantic request/response models
│   └── dependency.py        # Model lifecycle management
├── frontend/                 # Streamlit user interface
│   └── app.py               # Interactive model comparison interface
├── tests/                    # Unit, integration, and E2E test suite
│   ├── test_data.py       # Data preprocessing tests
│   ├── test_model.py      # Model loading and LoRA config tests
│   ├── test_inference.py  # Inference formatting tests
│   └── test_e2e.py        # End-to-end request/response validation
├── reports/                  # Generated evaluation reports
│   ├── train_summary.json   # Training run metadata
│   └── eval_report.json     # Test-set evaluation results
├── artifacts/                # Saved model adapters and outputs
│   └── trained_adapter/     # PEFT adapter directory (adapter_config.json, pytorch_model.bin)
├── docs/                     # All project documentation
│   ├── 00_MODEL_SELECTION_REPORT.md
│   ├── 01_DATASET_SELECTION_REPORT.md
│   ├── 02_HARDWARE_FEASIBILITY_REPORT.md
│   ├── 03_ARCHITECTURE_DIAGRAM.md     # To be created
│   ├── TECHNICAL_DECISIONS.md          # Conceptual explanations
│   └── FINAL_PRODUCTION_READINESS_AUDIT.md
├── docker/                   # Containerization
│   ├── Dockerfile           # CPU-optimized build
│   └── docker-compose.yml  # Service orchestration
├── .github/workflows/        # CI/CD pipelines
│   └── test.yml            # Lint, type-check, unit tests
├── requirements.txt          # Pinned dependencies
├── pyproject.toml            # Project metadata, optional deps
├── .env.example              # Environment variable templates
└── README.md                 # Project overview and setup
```

**Key Design Decisions:**

1. **LoRA over QLoRA for initial runs:** Standard PEFT LoRA (full-precision) on CPU. QLoRA 4-bit quantization design is included in the code but will be conditionally enabled based on CUDA availability.

2. **Adapter-based fine-tuning:** Base model weights remain frozen; only LoRA adapter matrices (A and B) are trained. At inference, adapter is loaded alongside base model via `PeftModel.from_pretrained()`.

3. **Schema-based evaluation:** Outputs validated against JSON schema; exact-match score computed on structural equivalence, not token-level comparison.

4. **Modular data pipeline:** Dataset can be reconstructed from raw JSONL → processed JSONL → train/val/test splits. Provenance tracked via split identifiers and preprocessing configuration hashes.

5. **Configuration-driven:** All hyperparameters, paths, and model settings centralized in YAML configs under `configs/`. Seed (42) ensures reproducibility.

6. **API-first design:** FastAPI backend with Streamlit frontend. Model loading logic separated from HTTP routing. Model and adapter loaded once at startup.

7. **Experiment tracking:** W&B or local JSON logging for every training run. Config hash, seed, and dataset version recorded per run.

**Inference Flow:**
```
User Input (instruction)
    │
    ▼
frontend → FastAPI /predict endpoint
    │
    ▼
src/inference/generator.py
    │
    ├─→ Load base model (Qwen2.5-0.5B-Instruct, float32, CPU)
    │     │
    │     └─→ If fine-tuned adapter exists:
    │           │
    │           └─→ PeftModel.from_pretrained(base_model, adapter_path)
    │
    ▼
Tokenize instruction using Qwen2.5 chat template
    │
    ▼
Model.generate() → raw token sequence
    │
    ▼
src/inference/validator.py → Parse JSON from output
    │
    ▼
Validate against expected schema
    │
    ▼
Return structured result to user
```

**Training Flow:**
```
Dataset (JSONL instruction + expected_output)
    │
    ▼
src/data/preprocessor.py → Filter, normalize, schema conversion
    │
    ▼
src/data/tokenizer.py → Qwen2.5 chat template formatting + tokenization
    │
    ▼
PEFT model = base_model + LoRA config (from configs/training.yaml)
    │
    ▼
SFTTrainer or custom training loop
    │
    │   ├─→ Forward pass → compute loss (cross-entropy on assistant tokens)
    │   ├─→ Backward pass → optimizer step (AdamW, lr from config)
    │   ├─→ Save checkpoint every save_steps
    │   └─→ Log training metrics every logging_steps
    │
    ▼
Artifacts: adapter/ (adapter_config.json + trained weights)
        training_summary.json (loss, lr, duration, seed, config hash)
```

**Fallback Paths:**
- If CUDA becomes available: QLoRA 4-bit activation via `bitsandbytes.compute_dtype = torch.float4` and `PeftModel.for_quantized_4bit`
- If memory error occurs: Reduce batch size to 1, increase gradient accumulation, reduce max_seq_length
- If training diverges: Reduce learning rate, increase gradient accumulation, add weight decay

**Reproducibility Guarantees:**
- Random seed (42) set for Python, NumPy, and torch before any initialization
- All code paths deterministic given fixed seed (where PyTorch ops allow)
- Dataset split identifiers stored in `data/splits/` with hash of preprocessing config
- Model checkpoints include optimizer state for resume capability