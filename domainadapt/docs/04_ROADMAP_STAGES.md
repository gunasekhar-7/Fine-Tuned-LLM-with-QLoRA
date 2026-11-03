# DomainAdapt - Stage 0 Roadmap

## Stage Objectives, Plan, and Deliverables

### Stage 0 — Research, Feasibility, and Architecture ✅ COMPLETED

**Objectives:**
1. Research candidate base models and their licenses
2. Compare model size, architecture, context length, quantization support, and GPU requirements
3. Evaluate dataset options and their licenses
4. Select a task and define its input/output contract
5. Estimate memory requirements for base model loading, LoRA training, and inference
6. Determine whether the project can realistically run on available Colab/Kaggle GPU hardware
7. Design the system architecture
8. Define measurable project success criteria

**Implementation Plan:**
- Research and document 5+ candidate base models (0.5B-7B parameter range)
- Evaluate 4+ candidate datasets for licensing, format, and domain relevance
- Compare use cases: support ticket classification, SQL generation, domain-specific instruction following, technical support assistant
- Select Qwen2.5-0.5B-Instruct as base model
- Select domain-specific instruction following with structured outputs as use case
- Create custom structured output dataset design (800 examples)
- Estimate CPU memory: ~8-10 GB total (model + optimizer + LoRA adapters)
- Design modular architecture with centralized configs, data pipeline, training loop, evaluation, inference, API, frontend
- Define success criteria: EM improvement from ~50% (base) to ~70%+ (fine-tuned) on test set

**Deliverables:**
- Problem definition ✅
- Model selection report ✅ (docs/00_MODEL_SELECTION_REPORT.md)
- Dataset selection report ✅ (docs/01_DATASET_SELECTION_REPORT.md)
- Hardware feasibility report ✅ (docs/02_HARDWARE_FEASIBILITY_REPORT.md)
- System architecture diagram ✅ (docs/03_ARCHITECTURE_DIAGRAM.md)
- Development roadmap ✅ (this file)
- Initial project risks and mitigations ✅ (documented in technical decisions)

**Acceptance criteria:** The selected model, task, dataset, training approach, and evaluation strategy are technically compatible and feasible before implementation begins. ✅

---

### Stage 1 — Repository Foundation and Engineering Setup

**Objectives:**
- Create a maintainable project structure following the specified layout
- Centralize configurations in YAML files under configs/
- Implement logging and exception handling
- Use environment variables for credentials
- Add dependency management and reproducible setup instructions
- Configure formatting, linting, and type checking
- Establish Git conventions
- Exclude datasets, model weights, secrets, and generated artifacts from Git

**Implementation Plan:**
1. Create directory structure (already partially created)
2. Write configs/model.yaml, configs/training.yaml, configs/evaluation.yaml (already created)
3. Write requirements.txt with pinned dependencies
4. Write pyproject.toml with project metadata
5. Set up .env.example with template variables
6. Configure ruff (linting), pyright (type checking)
7. Create basic test infrastructure
8. Initialize Git repository with .gitignore

**Deliverables:**
- Clean environment can install dependencies
- Application modules can be imported
- Basic tests pass
- Project setup is reproducible

**Acceptance criteria:** A clean environment can install dependencies, import the application modules, execute basic tests, and run the initial project setup.

---

### Stage 2 — Dataset Acquisition and Data Engineering

**Objectives:**
- Acquire the selected dataset from a legitimate source
- Verify dataset licensing and usage restrictions
- Inspect dataset size, fields, distribution, and quality
- Identify duplicates, missing values, malformed records, and inconsistent examples
- Remove irrelevant or low-quality examples using documented rules
- Normalize text without destroying meaningful information
- Convert records into the selected model's instruction/chat format
- Validate every training example against the expected schema
- Create reproducible train, validation, and test splits

**Implementation Plan:**
1. Create the structured output dataset (800 examples custom-generated)
2. Verify licensing (project-internal, no third-party restrictions)
3. Inspect dataset: size, field distribution, category balance
4. Apply filtering rules: remove >512-token instructions, invalid JSON outputs, duplicates
5. Normalize whitespace, standardize formatting
6. Convert to Qwen2.5 chat template format
7. Create train/validation/test splits (600/100/100) with reproducible identifiers
8. Save dataset version hash and preprocessing configuration

**Deliverables:**
- Reproducible preprocessing pipeline
- Validated training dataset (600 examples)
- Train/validation/test split with identifiers
- Dataset statistics report
- Data quality tests

**Acceptance criteria:** The dataset can be reconstructed from documented inputs and preprocessing steps, and no test examples leak into training.

---

### Stage 3 — Base Model Selection and Baseline Evaluation

**Objectives:**
- Load the untouched base model and tokenizer
- Verify model compatibility with the intended LoRA implementation
- Configure inference settings
- Build a baseline inference pipeline
- Evaluate the base model on the held-out evaluation set
- Save predictions and evaluation results
- Record model configuration, prompt template, generation parameters, and software versions

**Implementation Plan:**
1. Load Qwen2.5-0.5B-Instruct in float32 on CPU
2. Build inference pipeline with chat template formatting
3. Evaluate base model on 100-example test set
4. Compute Exact Match (EM), Schema validity, Invalid output rate
5. Save baseline predictions for comparison
6. Record software versions, random seed, configuration

**Deliverables:**
- Baseline evaluation pipeline
- Baseline predictions (base model on test set)
- Baseline metrics report (EM, schema validity, invalid rate)
- Reproducible inference configuration

**Acceptance criteria:** The base model has been evaluated on the same task and under the same evaluation protocol that will be used for the fine-tuned model.

---

### Stage 4 — LoRA and QLoRA Implementation

**Objectives:**
- Implement standard PEFT LoRA (full-precision) on CPU
- Design QLoRA 4-bit quantization path for GPU activation
- Build complete training pipeline with tokenization, chat formatting, dataset collation
- PEFT configuration from configs/training.yaml
- Training arguments, validation during training, checkpoint saving
- Adapter saving and configuration recording

**Implementation Plan:**
1. Configure LoRA: r=8, lora_alpha=32, lora_dropout=0.05, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
2. Load base model + apply LoRA adapters
3. Tokenize dataset using Qwen2.5 chat template
4. Build training loop with SFTTrainer or custom loop
5. Train for 3 epochs with gradient accumulation (effective batch size 8)
6. Save adapter checkpoint and training summary
7. Record: training loss, validation loss, learning rate, duration, GPU memory, trainable params, total params, throughput, seed, config hash

**Deliverables:**
- Complete fine-tuning run finishes successfully
- Loadable adapter saved to artifacts/trained_adapter/
- Configuration and training history recorded

**Acceptance criteria:** A complete fine-tuning run finishes successfully, saves a loadable adapter, and records its configuration and training history.

---

### Stage 5 — Hyperparameter Experiments and Model Optimization

**Objectives:**
- Design controlled experiments comparing LoRA rank (4, 8, 16), learning rate, batch size, gradient accumulation, epochs, warmup ratio, weight decay
- Track every run using W&B or local JSON logging
- Compare: validation loss, task-specific performance, training time, GPU memory, trainable parameter count, inference latency, model stability
- Use validation performance to select best configuration (no test set tuning)

**Implementation Plan:**
1. Run 3 experiments: r ∈ {4, 8, 16}, all other hyperparameters fixed at training config values
2. Log each run: config hash, seed, validation EM, training time, memory
3. Select best r based on validation EM
4. Document: experiment configurations, W&B records, comparison report, selected adapter config, reproducible best training run

**Deliverables:**
- Experiment configurations (3 runs)
- W&B experiment records or local JSON logs
- Hyperparameter comparison report
- Selected adapter configuration
- Reproducible best training run

**Acceptance criteria:** The final configuration is selected based on documented experimental evidence rather than intuition alone.

---

### Stage 6 — Fine-Tuned Model Evaluation and Comparison

**Objectives:**
- Evaluate the selected fine-tuned model using the untouched test set
- Compare three configurations: base model, fine-tuned LoRA adapter, merged model (if supported)
- Use identical test examples and consistent inference settings
- Include: overall task-specific metrics, per-category performance, failure analysis, invalid output rate, before-and-after comparison report

**Implementation Plan:**
1. Load base model, fine-tuned adapter, and (if applicable) merged model
2. Evaluate all three on the same 100-example test set
3. Compute EM, schema validity, invalid output rate per category
4. Perform failure analysis: categorize types of errors (wrong category, wrong action, invalid JSON, etc.)
5. Record inference latency for each configuration
6. Create before-and-after comparison report

**Deliverables:**
- Evaluation scripts
- Test-set predictions (base, fine-tuned, merged if applicable)
- Comparison report
- Error analysis
- Before-and-after metrics
- Reproducibility instructions

**Critical requirement:** Do not claim the fine-tuned model outperforms the base model unless results support that conclusion. If no improvement, investigate and document honestly.

**Acceptance criteria:** Evaluation scripts produce valid results on the untouched test set; comparison report is documented with before-and-after metrics.

---

### Stage 7 — Model Packaging and Inference Optimization

**Objectives:**
- Prepare the fine-tuned model for practical inference
- Implement base model and adapter loading
- Adapter configuration validation
- Model version identification
- Configurable generation parameters with safe limits
- Input validation, output validation, token limits
- GPU/CPU compatibility checks, memory management, inference error handling

**Implementation Plan:**
1. Implement adapter-based inference (base model + LoRA adapter via PeftModel)
2. Implement merged model inference (adapter weights merged into base)
3. Evaluate: adapter-based inference, merged inference, quantized inference (if GPU available)
4. Measure: latency, throughput, GPU memory, model loading time
5. Document differences between training-time QLoRA quantization and inference-time quantization

**Deliverables:**
- Model loading scripts with documented instructions
- Inference benchmarking results
- Quantization comparison (if applicable)

**Acceptance criteria:** A clean environment can load the selected model and adapter and produce valid outputs using documented instructions.

---

### Stage 8 — FastAPI Backend Development

**Objectives:**
- Build an inference API around the fine-tuned model
- Implement endpoints: GET /health, GET /model/info, POST /predict, POST /predict/batch, GET /metrics
- Pydantic request/response schemas
- Input validation, structured errors, request IDs
- Logging, API documentation (OpenAPI), rate limiting
- Model loading lifecycle management, readiness and liveness checks
- Graceful handling of model-loading and GPU failures

**Implementation Plan:**
1. Create FastAPI app in api/main.py
2. Define Pydantic schemas for requests (instruction) and responses (structured output + metadata)
3. Implement endpoints with input validation
4. Load model + adapter at startup (lifecycle management)
5. Add health checks, metrics, rate limiting
6. Add OpenAPI documentation
7. Test with curl/httpie

**Deliverables:**
- FastAPI application with all specified endpoints
- Pydantic schemas for request/response validation
- API documentation (auto-generated OpenAPI)
- Health and metrics endpoints

**Acceptance criteria:** The API reliably serves predictions, validates inputs, handles failures, and exposes useful health and model information.

---

### Stage 9 — Interactive Application

**Objectives:**
- Create a clean Streamlit or React interface demonstrating the real value of the fine-tuned model
- Include: model info and configuration, input form for the selected task, base model vs fine-tuned comparison, generated outputs, inference latency, evaluation metrics, example test cases, error and limitation explanations
- Let users compare the same input across base and fine-tuned models
- Make interface understandable to both technical and non-technical reviewers

**Implementation Plan:**
1. Create Streamlit app in frontend/app.py
2. Add model info section (selected model, LoRA config, dataset version)
3. Add input form for the troubleshooting task
4. Add base vs fine-tuned comparison toggle
5. Display generated outputs with latency timing
6. Show evaluation metrics (EM, schema validity)
7. Add example test cases dropdown
8. Add error/limitation expander

**Deliverables:**
- Streamlit interactive application
- Screenshots or demo video

**Acceptance criteria:** A reviewer can understand the task, test the model, inspect outputs, and see evidence of its measured performance.

---

### Stage 10 — Testing and Quality Assurance

**Objectives:**
- Build comprehensive test suite covering unit, integration, and end-to-end tests
- Unit tests: data preprocessing, dataset formatting, tokenization, configuration validation, LoRA configuration, model loading, inference formatting, output parsing
- Integration tests: adapter loading, API inference, model configuration, evaluation pipeline
- End-to-end tests: valid request/response, base vs fine-tuned comparison, malformed input handling, unavailable model resources
- Regression tests: output schemas, performance regressions, saved model artifacts, evaluation reproducibility
- Testing must not require GPU for every test; mock model components where appropriate

**Implementation Plan:**
1. Write unit tests in tests/ (pytest format)
2. Write integration tests
3. Write end-to-end tests
4. Add regression tests for output schemas and artifact validation
5. Run tests; ensure coverage focused on critical behavior

**Deliverables:**
- Unit test suite
- Integration test suite
- End-to-end test suite
- Regression test suite

**Acceptance criteria:** All required tests pass, with coverage focused on critical behavior rather than inflated coverage numbers.

---

### Stage 11 — Production Hardening and Security

**Objectives:**
- Review the entire system for reliability and security
- Implement or verify: secrets management, input size and token limits, dataset and artifact validation, safe model loading, dependency pinning, error boundaries, request timeouts, rate limiting, structured logging, resource monitoring, GPU memory safeguards, API security, sensitive data handling, model and dataset license compliance
- Never commit access tokens, private datasets, API keys, or large model weights
- Document limitations of public model checkpoints, generated responses, and domain-specific reliability

**Implementation Plan:**
1. Review all code for security best practices
2. Ensure no secrets in source code (use .env + .gitignore)
3. Add input token limits and output token limits
4. Add model loading safeguards (OOM protection, GPU memory checks)
5. Add dependency pinning in requirements.txt
6. Add structured logging (loguru or standard logging with structured format)
7. Document license compliance (MIT for model, CC0 or project-internal for dataset)
8. Create KNOWN_LIMITATIONS.md

**Deliverables:**
- Security audit findings
- KNOWN_LIMITATIONS.md
- Updated .gitignore and .env.example

**Acceptance criteria:** No known critical security issues remain; application has documented operational limitations.

---

### Stage 12 — Docker, CI/CD, and Deployment

**Objectives:**
- Containerize the application and establish automated development workflow
- Dockerfile, Docker Compose where appropriate
- Environment-based configuration
- GitHub Actions: lint, type-check, unit tests, API smoke tests
- Build verification, artifact and configuration versioning
- Deploy application using environment compatible with model's memory and inference requirements

**Implementation Plan:**
1. Write Dockerfile (CPU-optimized, multi-stage if needed)
2. Write docker-compose.yml for local development
3. Write GitHub Actions workflow (.github/workflows/test.yml)
4. Set environment-based configuration (production vs development)
5. Test build: `docker build .`
6. Test run: `docker run` or `docker compose up`
7. Document: deployment architecture, GPU requirements, environment variables, startup commands, model download instructions, operational cost considerations, shutdown and resource cleanup

**Deliverables:**
- Dockerfile
- docker-compose.yml
- GitHub Actions workflows
- Deployment documentation

**Acceptance criteria:** The project can be built and launched using documented commands; CI pipeline validates critical engineering checks.

---

### Stage 13 — Monitoring and Experiment Reproducibility

**Objectives:**
- Track: request count, request failures, inference latency, token generation throughput, model loading time, GPU memory, model version, adapter version, training experiment ID, evaluation run ID
- Ensure every published metric can be traced to: dataset version, model checkpoint, training configuration, evaluation configuration, software environment
- Use W&B for training experiment tracking and application monitoring where justified

**Implementation Plan:**
1. Add metric tracking to FastAPI endpoints (request count, failures, latency)
2. Log training experiment ID with each run
3. Log evaluation run ID with each evaluation
4. Include dataset version, model checkpoint hash, training config hash, eval config hash in all reports
5. Ensure traceability: given an evaluation result, can identify which model, dataset, and config produced it

**Deliverables:**
- Metrics tracking implementation
- Traceability documentation

**Acceptance criteria:** A reviewer can reproduce the selected experiment and understand which model produced a given evaluation result.

---

### Stage 14 — Documentation and Portfolio Presentation

**Objectives:**
- Create professional documentation:
  - README.md (already created)
  - docs/ARCHITECTURE.md
  - docs/DATASET_CARD.md
  - docs/MODEL_CARD.md
  - docs/TRAINING_GUIDE.md
  - docs/EVALUATION_REPORT.md
  - docs/DEPLOYMENT_GUIDE.md
  - docs/SECURITY.md
  - docs/KNOWN_LIMITATIONS.md
- Prepare portfolio deliverables: architecture diagram, training workflow diagram, before-and-after performance charts, W&B experiment screenshots, application screenshots, API documentation screenshot, short demonstration video, technical project summary, resume-ready bullet points

**Deliverables:**
- All documentation files
- Portfolio presentation materials

**Acceptance criteria:** README makes it easy for a recruiter or engineer to understand what was built, how it works, and what evidence supports its performance.

---

## Overall Project Acceptance

The project passes Stage 0 if all Stage 0 acceptance criteria are met. Subsequent stages proceed only if the previous stage's acceptance criteria are satisfied. Critical failures at any stage must be resolved before proceeding.

**Current status:** Stage 0 complete. All acceptance criteria met.