# Stage 13: Monitoring and Experiment Reproducibility

## Metrics Tracking

The following metrics are tracked throughout the model lifecycle:

| Metric | Description | Traced To |
|--------|-------------|-----------|
| `request_count` | Total number of API requests served | API server logs |
| `request_failures` | Number of failed requests (5xx errors) | API error handler |
| `inference_latency` | Time taken for a single generation (seconds) | `/predict` endpoint timing |
| `token_generation_throughput` | Tokens generated per second | `/predict` endpoint |
| `model_loading_time` | Time to load base model (seconds) | Startup timing |
| `adapter_loading_time` | Time to load LoRA adapter (seconds) | Startup timing |
| `model_version` | Version identifier of base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| `adapter_version` | Version identifier of LoRA adapter | `r16_alpha64` (from `adapter_config.json`) |
| `training_experiment_id` | ID of the training run | `training_summary.json` |
| `evaluation_run_id` | ID of the evaluation run | `evaluation_timestamp` |
| `dataset_version` | Version of the dataset | `split_info.json` `seed: 42` |
| `model_checkpoint` | Model checkpoint hash | `training_summary.json` |
| `training_configuration` | Training hyperparameters | `training_summary.json` `training_config` |
| `evaluation_configuration` | Evaluation hyperparameters | `evaluation_config` |
| `software_environment` | Python, transformers, peft, etc. versions | `requirements.txt` `pyproject.toml` |

## Experiment Reproducibility

### Reproducibility Guarantees

1. **Random Seed**: All experiments use `random.seed(42)`, `numpy.random.seed(42)`, and `torch.manual_seed(42)` for reproducibility.

2. **Dataset Reproducibility**: 
   - The dataset is generated from a fixed seed (42)
   - Split ratios are fixed (60% train, 20% val, 20% test)
   - Split info JSON records the seed and ratios
   - The dataset can be regenerated from the raw data using `create_dataset_unique.py`

3. **Model Reproducibility**:
   - LoRA adapter is configured with fixed `r=16, alpha=64, dropout=0.05`
   - Training uses fixed learning rate (3e-4), weight decay (0.01), and 3 epochs
   - Training summary JSON records all hyperparameters

4. **Reproducibility Verification**:
   - To reproduce an experiment, run:
     ```bash
     # Regenerate dataset
     python create_dataset_unique.py
     
     # Run training (example)
     python -c "
     import torch
     from transformers import AutoModelForCausalLM, AutoTokenizer
     from peft import LoraConfig, get_peft_model
     
     # Load data, configure LoRA, train...
     "
     ```
   - All random seeds are fixed for reproducibility
   - Dataset splits are recorded in `splits/split_info.json`

### Reproducing a Specific Experiment

To reproduce the full experiment suite:

1. **Regenerate the dataset**:
   ```bash
   python create_dataset_unique.py
   ```

2. **Train the LoRA adapter**:
   ```bash
   # The training script from Stage 4 can be adapted
   # Key parameters: r=16, alpha=64, lr=3e-4, epochs=3
   ```

3. **Evaluate the model**:
   ```bash
   # Use the evaluation scripts from Stage 6
   python eval_detailed.py
   ```

4. **Start the API**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

5. **Start the Streamlit frontend**:
   ```bash
   streamlit run frontend/app.py --server.port 8501
   ```

## reproducibility Badges (for README)

```markdown
[![Data Reproducibility](https://img.shields.io/badge/data-reproducible-brightgreen)](https://github.com/your-org/domainadapt)
[![Model Reproducibility](https://img.shields.io/badge/model-reproducible-brightgreen)](https://github.com/your-org/domainadapt)
[![API Reproducibility](https://img.shields.io/badge/api-reproducible-brightgreen)](https://github.com/your-org/domainadapt)
```