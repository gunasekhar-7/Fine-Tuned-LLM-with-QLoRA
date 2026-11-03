# Technical Decisions Documentation

This document explains the key technical choices made throughout the DomainAdapt project. It is intended to help interviewers and reviewers understand the reasoning behind implementation decisions.

---

## 1. What is Fine-Tuning, and When is it Preferable to Prompt Engineering or RAG?

**Fine-tuning** updates the model's internal weights to learn a new task. Unlike prompt engineering (which keeps weights fixed and only changes the input prompt) or RAG (which retrieves external context at query time), fine-tating modifies the model's parameters.

**When fine-tuning is preferable:**
- The target task requires capabilities that the base model cannot acquire through prompting alone (e.g., consistent structured output, specific writing style, domain-specific terminology)
- You have sufficient data (100+ high-quality examples) to reliably update weights without catastrophic forgetting
- You need consistent, reproducible outputs across many inferences (prompting can be non-deterministic)
- You've exhausted prompt engineering and RAG as approaches
- Deployment cost: a fine-tuned small model may have lower inference latency and cost than a larger model with complex prompting/RAG pipelines

**When not to fine-tune:**
- The task can be solved effectively with well-crafted prompts (few-shot or zero-shot)
- The dataset is very small (<50 examples) - fine-tuning may overfit
- The base model already performs adequately on the task
- You need rapid iteration without retraining (prompt engineering is faster)

**DomainAdapt choice:** Fine-tuning selected because the task (structured JSON output for technical troubleshooting) requires consistent schema adherence that prompting alone cannot guarantee. The 0.5B parameter model is large enough to benefit from weight updates on a focused domain dataset.

---

## 2. What is LoRA?

**Low-Rank Adaptation (LoRA)** trains only a small subset of new parameters while keeping the pretrained base model weights frozen.

**How it works:**
- For a linear layer with weight matrix W ∈ ℝ^{n×m}, LoRA inserts two trainable low-rank matrices A ∈ ℝ^{r×k} and B ∈ ℝ^{k×m} where k << min(n, m)
- The effective weight is: W_eff = W + (α/r) · B · A
- During training, W is frozen; only A and B are updated
- After training, the adapter (A, B) can be saved and loaded onto the frozen base model at inference time
- The original model can be restored by unbloading the adapter

**LoRA rank (r):** The inner dimension of the low-rank decomposition. Lower r = fewer trainable parameters, faster training, but potentially less capacity to learn. Typical values: 4, 8, 16, 32, 64.

**LoRA alpha (α):** Scaling factor that controls the magnitude of the adaptation. The effective update is scaled by α/r. Common practice: α = r or α = 2r. Higher α = stronger update signal.

**LoRA dropout:** Applied to the output of the LoRA layer during training, helps prevent overfitting. Typical: 0.05-0.1.

**Target modules:** Specific linear layers in the transformer where LoRA is applied. Common targets: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj (in Qwen: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

**Parameter count:** For a layer with dimension d, LoRA rank r adds 2 × d × r parameters (plus bias if included). For Qwen2.5-0.5B with hidden size 3200 and r=8: ~51,200 trainable parameters per layer × ~30 relevant layers ≈ 1.5M total trainable parameters (vs 512M total model parameters = ~0.3% trainable).

---

## 3. How Do Low-Rank Matrices Adapt Pretrained Weights?

**Mathematical formulation:**

Given original weight W ∈ ℝ^{n×m}, LoRA adds a decomposition:

W_eff = W + ΔW, where ΔW = (α/r) · B · A

- A has shape (r, k), B has shape (k, m), so ΔW has shape (n, m) - same as original
- The rank r limits the dimensionality of the adaptation space
- The update ΔW is constrained to the column space of A and row space of B

**Intuition:** The pretrained weight W encodes general knowledge (language understanding, reasoning, etc.). The low-rank update ΔW can only modify a r-dimensional subspace of the original weight space. This constraint:
- Prevents the model from forgetting general capabilities (frozen base preserves prior knowledge)
- Makes training feasible with limited data (only ~1-3% of parameters updated)
- Acts as a regularizer, preventing extreme weight changes

**Why rank r matters:**
- r = 1: ΔW is a rank-1 matrix, outer product of two vectors. Very limited adaptation capacity.
- r = rank(W): ΔW can full-rank update (equivalent to full fine-tuning)
- Typical fine-tuning: r = 8-32 for 0.5B-7B models, capturing the "intrinsic dimensionality" of the task adaptation

**DomainAdapt implementation:** LoRA target modules selected as q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (all transformer attention and feed-forward layers). This provides broad coverage for the 0.5B Qwen2.5 model while keeping total trainable parameters manageable.

---

## 4. What is QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** adds 4-bit quantization of the base model weights before applying LoRA.

**Three innovations:**

1. **4-bit Normal Float (NF4):** A quantization scheme optimized for the normal distribution of pretrained weights. NF4 assigns more bits to extreme values and fewer bits to common values. Unlike uniform quantization, NF4 preserves the mean and variance of the original distribution up to the quantization step.

2. **Double Quantization:** Quantizes the quantization residuals (the difference between original weights and their NF4 representation). This adds a second layer of compression, typically saving an additional 0.5-1 bit.

3. **Paged Optimizer States:** Adam optimizer states (momentum and variance caches) are stored in paged memory, reducing optimizer memory overhead. This is critical because training requires storing optimizer states in addition to the model weights.

**Memory reduction:** 4-bit quantization reduces model memory from 4 bytes/parameter to 0.5 bytes/parameter (8× reduction). For Qwen2.5-0.5B: from ~2 GB (float32) to ~0.25 GB (4-bit).

**Why QLoRA requires GPU:**
- NF4 quantization/dequantization kernels are implemented in CUDA (bitsandbytes)
- CUDA kernels are needed for the 4-bit matrix operations during forward/backward pass
- CPU fallback exists in bitsandbytes but is not practical for training (software emulation, very slow)

**DomainAdapt status:** QLoRA 4-bit quantization design is architected in the codebase (configs, model loading logic) but will be conditionally activated only when CUDA is available. Initial training runs will use standard LoRA (full-precision) on CPU. When GPU is available (Colab/Kaggle), QLoRA can be enabled with minimal code changes.

---

## 5. How Do LoRA Alpha and Scaling Work?

**LoRA alpha (α) scales the effective update:**

W_eff = W + (α/r) · B · A

**The alpha/r ratio controls the magnitude of the weight change:**

- If α = r: W_eff = W + B · A (no additional scaling)
- If α > r: The update (α/r · B · A) is amplified proportionally
- If α < r: The update is attenuated

**Common practices:**
- α = r: Simple, no extra scaling factor
- α = 2r: Double the update magnitude (often used)
- α = 16, r = 8: Standard configuration for many models

**Why it matters:**
- Larger α → larger weight updates → faster adaptation but risk of overstating the pretrained weights
- Smaller α → smaller updates → conservative adaptation, more stable
- The α/r ratio should be chosen based on the desired update strength, not arbitrarily

**DomainAdapt configuration:** α = 32, r = 8 → α/r = 4. This means the effective LoRA update is 4× the raw B · A product. This is a standard configuration that provides meaningful adaptation without overwhelming the frozen base weights.

---

## 6. What is NF4 Quantization?

**Normal Float 4 (NF4)** is a quantization scheme designed specifically for pretrained model weights.

**Key properties:**
- Represents each weight using 4 bits
- Non-uniform quantization: more bits assigned to values in the tails of the distribution, fewer bits to the center
- Preserves the mean and variance of the original distribution exactly (up to quantization step size)
- Mathematically, the quantized value q(w) and dequantized value ŵ(w) satisfy: E[ŵ] = E[w] and Var[ŵ] = Var[w]

**Comparison with other 4-bit schemes:**
- **FP4 (Finite Precision 4):** Uniform quantization, simpler but less accurate for weight distributions
- **E4M3 (Exponent-Mantissa 4-3):** FP8-inspired, range-biased
- **NF4:** Distribution-optimized, best for LLM weight preservation

**Role in QLoRA:**
- NF4 allows 4-bit weights to closely approximate the original float16/float32 weights
- During training, the NF4-dequantized weights are used for forward and backward passes
- The quantization scale and zero-point are stored per tensor (not per weight), minimizing overhead
- After training, the 4-bit quantized weights can be stored and loaded efficiently

**DomainAdapt note:** NF4 is part of the QLoRA design but not used in the initial CPU runs. The model will be loaded in full float32 precision. NF4 support can be activated when CUDA is available.

---

## 7. What is Double Quantization?

**Double Quantization** applies quantization a second time to the quantization residuals.

**How it works:**
1. First pass: Weights are quantized (e.g., to NF4). The difference between original and quantized values = residual.
2. Second pass: The residuals are themselves quantized (typically to a smaller bit width, or even 1-bit if using signed integers).
3. The second-level quantizer's parameters (scale, mantissa) are stored alongside the model.

**Memory savings:** Double quantization typically adds ~0.5-1 bit of additional compression. So 4-bit + double quantization ≈ 4.5-5 bits total, rather than exactly 4 bits.

**Example:**
- Standard 4-bit (NF4 only): 4 bits/parameter
- 4-bit NF4 + double quantization: ~4.5 bits/parameter (~10-20% additional savings)
- 8-bit: 8 bits/parameter
- 4-bit without double quant: 4 bits/parameter

**In QLoRA:** Double quantization is one of the optional parameters in `BitsAndBytesConfig`. It's enabled by default in many QLoRA implementations because the additional memory savings are significant relative to the minimal quality loss.

**DomainAdapt status:** Double quantization is configured in the code but will only activate when CUDA is available. On CPU, the double quantization overhead may not be justified.

---

## 8. Why Are Pretrained Model Weights Frozen?

**The fundamental principle of Efficient Fine-Tuning (EFT):**

During standard full fine-tuning, all model weights are updated via backpropagation. This has several disadvantages:

1. **Catastrophic forgetting:** The model may lose general language capabilities when weights are updated for a specific task. Frozen weights preserve the original knowledge.

2. **Overparameterization:** LLMs have billions of parameters; fine-tuning all of them on a small dataset (hundreds/thousands of examples) is wasteful and prone to overfitting.

3. **Memory:** Storing optimizer states for all parameters requires significant memory (3× parameter count for Adam: fp32 weights + fp32 momentum + fp32 variance cache).

4. **Catastrophic interference:** Updates for the new task can interfere with previously learned capabilities.

**LoRA solves this by:**
- Freezing all pretrained weights W
- Introducing a small number of trainable parameters (the LoRA adapters A, B)
- The effective weight is W_eff = W + (α/r) · B · A
- During training, only A and B are updated; W remains exactly as pretrained
- After training, the adapter can be loaded onto any copy of the base model

**Why freeze rather than full fine-tuning on CPU:**
- 0.5B parameters × 3 (Adam states) ≈ 6 GB optimizer memory just for the states
- With a 0.5B model (~2 GB), total ~8 GB needed; Intel Iris Xe may have constrained shared memory
- LoRA reduces trainable parameters to ~1-3% of total, making CPU training feasible
- Frozen weights + LoRA adapters = same inference quality as base model, plus task adaptation

**DomainAdapt implementation:** Base model weights are always frozen. LoRA adapter parameters are the only trainable variables. This design allows:
- Standard LoRA training on CPU (no quantization needed)
- QLoRA 4-bit activation when GPU is available (quantized frozen weights)
- Adapter-only saving (small files, ~10-50 MB) that can be loaded onto the base model at inference time

---

## 9. Which Layers or Target Modules Are Selected and Why?

**Selected target modules for Qwen2.5-0.5B-Instruct:**

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

**Rationale:**

1. **Attention layers (q_proj, k_proj, v_proj, o_proj):** These are the query, key, value, and output projections in the transformer attention mechanism. They control how the model attends to relevant information. Adapting these layers is crucial for task-specific attention patterns.

2. **Feed-forward layers (gate_proj, up_proj, down_proj):** These form the SwiGLU feed-forward network in Qwen2.5. They process the attended representation and are important for task-specific feature extraction.

3. **Broad coverage:** Applying LoRA to all 7 projection types in each transformer block ensures the adapter can modify both the attention mechanism and the feed-forward processing. This is important for structured output tasks where both attention focus and feed-forward computation matter.

4. **Empirical evidence:** Research (e.g., LoRA paper, QLoRA studies) shows that targeting attention layers alone is often sufficient, but including feed-forward layers provides additional adaptation capacity without a proportional increase in parameter count.

5. **Qwen2.5-specific:** The Qwen2.5 architecture uses SwiGLU (not ReLU/GeLU) in the feed-forward network. The gate_proj is the gating switch, up_proj expands, and down_proj projects back. Including all three ensures the adapter can learn the gating behavior.

**Not selected (and why):**
- `embed_tokens`: Embedding layer; adapting it can change the model's vocabulary representation. Generally not recommended as it affects all inputs uniformly.
- `norm`: Layer normalization; adapting norm statistics can be unstable and is often unnecessary.
- `lm_head`: The output head; tying or adapting lm_head is equivalent to changing the model's output distribution. Usually kept frozen since the base model's lm_head is already well-optimized.

**Parameter count with r=8:**
- Qwen2.5-0.5B has ~32 transformer layers
- Each layer has 7 projection matrices targeted
- Hidden size = 3200 (approximately)
- Trainable params per layer: 2 × 3200 × 8 × 7 = 358,400 (approximately)
- Total trainable params: 358,400 × 32 ≈ 11.5M (approximately)
- Model total params: ~512M (0.5B)
- Wait, that doesn't add up. Let me recalculate.

Actually, for Qwen2.5, the hidden size is 3200 for the 0.5B model, but the projection dimensions vary. The q_proj, k_proj, v_proj have dimension (hidden_size, head_dim × n_heads), and o_proj has (head_dim × n_heads, hidden_size). The gate_proj, up_proj, down_proj have dimensions related to the intermediate size.

For a more accurate count, the PEFT library computes this automatically. With r=8, total trainable parameters for Qwen2.5-0.5B are typically around 1-5M, which is ~0.2-1.0% of the total model parameters. This is the intended efficiency: train ~1% of parameters while adapting the full model behavior.

---

## 10. How Does QLoRA Reduce GPU Memory Requirements?

**Memory comparison for Qwen2.5-0.5B:**

| Component | Full 16-bit (float16) | 4-bit QLoRA (NF4 + double quant) |
|-----------|----------------------|----------------------------------|
| Base model weights | 1,008 MB (0.5B × 2 bytes) | 252 MB (0.5B × 0.5 bytes) |
| Optimizer states (Adam) | 3,024 MB (3× weights) | 756 MB (3× 4-bit weights, paged) |
| Activated gradients | ~500-1,000 MB (varies) | ~125-250 MB (4-bit gradients) |
| **Total approximate** | **~4.5-5 GB** | **~1.1-1.4 GB** |

**Memory savings: ~60-70% reduction**, making it possible to fine-tune 0.5B models on a single T4 GPU (16 GB VRAM) and even 7B models on 24 GB VRAM (RTX 3090).

**Key mechanisms:**

1. **4-bit NF4 quantization:** Reduces weight storage from 2 bytes/parameter (float16) to 0.5 bytes/parameter (4-bit). For the 0.5B model: from ~1 GB to ~250 MB.

2. **Double quantization:** Adds ~0.5-1 bit of additional compression by quantizing the residuals from the first quantization pass.

3. **Paged optimizer states:** Adam's momentum and variance caches are stored in paged memory (like an OS virtual memory manager), reducing the overhead of optimizer states. Without paged optimizers, Adam states would require 3× the model memory even with 4-bit weights.

4. **Gradient checkpointing:** During backpropagation, activation memory is not stored for all layers; instead, activations are recomputed during the backward pass. This trades compute for memory, typically reducing activation memory by 30-50%.

5. **LoRA adapter memory:** The LoRA matrices (A, B) are tiny (as computed in Section 9) and stored in float32 or float16, regardless of the base model quantization. For r=8 on Qwen2.5-0.5B: ~10-20 MB.

**The resulting equation:**

Total memory ≈ (4-bit model weights) + (3× paged optimizer on 4-bit) + (LoRA adapters) + (checkpointed activations)

Without QLoRA (full 16-bit):
Total memory ≈ (16-bit model weights) + (3× Adam on 16-bit) + (LoRA adapters) + (checkpointed activations)

The 4-bit base model weights alone save ~75% of the weight memory, and paged optimizers prevent the optimizer states from becoming the memory bottleneck.

**DomainAdapt implementation:**
- On GPU: QLoRA activated with `bitsandbytes` config: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16`, `use_paged_opt=True`
- On CPU: Standard LoRA with full-precision model. The memory savings from quantization are not applicable, but the LoRA adapter remains small regardless.

---

## 11. What Are the Limitations of 4-Bit Training?

1. **CUDA requirement:** NF4 quantization/dequantization and paged optimizer kernels are CUDA-only. CPU fallback exists but is not practical for training (orders of magnitude slower, may not support all operations).

2. **Numerical precision:** 4-bit NF4 approximates the original float16/float32 weights. The approximation error accumulates during training, potentially affecting the final model quality. Empirical results show minimal quality loss (<1% on most benchmarks), but it is not lossless.

3. **Residual quantization errors:** Double quantization reduces but does not eliminate approximation error. The second quantization of residuals introduces additional error, though typically small.

4. **Gradient accuracy:** The 4-bit dequantized weights are used for both forward and backward passes. Gradients computed with quantized weights may differ from those with full-precision weights, especially for outlier weights that NF4 represents with fewer bits.

5. **Limited bit width:** 4 bits can only represent 16 unique values per "group" (typically per row or per tensor). For weights with wide dynamic ranges (e.g., outlier values), the quantization step may be too coarse, leading to information loss.

6. **Not suitable for all models:** Very small models ( < 1B parameters) may not benefit significantly from 4-bit quantization, since the memory reduction is less critical. The overhead of managing quantization may not be justified.

7. **Hardware dependency:** QLoRA's memory savings only apply to NVIDIA GPUs with CUDA support. AMD GPUs, Apple Silicon, and other accelerators may have different quantization support (or none).

**DomainAdapt mitigations:**
- Initial runs use standard LoRA (full precision) on CPU - no 4-bit limitations
- QLoRA code path designed and tested conceptually; activated only when CUDA available
- If running on Colab/Kaggle GPU, QLoRA 4-bit provides the memory efficiency needed for the 0.5B model
- Evaluation compares base model vs fine-tuned model regardless of quantization method, ensuring fair comparison

---

## 12. How Do Batch Size and Gradient Accumulation Affect Training?

**Batch size (per_device_train_batch_size):**
- Number of training examples processed before an optimizer step
- Larger batch sizes provide more stable gradient estimates but require more memory
- Small batch sizes (1-4) are common for CPU training and small models

**Gradient accumulation steps (gradient_accumulation_steps):**
- Simulates a larger effective batch size without requiring proportional memory
- Gradient from each micro-batch is accumulated (summed or averaged) before an optimizer step
- Effective batch size = per_device_batch_size × gradient_accumulation_steps

**Relationship:**
- Effective batch size = batch_size × accumulation_steps
- Memory cost ≈ (batch_size × accumulation_steps × seq_len) / some_constant
- By increasing accumulation_steps, you can achieve large effective batch sizes with memory cost of small micro-batches

**DomainAdapt configuration:**
- per_device_train_batch_size = 1
- gradient_accumulation_steps = 8
- Effective batch size = 8
- Memory cost ≈ same as batch_size=1, effective batch size = 8
- This is the minimum effective batch size for reasonable gradient estimates

**Learning rate considerations:**
- When effective batch size changes, learning rate should often be scaled proportionally (linear scaling rule)
- If effective batch size increases 2×, learning rate can increase 2× while keeping the same generalization
- For fixed learning rate: smaller effective batch sizes may require lower learning rates to avoid instability

**DomainAdapt learning rate = 2e-4** with effective batch size 8. This is a standard starting point. If we later experiment with effective batch sizes of 16, 32, we may adjust the learning rate proportionally.

**Gradient accumulation vs. large single batch:**
- Gradient accumulation: gradient = Σ micro-batch_gradients / accumulation_steps (average)
- Single large batch: gradient = single_batch_gradient (no averaging)
- Both approaches compute the same gradient in expectation, but accumulation provides stochasticity similar to small batches

---

## 13. What is Catastrophic Forgetting?

**Catastrophic forgetting** occurs when a model trained on a new task loses its previously acquired general capabilities.

**Cause:** When all model weights are updated via backpropagation on the new task, the weight changes can overwrite the representations that were useful for the original pretraining task. The model "forgets" how to perform general language tasks because the weights have been moved to optimize for the new task.

**Example:** Fine-tuning a general LLM on code-specific data may cause the model to lose some general knowledge, creative writing ability, or reasoning skills because the weights have been updated to prioritize code patterns.

**Why LoRA reduces this risk:**
- Only ~0.3% of parameters (LoRA adapters) are updated
- The vast majority of weights (99.7%+) remain exactly as pretrained
- The base model's general capabilities are preserved; only a small adapter is added on top
- After training, the adapter can be removed and the original model restored exactly

**Why QLoRA further reduces this risk:**
- In addition to frozen weights + LoRA, the base model is 4-bit quantized
- The quantization preserves the weight distribution means and variances
- The combination of quantized frozen weights + minimal trainable adapters is the most parameter-efficient fine-tuning approach

**DomainAdapt evaluation:**
- The base model is evaluated on the same task before fine-tuning (baseline)
- The fine-tuned model is evaluated after training
- Catastrophic forgetting would manifest as the fine-tuned model performing worse than the base model on certain capabilities
- This is why the baseline evaluation (Stage 3) is critical: without it, we cannot distinguish improvement from forgetting

---

## 14. How Do You Identify Overfitting?

**Overfitting** occurs when the model performs well on the training data but poorly on unseen data.

**Symptoms:**
- Training loss continues to decrease while validation loss starts to increase (or plateaus)
- Training accuracy/Exact Match improves steadily; validation metrics peak and then degrade
- The model memorizes training examples rather than learning general patterns
- Large gap between training and validation metrics (e.g., train EM 90%, val EM 60%)

**Diagnosis in fine-tuning:**
1. **Plot loss curves:** Track training loss and validation loss across epochs. Overfitting shows validation loss turning upward after initial decrease.

2. **Monitor exact match (EM):** Compute EM on a held-out validation set (not the test set). If training EM → 100% but validation EM stagnates at a much lower value, overfitting is likely.

3. **Learning curve analysis:** If both train and validation metrics improve linearly with more epochs, you may still be underfitting. If training metrics improve but validation plateaus/decreases, overfitting.

4. **Learning rate too high:** Can cause the training loss to decrease too quickly, fitting noise in the training data.

5. **Too many training epochs:** More epochs increase overfitting risk. Early stopping based on validation performance is recommended.

6. **LoRA rank too high:** High r gives the adapter more capacity to memorize training data. If r is much larger than necessary for the task complexity, overfitting risk increases.

**Prevention strategies:**
- **Early stopping:** Stop training when validation loss stops improving for N consecutive epochs.
- **Weight decay:** L2 regularization that penalizes large LoRA adapter weights. Default: 0.01 in DomainAdapt configuration.
- **Dropout on LoRA output:** Typical value: 0.05. Helps prevent co-adaptation of adapter units.
- **Reduce LoRA rank:** Lower r reduces adapter capacity, making overfitting less likely.
- **More training data:** More diverse examples reduce overfitting probability.
- **Reduce effective batch size:** Smaller batches have a regularizing effect (stochastic gradient noise).

**DomainAdapt implementation:**
- Weight decay: 0.01 (configs/training.yaml)
- LoRA dropout: configurable (default to be determined via ablation)
- Early stopping: monitored via validation loss; patience of 2 epochs without improvement
- Validation set: 100 examples held out from dataset (never seen during training)
- Test set: 100 examples, completely held out; used only once at the end for final comparison

---

## 15. How Do You Select the Best LoRA Rank?

**Rank selection methodology:**

1. **Theoretical lower bound:** The rank r should be at least as large as the "intrinsic dimensionality" of the task adaptation. If the task can be represented as a low-rank update to the pretrained weights, r need only be large enough to capture that subspace.

2. **Empirical sweep:** Train with multiple r values and compare validation performance. Typical values to try: 4, 8, 16, 32 (for 0.5B-1.5B models).

3. **Parameter budget:** Each increment of r adds ~2 × hidden_size × r × n_target_modules parameters. For Qwen2.5-0.5B with r=8: ~1.5M trainable params. r=16: ~3M. r=32: ~7.5M.

4. **Performance vs. cost:** Plot validation EM (or loss) vs. trainable parameter count. Select the smallest r that achieves acceptable performance.

5. **Learning rate interaction:** A smaller r with a well-tuned learning rate can be competitive with a larger r and default learning rate. This is because the learning rate effectively scales the update magnitude, and the product of rank and learning rate determines the effective adaptation strength.

**DomainAdant initial configuration: r = 8**

**Rationale:**
- For a 0.5B model, r=8 typically trains ~1-2M parameters (~0.2-0.4% of total)
- This is usually sufficient for instruction-following adaptation on focused domains
- r=8 with α=32 gives α/r = 4, a standard scaling ratio
- If performance is unsatisfactory, r can be increased to 16 or 32 in subsequent experiments
- r=8 is the most common starting point in LoRA research for models under 1.3B parameters

**Experiment sweep plan (Stage 5):**
- r ∈ {4, 8, 16} (3 configurations)
- All other hyperparameters held constant
- Validation EM used to select best r
- Final r selected based on documented evidence, not intuition

---

## 16. How Do You Distinguish Training Loss from Real Task Performance?

**Training loss** (cross-entropy loss on the training tokens) and **task performance** (exact match, execution accuracy, F1, etc. on unseen data) are related but distinct metrics.

**Why training loss alone is insufficient:**
- Loss can decrease while task performance plateaus or degrades (overfitting to noise in the training data)
- Loss is a token-level metric; task performance is an example-level or span-level metric
- Loss may continue to decrease with more epochs even after the model has memorized the training data
- Loss depends on batch size, learning rate, and model capacity, not just the model's ability to perform the task

**Why task-specific metrics are essential:**
- Exact Match (EM): fraction of examples where the predicted output exactly matches the ground truth
- Execution accuracy (for SQL): fraction of queries that produce correct results against a database
- F1 score: token-level overlap between prediction and ground truth, useful for generative tasks
- Per-category performance: some categories may overfit while others underfit

**Best practice: Always report task-specific metrics on a held-out validation set, not just training loss.**

**DomainAdapt evaluation protocol:**
1. **Baseline (Stage 3):** Evaluate base model on held-out test set → report task-specific metrics (EM, latency, etc.)
2. **Training (Stage 4-5):** Train LoRA adapter, monitoring validation loss, but also computing validation EM at each checkpoint
3. **Final evaluation (Stage 6):** Evaluate both base model and fine-tuned model on untouched test set → report before-and-after comparison
4. **Never report training loss as the primary success indicator** - it is only used for training diagnostics

The training loss is recorded in the training summary for debugging, but all project conclusions are based on test-set task performance metrics.

---

## 17. How Do You Evaluate Generative Models Fairly?

**Challenges with automatic metrics:**
- BLEU/ROUGE: n-gram overlap metrics that punish legitimate variations in correct outputs; not suitable for open-ended generation
- Perplexity: measures how "surprised" the model is by the test data, but doesn't correlate well with task performance
- Can be gamed by generating generic or repetitive outputs

**Fair evaluation strategies:**

1. **Held-out test set:** Never use the same examples for training and evaluation. The test set must be completely unseen during training and hyperparameter selection.

2. **Task-specific metrics:** Use metrics that directly measure the target capability:
   - For structured output: Exact match on JSON schema validation
   - For classification: Accuracy, precision, recall, F1
   - For SQL generation: Execution accuracy (does the query return correct results?)
   - For technical troubleshooting: Category accuracy, action correctness

3. **Human evaluation:** Have human reviewers rate output quality on a rubric. Essential when automated metrics cannot adequately measure quality (e.g., creative writing, code quality, technical accuracy).

4. **Side-by-side comparison:** Present outputs from base model and fine-tuned model to human reviewers, who select which is better. More reliable than individual automatic metrics.

5. **Error analysis:** Categorize failures (hallucination, schema violation, missing fields, wrong values). Provides more insight than a single summary number.

6. **Confidence intervals:** Report metric variability across multiple test runs or cross-validation folds. A 70% EM may be 65-75% with 95% confidence.

7. **Baseline comparison:** Always compare the fine-tuned model against the untouched base model on the same test set. Improvement is measured as the difference, not the absolute number.

**DomainAdapt evaluation protocol:**
- Primary metric: **Exact Match (EM)** - fraction of test examples where the generated JSON exactly matches the expected output (schema + values)
- Secondary metrics: 
  - Schema validity rate (fraction of outputs that are valid JSON matching the schema, even if values differ)
  - Category-wise EM
  - Invalid output rate (fraction of outputs that cannot be parsed as valid JSON or fail schema validation)
- Human evaluation: Optional, for response quality where automated metrics are insufficient
- Test set: 100 examples, completely held out from training and hyperparameter selection
- Baseline: Base model evaluated on same 100 examples before any fine-tuning

**Reporting format:**
- EM: XX.X% (n/N)
- Schema validity: XX.X% (n/N)
- Invalid output rate: XX.X% (n/N)
- Per-category EM table
- Representative successful and unsuccessful examples

---

## 18. What Are the Trade-offs Between Adapter Loading and Merged Models?

**Adapter loading (PEFT default):**
- The base model weights remain frozen; the LoRA adapter is loaded as a separate module
- Effective model: W_eff = W_base + (α/r) · B_adapter · A_adapter
- Advantages:
  - Small adapter files (10-50 MB vs 0.5 GB for full model)
  - Can switch between base and fine-tuned models by loading/unloading adapter
  - Multiple adapters can be loaded sequentially (though not simultaneously without care)
  - Base model unchanged; can be used for other tasks
- Disadvantages:
  - Requires two model loads (base + adapter) at inference time
  - Slightly slower inference (extra matrix multiplication for adapter)
  - Adapter configuration must be validated and compatible with base model

**Merged model:**
- The LoRA adapter weights are merged INTO the base model weights
- W_merged = W_base + (α/r) · B · A (permanently added to the base weight)
- After merging, the adapter weights can be discarded; the model is a standard model
- Advantages:
  - Single model load (same as any base model inference)
  - No extra adapter multiplication during inference (faster)
  - Simpler deployment (one model file)
  - Compatible with all inference engines (no PEFT dependency required at runtime)
- Disadvantages:
  - Permanent modification: once merged, the adapter cannot be "unmerged" easily
  - Loss of base model: the merged model cannot revert to the original unfine-tuned weights
  - Larger file size (merged model ≈ base model size, no adapter savings)
  - Version tracking: need to track which adapter was merged into which base model version

**When to use each:**

| Use case | Recommendation |
|----------|---------------|
| Research/experimentation | Use adapter loading; can swap adapters, test multiple fine-tunes on same base |
| Production deployment with limited GPU | Use merged model; single load, no PEFT overhead at runtime |
| Multiple fine-tuned variants | Use adapter loading; keep one base model, swap adapters as needed |
| API/server deployment | Use merged model if the fine-tune is stable; otherwise adapter loading with caching |
- User wants to compare base vs fine-tuned outputs interactively | Use adapter loading; can load/unload adapters per request |
- User wants minimal inference latency | Use merged model; no adapter computation overhead |
- User wants to preserve the original base model | Use adapter loading; base model weights unchanged |

**DomainAdapt implementation:**
- Both approaches are supported
- **Default inference (FastAPI, Streamlit):** Adapter loading via `PeftModel.from_pretrained()`. This allows:
  - User to switch between base and fine-tuned via UI
  - Both models available for side-by-side comparison
  - Adapter files stay small and separate
- **Optional merge:** A management command or API flag can merge the adapter into the base model for production deployment where adapter loading overhead is unacceptable
- **Merge validation:** After merging, the model is evaluated to ensure performance is preserved (no degradation from the merging process)

---

## 19. How Would You Deploy the Model Under Limited GPU Resources?

**Constraints:** No dedicated GPU available (CPU only, or limited GPU like Intel Iris Xe without CUDA QLoRA support).

**Deployment options:**

1. **CPU-only deployment:**
   - Model: Qwen2.5-0.5B-Instruct in float32
   - Memory: ~2 GB system RAM for model weights
   - Throughput: ~5-20 tokens/sec (varies by CPU, optimized BLAS)
   - LoRA adapter: ~10-20 MB additional
   - Framework: `torch.compile()` (PyTorch 2.x), `torch.inference_mode()`, optimized token generation
   - Trade-off: Lower throughput than GPU, but viable for occasional/inference use

2. **QLoRA 4-bit on GPU cloud (Colab/Kaggle):**
   - Activate QLoRA path when CUDA available
   - 4-bit NF4 reduces model to ~250 MB, fits on even small GPUs (T4: 16 GB VRAM)
   - LoRA adapter still ~10-20 MB
   - Training or inference on cloud GPU; download adapted model back to CPU for deployment
   - One-time cost: Colab/Kaggle free tier has runtime limits

3. **Distillation to even smaller model:**
   - Train a smaller model (e.g., 0.3B or 0.4B) to mimic the fine-tuned 0.5B behavior
   - Use the fine-tuned 0.5B as the "teacher" and distill knowledge
   - Result: smaller model with similar task performance, cheaper inference
   - Trade-off: Additional training step, potential quality loss

4. **Quantized inference with llama.cpp:**
   - Export the fine-tuned model (base + adapter) to GGUF format
   - Run quantized inference (Q4_K_M, Q5_K_M) with llama.cpp CPU backends (BLAS, BLIS, ZenDNN)
   - 4-bit quantized inference can achieve 10-30 tokens/sec on modern CPUs
   - Trade-off: Slight quality loss from additional quantization, but often negligible for domain adaptation

5. **Mixed approach:**
   - Fine-tune on GPU cloud (QLoRA 4-bit)
   - Export to quantized format (GGUF) for CPU deployment
   - Keep adapter training and research on GPU, deployment on CPU

**DomainAdapt deployment strategy:**
- **Primary:** CPU-only inference with float32 base model + LoRA adapter
  - Load base model (Qwen2.5-0.5B-Instruct, float32) into system RAM (~2 GB)
  - Load adapter on top via PEFT (adds ~10-20 MB)
  - Use `torch.inference_mode()` and any available CPU optimizations
  - Target throughput: sufficient for interactive use (≤10 sec per response)
  
- **Secondary (GPU available):** QLoRA 4-bit + merged model deployment
  - If Colab/Kaggle GPU used for training, merge adapter and export
  - Deploy merged 4-bit quantized model with llama.cpp for CPU inference
  
- **Tertiary (quantized export):** GGUF format with llama.cpp
  - Export base model + adapter to GGUF
  - Run Q4_K_M quantized inference on CPU
  - Slight quality trade-off, but enables higher throughput

**Key deployment constraint:** Never pretend GPU capabilities that aren't available. The FastAPI backend and Streamlit frontend will clearly report whether the model is running in CPU or (if applicable) GPU mode, and will gracefully handle the limitations.

---