## Hardware Feasibility Report

**Available Hardware:**
- Platform: Windows 11 (win32)
- CPU: Intel Iris Xe Graphics (integrated, no discrete NVIDIA GPU)
- PyTorch: 2.14.1 (CPU only, CUDA not available)
- RAM: System RAM available (not quantified precisely, but typical laptop/desktop range)

**Software Stack:**
- Python 3.11.7
- PyTorch 2.14.1 + CPU operator set
- Transformers 5.19.0
- PEFT 0.21.2
- bitsandbytes 0.50.2 (CUDA-dependent; 4-bit kernels require CUDA)
- No CUDA runtime available

**QLoRA 4-Bit Feasibility:**
- ❌ NOT feasible on current hardware
- bitsandbytes 4-bit quantization (NF4, double quantization) requires CUDA kernels
- Multiple research projects confirm: "QLoRA's 4-bit kernels require CUDA and aren't available on CPU" (Salik702 project)
- bitsandbytes CPU fallback exists but is performance-prohibitive and not officially supported for training

**Alternative: Standard LoRA (Full-Precision) on CPU:**
- ✅ FEASIBLE
- LoRA trains only adapter matrices (rank r × dimension), ~1-3% of total parameters
- All forward/backward passes use standard PyTorch operations (CUDA not required)
- Optimizer states (Adam) computed in float32 on CPU
- Memory: 0.5B model × 4 bytes × ~3 (optimizer) = ~6 GB RAM needed
- Training throughput: ~0.5-2 steps/sec on CPU (vs ~50-100 steps/sec on GPU)
- Expected epochs: 3-5 (vs 1-3 on GPU, due to slower throughput)

**Memory Estimation:**
| Component | Memory (approx) |
|-----------|----------------|
| Base model (Qwen2.5-0.5B, float32) | ~2 GB |
| Adam optimizer states (3× params) | ~6 GB |
| LoRA adapter weights (r=8, 2×r×dim) | ~10-50 MB |
| Activations + gradients | ~1-2 GB (with gradient checkpointing) |
| **Total estimated** | **~9-10 GB RAM** |

**Configuration Adjustments for CPU:**
1. **Batch size:** 1 (per device); gradient accumulation steps: 8 → effective batch size = 8
2. **Max seq length:** 512 (reduces per-step memory and compute)
3. **Gradient checkpointing:** True (trades compute for memory)
4. **Fp16/bf16:** False (no CUDA support; CPU fp16 may use software emulation)
5. **Learning rate:** 2e-4 (standard; may need tuning on CPU)
6. **Epochs:** 3 (lower than typical GPU 3-5 due to slower overfitting risk on small dataset)

**GPU Cloud Option:**
- If QLoRA 4-bit quantization is required, Google Colab or Kaggle T4/T100 GPU runtime can be used
- Unsloth-optimized Colab notebooks can train Qwen2.5-0.5B-Instruct in ~10-20 minutes with 4-bit
- Fallback: Run standard LoRA on CPU for methodology demonstration, document QLoRA conceptually

**Decision:** Stage 1-4 will use standard LoRA (full-precision) on CPU. QLoRA 4-bit will be documented as a GPU-optional enhancement. All architecture and code will be designed to support QLoRA when GPU becomes available, but the initial training runs will demonstrate valid LoRA fine-tuning on CPU.