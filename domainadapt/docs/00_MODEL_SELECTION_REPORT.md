## Model Selection Report

**Selected Model:** Qwen/Qwen2.5-0.5B-Instruct

**Considered Models:**

1. **Qwen/Qwen2.5-0.5B-Instruct** (0.5B params)
   - Architecture: Qwen2.5 with RoPE, SwiGLU, RMSNorm, GQA
   - Context length: 32,768 tokens
   - Already instruction-tuned
   - Available on Hugging Face (MIT license)
   - CPU-compatible (float32 full precision)
   - Why selected: Smallest Qwen2.5 variant, good instruction-following baseline, fits CPU memory

2. **TinyLlama/TinyLlama-1.1B-Chat-v1.1** (1.1B params)
   - Architecture: Llama-style decoder
   - Context length: 32,768 tokens
   - Instruction-tuned chat variant
   - Apache 2.0 license
   - Larger than needed; similar feasibility to 0.5B model

3. **distilgpt2** (124M params)
   - Smallest option but very limited reasoning capacity
   - Not an LLM in the modern sense; transformer decoder only
   - Used in CPU-only research studies but too small for meaningful domain adaptation

4. **openbmb/MiniCPM4-0.5B** (0.5B params)
   - Competitor to Qwen2.5-0.5B
   - Less documented; fewer community resources

**Why Qwen2.5-0.5B-Instruct:**
- Best balance of size, capability, and CPU compatibility
- Modern Qwen2.5 architecture with SwiGLU and RMSNorm
- Instruction-tuned already, reducing need for extensive data
- Active Hugging Face community and documentation
- MIT license - permissive for research and commercial use
- 32K context length supports longer instructions

**Rejected:** All other models either too small (distilgpt2), too large for CPU memory (7B+), or insufficiently documented/minimal community support.

Hardware note: 0.5B parameters × 4 bytes/parameter (float32) = ~2 GB model memory. With optimizer states (Adam: 2× additional + momentum = ~3× total), ~6-8 GB RAM required. Intel Iris Xe shared memory may be constrained; batch size of 1 with gradient accumulation of 8 is the feasible configuration.