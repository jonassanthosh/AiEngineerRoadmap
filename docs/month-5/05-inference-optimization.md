---
sidebar_position: 5
slug: inference-optimization
title: "Inference Optimization"
---


# Inference Optimization

:::info[What You'll Learn]
- KV cache and why it matters for autoregressive generation
- FlashAttention and memory-efficient attention
- Continuous batching and PagedAttention (vLLM)
- Speculative decoding for faster generation
:::

:::note[Prerequisites]
[GPT Architecture](/curriculum/month-4/gpt-architecture) and [Quantization](quantization) from this month.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

Training an LLM is expensive but happens once. **Inference** happens millions of times and determines your cost-per-token, user-perceived latency, and throughput. This lesson covers the techniques that make LLM inference fast: KV-caching, speculative decoding, continuous batching, Flash Attention, and PagedAttention.

## The Autoregressive Bottleneck

LLMs generate text one token at a time. Each new token requires a full forward pass through the model, and that forward pass depends on **all previous tokens**. This creates a fundamental bottleneck.

For a sequence of length \(n\) with a model of \(L\) layers, \(d\) dimensions, and vocabulary size \(V\):

- **Prefill** (processing the prompt): processes all \(n\) tokens in parallel — compute-bound
- **Decode** (generating tokens): generates one token at a time — memory-bandwidth-bound

:::info[Prefill vs. Decode]
LLM inference has two distinct phases:
- **Prefill:** Process the entire input prompt in one forward pass. This is compute-bound because we're doing dense matrix multiplications on many tokens simultaneously.
- **Decode:** Generate output tokens one by one. This is memory-bandwidth-bound because each step loads the full model weights from GPU memory to compute a single token's output.

Most optimization effort focuses on the decode phase, since that's where users experience latency.
:::

## KV-Cache: Why It Matters

Without caching, generating token \(n+1\) would require recomputing attention over all \(n\) previous tokens — recalculating their key and value projections from scratch. The **KV-cache** stores the key and value vectors from all previous tokens, so each new token only needs to compute its own Q, K, V and attend to the cached keys/values.

```
Without KV-cache (generating token 5):
  Token 1 → compute K₁, V₁
  Token 2 → compute K₂, V₂
  Token 3 → compute K₃, V₃
  Token 4 → compute K₄, V₄
  Token 5 → compute K₅, V₅, attend to K₁₋₅, V₁₋₅

With KV-cache (generating token 5):
  Cache already has K₁₋₄, V₁₋₄ from previous steps
  Token 5 → compute K₅, V₅, append to cache, attend to K₁₋₅, V₁₋₅
```

This transforms generation from \(O(n^2)\) total computation to \(O(n)\) — each step does \(O(1)\) work instead of \(O(n)\).

```python title="KV-cache implementation"
import torch
import torch.nn as nn
import math

class CachedAttention(nn.Module):
    """Multi-head attention with KV-cache for efficient autoregressive generation."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            # Append new K, V to the cache
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        new_cache = (K, V)

        # Standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask: only needed during prefill (T > 1)
        if T > 1:
            causal_mask = torch.triu(torch.ones(T, K.size(2), device=x.device), diagonal=K.size(2) - T + 1)
            scores = scores.masked_fill(causal_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), new_cache

# Demonstration: prefill then decode
attn = CachedAttention(d_model=512, num_heads=8)

# Prefill: process 20 prompt tokens at once
prompt = torch.randn(1, 20, 512)
out, cache = attn(prompt, kv_cache=None)
print(f"Prefill output: {out.shape}")          # (1, 20, 512)
print(f"Cache K shape:  {cache[0].shape}")     # (1, 8, 20, 64)

# Decode: generate one token at a time
for step in range(5):
    new_token = torch.randn(1, 1, 512)
    out, cache = attn(new_token, kv_cache=cache)
    print(f"Step {step+1} — output: {out.shape}, cache length: {cache[0].shape[2]}")
```

:::tip[Line-by-Line Walkthrough]
- **`Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)`** — Projects input into queries, then reshapes into multi-head format: `(batch, heads, seq_len, head_dim)`.
- **`K = torch.cat([kv_cache[0], K], dim=2)`** — The key step for KV-caching: appends the new token's key vector to the cached keys from all previous tokens. The cache grows by one position each step.
- **`new_cache = (K, V)`** — Stores the updated keys and values for the next generation step. This is what makes subsequent steps O(1) instead of O(n).
- **`scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)`** — Standard scaled dot-product attention: the new token's query attends to all cached keys. The sqrt scaling prevents attention scores from growing too large.
- **`if T > 1:` (causal mask)** — During prefill (multiple tokens), we need a causal mask so each position can only see earlier positions. During decode (single token), the new token naturally attends to all cached positions.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `kv_cache.py`
2. Run: `python kv_cache.py`

**Expected output:**
```
Prefill output: torch.Size([1, 20, 512])
Cache K shape:  torch.Size([1, 8, 20, 64])
Step 1 — output: torch.Size([1, 1, 512]), cache length: 21
Step 2 — output: torch.Size([1, 1, 512]), cache length: 22
Step 3 — output: torch.Size([1, 1, 512]), cache length: 23
Step 4 — output: torch.Size([1, 1, 512]), cache length: 24
Step 5 — output: torch.Size([1, 1, 512]), cache length: 25
```
Notice the cache grows by 1 each step.

</details>

### KV-Cache Memory Requirements

The KV-cache can consume substantial memory, especially for long sequences and large models:

:::note[KV-Cache Size Formula]

:::info[Plain English: What Is This Formula Doing?]
The KV-cache is like a notebook where the model writes down "cheat notes" for every word it has seen so far. Every layer in the model keeps its own notebook, and each word gets two entries (a Key and a Value). For a big model processing a long conversation, this notebook can get enormous — sometimes even bigger than the model itself. This formula tells you exactly how big that notebook will be.
:::

\[
\text{KV cache size} = 2 \times L \times n \times h \times d_k \times b \times \text{bytes\_per\_element}
\]

**Reading the formula:** The total KV-cache memory equals: *2* (for both Keys and Values) × *L* (number of layers) × *n* (sequence length — how many tokens) × *h* (number of attention heads) × *d_k* (dimension per head) × *b* (batch size — how many sequences at once) × *bytes_per_element* (2 for FP16, 1 for INT8). Multiply all these together and you get the total memory in bytes.

where \(L\) = layers, \(n\) = sequence length, \(h\) = attention heads, \(d_k\) = head dimension, \(b\) = batch size, and the factor of 2 is for both K and V.

For Llama-3.1-70B (\(L=80\), \(h=64\), \(d_k=128\)) with a single sequence of 8192 tokens in FP16:

:::info[Plain English: What Is This Formula Doing?]
This plugs in real numbers for the Llama-3.1-70B model. With 80 layers, 64 attention heads, and a sequence of 8192 tokens stored in 16-bit precision, the KV-cache alone eats up 20.5 GB of GPU memory — just for one sequence! That's nearly as much as some entire models. This is why efficient KV-cache management (like PagedAttention) is so critical.
:::

\[
2 \times 80 \times 8192 \times 64 \times 128 \times 2 \text{ bytes} = \textbf{20.5 GB}
\]

**Reading the formula:** *2* for K and V × *80* layers × *8192* tokens × *64* heads × *128* dimensions per head × *2* bytes per FP16 value = 20.5 GB. That's the memory needed just for the cache of one sequence. Add the model weights on top of that, and you need a very large GPU.
:::

## Speculative Decoding

Speculative decoding accelerates generation by using a **small, fast draft model** to propose multiple tokens, then verifying them with the **large target model** in a single forward pass.

```
Standard decoding (5 tokens, 5 forward passes of large model):
  Large Model → token₁ → Large Model → token₂ → ... → Large Model → token₅

Speculative decoding (5 tokens, 1 draft + 1 verify):
  Draft Model → [t₁, t₂, t₃, t₄, t₅]  (fast, ~5× cheaper)
       │
  Large Model verifies all 5 at once   (single forward pass)
       │
  Accept: [t₁, t₂, t₃] ✓  Reject: [t₄, t₅] ✗
  Resample t₄ from large model's distribution
  Final: [t₁, t₂, t₃, t₄]  (4 tokens in 2 forward passes instead of 4)
```

The key property: speculative decoding produces **exactly the same distribution** as standard decoding from the large model. It's a lossless speedup — no quality degradation.

:::info[When Speculative Decoding Helps Most]
Speculative decoding works best when:
- The draft model has a high **acceptance rate** (its tokens match the target model's choices)
- The target model is significantly larger than the draft model
- The workload is latency-sensitive (single-user generation)

Typical speedups: 2–3× for well-matched draft/target pairs. The speedup is lower for creative/diverse generation and higher for deterministic/factual tasks.
:::

```python title="Speculative decoding concept"
import torch
import torch.nn.functional as F

def speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    gamma: int = 5,         # Number of speculative tokens
    temperature: float = 1.0,
):
    """Simplified speculative decoding (conceptual implementation)."""
    device = input_ids.device

    # Step 1: Draft model proposes gamma tokens autoregressively
    draft_ids = input_ids.clone()
    draft_probs_list = []
    for _ in range(gamma):
        with torch.no_grad():
            logits = draft_model(draft_ids).logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_probs_list.append(probs)
            draft_ids = torch.cat([draft_ids, next_token], dim=1)

    draft_tokens = draft_ids[:, input_ids.shape[1]:]  # The gamma proposed tokens

    # Step 2: Target model evaluates all gamma tokens in ONE forward pass
    with torch.no_grad():
        target_logits = target_model(draft_ids).logits
        # Get target probabilities at each drafted position
        target_probs = F.softmax(
            target_logits[:, input_ids.shape[1]-1:-1, :] / temperature, dim=-1
        )

    # Step 3: Accept/reject each token
    accepted = []
    for i in range(gamma):
        draft_p = draft_probs_list[i]
        target_p = target_probs[:, i, :]
        token = draft_tokens[:, i]

        # Acceptance criterion
        draft_prob = draft_p[0, token[0]]
        target_prob = target_p[0, token[0]]
        acceptance_ratio = target_prob / (draft_prob + 1e-10)

        if torch.rand(1).item() < acceptance_ratio.item():
            accepted.append(token)
        else:
            # Reject: sample from adjusted distribution
            adjusted = torch.clamp(target_p - draft_p, min=0)
            adjusted = adjusted / adjusted.sum()
            resampled = torch.multinomial(adjusted, 1)
            accepted.append(resampled.squeeze(-1))
            break  # Stop after first rejection

    accepted_tokens = torch.stack(accepted, dim=1)
    return torch.cat([input_ids, accepted_tokens], dim=1)

# In practice, use HuggingFace's built-in speculative decoding:
# model.generate(..., assistant_model=draft_model)
```

:::tip[Line-by-Line Walkthrough]
- **`for _ in range(gamma):`** — The draft model generates `gamma` (default 5) tokens one at a time. This is fast because the draft model is small.
- **`draft_probs_list.append(probs)`** — Saves the draft model's probability distribution at each step. We'll need these later to decide whether to accept or reject each token.
- **`target_logits = target_model(draft_ids).logits`** — The target model evaluates ALL draft tokens in a single forward pass. This is the key efficiency gain — one big forward pass instead of five.
- **`acceptance_ratio = target_prob / (draft_prob + 1e-10)`** — For each drafted token, computes the ratio of the target model's probability to the draft model's probability. If the target agrees (high target prob), the token is likely accepted.
- **`if torch.rand(1).item() < acceptance_ratio.item():`** — The acceptance test: if the target model assigns higher probability, the token is always accepted. If lower, it's accepted with probability equal to the ratio. This ensures the final output matches the target model's distribution exactly.
- **`adjusted = torch.clamp(target_p - draft_p, min=0)`** — On rejection, we sample from the "residual" distribution (target minus draft), ensuring we still produce tokens from the target's distribution.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers torch
```

**Steps:**
1. This is a conceptual implementation. To use speculative decoding in practice, use HuggingFace's built-in support:
```python
from transformers import AutoModelForCausalLM
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
draft = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
outputs = target.generate(input_ids, assistant_model=draft, max_new_tokens=100)
```

**Expected output:**
Generated text identical in distribution to what the target model would produce alone, but ~2-3× faster due to draft-then-verify batching.

</details>

## Continuous Batching

Traditional **static batching** waits until a batch is full, processes all sequences together, and waits for the longest sequence to finish before starting the next batch. This wastes GPU cycles because short sequences finish early and sit idle.

**Continuous batching** (also called **iteration-level batching**) inserts new requests into the batch as soon as any sequence finishes, keeping the GPU fully utilized at all times.

```
Static batching:                    Continuous batching:
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│ Req A: ████████████         │    │ Req A: █████████                 │
│ Req B: ████████████████████ │    │ Req B: ██████████████████        │
│ Req C: ████████████         │    │ Req C: █████████   Req D: ██████ │
│        ^^^^^^^^^^^^         │    │ Req E: ████████████              │
│        wasted compute       │    │ No wasted compute!               │
└─────────────────────────────┘    └──────────────────────────────────┘

  Wait for all to finish              New requests fill gaps instantly
```

:::tip[Throughput Impact]
Continuous batching can improve throughput by **2–5×** compared to static batching, depending on the variance in sequence lengths. This is the primary reason frameworks like vLLM and TGI are dramatically faster than naive HuggingFace `model.generate()` serving.
:::

## Flash Attention

Standard attention computes the full \(n \times n\) attention matrix and stores it in GPU high-bandwidth memory (HBM). **Flash Attention** (Dao et al., 2022) restructures the computation to avoid materializing this matrix, performing the attention computation in fast on-chip SRAM instead.

```
Standard Attention:                Flash Attention:
  1. Compute S = QK^T  (n×n)         1. Tile Q, K, V into blocks
  2. Store S in HBM                  2. For each block:
  3. Compute P = softmax(S)             - Load Q, K, V block to SRAM
  4. Store P in HBM                     - Compute local attention in SRAM
  5. Compute O = PV                     - Accumulate output with online softmax
  6. Load P from HBM                 3. No n×n matrix ever hits HBM

  Memory: O(n²)                      Memory: O(n)
  IO: many HBM round-trips           IO: minimal HBM access
```

:::note[Flash Attention Complexity]
Flash Attention has the same asymptotic **compute** complexity as standard attention — \(O(n^2 d)\). But it reduces **memory IO** from \(O(n^2 + nd)\) to \(O(n^2 d^2 / M)\), where \(M\) is the SRAM size. In practice, this yields a **2–4× wall-clock speedup** and enables much longer sequence lengths because the \(O(n^2)\) attention matrix never needs to be stored.
:::

```python title="Using Flash Attention in practice"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flash Attention 2 is enabled via attn_implementation parameter
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # Use Flash Attention
    device_map="auto",
)

# PyTorch 2.0+ includes scaled_dot_product_attention with Flash Attention
# This is used automatically by many models
import torch.nn.functional as F

B, H, T, D = 2, 32, 4096, 128  # batch, heads, seq_len, head_dim
Q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
K = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
V = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

# PyTorch automatically selects the best kernel (Flash, Memory-Efficient, or Math)
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print(f"Flash Attention output: {output.shape}")  # (2, 32, 4096, 128)
```

:::tip[Line-by-Line Walkthrough]
- **`attn_implementation="flash_attention_2"`** — Tells HuggingFace to use Flash Attention 2 for all attention layers in the model. This is the easiest way to get a 2–4× speedup on long sequences.
- **`B, H, T, D = 2, 32, 4096, 128`** — Sets up a realistic attention scenario: 2 sequences in a batch, 32 attention heads, 4096 tokens long, 128 dimensions per head. Standard attention would need a 4096×4096 matrix per head — Flash Attention avoids storing it.
- **`torch.backends.cuda.sdp_kernel(enable_flash=True, ...)`** — Explicitly selects the Flash Attention kernel. PyTorch 2.0+ can auto-select, but this forces it for benchmarking purposes.
- **`F.scaled_dot_product_attention(Q, K, V, is_causal=True)`** — PyTorch's built-in fused attention that automatically uses Flash Attention when available. `is_causal=True` adds the causal mask for autoregressive models.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers torch flash-attn
```
You need a CUDA GPU (Ampere or later recommended for best Flash Attention support).

**Steps:**
1. Save to `flash_attn_demo.py`
2. Run: `python flash_attn_demo.py`

**Expected output:**
```
Flash Attention output: torch.Size([2, 32, 4096, 128])
```
If Flash Attention isn't available on your GPU, it will fall back to the math or memory-efficient kernel.

</details>

## PagedAttention (vLLM)

The KV-cache for each request has a variable and unpredictable size (it grows with each generated token). Traditional systems pre-allocate the maximum possible KV-cache for each request, wasting memory. **PagedAttention** (Kwon et al., 2023) manages KV-cache memory like a virtual memory system, allocating memory in fixed-size **pages** on demand.

```
Traditional KV-cache:               PagedAttention:
┌────────────────────────────┐      ┌──────────────────────────────┐
│ Request 1: ████████░░░░░░░ │      │ Page table:                  │
│ Request 2: ████░░░░░░░░░░░ │      │  Req 1 → [P3, P7, P1]       │
│ Request 3: ██████████░░░░░ │      │  Req 2 → [P5, P2]           │
│            ^^^^^^^^^^^^^^^^ │      │  Req 3 → [P4, P8, P6]       │
│            wasted memory    │      │                              │
└────────────────────────────┘      │ Pages allocated on demand    │
                                    │ No waste, no fragmentation   │
  ~60-80% memory waste              └──────────────────────────────┘
                                      ~5% memory waste
```

PagedAttention is the core innovation behind **vLLM** and enables:
- **Near-zero memory waste** — pages are allocated only as needed
- **Memory sharing** across requests (e.g., shared system prompts)
- **2–4× higher throughput** compared to static allocation

## TensorRT-LLM

NVIDIA's TensorRT-LLM is a high-performance inference library that combines multiple optimizations:

- **Kernel fusion:** Combines multiple operations (e.g., LayerNorm + Linear) into a single GPU kernel
- **Quantization:** INT8 and FP8 quantization with hardware-optimized kernels
- **In-flight batching:** NVIDIA's term for continuous batching
- **Paged KV-cache:** Similar to PagedAttention
- **Tensor parallelism:** Splits the model across multiple GPUs

```python title="TensorRT-LLM model compilation (conceptual)"
# TensorRT-LLM provides a builder pattern for compiling optimized models.
# This is a conceptual walkthrough — actual API may vary by version.

# Step 1: Convert a HuggingFace model to TensorRT-LLM format
# (usually done via CLI)
#
# python convert_checkpoint.py \\
#     --model_dir meta-llama/Llama-3.2-1B-Instruct \\
#     --output_dir ./trt_ckpt \\
#     --dtype float16 \\
#     --tp_size 1

# Step 2: Build the TensorRT engine
# trtllm-build \\
#     --checkpoint_dir ./trt_ckpt \\
#     --output_dir ./trt_engine \\
#     --gemm_plugin float16 \\
#     --max_batch_size 32 \\
#     --max_input_len 2048 \\
#     --max_seq_len 4096 \\
#     --paged_kv_cache enable

# Step 3: Run inference
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./trt_engine")

prompts = [
    "Explain machine learning in simple terms:",
    "Write a Python function to sort a list:",
]

outputs = runner.generate(
    prompts,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\\n")
```

:::tip[Line-by-Line Walkthrough]
- **`convert_checkpoint.py`** — Converts a HuggingFace model into TensorRT-LLM's internal format. `--tp_size 1` means single-GPU; increase for multi-GPU tensor parallelism.
- **`trtllm-build`** — Compiles the model into an optimized TensorRT engine with fused kernels, quantization, and memory planning. This is a one-time cost that produces a fast inference engine.
- **`--paged_kv_cache enable`** — Enables PagedAttention-style KV-cache management for efficient memory use with variable-length sequences.
- **`ModelRunner.from_dir("./trt_engine")`** — Loads the compiled engine for inference. This is much faster than loading a raw HuggingFace model.
- **`runner.generate(prompts, ...)`** — Runs batched inference with the compiled engine. Multiple prompts are processed efficiently with continuous batching.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tensorrt-llm
```
TensorRT-LLM requires NVIDIA GPUs and specific driver versions. Installation is more involved than other libraries — see the [official documentation](https://github.com/NVIDIA/TensorRT-LLM).

**Steps:**
1. Convert the model checkpoint (Step 1 CLI command)
2. Build the TensorRT engine (Step 2 CLI command)
3. Run inference with the Python script (Step 3)

**Expected output:**
Generated text for each prompt. TensorRT-LLM typically achieves 2–5× throughput improvement over raw HuggingFace inference.

</details>

## Benchmarking Throughput and Latency

When evaluating inference performance, measure these key metrics:

| Metric | Definition | Typical Values (7B model) |
|---|---|---|
| **Time to First Token (TTFT)** | Time from request to first output token | 50–200 ms |
| **Token throughput** | Output tokens per second (per request) | 30–100 tok/s |
| **Total throughput** | Tokens per second across all concurrent requests | 500–5000 tok/s |
| **Latency (P50/P99)** | Median/tail end-to-end response time | 1–5 s / 5–15 s |

```python title="Inference benchmarking script"
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_inference(model, tokenizer, prompts: list[str],
                        max_new_tokens: int = 128, n_warmup: int = 2):
    """Benchmark inference throughput and latency."""
    device = next(model.parameters()).device

    # Warmup
    for _ in range(n_warmup):
        inputs = tokenizer(prompts[0], return_tensors="pt").to(device)
        model.generate(**inputs, max_new_tokens=10, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        output_len = outputs.shape[1] - input_len
        tokens_per_sec = output_len / elapsed

        results.append({
            "prompt_tokens": input_len,
            "output_tokens": output_len,
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
        })

    # Summary
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_latency = sum(r["elapsed_sec"] for r in results) / len(results)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Avg tokens/sec':<25} {avg_tps:>10.1f}")
    print(f"{'Avg latency (sec)':<25} {avg_latency:>10.3f}")
    print(f"{'Avg output tokens':<25} {sum(r['output_tokens'] for r in results)/len(results):>10.0f}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{'Peak GPU memory (GB)':<25} {peak_mem:>10.2f}")

    return results

# Usage
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

test_prompts = [
    "Explain what a neural network is in 3 sentences.",
    "Write a haiku about programming.",
    "List 5 benefits of exercise.",
    "What is the capital of Japan and why is it significant?",
]

results = benchmark_inference(model, tokenizer, test_prompts)
```

:::tip[Line-by-Line Walkthrough]
- **`model.generate(**inputs, max_new_tokens=10, do_sample=False)`** — Warmup runs: the first few forward passes on a GPU are slower due to CUDA kernel compilation and memory allocation. We run a few throwaway generations first.
- **`torch.cuda.synchronize()`** — Forces the CPU to wait until all GPU operations complete. Without this, timing measurements would be inaccurate because GPU operations are asynchronous.
- **`time.perf_counter()`** — High-resolution timer for measuring elapsed wall-clock time.
- **`output_len = outputs.shape[1] - input_len`** — Calculates how many new tokens were generated by subtracting the prompt length from the total output length.
- **`tokens_per_sec = output_len / elapsed`** — The key metric: how many tokens per second the model generates. Higher is better.
- **`torch.cuda.max_memory_allocated() / 1e9`** — Reports the peak GPU memory usage in gigabytes during the benchmark.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers torch accelerate
```
You need a CUDA GPU.

**Steps:**
1. Save to `benchmark.py`
2. Run: `python benchmark.py`

**Expected output:**
```
Metric                       Value
-------------------------------------
Avg tokens/sec                 45.2
Avg latency (sec)             2.832
Avg output tokens               128
Peak GPU memory (GB)           2.50
```
Values vary significantly by GPU. A100 will be much faster than a consumer GPU.

</details>

---

## Exercises

:::tip[Exercise 1: KV-Cache Size Calculator — beginner]

Write a function that calculates the KV-cache memory requirements for any model configuration. Use it to compute the cache size for Llama-3.1-8B, Llama-3.1-70B, and Llama-3.1-405B at sequence lengths 2048, 8192, and 32768. At what point does the KV-cache exceed the model weights in memory?

<details>
<summary>Hints</summary>

1. Use the formula: 2 × layers × seq_len × heads × head_dim × bytes_per_element
2. For FP16, bytes_per_element = 2; for INT8 KV-cache, it's 1
3. Calculate for batch sizes 1, 8, 32, 128
4. Compare KV-cache size to model weight size

</details>

:::

:::tip[Exercise 2: Speculative Decoding Speedup — intermediate]

Implement speculative decoding using HuggingFace's `model.generate(assistant_model=draft_model)`. Measure:
1. Acceptance rate across different prompt types (factual, creative, code)
2. Speedup over standard decoding at different gamma values
3. Verify that outputs are statistically indistinguishable from standard decoding

<details>
<summary>Hints</summary>

1. Use a small model (e.g. 1B) as draft and a larger model (e.g. 3B) as target
2. Measure acceptance rate at different temperatures
3. Compare generation time with and without speculation
4. Try different gamma values (number of speculative tokens): 3, 5, 8, 12

</details>

:::

:::tip[Exercise 3: Flash Attention Benchmark — intermediate]

Benchmark Flash Attention vs. standard (math) attention across sequence lengths from 256 to 16384. For each sequence length, measure:
1. Forward pass time
2. Backward pass time
3. Peak GPU memory
Plot the results. At what sequence length does Flash Attention's advantage become dramatic?

<details>
<summary>Hints</summary>

1. Use torch.nn.functional.scaled_dot_product_attention
2. Compare with enable_flash=True vs. enable_math=True
3. Test sequence lengths from 256 to 16384
4. Measure both time and peak GPU memory

</details>

:::

---

## Resources

- **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)** _(paper)_ by Dao et al., 2022 — Flash Attention — reduces attention memory from O(n²) to O(n) by computing in SRAM tiles.

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** _(paper)_ by Kwon et al., 2023 — PagedAttention (vLLM) — applies virtual memory concepts to KV-cache management.

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)** _(paper)_ by Leviathan et al., 2022 — Speculative decoding — use a small draft model to accelerate generation from a large model.

- **[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)** _(paper)_ by Yu et al., 2022 — Introduces iteration-level (continuous) batching for LLM serving.

- **[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)** _(paper)_ by Dao, 2023 — Flash Attention 2 — further optimizations yielding up to 2× speedup over Flash Attention 1.

- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** _(tool)_ by NVIDIA — NVIDIA's high-performance LLM inference library with kernel fusion, quantization, and tensor parallelism.
