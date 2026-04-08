---
sidebar_position: 3
slug: quantization
title: "Quantization"
---


# Quantization

:::info[What You'll Learn]
- How floating-point precision affects model size and speed
- INT8 and INT4 quantization techniques
- Post-training quantization (GPTQ, AWQ) vs. quantization-aware training
- Practical trade-offs: quality vs. speed vs. memory
:::

:::note[Prerequisites]
[Fine-Tuning Strategies](fine-tuning-strategies) from this month and [Training Infrastructure](/curriculum/month-4/training-infrastructure) from Month 4.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

A 70B-parameter model in FP16 requires **140 GB** of memory — more than fits on any single consumer GPU. Quantization shrinks models by representing weights (and sometimes activations) with fewer bits, reducing memory usage and increasing inference speed with minimal quality loss.

Quantization is the single most impactful optimization for making LLMs practical. It's how you run a 70B model on a 24GB GPU, or a 7B model on a phone.

## Why Quantize?

### Memory Savings

The memory footprint of a model scales linearly with the bit-width of its parameters:

| Format | Bits per Param | 7B Model Size | 70B Model Size |
|---|---|---|---|
| FP32 | 32 | 28 GB | 280 GB |
| FP16 / BF16 | 16 | 14 GB | 140 GB |
| INT8 | 8 | 7 GB | 70 GB |
| INT4 | 4 | 3.5 GB | 35 GB |

Going from FP16 to INT4 gives a **4× memory reduction**. A 70B model shrinks from 140 GB to 35 GB — within reach of two consumer GPUs.

### Speed Improvements

Quantized models are faster for two reasons:
1. **Memory bandwidth.** LLM inference is memory-bound during token generation. Fewer bits per weight = less data transferred from memory = faster.
2. **Compute throughput.** INT8 and INT4 operations are faster than FP16 on modern GPUs. NVIDIA's A100 delivers 624 TOPS for INT8 vs. 312 TFLOPS for FP16.

:::info[Arithmetic Intensity of LLM Inference]
During autoregressive decoding (generating one token at a time), each token requires reading **all** model weights from memory but performing relatively little computation per weight. This makes inference **memory-bandwidth-bound**, not compute-bound. Reducing the number of bits per weight directly translates to faster token generation.
:::

## Number Formats

### FP16 and BF16

Standard 16-bit formats used during training and inference:

```
FP16 (IEEE 754 half-precision):
┌───┬─────────┬──────────────────┐
│ S │ Exp (5) │ Mantissa (10)    │  Range: ±65504, Precision: ~3.3 decimal digits
└───┴─────────┴──────────────────┘

BF16 (Brain Float 16):
┌───┬──────────────┬─────────────┐
│ S │ Exp (8)      │ Mantissa (7)│  Range: same as FP32, Precision: ~2.4 decimal digits
└───┴──────────────┴─────────────┘
```

BF16 has the same range as FP32 but lower precision. It's preferred for training because the range prevents overflow in gradients and activations.

### INT8

8-bit integer quantization maps floating-point values to 256 discrete levels:

:::note[Symmetric Quantization]
For a floating-point tensor \(X\) with maximum absolute value \(\alpha = \max(|X|)\):

:::info[Plain English: What Is This Formula Doing?]
Think of quantization like reducing the resolution of a photo. A high-resolution photo (FP16) captures every subtle shade of color, but takes up lots of storage. Quantization is like saving that photo as a lower-resolution version (INT8 or INT4) — you lose some subtle detail, but the picture is still recognizable and takes up much less space. The "scale factor" is like knowing the original brightness range so you can approximately reconstruct the original colors.
:::

\[
X_{\text{int8}} = \text{round}\left(\frac{X}{\alpha} \times 127\right)
\]

**Reading the formula:** *X_int8* is the quantized (compressed) version of the original tensor *X*. We divide each value in *X* by the maximum absolute value *α* (normalizing everything to the range [-1, 1]), then multiply by 127 (the maximum value an INT8 number can represent), and round to the nearest integer.

:::info[Plain English: What Is This Formula Doing?]
This is the reverse step — "unzipping" the compressed photo back to a viewable image. We take the low-resolution integer values and scale them back up to approximate the original floating-point numbers. The result isn't perfect (some detail was lost during rounding), but it's close enough that the model still works well.
:::

\[
\hat{X} = \frac{X_{\text{int8}}}{127} \times \alpha
\]

**Reading the formula:** *X̂* (X-hat) is the reconstructed (dequantized) value — our approximation of the original. We reverse the process: divide the integer by 127 and multiply by the original scale *α*. This gives us back an approximation of the original floating-point number, with some small rounding error.

The **scale factor** \(s = \alpha / 127\) determines the mapping. The quantization error is bounded by \(s/2\).
:::

### INT4

4-bit quantization uses only 16 discrete levels. This sounds extreme, but LLM weights follow a roughly normal distribution — most values are near zero, with few outliers. Careful quantization can preserve most of the information.

### FP8

FP8 formats provide a middle ground between INT8 and FP16:

```
E4M3 (4 exponent, 3 mantissa bits): Range [-448, 448], good for weights
E5M2 (5 exponent, 2 mantissa bits): Range [-57344, 57344], good for gradients
```

FP8 is supported on NVIDIA's H100 and later GPUs, and is increasingly used for both training and inference.

```python title="Visualizing quantization effects"
import torch
import matplotlib.pyplot as plt
import numpy as np

def quantize_symmetric(tensor, bits):
    """Symmetric uniform quantization to n bits."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    alpha = tensor.abs().max()
    scale = alpha / qmax
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    dequantized = quantized * scale
    return dequantized, scale

# Simulate typical LLM weight distribution (approximately normal)
torch.manual_seed(42)
weights = torch.randn(10000) * 0.02  # Typical LLM weight scale

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
configs = [(8, "INT8 (256 levels)"), (4, "INT4 (16 levels)"), (2, "INT2 (4 levels)")]

for ax, (bits, title) in zip(axes, configs):
    dequantized, scale = quantize_symmetric(weights, bits)
    error = (weights - dequantized).abs()

    ax.hist(weights.numpy(), bins=100, alpha=0.5, label="Original", density=True)
    ax.hist(dequantized.numpy(), bins=100, alpha=0.5, label=f"{title}", density=True)
    ax.set_title(f"{title}\\nMSE: {error.pow(2).mean():.2e} | Max Error: {error.max():.4f}")
    ax.legend()
    ax.set_xlabel("Weight Value")

plt.tight_layout()
plt.savefig("quantization_comparison.png", dpi=150)
plt.show()
```

:::tip[Line-by-Line Walkthrough]
- **`qmin = -(2 ** (bits - 1))` / `qmax = 2 ** (bits - 1) - 1`** — Calculates the range of integer values for the given bit-width. For 8 bits: -128 to 127. For 4 bits: -8 to 7.
- **`alpha = tensor.abs().max()`** — Finds the largest absolute value in the tensor. This becomes the "ruler" — the scale of the original data.
- **`scale = alpha / qmax`** — Computes how much each integer step represents in the original floating-point space.
- **`torch.clamp(torch.round(tensor / scale), qmin, qmax)`** — The core quantization: divides each weight by the scale, rounds to the nearest integer, and clamps to the valid integer range.
- **`dequantized = quantized * scale`** — Converts back to floating point by multiplying the integers by the scale. This is what the model actually uses during inference.
- **`error = (weights - dequantized).abs()`** — Computes the quantization error — the difference between original and reconstructed values.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch matplotlib numpy
```

**Steps:**
1. Save to `quant_viz.py`
2. Run: `python quant_viz.py`

**Expected output:**
A figure with three histograms comparing original weight distributions with INT8, INT4, and INT2 quantized versions. INT8 is nearly identical to the original, INT4 shows slight discretization, and INT2 is visibly coarse. The figure is saved as `quantization_comparison.png`.

</details>

## Post-Training Quantization (PTQ)

PTQ quantizes a model **after training** without any further training or fine-tuning. You take a trained FP16 model and convert its weights to a lower precision. This is the most common approach for deploying LLMs.

### Naive (Round-to-Nearest) Quantization

The simplest approach: compute a scale factor per tensor (or per channel) and round each weight to the nearest quantized value.

```python title="Basic per-channel quantization"
import torch

def quantize_per_channel(weight: torch.Tensor, bits: int = 8):
    """Per-channel symmetric quantization for a 2D weight matrix."""
    qmax = 2 ** (bits - 1) - 1
    # Compute scale per output channel (row)
    scales = weight.abs().amax(dim=1) / qmax             # (out_features,)
    scales = scales.clamp(min=1e-8)                       # Avoid division by zero
    quantized = torch.clamp(
        torch.round(weight / scales.unsqueeze(1)), -qmax, qmax
    ).to(torch.int8)
    return quantized, scales

def dequantize_per_channel(quantized: torch.Tensor, scales: torch.Tensor):
    return quantized.float() * scales.unsqueeze(1)

# Example
W = torch.randn(4096, 4096) * 0.02
W_q, scales = quantize_per_channel(W, bits=8)
W_deq = dequantize_per_channel(W_q, scales)

print(f"Original dtype: {W.dtype}, size: {W.element_size() * W.numel() / 1e6:.1f} MB")
print(f"Quantized dtype: {W_q.dtype}, size: {W_q.element_size() * W_q.numel() / 1e6:.1f} MB")
print(f"Max abs error: {(W - W_deq).abs().max():.6f}")
print(f"Relative error: {(W - W_deq).norm() / W.norm():.6f}")
```

:::tip[Line-by-Line Walkthrough]
- **`scales = weight.abs().amax(dim=1) / qmax`** — Computes a separate scale factor for each row (output channel) of the weight matrix. Per-channel is more accurate than per-tensor because each row can have its own range.
- **`scales = scales.clamp(min=1e-8)`** — Prevents division-by-zero for rows that happen to be all zeros.
- **`torch.round(weight / scales.unsqueeze(1))`** — Divides each row by its scale, then rounds to the nearest integer. The `unsqueeze(1)` broadcasts the per-row scale across all columns.
- **`.to(torch.int8)`** — Converts the result to 8-bit integer type, saving memory (1 byte per value instead of 4).
- **`quantized.float() * scales.unsqueeze(1)`** — Dequantization: converts back to float and multiplies by the scale to approximate the original values.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `per_channel_quant.py`
2. Run: `python per_channel_quant.py`

**Expected output:**
```
Original dtype: torch.float32, size: 67.1 MB
Quantized dtype: torch.int8, size: 16.8 MB
Max abs error: 0.000162
Relative error: 0.002847
```
The quantized version is 4× smaller with very low error.

</details>

:::warning[The Outlier Problem]
Naive quantization breaks down when weight or activation tensors contain **outliers** — a few values that are much larger than the rest. With symmetric quantization, a single outlier forces the scale factor to be large, wasting most of the quantization range on values near zero. This is why advanced methods like GPTQ and AWQ exist.
:::

## GPTQ: Accurate Post-Training Quantization

GPTQ (Frantar et al., 2022) is a one-shot weight quantization method based on the **Optimal Brain Quantization** framework. Instead of independently rounding each weight, GPTQ quantizes weights **one at a time** and adjusts the remaining (not yet quantized) weights to compensate for the quantization error.

The key idea: after quantizing weight \(w_i\), update the remaining unquantized weights to minimize the overall output error. This compensatory adjustment significantly reduces quantization loss compared to naive rounding.

:::note[GPTQ Error Compensation]
GPTQ minimizes the squared error between the original and quantized layer outputs:

:::info[Plain English: What Is This Formula Doing?]
Imagine you're replacing an orchestra of 100 musicians with only 16 different sound samples (4-bit quantization). If you just replace each musician with the nearest sample, the overall sound might be off. GPTQ is smarter: after replacing one musician, it adjusts the remaining real musicians slightly to compensate, keeping the overall harmony as close to the original as possible. It uses a small set of real music recordings (calibration data) to guide these adjustments.
:::

\[
\min_{\hat{W}} \| WX - \hat{W}X \|_2^2
\]

**Reading the formula:** We want to find the quantized weight matrix *Ŵ* (W-hat) that minimizes the squared difference between the original layer's output (*WX*) and the quantized layer's output (*ŴX*). *W* is the original weight matrix, *X* is a small calibration dataset (input activations), and *‖...‖²* measures the total squared error. In other words: "make the quantized layer's outputs as close as possible to the original layer's outputs on real data."

where \(X\) is a small calibration dataset (128–256 samples). It uses the inverse Hessian \(H = 2X X^T\) to determine the optimal update to remaining weights after quantizing each column.
:::

```python title="Quantizing a model with GPTQ using AutoGPTQ"
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_id = "meta-llama/Llama-3.2-1B"
quant_output = "./llama-1b-gptq-4bit"

# Quantization configuration
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,      # Quantize in groups of 128 for better accuracy
    desc_act=True,       # Quantize in order of activation magnitude (more accurate)
    damp_percent=0.01,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model for quantization
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
)

# Prepare calibration data (128 examples from a diverse dataset)
from datasets import load_dataset

calibration_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:128]")
calibration_texts = [text for text in calibration_data["text"] if len(text) > 100]

calibration_examples = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calibration_texts[:128]
]

# Quantize (takes ~5-30 minutes depending on model size)
model.quantize(calibration_examples)

# Save quantized model
model.save_quantized(quant_output)
tokenizer.save_pretrained(quant_output)
print(f"Quantized model saved to {quant_output}")
```

:::tip[Line-by-Line Walkthrough]
- **`bits=4`** — Quantize weights to 4 bits (16 discrete levels per weight).
- **`group_size=128`** — Instead of one scale factor for an entire row, use a separate scale for every group of 128 consecutive weights. This gives much better accuracy because each small group can have its own range.
- **`desc_act=True`** — Quantize columns in decreasing order of activation magnitude. Columns that have the biggest impact on outputs are quantized first, when compensation is most effective.
- **`load_dataset("wikitext", ...)`** — Loads a small calibration dataset. GPTQ uses this to measure how quantization affects real outputs and to compute the compensatory weight adjustments.
- **`model.quantize(calibration_examples)`** — Runs the GPTQ algorithm: feeds the calibration data through the model, quantizes weights one column at a time, and adjusts remaining weights to compensate after each step.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install auto-gptq transformers datasets torch
```
You need a CUDA GPU.

**Steps:**
1. Save to `gptq_quantize.py` (add `import torch` at the top)
2. Run: `python gptq_quantize.py`

**Expected output:**
Progress bars showing each layer being quantized. The process takes 5–30 minutes for a 1B model. Final output:
```
Quantized model saved to ./llama-1b-gptq-4bit
```
The quantized model directory will be ~600 MB instead of ~2.5 GB.

</details>

### Group Quantization

Instead of using one scale factor per entire tensor or per channel, **group quantization** uses a separate scale factor for every `group_size` elements (typically 128). This dramatically improves accuracy at minimal memory cost — the scale factors themselves are tiny compared to the weights.

```
Weight row:  [w1, w2, ..., w128 | w129, w130, ..., w256 | ...]
              └── group 1 ──────┘ └── group 2 ──────────┘
              scale_1               scale_2

Each group gets its own scale factor, so outliers in one group
don't affect the quantization of other groups.
```

## AWQ: Activation-Aware Weight Quantization

AWQ (Lin et al., 2023) takes a different approach: not all weights are equally important. Weights connected to channels with **large activation magnitudes** have a disproportionate impact on output quality. AWQ identifies these "salient" channels and applies per-channel scaling to protect them before quantization.

:::info[AWQ's Key Insight]
Only ~1% of weight channels are critical for model quality, and they correspond to channels with large activation values. By scaling these channels up before quantization (and compensating with the inverse scale on the activations), you preserve the most important information while still quantizing to 4 bits.
:::

```python title="Quantizing with AWQ"
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
quant_output = "./llama-1b-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

quant_config = {
    "zero_point": True,       # Asymmetric quantization (better for skewed distributions)
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",       # Optimized GEMM kernel for fast inference
}

# Quantize with calibration data
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_output)
tokenizer.save_pretrained(quant_output)

# Load and use the quantized model
model = AutoAWQForCausalLM.from_quantized(quant_output, fuse_layers=True)
print("Model loaded in 4-bit AWQ format")
```

:::tip[Line-by-Line Walkthrough]
- **`"zero_point": True`** — Uses asymmetric quantization, which can handle weight distributions that aren't centered at zero. This is better for layers with biased distributions.
- **`"q_group_size": 128`** — Same group quantization as GPTQ: separate scale per 128 weights for better accuracy.
- **`"version": "GEMM"`** — Selects an optimized matrix multiplication kernel designed for quantized weights. This makes inference faster than naive dequantize-then-multiply.
- **`model.quantize(tokenizer, quant_config=quant_config)`** — Runs AWQ: identifies the most important weight channels using activation statistics, scales them up to protect them, then quantizes everything to 4 bits.
- **`AutoAWQForCausalLM.from_quantized(quant_output, fuse_layers=True)`** — Loads the quantized model and fuses operations (e.g., combining LayerNorm + Linear) for extra speed.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install autoawq transformers torch
```
You need a CUDA GPU.

**Steps:**
1. Save to `awq_quantize.py`
2. Run: `python awq_quantize.py`

**Expected output:**
Quantization progress messages, followed by:
```
Model loaded in 4-bit AWQ format
```

</details>

## bitsandbytes Library

The `bitsandbytes` library by Tim Dettmers provides the easiest path to quantized inference. It integrates directly with HuggingFace Transformers — you add two lines to your model loading code and get 8-bit or 4-bit quantization.

```python title="Quantized inference with bitsandbytes"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# --- INT8 quantization ---
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    load_in_8bit=True,
    device_map="auto",
)

# --- 4-bit NF4 quantization ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat4: optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16 for speed
    bnb_4bit_use_double_quant=True,        # Quantize the quantization constants
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Compare memory usage
def model_memory_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

print(f"8-bit model memory: ~{model_memory_mb(model_8bit):.0f} MB")
print(f"4-bit model memory: ~{model_memory_mb(model_4bit):.0f} MB")

# Generate text with quantized model
inputs = tokenizer("Explain quantum entanglement simply:", return_tensors="pt").to("cuda")
outputs = model_4bit.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

:::tip[Line-by-Line Walkthrough]
- **`load_in_8bit=True`** — The simplest way to load a model in 8-bit quantization. Just one flag and you get ~2× memory savings.
- **`bnb_4bit_quant_type="nf4"`** — Uses NormalFloat4, a 4-bit format whose quantization levels are spaced according to a normal distribution's shape. Since neural network weights are roughly bell-curve shaped, this wastes fewer levels on the tails.
- **`bnb_4bit_compute_dtype=torch.bfloat16`** — Even though weights are stored in 4 bits, computations are done in BF16 for numerical stability and speed.
- **`bnb_4bit_use_double_quant=True`** — Quantizes the scale factors themselves (which are normally stored in FP32), saving additional memory.
- **`model_memory_mb(model)`** — Helper function that calculates the total memory used by all model parameters.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers bitsandbytes torch accelerate
```
You need a CUDA GPU with at least 4 GB VRAM.

**Steps:**
1. Save to `bnb_quantize.py`
2. Run: `python bnb_quantize.py`

**Expected output:**
```
8-bit model memory: ~1235 MB
4-bit model memory: ~618 MB
Explain quantum entanglement simply: ...
```
Followed by a generated explanation of quantum entanglement.

</details>

:::tip[NF4 vs. FP4]
NormalFloat4 (NF4) is a 4-bit data type optimized for the fact that neural network weights follow an approximately normal distribution. It spaces quantization levels according to the normal distribution's quantiles, giving more precision near zero (where most weights live) and less precision in the tails. This makes NF4 measurably better than naive 4-bit quantization.
:::

## Quality vs. Speed Tradeoffs

The impact of quantization on model quality depends on the model size, the task, and the quantization method:

| Model Size | INT8 Quality Loss | INT4 Quality Loss | Notes |
|---|---|---|---|
| 1B | Noticeable | Significant | Small models are more sensitive |
| 7B | Minimal (~0.1%) | Small (~1-3%) | Sweet spot for INT4 |
| 13B+ | Negligible | Minimal (~0.5-1%) | Larger models tolerate quantization better |
| 70B+ | Negligible | Negligible (~0.3%) | Almost no quality loss at this scale |

:::info[Bigger Models Quantize Better]
Larger models have more redundancy in their weight matrices, so quantization removes less useful information. A 70B model in INT4 typically outperforms a 13B model in FP16 — you get better quality **and** use less memory by choosing a larger quantized model over a smaller full-precision one.
:::

```python title="Benchmarking quantization quality"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import time

def evaluate_perplexity(model, tokenizer, dataset, max_samples=100, max_length=512):
    """Evaluate perplexity on a dataset as a quality proxy."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        text = example["text"]
        if len(text) < 50:
            continue

        inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                          truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def benchmark_speed(model, tokenizer, prompt="The meaning of life is", n_tokens=100, n_runs=5):
    """Benchmark tokens per second."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    model.generate(**inputs, max_new_tokens=10, do_sample=False)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_per_sec = n_tokens / avg_time
    return tokens_per_sec

# Compare quantization levels on a model
eval_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

results = {}
for label, load_kwargs in [
    ("FP16", {"torch_dtype": torch.float16}),
    ("INT8", {"load_in_8bit": True}),
    ("INT4-NF4", {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)}),
]:
    print(f"\\nEvaluating {label}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **load_kwargs)

    ppl = evaluate_perplexity(model, tokenizer, eval_data)
    speed = benchmark_speed(model, tokenizer)
    mem = torch.cuda.max_memory_allocated() / 1e9

    results[label] = {"perplexity": ppl, "tokens/sec": speed, "gpu_mem_gb": mem}
    print(f"  Perplexity: {ppl:.2f} | Speed: {speed:.1f} tok/s | Memory: {mem:.2f} GB")

    del model
    torch.cuda.empty_cache()
```

:::tip[Line-by-Line Walkthrough]
- **`evaluate_perplexity(...)`** — Measures how "surprised" the model is by held-out text. Lower perplexity = better model. This is the standard way to check if quantization degraded quality.
- **`outputs = model(**inputs, labels=inputs["input_ids"])`** — Passes text through the model and computes the cross-entropy loss. The loss tells us how well the model predicts each token.
- **`torch.exp(torch.tensor(avg_loss)).item()`** — Converts average loss to perplexity (e^loss). Perplexity is more intuitive: a perplexity of 10 means the model is as uncertain as choosing from 10 equally likely options.
- **`benchmark_speed(...)`** — Measures how many tokens the model generates per second by timing multiple generation runs and averaging.
- **`torch.cuda.max_memory_allocated() / 1e9`** — Reports peak GPU memory usage in gigabytes, so you can compare memory footprints across quantization levels.
- **`del model; torch.cuda.empty_cache()`** — Frees GPU memory before loading the next model variant. Without this, you'd run out of memory.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers bitsandbytes datasets torch accelerate
```
You need a CUDA GPU with at least 8 GB VRAM (to fit the FP16 version).

**Steps:**
1. Save to `quant_benchmark.py`
2. Run: `python quant_benchmark.py`

**Expected output:**
```
Evaluating FP16...
  Perplexity: 15.23 | Speed: 42.1 tok/s | Memory: 2.50 GB

Evaluating INT8...
  Perplexity: 15.28 | Speed: 38.5 tok/s | Memory: 1.30 GB

Evaluating INT4-NF4...
  Perplexity: 15.65 | Speed: 45.2 tok/s | Memory: 0.72 GB
```
Exact values will vary by GPU. Notice perplexity barely changes while memory drops dramatically.

</details>

---

## Exercises

:::tip[Exercise 1: Quantization Error Analysis — beginner]

Take a weight matrix from a real pretrained model and quantize it at different bit widths (8, 4, 3, 2). For each:
1. Compute the MSE and maximum absolute error
2. Plot the error distribution
3. Visualize which weights have the largest errors
4. Does per-channel quantization significantly reduce error vs. per-tensor?

<details>
<summary>Hints</summary>

1. Load a real model and extract a weight matrix
2. Quantize it at 8, 4, 3, and 2 bits
3. Plot the distribution of quantization errors
4. Check if errors are correlated with weight magnitude

</details>

:::

:::tip[Exercise 2: GPTQ vs. AWQ vs. bitsandbytes — intermediate]

Quantize a 7B model to 4 bits using GPTQ, AWQ, and bitsandbytes NF4. Compare them on:
- Perplexity (WikiText-2)
- Generation quality (sample outputs on 10 diverse prompts)
- Inference speed (tokens per second)
- GPU memory usage
Which method gives the best quality? Which is fastest?

<details>
<summary>Hints</summary>

1. Quantize the same model with all three methods to INT4
2. Evaluate perplexity on WikiText-2 or C4
3. Benchmark generation speed (tokens/sec) and memory usage
4. Try both short and long prompt scenarios

</details>

:::

:::tip[Exercise 3: Quantization-Aware Fine-Tuning — advanced]

Investigate whether it matters if you quantize before or after fine-tuning. Take a 7B model and compare three approaches on a downstream task:
1. Quantize to 4-bit, then fine-tune with QLoRA
2. Fine-tune in FP16, then quantize to 4-bit
3. Fine-tune with QLoRA, merge, then quantize

Which approach gives the best quality? Which is most practical?

<details>
<summary>Hints</summary>

1. Quantize a model to 4-bit, then fine-tune with QLoRA on a downstream task
2. Compare against: (a) fine-tuning the FP16 model and then quantizing, (b) direct 4-bit inference without fine-tuning
3. The order matters: quantize-then-fine-tune vs. fine-tune-then-quantize

</details>

:::

---

## Resources

- **[GPTQ: Accurate Post-Training Quantization for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323)** _(paper)_ by Frantar et al., 2022 — The GPTQ paper — one-shot weight quantization with error compensation.

- **[AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)** _(paper)_ by Lin et al., 2023 — AWQ identifies and protects the most important weight channels during quantization.

- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** _(paper)_ by Dettmers et al., 2023 — QLoRA combines NF4 quantization with LoRA for efficient fine-tuning.

- **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** _(tool)_ by Tim Dettmers — GPU-accelerated 8-bit and 4-bit operations for PyTorch — the easiest quantization library.

- **[A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)** _(paper)_ by Gholami et al., 2021 — Comprehensive survey covering quantization theory, methods, and hardware considerations.

- **[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)** _(tool)_ — User-friendly GPTQ implementation — quantize and run models with a few lines of code.
