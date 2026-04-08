---
sidebar_position: 1
slug: reading-papers
title: "Reading and Implementing ML Papers"
---


# Reading and Implementing ML Papers

The ability to read a machine learning paper, understand its contributions, and turn it into working code is the single most important skill that separates an AI engineer from someone who only uses APIs. In this lesson, you'll learn a systematic approach to reading papers and a practical workflow for going from PDF to PyTorch.

:::tip[Why Read Papers?]
Every library you use — Transformers, Flash Attention, LoRA, vLLM — started as a paper. If you can read and implement papers, you can adopt techniques months before they appear in libraries, debug issues others can't, and even contribute novel improvements.
:::

## The Three-Pass Method

Most researchers don't read a paper once from start to finish. They use a multi-pass approach that lets them quickly triage papers and invest time proportionally to relevance.

### Pass 1: Survey (5–10 minutes)

The goal is to decide whether this paper is worth your time.

1. Read the **title, abstract, and introduction** carefully.
2. Read the **section headings** — this tells you the structure.
3. Read the **conclusion** — authors summarize their actual contributions here (often more honestly than the abstract).
4. Glance at the **figures and tables** — especially Figure 1 (usually the architecture overview) and Table 1 (usually the main results).
5. Check the **references** — are there foundational papers you recognize?

After Pass 1, you should be able to answer:
- What problem does this paper solve?
- What's the general approach?
- Is it relevant to what I'm working on?

### Pass 2: Comprehension (1–2 hours)

Now read the full paper, but skip dense proofs and derivations on this pass.

1. Study the **figures and diagrams** carefully. Each one encodes a lot of information.
2. Mark terms and notation you don't understand.
3. Read the **method section** closely — this is where the actual contribution lives.
4. Study the **experimental setup** — what datasets, baselines, and metrics did they use?
5. Read the **results** — but focus on the delta (improvement over baselines), not the absolute numbers.

:::info[Reading Ablation Studies]
Ablation tables are often the most informative part of a paper. They show what happens when you remove individual components of the proposed method. If removing component X drops performance by 8% but removing component Y only drops it by 0.3%, you know X is the real contribution and Y is optional.
:::

### Pass 3: Reimplementation (4–8 hours)

This is where deep understanding happens. You mentally (or actually) re-derive the key equations and attempt to reimplement the core algorithm.

1. Work through the **math** line by line. Verify dimensions, check edge cases.
2. Try to **re-derive** key equations from first principles.
3. Identify any **gaps** between what the paper describes and what you'd need to code.
4. Check the **appendix** — implementation details are often buried there.
5. Look for an **official codebase** and compare your understanding against it.

## Understanding ML Notation

ML papers use notation heavily. Here's a reference table for the most common conventions.

| Symbol | Typical Meaning |
|--------|----------------|
| \( x \in \mathbb{R}^d \) | Input vector of dimension d |
| \( W \in \mathbb{R}^{m \times n} \) | Weight matrix (m rows, n columns) |
| \( \theta \) | All learnable parameters |
| \( \mathcal{L} \) | Loss function |
| \( \nabla_\theta \mathcal{L} \) | Gradient of loss w.r.t. parameters |
| \( \mathbb{E}_{x \sim p} \) | Expectation over distribution p |
| \( \text{softmax}(z)_i \) | \( e^{z_i} / \sum_j e^{z_j} \) |
| \( \| \cdot \| \) | Norm (usually L2 unless specified) |
| \( \odot \) | Element-wise (Hadamard) product |
| \( \oplus \) | Concatenation |
| \( [B, T, D] \) | Tensor shape: batch, sequence length, dimension |

:::warning[Notation Is Not Standardized]
Different authors use different notation for the same thing. The query/key/value matrices in attention are variously called \( W_Q, W_K, W_V \) or \( W^Q, W^K, W^V \) or even \( P_q, P_k, P_v \). Always check how a paper defines its terms — don't assume from other papers.
:::

## Identifying Key Contributions

Not every paper introduces something genuinely new. Here's how to separate signal from noise:

**Signs of a key contribution:**
- A new architectural component that replaces an existing one with clear benefits
- A training technique that enables something previously impossible (e.g., training at larger scale)
- A theoretical insight that explains why existing methods work (or fail)
- An empirical finding that changes how practitioners think about a problem

**Signs of incremental work:**
- "We add X to Y and get 0.5% improvement on benchmark Z"
- The method only works on a specific dataset or narrow setting
- The ablation study shows most components contribute negligibly
- No comparison against the strongest baselines

:::tip[Track the Lineage]
Every paper builds on prior work. When you read a paper, note which prior papers it directly extends. Tracing this lineage — for example, Attention → Transformer → GPT → GPT-2 → GPT-3 → InstructGPT — helps you understand *why* certain design choices were made.
:::

## From Paper to Code: A Practical Workflow

Here's a step-by-step workflow for implementing a paper in PyTorch.

### Step 1: Identify the Core Algorithm

Strip away the prose and identify the exact mathematical operations. Write pseudocode.

### Step 2: Set Up Tensor Shape Annotations

Before writing any code, map every variable to its tensor shape. This is the single most effective debugging technique for neural network code.

```python title="Shape annotation conventions"
# Always annotate shapes in comments:
# x: [B, T, D]       — batch, sequence length, model dimension
# W_q: [D, D_k]      — projection to key/query dimension
# attn: [B, H, T, T] — attention weights per head

# Example: multi-head attention shapes
B, T, D = 4, 128, 512
H = 8          # number of heads
D_k = D // H   # per-head dimension = 64

import torch
x = torch.randn(B, T, D)
W_q = torch.randn(D, D)

# Project: [B, T, D] @ [D, D] → [B, T, D]
Q = x @ W_q

# Reshape to heads: [B, T, D] → [B, T, H, D_k] → [B, H, T, D_k]
Q = Q.view(B, T, H, D_k).transpose(1, 2)
print(f"Q shape: {Q.shape}")  # [4, 8, 128, 64]
```

:::tip[Line-by-Line Walkthrough]
- **`B, T, D = 4, 128, 512`** — Sets up our dimensions: 4 sequences in the batch, each 128 tokens long, each token represented by 512 numbers.
- **`H = 8` / `D_k = D // H`** — We split attention into 8 heads, each working with 64 dimensions (512 ÷ 8).
- **`x = torch.randn(B, T, D)`** — Creates random input data shaped like a real batch of token embeddings.
- **`W_q = torch.randn(D, D)`** — A random weight matrix that projects input into "query" space.
- **`Q = x @ W_q`** — Matrix multiplication: every token's 512-number vector gets transformed into a new 512-number query vector.
- **`Q = Q.view(B, T, H, D_k).transpose(1, 2)`** — Reshapes the flat 512 dimensions into 8 heads of 64 each, then rearranges so the "head" dimension comes before the "time" dimension (needed for parallel head computation).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `shape_annotations.py`
2. Open a terminal and run: `python shape_annotations.py`

**Expected output:**
```
Q shape: torch.Size([4, 8, 128, 64])
```

</details>

### Step 3: Build and Test Components Individually

Never build the whole model at once. Implement each component as a standalone module and verify shapes.

### Step 4: Compare Against Reference

If the paper has official code, compare your implementation against it layer by layer.

```python title="Comparing your implementation against a reference"
import torch
import torch.nn as nn

def compare_outputs(your_module, ref_module, input_tensor, atol=1e-5):
    """Compare outputs of two modules with the same weights."""
    # Copy weights from reference to your module
    your_module.load_state_dict(ref_module.state_dict())

    with torch.no_grad():
        your_out = your_module(input_tensor)
        ref_out = ref_module(input_tensor)

    match = torch.allclose(your_out, ref_out, atol=atol)
    if match:
        print("✓ Outputs match!")
    else:
        diff = (your_out - ref_out).abs()
        print(f"✗ Max difference: {diff.max().item():.2e}")
        print(f"  Mean difference: {diff.mean().item():.2e}")
    return match
```

:::tip[Line-by-Line Walkthrough]
- **`your_module.load_state_dict(ref_module.state_dict())`** — Copies all the learned weights from the reference module into yours so both start with identical parameters.
- **`with torch.no_grad():`** — Tells PyTorch not to track gradients — we're just comparing outputs, not training.
- **`your_out = your_module(input_tensor)`** — Runs the same input through your implementation.
- **`torch.allclose(your_out, ref_out, atol=atol)`** — Checks if every number in both outputs is within `atol` (absolute tolerance) of each other — tiny floating-point differences are OK.
- **`diff = (your_out - ref_out).abs()`** — If they don't match, compute the absolute difference at every position to find where they diverge.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `compare_outputs.py`
2. You'll need to define `your_module` and `ref_module` as two `nn.Module` instances with the same architecture
3. Run: `python compare_outputs.py`

**Expected output:**
```
✓ Outputs match!
```
(If your implementation is correct, the max difference will be near zero.)

</details>

## Worked Example: Implementing RMSNorm from the LLaMA Paper

Let's walk through a full implementation example. RMSNorm (Root Mean Square Layer Normalization) was used in LLaMA instead of standard LayerNorm. The paper references the original RMSNorm paper by Zhang & Sennrich (2019).

**The math from the paper:**

:::info[Plain English: What Does This Formula Mean?]
Imagine you have a row of numbers (like test scores). First, you figure out how "big" those numbers are on average by squaring them all, averaging the squares, and taking the square root — that gives you one number called the RMS (Root Mean Square). Then you shrink each original number by dividing it by the RMS so they're all on a similar scale. Finally, you multiply each scaled number by a learnable "gain knob" that the model adjusts during training. It's like normalizing everyone's scores so they're comparable, then letting the model decide how much to amplify each one.
:::

\[
\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} \cdot g_i, \quad \text{where } \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^n a_i^2}
\]

**Reading the formula:** \( \bar{a}_i \) is the normalized output for the *i*-th element. \( a_i \) is the original input value. \( \text{RMS}(\mathbf{a}) \) is the Root Mean Square — a single number measuring how large the values in the vector are overall. \( n \) is how many elements are in the vector. \( g_i \) is a learnable gain parameter (one per element) that the model tunes during training. The \( \sum \) symbol means "add up all the squared values from 1 to *n*."

**Step 1: Pseudocode**
```
RMSNorm(x, gain):
    rms = sqrt(mean(x^2, dim=-1))
    return (x / rms) * gain
```

**Step 2: Implementation**

```python title="rmsnorm.py — Implementing RMSNorm from the LLaMA paper"
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Used in LLaMA instead of standard LayerNorm.
    Key difference: no mean centering, no bias — just scale normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


# ---- Verify shapes and behavior ----
B, T, D = 2, 10, 64
x = torch.randn(B, T, D)

rmsnorm = RMSNorm(D)
out = rmsnorm(x)
print(f"Input shape:  {x.shape}")   # [2, 10, 64]
print(f"Output shape: {out.shape}") # [2, 10, 64]

# Compare: RMSNorm should NOT center the mean
print(f"Input mean:   {x.mean(dim=-1)[0, :3]}")
print(f"Output mean:  {out.mean(dim=-1)[0, :3]}")  # non-zero (unlike LayerNorm)

# Compare against PyTorch LayerNorm
ln = nn.LayerNorm(D)
ln_out = ln(x)
print(f"LayerNorm output mean: {ln_out.mean(dim=-1)[0, :3]}")  # ~0
```

:::tip[Line-by-Line Walkthrough]
- **`self.eps = eps`** — A tiny number (0.000001) added to avoid dividing by zero.
- **`self.weight = nn.Parameter(torch.ones(dim))`** — Creates the learnable gain parameter \( g_i \), initialized to all ones (so initially it doesn't change the normalized values).
- **`rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)`** — Squares every element, averages across the last dimension (the feature dimension), takes the square root, and adds `eps` for numerical safety. This is the RMS value.
- **`x_norm = x / rms`** — Divides each element by the RMS, bringing everything to a similar scale.
- **`return x_norm * self.weight`** — Multiplies by the learnable gain so the model can adjust the scale of each feature.
- **`out.mean(dim=-1)[0, :3]`** — Shows that RMSNorm does NOT center the mean to zero (unlike standard LayerNorm).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `rmsnorm.py`
2. Run: `python rmsnorm.py`

**Expected output:**
```
Input shape:  torch.Size([2, 10, 64])
Output shape: torch.Size([2, 10, 64])
Input mean:   tensor([...])
Output mean:  tensor([...])
LayerNorm output mean: tensor([...])
```
The output mean for RMSNorm will be non-zero, while LayerNorm's output mean will be approximately zero.

</details>

**Step 3: Compare against the official LLaMA implementation**

```python title="Comparing against the official Meta implementation"
import torch
import torch.nn as nn
import torch.nn.functional as F

# Meta's implementation from the LLaMA repo (simplified)
class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

# ---- Compare ----
D = 128
ours = RMSNorm(D)
theirs = LlamaRMSNorm(D)

# Copy weights
theirs.weight.data = ours.weight.data.clone()

x = torch.randn(4, 32, D)
with torch.no_grad():
    our_out = ours(x)
    their_out = theirs(x)

print(f"Max diff: {(our_out - their_out).abs().max().item():.2e}")
# Should be ~1e-7 or less (floating point precision)
```

:::tip[Line-by-Line Walkthrough]
- **`class LlamaRMSNorm`** — Meta's official version of RMSNorm from the LLaMA repository.
- **`torch.rsqrt(...)`** — The reciprocal square root (1 / √x) — mathematically equivalent to our `x / sqrt(...)` but slightly faster on GPUs because it's a single operation.
- **`theirs.weight.data = ours.weight.data.clone()`** — Copies our learnable weights into their module so both have identical parameters.
- **`(our_out - their_out).abs().max()`** — Finds the single largest difference between our outputs and theirs. It should be tiny (around 0.0000001), showing our implementation is correct.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save this code along with the `RMSNorm` class from above into a single file, e.g. `compare_rmsnorm.py`
2. Run: `python compare_rmsnorm.py`

**Expected output:**
```
Max diff: 1.19e-07
```
(The exact number will vary, but it should be around 1e-7 or smaller — just floating-point rounding differences.)

</details>

:::info[Why RMSNorm Instead of LayerNorm?]
RMSNorm skips the mean-centering step of LayerNorm. This makes it faster (fewer operations) and empirically works just as well for large language models. The LLaMA paper showed this saves about 10% of training time with no loss in quality.
:::

## Common Paper-Reading Pitfalls

1. **Missing hyperparameters** — Check the appendix and any referenced code repositories. Key settings like learning rate schedules are often not in the main text.

2. **Ambiguous descriptions** — "We use a standard Transformer encoder" could mean many things. Check the code or supplementary material.

3. **Cherry-picked results** — Look at all tables, not just the highlighted ones. Check if standard deviations are reported.

4. **Reproducibility gaps** — Some results require compute that's impractical to reproduce. Focus on the *technique*, not the exact numbers.

5. **Notation overload** — The same paper might use \( h \) for both "hidden dimension" and "number of heads" in different sections. Build a notation glossary as you read.

## Building a Paper-Reading Habit

- **Read one paper per week** consistently — it compounds.
- **Keep a log** — title, date read, key contribution, relevance to your work.
- **Implement something small** from every paper you read deeply (even just one function).
- **Discuss with others** — join a reading group or post summaries online.

:::tip[Exercise 1: Three-Pass Reading — beginner]

Apply the three-pass method to the **"Attention Is All You Need"** paper (Vaswani et al., 2017).

1. **Pass 1** (10 min): Write down the paper's main contribution in one sentence.
2. **Pass 2** (1 hour): Draw the architecture diagram from memory. List 3 things you don't fully understand.
3. **Pass 3** (optional): Pick one component (e.g., positional encoding) and implement it from scratch.

<details>
<summary>Hints</summary>

1. Start with the abstract and conclusion
2. Time yourself: don't spend more than 10 minutes on Pass 1
3. For Pass 2, focus on Figures 1-3 and Table 1

</details>

:::

:::tip[Exercise 2: Implement SwiGLU — intermediate]

torch.Tensor:
        # x: [B, T, D]
        return self.w2(F.silu(self.w1(x)) * self.v(x))

# Test
B, T, D, D_FF = 2, 16, 512, 1024
x = torch.randn(B, T, D)
swiglu = SwiGLU(D, D_FF)
out = swiglu(x)
print(f"SwiGLU output shape: {out.shape}")  # [2, 16, 512]

# Compare parameter count vs standard FFN
standard_ffn = nn.Sequential(
    nn.Linear(D, D_FF),
    nn.ReLU(),
    nn.Linear(D_FF, D),
)
print(f"Standard FFN params: {sum(p.numel() for p in standard_ffn.parameters()):,}")
print(f"SwiGLU params:       {sum(p.numel() for p in swiglu.parameters()):,}")
```
}>

Read the SwiGLU section of the **LLaMA paper** (Touvron et al., 2023) or the original **"GLU Variants Improve Transformer"** paper (Shazeer, 2020).

1. Implement the SwiGLU activation function as a PyTorch module.
2. Verify the output shapes are correct.
3. Compare the parameter count to a standard ReLU feed-forward network with the same hidden dimension.

<details>
<summary>Hints</summary>

1. SwiGLU(x) = (xW₁ ⊙ Swish(xV)) W₂
2. Swish(x) = x · σ(βx), where β is often set to 1
3. The gating mechanism means the FFN has 3 weight matrices instead of 2

</details>

:::

:::tip[Exercise 3: Paper Reproduction Log — advanced]

Choose a paper published in the last 12 months. Maintain a **reproduction log** as you implement it:

1. Create a markdown document tracking: date, what you attempted, what worked, what didn't, and what you learned.
2. Implement the paper's core algorithm from scratch (don't copy the official repo).
3. Run the smallest experiment from the paper and compare your results.
4. Write a 500-word summary of what you learned and what the paper leaves out.

<details>
<summary>Hints</summary>

1. Pick a paper with available code for verification
2. Start with the smallest experiment in the paper
3. Track every decision and deviation in your log

</details>

:::

## Resources

- **[How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)** _(paper)_ by S. Keshav — The classic three-pass method for reading research papers efficiently.

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** _(paper)_ by Vaswani et al. — The foundational Transformer paper — ideal for practicing paper-reading skills.

- **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)** _(paper)_ by Touvron et al. — The LLaMA paper, featuring RMSNorm, SwiGLU, and RoPE — great for implementation practice.

- **[Papers With Code](https://paperswithcode.com)** _(tool)_ — Links papers to their official code implementations — invaluable for verification.

- **[Yannic Kilcher's YouTube Channel](https://www.youtube.com/@YannicKilcher)** _(video)_ by Yannic Kilcher — Detailed video walkthroughs of ML papers, great for building reading intuition.

- **[Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** _(tutorial)_ by Harvard NLP — Line-by-line annotated implementation of 'Attention Is All You Need'.
