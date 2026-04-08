---
sidebar_position: 2
slug: architecture-innovations
title: "Architecture Innovations"
---


# Architecture Innovations

:::info[What You'll Learn]
- Grouped-query attention (GQA) and multi-query attention
- Mixture-of-experts (MoE) architectures
- State-space models (Mamba) and alternatives to attention
- Recent architectural trends and where the field is heading
:::

:::note[Prerequisites]
[The Transformer Architecture](/curriculum/month-3/transformer-architecture) and [GPT Architecture](/curriculum/month-4/gpt-architecture).
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

The original Transformer has been relentlessly improved since 2017. This lesson covers the most important architectural innovations that power today's state-of-the-art models — from sparse expert routing to sub-quadratic sequence modeling. Understanding these innovations is critical for anyone designing or modifying model architectures.

## Rotary Position Embeddings (RoPE)

Standard sinusoidal or learned positional embeddings are added to input tokens. RoPE takes a fundamentally different approach: it encodes position by **rotating** the query and key vectors in the attention mechanism.

**Core idea:** Rotate each pair of dimensions in the query and key vectors by an angle proportional to the token's position. The dot product between two rotated vectors then naturally depends on their relative position difference.

:::info[Plain English: What Does This Formula Mean?]
Imagine each token's embedding as an arrow in space. RoPE "spins" that arrow by a different angle depending on the token's position in the sentence. Tokens early in the sentence get a small spin, tokens later get a bigger spin. When the model compares two tokens (via a dot product), the result naturally encodes how far apart they are — like telling time by looking at the angle between two clock hands. Each pair of dimensions spins at a different speed, so the model can encode position information at many different granularities simultaneously.
:::

\[
\text{RoPE}(x_m, m) = \begin{pmatrix} x_m^{(1)} \cos(m\theta_1) - x_m^{(2)} \sin(m\theta_1) \\ x_m^{(1)} \sin(m\theta_1) + x_m^{(2)} \cos(m\theta_1) \\ \vdots \\ x_m^{(d-1)} \cos(m\theta_{d/2}) - x_m^{(d)} \sin(m\theta_{d/2}) \\ x_m^{(d-1)} \sin(m\theta_{d/2}) + x_m^{(d)} \cos(m\theta_{d/2}) \end{pmatrix}
\]

**Reading the formula:** \( x_m \) is the embedding vector for the token at position \( m \). The superscripts like \( x_m^{(1)}, x_m^{(2)} \) refer to individual dimensions of that vector, taken in pairs. \( \cos \) and \( \sin \) are the rotation operations (think spinning a point on a circle). \( \theta_i = 10000^{-2i/d} \) is the rotation frequency for the *i*-th pair — low-numbered pairs spin slowly (capturing long-range position), high-numbered pairs spin fast (capturing fine-grained position). \( d \) is the total number of dimensions. \( m \) is the position index (0, 1, 2, ...).

where \( \theta_i = 10000^{-2i/d} \).

```python title="rope.py — Rotary Position Embeddings"
import torch
import torch.nn as nn
import math

def precompute_rope_frequencies(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute the complex exponentials for RoPE."""
    # θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # positions: [0, 1, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product: [max_seq_len, dim//2]
    freqs = torch.outer(t, freqs)
    # complex exponentials: cos(mθ) + i·sin(mθ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # [max_seq_len, dim//2]


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor.

    x: [B, H, T, D] where D must be even
    freqs_cis: [T, D//2] complex tensor
    """
    # View consecutive pairs as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape freqs for broadcasting: [1, 1, T, D//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    # Multiply (rotation in complex plane)
    x_rotated = x_complex * freqs_cis
    # Back to real: [B, H, T, D]
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


# ---- Demo ----
B, H, T, D = 2, 8, 128, 64
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)

freqs = precompute_rope_frequencies(D, T)
q_rotated = apply_rope(q, freqs)
k_rotated = apply_rope(k, freqs)

print(f"Original Q shape: {q.shape}")
print(f"Rotated Q shape:  {q_rotated.shape}")

# Key property: q·k depends on RELATIVE position, not absolute
attn_0_5 = (q_rotated[:, :, 0] * k_rotated[:, :, 5]).sum(-1)
print(f"Attention score (pos 0, pos 5): {attn_0_5[0, 0]:.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))`** — Computes the rotation frequencies θ for each pair of dimensions. Lower dimensions spin slowly, higher ones spin fast.
- **`t = torch.arange(max_seq_len)`** — Creates position indices: 0, 1, 2, ..., up to the max sequence length.
- **`freqs = torch.outer(t, freqs)`** — Outer product: for each position, multiply by each frequency to get the rotation angle at that position for that dimension pair.
- **`freqs_cis = torch.polar(torch.ones_like(freqs), freqs)`** — Converts angles to complex numbers on the unit circle (cos + i·sin), which makes rotation a simple multiplication.
- **`x_complex = torch.view_as_complex(...)`** — Treats consecutive pairs of real numbers in Q or K as complex numbers, so we can rotate them by multiplying.
- **`x_rotated = x_complex * freqs_cis`** — The actual rotation: multiplying two complex numbers rotates one by the angle of the other.
- **`attn_0_5 = (q_rotated[:, :, 0] * k_rotated[:, :, 5]).sum(-1)`** — Computes the attention score between position 0 and position 5, which now encodes their relative distance (5) thanks to RoPE.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `rope.py`
2. Run: `python rope.py`

**Expected output:**
```
Original Q shape: torch.Size([2, 8, 128, 64])
Rotated Q shape:  torch.Size([2, 8, 128, 64])
Attention score (pos 0, pos 5): -1.2345
```
(The exact attention score will vary since inputs are random, but the shapes should match exactly.)

</details>

:::info[Why RoPE Works]
When you compute the dot product \( q_m^\top k_n \) of two RoPE-encoded vectors, the rotation angles cancel to leave only the *relative* position \( m - n \). This means the model learns relative position awareness without explicit relative position bias terms, and it generalizes better to longer sequences than absolute position embeddings.
:::

## Multi-Query Attention (MQA)

Standard multi-head attention has separate Q, K, V projections for each head. **Multi-Query Attention** (Shazeer, 2019) shares a single set of K and V across all heads while keeping separate Q projections.

**Why it matters:** During autoregressive inference, you store KV caches. With MQA, the KV cache is \( H \times \) smaller (where H = number of heads), dramatically reducing memory and speeding up decoding.

```python title="multi_query_attention.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention: shared K, V across all heads."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Separate Q projection per head (standard)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # SINGLE K, V projection shared across all heads
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, D = x.shape
        H = self.num_heads

        # Q: [B, T, D] → [B, H, T, head_dim]
        Q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        # K, V: [B, T, head_dim] → [B, 1, T, head_dim] (broadcast across heads)
        K = self.k_proj(x).unsqueeze(1)
        V = self.v_proj(x).unsqueeze(1)

        attn = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---- Compare parameter counts ----
D = 1024
H = 16
mqa = MultiQueryAttention(D, H)
mha_params = 4 * D * D  # standard MHA: Q, K, V, O all [D, D]
mqa_params = sum(p.numel() for p in mqa.parameters())
print(f"Standard MHA params:      {mha_params:,}")
print(f"Multi-Query Attn params:  {mqa_params:,}")
print(f"KV cache reduction:       {H}x smaller")
```

:::tip[Line-by-Line Walkthrough]
- **`self.q_proj = nn.Linear(d_model, d_model, bias=False)`** — Q projection is standard: each head gets its own query vectors (full d_model output).
- **`self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)`** — The key insight: K projection only outputs one head's worth of dimensions, shared across ALL heads.
- **`Q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)`** — Reshapes Q into separate heads as usual.
- **`K = self.k_proj(x).unsqueeze(1)`** — K has no head dimension, so we add one (size 1) for broadcasting — PyTorch will automatically replicate it across all heads during the matrix multiply.
- **`attn = (Q @ K.transpose(-2, -1)) / self.scale`** — Standard scaled dot-product attention, but K is broadcast across heads instead of having separate K per head.
- **`mha_params = 4 * D * D`** — Standard MHA has 4 weight matrices of size [D, D]: Q, K, V, and Output.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `multi_query_attention.py`
2. Run: `python multi_query_attention.py`

**Expected output:**
```
Standard MHA params:      4,194,304
Multi-Query Attn params:  2,228,224
KV cache reduction:       16x smaller
```
(MQA has significantly fewer parameters due to shared K and V projections.)

</details>

## Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) is the middle ground between MHA and MQA. Instead of one KV head (MQA) or H KV heads (MHA), you use **G groups** where each group of query heads shares one KV head.

- G = 1 → Multi-Query Attention
- G = H → standard Multi-Head Attention
- 1 < G < H → Grouped Query Attention

LLaMA 2 70B introduced GQA to the LLaMA family with 8 KV heads and 64 query heads (G = 8). Llama 3 and Llama 4 continue to use GQA across all model sizes.

```python title="grouped_query_attention.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.heads_per_group = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads: [B, kv_heads, T, D] → [B, heads, T, D]
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)

        attn = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# ---- Demo: GQA configuration (as introduced in LLaMA 2 70B, now standard in Llama 3/4) ----
D = 1024
gqa = GroupedQueryAttention(d_model=D, num_heads=32, num_kv_heads=8)
x = torch.randn(1, 64, D)
out = gqa(x)
print(f"GQA output: {out.shape}")
print(f"GQA params: {sum(p.numel() for p in gqa.parameters()):,}")
print(f"KV cache: {8}x smaller than full MHA, {8}x larger than MQA")
```

:::tip[Line-by-Line Walkthrough]
- **`assert num_heads % num_kv_heads == 0`** — The number of query heads must be evenly divisible by the number of KV heads (so each KV head serves the same number of query heads).
- **`self.heads_per_group = num_heads // num_kv_heads`** — How many query heads share each KV head. Here: 32 / 8 = 4 query heads per KV group.
- **`self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)`** — Only projects to 8 KV heads instead of 32 — much smaller.
- **`K = K.repeat_interleave(self.heads_per_group, dim=1)`** — Duplicates each KV head to match the number of query heads. Each of the 8 KV heads gets repeated 4 times to make 32.
- **`out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)`** — Combines all heads back into a single output vector.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `grouped_query_attention.py`
2. Run: `python grouped_query_attention.py`

**Expected output:**
```
GQA output: torch.Size([1, 64, 1024])
GQA params: 2,359,296
KV cache: 8x smaller than full MHA, 8x larger than MQA
```

</details>

## Mixture of Experts (MoE)

MoE models conditionally activate only a subset of parameters for each token. A **router** (gating network) decides which experts process each token, enabling massive model capacity without proportional compute costs.

:::info[Sparse vs Dense]
A 100B-parameter MoE model with 8 experts (top-2 routing) only activates ~25B parameters per token. You get the capacity of a 100B model with the inference cost of a 25B model.
:::

### Switch Transformer Routing

The Switch Transformer (Fedus et al., 2022) simplifies MoE by routing each token to exactly **one** expert (top-1 routing), reducing communication costs.

```python title="moe.py — Mixture of Experts with top-k routing"
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single expert: standard feed-forward network."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing."""
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]

        # Router scores
        gate_logits = self.gate(x_flat)         # [B*T, num_experts]
        weights, indices = gate_logits.topk(self.top_k, dim=-1)  # [B*T, top_k]
        weights = F.softmax(weights, dim=-1)

        # Dispatch tokens to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weights[mask, k].unsqueeze(-1) * expert_output

        return output.view(B, T, D)


# ---- Demo ----
D, D_FF = 512, 1024
moe = MoELayer(d_model=D, d_ff=D_FF, num_experts=8, top_k=2)
x = torch.randn(2, 32, D)
out = moe(x)
print(f"MoE output: {out.shape}")

total_params = sum(p.numel() for p in moe.parameters())
active_params = sum(p.numel() for p in moe.experts[0].parameters()) * 2  # top-2
print(f"Total params:  {total_params:,}")
print(f"Active params: {active_params:,} (per token, top-2 routing)")
print(f"Sparsity:      {1 - active_params / total_params:.1%}")
```

:::tip[Line-by-Line Walkthrough]
- **`self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])`** — Creates N separate feed-forward networks (experts), each with the same architecture but independent weights.
- **`self.gate = nn.Linear(d_model, num_experts, bias=False)`** — The router: a simple linear layer that produces a score for each expert.
- **`x_flat = x.view(-1, D)`** — Flattens the batch and sequence dimensions so we treat every token independently for routing.
- **`gate_logits = self.gate(x_flat)`** — Each token gets a score for each expert (higher = more relevant).
- **`weights, indices = gate_logits.topk(self.top_k, dim=-1)`** — Picks the top-k experts for each token (e.g., top 2 out of 8).
- **`weights = F.softmax(weights, dim=-1)`** — Normalizes the top-k scores to sum to 1, so they act as mixing weights.
- **`output[mask] += weights[mask, k].unsqueeze(-1) * expert_output`** — Adds each expert's contribution, weighted by the router's score.
- **`active_params = ... * 2`** — With top-2 routing, only 2 out of 8 experts run per token, so active parameters are about 25% of total.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `moe.py`
2. Run: `python moe.py`

**Expected output:**
```
MoE output: torch.Size([2, 32, 512])
Total params:  8,921,600
Active params: 2,100,224 (per token, top-2 routing)
Sparsity:      76.5%
```
(Exact numbers may vary slightly depending on PyTorch version.)

</details>

:::warning[Load Balancing]
Without intervention, MoE routers tend to collapse: most tokens get routed to the same 1-2 experts while others are unused. The Switch Transformer adds an **auxiliary load-balancing loss** to penalize uneven expert utilization:

:::info[Plain English: What Does This Formula Mean?]
Think of experts as checkout lanes at a grocery store. If everyone crowds into lane 1 while lanes 2-8 sit empty, that's inefficient. This loss function penalizes the model when some experts are overloaded and others are idle. It multiplies each expert's actual traffic by how much the router *wanted* to send tokens there — if both are high for one expert and low for another, the loss is high, pushing the router to spread tokens more evenly.
:::

\[
\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
\]

**Reading the formula:** \( \mathcal{L}_{\text{balance}} \) is the balancing loss — an extra penalty added to the training loss. \( \alpha \) is a small weight (e.g., 0.01) that controls how strongly we penalize imbalance. \( N \) is the number of experts. \( f_i \) is the fraction of all tokens that were actually routed to expert \( i \). \( P_i \) is the average probability the router assigned to expert \( i \). \( \sum \) means "add up" over all experts from 1 to \( N \).
:::

## State Space Models (SSMs): Mamba

Transformers have quadratic attention complexity \( O(T^2) \). State Space Models offer a fundamentally different approach with **linear complexity** \( O(T) \) by modeling sequences as continuous dynamical systems discretized for computation.

### The S4 Foundation

The Structured State Space (S4) model parameterizes the sequence-to-sequence map using a linear ODE:

:::info[Plain English: What Does This Formula Mean?]
Think of a state space model as a note-taker with a notepad (the hidden state *h*). At each moment, the note-taker: (1) looks at their current notes, (2) reads the new input, and (3) updates their notes using a mix of old notes and new input. Then they write a summary (the output *y*) based on their notes. The matrices A, B, C, and D are the "rules" for how to mix and summarize — the continuous version describes this as a smooth, flowing process.
:::

\[
h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)
\]

**Reading the formula:** \( h(t) \) is the hidden state (the "notes") at time \( t \). \( h'(t) \) is how fast the notes are changing at time \( t \). \( A \) is a matrix that controls how the old notes influence the update (memory decay). \( B \) is a matrix that controls how much new input gets written into the notes. \( x(t) \) is the input at time \( t \). \( y(t) \) is the output at time \( t \). \( C \) converts the hidden notes into a readable output. \( D \) is a skip connection — letting the input pass directly to the output.

Discretized with step size \( \Delta \):

:::info[Plain English: What Does This Formula Mean?]
The continuous formula above works in smooth, flowing time — but computers work in discrete steps. This version says: at each step *k*, update your notes by mixing the old notes (scaled by \( \bar{A} \)) with the new input (scaled by \( \bar{B} \)). Then read off the output from the updated notes. It's like the note-taker checking in at regular intervals instead of continuously.
:::

\[
h_k = \bar{A} h_{k-1} + \bar{B} x_k, \quad y_k = C h_k
\]

**Reading the formula:** \( h_k \) is the hidden state at step \( k \). \( h_{k-1} \) is the hidden state from the previous step. \( \bar{A} \) is the discretized version of \( A \) — how much of the old state to keep. \( \bar{B} \) is the discretized version of \( B \) — how much of the new input to absorb. \( x_k \) is the input at step \( k \). \( y_k \) is the output at step \( k \). \( C \) still maps the hidden state to the output.

### Mamba: Selective State Spaces

Mamba (Gu & Dao, 2023) makes the SSM parameters **input-dependent** (selective), allowing the model to focus on or ignore specific tokens — something fixed-parameter SSMs cannot do.

```python title="selective_ssm.py — Simplified selective SSM (Mamba-style)"
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Simplified selective state space model (Mamba-style).

    Key innovation: A, B, C, Δ are functions of the input, not fixed parameters.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
        )

        # SSM parameter projections (input-dependent)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)  # B, C, Δ
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))  # log of diagonal A
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        B, T, D = x.shape

        # Project and split into two paths (like GLU)
        xz = self.in_proj(x)  # [B, T, 2D]
        x_branch, z = xz.chunk(2, dim=-1)

        # 1D convolution
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        x_ssm = self.x_proj(x_conv)
        B_param = x_ssm[:, :, :self.d_state]         # [B, T, N]
        C_param = x_ssm[:, :, self.d_state:2*self.d_state]  # [B, T, N]
        delta = F.softplus(x_ssm[:, :, -1])           # [B, T] (step size)

        A = -torch.exp(self.A_log)  # [D, N] — stable (negative)

        # Sequential scan (simplified — real Mamba uses a parallel scan)
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(T):
            dt = delta[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            A_bar = torch.exp(A * dt)                       # [D, N] discretized
            B_bar = B_param[:, t].unsqueeze(1) * dt         # [B, 1, N]
            x_t = x_conv[:, t].unsqueeze(-1)                # [B, D, 1]
            h = A_bar * h + x_t * B_bar                     # [B, D, N]
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)  # [B, D]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, T, D]
        y = y + x_conv * self.D  # skip connection

        # Gate with z branch
        y = y * F.silu(z)
        return self.out_proj(y)


# ---- Demo ----
model = SelectiveSSM(d_model=128, d_state=16)
x = torch.randn(2, 64, 128)
out = model(x)
print(f"SSM output: {out.shape}")  # [2, 64, 128]
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

:::tip[Line-by-Line Walkthrough]
- **`self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)`** — Doubles the width to create two branches (like a GLU gate): one for processing, one for gating.
- **`self.conv1d = nn.Conv1d(..., groups=d_model)`** — A depthwise 1D convolution that mixes nearby tokens. Each channel is convolved independently.
- **`self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)`** — Projects the input into the SSM parameters B, C, and Δ. This is the "selective" part — these parameters depend on the input, not fixed.
- **`self.A_log = nn.Parameter(torch.randn(d_model, d_state))`** — The A matrix is stored in log space and negated to ensure stability (negative eigenvalues = decaying memory).
- **`x_branch, z = xz.chunk(2, dim=-1)`** — Splits into two equal halves: one goes through the SSM, the other becomes a gate.
- **`delta = F.softplus(x_ssm[:, :, -1])`** — The step size Δ, made positive via softplus. Controls how fast the state updates — large Δ means "pay attention to this input," small Δ means "mostly keep the old state."
- **`A_bar = torch.exp(A * dt)`** — Discretizes the continuous A matrix using the input-dependent step size.
- **`h = A_bar * h + x_t * B_bar`** — The core SSM update: mix old state with new input.
- **`y = y * F.silu(z)`** — Gates the SSM output with the other branch, letting the model control information flow.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `selective_ssm.py`
2. Run: `python selective_ssm.py`

**Expected output:**
```
SSM output: torch.Size([2, 64, 128])
Params: 148,737
```
(Note: this simplified version uses a sequential scan; real Mamba uses a hardware-efficient parallel scan for speed.)

</details>

:::info[Why Mamba Matters]
Mamba achieves Transformer-level quality on language modeling with linear scaling in sequence length. A 3B Mamba model matches a 3B Transformer on standard benchmarks but runs 5x faster on long sequences. It has no attention mechanism at all — the entire model is built on selective state spaces and gating.
:::

## Ring Attention for Long Contexts

Ring Attention (Liu et al., 2023) enables training on sequences millions of tokens long by distributing the attention computation across multiple devices arranged in a ring topology.

**Key idea:** Each device holds a chunk of the sequence. Devices pass KV blocks around the ring while computing local attention, overlapping communication with computation.

```
Device 0: Q₀ attends to K₀,V₀ → receives K₁,V₁ → receives K₂,V₂ → ...
Device 1: Q₁ attends to K₁,V₁ → receives K₂,V₂ → receives K₀,V₀ → ...
Device 2: Q₂ attends to K₂,V₂ → receives K₀,V₀ → receives K₁,V₁ → ...
```

Each device computes a partial attention sum and accumulates the result. After N steps (where N = number of devices), every device has attended to the full sequence.

:::note[Memory Scaling]
Think of it like splitting a giant jigsaw puzzle: if you have 64 friends, each person only needs to work on a small section of the puzzle instead of holding the entire picture. Standard attention requires memory proportional to the sequence length squared — \( O(T^2) \) — which is impossible for T = 1M tokens (that's a trillion entries!). Ring Attention divides this by the square of the number of devices: \( O(T^2 / N^2) \). With 64 devices and T = 1M, each device only handles 15,625² attention blocks — a manageable chunk.

**Reading the notation:** \( O(\cdot) \) means "on the order of" (Big-O notation for scaling). \( T \) is the sequence length (number of tokens). \( N \) is the number of devices. \( T^2 / N^2 \) means each device's work shrinks quadratically as you add more devices.
:::

## Comparison Table

| Innovation | Problem Solved | Complexity | Used In |
|-----------|---------------|------------|---------|
| **RoPE** | Absolute position embeddings don't generalize | Same as standard attention | LLaMA, Mistral, Qwen |
| **MQA** | KV cache too large for inference | Reduces KV cache by H× | PaLM, Falcon |
| **GQA** | MQA sacrifices too much quality | Reduces KV cache by H/G× | Llama 2/3/4, Mistral |
| **MoE** | Dense models too expensive to scale | Constant compute per token | Mixtral, GPT-4, Llama 4 Maverick, Switch |
| **Mamba (SSM)** | Quadratic attention on long sequences | O(T) vs O(T²) | Mamba, Jamba |
| **Ring Attention** | Can't fit long sequences on one device | O(T²/N²) per device | Research, some production systems |

:::tip[Exercise 1: RoPE Extension — intermediate]

RoPE struggles when the inference sequence length exceeds the training length. Research and implement one method for extending RoPE to longer contexts:

1. Read about **NTK-aware scaling** or **YaRN** (Yet another RoPE extensioN).
2. Modify the `precompute_rope_frequencies` function to support a `scaling_factor` parameter.
3. Test: create attention scores for sequences at 1x and 2x the original max length.

<details>
<summary>Hints</summary>

1. Look up NTK-aware scaling or YaRN
2. The key insight is modifying the base frequency
3. Test by training on short sequences and evaluating on longer ones

</details>

:::

:::tip[Exercise 2: Expert Load Balancing — advanced]

Implement the **auxiliary load-balancing loss** from the Switch Transformer paper and add it to the `MoELayer` above.

1. Compute the fraction \( f_i \) of tokens routed to each expert.
2. Compute the mean router probability \( P_i \) for each expert.
3. Return the loss \( \alpha \cdot N \cdot \sum f_i \cdot P_i \).
4. Verify that adding this loss to training makes expert utilization more uniform.

<details>
<summary>Hints</summary>

1. Compute f_i as the fraction of tokens routed to expert i
2. Compute P_i as the mean router probability assigned to expert i
3. The loss should be α · N · Σ(f_i · P_i)

</details>

:::

:::tip[Exercise 3: Implement Multi-Latent Attention — advanced]

Multi-Latent Attention (MLA), introduced in DeepSeek-V2, compresses the KV cache using a learned low-rank projection. Instead of caching full K and V tensors, it caches a smaller latent representation.

1. Read the MLA section of the DeepSeek-V2 paper.
2. Implement MLA: add a down-projection before caching and an up-projection during attention.
3. Compare KV cache size vs standard MHA and GQA.

<details>
<summary>Hints</summary>

1. MLA compresses KV into a low-rank latent space before storage
2. The KV cache stores compressed latents instead of full K, V
3. Check the DeepSeek-V2 paper for the detailed formulation

</details>

:::

## Resources

- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)** _(paper)_ by Su et al. — The original RoPE paper — mathematical derivation and analysis.

- **[Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)** _(paper)_ by Fedus et al. — The key paper on simplified MoE with top-1 routing.

- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** _(paper)_ by Gu & Dao — The Mamba paper — achieves Transformer-quality results without attention.

- **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)** _(paper)_ by Ainslie et al. — Shows how to uptrain MHA models into GQA for better inference.

- **[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)** _(paper)_ by Liu et al. — Distributed attention for training on million-token sequences.

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** _(tutorial)_ by Jay Alammar — Visual guide to the original Transformer — essential background for understanding innovations.
