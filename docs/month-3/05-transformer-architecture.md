---
sidebar_position: 5
slug: transformer-architecture
title: "The Transformer Architecture"
---


# The Transformer Architecture

:::info[What You'll Learn]
- Self-attention and why it replaces recurrence
- Scaled dot-product attention and multi-head attention
- Positional encoding for sequence order
- The full encoder-decoder Transformer architecture
- Residual connections and layer normalization
:::

:::note[Prerequisites]
[Attention Mechanism](attention-mechanism) from this month and [Math Foundations](/curriculum/month-1/math-foundations) from Month 1.
:::

**Estimated time:** Reading: ~50 min | Exercises: ~3 hours

The Transformer, introduced in the 2017 paper **"Attention Is All You Need"** by Vaswani et al., is the architecture behind virtually every modern language model — GPT, BERT, T5, LLaMA, and hundreds more. It replaced recurrence entirely with **self-attention**, enabling massive parallelization and dramatically improving performance on sequence tasks.

This is the most important lesson in Month 3. Take your time with it.

## Why Replace Recurrence?

RNNs process tokens sequentially — token 1, then token 2, then token 3. This creates two problems:

1. **No parallelism.** Each step depends on the previous step's output, so you can't process tokens simultaneously on a GPU.
2. **Long-range dependencies.** Information from early tokens must survive through many sequential steps to reach later tokens. Even LSTMs struggle with sequences beyond a few hundred tokens.

:::info[The Transformer's Key Innovation]
Instead of processing tokens one at a time, the Transformer processes **all tokens simultaneously** using self-attention. Every token can directly attend to every other token in a single operation — no information bottleneck, no sequential dependency.
:::

## Paper Walkthrough: "Attention Is All You Need"

The paper proposes an **encoder-decoder** architecture built entirely from attention layers and feed-forward networks — no recurrence, no convolutions.

```
                    ┌─────────────────────────────┐
                    │     Output Probabilities     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │      Linear + Softmax        │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │           DECODER STACK (N×)             │
              │  ┌────────────────────────────────────┐  │
              │  │  Masked Multi-Head Self-Attention   │  │
              │  │  + Residual & LayerNorm             │  │
              │  ├────────────────────────────────────┤  │
              │  │  Multi-Head Cross-Attention         │  │
              │  │  (Q from decoder, K/V from encoder) │  │
              │  │  + Residual & LayerNorm             │  │
              │  ├────────────────────────────────────┤  │
              │  │  Feed-Forward Network               │  │
              │  │  + Residual & LayerNorm             │  │
              │  └────────────────────────────────────┘  │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │           ENCODER STACK (N×)             │
              │  ┌────────────────────────────────────┐  │
              │  │  Multi-Head Self-Attention           │  │
              │  │  + Residual & LayerNorm             │  │
              │  ├────────────────────────────────────┤  │
              │  │  Feed-Forward Network               │  │
              │  │  + Residual & LayerNorm             │  │
              │  └────────────────────────────────────┘  │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │     Input Embeddings + Positional Enc    │
              └─────────────────────────────────────────┘
```

The original Transformer uses \( N = 6 \) layers in both the encoder and decoder, with model dimension \( d_{\text{model}} = 512 \), 8 attention heads, and feed-forward inner dimension 2048.

## Self-Attention

In the encoder-decoder attention from the previous lesson, queries came from the decoder and keys/values from the encoder. In **self-attention**, queries, keys, and values all come from the **same sequence**. Every token attends to every other token (including itself) in the same layer.

### Scaled Dot-Product Attention

:::note[Scaled Dot-Product Attention]

:::info[Plain English: What Does This Formula Mean?]
Imagine a classroom where every student (token) can pass notes to every other student. Each student writes a question on a note (**query**), and every student also wears a name tag (**key**) and holds a piece of information (**value**). To decide who to listen to, each student compares their question to everyone's name tags (dot product), divides by a calming factor (√d_k) so no one shouts too loudly, then collects a weighted blend of everyone's information — paying most attention to the best-matching name tags.
:::

Given matrices \(\mathbf{Q}\) (queries), \(\mathbf{K}\) (keys), and \(\mathbf{V}\) (values):

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
\]

**Reading the formula:** *Q* is the matrix of queries (what each token is looking for). *K^T* is the transposed key matrix (what each token offers). Multiplying *QK^T* gives a score for every pair of tokens. Dividing by *√d_k* prevents scores from becoming too extreme. *softmax* turns scores into weights (positive numbers summing to 1 per row). Multiplying by *V* produces the final output — a blended summary for each token based on what it attended to.

where:
- \(\mathbf{Q} \in \mathbb{R}^{n \times d_k}\) — query matrix (one row per token)
- \(\mathbf{K} \in \mathbb{R}^{m \times d_k}\) — key matrix
- \(\mathbf{V} \in \mathbb{R}^{m \times d_v}\) — value matrix
- \(\sqrt{d_k}\) — scaling factor to prevent large dot products
:::

#### Why Scale by \(\sqrt{d_k}\)?

When \(d_k\) is large, the dot products \(\mathbf{q} \cdot \mathbf{k}\) tend to have large magnitude (variance grows with dimension). Large values push softmax into regions where gradients are extremely small. Dividing by \(\sqrt{d_k}\) normalizes the variance back to approximately 1.

```python title="Scaled dot-product attention"
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q: (B, num_queries, d_k)
    K: (B, num_keys,    d_k)
    V: (B, num_keys,    d_v)
    mask: (B, num_queries, num_keys) or broadcastable
    Returns: output (B, num_queries, d_v), weights (B, num_queries, num_keys)
    """
    d_k = K.shape[-1]
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)   # (B, Q, K)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)                      # (B, Q, K)
    output = torch.bmm(attn_weights, V)                           # (B, Q, d_v)

    return output, attn_weights

# Example: 4 tokens, each 64-dimensional
B, T, D = 1, 4, 64
x = torch.randn(B, T, D)

output, weights = scaled_dot_product_attention(x, x, x)
print(f"Output shape:  {output.shape}")   # (1, 4, 64)
print(f"Weights shape: {weights.shape}")  # (1, 4, 4)
print(f"Weights sum per query: {weights.sum(dim=-1)}")  # all 1.0
```

:::tip[Line-by-Line Walkthrough]
- **`torch.bmm(Q, K.transpose(1, 2))`** — Computes the dot product between every pair of query and key vectors. For 4 tokens, this produces a 4×4 matrix of scores — every token compared to every other token.
- **`/ math.sqrt(d_k)`** — The scaling factor. Without it, high-dimensional dot products have high variance, pushing softmax to extreme values. Dividing by √64 = 8 keeps things balanced.
- **`scores.masked_fill(mask == 0, float('-inf'))`** — Sets masked positions to negative infinity. After softmax, these become exactly 0 — the model literally cannot attend to masked positions (used for padding and causal masking).
- **`F.softmax(scores, dim=-1)`** — Converts each row of scores to attention weights summing to 1.
- **`torch.bmm(attn_weights, V)`** — Multiplies the attention weights by the value matrix: each token's output is a weighted blend of all value vectors.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `scaled_attention.py`
2. Run: `python scaled_attention.py`

**Expected output:**
```
Output shape:  torch.Size([1, 4, 64])
Weights shape: torch.Size([1, 4, 4])
Weights sum per query: tensor([[1., 1., 1., 1.]])
```

</details>

### Understanding the Attention Matrix

When self-attention processes a 4-token sequence, the attention weight matrix is 4×4. Each row tells you how much each token attends to every other token:

```
            the    cat    sat    down
  the    [ 0.15   0.35   0.25   0.25 ]   ← "the" attends to all tokens
  cat    [ 0.10   0.40   0.30   0.20 ]   ← "cat" focuses on itself and "sat"
  sat    [ 0.05   0.30   0.35   0.30 ]
  down   [ 0.05   0.20   0.25   0.50 ]   ← "down" focuses on itself
```

:::tip[Self-Attention as Soft Lookup]
Think of self-attention as a **differentiable dictionary lookup**. The query says "what am I looking for?", the keys say "what do I contain?", and the values say "here's my information." The softmax over scores determines how much each value contributes — it's a soft, weighted average instead of a hard lookup.
:::

## Multi-Head Attention

A single attention head can only focus on one type of relationship at a time. **Multi-head attention** runs \(h\) attention operations in parallel, each with its own learned projections, and concatenates the results.

:::note[Multi-Head Attention]

:::info[Plain English: What Does This Formula Mean?]
Imagine you have 8 different highlighters, each a different color. One highlighter marks grammar relationships, another marks topic words, another marks nearby words, and so on. Multi-head attention is like highlighting the same text 8 different ways simultaneously, then combining all the highlighted notes into one rich summary. Each "head" notices different patterns in the text.
:::

\[
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O
\]

**Reading the formula:** We run *h* separate attention operations (heads) in parallel. Each head produces its own output. *Concat(...)* glues all heads' outputs side by side into one long vector. *W^O* is a learned matrix that projects this concatenated result back to the model's dimension.

where each head is:

\[
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
\]

**Reading the formula:** Each head first projects Q, K, and V through its own private weight matrices (*W_i^Q*, *W_i^K*, *W_i^V*), giving each head a different "perspective" on the data. Then it runs standard scaled dot-product attention on these projections.

Each head operates on a subspace of dimension \(d_k = d_{\text{model}} / h\). With 512 dimensions and 8 heads, each head processes 64-dimensional queries, keys, and values.
:::

Why multiple heads? Different heads can learn different types of relationships:
- Head 1 might learn **syntactic dependencies** (subject-verb agreement)
- Head 2 might learn **positional relationships** (adjacent words)
- Head 3 might learn **semantic similarity** (synonyms, coreference)

```python title="Multi-head attention module"
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B = query.shape[0]

        # Linear projections: (B, T, d_model) → (B, T, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads: (B, T, d_model) → (B, num_heads, T, d_k)
        Q = Q.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        d_k = self.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, h, T, d_k)

        # Concatenate heads: (B, h, T, d_k) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)
        return output

import torch.nn as nn
import math

# Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
out = mha(x, x, x)
print(f"Input:  {x.shape}")   # (2, 10, 512)
print(f"Output: {out.shape}")  # (2, 10, 512)
```

:::tip[Line-by-Line Walkthrough]
- **`self.d_k = d_model // num_heads`** — Each head gets a slice of the total dimension. With 512 dimensions and 8 heads, each head works with 64 dimensions.
- **`self.W_q`, `self.W_k`, `self.W_v`** — Three separate linear layers that project the input into query, key, and value representations. These are the learned "lenses" through which each head views the data.
- **`.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)`** — Reshapes the projected vectors to split them into multiple heads. The 512-dim vector becomes 8 separate 64-dim vectors, one per head.
- **`torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)`** — Scaled dot-product attention, computed for all heads in parallel using a single batched matrix multiply.
- **`.transpose(1, 2).contiguous().view(B, -1, self.d_model)`** — After attention, concatenates all heads' outputs back into a single 512-dim vector. `.contiguous()` is needed after transpose for the reshape to work.
- **`self.W_o(context)`** — A final linear layer that mixes information across heads, producing the final output.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `multi_head_attention.py`
2. Run: `python multi_head_attention.py`

**Expected output:**
```
Input:  torch.Size([2, 10, 512])
Output: torch.Size([2, 10, 512])
```

</details>

## Positional Encoding

Self-attention is **permutation-invariant** — shuffling the input tokens produces shuffled output, but the attention weights don't change. The model has no inherent sense of word order. Positional encodings inject sequence position information.

### Sinusoidal Positional Encoding

The original Transformer uses sinusoidal functions of different frequencies:

:::note[Sinusoidal Position Encoding]

:::info[Plain English: What Does This Formula Mean?]
Since the Transformer processes all words at once (not one by one like an RNN), it has no built-in sense of word order. Positional encoding is like giving each word a unique "seat number" so the model knows that "the" comes before "cat." Instead of just numbering seats 1, 2, 3, the encoding uses a clever pattern of sine and cosine waves at different frequencies — like a musical chord that sounds different for every seat. This lets the model figure out both exact positions and relative distances between words.
:::

For position \( \text{pos} \) and dimension \( i \):

\[
PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
\]

**Reading the formula:** For even-numbered dimensions (*2i*), the positional encoding value is a sine wave. *pos* is the word's position in the sequence (0, 1, 2, ...). The denominator *10000^(2i/d_model)* creates a different frequency for each dimension — low dimensions oscillate quickly (like treble notes), high dimensions oscillate slowly (like bass notes).

\[
PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
\]

**Reading the formula:** For odd-numbered dimensions (*2i+1*), it's the same but with cosine instead of sine. Together, the sine-cosine pair for each dimension creates a unique "fingerprint" for every position.

Each dimension uses a different frequency, creating a unique "fingerprint" for each position. The sinusoidal form also allows the model to learn relative positions, because \(PE_{\text{pos}+k}\) can be expressed as a linear function of \(PE_{\text{pos}}\).
:::

```python title="Sinusoidal positional encoding"
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Visualize the positional encoding
import matplotlib.pyplot as plt

pe = PositionalEncoding(d_model=128)
dummy = torch.zeros(1, 100, 128)
encoded = pe(dummy)

plt.figure(figsize=(12, 4))
plt.imshow(encoded[0].detach().numpy().T, aspect='auto', cmap='RdBu')
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=150)
plt.show()
```

:::tip[Line-by-Line Walkthrough]
- **`pe = torch.zeros(max_len, d_model)`** — Pre-computes positional encodings for up to 5000 positions. These are computed once and reused forever.
- **`div_term = torch.exp(...)`** — Computes the frequency for each dimension. Low dimensions have high frequency (rapid oscillation), high dimensions have low frequency (slow oscillation). This is mathematically equivalent to 1/10000^(2i/d_model) but computed in log space for numerical stability.
- **`pe[:, 0::2] = torch.sin(...)` / `pe[:, 1::2] = torch.cos(...)`** — Fills even dimensions with sine values and odd dimensions with cosine values.
- **`self.register_buffer('pe', pe)`** — Stores the positional encoding as a non-trainable buffer (it moves to GPU with the model but doesn't have gradients).
- **`x = x + self.pe[:, :x.size(1)]`** — Adds positional information to the word embeddings. After this, each token knows both its meaning and its position.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch matplotlib
```

**Steps:**
1. Save to `positional_encoding.py` (include `import torch, math, torch.nn as nn`)
2. Run: `python positional_encoding.py`

**Expected output:** A colorful heatmap image (`positional_encoding.png`) showing wave patterns — horizontal stripes of alternating red and blue with varying frequencies across dimensions.

</details>

### Learned Positional Embeddings

An alternative (used by BERT and GPT-2) is to learn position embeddings as parameters:

```python title="Learned positional embeddings"
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.position_embedding(positions)
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Embedding(max_len, d_model)`** — Instead of using fixed sine/cosine patterns, this creates a learnable lookup table where each position (0, 1, 2, ..., 511) gets its own trainable vector. The model figures out the best position representations during training.
- **`torch.arange(x.size(1), device=x.device)`** — Generates position indices [0, 1, 2, ..., T-1] for the current sequence length.
- **`x + self.position_embedding(positions)`** — Adds the learned position vectors to the word embeddings, just like the sinusoidal version.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `learned_pos.py`
2. Add: `lpe = LearnedPositionalEncoding(128); print(lpe(torch.randn(2, 10, 128)).shape)`
3. Run: `python learned_pos.py`

**Expected output:**
```
torch.Size([2, 10, 128])
```

</details>

:::info[Sinusoidal vs. Learned Positions]
- **Sinusoidal:** Generalizes to longer sequences than seen during training. No extra parameters.
- **Learned:** Slightly better performance in practice. But can't extrapolate beyond `max_len`.
- The original paper found both perform similarly. Modern models mostly use learned embeddings.
:::

## Feed-Forward Networks

Each Transformer layer contains a **position-wise feed-forward network** — the same two-layer MLP applied independently to each token position:

:::info[Plain English: What Does This Formula Mean?]
After tokens have "talked" to each other through attention, each token needs time to "think" on its own. The feed-forward network is like a private processing step: each token passes through two layers — first expanding to a wider space (like spreading out your notes on a big desk), applying a filter (ReLU keeps only positive values), then compressing back down. Every token goes through the exact same network independently.
:::

\[
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
\]

**Reading the formula:** *x* is a single token's representation. *W_1* and *b_1* are the weights and bias of the first linear layer, which expands the dimension (e.g., 512 → 2048). *ReLU* sets any negative values to zero (a simple nonlinearity). *W_2* and *b_2* are the second layer, which compresses back down (2048 → 512). The result is the same size as the input.

The inner dimension is typically 4× the model dimension (e.g., 2048 for \(d_{\text{model}} = 512\)).

```python title="Position-wise feed-forward network"
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# The FFN expands and contracts:  512 → 2048 → 512
ffn = FeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 10, 512)
print(f"FFN output: {ffn(x).shape}")  # (2, 10, 512)
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Linear(d_model, d_ff)`** — First layer expands 512 dimensions to 2048. This wider space gives the network more room to learn complex transformations.
- **`F.relu(self.linear1(x))`** — Applies ReLU (Rectified Linear Unit): keeps positive values, sets negative values to 0. This nonlinearity is what allows the network to learn complex patterns.
- **`self.linear2(self.dropout(...))`** — Applies dropout for regularization, then compresses back from 2048 to 512 dimensions.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `ffn.py` (include imports for `torch`, `torch.nn`, `torch.nn.functional as F`)
2. Run: `python ffn.py`

**Expected output:**
```
FFN output: torch.Size([2, 10, 512])
```

</details>

:::info[Why Feed-Forward After Attention?]
Self-attention lets tokens *exchange* information. The FFN lets each token *process* that gathered information independently. Think of attention as "communication" and FFN as "computation." Recent research suggests the FFN layers store factual knowledge, while attention layers handle relational reasoning.
:::

## Layer Normalization and Residual Connections

Every sub-layer (attention or FFN) is wrapped with a **residual connection** and **layer normalization**:

:::info[Plain English: What Does This Formula Mean?]
The residual connection is like a safety net: the output of each sub-layer is *added* to its input (`x + Sublayer(x)`). This way, even if the sub-layer learns nothing useful, the original information passes through unchanged. Layer normalization then adjusts the numbers so they're nicely centered around zero with a consistent spread — like adjusting the brightness and contrast of a photo.
:::

\[
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
\]

**Reading the formula:** *x* is the input. *Sublayer(x)* is the output from attention or FFN. Adding them together is the "residual connection" — a shortcut that preserves the original signal. *LayerNorm* then normalizes the result to stabilize training.

:::note[Layer Normalization]

:::info[Plain English: What Does This Formula Mean?]
Imagine you have a row of numbers that are all over the place — some very big, some tiny. Layer normalization subtracts the average (centering them around zero) and divides by the spread (making them roughly between -1 and 1). Then learned parameters *γ* and *β* let the model adjust the scale and shift to whatever works best. It's like standardizing test scores so they're all on the same scale.
:::

For an input vector \(\mathbf{x} \in \mathbb{R}^d\):

\[
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

**Reading the formula:** *x* is the input vector. *μ* is the mean (average) of all values in *x*. *σ²* is the variance (how spread out the values are). Subtracting *μ* centers the values; dividing by *√(σ² + ε)* makes the spread consistent. *ε* is a tiny number (like 0.00001) to prevent division by zero. *γ* (scale) and *β* (shift) are learned parameters that let the model adjust the normalized output.

where \(\mu\) and \(\sigma^2\) are the mean and variance computed across the feature dimension, and \(\gamma, \beta\) are learned scale and shift parameters.
:::

**Residual connections** solve the vanishing gradient problem by providing a direct gradient path through the network. **Layer normalization** stabilizes training by normalizing activations.

```python title="Residual connection with layer norm"
class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # Pre-norm variant (used in most modern implementations)
        return x + self.dropout(sublayer(self.norm(x)))
```

:::tip[Line-by-Line Walkthrough]
- **`self.norm = nn.LayerNorm(d_model)`** — Creates a layer normalization module that will normalize across the feature dimension.
- **`return x + self.dropout(sublayer(self.norm(x)))`** — This single line implements the full pattern: (1) normalize the input, (2) pass through the sub-layer (attention or FFN), (3) apply dropout, (4) add back the original input (residual connection). The `x +` part ensures information can flow through unchanged if needed.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `sublayer.py`
2. Add: `sl = SublayerConnection(512); print(sl(torch.randn(2, 10, 512), lambda x: x).shape)`
3. Run: `python sublayer.py`

**Expected output:**
```
torch.Size([2, 10, 512])
```

</details>

:::tip[Pre-Norm vs. Post-Norm]
The original paper uses **post-norm**: `LayerNorm(x + Sublayer(x))`. Most modern implementations use **pre-norm**: `x + Sublayer(LayerNorm(x))`. Pre-norm is more stable during training (especially for deep models) and doesn't require a learning rate warmup.
:::

## Encoder Stack

Each encoder layer has two sub-layers:
1. Multi-head self-attention
2. Position-wise feed-forward network

Both wrapped with residual connections and layer norm.

```python title="Transformer encoder layer"
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attention(self.norm1(x), self.norm1(x),
                                        self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

:::tip[Line-by-Line Walkthrough]
- **`EncoderLayer`** — One Transformer encoder layer with two sub-blocks: self-attention followed by feed-forward, each with layer norm and residual connections.
- **`self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)`** — In self-attention, Q, K, and V all come from the same input (after normalization). Every token attends to every other token.
- **`x = x + self.dropout1(attn_out)`** — Residual connection: adds the attention output back to the input. Information can flow through unchanged.
- **`nn.ModuleList([EncoderLayer(...) for _ in range(num_layers)])`** — Creates N identical encoder layers stacked on top of each other. The original Transformer uses N=6.
- **`self.embedding(src) * math.sqrt(self.d_model)`** — Scales the embeddings up. Without this, the embedding values are too small compared to positional encodings, and position information drowns out word meaning.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save all components (MultiHeadAttention, FeedForward, PositionalEncoding, EncoderLayer, TransformerEncoder) to `transformer_encoder.py`
2. Add: `enc = TransformerEncoder(10000, 512, 8, 2048, 6); print(enc(torch.randint(0, 10000, (2, 20))).shape)`
3. Run: `python transformer_encoder.py`

**Expected output:**
```
torch.Size([2, 20, 512])
```

</details>

Note the scaling factor `* math.sqrt(self.d_model)` applied to embeddings — this ensures the embedding magnitudes are comparable to the positional encoding magnitudes.

## Decoder Stack

Each decoder layer has **three** sub-layers:
1. **Masked** multi-head self-attention (prevents attending to future tokens)
2. Multi-head cross-attention (queries from decoder, keys/values from encoder)
3. Position-wise feed-forward network

### The Causal Mask

During training, the decoder sees the entire target sequence at once (for parallelism). The causal mask ensures that position \(i\) can only attend to positions \(\leq i\), preserving the autoregressive property.

```python title="Creating a causal (look-ahead) mask"
def create_causal_mask(size: int) -> torch.Tensor:
    """Create upper-triangular mask to prevent attending to future tokens."""
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0)  # (1, T, T)
    return mask  # 1 = attend, 0 = mask out

mask = create_causal_mask(5)
print(mask[0])
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

:::tip[Line-by-Line Walkthrough]
- **`torch.tril(torch.ones(size, size))`** — Creates a lower-triangular matrix of 1s. The lower triangle (including the diagonal) is 1, and the upper triangle is 0. This ensures token 3 can see tokens 1, 2, 3 but not tokens 4 or 5 — preventing "peeking into the future."
- **`.unsqueeze(0)`** — Adds a batch dimension so the mask can be broadcast across all items in a batch.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `causal_mask.py`
2. Run: `python causal_mask.py`

**Expected output:** A 5×5 matrix with 1s in the lower triangle and 0s in the upper triangle.

</details>

```python title="Transformer decoder layer"
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Masked self-attention
        normed = self.norm1(x)
        attn_out = self.masked_self_attention(normed, normed, normed, tgt_mask)
        x = x + self.dropout1(attn_out)

        # Cross-attention (Q from decoder, K/V from encoder)
        normed = self.norm2(x)
        enc_normed = encoder_output  # encoder output already normed
        cross_out = self.cross_attention(normed, enc_normed, enc_normed, src_mask)
        x = x + self.dropout2(cross_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm3(x))
        x = x + self.dropout3(ff_out)

        return x
```

:::tip[Line-by-Line Walkthrough]
- **`self.masked_self_attention(normed, normed, normed, tgt_mask)`** — The decoder first attends to its own tokens, but with a causal mask (`tgt_mask`) that prevents looking at future tokens. This is essential for autoregressive generation.
- **`self.cross_attention(normed, enc_normed, enc_normed, src_mask)`** — Cross-attention: the decoder's queries look at the encoder's keys and values. This is how the decoder accesses the input sequence. Like a translator reading the source text while writing the translation.
- **Three separate layer norms** — Each sub-layer (masked self-attention, cross-attention, FFN) has its own normalization, keeping gradients healthy through the deep network.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save together with MultiHeadAttention and FeedForward to `decoder_layer.py`
2. Add: `dl = DecoderLayer(512, 8, 2048); print(dl(torch.randn(2, 10, 512), torch.randn(2, 20, 512)).shape)`
3. Run: `python decoder_layer.py`

**Expected output:**
```
torch.Size([2, 10, 512])
```

</details>

## Attention Score Visualization

Let's build an interactive visualization to see how attention scores change with different input patterns.

```python title="Attention pattern analysis"
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_patterns():
    """Demonstrate different attention patterns that heads learn."""
    torch.manual_seed(42)
    seq_len = 8
    d_k = 16
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "dog"]

    # Pattern 1: Identity-like (each token attends to itself)
    Q1 = torch.eye(seq_len, d_k)
    K1 = torch.eye(seq_len, d_k)
    scores1 = F.softmax(Q1 @ K1.T / math.sqrt(d_k), dim=-1)

    # Pattern 2: Adjacent attention (bigram-like)
    Q2 = torch.randn(seq_len, d_k)
    K2 = torch.randn(seq_len, d_k)
    # Bias toward adjacent positions
    position_bias = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            position_bias[i, j] = -abs(i - j)
    scores2 = F.softmax((Q2 @ K2.T / math.sqrt(d_k)) + position_bias * 2, dim=-1)

    # Pattern 3: Causal (decoder-style)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    raw_scores = Q2 @ K2.T / math.sqrt(d_k)
    raw_scores = raw_scores.masked_fill(causal_mask == 0, float('-inf'))
    scores3 = F.softmax(raw_scores, dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Head A: Self-Focus", "Head B: Local Context", "Head C: Causal"]

    for ax, scores, title in zip(axes, [scores1, scores2, scores3], titles):
        im = ax.imshow(scores.detach().numpy(), cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Key (attending to)")
        ax.set_ylabel("Query (from)")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Attention Weight")
    plt.suptitle("Different Attention Head Patterns", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("attention_heads.png", dpi=150, bbox_inches='tight')
    plt.show()

visualize_attention_patterns()
```

:::tip[Line-by-Line Walkthrough]
- **Pattern 1 (Identity):** Uses identity matrices for Q and K, so each token primarily attends to itself. Useful for tasks where a token's own features are most important.
- **Pattern 2 (Adjacent):** Adds a position bias that penalizes distant tokens, making attention focus on nearby words — like reading a bigram model.
- **Pattern 3 (Causal):** Applies a causal mask so tokens can only attend to previous positions and themselves — this is exactly how decoder self-attention works in GPT-style models.
- **`np.random.dirichlet(...)`** / **`F.softmax(...)`** — Both produce probability distributions (rows sum to 1), but through different methods. Softmax is used in the actual attention computation.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch matplotlib numpy
```

**Steps:**
1. Save to `attention_patterns.py`
2. Run: `python attention_patterns.py`

**Expected output:** A figure with three side-by-side heatmaps: one showing diagonal self-focus, one showing local/adjacent attention, and one showing a lower-triangular causal pattern. Saved as `attention_heads.png`.

</details>

## Computational Complexity

:::info[Attention Complexity]
Self-attention has \(O(n^2 \cdot d)\) complexity where \(n\) is sequence length and \(d\) is the model dimension. This quadratic scaling in sequence length is both the Transformer's power (every token sees every other token) and its limitation (processing 10,000+ token sequences is expensive).

For comparison:
- **RNN:** \(O(n \cdot d^2)\) — linear in sequence length, quadratic in dimension
- **Self-attention:** \(O(n^2 \cdot d)\) — quadratic in sequence length, linear in dimension

For typical NLP (n ≈ 512, d ≈ 512), self-attention is faster due to GPU parallelism. For very long sequences, efficient attention variants (Longformer, Flash Attention) are needed.
:::

## The Full Picture

Let's see how data flows through the complete Transformer:

1. **Input tokens** are converted to embeddings and added to positional encodings.
2. The **encoder** processes the input through N layers of self-attention + FFN, producing contextualized representations.
3. The **decoder** receives target tokens (shifted right), processes them through masked self-attention, then cross-attends to the encoder output, and passes through FFN.
4. The decoder's output is projected to vocabulary size and passed through softmax to get token probabilities.

```python title="Complete Transformer architecture (forward pass)"
class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.src_pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.tgt_pos = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def encode(self, src: torch.Tensor, src_mask=None) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pos(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               src_mask=None, tgt_mask=None) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask=None, tgt_mask=None) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.output_projection(decoded)

# Instantiate and test
model = Transformer(src_vocab=10000, tgt_vocab=10000)
src = torch.randint(0, 10000, (2, 20))   # batch=2, src_len=20
tgt = torch.randint(0, 10000, (2, 15))   # batch=2, tgt_len=15
tgt_mask = create_causal_mask(15).unsqueeze(1)  # (1, 1, 15, 15)

output = model(src, tgt, tgt_mask=tgt_mask)
print(f"Output shape: {output.shape}")  # (2, 15, 10000)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~44M for default config
```

:::tip[Line-by-Line Walkthrough]
- **`self.src_embedding` / `self.tgt_embedding`** — Separate embedding lookup tables for source and target vocabularies. Each converts word indices into dense vectors.
- **`self.encoder_layers` / `self.decoder_layers`** — Stacks of N identical layers (default 6). Each encoder layer does self-attention + FFN; each decoder layer does masked self-attention + cross-attention + FFN.
- **`self.output_projection = nn.Linear(d_model, tgt_vocab)`** — Converts the decoder's 512-dimensional output into scores for every word in the target vocabulary. The highest score determines the predicted word.
- **`encode()` method** — Embeds source tokens, adds positional encoding, and passes through all encoder layers. Returns a "memory" tensor that the decoder will read from.
- **`decode()` method** — Embeds target tokens, adds positional encoding, and passes through all decoder layers. Each decoder layer cross-attends to the encoder memory.
- **`create_causal_mask(15).unsqueeze(1)`** — Creates a 15×15 causal mask and adds a dimension for broadcasting across attention heads.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save all classes (MultiHeadAttention, FeedForward, PositionalEncoding, EncoderLayer, DecoderLayer, Transformer, create_causal_mask) to `full_transformer.py`
2. Run: `python full_transformer.py`

**Expected output:**
```
Output shape: torch.Size([2, 15, 10000])
Total parameters: 44,140,544
```
(Exact parameter count depends on configuration.)

</details>

---

## Exercises

:::tip[Exercise 1: Attention Head Analysis — beginner]

Modify the `MultiHeadAttention` module to also return the attention weights. Feed a sample sentence through a randomly initialized Transformer encoder and visualize the attention patterns for all 8 heads. Even without training, can you spot different patterns between heads?

<details>
<summary>Hints</summary>

1. Extract attention weights from each head by modifying the MultiHeadAttention forward method to return them
2. Plot 8 separate heatmaps in a 2x4 grid
3. Look for patterns: diagonal, vertical stripes, block patterns

</details>

:::

:::tip[Exercise 2: Positional Encoding Properties — intermediate]

Verify two key properties of sinusoidal positional encodings:
1. The dot product \(PE_{\text{pos}} \cdot PE_{\text{pos}+k}\) depends only on the offset \(k\), not on the absolute position. Plot this relationship.
2. For any fixed offset \(k\), \(PE_{\text{pos}+k}\) can be expressed as a linear transformation of \(PE_{\text{pos}}\). Find and verify this transformation matrix.

<details>
<summary>Hints</summary>

1. Compute PE for positions 0-100 and plot dot products between positions
2. The dot product between PE(pos) and PE(pos+k) should depend only on k, not pos
3. Try computing PE(pos+k) as a linear transformation of PE(pos)

</details>

:::

:::tip[Exercise 3: Complexity Benchmarking — advanced]

Empirically verify the \(O(n^2)\) complexity of self-attention. Benchmark the forward pass time for increasing sequence lengths (64 to 2048) while keeping \(d_{\text{model}}\) constant. Plot the results and fit a polynomial to confirm quadratic scaling. Then compare against an LSTM on the same sequence lengths — at what length does the LSTM become faster?

<details>
<summary>Hints</summary>

1. Use torch.cuda.Event for precise GPU timing
2. Test sequence lengths: 64, 128, 256, 512, 1024, 2048
3. Plot time vs sequence length on a log-log scale
4. The slope should be approximately 2 for self-attention

</details>

:::

---

## Resources

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** _(paper)_ by Vaswani et al., 2017 — The original Transformer paper — essential reading for any AI engineer.

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** _(tutorial)_ by Jay Alammar — The best visual guide to understanding the Transformer architecture.

- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** _(tutorial)_ by Harvard NLP — Line-by-line PyTorch implementation with detailed annotations — an invaluable reference.

- **[Transformer from Scratch (YouTube)](https://www.youtube.com/watch?v=U0s0f995w14)** _(video)_ by Andrej Karpathy — Karpathy builds a Transformer from scratch, explaining every component.

- **[Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238)** _(paper)_ by Phuong & Hutter, 2022 — Rigorous mathematical description of Transformer algorithms — great for deep understanding.

- **[A Survey of Transformers](https://arxiv.org/abs/2106.04554)** _(paper)_ by Lin et al., 2022 — Comprehensive survey covering Transformer variants, efficiency improvements, and applications.
