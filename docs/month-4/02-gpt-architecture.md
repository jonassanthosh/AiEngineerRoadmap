---
sidebar_position: 2
slug: gpt-architecture
title: "The GPT Architecture"
---


# The GPT Architecture

:::info[What You'll Learn]
- Decoder-only Transformer architecture
- Causal (autoregressive) self-attention masking
- How GPT generates text token by token
- The evolution from GPT-1 to GPT-4
:::

:::note[Prerequisites]
[Scaling Laws](scaling-laws) from this month and [The Transformer Architecture](/curriculum/month-3/transformer-architecture) from Month 3.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

GPT (Generative Pre-trained Transformer) is the most influential family of language models. Its core idea is simple but powerful: take the Transformer decoder, train it on massive text corpora to predict the next token, and scale it up. This lesson walks through the architecture, its evolution from GPT-1 to GPT-3, and shows you how to build a GPT block from scratch in PyTorch.

## Decoder-Only Transformer Recap

In Month 3, you learned about the full encoder-decoder Transformer. GPT uses only the **decoder** half — but with an important modification. There is no encoder, so there are no cross-attention layers. The decoder simply attends to previous tokens in the sequence using **causal (masked) self-attention**.

```
  Input tokens:    [The] [cat] [sat] [on]  [the]
                     ↓     ↓     ↓    ↓     ↓
              ┌──────────────────────────────────┐
              │    Token + Positional Embeddings  │
              └──────────┬───────────────────────┘
                         │
              ┌──────────┴───────────────────────┐
              │  Causal Self-Attention (masked)   │
              │  + Residual & LayerNorm           │
              ├──────────────────────────────────┤
              │  Feed-Forward Network             │
              │  + Residual & LayerNorm           │
              └──────────┬───────────────────────┘
                         │           × N layers
              ┌──────────┴───────────────────────┐
              │  Linear → Vocabulary Logits       │
              │  Softmax → P(next token)          │
              └──────────────────────────────────┘
  
  Prediction:  [cat] [sat] [on] [the] [mat]
```

Each token can only attend to itself and tokens to its **left**. This is enforced by a causal attention mask — an upper-triangular matrix of negative infinity values that prevents information from flowing backward.

:::info[Why Decoder-Only?]
A decoder-only architecture is simpler, easier to scale, and naturally suited to **generation** — the model produces one token at a time, conditioning on everything before it. This is exactly what you need for text generation, code completion, and chat. Encoder-decoder models are better for tasks with a clear input-output split (like translation), but decoder-only models handle those tasks well enough with appropriate prompting.
:::

## Causal (Autoregressive) Language Modeling

GPT is trained with a single objective: **predict the next token** given all previous tokens.

:::info[Plain English: What Is the Autoregressive Objective?]
Imagine playing a "guess the next word" game. Someone shows you the beginning of a sentence — "The cat sat on the" — and you guess what comes next ("mat"). The model plays this game for every single position in every sentence it reads. Its score is how confident it was in the correct next word. Training makes the model better and better at this game.
:::

:::note[Autoregressive Objective]
Given a sequence of tokens \( x_1, x_2, \ldots, x_T \), the model maximizes:

$$
\mathcal{L} = \sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)
$$

This is equivalent to minimizing the cross-entropy loss between the model's predicted distribution and the true next token, averaged over all positions.
:::

**Reading the formula:** **𝓛** is the total score we want to maximize (the log-likelihood). **x_t** is the token at position t. **P(x_t | x_1, ... x_{t-1}; θ)** is the model's predicted probability for the correct token at position t, given all previous tokens. **θ** represents the model's learnable weights. **log** converts probabilities to a scale where we can add them up. **Σ from t=1 to T** means "add up the score at every position in the sequence." In plain terms: the model gets a point for each position based on how confident it was in the right answer, and we add up all those points.

During training, the model processes the entire sequence in parallel. At position \( t \), the causal mask ensures the model only sees tokens \( x_1 \) through \( x_t \), so it predicts \( x_{t+1} \). This gives you \( T-1 \) training examples from a single sequence — highly efficient.

During inference (generation), the model produces one token at a time:

1. Feed in the prompt tokens
2. Get the probability distribution over the vocabulary for the next position
3. Sample a token (or take the argmax)
4. Append it to the sequence
5. Repeat

## GPT-1, GPT-2, GPT-3: The Evolution

The GPT family demonstrates the power of scaling laws in practice. The architecture barely changed — the magic was in scale and data.

### GPT-1 (2018)

| Property | Value |
|----------|-------|
| Parameters | 117M |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Context length | 512 |
| Training data | BookCorpus (~800M tokens) |

GPT-1 introduced the idea of **unsupervised pre-training followed by supervised fine-tuning**. The pre-trained model learned general language understanding, then was fine-tuned on specific tasks (classification, entailment, similarity). This was a paradigm shift from training task-specific models from scratch.

### GPT-2 (2019)

| Property | Value |
|----------|-------|
| Parameters | 1.5B (largest) |
| Layers | 48 |
| Hidden dim | 1600 |
| Attention heads | 25 |
| Context length | 1024 |
| Training data | WebText (~40GB, ~10B tokens) |

GPT-2 demonstrated that scaling up the same architecture and training on diverse web text produced a model capable of **zero-shot** task performance. No fine-tuning needed — just prompt the model with a task description.

Key architectural changes from GPT-1:
- Layer normalization moved to the **input** of each block (pre-norm) instead of the output
- An additional layer norm added after the final self-attention block
- Residual path weights scaled by \( 1/\sqrt{N} \) where \( N \) is the number of layers

### GPT-3 (2020)

| Property | Value |
|----------|-------|
| Parameters | 175B |
| Layers | 96 |
| Hidden dim | 12288 |
| Attention heads | 96 |
| Context length | 2048 |
| Training data | ~300B tokens (Common Crawl, books, Wikipedia) |

GPT-3 was a 100x scale-up that demonstrated **few-shot learning**: by providing a few examples in the prompt, the model could perform tasks it was never explicitly trained on. This was the birth of "in-context learning."

:::info[In-Context Learning]
GPT-3 showed that large language models can learn from examples provided in the prompt, without any gradient updates. This is fundamentally different from fine-tuning — the model's weights don't change. The mechanism behind in-context learning is still debated, but it likely involves the model's attention layers implementing something like implicit gradient descent during the forward pass.
:::

## Architecture Details

Let's examine each component of a GPT block in detail.

### 1. Token and Position Embeddings

The input to GPT is a sequence of token IDs. These are converted to dense vectors via two embedding tables:

- **Token embedding**: Maps each vocabulary token to a vector of dimension \( d_{\text{model}} \). Vocabulary size is typically 50,000–100,000.
- **Position embedding**: Maps each position (0, 1, 2, ..., context_length-1) to a vector of the same dimension. GPT-1/2/3 use **learned** position embeddings (unlike the sinusoidal encodings in the original Transformer).

The two embeddings are **summed** (not concatenated) to produce the input to the first Transformer block.

### 2. Causal Self-Attention

:::info[Plain English: How Does Causal Attention Work?]
Picture a classroom where students sit in a row. Each student can pass notes to anyone sitting to their left or to themselves — but **never** to someone sitting to their right. This is causal attention: each word can "look at" (attend to) previous words for context, but can't peek at future words. The model scores how relevant each past word is to the current word, then blends their information together.
:::

The attention mechanism is identical to standard multi-head attention, except for the causal mask:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

**Reading the formula:** **Q** (Query) represents "what am I looking for?" for each position. **K** (Key) represents "what do I contain?" for each position. **V** (Value) represents "what information do I provide?" for each position. **QKᵀ** computes a compatibility score between every pair of positions. **√d_k** is a scaling factor (the square root of the key dimension) that prevents scores from getting too large. **M** is the causal mask that blocks future positions. **softmax** turns raw scores into probabilities (they sum to 1). The final multiplication by **V** produces a weighted mix of value vectors.

where \( M \) is the causal mask:

:::info[Plain English: What Is the Causal Mask?]
Think of a one-way mirror in a row of windows. Each window (position) can see everything to its left and itself, but the glass turns opaque for anything to the right. The causal mask is like those one-way mirrors — it lets each word look backward in time but blocks it from peeking at future words. We do this by putting "negative infinity" in the blocked spots, which means "zero chance of looking there" after we run softmax.
:::

$$
M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
$$

**Reading the formula:** **M** is a grid of numbers. **i** is the row (the current position), **j** is the column (the position being attended to). When i ≥ j (current position is at or after the attended position), the mask is 0 — meaning "allowed to look." When i < j (trying to look into the future), the mask is −∞ (negative infinity) — which makes the softmax output zero, effectively blocking that connection.

This ensures position \( i \) can only attend to positions \( \leq i \).

### 3. Feed-Forward Network

:::info[Plain English: What Does the Feed-Forward Network Do?]
After attention figures out *which* words are relevant, the feed-forward network decides *what to do* with that information. Think of it like a two-step filter: first, it expands the information into a wider space (like brainstorming many possibilities), then squishes it back down (like picking the best ideas). This is where much of the model's "thinking" happens.
:::

Each attention layer is followed by a position-wise feed-forward network:

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

**Reading the formula:** **x** is the input vector at a single position (after attention). **W₁** and **b₁** are the weights and bias of the first linear layer — this *expands* the vector to 4× its size. **GELU** is an activation function (a smooth on/off switch for each neuron). **W₂** and **b₂** are the second linear layer — this *compresses* back to the original size. The whole thing is: expand → activate → compress.

GPT uses **GELU** (Gaussian Error Linear Unit) instead of ReLU. The inner dimension is typically \( 4 \times d_{\text{model}} \).

### 4. Output Projection (Weight Tying)

The final output layer projects from \( d_{\text{model}} \) to vocabulary size. GPT ties this weight matrix with the **token embedding matrix** — the same matrix is used for both input embedding and output projection (transposed). This reduces parameter count and improves performance.

## Full GPT Block in PyTorch

```python title="Complete GPT implementation"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask: upper triangle = True (will be masked out)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, d_k)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class GPTBlock(nn.Module):
    """A single GPT Transformer block with pre-norm."""
    def __init__(self, d_model, n_heads, context_length, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 context_length, dropout=0.1):
        super().__init__()
        self.context_length = context_length

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            GPTBlock(d_model, n_heads, context_length, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context_length

        tok = self.tok_emb(idx)                          # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = self.drop(tok + pos)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# Instantiate a small GPT
model = GPT(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    context_length=1024,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"GPT model with {total_params/1e6:.1f}M parameters")

# Forward pass test
x = torch.randint(0, 50257, (2, 128))
logits, loss = model(x, targets=x)
print(f"Logits shape: {logits.shape}")  # (2, 128, 50257)
```

:::tip[Line-by-Line Walkthrough]
- **`self.qkv = nn.Linear(d_model, 3 * d_model)`** — A single linear layer that projects the input into Queries, Keys, and Values simultaneously (3× the model dimension). This is more efficient than three separate layers.
- **`mask = torch.triu(torch.ones(...), diagonal=1).bool()`** — Creates the causal mask: an upper-triangular matrix of `True` values that will be used to block future positions in attention.
- **`att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)`** — Computes the attention scores: dot product of queries and keys, scaled down by √d_k to keep values in a stable range.
- **`att = att.masked_fill(self.mask[:T, :T], float('-inf'))`** — Fills future positions with negative infinity so softmax makes them zero (no peeking ahead).
- **`self.head.weight = self.tok_emb.weight`** — Weight tying: the output layer shares the same matrix as the input embedding, so the model uses the same "dictionary" for reading and writing.
- **`nn.init.normal_(module.weight, std=0.02)`** — Initializes all weights from a narrow bell curve centered at 0. Small initial weights prevent the model from making wild predictions before training.
- **`idx_cond = idx[:, -self.context_length:]`** — During generation, crops the input to the maximum context length (the model can only see this many past tokens).
- **`next_token = torch.multinomial(probs, num_samples=1)`** — Randomly samples one token from the probability distribution. This is what makes generation non-deterministic.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the entire code block to a file, e.g. `gpt_model.py`
2. Run: `python gpt_model.py`

**Expected output:**
```
GPT model with 124.4M parameters
Logits shape: torch.Size([2, 128, 50257])
```
The model instantiates and runs a forward pass with random input (no GPU required, but it will be slow on CPU for large models).

</details>

### Key Design Decisions in This Code

**Pre-norm vs post-norm.** GPT-2 and later models apply LayerNorm **before** each sub-layer (`ln → attn → residual`), not after. This stabilizes training at large scales because the residual path carries unmodified gradients.

**Weight tying.** The token embedding matrix and the output projection matrix share the same weights. This is mathematically sensible: if the embedding for "cat" is a vector \( e_{\text{cat}} \), then the logit for "cat" at the output is \( x^T e_{\text{cat}} \) — the dot product between the hidden state and the same embedding.

**Weight initialization.** Weights are initialized from \( \mathcal{N}(0, 0.02) \). This is important for training stability. Some implementations also scale the residual projection weights by \( 1/\sqrt{2N} \) to prevent the residual stream from growing with depth.

## The KV Cache: Making Generation Fast

During generation, the model recomputes attention over all previous tokens at every step. This is wasteful — the keys and values for previous positions don't change. The **KV cache** stores previously computed keys and values so they're only computed once.

```python title="KV cache concept"
# Without KV cache: O(T^2) per token generated
# Step 1: process tokens [0..99], compute attention over all 100
# Step 2: process tokens [0..100], recompute attention over all 101
# Step 3: process tokens [0..101], recompute attention over all 102
# Total work for generating 100 tokens: ~100 * 200 * d = O(T^2 * d)

# With KV cache: O(T) per token generated
# Step 1: process tokens [0..99], cache all K,V
# Step 2: process ONLY token [100], use cached K,V + new K,V
# Step 3: process ONLY token [101], use cached K,V + new K,V
# Total work for generating 100 tokens: ~100 * d = O(T * d)

# This is why KV cache is essential for production LLM serving.
# Memory cost: O(n_layers * T * d_model) per sequence.
```

:::tip[Line-by-Line Walkthrough]
This is a conceptual pseudocode block (no runnable code). The key idea:
- **Without KV cache:** every time you generate a new token, you recompute attention for *all* previous tokens from scratch. Generating 100 tokens means repeating work ~100 times — cost grows quadratically.
- **With KV cache:** you compute Keys and Values for each token only *once*, then store (cache) them. When generating the next token, you only compute Q/K/V for the new token and reuse all previous K/V values. This makes generation linear in sequence length — a massive speedup.
:::

:::tip[Production Optimization]
In production, the KV cache is the primary memory bottleneck during serving. Techniques like **paged attention** (vLLM), **multi-query attention** (MQA), and **grouped-query attention** (GQA) all aim to reduce KV cache memory. We'll cover these in Month 5.
:::

## Summary

| Component | Role |
|-----------|------|
| Token embedding | Maps token IDs to dense vectors |
| Position embedding | Encodes position information (learned in GPT) |
| Causal self-attention | Each token attends only to previous tokens |
| Feed-forward network | Non-linear transformation at each position |
| Pre-norm LayerNorm | Stabilizes training at scale |
| Weight tying | Shares embedding and output projection weights |
| KV cache | Avoids redundant computation during generation |

---

## Exercises

:::tip[Count the Parameters — beginner]

Calculate the total parameter count for GPT-2 (small): vocab_size=50257, d_model=768, n_heads=12, n_layers=12, context_length=1024. Break it down by component. How does your count compare to the 117M figure usually cited?

<div>
**Solution:**
- Token embedding: 50257 × 768 = 38.6M
- Position embedding: 1024 × 768 = 0.8M
- Per block attention (QKV + proj): 4 × 768² = 2.4M
- Per block FFN: 2 × 768 × 3072 = 4.7M
- Per block LayerNorm (×2): 2 × 2 × 768 = 3K
- 12 blocks: 12 × (2.4M + 4.7M + 3K) = 85.2M
- Final LayerNorm: 1.5K
- Output projection: tied with token embedding (0 extra)
- **Total: ~124.4M** (the "117M" figure excludes position embeddings and some components)
<details>
<summary>Hints</summary>

1. Token embedding: vocab_size × d_model
2. Position embedding: context_length × d_model
3. Each attention block: 4 × d_model² (for QKV + output projection)
4. Each FFN: 2 × 4 × d_model² (two linear layers)
5. Don't forget LayerNorm parameters

</details>

:::

:::tip[Implement Top-p (Nucleus) Sampling — intermediate]

The `generate` method above supports top-k sampling. Extend it to support **top-p (nucleus) sampling**, where you keep the smallest set of tokens whose cumulative probability exceeds a threshold \( p \). This tends to produce more natural text than top-k.

<div>
**Solution:**

```python
def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float('-inf')
    
    probs = F.softmax(sorted_logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, token)
```

:::tip[Line-by-Line Walkthrough]
- **`sorted_logits, sorted_indices = torch.sort(logits, descending=True)`** — Sorts the raw scores from highest to lowest so we can work from the most likely tokens downward.
- **`cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)`** — Converts sorted scores to probabilities and computes a running total. For example, if the top 3 probabilities are 0.4, 0.3, 0.2, the cumulative sums are 0.4, 0.7, 0.9.
- **`sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p`** — Identifies all tokens whose cumulative probability *before adding them* exceeds the threshold p. These are the tokens we want to cut off.
- **`sorted_logits[sorted_mask] = float('-inf')`** — Kills the low-probability tail by setting their scores to −∞, so they get zero probability after softmax.
- **`return sorted_indices.gather(-1, token)`** — Maps the sampled token back to its original position in the vocabulary (undoing the sort).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Add `import torch` and `import torch.nn.functional as F` at the top of your script.
2. Create a dummy logits tensor: `logits = torch.randn(1, 50257)` and call `top_p_sample(logits)`.

**Expected output:** A single tensor containing the token ID sampled from the nucleus of the distribution. The exact token varies each run since it's random.

</details>

<details>
<summary>Hints</summary>

1. Sort the probabilities in descending order
2. Compute the cumulative sum
3. Find the cutoff where cumulative probability exceeds p
4. Zero out everything below the cutoff, renormalize

</details>

:::

:::tip[Attention Pattern Analysis — advanced]

Modify the `CausalSelfAttention` class to optionally return the attention weights. Then load a pre-trained GPT-2 model (from Hugging Face) and visualize the attention patterns for a sample sentence across different layers and heads. What patterns do you observe?

<div>
**Solution approach:**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

inputs = tokenizer("The cat sat on the mat", return_tensors="pt")
outputs = model(**inputs)

# outputs.attentions is a tuple of (n_layers) tensors,
# each of shape (batch, heads, seq_len, seq_len)
import matplotlib.pyplot as plt
attn = outputs.attentions[0][0]  # layer 0, batch 0
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(attn[i].detach().numpy(), cmap='viridis')
    ax.set_title(f'Head \{i\}')
```

:::tip[Line-by-Line Walkthrough]
- **`GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)`** — Loads a pre-trained GPT-2 model and tells it to return attention weights alongside its predictions.
- **`outputs.attentions[0][0]`** — Extracts the attention weights from layer 0, batch item 0. This is a tensor of shape (12, seq_len, seq_len) — one attention matrix per head.
- **`ax.imshow(attn[i].detach().numpy(), cmap='viridis')`** — Plots each attention head as a heatmap. Bright spots show where the model is "looking." The x-axis is "attending to" and the y-axis is "attending from."
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch transformers matplotlib
```

**Steps:**
1. Save the code to `attention_viz.py`
2. Run: `python attention_viz.py`
3. First run downloads GPT-2 (~500 MB).

**Expected output:** A 3×4 grid of heatmaps, each showing one attention head's pattern for the sentence "The cat sat on the mat." You'll see diagonal patterns (attend to previous token), vertical stripes (attend to first token), and broader patterns.

</details>

Common patterns: some heads attend to the previous token (diagonal), some attend to the first token (vertical stripe at position 0), some form broader contextual patterns.
<details>
<summary>Hints</summary>

1. Register forward hooks on the attention layers
2. Extract the attention weights after softmax
3. Use matplotlib's imshow to visualize
4. Look for patterns: diagonal, vertical stripes, block structure

</details>

:::

---

## Resources

- **[Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** _(paper)_ by Radford et al. — The original GPT paper introducing unsupervised pre-training for NLP.

- **[Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** _(paper)_ by Radford et al. — GPT-2 paper demonstrating zero-shot task transfer.

- **[Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)** _(paper)_ by Brown et al. — The GPT-3 paper introducing few-shot in-context learning.

- **[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)** _(tutorial)_ by Jay Alammar — Visual walkthrough of the GPT-2 architecture with detailed diagrams.

- **[minGPT by Andrej Karpathy](https://github.com/karpathy/minGPT)** _(tool)_ by Andrej Karpathy — A clean, minimal PyTorch re-implementation of GPT for educational purposes.

- **[Let's Build GPT: From Scratch, in Code](https://www.youtube.com/watch?v=kCc8FmEb1nY)** _(video)_ by Andrej Karpathy — A 2-hour video building a GPT model from scratch — the single best resource for understanding GPT.
