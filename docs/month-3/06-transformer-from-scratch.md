---
sidebar_position: 6
slug: transformer-from-scratch
title: "Building a Transformer from Scratch"
---


# Building a Transformer from Scratch

In the previous lesson, you learned the theory behind every Transformer component. Now it's time to build one from the ground up in PyTorch and train it on a real task. By the end of this lesson, you'll have a fully working Transformer that can translate between two languages.

We'll build each module independently, test it, then assemble them into a complete encoder-decoder Transformer.

:::tip[How to Approach This Lesson]
Type out the code yourself rather than copy-pasting. The act of writing each module will solidify your understanding of how the pieces fit together. Run each section independently to verify shapes before moving on.
:::

## Module 1: Multi-Head Attention

This is the core building block. We need a single module that handles self-attention (encoder), masked self-attention (decoder), and cross-attention (decoder attending to encoder).

```python title="multi_head_attention.py"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project and reshape: (B, T, d_model) → (B, h, T, head_dim)
        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, T_q, T_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        attn_output = torch.matmul(attn_weights, V)  # (B, h, T_q, head_dim)

        # Concatenate heads: (B, h, T_q, head_dim) → (B, T_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        return self.out_proj(attn_output)

# ---- Test ----
mha = MultiHeadAttention(d_model=64, num_heads=4)

# Self-attention test
x = torch.randn(2, 10, 64)
out = mha(x, x, x)
assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"

# Cross-attention test (different sequence lengths for Q vs K/V)
q = torch.randn(2, 8, 64)
kv = torch.randn(2, 12, 64)
out = mha(q, kv, kv)
assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"

print("✓ MultiHeadAttention tests passed")
```

:::tip[Line-by-Line Walkthrough]
- **`self.head_dim = d_model // num_heads`** — Splits the total dimension evenly among heads. With 64 dims and 4 heads, each head works with 16 dimensions.
- **`self.q_proj`, `self.k_proj`, `self.v_proj`** — Three separate linear layers that project the input into queries, keys, and values. Each head gets its own slice of these projections.
- **`.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)`** — Reshapes the projected tensor from `(B, T, d_model)` to `(B, num_heads, T, head_dim)`, splitting the embedding dimension into separate heads so each head processes independently.
- **`torch.matmul(Q, K.transpose(-2, -1)) / self.scale`** — Scaled dot-product: computes similarity between every query-key pair, then divides by √head_dim to keep values in a reasonable range.
- **`attn_scores.masked_fill(mask == 0, float("-inf"))`** — Sets masked positions to negative infinity so they become zero after softmax — the model cannot attend to those positions.
- **`.transpose(1, 2).contiguous().view(B, T_q, self.d_model)`** — Concatenates all heads back together by reshaping from `(B, heads, T, head_dim)` back to `(B, T, d_model)`.
- **`self.out_proj(attn_output)`** — A final linear layer that mixes information across heads.
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
✓ MultiHeadAttention tests passed
```

</details>

## Module 2: Positional Encoding

```python title="positional_encoding.py"
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the positional encoding frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ---- Test ----
pe = PositionalEncoding(d_model=64)
x = torch.zeros(2, 20, 64)
out = pe(x)
assert out.shape == (2, 20, 64)
# Verify positions are different
assert not torch.allclose(out[0, 0], out[0, 1]), "Different positions should have different encodings"
print("✓ PositionalEncoding tests passed")
```

:::tip[Line-by-Line Walkthrough]
- **`pe = torch.zeros(max_len, d_model)`** — Pre-allocates a matrix of zeros for up to 5000 positions. Each row will store the positional encoding for one position.
- **`div_term = torch.exp(...)`** — Computes different frequencies for each pair of dimensions. Low dimensions oscillate fast (like high-pitched notes), high dimensions oscillate slowly (like bass notes). This creates a unique pattern for every position.
- **`pe[:, 0::2] = torch.sin(...)` / `pe[:, 1::2] = torch.cos(...)`** — Even dimensions get sine values, odd dimensions get cosine values. Together they form a unique "fingerprint" for each position.
- **`self.register_buffer("pe", ...)`** — Stores the encoding as a non-trainable buffer. It moves to GPU with the model but has no gradients — positions are fixed, not learned.
- **`x = x + self.pe[:, :x.size(1)]`** — Adds the positional encoding to the word embeddings. After this, each token knows both what it means and where it sits in the sequence.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `positional_encoding.py` (include `import torch, torch.nn as nn, math`)
2. Run: `python positional_encoding.py`

**Expected output:**
```
✓ PositionalEncoding tests passed
```

</details>

## Module 3: Feed-Forward Network

```python title="feed_forward.py"
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))

# ---- Test ----
ffn = PositionWiseFFN(d_model=64, d_ff=256)
x = torch.randn(2, 10, 64)
out = ffn(x)
assert out.shape == (2, 10, 64)
print("✓ PositionWiseFFN tests passed")
```

:::tip[Line-by-Line Walkthrough]
- **`self.w1 = nn.Linear(d_model, d_ff)`** — First layer expands the dimension (e.g., 64 → 256), giving the network a larger space to learn complex transformations.
- **`F.relu(self.w1(x))`** — Applies the ReLU activation, which zeroes out negative values. This nonlinearity is essential for learning complex patterns.
- **`self.w2(self.dropout(...))`** — Applies dropout (randomly zeroing values during training to prevent overfitting), then compresses back to the original dimension (256 → 64).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `feed_forward.py` (include `import torch, torch.nn as nn, torch.nn.functional as F`)
2. Run: `python feed_forward.py`

**Expected output:**
```
✓ PositionWiseFFN tests passed
```

</details>

## Module 4: Encoder Block

```python title="encoder_block.py"
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, src_mask))

        # Pre-norm FFN with residual
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))

        return x

# ---- Test ----
encoder_block = EncoderBlock(d_model=64, num_heads=4, d_ff=256)
x = torch.randn(2, 10, 64)
out = encoder_block(x)
assert out.shape == (2, 10, 64)
print("✓ EncoderBlock tests passed")
```

:::tip[Line-by-Line Walkthrough]
- **`self.norm1(x)` → `self.self_attn(normed, normed, normed, src_mask)`** — Pre-norm pattern: normalize first, then pass through self-attention. All three inputs (Q, K, V) are the same — this is **self**-attention where every token attends to every other token.
- **`x = x + self.dropout1(...)`** — The residual connection: adds the attention output back to the original input. Even if attention does nothing useful, the original signal passes through.
- **`self.ffn(normed)`** — The feed-forward network processes each token independently after the attention step. Think of attention as "group discussion" and FFN as "individual thinking."
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save with MultiHeadAttention and PositionWiseFFN to `encoder_block.py`
2. Run: `python encoder_block.py`

**Expected output:**
```
✓ EncoderBlock tests passed
```

</details>

## Module 5: Decoder Block

The decoder block has three sub-layers: masked self-attention, cross-attention, and FFN.

```python title="decoder_block.py"
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, tgt_mask))

        # Cross-attention: query from decoder, key/value from encoder
        normed = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(normed, encoder_output, encoder_output, src_mask))

        # Feed-forward
        normed = self.norm3(x)
        x = x + self.dropout3(self.ffn(normed))

        return x

# ---- Test ----
decoder_block = DecoderBlock(d_model=64, num_heads=4, d_ff=256)
tgt = torch.randn(2, 8, 64)
memory = torch.randn(2, 12, 64)
out = decoder_block(tgt, memory)
assert out.shape == (2, 8, 64)
print("✓ DecoderBlock tests passed")
```

:::tip[Line-by-Line Walkthrough]
- **`self.self_attn(normed, normed, normed, tgt_mask)`** — Masked self-attention: the decoder attends to its own tokens, but `tgt_mask` prevents it from seeing future tokens. Like reading a sentence left to right without peeking ahead.
- **`self.cross_attn(normed, encoder_output, encoder_output, src_mask)`** — Cross-attention: the query comes from the decoder ("what am I looking for?"), but keys and values come from the encoder ("what did the input say?"). This is how the decoder reads the source sequence.
- **`self.ffn(normed)`** — After talking to itself and the encoder, each token processes everything through its own feed-forward network.
- **Three norm layers + three dropout layers** — Each of the three sub-blocks (masked self-attn, cross-attn, FFN) gets its own normalization and dropout for stable training.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save with MultiHeadAttention and PositionWiseFFN to `decoder_block.py`
2. Run: `python decoder_block.py`

**Expected output:**
```
✓ DecoderBlock tests passed
```

</details>

## Module 6: Full Transformer Model

Now we assemble everything into the complete model.

```python title="transformer.py — The complete model"
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # (B, 1, 1, src_len) — mask padding tokens
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        B, T = tgt.shape
        # Padding mask
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T, T)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)                    # (1, 1, T, T)
        return pad_mask & causal_mask  # (B, 1, T, T)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, src_mask, tgt_mask)

        return self.output_proj(decoded)

# ---- Test ----
model = Transformer(src_vocab_size=100, tgt_vocab_size=100)
src = torch.randint(1, 100, (4, 12))
tgt = torch.randint(1, 100, (4, 10))
out = model(src, tgt)
assert out.shape == (4, 10, 100)

params = sum(p.numel() for p in model.parameters())
print(f"✓ Transformer tests passed — {params:,} parameters")
```

:::tip[Line-by-Line Walkthrough]
- **`self.src_embedding` / `self.tgt_embedding`** — Separate embedding tables for source and target vocabularies. `padding_idx=pad_idx` ensures the padding token always maps to a zero vector.
- **`self.pos_encoding`** — Shared positional encoding for both source and target (the same sine/cosine patterns work for any sequence).
- **`make_src_mask(src)`** — Creates a mask that marks padding tokens as "ignore." Shape `(B, 1, 1, src_len)` so it broadcasts correctly across attention heads.
- **`make_tgt_mask(tgt)`** — Creates a combined mask: (1) padding mask (ignore pad tokens) AND (2) causal mask (prevent seeing future tokens). Both conditions must be true to attend.
- **`self.src_embedding(src) * math.sqrt(self.d_model)`** — Scales embeddings by √d_model so their magnitude matches the positional encodings.
- **`self.output_proj(decoded)`** — Converts the decoder's representation into scores over the target vocabulary. The highest score is the predicted next word.
- **`nn.init.xavier_uniform_(p)`** — Initializes all weight matrices with Xavier uniform initialization, which keeps signal magnitudes stable through deep networks.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save all modules (MultiHeadAttention, PositionalEncoding, PositionWiseFFN, EncoderBlock, DecoderBlock, Transformer) to `transformer.py`
2. Run: `python transformer.py`

**Expected output:**
```
✓ Transformer tests passed — 1,435,236 parameters
```
(Exact parameter count depends on vocab size and model dimensions.)

</details>

:::warning[Common Implementation Bugs]
1. **Forgetting the embedding scale factor:** The embeddings are multiplied by \(\sqrt{d_{\text{model}}}\) to match the magnitude of positional encodings. Skip this and training will be unstable.
2. **Wrong mask shape:** The mask must broadcast correctly to `(B, num_heads, T_q, T_k)`. A common bug is using `(B, T)` instead of `(B, 1, 1, T)`.
3. **Not using `.contiguous()` after transpose:** Reshaping after transpose without `.contiguous()` causes silent errors or crashes.
:::

## Training on a Toy Translation Task

Let's train our Transformer on a synthetic task: translating sequences of numbers with a simple rule. This lets us verify the model works without needing a real dataset.

Our task: reverse the input sequence and add 1 to each element (modular arithmetic). For example, `[3, 7, 1, 5]` → `[6, 2, 8, 4]`.

```python title="Dataset and training loop"
import random

# Configuration
PAD, SOS, EOS = 0, 1, 2
NUM_TOKENS = 10
VOCAB_SIZE = NUM_TOKENS + 3  # 0=PAD, 1=SOS, 2=EOS, 3-12=digits 0-9
OFFSET = 3

def generate_pair(seq_len: int = 6):
    """Generate a (source, target) pair: reverse and add 1."""
    src_digits = [random.randint(0, NUM_TOKENS - 1) for _ in range(seq_len)]
    tgt_digits = [(d + 1) % NUM_TOKENS for d in reversed(src_digits)]

    src = [SOS] + [d + OFFSET for d in src_digits] + [EOS]
    tgt = [SOS] + [d + OFFSET for d in tgt_digits] + [EOS]
    return src, tgt

def make_batch(batch_size: int, seq_len: int = 6):
    pairs = [generate_pair(seq_len) for _ in range(batch_size)]
    src = torch.tensor([p[0] for p in pairs])
    tgt = torch.tensor([p[1] for p in pairs])
    return src, tgt

# Model setup
model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=64,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=256,
    dropout=0.1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

# Learning rate warmup schedule (from the original paper)
def lr_schedule(step, d_model=64, warmup_steps=400):
    if step == 0:
        step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

# Training
EPOCHS = 40
BATCH_SIZE = 64
SEQ_LEN = 6

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for _ in range(50):  # 50 batches per epoch
        src, tgt = make_batch(BATCH_SIZE, SEQ_LEN)
        tgt_input = tgt[:, :-1]   # decoder input: everything except last token
        tgt_output = tgt[:, 1:]   # decoder target: everything except first token

        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / 50
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
```

:::tip[Line-by-Line Walkthrough]
- **`generate_pair(seq_len=6)`** — Creates one training example: a random sequence of digits, reversed and incremented by 1 (with wraparound). For example, `[3, 7, 1]` → `[2, 8, 4]`. SOS and EOS tokens mark sequence boundaries.
- **`tgt_input = tgt[:, :-1]` / `tgt_output = tgt[:, 1:]`** — The decoder receives the target shifted right (everything except the last token) and tries to predict the target shifted left (everything except the first token). This is standard teacher forcing.
- **`criterion = nn.CrossEntropyLoss(ignore_index=PAD)`** — Measures how wrong the predictions are, while ignoring padding tokens (which carry no information).
- **`lr_schedule(step, ...)`** — Implements the Transformer's learning rate schedule: linear warmup followed by inverse square root decay. This prevents the model from taking too-large steps early in training.
- **`clip_grad_norm_(model.parameters(), 1.0)`** — Caps gradient magnitudes at 1.0 to prevent exploding gradients during training.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save all modules plus this training code to `train_transformer.py`
2. Run: `python train_transformer.py`
3. Training takes 1-2 minutes on CPU.

**Expected output:**
```
Epoch   5 | Loss: 1.8234 | LR: 0.006250
Epoch  10 | Loss: 0.5678 | LR: 0.004419
Epoch  15 | Loss: 0.1234 | LR: 0.003608
...
Epoch  40 | Loss: 0.0089 | LR: 0.002210
```
(Exact values will vary, but loss should decrease steadily.)

</details>

:::info[Decoder Input vs. Target]
The decoder input is the target sequence **shifted right** — it receives `[SOS, tok1, tok2, ..., tokN]` and must predict `[tok1, tok2, ..., tokN, EOS]`. This is the standard "teacher forcing" setup for autoregressive models.
:::

## Inference with Greedy Decoding

At inference time, we don't have the target sequence. We generate one token at a time, feeding each prediction back as input.

```python title="Greedy decoding for the trained model"
@torch.no_grad()
def translate(model, src_tokens: list[int], max_len: int = 20) -> list[int]:
    """Translate a source sequence using greedy decoding."""
    model.eval()
    src = torch.tensor([src_tokens])
    src_mask = model.make_src_mask(src)
    memory = model.encode(src, src_mask)

    # Start with SOS token
    tgt_tokens = [SOS]

    for _ in range(max_len):
        tgt = torch.tensor([tgt_tokens])
        tgt_mask = model.make_tgt_mask(tgt)

        decoded = model.decode(tgt, memory, src_mask, tgt_mask)
        logits = model.output_proj(decoded[:, -1, :])  # last position
        next_token = logits.argmax(dim=-1).item()

        if next_token == EOS:
            break
        tgt_tokens.append(next_token)

    return tgt_tokens[1:]  # exclude SOS

# Test on examples
print("\\n--- Translation Results ---")
correct = 0
total = 20

for _ in range(total):
    digits = [random.randint(0, 9) for _ in range(SEQ_LEN)]
    expected = [(d + 1) % 10 for d in reversed(digits)]

    src = [SOS] + [d + OFFSET for d in digits] + [EOS]
    result = translate(model, src)
    result_digits = [t - OFFSET for t in result]

    match = "✓" if result_digits == expected else "✗"
    if result_digits == expected:
        correct += 1
    print(f"  {match}  {digits} → {result_digits}  (expected {expected})")

print(f"\\nAccuracy: {correct}/{total} = {100*correct/total:.0f}%")
```

:::tip[Line-by-Line Walkthrough]
- **`model.encode(src, src_mask)`** — Runs the encoder once on the input to produce a "memory" tensor. The decoder will reference this memory at every step.
- **`tgt_tokens = [SOS]`** — Starts generation with just the start-of-sentence token.
- **`model.decode(tgt, memory, src_mask, tgt_mask)`** — Runs the decoder on all tokens generated so far, using the encoder memory. The causal mask ensures each position only sees previous positions.
- **`logits.argmax(dim=-1).item()`** — Greedy decoding: picks the single most likely next token. Simple but fast.
- **`if next_token == EOS: break`** — Stops generating when the model outputs the end-of-sentence token.
- **The test loop** generates 20 random sequences, runs greedy decoding, and checks if the output matches the expected reverse-and-increment result.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Append this code to the training script (`train_transformer.py`)
2. Run the full script: `python train_transformer.py`

**Expected output:**
```
--- Translation Results ---
  ✓  [3, 7, 1, 5, 9, 2] → [3, 0, 6, 2, 8, 4]  (expected [3, 0, 6, 2, 8, 4])
  ...
Accuracy: 18/20 = 90%
```
(Accuracy should be 80-100% after 40 epochs.)

</details>

## Examining What the Model Learned

```python title="Inspecting attention patterns"
import matplotlib.pyplot as plt

def get_attention_weights(model, src_tokens, tgt_tokens):
    """Extract attention weights from the last encoder and decoder layers."""
    model.eval()
    src = torch.tensor([src_tokens])
    tgt = torch.tensor([tgt_tokens])

    src_mask = model.make_src_mask(src)
    tgt_mask = model.make_tgt_mask(tgt)

    # Run encoder
    x = model.pos_encoding(model.src_embedding(src) * math.sqrt(model.d_model))
    for layer in model.encoder_layers:
        x = layer(x, src_mask)
    memory = model.encoder_norm(x)

    # Run decoder and capture cross-attention
    x = model.pos_encoding(model.tgt_embedding(tgt) * math.sqrt(model.d_model))
    for layer in model.decoder_layers:
        # We'd need to modify DecoderBlock to return weights
        # For now, manually compute cross-attention weights
        normed = layer.norm1(x)
        x = x + layer.dropout1(layer.self_attn(normed, normed, normed, tgt_mask))
        normed = layer.norm2(x)

        # Get cross-attention scores
        B, T_q, _ = normed.shape
        T_k = memory.shape[1]
        h = layer.cross_attn.num_heads
        d_k = layer.cross_attn.head_dim

        Q = layer.cross_attn.q_proj(normed).view(B, T_q, h, d_k).transpose(1, 2)
        K = layer.cross_attn.k_proj(memory).view(B, T_k, h, d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / layer.cross_attn.scale
        weights = F.softmax(scores, dim=-1)  # (1, h, T_q, T_k)

        x = x + layer.dropout2(layer.cross_attn(normed, memory, memory, src_mask))
        normed = layer.norm3(x)
        x = x + layer.dropout3(layer.ffn(normed))

    return weights[0].detach()  # (h, T_q, T_k)

# Visualize for a specific example
digits = [3, 7, 1, 5, 9, 2]
expected = [(d + 1) % 10 for d in reversed(digits)]
src = [SOS] + [d + OFFSET for d in digits] + [EOS]
tgt = [SOS] + [d + OFFSET for d in expected]

weights = get_attention_weights(model, src, tgt)
src_labels = ["SOS"] + [str(d) for d in digits] + ["EOS"]
tgt_labels = ["SOS"] + [str(d) for d in expected]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for head_idx in range(4):
    ax = axes[head_idx]
    ax.imshow(weights[head_idx].numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(src_labels)))
    ax.set_yticks(range(len(tgt_labels)))
    ax.set_xticklabels(src_labels, fontsize=8)
    ax.set_yticklabels(tgt_labels, fontsize=8)
    ax.set_title(f"Head {head_idx + 1}")
    ax.set_xlabel("Source")
    if head_idx == 0:
        ax.set_ylabel("Target")

plt.suptitle("Cross-Attention Weights (Last Decoder Layer)")
plt.tight_layout()
plt.savefig("cross_attention_heads.png", dpi=150)
plt.show()
```

:::tip[Line-by-Line Walkthrough]
- **`get_attention_weights(model, src_tokens, tgt_tokens)`** — Manually runs the encoder and decoder forward pass while extracting the cross-attention weights from the last decoder layer.
- **`layer.cross_attn.q_proj(normed)` / `layer.cross_attn.k_proj(memory)`** — Manually computes the query and key projections so we can extract raw attention scores between decoder and encoder positions.
- **`F.softmax(scores, dim=-1)`** — Converts raw scores to attention weights (probabilities summing to 1 per row). Each row shows how much the decoder position attends to each encoder position.
- **`fig, axes = plt.subplots(1, 4, ...)`** — Creates a 1×4 grid of heatmaps, one for each attention head, so you can see what different heads have learned to focus on.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch matplotlib
```

**Steps:**
1. Append this visualization code to the training script
2. Run the full script: `python train_transformer.py`

**Expected output:** A figure with 4 heatmaps (one per attention head) saved as `cross_attention_heads.png`. For the reverse-and-increment task, at least one head should show an **anti-diagonal** pattern — the first output position attends to the last input position.

</details>

For the reversal + increment task, you should see an **anti-diagonal** pattern in at least one attention head — the decoder's first output position attends to the encoder's last input position, and so on.

---

## Exercises

:::tip[Exercise 1: Copy Task — beginner]

Train the Transformer on the simplest possible task: **copying** the input sequence. If the model can't learn to copy, there's a bug in your implementation. Verify 100% accuracy on sequences of length 6, then test on longer sequences (10, 15, 20) to see how well it generalizes.

<details>
<summary>Hints</summary>

1. The target is identical to the source
2. This should converge very quickly — if it doesn't, you have a bug
3. Check that your causal mask is correct

</details>

:::

:::tip[Exercise 2: Learning Rate Warmup Ablation — intermediate]

The original Transformer paper uses a specific learning rate schedule with warmup. Investigate its importance by training with different warmup durations and without warmup at all. Plot the training loss curves. At what point does removing warmup cause training to fail?

<details>
<summary>Hints</summary>

1. Train with warmup_steps = 0, 100, 400, 1000, 4000
2. Plot loss curves for all five on the same graph
3. Also try constant learning rates: 1e-3, 5e-4, 1e-4

</details>

:::

:::tip[Exercise 3: Scaling Laws — advanced]

Investigate how model size affects performance. Train Transformers of different sizes (varying `d_model`, `num_layers`, and `num_heads`) on the reverse-and-increment task with increasing sequence lengths. Plot the relationship between parameter count and accuracy. Do you observe scaling laws similar to those in LLM research?

<details>
<summary>Hints</summary>

1. Vary d_model (32, 64, 128, 256) while keeping num_heads proportional
2. Vary num_layers (1, 2, 4, 6, 8)
3. Test each model on sequences of length 6, 10, 15, 20
4. Plot parameter count vs. accuracy for each sequence length

</details>

:::

:::tip[Exercise 4: Real Translation Task — advanced]

Extend your Transformer to handle a real translation task. Use the Multi30k English-German dataset (available via `torchtext` or Hugging Face `datasets`). Build proper vocabularies, implement batching with padding, and evaluate using BLEU score. How does your from-scratch Transformer compare to the PyTorch built-in `nn.Transformer`?

<details>
<summary>Hints</summary>

1. Use torchtext or datasets library to load Multi30k
2. Build vocabularies for source and target languages
3. Use BPE tokenization for better OOV handling
4. Train for at least 20 epochs; use BLEU score for evaluation

</details>

:::

---

## Resources

- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** _(tutorial)_ by Harvard NLP — The gold-standard annotated implementation — follow along to verify your own.

- **[Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)** _(video)_ by Andrej Karpathy — Building a GPT (decoder-only Transformer) from scratch with thorough explanations.

- **[PyTorch nn.Transformer Docs](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)** _(tool)_ — PyTorch's built-in Transformer module — compare your implementation against it.

- **[minGPT](https://github.com/karpathy/minGPT)** _(tool)_ by Andrej Karpathy — Clean, minimal GPT implementation in PyTorch — excellent reference code.
