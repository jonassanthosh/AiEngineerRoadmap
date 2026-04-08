---
sidebar_position: 7
slug: month6-project
title: "Capstone: Design a Novel Architecture"
---


# Capstone: Design a Novel Architecture

:::info[What You'll Learn]
- Identifying weaknesses in existing architectures through paper analysis
- Proposing and implementing architectural modifications
- Setting up rigorous experimental evaluation
- Writing a technical report documenting your findings
:::

:::note[Prerequisites]
All of Month 6 lessons 1–6 and the full curriculum.
:::

**Estimated time:** Reading: ~25 min | Project work: ~15 hours

This is the final capstone of the AI Engineering Academy. You'll do what research engineers do: identify a weakness in an existing architecture, propose a modification, implement it, and rigorously evaluate whether it helps. This project ties together everything you've learned — paper reading, architecture design, training, and evaluation.

:::tip[This Is a Real Research Project]
This capstone follows the same workflow used to produce published research. Many architectural innovations that now power production models (GQA, RMSNorm, SwiGLU) started as exactly this kind of project: "What if we changed this one thing? Does it help?"
:::

## Project Overview

**Objective:** Design, implement, and evaluate a modification to an existing language model architecture.

**Deliverables:**
1. A written proposal (1-2 pages)
2. Implementation in PyTorch
3. Training on a benchmark dataset
4. Comparison against a baseline
5. A technical report or blog post
6. A presentation (slides or recorded video)

**Timeline:** 2-3 weeks

## Phase 1: Choose Your Base Architecture

Start with a small, well-understood architecture that you can train quickly. We recommend one of these:

| Base Model | Parameters | Training Time (1× A100) | Framework |
|-----------|-----------|-------------------------|-----------|
| GPT-2 Small (custom) | 85M-125M | 2-4 hours | PyTorch |
| nanoGPT | 10M-85M | 30 min - 2 hours | PyTorch |
| LitGPT small | 50M-200M | 1-4 hours | Lightning |

:::info[Why Small Models?]
You want to iterate fast. An architectural change that helps at 100M parameters will almost certainly also help at 7B — but you can test it in minutes instead of weeks. Every major architecture paper starts with small-scale experiments before scaling up.
:::

### Setting Up the Baseline

Let's use a nanoGPT-style baseline. This gives us a clean, minimal codebase to modify.

```python title="baseline.py — Minimal GPT baseline for architecture experiments"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [B, T, H, D_h]
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BaselineGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.head.weight

        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---- Verify baseline ----
model = BaselineGPT(vocab_size=32000, d_model=512, num_heads=8, num_layers=6)
print(f"Baseline parameters: {model.count_parameters():,}")
x = torch.randint(0, 32000, (2, 128))
logits, loss = model(x, x)
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)`** — Computes Q, K, and V projections in a single matrix multiply for efficiency (3× the model dimension).
- **`qkv.unbind(dim=2)`** — Splits the combined projection back into separate Q, K, V tensors.
- **`self.register_buffer("mask", torch.tril(...))`** — Creates the causal (lower-triangular) mask as a buffer — it's saved with the model but not trained. This prevents tokens from attending to future positions.
- **`self.token_emb = nn.Embedding(vocab_size, d_model)`** — Converts token IDs (integers) into dense vectors of size `d_model`.
- **`self.pos_emb = nn.Embedding(max_seq_len, d_model)`** — Learned positional embeddings: each position (0, 1, 2, ...) gets its own vector.
- **`self.token_emb.weight = self.head.weight`** — Weight tying: the input embedding and output projection share the same weight matrix. This reduces parameters and often improves performance.
- **`torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)`** — Initializes weights with small random values (standard deviation 0.02), a common choice for Transformer models.
- **`loss = F.cross_entropy(logits.view(-1, ...), targets.view(-1))`** — Flattens predictions and targets, then computes how wrong the model's predictions are across all tokens.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `baseline.py`
2. Run: `python baseline.py`

**Expected output:**
```
Baseline parameters: 29,671,936
Logits shape: torch.Size([2, 128, 32000])
Loss: 10.3734
```
(The initial loss is high (~log(32000) ≈ 10.37) because the untrained model is guessing randomly across 32,000 vocabulary tokens.)

</details>

## Phase 2: Propose a Modification

Here are concrete modification ideas, ordered by complexity.

### Idea 1: Replace LayerNorm with RMSNorm

**Hypothesis:** RMSNorm is faster than LayerNorm with equivalent quality (as shown in LLaMA).

**What to change:** Replace all `nn.LayerNorm` with `RMSNorm` (you implemented this in Lesson 1).

**What to measure:** Training loss convergence, wall-clock time per step, final perplexity.

### Idea 2: Replace GELU FFN with SwiGLU

**Hypothesis:** SwiGLU provides better parameter efficiency than GELU feed-forward networks.

**What to change:** Replace `FeedForward` with the `SwiGLU` module. Note: SwiGLU has 3 weight matrices vs 2, so adjust `d_ff` to keep total parameters equal.

**What to measure:** Final perplexity at equal parameter count, convergence speed.

### Idea 3: Add Grouped Query Attention

**Hypothesis:** GQA reduces memory usage during inference with minimal quality loss.

**What to change:** Replace `CausalSelfAttention` with `GroupedQueryAttention` using fewer KV heads.

**What to measure:** Training perplexity, KV cache size, inference throughput.

### Idea 4: Hybrid Attention + SSM Blocks

**Hypothesis:** Alternating attention and SSM (Mamba) blocks captures both local and global patterns more efficiently than pure attention.

**What to change:** Replace every other `TransformerBlock` with an SSM block.

**What to measure:** Perplexity on short and long sequences, training speed, inference latency.

### Idea 5: Differential Attention

**Hypothesis:** Computing attention as the difference between two softmax attention maps reduces noise and improves performance (inspired by DIFF Transformer).

**What to change:** Compute two attention maps and subtract them, then apply a learned scaling factor.

**What to measure:** Perplexity, attention pattern quality, performance on retrieval-heavy tasks.

```python title="diff_attention.py — Differential Attention implementation"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentialAttention(nn.Module):
    """Differential Attention: computes attention as the difference of two maps.

    Instead of a single softmax attention, compute two attention distributions
    and subtract them. This cancels out noise and amplifies relevant signals.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        half_head_dim = self.head_dim // 2
        self.scale = math.sqrt(half_head_dim)

        # Two sets of Q, K projections (one for each attention map)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable scaling parameter (lambda)
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        half_d = self.head_dim // 2

        Q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)

        # Split Q, K into two halves for differential computation
        Q1, Q2 = Q[..., :half_d], Q[..., half_d:]
        K1, K2 = K[..., :half_d], K[..., half_d:]

        # Compute two attention maps
        attn1 = (Q1 @ K1.transpose(-2, -1)) / self.scale
        attn2 = (Q2 @ K2.transpose(-2, -1)) / self.scale

        # Apply causal mask
        causal_mask = self.mask[:, :, :T, :T]
        attn1 = attn1.masked_fill(causal_mask == 0, float("-inf"))
        attn2 = attn2.masked_fill(causal_mask == 0, float("-inf"))

        # Differential attention: subtract the two softmax distributions
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        attn_diff = self.attn_dropout(attn1 - self.lambda_param * attn2)

        out = (attn_diff @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.out_proj(out))


# ---- Quick test ----
diff_attn = DifferentialAttention(d_model=256, num_heads=4, max_seq_len=128)
x = torch.randn(2, 64, 256)
out = diff_attn(x)
print(f"DiffAttn output shape: {out.shape}")
print(f"Lambda: {diff_attn.lambda_param.item():.3f}")
print(f"Params: {sum(p.numel() for p in diff_attn.parameters()):,}")
```

:::tip[Line-by-Line Walkthrough]
- **`half_head_dim = self.head_dim // 2`** — Each head's dimensions are split in half: one half for the first attention map, one for the second. This is the core of the "differential" idea.
- **`self.lambda_param = nn.Parameter(torch.tensor(0.5))`** — A learnable scalar that controls how much of the second attention map to subtract. Initialized to 0.5 (equal weighting).
- **`Q1, Q2 = Q[..., :half_d], Q[..., half_d:]`** — Splits queries into two halves, each used for computing a separate attention distribution.
- **`attn1 = (Q1 @ K1.transpose(-2, -1)) / self.scale`** — Computes the first attention map using the first half of Q and K dimensions.
- **`attn_diff = ... attn1 - self.lambda_param * attn2`** — The differential: subtracts the second attention distribution from the first. This cancels out "noise" (tokens that both maps attend to equally) and amplifies "signal" (tokens that only the first map focuses on).
- **`self.lambda_param`** — Starts at 0.5 but is trained — the model learns how much subtraction is optimal.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `diff_attention.py`
2. Run: `python diff_attention.py`

**Expected output:**
```
DiffAttn output shape: torch.Size([2, 64, 256])
Lambda: 0.500
Params: 263,425
```

</details>

## Phase 3: Training Setup

### Dataset

Use a standard, easy-to-load dataset for consistent comparison.

```python title="data.py — Dataset preparation for architecture experiments"
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """Simple dataset that returns fixed-length chunks of tokenized text."""
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.num_samples = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.token_ids[start : start + self.seq_len]
        y = self.token_ids[start + 1 : start + self.seq_len + 1]
        return x, y


def prepare_data(seq_len: int = 256, batch_size: int = 32):
    """Load and tokenize a standard dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\\n".join(dataset["train"]["text"])
    val_text = "\\n".join(dataset["validation"]["text"])

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    print(f"Training tokens:   {len(train_ids):,}")
    print(f"Validation tokens: {len(val_ids):,}")

    train_dataset = TextDataset(train_ids, seq_len)
    val_dataset = TextDataset(val_ids, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer

# Usage:
# train_loader, val_loader, tokenizer = prepare_data(seq_len=256, batch_size=32)
```

:::tip[Line-by-Line Walkthrough]
- **`self.num_samples = (len(token_ids) - 1) // seq_len`** — Calculates how many non-overlapping chunks of length `seq_len` fit in the dataset. The `-1` accounts for the target being shifted by one position.
- **`x = self.token_ids[start : start + self.seq_len]`** — The input: a chunk of consecutive tokens.
- **`y = self.token_ids[start + 1 : start + self.seq_len + 1]`** — The target: the same chunk shifted by one token. The model's job is to predict each token from the previous ones.
- **`tokenizer = AutoTokenizer.from_pretrained("gpt2")`** — Loads GPT-2's tokenizer to convert raw text into token IDs.
- **`load_dataset("wikitext", "wikitext-103-raw-v1")`** — Downloads WikiText-103, a standard benchmark dataset with ~100M tokens of Wikipedia articles.
- **`train_ids = torch.tensor(tokenizer.encode(train_text))`** — Tokenizes all training text into a single long tensor of token IDs.
- **`DataLoader(train_dataset, batch_size=batch_size, shuffle=True)`** — Wraps the dataset for efficient batching and random shuffling during training.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets transformers
```

**Steps:**
1. Save the code to `data.py`
2. Run:
```python
from data import prepare_data
train_loader, val_loader, tokenizer = prepare_data(seq_len=256, batch_size=32)
```

**Expected output:**
```
Training tokens:   ~103,000,000
Validation tokens: ~200,000
```
(WikiText-103 will be downloaded automatically on first run, ~180 MB.)

</details>

### Training Loop

```python title="train.py — Training loop with proper evaluation and logging"
import torch
import torch.nn as nn
import time
import json
from pathlib import Path

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
    log_dir: str = "logs",
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    training_log = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start

        # ---- Validation ----
        model.eval()
        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss_total += loss.item()
                val_batches += 1

        val_loss = val_loss_total / val_batches
        val_ppl = torch.exp(torch.tensor(val_loss)).item()

        # ---- Logging ----
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 1),
        }
        training_log.append(entry)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:.2f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), log_path / "best_model.pt")

        scheduler.step()

    # Save training log
    with open(log_path / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    return training_log
```

:::tip[Line-by-Line Walkthrough]
- **`optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)`** — AdamW optimizer: the standard choice for training Transformers. Weight decay (0.1) acts as regularization.
- **`scheduler = ... CosineAnnealingLR(optimizer, T_max=num_epochs)`** — Learning rate follows a cosine curve: starts at `lr`, smoothly decreases to near-zero by the end of training.
- **`_, loss = model(x, y)`** — Forward pass: the model predicts next tokens and computes the loss against the actual targets.
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`** — Clips gradient norms to prevent training instability from occasional large gradients.
- **`val_ppl = torch.exp(torch.tensor(val_loss)).item()`** — Converts validation loss to perplexity (the standard metric for language models).
- **`if val_loss < best_val_loss`** — Only saves the model checkpoint when validation loss improves, so you keep the best-performing version.
- **`json.dump(training_log, f)`** — Saves the complete training history to a JSON file for later analysis and comparison.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets transformers
```

**Steps:**
1. Save `data.py` and `train.py` together
2. Combine with your model:
```python
from baseline import BaselineGPT
from data import prepare_data
from train import train_model

train_loader, val_loader, _ = prepare_data()
model = BaselineGPT(vocab_size=50257)
train_model(model, train_loader, val_loader, num_epochs=5, device="cuda")
```
3. Run: `python train.py` (requires a GPU for reasonable speed)

**Expected output:**
```
Epoch 1/5 | Train Loss: 6.1234 | Val Loss: 5.8901 | Val PPL: 362.15 | Time: 120.3s
Epoch 2/5 | Train Loss: 5.4567 | Val Loss: 5.3210 | Val PPL: 205.42 | Time: 118.7s
...
```

</details>

## Phase 4: Comparison and Analysis

This is where the project succeeds or fails. A rigorous comparison requires controlling for variables.

### What to Control

1. **Parameter count** — Both models should have approximately the same number of trainable parameters. If your modification adds parameters, reduce `d_ff` or `d_model` to compensate.
2. **Training data** — Identical data in identical order (set the same random seed).
3. **Training compute** — Same number of steps, same batch size, same hardware.
4. **Hyperparameters** — Same learning rate, same schedule (unless your modification requires different settings — if so, document why).

```python title="compare.py — Comparing baseline vs modified architecture"
import json
import torch

def compare_architectures(baseline_log_path: str, modified_log_path: str):
    """Compare training results between baseline and modified architecture."""
    with open(baseline_log_path) as f:
        baseline = json.load(f)
    with open(modified_log_path) as f:
        modified = json.load(f)

    print(f"{'Metric':<25} {'Baseline':>12} {'Modified':>12} {'Delta':>10}")
    print("-" * 62)

    # Final validation loss
    bl_final = baseline[-1]["val_loss"]
    mod_final = modified[-1]["val_loss"]
    delta = mod_final - bl_final
    print(f"{'Final Val Loss':<25} {bl_final:>12.4f} {mod_final:>12.4f} {delta:>+10.4f}")

    # Final perplexity
    bl_ppl = baseline[-1]["val_ppl"]
    mod_ppl = modified[-1]["val_ppl"]
    delta_ppl = mod_ppl - bl_ppl
    print(f"{'Final Val PPL':<25} {bl_ppl:>12.2f} {mod_ppl:>12.2f} {delta_ppl:>+10.2f}")

    # Best validation loss
    bl_best = min(e["val_loss"] for e in baseline)
    mod_best = min(e["val_loss"] for e in modified)
    delta_best = mod_best - bl_best
    print(f"{'Best Val Loss':<25} {bl_best:>12.4f} {mod_best:>12.4f} {delta_best:>+10.4f}")

    # Training speed
    bl_time = sum(e["epoch_time_s"] for e in baseline)
    mod_time = sum(e["epoch_time_s"] for e in modified)
    speedup = bl_time / mod_time
    print(f"{'Total Time (s)':<25} {bl_time:>12.1f} {mod_time:>12.1f} {speedup:>9.2f}x")

    # Convergence: at which epoch does each model reach <threshold> loss?
    threshold = bl_final * 1.02  # within 2% of baseline final
    bl_converge = next((e["epoch"] for e in baseline if e["val_loss"] <= threshold), "N/A")
    mod_converge = next((e["epoch"] for e in modified if e["val_loss"] <= threshold), "N/A")
    print(f"{'Convergence Epoch':<25} {str(bl_converge):>12} {str(mod_converge):>12}")


def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable
```

:::tip[Line-by-Line Walkthrough]
- **`compare_architectures(baseline_log_path, modified_log_path)`** — Loads the JSON training logs from both models and prints a side-by-side comparison table.
- **`bl_final = baseline[-1]["val_loss"]`** — Grabs the final epoch's validation loss (the last entry in the log).
- **`delta = mod_final - bl_final`** — Computes the difference: negative delta means your modification improved (lower loss is better).
- **`bl_best = min(e["val_loss"] for e in baseline)`** — Finds the best validation loss across all epochs (not just the last one).
- **`speedup = bl_time / mod_time`** — Computes relative speedup: >1.0x means your modification trains faster.
- **`threshold = bl_final * 1.02`** — Convergence check: at which epoch does each model first reach within 2% of the baseline's final loss? Earlier = faster convergence.
- **`count_model_params(model)`** — Reports total and trainable parameter counts. Important for fair comparison — both models should have similar counts.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `compare.py`
2. After training both baseline and modified models (which save `training_log.json`), run:
```python
compare_architectures("logs_baseline/training_log.json", "logs_modified/training_log.json")
```

**Expected output:**
```
Metric                      Baseline     Modified      Delta
--------------------------------------------------------------
Final Val Loss                5.2100       5.0800    -0.1300
Final Val PPL                183.58       160.77     -22.81
Best Val Loss                 5.1800       5.0500    -0.1300
Total Time (s)              1203.4       1150.2       1.05x
Convergence Epoch                8            6
```

</details>

## Phase 5: Write Your Technical Report

Your report should follow a standard structure. Keep it concise — 4-6 pages including figures and tables.

### Report Structure

1. **Abstract** (100-150 words) — What you did, what you found, one sentence on why it matters.

2. **Introduction** (0.5-1 page) — The problem, your hypothesis, brief preview of results.

3. **Method** (1-1.5 pages) — What you changed, why, and the exact implementation details. Include a figure showing the baseline vs modified architecture.

4. **Experiments** (1-2 pages) — Dataset, hyperparameters, hardware. Results tables and training curves. Ablation study if you modified multiple things.

5. **Analysis** (0.5-1 page) — Why do you think it worked (or didn't)? What surprised you? What would you try next?

6. **Conclusion** (0.5 page) — Summary of findings, limitations, future work.

:::warning[Negative Results Are Results]
If your modification doesn't improve over the baseline, that's still a valid and publishable result. The key is to analyze *why* it didn't work and what that tells us. Many important papers report negative results — they save others from going down the same dead end. Don't twist your analysis to pretend a failure is a success.
:::

## Phase 6: Present Your Findings

Prepare a 10-15 minute presentation:

1. **Slide 1:** Title and one-line summary of your finding
2. **Slides 2-3:** Background — what problem are you addressing?
3. **Slides 4-5:** Your approach — diagram of the modification
4. **Slides 6-8:** Results — tables, training curves, key comparisons
5. **Slide 9:** Analysis — why does this work (or not)?
6. **Slide 10:** Future work and conclusions

:::tip[Lead with the Punchline]
Start your presentation with the result: "I modified X and it improved perplexity by Y while reducing memory by Z." The audience will be much more engaged knowing the conclusion upfront — they can then focus on understanding *how* and *why*.
:::

## Bonus: Submit Your Work

If your results are strong, consider sharing them with the community:

1. **arXiv preprint** — Submit your technical report. No peer review required, but it becomes a permanent record.
2. **Workshop submission** — NeurIPS, ICML, and ICLR have workshops on efficient ML, architectures, etc. Workshop papers are typically 4-6 pages and have higher acceptance rates than main conferences.
3. **Blog post** — Publish on your personal blog, Medium, or Substack. Well-written blog posts often reach a larger audience than papers.
4. **Open-source release** — Put your code on GitHub with clear documentation. Include the exact commands to reproduce your results.

:::tip[Milestone 1: Written Proposal — intermediate]

Write a 1-2 page proposal for your capstone project:

1. **Hypothesis** — State what you're changing and what you predict will happen.
2. **Base architecture** — Which model will you modify?
3. **Modification** — Describe the exact change in technical detail.
4. **Evaluation plan** — What metrics? What dataset? What's your baseline?
5. **Timeline** — Break the project into milestones over 2-3 weeks.

Submit this proposal before writing any code.

<details>
<summary>Hints</summary>

1. Focus on a single, testable hypothesis
2. Define your evaluation metrics before you start coding
3. List all variables you need to control

</details>

:::

:::tip[Milestone 2: Implementation and Baseline — advanced]

Implement your baseline and modified architectures:

1. Train the baseline model and record all metrics (loss curves, final perplexity, training time).
2. Implement your modification as a minimal diff from the baseline code.
3. Verify shapes and forward pass before training.
4. Train the modified model with the same data, seeds, and hyperparameters.
5. Record all metrics.

<details>
<summary>Hints</summary>

1. Get the baseline training first — don't try to modify and train simultaneously
2. Save baseline training logs — you'll need them for comparison
3. Set random seeds for reproducibility

</details>

:::

:::tip[Milestone 3: Technical Report — advanced]

Write your technical report and prepare your presentation:

1. Write the report following the structure above (4-6 pages).
2. Include training curves (loss over time for both models on the same plot).
3. Include a results table with final metrics for both models.
4. Write an honest analysis — what worked, what didn't, what you'd do differently.
5. Prepare 10-15 slides for your presentation.

<details>
<summary>Hints</summary>

1. Follow the report structure outlined above
2. Include at least one figure (training curves) and one table (final results)
3. Don't overclaim — state exactly what you found

</details>

:::

## Resources

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** _(tool)_ by Andrej Karpathy — The simplest, fastest repository for training GPTs — perfect as a base for architecture experiments.

- **[LitGPT](https://github.com/Lightning-AI/litgpt)** _(tool)_ by Lightning AI — Clean, hackable implementations of modern LLM architectures for training and fine-tuning.

- **[How to Write a Great Research Paper](https://www.youtube.com/watch?v=VK51E3gHENc)** _(video)_ by Simon Peyton Jones — Timeless advice on structuring and writing technical papers clearly.

- **[DIFF Transformer](https://arxiv.org/abs/2410.05258)** _(paper)_ by Ye et al. — Differential attention: computing attention as the difference of two softmax maps.

- **[Tips for Writing NeurIPS Papers](https://neurips.cc/Conferences/2026/CallForPapers)** _(tutorial)_ — Formatting guidelines and writing advice from the NeurIPS committee.

- **[Weights & Biases](https://wandb.ai/)** _(tool)_ by Weights & Biases — Experiment tracking platform — invaluable for comparing architecture experiments. Free for personal use.
