---
sidebar_position: 6
slug: nanogpt
title: "Pretraining a Small GPT (nanoGPT)"
---


# Pretraining a Small GPT (nanoGPT)

Theory is essential, but there's no substitute for training a language model yourself. In this lesson, we'll walk through Andrej Karpathy's **nanoGPT** — a clean, minimal implementation of GPT that you can train on a single GPU. By the end, you'll have trained a model that generates coherent text and understand every line of the training pipeline.

:::tip[Prerequisite]
Make sure you've worked through the GPT architecture lesson first. This lesson builds directly on that code and assumes you understand causal self-attention, the GPT block, and the autoregressive training objective.
:::

## Overview: What We'll Build

We'll implement a complete pre-training pipeline:

1. **Data preparation** — Download text, tokenize it, create train/val splits
2. **Model definition** — A GPT architecture (we'll reuse and refine the code from Lesson 2)
3. **Training loop** — Gradient accumulation, learning rate scheduling, logging
4. **Text generation** — Sample from the trained model
5. **Hyperparameter tuning** — Understand what to tweak and why

The approach mirrors Karpathy's nanoGPT but is structured for learning. We'll train on Shakespeare's complete works (~1MB of text) — small enough to train in minutes, large enough to produce recognizable English.

## Step 1: Data Preparation

```python title="Download and prepare the Shakespeare dataset"
import os
import requests
import numpy as np
import tiktoken

# Download Shakespeare
data_dir = "data/shakespeare"
os.makedirs(data_dir, exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filepath = os.path.join(data_dir, "input.txt")

if not os.path.exists(filepath):
    print("Downloading Shakespeare...")
    text = requests.get(url).text
    with open(filepath, 'w') as f:
        f.write(text)
else:
    with open(filepath, 'r') as f:
        text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(f"Sample: {text[:200]}")
print()

# Tokenize with tiktoken (GPT-2 tokenizer)
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
tokens = np.array(tokens, dtype=np.uint16)
print(f"Token count: {len(tokens):,}")
print(f"Vocabulary used: {len(set(tokens)):,} unique tokens")

# Train/val split (90/10)
split_idx = int(len(tokens) * 0.9)
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

train_tokens.tofile(os.path.join(data_dir, "train.bin"))
val_tokens.tofile(os.path.join(data_dir, "val.bin"))
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")
```

:::tip[Line-by-Line Walkthrough]
- **`os.makedirs(data_dir, exist_ok=True)`** — Creates the data directory if it doesn't already exist.
- **`text = requests.get(url).text`** — Downloads Shakespeare's complete works as raw text from GitHub.
- **`enc = tiktoken.get_encoding("gpt2")`** — Loads GPT-2's tokenizer. This splits text into subword tokens that the model will learn to predict.
- **`tokens = enc.encode(text)`** — Converts the entire text into a list of integer token IDs. Shakespeare's ~1MB of text becomes ~300K tokens.
- **`tokens = np.array(tokens, dtype=np.uint16)`** — Stores tokens as 16-bit unsigned integers (saves memory — token IDs for GPT-2 fit in 16 bits since vocab < 65535).
- **`split_idx = int(len(tokens) * 0.9)`** — Puts 90% of the data in training and 10% in validation.
- **`.tofile(...)`** — Saves the token arrays as raw binary files. This format is fast to load with `np.memmap` later.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy tiktoken requests
```

**Steps:**
1. Save the code to `prepare_data.py`
2. Run: `python prepare_data.py`

**Expected output:**
```
Downloading Shakespeare...
Dataset size: 1,115,394 characters
Sample: First Citizen:
Before we proceed any further...

Token count: 338,025
Vocabulary used: 4,743 unique tokens
Train tokens: 304,222
Val tokens: 33,803
```

</details>

### Character-Level vs BPE Tokenization

You have two choices for a small project like this:

| Approach | Vocab Size | Seq Length | Quality | Training Speed |
|----------|-----------|------------|---------|----------------|
| Character-level | ~65 | Long (~1M chars) | Lower per-token | Faster per step |
| BPE (GPT-2 tokenizer) | 50,257 | Shorter (~300K tokens) | Higher per-token | Slower per step |

For learning, character-level is simpler (no tokenizer dependency) and trains faster. For better text quality, BPE is superior. We'll use BPE here, but the code works with either — just change the vocabulary size and encoding.

```python title="Character-level alternative"
# If you prefer character-level tokenization:
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Character vocabulary: {vocab_size} chars")
print(f"Characters: {''.join(chars)}")

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda ids: ''.join([idx_to_char[i] for i in ids])

# Test round-trip
test = "Hello, World!"
assert decode(encode(test)) == test
```

:::tip[Line-by-Line Walkthrough]
- **`chars = sorted(set(text))`** — Extracts every unique character in the text and sorts them. This becomes your vocabulary (typically ~65 characters for English text).
- **`char_to_idx = {ch: i for i, ch in enumerate(chars)}`** — Creates a dictionary mapping each character to a unique integer (e.g., 'a' → 0, 'b' → 1, ...).
- **`encode = lambda s: [char_to_idx[c] for c in s]`** — Converts a string into a list of integers by looking up each character.
- **`decode = lambda ids: ''.join([idx_to_char[i] for i in ids])`** — Converts a list of integers back into a string.
- **`assert decode(encode(test)) == test`** — Verifies that encoding then decoding gives back the original text (a round-trip test).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
No extra packages needed — uses only Python builtins.

**Steps:**
1. This code requires the `text` variable from the data preparation step above.
2. Add it after loading the Shakespeare text.
3. Run: `python prepare_data.py`

**Expected output:**
```
Character vocabulary: 65 chars
Characters:  !"&'()*+,-./0123456789:;?ABCD...xyz
```
(No assertion error means the round-trip test passed.)

</details>

## Step 2: Data Loading

```python title="Efficient data loading for language modeling"
import torch
import numpy as np

class TextDataset:
    def __init__(self, data_path, context_length):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length - 1

    def get_batch(self, batch_size, device='cuda'):
        """Sample a random batch of (input, target) pairs."""
        ix = torch.randint(len(self.data) - self.context_length, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.context_length].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.context_length].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


# Usage
train_dataset = TextDataset("data/shakespeare/train.bin", context_length=256)
val_dataset = TextDataset("data/shakespeare/val.bin", context_length=256)

# Quick test
x, y = train_dataset.get_batch(4, device='cpu')
print(f"Input shape:  {x.shape}")   # (4, 256)
print(f"Target shape: {y.shape}")   # (4, 256)
print(f"Input[0,:5]:  {x[0,:5]}")
print(f"Target[0,:5]: {y[0,:5]}")   # shifted by 1
```

:::tip[Line-by-Line Walkthrough]
- **`self.data = np.memmap(data_path, dtype=np.uint16, mode='r')`** — Memory-maps the binary token file. The data stays on disk and is loaded into RAM on demand, which is essential for large datasets.
- **`ix = torch.randint(len(self.data) - self.context_length, (batch_size,))`** — Picks random starting positions in the dataset. Each position will produce one training example.
- **`self.data[i:i+self.context_length]`** — Grabs a contiguous chunk of tokens as the input sequence (x).
- **`self.data[i+1:i+1+self.context_length]`** — Grabs the *next* chunk shifted by one position as the target sequence (y). So if x = [The, cat, sat], then y = [cat, sat, on] — each target token is the next word after the input token.
- **`return x.to(device), y.to(device)`** — Moves the tensors to GPU (or CPU).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy
```
Requires the binary files `train.bin` and `val.bin` from the data preparation step.

**Steps:**
1. Run the data preparation code first to create `data/shakespeare/train.bin` and `val.bin`.
2. Add this code after the data prep or in the same script.
3. Run: `python prepare_data.py`

**Expected output:**
```
Input shape:  torch.Size([4, 256])
Target shape: torch.Size([4, 256])
Input[0,:5]:  tensor([  464,  3979,  2486,    25,   198])
Target[0,:5]: tensor([ 3979,  2486,    25,   198, 12050])
```
(Token IDs will vary. Notice that target is input shifted by 1.)

</details>

:::info[Why memmap?]
`np.memmap` maps the file directly into virtual memory without loading it all into RAM. This is essential for larger datasets — you can work with files much larger than your RAM. For our tiny Shakespeare dataset it doesn't matter, but it's a good habit.
:::

## Step 3: Model Architecture

We'll use the same GPT architecture from Lesson 2, but sized for our small experiment:

```python title="nanoGPT model configuration"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['d_model'] % config['n_heads'] == 0
        self.n_heads = config['n_heads']
        self.d_k = config['d_model'] // config['n_heads']

        self.qkv = nn.Linear(config['d_model'], 3 * config['d_model'])
        self.proj = nn.Linear(config['d_model'], config['d_model'])
        self.attn_drop = nn.Dropout(config['dropout'])
        self.proj_drop = nn.Dropout(config['dropout'])

        mask = torch.triu(
            torch.ones(config['context_length'], config['context_length']),
            diagonal=1
        ).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['d_model'], 4 * config['d_model']),
            nn.GELU(),
            nn.Linear(4 * config['d_model'], config['d_model']),
            nn.Dropout(config['dropout']),
        )
    def forward(self, x):
        return self.net(x)

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config['d_model'])
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_emb = nn.Embedding(config['context_length'], config['d_model'])
        self.drop = nn.Dropout(config['dropout'])
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # Scale residual projections
        for name, p in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config['n_layers']))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# Model configurations
configs = {
    'nano': {
        'vocab_size': 50257, 'd_model': 128, 'n_heads': 4,
        'n_layers': 4, 'context_length': 256, 'dropout': 0.1,
    },
    'small': {
        'vocab_size': 50257, 'd_model': 384, 'n_heads': 6,
        'n_layers': 6, 'context_length': 256, 'dropout': 0.1,
    },
    'medium': {
        'vocab_size': 50257, 'd_model': 768, 'n_heads': 12,
        'n_layers': 12, 'context_length': 1024, 'dropout': 0.1,
    },
}

for name, cfg in configs.items():
    model = NanoGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:8s}: {n_params/1e6:.1f}M parameters")
```

:::tip[Line-by-Line Walkthrough]
- **`self.qkv = nn.Linear(config['d_model'], 3 * config['d_model'])`** — Computes Q, K, V in a single matrix multiplication (more efficient than three separate ones).
- **`self.register_buffer("mask", mask)`** — Stores the causal mask as a non-trainable buffer that moves with the model to GPU automatically.
- **`self.head.weight = self.tok_emb.weight`** — Weight tying: reuses the token embedding matrix for the output projection, reducing parameters and improving learning.
- **`nn.init.normal_(p, std=0.02 / math.sqrt(2 * config['n_layers']))`** — Scales down the residual projection weights to prevent the residual stream from growing with depth. The `2 * n_layers` factor accounts for the two residual connections per block (attention + FFN).
- **`logits = logits[:, -1, :] / temperature`** — During generation, takes only the last position's predictions and divides by temperature to control randomness.
- **`logits[logits < v[:, [-1]]] = float('-inf')`** — Top-k filtering: keeps only the k highest-probability tokens, setting all others to −∞ so they get zero probability after softmax.
- **`configs = {'nano': ..., 'small': ..., 'medium': ...}`** — Three pre-defined model sizes for experimentation, from 0.8M to 85M parameters.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `nanogpt_model.py`
2. Run: `python nanogpt_model.py`

**Expected output:**
```
nano    : 0.8M parameters
small   : 10.6M parameters
medium  : 85.0M parameters
```
(This only creates the model architectures — no training yet.)

</details>

## Step 4: Training Loop

```python title="Complete training loop with all the essentials"
import torch
import time
import math
import tiktoken

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50, batch_size=32):
    """Estimate train and val loss over multiple batches."""
    model.eval()
    losses = {}
    for name, dataset in [('train', train_data), ('val', val_data)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = dataset.get_batch(batch_size)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[name] = total_loss / eval_iters
    model.train()
    return losses

def train_nanogpt():
    # --- Configuration ---
    config = {
        'vocab_size': 50257,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 6,
        'context_length': 256,
        'dropout': 0.1,
    }

    max_steps = 5000
    batch_size = 64
    grad_accum_steps = 4       # effective batch = 64 × 4 = 256 sequences
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = 200
    eval_interval = 250
    save_interval = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Data ---
    train_data = TextDataset("data/shakespeare/train.bin", config['context_length'])
    val_data = TextDataset("data/shakespeare/val.bin", config['context_length'])

    # --- Model ---
    model = NanoGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, betas=(0.9, 0.95),
        weight_decay=0.1, fused=(device == 'cuda')
    )

    # --- Tokenizer for generation samples ---
    enc = tiktoken.get_encoding("gpt2")

    # --- Training ---
    model.train()
    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(max_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_data.get_batch(batch_size, device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps  # scale for accumulation
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Timing
        dt = time.time() - t0
        tokens_per_sec = (batch_size * config['context_length'] *
                          grad_accum_steps * (step + 1)) / dt

        # Logging
        if step % 50 == 0:
            print(f"step {step:5d} | loss {total_loss:.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")

        # Evaluation
        if step > 0 and step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data,
                                   batch_size=batch_size)
            print(f"  → eval | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  → saved best model (val loss: {best_val_loss:.4f})")

            # Generate a sample
            prompt = enc.encode("ROMEO:")
            prompt_tensor = torch.tensor([prompt], device=device)
            generated = model.generate(prompt_tensor, max_new_tokens=200)
            text = enc.decode(generated[0].tolist())
            print(f"  → sample: {text[:300]}")
            print()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {(time.time() - t0)/60:.1f} minutes")


# Run training
train_nanogpt()
```

:::tip[Line-by-Line Walkthrough]
- **`get_lr(step, warmup_steps, max_steps, max_lr, min_lr)`** — Implements cosine learning rate scheduling with linear warmup. The LR starts at 0, ramps up linearly, then decays following a cosine curve.
- **`loss = loss / grad_accum_steps`** — Divides the loss by the number of accumulation steps so the final accumulated gradient has the correct magnitude (equivalent to a single large batch).
- **`with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):`** — Runs the forward pass in bf16 mixed precision for faster computation and lower memory usage.
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`** — Prevents training instability by capping the total gradient magnitude at 1.0.
- **`tokens_per_sec = (...)`** — Tracks throughput to ensure your GPU is being used efficiently. On an A100, you should see 100K+ tokens/second for small models.
- **`prompt = enc.encode("ROMEO:")`** — Generates a text sample during evaluation to qualitatively check if the model is learning Shakespeare's style.
- **`torch.save(model.state_dict(), "best_model.pt")`** — Saves the model weights when validation loss improves, so you always keep the best version.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy tiktoken
```
A CUDA-capable GPU is recommended but not required (CPU will be very slow).

**Steps:**
1. First run the data preparation code to create `data/shakespeare/train.bin` and `val.bin`.
2. Combine the TextDataset class, NanoGPT model, and this training loop into one script (or use imports).
3. Save to `train_nanogpt.py` and run: `python train_nanogpt.py`

**Expected output:**
```
Model: 10.6M parameters
step     0 | loss 10.8145 | lr 0.00e+00 | 45231 tok/s
step    50 | loss 6.2341 | lr 7.50e-05 | 52103 tok/s
...
step  5000 | loss 1.7823 | lr 3.00e-05 | 55000 tok/s
  → sample: ROMEO: What is the matter with the world?...
Training complete. Best val loss: 1.8234
Total time: 12.3 minutes
```
(On a GPU. CPU training will take hours.)

</details>

## Step 5: Understanding the Training Dynamics

### Learning Rate Schedule

The cosine schedule with warmup is standard for LLM training:

1. **Linear warmup** (steps 0–200): LR increases from 0 to `max_lr`. This prevents large, noisy gradients from destabilizing the model early in training.
2. **Cosine decay** (steps 200–5000): LR smoothly decreases to `min_lr`. The model does coarse learning first, then fine-tuning.

:::warning[Common Training Failures]
If your loss doesn't decrease:
- **Learning rate too high:** Loss oscillates or goes to NaN. Try reducing max_lr by 10×.
- **Learning rate too low:** Loss decreases very slowly. Try increasing max_lr by 3–10×.
- **Batch size too small:** Noisy gradients. Use gradient accumulation to increase effective batch size.
- **Model too large for data:** Overfits immediately (train loss drops, val loss increases). Use a smaller model or more data.
- **NaN loss:** Usually caused by fp16 overflow. Switch to bf16 or reduce learning rate.
:::

### Gradient Accumulation

When your GPU can't fit a large batch, gradient accumulation simulates it. Instead of one forward-backward pass with batch size 256, do 4 passes with batch size 64 and accumulate the gradients before stepping the optimizer.

The key subtlety: **you must divide the loss by `grad_accum_steps`** before calling `loss.backward()`. Otherwise, the accumulated gradient will be `grad_accum_steps × ` too large.

### What to Monitor

| Metric | What It Tells You |
|--------|-------------------|
| Training loss | Is the model learning? Should decrease smoothly. |
| Validation loss | Is it overfitting? Gap between train and val loss. |
| Learning rate | Is the schedule behaving correctly? |
| Tokens/second | Is your GPU being utilized efficiently? |
| Gradient norm | Are gradients healthy? Should be ~0.1–10. |
| Generated samples | Qualitative check — is the output improving? |

## Step 6: Generating Text

```python title="Text generation with different strategies"
import tiktoken
import torch

enc = tiktoken.get_encoding("gpt2")

def generate_text(model, prompt, max_tokens=300, temperature=0.8,
                  top_k=40, device='cuda'):
    """Generate text from a prompt string."""
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], device=device)

    with torch.no_grad():
        output = model.generate(x, max_new_tokens=max_tokens,
                                temperature=temperature, top_k=top_k)
    return enc.decode(output[0].tolist())


# Load the best model
model = NanoGPT(config).to(device)
model.load_state_dict(torch.load("best_model.pt"))

# Try different prompts
prompts = [
    "ROMEO:",
    "To be, or not to be,",
    "KING HENRY:",
    "Once upon a time",
]

for prompt in prompts:
    print(f"--- Prompt: '{prompt}' ---")
    text = generate_text(model, prompt, max_tokens=200)
    print(text)
    print()

# Compare temperature settings
prompt = "JULIET:"
for temp in [0.2, 0.5, 0.8, 1.0, 1.5]:
    text = generate_text(model, prompt, max_tokens=100, temperature=temp)
    print(f"Temperature {temp}: {text[:150]}...")
    print()
```

:::tip[Line-by-Line Walkthrough]
- **`model.eval()`** — Switches the model to evaluation mode, which disables dropout (we want deterministic predictions during generation).
- **`tokens = enc.encode(prompt)`** — Converts the text prompt to token IDs using the same tokenizer used during training.
- **`output = model.generate(x, max_new_tokens=max_tokens, ...)`** — Generates up to `max_tokens` new tokens, one at a time, by repeatedly predicting the next token and appending it.
- **`model.load_state_dict(torch.load("best_model.pt"))`** — Loads the best checkpoint saved during training.
- **The temperature loop** — Demonstrates how temperature affects generation: 0.2 produces safe/repetitive text, 0.8 gives a good balance, 1.5 produces creative but chaotic text.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch tiktoken
```
Requires a trained model checkpoint (`best_model.pt`) from the training step.

**Steps:**
1. Make sure you've trained the model and have `best_model.pt` in your working directory.
2. Save this code (along with the model definition and config) to `generate.py`.
3. Run: `python generate.py`

**Expected output:**
```
--- Prompt: 'ROMEO:' ---
ROMEO:
What is the matter with the world? I am
The man that doth not love the state of France...

Temperature 0.2: ROMEO: I am the the the...  (very repetitive)
Temperature 0.8: ROMEO: What say you, my lord...  (natural)
Temperature 1.5: ROMEO: Zounds! Fie upon't...  (creative but chaotic)
```

</details>

### How Temperature Affects Generation

| Temperature | Effect |
|-------------|--------|
| 0.1–0.3 | Very deterministic, repetitive, "safe" text |
| 0.5–0.8 | Good balance of coherence and variety |
| 1.0 | Matches the training distribution |
| 1.2–2.0 | Creative but increasingly incoherent |

Temperature works by dividing the logits before softmax: \( P(x_i) \propto \exp(z_i / T) \). Low temperature sharpens the distribution (high-probability tokens become even more likely). High temperature flattens it (all tokens become more equally likely).

## Step 7: Hyperparameter Tuning

The most impactful hyperparameters for a small GPT, in rough order of importance:

### 1. Learning Rate
The single most important hyperparameter. For small models (1–50M params), try `3e-4`. For medium models (50–500M), try `1e-4`. Always use a schedule (cosine with warmup).

### 2. Model Size (d_model, n_layers)
More parameters generally help, but only if you have enough data. For Shakespeare (~300K tokens), a 10M parameter model is plenty. Going larger will overfit.

### 3. Context Length
Longer context lets the model learn longer-range dependencies but increases memory quadratically. For Shakespeare, 256 tokens captures most of a speech or dialogue. 1024 would capture entire scenes.

### 4. Batch Size
Larger effective batch sizes (via gradient accumulation) give more stable gradients. For small datasets, 128–512 sequences per step is a good range.

### 5. Weight Decay
0.1 is standard. Weight decay prevents overfitting and interacts with the learning rate. Exclude bias and LayerNorm parameters from weight decay.

```python title="Proper weight decay configuration"
def configure_optimizer(model, lr, weight_decay=0.1):
    """Apply weight decay only to 2D weight tensors (not biases or LayerNorm)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Weight decay params: {n_decay:,} | No decay: {n_no_decay:,}")

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
```

:::tip[Line-by-Line Walkthrough]
- **`if param.ndim < 2 or 'ln' in name or 'bias' in name:`** — Identifies parameters that should *not* get weight decay: 1D parameters (biases, LayerNorm scales/shifts), and anything with "ln" in the name. Weight decay on these hurts performance.
- **`decay_params.append(param)` / `no_decay_params.append(param)`** — Separates parameters into two groups with different weight decay settings.
- **`param_groups = [{'params': decay_params, 'weight_decay': 0.1}, ...]`** — Creates two optimizer parameter groups: one with weight decay 0.1 (for weight matrices) and one with 0.0 (for biases and norms).
- **`betas=(0.9, 0.95)`** — AdamW momentum parameters. 0.9 for the first moment (gradient moving average) and 0.95 for the second moment (squared gradient moving average). The 0.95 value (instead of the default 0.999) causes the optimizer to forget old gradient information faster, which helps with non-stationary training data.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Use `configure_optimizer(model, lr=3e-4)` as a drop-in replacement for `torch.optim.AdamW(model.parameters(), ...)` in your training script.

**Expected output:**
```
Weight decay params: 10,340,352 | No decay: 4,992
```
(Shows how many parameters get weight decay vs not. Most parameters are in weight matrices.)

</details>

## What Good Training Looks Like

After training the small config (~10M params) on Shakespeare for 5000 steps, you should see:

- Training loss dropping from ~10 (random, since ln(50257) ≈ 10.8) to ~1.5–2.0
- Validation loss tracking close to training loss (not overfitting with this model size)
- Generated text that looks like Shakespeare — iambic pentameter, character names, dramatic dialogue — even if the content doesn't make sense

```
ROMEO:
What is the matter with the world? I am
The man that doth not love the state of France,
And yet I am not so; for I am not
A man of blood, and yet I would not speak.
```

The model captures Shakespeare's style — the meter, vocabulary, character structure, and dialogue formatting — even though the content is nonsensical. A larger model trained on more data would produce increasingly coherent text.

---

## Exercises

:::tip[Train on a Different Dataset — beginner]

Swap out Shakespeare for a different text dataset. Good options:
- A novel from Project Gutenberg
- Python code (e.g., from a GitHub repo)
- Wikipedia articles
- Song lyrics

Train the small config (10M params) and compare the generated text quality. How does the dataset affect what the model learns?

<div>
**Key observations you'll see:**
- **Code:** The model learns syntax, indentation, and variable naming conventions. Generated code looks syntactically plausible but is logically wrong.
- **Wikipedia:** More factual-sounding, with dates and proper nouns. Structure is encyclopedic.
- **Lyrics:** The model picks up rhyming patterns and verse structure.
- The model always learns the **form** of the data before the **content**.
<details>
<summary>Hints</summary>

1. Try Project Gutenberg books, song lyrics, or code
2. Adjust context_length based on the structure of your data
3. You may need to adjust the number of training steps

</details>

:::

:::tip[Implement Perplexity Evaluation — intermediate]

Implement a function that computes the perplexity of your trained model on the validation set. Track how perplexity changes during training by evaluating every 500 steps. Plot the learning curve (perplexity vs training step).

<div>
**Solution:**

```python
@torch.no_grad()
def compute_perplexity(model, dataset, batch_size=32, n_batches=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x, y = dataset.get_batch(batch_size)
        _, loss = model(x, y)
        total_loss += loss.item() * x.shape[0] * x.shape[1]
        total_tokens += x.shape[0] * x.shape[1]
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return perplexity
```

:::tip[Line-by-Line Walkthrough]
- **`@torch.no_grad()`** — Disables gradient tracking for the entire function since we're only evaluating, not training.
- **`total_loss += loss.item() * x.shape[0] * x.shape[1]`** — Accumulates the total loss weighted by the number of tokens in the batch (batch_size × sequence_length), so we get a proper per-token average.
- **`avg_loss = total_loss / total_tokens`** — Computes the average cross-entropy loss per token across all batches.
- **`perplexity = math.exp(avg_loss)`** — Converts average loss to perplexity. A loss of 3.0 means perplexity ~20 (the model is as uncertain as choosing from 20 equally likely options).
- **`model.train()`** — Switches back to training mode at the end so dropout and other training-time behaviors are re-enabled.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy
```
Requires the `TextDataset` class and a trained `NanoGPT` model from earlier in this lesson.

**Steps:**
1. Add `import math` at the top of your script.
2. Call: `ppl = compute_perplexity(model, val_dataset)`
3. Print: `print(f"Perplexity: {ppl:.1f}")`

**Expected output:**
```
Perplexity: 45.2
```
(Value depends on model size and training duration. Untrained models start at ~50,000; well-trained small models reach 20–100.)

</details>

Expect perplexity to drop from ~50000 (random) to 20–100 depending on dataset and model size.
<details>
<summary>Hints</summary>

1. Perplexity = exp(average cross-entropy loss)
2. Compute loss over the entire validation set
3. Lower perplexity = better model
4. Compare across model sizes and training steps

</details>

:::

:::tip[Add Flash Attention — advanced]

Replace the manual attention computation in `CausalSelfAttention` with PyTorch's `F.scaled_dot_product_attention`, which automatically uses Flash Attention when available. Benchmark the training speed (tokens/second) before and after. What speedup do you observe?

<div>
**Solution:**

```python
class CausalSelfAttentionFlash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.d_k = config['d_model'] // config['n_heads']
        self.qkv = nn.Linear(config['d_model'], 3 * config['d_model'])
        self.proj = nn.Linear(config['d_model'], config['d_model'])
        self.attn_drop = config['dropout']
        self.proj_drop = nn.Dropout(config['dropout'])

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))
```

:::tip[Line-by-Line Walkthrough]
- **`self.attn_drop = config['dropout']`** — Stores the dropout probability as a float (not a Dropout module) because `scaled_dot_product_attention` handles dropout internally.
- **`F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)`** — PyTorch's built-in attention that automatically selects Flash Attention or Memory-Efficient Attention depending on your GPU. `is_causal=True` applies the causal mask without needing to construct it explicitly.
- **`dropout_p=self.attn_drop if self.training else 0.0`** — Applies dropout during training but disables it during inference.
- Notice the **entire manual attention block** (score computation, masking, softmax, dropout) is replaced by this single function call — same result, much faster.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch>=2.0
```
Requires PyTorch 2.0+ for `F.scaled_dot_product_attention`. A CUDA GPU is needed for Flash Attention (CPU falls back to standard attention).

**Steps:**
1. Replace the `CausalSelfAttention` class in your nanoGPT code with `CausalSelfAttentionFlash`.
2. Run your training script as before.
3. Compare tokens/second before and after.

**Expected output:** Training runs 1.5–3× faster with the same loss curves. Memory usage drops significantly for longer sequences (e.g., context_length=1024).

</details>

Expect 1.5–3× speedup on modern GPUs, with memory usage dropping significantly for long sequences.
<details>
<summary>Hints</summary>

1. PyTorch 2.0+ has F.scaled_dot_product_attention
2. It automatically selects Flash Attention or Memory Efficient Attention
3. Drop-in replacement for manual attention computation
4. Benchmark the speedup with torch.cuda.Event timing

</details>

:::

---

## Resources

- **[nanoGPT Repository](https://github.com/karpathy/nanoGPT)** _(tool)_ by Andrej Karpathy — The cleanest, most educational GPT implementation. Study the code line by line.

- **[Let's Build GPT: From Scratch, in Code (Video)](https://www.youtube.com/watch?v=kCc8FmEb1nY)** _(video)_ by Andrej Karpathy — The 2-hour companion video where Karpathy builds the model from first principles.

- **[Let's Reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)** _(video)_ by Andrej Karpathy — A follow-up video on reproducing GPT-2 at full scale with modern training infrastructure.

- **[FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)** _(paper)_ by Dao et al. — The algorithm behind fast attention — essential for efficient Transformer training.

- **[PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)** _(tutorial)_ — Official docs for PyTorch's scaled_dot_product_attention with Flash Attention support.
