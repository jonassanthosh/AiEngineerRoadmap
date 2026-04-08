---
sidebar_position: 7
slug: month4-project
title: "Project: Train a Language Model"
---


# Project: Train a Language Model

:::info[What You'll Learn]
- Building a complete language model training pipeline
- Data collection and preparation for pretraining
- Training a small GPT model (5–50M parameters)
- Generating and evaluating model outputs
:::

:::note[Prerequisites]
All of Month 4 lessons 1–6.
:::

**Estimated time:** Reading: ~30 min | Project work: ~10 hours

This is the Month 4 capstone project. You'll build a complete language model training pipeline from scratch — data collection, tokenization, model definition, training, evaluation, and text generation. Everything you've learned this month comes together here.

:::tip[Project Scope]
You'll train a small GPT model (5–50M parameters) on a dataset of your choice. The goal isn't to build something production-grade — it's to deeply understand every stage of the pipeline by building it yourself. By the end, you'll have a model that generates coherent, domain-specific text.
:::

## Project Overview

### Requirements

1. **Choose a dataset** — at least 1MB of text, preferably 5–50MB
2. **Build a tokenizer** or adapt an existing one
3. **Define a GPT model** using the architecture from this month's lessons
4. **Train the model** with proper learning rate scheduling, gradient accumulation, and logging
5. **Evaluate** using perplexity on a held-out set
6. **Generate text** and analyze the quality
7. **Write a brief report** documenting your choices, results, and observations

### Suggested Datasets

| Dataset | Size | What You'll Learn |
|---------|------|-------------------|
| Shakespeare | ~1MB | Dialogue structure, poetic language |
| Python source code | 5–50MB | Syntax, indentation, variable naming |
| Recipes | 5–20MB | Structured text, ingredient lists, instructions |
| Wikipedia articles | 10–100MB | Factual prose, encyclopedic style |
| Song lyrics | 5–20MB | Rhyming, verse structure, emotional language |
| Legal documents | 10–50MB | Formal language, clause structure |
| Academic papers (abstracts) | 5–20MB | Scientific writing, citation patterns |

:::info[Dataset Size vs Model Size]
Rule of thumb: you want at least 20 tokens per parameter for compute-optimal training (Chinchilla). For a 10M parameter model, that's 200M tokens (~800MB of text). But for a learning project with a small model, even 1MB of text will produce interesting results — the model will overfit, but you'll still see it learn the style and structure of your data.
:::

## Step 1: Data Collection and Preparation

```python title="Data collection pipeline"
import os
import requests
import glob
import re

def download_gutenberg(book_id, save_dir="data/project"):
    """Download a book from Project Gutenberg."""
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    filepath = os.path.join(save_dir, f"book_{book_id}.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"Downloaded book {book_id}: {len(response.text):,} chars")
    return filepath

def collect_code_dataset(repo_dir, extensions=('.py',)):
    """Collect source code files from a local repository."""
    texts = []
    for ext in extensions:
        for filepath in glob.glob(f"{repo_dir}/**/*{ext}", recursive=True):
            if 'venv' in filepath or 'node_modules' in filepath:
                continue
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read()
                if len(content) > 100:
                    texts.append(f"# FILE: {os.path.basename(filepath)}\\n{content}")
    return "\\n\\n".join(texts)

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\\r\\n', '\\n', text)       # normalize line endings
    text = re.sub(r'\\n{3,}', '\\n\\n', text)   # collapse multiple blank lines
    text = re.sub(r'[ \\t]+$', '', text, flags=re.MULTILINE)  # trailing whitespace
    return text.strip()

def prepare_dataset(text, output_dir, val_fraction=0.1):
    """Tokenize text and create train/val binary files."""
    import numpy as np
    import tiktoken

    os.makedirs(output_dir, exist_ok=True)

    # Save raw text for reference
    with open(os.path.join(output_dir, "raw.txt"), 'w') as f:
        f.write(text)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)

    split = int(len(tokens) * (1 - val_fraction))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_tokens.tofile(os.path.join(output_dir, "train.bin"))
    val_tokens.tofile(os.path.join(output_dir, "val.bin"))

    stats = {
        'total_chars': len(text),
        'total_tokens': len(tokens),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'vocab_used': len(set(tokens)),
        'chars_per_token': len(text) / len(tokens),
    }

    print("Dataset statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:,}")

    return stats


# Example: collect Shakespeare
book_ids = [1513, 1514, 1515, 1516, 1517]  # Various Shakespeare plays
texts = []
for bid in book_ids:
    path = download_gutenberg(bid)
    with open(path) as f:
        texts.append(f.read())

full_text = clean_text("\\n\\n".join(texts))
stats = prepare_dataset(full_text, "data/project/processed")
```

:::tip[Line-by-Line Walkthrough]
- **`download_gutenberg(book_id, save_dir="data/project")`** — Downloads a free book from Project Gutenberg by its numeric ID and saves it as a text file.
- **`collect_code_dataset(repo_dir, extensions=('.py',))`** — Alternatively, collects Python source files from a local repository, skipping virtual environments and node_modules.
- **`re.sub(r'\\n{3,}', '\\n\\n', text)`** — Collapses three or more consecutive newlines into two, cleaning up excessive whitespace.
- **`enc = tiktoken.get_encoding("gpt2")`** — Uses GPT-2's tokenizer to convert text into token IDs for training.
- **`tokens = np.array(tokens, dtype=np.uint16)`** — Stores token IDs as 16-bit integers (saves disk space and memory).
- **`train_tokens.tofile(...)`** — Writes the token arrays as raw binary files for fast memory-mapped loading during training.
- **`'chars_per_token': len(text) / len(tokens)`** — Computes the compression ratio — higher means the tokenizer is more efficient on this text.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy tiktoken requests
```

**Steps:**
1. Save the code to `collect_data.py`
2. Run: `python collect_data.py`
3. This downloads Shakespeare plays from Project Gutenberg (requires internet).

**Expected output:**
```
Downloaded book 1513: 184,234 chars
Downloaded book 1514: 123,456 chars
...
Dataset statistics:
  total_chars: 623,456
  total_tokens: 167,890
  train_tokens: 151,101
  val_tokens: 16,789
  vocab_used: 5,234
  chars_per_token: 3.71
```

</details>

## Step 2: Tokenizer Selection

For this project, you have three options:

### Option A: Use GPT-2's Tokenizer (Recommended)
The simplest approach. GPT-2's tokenizer (via tiktoken) is well-tested and produces good subword splits for English text.

### Option B: Train Your Own BPE Tokenizer
More work, but educational. Use the HuggingFace `tokenizers` library to train a custom BPE tokenizer on your dataset. This is especially worthwhile if your data is domain-specific (code, recipes, legal text).

### Option C: Character-Level
The simplest possible tokenizer. Good for learning, but produces longer sequences and lower-quality per-token predictions.

```python title="Option B: Training a custom BPE tokenizer"
from tokenizers import ByteLevelBPETokenizer

def train_custom_tokenizer(text_file, vocab_size=8192, save_dir="tokenizer"):
    """Train a BPE tokenizer from scratch."""
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[text_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        show_progress=True,
    )

    tokenizer.save_model(save_dir)
    print(f"Tokenizer saved to {save_dir}/")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    test_text = "To be, or not to be, that is the question."
    output = tokenizer.encode(test_text)
    print(f"Test: '{test_text}'")
    print(f"Tokens: {output.tokens}")
    print(f"Count: {len(output.tokens)}")

    return tokenizer

tokenizer = train_custom_tokenizer(
    "data/project/processed/raw.txt",
    vocab_size=8192
)
```

:::tip[Line-by-Line Walkthrough]
- **`ByteLevelBPETokenizer()`** — Creates a byte-level BPE tokenizer that starts with 256 byte tokens and learns merges. This can handle any input (including emojis and non-English text).
- **`tokenizer.train(files=[text_file], vocab_size=vocab_size, min_frequency=2, ...)`** — Learns BPE merge rules from your text file. `min_frequency=2` means a pair must appear at least twice to be merged. `special_tokens` reserves IDs for padding, start, end, and unknown tokens.
- **`tokenizer.save_model(save_dir)`** — Saves the vocabulary and merge rules so you can reload the tokenizer later without retraining.
- **`tokenizer.encode(test_text)`** — Tests the trained tokenizer on a sample sentence to verify it works correctly.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tokenizers
```

**Steps:**
1. First run the data collection pipeline to create `data/project/processed/raw.txt`.
2. Save this code to `train_tokenizer.py`
3. Run: `python train_tokenizer.py`

**Expected output:**
```
Tokenizer saved to tokenizer/
Vocabulary size: 8192
Test: 'To be, or not to be, that is the question.'
Tokens: ['To', 'Ġbe', ',', 'Ġor', 'Ġnot', 'Ġto', 'Ġbe', ',', 'Ġthat', 'Ġis', 'Ġthe', 'Ġquestion', '.']
Count: 13
```
(`Ġ` represents a space character in byte-level BPE.)

</details>

## Step 3: Model Definition

Use the `NanoGPT` architecture from Lesson 6. Choose a configuration that matches your dataset size and available GPU memory.

```python title="Model configuration guide"
import torch

# Determine available compute
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
else:
    print("No GPU — training will be slow but still possible!")
    gpu_mem = 0

def suggest_config(n_tokens, gpu_mem_gb):
    """Suggest model configuration based on dataset and GPU."""
    configs = {
        'tiny': {
            'desc': 'Quick experiments, CPU-friendly',
            'd_model': 128, 'n_heads': 4, 'n_layers': 4,
            'context_length': 128, 'batch_size': 32,
            'params_approx': '~1M',
        },
        'small': {
            'desc': 'Good for 1-5MB datasets',
            'd_model': 256, 'n_heads': 8, 'n_layers': 6,
            'context_length': 256, 'batch_size': 64,
            'params_approx': '~8M',
        },
        'medium': {
            'desc': 'Good for 5-50MB datasets',
            'd_model': 384, 'n_heads': 6, 'n_layers': 6,
            'context_length': 256, 'batch_size': 64,
            'params_approx': '~20M',
        },
        'large': {
            'desc': 'Needs 8GB+ GPU, 10-100MB dataset',
            'd_model': 768, 'n_heads': 12, 'n_layers': 12,
            'context_length': 512, 'batch_size': 32,
            'params_approx': '~85M',
        },
    }

    print(f"Dataset: {n_tokens:,} tokens | GPU: {gpu_mem_gb:.0f} GB")
    print()

    if gpu_mem_gb < 4:
        recommended = 'tiny'
    elif gpu_mem_gb < 8:
        recommended = 'small'
    elif gpu_mem_gb < 16:
        recommended = 'medium'
    else:
        recommended = 'large'

    for name, cfg in configs.items():
        marker = " ← RECOMMENDED" if name == recommended else ""
        print(f"  {name:8s} ({cfg['params_approx']:>5s}): {cfg['desc']}{marker}")

    return configs[recommended]

config = suggest_config(n_tokens=300_000, gpu_mem_gb=gpu_mem)
```

:::tip[Line-by-Line Walkthrough]
- **`torch.cuda.get_device_name(0)`** — Gets the name of your GPU (e.g., "NVIDIA A100").
- **`torch.cuda.get_device_properties(0).total_memory / 1e9`** — Gets total GPU memory in GB, used to determine what model size will fit.
- **`suggest_config(n_tokens, gpu_mem_gb)`** — Recommends a model configuration based on your dataset size and GPU memory. Tiny (1M params) for CPU/small GPU, up to large (85M params) for 16GB+ GPUs.
- **The configs dict** — Pre-built configurations with balanced hyperparameters. Each specifies d_model (hidden size), n_heads (attention heads), n_layers (depth), context_length, and batch_size.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `check_gpu.py`
2. Run: `python check_gpu.py`

**Expected output (with GPU):**
```
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Dataset: 300,000 tokens | GPU: 24 GB

  tiny     (  ~1M): Quick experiments, CPU-friendly
  small    (  ~8M): Good for 1-5MB datasets
  medium   ( ~20M): Good for 5-50MB datasets ← RECOMMENDED
  large    ( ~85M): Needs 8GB+ GPU, 10-100MB dataset
```

**Expected output (without GPU):**
```
No GPU — training will be slow but still possible!
```

</details>

## Step 4: Training

```python title="Full training script"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import os

class TrainingLogger:
    """Log training metrics to a JSON file for later analysis."""
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(
            log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.entries = []

    def log(self, step, **metrics):
        entry = {'step': step, 'timestamp': time.time(), **metrics}
        self.entries.append(entry)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\\n')

    def plot(self):
        import matplotlib.pyplot as plt
        steps = [e['step'] for e in self.entries if 'train_loss' in e]
        train_loss = [e['train_loss'] for e in self.entries if 'train_loss' in e]
        val_entries = [e for e in self.entries if 'val_loss' in e]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, train_loss, label='Train Loss', alpha=0.7)
        if val_entries:
            ax.plot([e['step'] for e in val_entries],
                    [e['val_loss'] for e in val_entries],
                    label='Val Loss', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig('training_curve.png', dpi=150)
        plt.show()


def train_project_model(config, data_dir, max_steps=5000,
                        device='cuda', save_dir='checkpoints'):
    """Complete training pipeline for the capstone project."""
    os.makedirs(save_dir, exist_ok=True)
    logger = TrainingLogger()

    # Data
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'),
                           dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'),
                         dtype=np.uint16, mode='r')

    ctx_len = config['context_length']
    batch_size = config.get('batch_size', 64)
    grad_accum = config.get('grad_accum_steps', 4)

    def get_batch(data, batch_size):
        ix = torch.randint(len(data) - ctx_len, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+ctx_len].astype(np.int64))
                         for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+ctx_len].astype(np.int64))
                         for i in ix])
        return x.to(device), y.to(device)

    # Model
    model = NanoGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M parameters")

    # Optimizer with proper weight decay
    decay_params = [p for n, p in model.named_parameters()
                    if p.ndim >= 2 and 'ln' not in n]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.ndim < 2 or 'ln' in n]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.get('max_lr', 3e-4), betas=(0.9, 0.95))

    # Learning rate schedule
    max_lr = config.get('max_lr', 3e-4)
    min_lr = max_lr / 10
    warmup_steps = max_steps // 20

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))

    # Training
    model.train()
    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log every step
        logger.log(step, train_loss=total_loss, lr=lr)

        # Print progress
        if step % 100 == 0:
            dt = time.time() - t0
            tps = batch_size * ctx_len * grad_accum * (step + 1) / dt
            print(f"step {step:5d}/{max_steps} | loss {total_loss:.4f} | "
                  f"lr {lr:.2e} | {tps:.0f} tok/s")

        # Evaluate periodically
        if step > 0 and step % 500 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(50):
                    x, y = get_batch(val_data, batch_size)
                    _, vloss = model(x, y)
                    val_losses.append(vloss.item())
            val_loss = sum(val_losses) / len(val_losses)
            perplexity = math.exp(val_loss)

            logger.log(step, val_loss=val_loss, perplexity=perplexity)
            print(f"  eval | val_loss: {val_loss:.4f} | "
                  f"perplexity: {perplexity:.1f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'config': config,
                    'step': step,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best.pt'))
                print(f"  saved best model (val_loss: {best_val_loss:.4f})")

            model.train()

    total_time = time.time() - t0
    print(f"\\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {math.exp(best_val_loss):.1f}")

    logger.plot()
    return model, logger
```

:::tip[Line-by-Line Walkthrough]
- **`TrainingLogger`** — A simple class that writes training metrics (loss, learning rate) to a JSONL file after each step, and can plot the learning curve at the end.
- **`decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'ln' not in n]`** — Separates weight matrices (which get weight decay) from biases and LayerNorm parameters (which don't).
- **`with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):`** — Runs the forward pass in bf16 for 2× speed and half the memory.
- **`loss = loss / grad_accum`** — Scales the loss before accumulating gradients, so the final gradient is correct for the effective batch size.
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`** — Gradient clipping to prevent training explosions.
- **`perplexity = math.exp(val_loss)`** — Converts cross-entropy loss to perplexity for a more interpretable metric.
- **`torch.save({'model': model.state_dict(), 'config': config, ...}, ...)`** — Saves a checkpoint with the model weights, configuration, step number, and validation loss — everything needed to resume training or use the model later.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy tiktoken matplotlib
```
A CUDA GPU is strongly recommended.

**Steps:**
1. Combine the data preparation, model definition, and this training script into one file.
2. Make sure `data/project/processed/train.bin` and `val.bin` exist from the data pipeline.
3. Run: `python train_project.py`

**Expected output:**
```
Model: 20.1M parameters
step     0/5000 | loss 10.8145 | lr 0.00e+00 | 45000 tok/s
step   100/5000 | loss 5.2341 | lr 1.50e-04 | 52000 tok/s
...
  eval | val_loss: 2.3456 | perplexity: 10.4
  saved best model (val_loss: 2.3456)
...
Training complete in 15.2 minutes
Best validation loss: 1.8234
Best perplexity: 6.2
```
A `training_curve.png` file will be generated showing loss vs steps.

</details>

## Step 5: Evaluation

### Perplexity

Perplexity is the standard metric for language models. It measures how "surprised" the model is by the test data.

:::info[Plain English: What Is Perplexity?]
Imagine you're reading a book and trying to guess each next word. Sometimes you're pretty sure ("The cat sat on the ___" → "mat"), and sometimes you have no idea. Perplexity measures how "confused" the model is on average. A perplexity of 50 means the model is roughly as uncertain as picking randomly from 50 options at each step. A perplexity of 10 means it's narrowed it down to about 10 plausible options. Lower perplexity = better model.
:::

:::note[Perplexity]
$$
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)
$$

Perplexity is the exponential of the average cross-entropy loss. Lower is better. A perplexity of 50 means the model is, on average, as uncertain as if it had to choose uniformly among 50 equally likely next tokens.
:::

**Reading the formula:** **PPL** is perplexity. **exp(...)** is the exponential function (e raised to a power). **N** is the total number of tokens in the test data. **Σ from i=1 to N** means "add up over all tokens." **log P(x_i | x_{<i})** is the log-probability the model assigned to the correct token x_i, given all tokens that came before it. The **−1/N** computes the average negative log-probability (the cross-entropy loss). Then **exp** converts that average loss back into a "number of choices" interpretation.

| Perplexity Range | Interpretation |
|-----------------|----------------|
| 1 | Perfect prediction (impossible in practice) |
| 10–30 | Excellent for domain-specific text |
| 30–100 | Good for general text |
| 100–500 | The model captures some structure |
| 500+ | Barely better than random |

```python title="Comprehensive evaluation"
import tiktoken
import math
import torch

enc = tiktoken.get_encoding("gpt2")

def evaluate_model(model, val_data, config, device='cuda'):
    """Run full evaluation suite."""
    model.eval()
    ctx_len = config['context_length']

    # 1. Perplexity
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(200, len(val_data) // (64 * ctx_len))

    with torch.no_grad():
        for i in range(n_batches):
            ix = torch.randint(len(val_data) - ctx_len, (64,))
            x = torch.stack([
                torch.from_numpy(val_data[j:j+ctx_len].astype(np.int64))
                for j in ix
            ]).to(device)
            y = torch.stack([
                torch.from_numpy(val_data[j+1:j+1+ctx_len].astype(np.int64))
                for j in ix
            ]).to(device)

            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.1f}")

    # 2. Generate samples at different temperatures
    print("\\n--- Generated Samples ---")
    prompts = ["The ", "Once upon a ", "To be or not"]
    for prompt in prompts:
        tokens = enc.encode(prompt)
        x = torch.tensor([tokens], device=device)
        for temp in [0.5, 0.8, 1.0]:
            out = model.generate(x, max_new_tokens=100,
                                 temperature=temp, top_k=40)
            text = enc.decode(out[0].tolist())
            print(f"  [{prompt}...] (T={temp}): {text[:200]}")
        print()

    # 3. Top-k accuracy
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for _ in range(50):
            ix = torch.randint(len(val_data) - ctx_len, (32,))
            x = torch.stack([
                torch.from_numpy(val_data[j:j+ctx_len].astype(np.int64))
                for j in ix
            ]).to(device)
            y = torch.stack([
                torch.from_numpy(val_data[j+1:j+1+ctx_len].astype(np.int64))
                for j in ix
            ]).to(device)

            logits, _ = model(x, y)
            preds = logits.argmax(dim=-1)
            top5 = logits.topk(5, dim=-1).indices

            correct_top1 += (preds == y).sum().item()
            correct_top5 += (top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
            total += y.numel()

    print(f"Top-1 Accuracy: {correct_top1/total:.2%}")
    print(f"Top-5 Accuracy: {correct_top5/total:.2%}")

    return {'perplexity': perplexity, 'loss': avg_loss,
            'top1_acc': correct_top1/total, 'top5_acc': correct_top5/total}
```

:::tip[Line-by-Line Walkthrough]
- **`n_batches = min(200, len(val_data) // (64 * ctx_len))`** — Limits evaluation to 200 batches so it doesn't take too long, while still covering enough data for a reliable estimate.
- **`total_loss += loss.item() * x.numel()`** — Accumulates the total loss weighted by the number of tokens, so we get a proper average across all tokens (not across batches).
- **`perplexity = math.exp(avg_loss)`** — Converts average cross-entropy loss to perplexity. For example, a loss of 3.0 → perplexity of ~20.
- **`model.generate(x, max_new_tokens=100, temperature=temp, top_k=40)`** — Generates text samples at different temperatures to show the diversity-quality tradeoff.
- **`preds = logits.argmax(dim=-1)`** — Top-1 prediction: picks the single most likely next token at each position.
- **`top5 = logits.topk(5, dim=-1).indices`** — Top-5 predictions: checks if the correct token is among the 5 most likely predictions.
- **`(top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()`** — Counts how often the correct token appears in the top-5 predictions.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy tiktoken
```
Requires a trained model checkpoint.

**Steps:**
1. Load your trained model and validation data.
2. Call `evaluate_model(model, val_data, config)`.
3. This produces perplexity, accuracy metrics, and generated text samples.

**Expected output:**
```
Validation Loss: 1.8234
Perplexity: 6.2

--- Generated Samples ---
  [The ...] (T=0.5): The king is dead...
  [The ...] (T=0.8): The summer air was thick...
  [The ...] (T=1.0): The wandering knight did speak...

Top-1 Accuracy: 35.2%
Top-5 Accuracy: 62.8%
```
(Values depend on your dataset and model size.)

</details>

## Step 6: Analysis and Report

After training, write a brief report (1–2 pages) covering:

### Required Sections

1. **Dataset Description**
   - What data did you use? How much?
   - How did you clean and prepare it?
   - What tokenizer did you use and why?

2. **Model Configuration**
   - How many parameters? How many layers?
   - Why did you choose this size?
   - Any architectural modifications from the base GPT?

3. **Training Details**
   - How many steps? What learning rate?
   - How long did training take?
   - Include your training curve (loss vs step)

4. **Results**
   - Final perplexity on the validation set
   - Include 3–5 generated text samples
   - What did the model learn well? What does it struggle with?

5. **Observations**
   - How did the generated text quality change during training?
   - Did you notice overfitting? How could you address it?
   - What would you do differently with more compute?

---

## Bonus Challenges

These are optional extensions for students who want to go deeper.

:::tip[Bonus 1: Compare Model Sizes — intermediate]

Train models with 1M, 5M, 20M, and 50M parameters on the same dataset with the same number of training tokens. Plot the validation loss vs parameter count. Do you observe a power law? How does it compare to the Kaplan/Chinchilla scaling laws?

<div>
**What to look for:**
- On a log-log plot of loss vs parameters, the points should roughly fall on a straight line (power law)
- The slope tells you the scaling exponent — compare to Kaplan's α_N ≈ 0.076
- Smaller models may need more training steps to converge, so make sure to use a fixed token budget rather than a fixed step budget
<details>
<summary>Hints</summary>

1. Train 3-4 models with the same number of total training tokens
2. Keep all hyperparameters the same except d_model and n_layers
3. Plot perplexity vs parameter count on a log-log scale

</details>

:::

:::tip[Bonus 2: Implement Beam Search — intermediate]

Implement beam search generation as an alternative to sampling. Compare the quality of generated text between beam search (beam width 5) and nucleus sampling (top-p = 0.9) for the same prompts.

<div>
**Solution sketch:**

```python
def beam_search(model, prompt_ids, beam_width=5, max_tokens=100):
    beams = [(prompt_ids, 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_tokens):
        candidates = []
        for seq, score in beams:
            logits, _ = model(seq.unsqueeze(0))
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            top_k = log_probs.topk(beam_width)
            
            for lp, idx in zip(top_k.values, top_k.indices):
                new_seq = torch.cat([seq, idx.unsqueeze(0)])
                candidates.append((new_seq, score + lp.item()))
        
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return beams[0][0]  # return best beam
```

:::tip[Line-by-Line Walkthrough]
- **`beams = [(prompt_ids, 0.0)]`** — Starts with a single beam containing the prompt and a score of 0.0 (log-probability).
- **`logits, _ = model(seq.unsqueeze(0))`** — Runs the model on the current sequence to predict the next token.
- **`log_probs = F.log_softmax(logits[0, -1], dim=-1)`** — Converts the last position's logits to log-probabilities (we work in log-space so we can add scores instead of multiplying tiny probabilities).
- **`top_k = log_probs.topk(beam_width)`** — For each beam, picks the top-k most likely next tokens to expand.
- **`candidates.append((new_seq, score + lp.item()))`** — Each candidate is the current sequence extended by one token, with the cumulative log-probability score.
- **`beams = sorted(candidates, ..., reverse=True)[:beam_width]`** — Keeps only the `beam_width` best candidates across all expansions.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```
Requires a trained NanoGPT model and `import torch.nn.functional as F`.

**Steps:**
1. Load your trained model and encode a prompt: `prompt_ids = torch.tensor(enc.encode("ROMEO:"))`.
2. Call: `result = beam_search(model, prompt_ids, beam_width=5, max_tokens=100)`.
3. Decode: `print(enc.decode(result.tolist()))`.

**Expected output:** Coherent, somewhat repetitive text (beam search tends to produce safer outputs than sampling). Compare with nucleus sampling from the same prompt to see the quality-diversity tradeoff.

</details>

<details>
<summary>Hints</summary>

1. Maintain a list of (sequence, score) pairs called 'beams'
2. At each step, expand each beam with the top-k next tokens
3. Keep only the best beam_width sequences
4. Score is the sum of log probabilities

</details>

:::

:::tip[Bonus 3: Train a Code Model — advanced]

Train your GPT on Python source code instead of natural language. Collect at least 10MB of Python code from open-source repositories. After training, evaluate the model by generating Python functions and checking:
1. Is the generated code syntactically valid? (Use `ast.parse`)
2. Does it follow Python conventions? (Use `pylint`)
3. Can it complete partial functions?

<div>
**Evaluation approach:**

```python
import ast

def check_syntax(code_string):
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False

generated_samples = [generate_text(model, "def ", max_tokens=200) 
                     for _ in range(100)]
valid = sum(check_syntax(s) for s in generated_samples)
print(f"Syntactically valid: \{valid\}/100 (\{valid\}%)")
```

:::tip[Line-by-Line Walkthrough]
- **`ast.parse(code_string)`** — Uses Python's built-in AST (Abstract Syntax Tree) parser to check if the generated code is syntactically valid — the same parser Python itself uses.
- **`return True / except SyntaxError: return False`** — If `ast.parse` succeeds, the code is valid Python syntax. If it throws a `SyntaxError`, the generated code has syntax problems.
- **`generate_text(model, "def ", max_tokens=200)`** — Generates 100 Python function samples starting with `"def "`, giving the model a chance to produce complete function definitions.
- **`valid = sum(check_syntax(s) for s in generated_samples)`** — Counts how many of the 100 generated samples are syntactically correct, giving you a success rate.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch tiktoken
```
Requires a trained code model and the `generate_text` function from the evaluation section.

**Steps:**
1. Train your model on Python code instead of Shakespeare.
2. Define `generate_text` from the evaluation section.
3. Run this evaluation code.

**Expected output:**
```
Syntactically valid: 42/100 (42%)
```
(A well-trained small code model typically achieves 30–60% syntax validity.)

</details>

A well-trained code model should produce syntactically valid Python 30–60% of the time, even at small scale.
<details>
<summary>Hints</summary>

1. Collect Python files from popular open-source repos
2. Add a file separator token between files
3. Consider adding indentation-aware tokenization
4. Evaluate by checking if generated code is syntactically valid

</details>

:::

:::tip[Bonus 4: Add Attention Visualization — advanced]

Add attention weight extraction to your model and create visualizations showing which tokens attend to which other tokens. Analyze the attention patterns:
- Do any heads specialize (e.g., attend to the previous word, or to punctuation)?
- How do attention patterns change across layers?
- Can you find "induction heads" — heads that copy patterns from earlier in the sequence?

<div>
**What to look for:**
- **Layer 0:** Often has simple positional patterns (attend to adjacent tokens)
- **Later layers:** More semantic patterns (attend to related content words)
- **Induction heads:** Heads that look for pattern `[A][B]...[A]` and predict `[B]` — this is a key mechanism behind in-context learning
<details>
<summary>Hints</summary>

1. Modify the attention layer to optionally return attention weights
2. Use matplotlib's imshow for visualization
3. Look for interpretable patterns in different heads/layers
4. Try it on text where you know the grammatical structure

</details>

:::

---

## Submission Checklist

Before marking this project as complete, make sure you have:

- [ ] A working data pipeline (download → clean → tokenize → train/val split)
- [ ] A trained model with saved checkpoints
- [ ] A training curve plot showing loss vs step
- [ ] Perplexity numbers on the validation set
- [ ] At least 5 generated text samples at different temperatures
- [ ] A written report covering dataset, model, training, and observations
- [ ] All code is clean, documented, and runnable

:::tip[What Makes a Great Submission]
The best projects aren't necessarily the ones with the lowest perplexity. They're the ones where the student demonstrates deep understanding: Why did you choose these hyperparameters? What did the training dynamics tell you? Where did the model succeed and fail, and why? These questions matter more than the final number.
:::

---

## Resources

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** _(tool)_ by Andrej Karpathy — The reference implementation this project is based on. Study the training script and config.

- **[Project Gutenberg](https://www.gutenberg.org/)** _(tool)_ — Free books in plain text format — an excellent source of training data.

- **[The Pile](https://pile.eleuther.ai/)** _(tool)_ by EleutherAI — A large, diverse dataset for language model training. Use a subset for this project.

- **[Weights & Biases (wandb)](https://wandb.ai/)** _(tool)_ — Free experiment tracking tool. Log your training runs for beautiful visualizations and comparisons.

- **[How to Train Your Own LLM (Blog)](https://blog.replit.com/llm-training)** _(tutorial)_ by Replit — Practical walkthrough of training a code LLM from scratch, with lessons learned.
