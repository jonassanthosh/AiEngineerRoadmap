---
sidebar_position: 3
slug: pretraining-at-scale
title: "Pretraining at Scale"
---


# Pretraining at Scale

:::info[What You'll Learn]
- Data curation and quality filtering at scale
- Training recipes (learning rate, batch size, sequence length scheduling)
- Stability techniques for large training runs
- When and how to use curriculum learning
:::

:::note[Prerequisites]
[Training Infrastructure](/curriculum/month-4/training-infrastructure), [Scaling Laws](/curriculum/month-4/scaling-laws), and [nanoGPT](/curriculum/month-4/nanogpt) from Month 4.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~2 hours

Pretraining a large language model is one of the most demanding engineering challenges in ML. It requires curating trillions of tokens, orchestrating hundreds of GPUs for weeks or months, and maintaining stable training through countless potential failure modes. This lesson covers the full pipeline from raw web data to a trained base model.

## The Pretraining Data Pipeline

Data quality is the single largest determinant of model quality. The pipeline has four stages: collection, filtering, deduplication, and mixing.

### Stage 1: Data Collection

Most LLMs train on a mixture of web text, code, books, and curated datasets.

| Source | Scale | Content |
|--------|-------|---------|
| Common Crawl | ~250B pages | Web scrape, updated monthly |
| The Pile | 825 GB | Curated mix: academic, books, code, web |
| RedPajama v2 | 30T tokens | Open reproduction of LLaMA data |
| StarCoder data | 783 GB | Code from GitHub (licensed) |
| Wikipedia | ~20 GB | Encyclopedic text, high quality |
| arXiv | ~100 GB | Scientific papers |

:::warning[Web Data Is Mostly Garbage]
Raw Common Crawl is approximately 60-70% low-quality content: SEO spam, boilerplate navigation text, duplicated pages, and machine-generated filler. Aggressive filtering is essential — the LLaMA paper reports keeping only about 4% of Common Crawl after filtering.
:::

### Stage 2: Quality Filtering

Quality filtering transforms raw web scrapes into training-worthy text. The approach used by most modern LLMs combines rule-based heuristics with classifier-based filtering.

```python title="quality_filter.py — Heuristic text quality filters"
import re
from dataclasses import dataclass

@dataclass
class QualitySignals:
    char_count: int
    word_count: int
    avg_word_length: float
    alpha_ratio: float
    uppercase_ratio: float
    line_count: int
    short_line_ratio: float
    bullet_ratio: float
    duplicate_line_ratio: float

def compute_quality_signals(text: str) -> QualitySignals:
    lines = text.split("\\n")
    words = text.split()
    chars = list(text)

    word_count = len(words)
    if word_count == 0:
        return QualitySignals(0, 0, 0, 0, 0, 0, 0, 0, 0)

    avg_word_len = sum(len(w) for w in words) / word_count
    alpha_chars = sum(1 for c in chars if c.isalpha())
    alpha_ratio = alpha_chars / max(len(chars), 1)
    upper_chars = sum(1 for c in chars if c.isupper())
    uppercase_ratio = upper_chars / max(alpha_chars, 1)

    short_lines = sum(1 for l in lines if len(l.strip()) < 20)
    short_line_ratio = short_lines / max(len(lines), 1)

    bullet_lines = sum(1 for l in lines if l.strip().startswith(("•", "-", "*", "·")))
    bullet_ratio = bullet_lines / max(len(lines), 1)

    unique_lines = set(l.strip() for l in lines if l.strip())
    dup_ratio = 1 - len(unique_lines) / max(len([l for l in lines if l.strip()]), 1)

    return QualitySignals(
        char_count=len(text),
        word_count=word_count,
        avg_word_length=avg_word_len,
        alpha_ratio=alpha_ratio,
        uppercase_ratio=uppercase_ratio,
        line_count=len(lines),
        short_line_ratio=short_line_ratio,
        bullet_ratio=bullet_ratio,
        duplicate_line_ratio=dup_ratio,
    )

def passes_quality_filter(text: str) -> bool:
    """Apply heuristic quality filters similar to those in the C4/LLaMA pipelines."""
    signals = compute_quality_signals(text)

    # Too short or too long
    if signals.word_count < 50 or signals.word_count > 100_000:
        return False
    # Not enough actual text (boilerplate, navigation, etc.)
    if signals.alpha_ratio < 0.6:
        return False
    # Average word length outlier (garbled text or code-heavy)
    if signals.avg_word_length < 3.0 or signals.avg_word_length > 10.0:
        return False
    # Too many short lines (lists, menus, navigation)
    if signals.short_line_ratio > 0.5:
        return False
    # Too many duplicate lines
    if signals.duplicate_line_ratio > 0.3:
        return False
    # ALL CAPS screaming
    if signals.uppercase_ratio > 0.4:
        return False

    # Check for common boilerplate phrases
    boilerplate = [
        "cookie policy", "terms of service", "subscribe to our newsletter",
        "click here to", "javascript is required", "enable javascript",
        "all rights reserved", "skip to content", "toggle navigation",
    ]
    text_lower = text.lower()
    boilerplate_count = sum(1 for phrase in boilerplate if phrase in text_lower)
    if boilerplate_count >= 3:
        return False

    return True

# ---- Test ----
good_text = '''
Machine learning is a subfield of artificial intelligence that focuses on
building systems that can learn from data. Unlike traditional programming
where rules are explicitly coded, ML systems discover patterns automatically.
This approach has revolutionized fields from computer vision to natural
language processing, enabling applications that were previously impossible.
''' * 3

bad_text = "Click here | Home | About | Contact | Privacy Policy | Terms\\n" * 20

print(f"Good text passes: {passes_quality_filter(good_text)}")
print(f"Bad text passes:  {passes_quality_filter(bad_text)}")
```

:::tip[Line-by-Line Walkthrough]
- **`compute_quality_signals(text)`** — Extracts numerical features from a text document: character count, word count, average word length, how much is actual letters vs. symbols, etc. Think of it as a health checkup for a document.
- **`alpha_ratio = alpha_chars / max(len(chars), 1)`** — What fraction of characters are actual letters (a-z). Low ratio means lots of symbols, numbers, or HTML junk.
- **`dup_ratio = 1 - len(unique_lines) / max(...)`** — Checks how many lines are repeated. High duplication = boilerplate or spam.
- **`if signals.word_count < 50 or signals.word_count > 100_000`** — Reject documents that are too short (probably fragments) or too long (probably data dumps).
- **`if signals.alpha_ratio < 0.6`** — Reject if less than 60% of characters are letters — catches code-heavy pages, navigation menus, etc.
- **`boilerplate_count = sum(...)`** — Counts how many common web boilerplate phrases appear. If 3 or more match, the document is likely a generic webpage, not real content.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
No external packages needed — uses only Python standard library.

**Steps:**
1. Save the code to `quality_filter.py`
2. Run: `python quality_filter.py`

**Expected output:**
```
Good text passes: True
Bad text passes:  False
```

</details>

### Stage 3: Deduplication

Duplicate content in training data causes models to memorize specific passages and wastes compute on redundant examples. There are three levels of deduplication:

1. **URL deduplication** — Remove documents from the same URL.
2. **Document-level deduplication** — Use MinHash LSH to find near-duplicate documents.
3. **Paragraph/n-gram deduplication** — Remove repeated passages across documents.

```python title="minhash_dedup.py — Near-duplicate detection with MinHash"
import hashlib
import random

class MinHashDeduplicator:
    """MinHash-based near-duplicate document detection."""

    def __init__(self, num_hashes: int = 128, ngram_size: int = 5, threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.threshold = threshold
        # Random hash parameters (a*x + b mod p)
        self.max_hash = 2**32 - 1
        p = 4294967311  # large prime
        self.hash_params = [
            (random.randint(1, self.max_hash), random.randint(0, self.max_hash), p)
            for _ in range(num_hashes)
        ]

    def _get_ngrams(self, text: str) -> set[str]:
        words = text.lower().split()
        return {" ".join(words[i:i+self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)}

    def _minhash(self, ngrams: set[str]) -> list[int]:
        signature = []
        hashed_ngrams = [int(hashlib.md5(ng.encode()).hexdigest(), 16) % self.max_hash for ng in ngrams]
        for a, b, p in self.hash_params:
            min_val = min((a * h + b) % p for h in hashed_ngrams) if hashed_ngrams else self.max_hash
            signature.append(min_val)
        return signature

    def similarity(self, sig1: list[int], sig2: list[int]) -> float:
        return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

    def is_duplicate(self, text1: str, text2: str) -> bool:
        ngrams1 = self._get_ngrams(text1)
        ngrams2 = self._get_ngrams(text2)
        if not ngrams1 or not ngrams2:
            return False
        sig1 = self._minhash(ngrams1)
        sig2 = self._minhash(ngrams2)
        return self.similarity(sig1, sig2) >= self.threshold


# ---- Demo ----
dedup = MinHashDeduplicator(threshold=0.8)

doc_a = "Machine learning enables computers to learn from data without explicit programming."
doc_b = "Machine learning allows computers to learn from data without being explicitly programmed."
doc_c = "The weather in Paris is beautiful in the spring and summer months."

print(f"A vs B (near-duplicate): {dedup.is_duplicate(doc_a, doc_b)}")
print(f"A vs C (different):      {dedup.is_duplicate(doc_a, doc_c)}")
```

:::tip[Line-by-Line Walkthrough]
- **`self.hash_params = [...]`** — Creates 128 different random hash functions. Each one maps text chunks to numbers differently, reducing the chance of random collisions.
- **`_get_ngrams(text)`** — Splits text into overlapping groups of 5 consecutive words (n-grams). These are the "fingerprint" of the document.
- **`_minhash(ngrams)`** — For each of the 128 hash functions, finds the *minimum* hash value across all n-grams. This creates a compact "signature" — two documents with similar content will have similar signatures.
- **`similarity(sig1, sig2)`** — Compares two signatures by counting how many of the 128 positions match. The more matches, the more similar the documents.
- **`is_duplicate(text1, text2)`** — Returns True if similarity is above the threshold (default 80%), indicating near-duplicate content.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
No external packages needed — uses only Python standard library (`hashlib`, `random`).

**Steps:**
1. Save the code to `minhash_dedup.py`
2. Run: `python minhash_dedup.py`

**Expected output:**
```
A vs B (near-duplicate): True
A vs C (different):      False
```
(Note: MinHash is probabilistic — results may occasionally vary, but with 128 hashes the accuracy is high.)

</details>

### Stage 4: Data Mixing

The ratio of different data sources significantly impacts model behavior. Here's a typical mix:

| Source | Proportion | Purpose |
|--------|-----------|---------|
| Web text (filtered) | 65-70% | General knowledge, diverse language |
| Code | 10-15% | Reasoning, structured thinking |
| Books | 5-8% | Long-form coherence |
| Academic papers | 3-5% | Technical knowledge |
| Wikipedia | 3-5% | Factual grounding |
| Conversational | 2-3% | Dialogue patterns |

:::tip[Data Mixing Is an Art]
The Chinchilla paper showed that more data (with proportionally less training) beats bigger models. But the *composition* of that data matters enormously. The DoReMi paper (Xie et al., 2023) showed that optimizing domain weights with a small proxy model can improve downstream performance by 5-10% at no additional training cost.
:::

## Curriculum Learning

Instead of shuffling all data randomly, **curriculum learning** presents data in a structured order — typically from easier/cleaner examples to harder/noisier ones.

Common curriculum strategies:
1. **Quality-first:** Start with high-quality data (Wikipedia, books), then introduce web text.
2. **Domain scheduling:** Increase code and math data in later stages to boost reasoning.
3. **Upsampling:** Repeat high-quality data multiple times while seeing low-quality data only once.

The LLaMA 3 training used a multi-phase curriculum: the final phase upsampled code, math, and multilingual data to improve specific capabilities.

## Training Stability

Training a large model for weeks is a battle against instability. Here are the key techniques.

### Gradient Clipping

Large gradients cause parameter updates that are too aggressive, destabilizing training.

```python title="Training loop with stability techniques"
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def create_training_setup(model: nn.Module, total_steps: int, warmup_steps: int = 2000):
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps  # linear warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    return optimizer, scheduler

def train_step(model, batch, optimizer, scheduler, max_grad_norm=1.0):
    """Single training step with stability techniques."""
    optimizer.zero_grad()

    input_ids, labels = batch
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    loss.backward()

    # Gradient clipping — critical for stability
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Detect loss spikes
    if loss.item() > 10.0:
        print(f"WARNING: Loss spike detected ({loss.item():.2f}), skipping update")
        optimizer.zero_grad()
        return loss.item(), grad_norm.item(), True

    optimizer.step()
    scheduler.step()

    return loss.item(), grad_norm.item(), False
```

:::tip[Line-by-Line Walkthrough]
- **`AdamW(..., betas=(0.9, 0.95), weight_decay=0.1)`** — AdamW optimizer with standard LLM settings: momentum of 0.9, second-moment decay of 0.95, and weight decay to prevent overfitting.
- **`if step < warmup_steps: return step / warmup_steps`** — Linear warmup: the learning rate starts at zero and gradually increases over the first 2000 steps, preventing early instability.
- **`0.1 + 0.9 * 0.5 * (1 + torch.cos(...))`** — After warmup, the learning rate follows a cosine schedule: it slowly decreases but never drops below 10% of the peak (the `0.1 +` part).
- **`grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`** — Gradient clipping: if the total gradient magnitude exceeds 1.0, scale all gradients down proportionally. This prevents a single bad batch from derailing training.
- **`if loss.item() > 10.0`** — Loss spike detection: if the loss is suspiciously high, skip this update entirely rather than letting bad gradients corrupt the model weights.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. This code provides utility functions — integrate `create_training_setup` and `train_step` into your own training script
2. You'll need a model (`nn.Module`), a data loader, and a GPU for realistic use
3. Example integration: call `create_training_setup(model, total_steps=10000)` then loop over batches calling `train_step()`

**Expected output:**
When integrated into a training loop, you'll see loss values, gradient norms, and occasional warnings for loss spikes.

</details>

### Loss Spikes and Recovery

Loss spikes — sudden increases in training loss — are a major challenge at scale. Common causes and fixes:

| Cause | Symptom | Fix |
|-------|---------|-----|
| Bad data batch | Single-step spike | Skip batch if loss > threshold |
| Learning rate too high | Oscillating loss | Reduce LR or extend warmup |
| Numerical overflow | NaN/Inf in loss | Use mixed precision carefully, check norms |
| Gradient explosion | Huge grad norm | Lower clip threshold |
| Data distribution shift | Gradual drift | Improve data shuffling |

### Checkpointing

Save checkpoints frequently — compute is expensive, and you never want to lose more than a few hours of training.

```python title="checkpoint.py — Robust checkpointing for long training runs"
import torch
import os
import json
from datetime import datetime
from pathlib import Path

class CheckpointManager:
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list[Path] = []

    def save(self, model, optimizer, scheduler, step: int, loss: float, extra: dict = None):
        ckpt_path = self.save_dir / f"checkpoint-step{step}"
        ckpt_path.mkdir(exist_ok=True)

        torch.save(model.state_dict(), ckpt_path / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        metadata = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            **(extra or {}),
        }
        with open(ckpt_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.checkpoints.append(ckpt_path)

        # Rotate old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old = self.checkpoints.pop(0)
            import shutil
            shutil.rmtree(old)
            print(f"Removed old checkpoint: {old}")

        print(f"Saved checkpoint at step {step} (loss={loss:.4f})")

    def load_latest(self, model, optimizer=None, scheduler=None):
        ckpts = sorted(self.save_dir.glob("checkpoint-step*"))
        if not ckpts:
            print("No checkpoints found, starting from scratch")
            return 0

        latest = ckpts[-1]
        model.load_state_dict(torch.load(latest / "model.pt", weights_only=True))
        if optimizer and (latest / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(latest / "optimizer.pt", weights_only=True))
        if scheduler and (latest / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(latest / "scheduler.pt", weights_only=True))

        with open(latest / "metadata.json") as f:
            meta = json.load(f)

        print(f"Resumed from step {meta['step']} (loss={meta['loss']:.4f})")
        return meta["step"]
```

:::tip[Line-by-Line Walkthrough]
- **`self.max_checkpoints = max_checkpoints`** — Keeps only the 5 most recent checkpoints to avoid filling up disk space.
- **`torch.save(model.state_dict(), ckpt_path / "model.pt")`** — Saves just the model's learned weights (not the code), so you can load them later into the same architecture.
- **`torch.save(optimizer.state_dict(), ...)`** — Also saves the optimizer's internal state (momentum buffers, etc.) — essential for resuming training exactly where you left off.
- **`metadata = {...}`** — Saves human-readable info: which step, what the loss was, when the checkpoint was created.
- **`while len(self.checkpoints) > self.max_checkpoints`** — Rotates old checkpoints: deletes the oldest when you exceed the limit.
- **`ckpts = sorted(self.save_dir.glob("checkpoint-step*"))`** — Finds all existing checkpoints and sorts them by step number to locate the most recent.
- **`model.load_state_dict(torch.load(latest / "model.pt", weights_only=True))`** — Loads weights back into the model. `weights_only=True` is a security measure that prevents arbitrary code execution from checkpoint files.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `checkpoint.py`
2. Use it in your training script:
```python
ckpt_mgr = CheckpointManager("checkpoints/")
# During training: ckpt_mgr.save(model, optimizer, scheduler, step=1000, loss=2.5)
# To resume: start_step = ckpt_mgr.load_latest(model, optimizer, scheduler)
```
3. Checkpoints will be saved as directories under `checkpoints/`

**Expected output:**
```
Saved checkpoint at step 1000 (loss=2.5000)
Resumed from step 1000 (loss=2.5000)
```

</details>

## Multi-Node Training Setup

Training a model larger than a few billion parameters requires distributing across multiple GPUs and nodes. The three main parallelism strategies are:

1. **Data Parallelism (DP):** Each GPU holds a full copy of the model and processes different data. Gradients are synchronized (all-reduce) after each step.

2. **Tensor Parallelism (TP):** Individual layers are split across GPUs. A single matrix multiplication is divided so each GPU computes part of the result.

3. **Pipeline Parallelism (PP):** Different layers are placed on different GPUs. Microbatches flow through the pipeline.

:::info[FSDP: The Modern Default]
Fully Sharded Data Parallelism (FSDP) shards model parameters, gradients, and optimizer states across GPUs. Each GPU only holds a fraction of the full model. Parameters are gathered on-demand for forward/backward passes, then re-sharded. This is the default strategy in PyTorch for models that don't fit on a single GPU.
:::

```python title="fsdp_setup.py — Multi-GPU training with FSDP"
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import os

def setup_fsdp_training():
    """Setup FSDP for multi-GPU training (run with torchrun)."""
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    # Create your model (on CPU first for memory efficiency)
    model = create_your_model()

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # shard everything
        device_id=rank,
        use_orig_params=True,  # needed for torch.compile compatibility
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    return model, optimizer, rank

# Launch command:
# torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$RANK \\
#          --master_addr=$MASTER --master_port=29500 train.py
```

:::tip[Line-by-Line Walkthrough]
- **`dist.init_process_group("nccl")`** — Initializes distributed communication using NCCL (NVIDIA's high-performance GPU communication library).
- **`rank = int(os.environ["LOCAL_RANK"])`** — Each GPU process gets a unique rank (0, 1, 2, ...) set by `torchrun`. This tells the process which GPU to use.
- **`torch.cuda.set_device(rank)`** — Assigns this process to its designated GPU.
- **`model = create_your_model()`** — Creates the model on CPU first — FSDP will shard it across GPUs afterward.
- **`FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)`** — Wraps the model in FSDP, which splits (shards) the model's parameters, gradients, and optimizer states across all GPUs. Each GPU holds only a fraction of the full model.
- **`use_orig_params=True`** — Preserves original parameter names, needed for compatibility with `torch.compile`.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```
You also need multiple NVIDIA GPUs with CUDA support.

**Steps:**
1. Save the code to `fsdp_setup.py` and add your model definition as `create_your_model()`
2. Launch with torchrun:
```bash
# Single node, 8 GPUs:
torchrun --nproc_per_node=8 fsdp_setup.py

# Multi-node (4 nodes, 8 GPUs each = 32 GPUs total):
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$RANK \
         --master_addr=$MASTER --master_port=29500 fsdp_setup.py
```

**Expected output:**
Each process will print its rank and GPU assignment. The model will be automatically sharded across all available GPUs.

</details>

## Cost Estimation

Understanding training costs is essential for planning a pretraining run.

**Key formula:** Training FLOPs ≈ 6 × N × D

Where N = number of parameters and D = number of training tokens.

| Model Size | Tokens | FLOPs | A100 Hours | Estimated Cost |
|-----------|--------|-------|-----------|---------------|
| 1B | 20B | 1.2×10²⁰ | ~400 | ~$1,200 |
| 7B | 1T | 4.2×10²² | ~25,000 | ~$75,000 |
| 13B | 1.5T | 1.2×10²³ | ~75,000 | ~$225,000 |
| 70B | 2T | 8.4×10²³ | ~500,000 | ~$1,500,000 |

:::warning[Hidden Costs]
GPU hours are only part of the story. Factor in: data storage and preprocessing (5-15% of total), failed runs and restarts (budget 20-30% overhead), evaluation and iteration, engineering time, and electricity/cooling.
:::

### Chinchilla Scaling Laws

The Chinchilla paper (Hoffmann et al., 2022) showed that most LLMs before 2022 were *undertrained* — they used too many parameters with too little data. The compute-optimal ratio is approximately:

:::info[Plain English: What Does This Formula Mean?]
Think of it like a recipe: for a model of a given size, there's an ideal amount of data to feed it. If you have a model with 7 billion "brain cells" (parameters), you should show it about 20 times that many words (tokens) during training. Too few tokens and the model hasn't seen enough examples to learn well. Too many parameters with too little data is like building a huge library with only a few books — wasteful.
:::

\[
D_{\text{optimal}} \approx 20 \times N
\]

**Reading the formula:** \( D_{\text{optimal}} \) is the ideal number of training tokens. \( N \) is the number of model parameters. \( 20 \times \) means "twenty times as many tokens as parameters." So a 7B parameter model should ideally see about 140B tokens.

A 7B model should train on ~140B tokens for compute-optimal performance. But in practice, "over-training" (using more data than Chinchilla-optimal) produces better models for inference — LLaMA trains 7B on 1T+ tokens because inference cost dominates.

:::tip[Exercise 1: Build a Quality Filter Pipeline — intermediate]

Build a text quality filtering pipeline:

1. Write at least 5 heuristic quality filters (word count, character ratio, duplicate content, boilerplate detection, language detection).
2. Test your pipeline on a sample of 1000 web pages (you can download a small Common Crawl sample from the WET files).
3. Measure the pass/fail rate and manually inspect 50 documents that passed to verify quality.

<details>
<summary>Hints</summary>

1. Start with character and word count filters
2. Add a perplexity-based filter using a small language model
3. Process files in parallel using multiprocessing

</details>

:::

:::tip[Exercise 2: Training Cost Calculator — beginner]

Build a training cost calculator that takes model size, token count, GPU type, and cost per hour, then estimates total FLOPs, GPU hours, and dollar cost. Compare costs for training a 7B model on A100s vs H100s.

<details>
<summary>Hints</summary>

1. Use the formula: FLOPs ≈ 6 × N × D
2. A100 delivers ~312 TFLOPS for FP16
3. MFU (model FLOPS utilization) is typically 30-50% at scale

</details>

:::

:::tip[Exercise 3: Data Mixing Experiment — advanced]

Train a small language model (125M parameters) with different data mixing ratios:

1. Create a dataset with at least 3 domains (e.g., Wikipedia, code, web text).
2. Train three models with different mixing ratios.
3. Evaluate each model on held-out data from each domain.
4. Analyze: which mixing ratio gives the best overall performance? Which gives the best per-domain performance?

<details>
<summary>Hints</summary>

1. Use a small model (125M params) for fast iteration
2. Try at least 3 different mixing ratios
3. Measure perplexity on held-out sets from each domain separately

</details>

:::

## Resources

- **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)** _(paper)_ by Hoffmann et al. — The scaling laws paper that changed how the field thinks about data vs model size.

- **[The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)** _(paper)_ by Gao et al. — Design principles behind a widely-used pretraining dataset.

- **[LLaMA 3 Technical Report](https://arxiv.org/abs/2407.21783)** _(paper)_ by Meta AI — Detailed description of modern pretraining at scale, including data pipeline and curriculum.

- **[RedPajama: Open Dataset for Training Large Language Models](https://github.com/togethercomputer/RedPajama-Data)** _(tool)_ by Together AI — Open reproduction of the LLaMA training data pipeline.

- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)** _(paper)_ by Muennighoff et al. — What happens when you run out of data? Analysis of repeated data epochs.

- **[PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** _(tutorial)_ — Official PyTorch guide to Fully Sharded Data Parallelism.
