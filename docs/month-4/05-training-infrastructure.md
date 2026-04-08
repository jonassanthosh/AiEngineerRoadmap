---
sidebar_position: 5
slug: training-infrastructure
title: "Training Infrastructure"
---


# Training Infrastructure

Training a modern LLM requires distributing computation across dozens to thousands of GPUs. A single GPU cannot hold even a medium-sized model in memory, let alone train it efficiently. This lesson covers the hardware landscape, the key parallelism strategies, and the practical tools for distributed training.

## The Hardware Landscape

### GPUs (NVIDIA)

NVIDIA dominates the LLM training hardware market. Here's the progression of key training GPUs:

| GPU | VRAM | BF16 TFLOPS | Interconnect | Year |
|-----|------|-------------|--------------|------|
| V100 | 32 GB | 125 | NVLink 300 GB/s | 2017 |
| A100 | 80 GB | 312 | NVLink 600 GB/s | 2020 |
| H100 | 80 GB | 990 | NVLink 900 GB/s | 2023 |
| H200 | 141 GB | 990 | NVLink 900 GB/s | 2024 |
| B200 | 192 GB | 2250 | NVLink 1800 GB/s | 2025 |
| GB200 (Grace Blackwell) | 384 GB (2×192) | 4500 (2-chip) | NVLink 1800 GB/s | 2025 |
| B300 (Blackwell Ultra) | 288 GB | 4500 | NVLink 3600 GB/s | 2026 |

:::info[Why VRAM Is the Bottleneck]
For a 7B parameter model in fp16, you need:
- **Model weights:** 7B × 2 bytes = 14 GB
- **Optimizer states (Adam):** 7B × 8 bytes = 56 GB (momentum + variance in fp32, plus fp32 weight copy)
- **Gradients:** 7B × 2 bytes = 14 GB
- **Activations:** Varies with batch size, often 10–50 GB

**Total: ~84–134 GB** just for a 7B model. A single 80 GB A100 can't even hold it. That's why distributed training is not optional — it's required.
:::

### TPUs (Google)

Google's Tensor Processing Units are custom ASICs designed for matrix multiplication. TPU v4 pods can scale to thousands of chips with high-bandwidth interconnects. Google uses TPUs to train Gemini and PaLM. They're available through Google Cloud.

### The Interconnect Matters

Communication between GPUs is often the bottleneck. A single GPU might compute in 1ms what takes 10ms to communicate. Key interconnect technologies:

- **NVLink:** High-speed GPU-to-GPU within a single node (600–1800 GB/s)
- **InfiniBand:** GPU-to-GPU across nodes (400 Gb/s per port)
- **Ethernet:** Cheaper but slower (100–400 Gb/s), increasingly used with RoCE

## Data Parallelism (DDP)

Data parallelism is the simplest and most common form of distributed training. The idea: replicate the model on each GPU, split the data across GPUs, and synchronize gradients after each step.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│             │  │             │  │             │  │             │
│  Model Copy │  │  Model Copy │  │  Model Copy │  │  Model Copy │
│             │  │             │  │             │  │             │
│  Batch 0    │  │  Batch 1    │  │  Batch 2    │  │  Batch 3    │
│     ↓       │  │     ↓       │  │     ↓       │  │     ↓       │
│  Gradients  │  │  Gradients  │  │  Gradients  │  │  Gradients  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┼────────────────┼────────────────┘
                        │
                   AllReduce
                  (average gradients)
                        │
       ┌────────────────┼────────────────┼────────────────┐
       ↓                ↓                ↓                ↓
  Update weights   Update weights   Update weights   Update weights
  (identical)      (identical)      (identical)      (identical)
```

### How AllReduce Works

AllReduce is the collective communication operation that averages gradients across all GPUs. There are several implementations:

- **Ring AllReduce:** Each GPU sends a chunk of gradients to the next GPU in a ring. After two full rounds, all GPUs have the complete average. Communication cost is \( O(N) \) in data size, independent of the number of GPUs.
- **Tree AllReduce:** Uses a tree topology for reduce-then-broadcast. Can be faster for small messages.

```python title="PyTorch DDP — basic setup"
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, dataset):
    setup(rank, world_size)

    model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6).to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # ensure different shuffling per epoch
        for batch in dataloader:
            batch = batch.to(rank)
            output = model(batch, batch)
            loss = output.mean()

            optimizer.zero_grad()
            loss.backward()
            # DDP automatically synchronizes gradients via AllReduce
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} complete")

    cleanup()

# Launch with: torchrun --nproc_per_node=4 script.py
```

:::tip[Line-by-Line Walkthrough]
- **`dist.init_process_group("nccl", rank=rank, world_size=world_size)`** — Initializes the distributed backend. `"nccl"` is NVIDIA's optimized GPU communication library. `rank` is this GPU's ID (0, 1, 2, ...), `world_size` is the total number of GPUs.
- **`model = DDP(model, device_ids=[rank])`** — Wraps the model with DistributedDataParallel. This automatically synchronizes gradients across all GPUs after each backward pass using AllReduce.
- **`sampler = DistributedSampler(dataset, ...)`** — Ensures each GPU sees a different portion of the data. Without this, every GPU would process the same batches — wasting compute.
- **`sampler.set_epoch(epoch)`** — Changes the random seed for shuffling each epoch, so data is distributed differently. Without this, every epoch would use the same split.
- **`if rank == 0:`** — Only the first GPU prints logs and saves checkpoints, to avoid duplicate output.
- **`cleanup()`** — Shuts down the distributed process group cleanly.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```
Requires a machine with multiple NVIDIA GPUs and NCCL installed.

**Steps:**
1. Save the code to `ddp_training.py`
2. Launch with: `torchrun --nproc_per_node=4 ddp_training.py` (uses 4 GPUs)
3. For a single-GPU test: `torchrun --nproc_per_node=1 ddp_training.py`

**Expected output:**
```
Epoch 0 complete
Epoch 1 complete
...
Epoch 9 complete
```
(Only printed from rank 0. All GPUs process data in parallel.)

</details>

:::tip[torchrun vs mp.spawn]
Use `torchrun` (or `torch.distributed.launch`) to launch distributed training. It handles process creation, environment variables, and fault tolerance. Avoid `mp.spawn` for production training — `torchrun` can recover from individual worker failures.
:::

## Model Parallelism

When a model is too large for a single GPU, you must split it across multiple GPUs. There are two main approaches.

### Tensor Parallelism (TP)

Split individual **layers** across GPUs. For example, a linear layer with weight \( W \in \mathbb{R}^{m \times n} \) can be column-split across 2 GPUs:

```
GPU 0: W[:, :n/2]  →  output[:, :n/2]
GPU 1: W[:, n/2:]  →  output[:, n/2:]

Then concatenate (or AllReduce) to get the full output.
```

For attention, you can assign different attention heads to different GPUs — they're naturally independent until the output projection.

:::info[Megatron-LM Style Tensor Parallelism]
NVIDIA's Megatron-LM introduced an efficient tensor parallelism scheme for Transformers:
- **Attention:** Each GPU handles a subset of attention heads. After the output projection, an AllReduce synchronizes the results.
- **FFN:** The first linear layer is column-parallel, the second is row-parallel. This requires only one AllReduce per layer.

The total communication is 2 AllReduces per Transformer block, regardless of how many GPUs you use for TP.
:::

### Pipeline Parallelism (PP)

Split different **layers** across different GPUs. GPU 0 handles layers 1–12, GPU 1 handles layers 13–24, etc.

```
GPU 0          GPU 1          GPU 2          GPU 3
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Layers   │   │ Layers   │   │ Layers   │   │ Layers   │
│  1 - 6   │──→│  7 - 12  │──→│ 13 - 18  │──→│ 19 - 24  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

The problem with naive pipeline parallelism: **bubble overhead**. While GPU 3 processes micro-batch 1, GPUs 0–2 are idle (the pipeline isn't full yet). The solution is **micro-batching**: split each mini-batch into many micro-batches and pipeline them.

**GPipe** and **PipeDream** are two popular scheduling strategies:
- **GPipe:** Forward all micro-batches, then backward all micro-batches (simple but large memory for activations)
- **1F1B (One Forward One Backward):** Interleave forward and backward passes to reduce peak memory

## ZeRO Optimization (DeepSpeed)

ZeRO (Zero Redundancy Optimizer) is Microsoft DeepSpeed's approach to memory-efficient data parallelism. The key insight: in standard DDP, every GPU stores a **full copy** of the model weights, optimizer states, and gradients. Most of this is redundant.

ZeRO has three stages:

| Stage | What's Sharded | Memory Savings |
|-------|---------------|----------------|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | Optimizer states + gradients | ~8× |
| ZeRO-3 | Optimizer states + gradients + parameters | ~N× (N = num GPUs) |

:::info[ZeRO-3 (Full Sharding)]
With ZeRO-3, each GPU stores only \( 1/N \)-th of the model. When a layer needs the full parameters for a forward or backward pass, they're gathered from all GPUs on-the-fly via AllGather, used, and then discarded. This trades communication for memory — you can train models that are \( N \times \) larger than a single GPU's memory.
:::

```python title="DeepSpeed ZeRO configuration"
# deepspeed_config.json
config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,          # dynamic loss scaling
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,               # ZeRO Stage 3
        "offload_optimizer": {    # optionally offload to CPU
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,     # overlap communication with compute
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": False
}

# Launch with: deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json
```

:::tip[Line-by-Line Walkthrough]
- **`"train_batch_size": 256`** — The total effective batch size across all GPUs. DeepSpeed divides this by the number of GPUs and gradient accumulation steps to get the per-GPU micro-batch size.
- **`"gradient_accumulation_steps": 8`** — Simulates a larger batch by accumulating gradients over 8 micro-steps before updating weights.
- **`"loss_scale": 0`** — Enables dynamic loss scaling for fp16 training. It starts high and automatically adjusts to prevent overflow/underflow.
- **`"stage": 3`** — ZeRO Stage 3 shards everything (parameters, gradients, and optimizer states) across GPUs. This gives maximum memory savings.
- **`"offload_optimizer": {"device": "cpu"}`** — Moves optimizer states to CPU RAM, freeing GPU memory. This is slower but lets you train models that wouldn't otherwise fit.
- **`"overlap_comm": True`** — Overlaps GPU communication (AllGather/AllReduce) with computation so you don't waste time waiting.
- **`"gradient_clipping": 1.0`** — Clips gradients to a maximum norm of 1.0, preventing training instability from sudden large gradients.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install deepspeed torch
```
Requires NVIDIA GPUs with CUDA support.

**Steps:**
1. Save the config dict as `ds_config.json` (convert to proper JSON)
2. Write your training script as `train.py` using the DeepSpeed API
3. Launch: `deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json`

**Expected output:**
DeepSpeed prints a configuration summary at startup, followed by training logs showing loss, throughput, and memory usage per step.

</details>

## FSDP (Fully Sharded Data Parallel)

PyTorch's native answer to ZeRO-3. FSDP shards model parameters, gradients, and optimizer states across GPUs — exactly like ZeRO-3 — but integrated directly into PyTorch, without needing the DeepSpeed library.

```python title="PyTorch FSDP setup"
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import torch.distributed as dist

def train_fsdp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = build_transformer_model()  # your model

    # Configure mixed precision
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # like ZeRO-3
        mixed_precision=mp_policy,
        device_id=rank,
        use_orig_params=True,  # needed for torch.compile compatibility
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, batch in enumerate(dataloader):
        batch = batch.to(rank)
        loss = model(batch).loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0 and step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    dist.destroy_process_group()
```

:::tip[Line-by-Line Walkthrough]
- **`MixedPrecision(param_dtype=torch.bfloat16, ...)`** — Configures FSDP to use bf16 for parameters, gradient reduction, and buffers. bf16 halves memory usage and speeds up computation.
- **`FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, ...)`** — Wraps the model with full sharding (like ZeRO-3). Each GPU only stores 1/N-th of the parameters, gradients, and optimizer states.
- **`use_orig_params=True`** — Preserves the original parameter structure, which is needed for compatibility with `torch.compile` (PyTorch's JIT compiler).
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`** — Clips gradient norms to prevent training instability. FSDP handles this correctly across sharded parameters.
- **`dist.destroy_process_group()`** — Cleans up distributed processes at the end.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch>=2.0
```
Requires multiple NVIDIA GPUs.

**Steps:**
1. Save the code to `fsdp_training.py`
2. Launch: `torchrun --nproc_per_node=4 fsdp_training.py`

**Expected output:**
```
Step 0, Loss: 12.3456
Step 100, Loss: 8.2345
...
```
(Only printed from rank 0. FSDP automatically handles parameter gathering and gradient sharding.)

</details>

### FSDP vs DeepSpeed ZeRO

| Feature | FSDP | DeepSpeed ZeRO |
|---------|------|---------------|
| Native PyTorch | Yes | No (separate library) |
| CPU offloading | Limited | Full support |
| NVMe offloading | No | Yes (ZeRO-Infinity) |
| `torch.compile` compat | Yes | Limited |
| Maturity | Newer (stable since PyTorch 2.0) | Battle-tested |
| Community | Growing | Large |

## Mixed Precision Training

Training in full fp32 is wasteful. Modern GPUs have specialized tensor cores that are 2–8× faster for lower-precision formats.

| Format | Bits | Range | Precision | Use Case |
|--------|------|-------|-----------|----------|
| fp32 | 32 | ±3.4e38 | High | Master weights, accumulation |
| fp16 | 16 | ±65504 | Low | Activations, gradients (with care) |
| bf16 | 16 | ±3.4e38 | Very low | Activations, gradients (safer) |
| tf32 | 19 | ±3.4e38 | Medium | Enabled by default on A100+ |

:::info[bf16 vs fp16]
**fp16** has limited range (max ~65504). Gradients or activations that exceed this range cause **overflow**, which breaks training. You need **loss scaling** — multiply the loss by a large number before backward, then divide gradients by the same number.

**bf16** (brain floating point) has the same range as fp32 but fewer mantissa bits. It doesn't overflow in practice, so you don't need loss scaling. This makes it much simpler to use. bf16 is the standard for modern LLM training.
:::

```python title="Mixed precision training with PyTorch"
import torch
from torch.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# For fp16: use GradScaler to handle loss scaling
scaler = GradScaler()

for batch in dataloader:
    batch = batch.cuda()

    # Forward pass in fp16
    with autocast(device_type='cuda', dtype=torch.float16):
        loss = model(batch).loss

    # Backward pass with scaled loss
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# For bf16: no scaler needed (simpler)
for batch in dataloader:
    batch = batch.cuda()
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(batch).loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

:::tip[Line-by-Line Walkthrough]
- **`scaler = GradScaler()`** — Creates a gradient scaler for fp16 training. It multiplies the loss by a large number before backward, then divides gradients by the same number — this prevents small gradients from being rounded to zero in fp16.
- **`with autocast(device_type='cuda', dtype=torch.float16):`** — Tells PyTorch to automatically run operations in fp16 where it's safe (matrix multiplies, convolutions) and keep fp32 where it's not (loss computation, softmax).
- **`scaler.scale(loss).backward()`** — Multiplies the loss by the scale factor before computing gradients. This prevents underflow in fp16.
- **`scaler.unscale_(optimizer)`** — Divides gradients back to their true values before gradient clipping.
- **`scaler.step(optimizer)`** — Steps the optimizer, but skips the step if any gradients are NaN/Inf (which means the scale was too high).
- **`scaler.update()`** — Adjusts the scale factor: increases it if no overflow occurred, decreases it if overflow was detected.
- The **bf16 version** is much simpler — no scaler needed because bf16 has the same range as fp32 (it just has less precision).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```
Requires an NVIDIA GPU. bf16 requires Ampere architecture or newer (A100, RTX 3090+).

**Steps:**
1. Replace `MyModel()` and `dataloader` with your actual model and data.
2. Save the code to `mixed_precision.py`
3. Run: `python mixed_precision.py`

**Expected output:**
Training runs 2-3× faster than fp32 with roughly half the GPU memory usage. Loss values should be similar to fp32 training.

</details>

## Combining Parallelism Strategies (3D Parallelism)

Large-scale training combines all three strategies:

```
┌──────────────────────────────────────────┐
│              3D Parallelism              │
│                                          │
│  Data Parallel (DP/FSDP): 8 replicas     │
│    ├── Pipeline Parallel (PP): 4 stages  │
│    │     ├── Tensor Parallel (TP): 8 GPUs│
│    │     ├── Tensor Parallel (TP): 8 GPUs│
│    │     ├── Tensor Parallel (TP): 8 GPUs│
│    │     └── Tensor Parallel (TP): 8 GPUs│
│    └── ... (×4 PP stages)                │
│  Total: 8 × 4 × 8 = 256 GPUs            │
└──────────────────────────────────────────┘
```

Choosing the right combination depends on:
- **TP** uses fast intra-node NVLink, so it's bounded by the number of GPUs per node (typically 8)
- **PP** works across nodes but introduces pipeline bubbles
- **DP/FSDP** scales to any number of nodes with reasonable communication cost

## Gradient Checkpointing (Activation Recomputation)

A complementary memory optimization: instead of storing all intermediate activations for the backward pass, **recompute** them during the backward pass. This trades compute for memory — typically 30–40% slower but uses 60–70% less activation memory.

```python title="Gradient checkpointing in PyTorch"
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpointing(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Wrap sub-layers in checkpoint — activations recomputed in backward
        x = x + checkpoint(
            lambda inp: self.attn(inp, inp, inp)[0],
            self.ln1(x),
            use_reentrant=False
        )
        x = x + checkpoint(self.ff, self.ln2(x), use_reentrant=False)
        return x
```

:::tip[Line-by-Line Walkthrough]
- **`from torch.utils.checkpoint import checkpoint`** — Imports PyTorch's gradient checkpointing utility. This trades compute for memory by not saving intermediate activations.
- **`checkpoint(lambda inp: self.attn(inp, inp, inp)[0], self.ln1(x), use_reentrant=False)`** — Instead of saving the intermediate activations from the attention layer (which can be huge), this recomputes them during the backward pass. The `lambda` wraps the attention call, and `use_reentrant=False` uses the newer, safer checkpointing implementation.
- **`checkpoint(self.ff, self.ln2(x), use_reentrant=False)`** — Same idea for the feed-forward layer: discard activations after the forward pass, recompute them during backward.
- The residual connections (`x = x + ...`) remain outside the checkpoint, so the residual stream activations are still saved normally.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Use `TransformerBlockWithCheckpointing` as a drop-in replacement for your regular Transformer block.
2. No special launch command needed — gradient checkpointing works in regular training scripts.

**Expected output:**
Training will be 30-40% slower per step, but activation memory drops by 60-70%. This lets you use larger batch sizes or longer sequences.

</details>

## Summary

| Strategy | Splits | Reduces | Communication |
|----------|--------|---------|---------------|
| Data Parallelism | Data across GPUs | Training time | AllReduce (gradients) |
| Tensor Parallelism | Layers across GPUs | Memory per layer | AllReduce (activations) |
| Pipeline Parallelism | Model stages across GPUs | Memory per GPU | Point-to-point (activations) |
| ZeRO / FSDP | States across GPUs | Redundant memory | AllGather (params on demand) |
| Mixed Precision | — | Memory + compute | — |
| Gradient Checkpointing | — | Activation memory | — (trades compute) |

---

## Exercises

:::tip[Memory Estimation — beginner]

Calculate the total GPU memory needed to train a 13B parameter model in bf16 with AdamW optimizer. Include model weights, optimizer states, and gradients. Assume you don't count activations. How many A100 80GB GPUs would you need at minimum?

<div>
**Solution:**
- Model weights (bf16): 13B × 2 bytes = 26 GB
- Gradients (bf16): 13B × 2 bytes = 26 GB
- Optimizer states (fp32): 13B × 4 bytes × 3 = 156 GB (master weights + momentum + variance)
- **Total: ~208 GB** (without activations)
- Minimum GPUs: 208 / 80 = 2.6 → **at least 3 A100 80GB GPUs** using FSDP/ZeRO-3
- In practice, with activations and memory fragmentation, you'd need 4–8 GPUs.
<details>
<summary>Hints</summary>

1. Model params: count × bytes_per_param
2. Adam state: 2 fp32 copies per param (momentum + variance) + fp32 master weights
3. Use 2 bytes for fp16/bf16, 4 bytes for fp32

</details>

:::

:::tip[DDP Training Script — intermediate]

Write a complete, runnable PyTorch DDP training script for a small Transformer language model. Include proper initialization, data loading with `DistributedSampler`, gradient clipping, logging (only from rank 0), and checkpoint saving. Test with `torchrun --nproc_per_node=2`.

<div>
**Solution:** The key elements are:
1. `dist.init_process_group("nccl")` for GPU communication
2. `DistributedSampler` to partition data across GPUs
3. `DDP(model)` wrapper for automatic gradient sync
4. `sampler.set_epoch(epoch)` for proper shuffling
5. `if rank == 0:` guards for logging and checkpointing
6. `dist.destroy_process_group()` for clean shutdown
<details>
<summary>Hints</summary>

1. Use torchrun for launching
2. Use DistributedSampler for data loading
3. Synchronize metrics with dist.all_reduce
4. Save checkpoints only from rank 0

</details>

:::

:::tip[Parallelism Strategy Design — advanced]

You have a cluster of 64 A100 GPUs (8 nodes × 8 GPUs per node, connected by InfiniBand). You want to train a 30B parameter model. Design a parallelism strategy specifying TP, PP, and DP dimensions. Justify your choices based on communication patterns and memory requirements.

<div>
**Solution approach:**
- **TP = 4** within each node (uses fast NVLink; 8 would be fine too but 4 saves communication)
- **PP = 2** across pairs of nodes (splits 30B into 15B per stage, manageable memory)
- **DP = 8** across the remaining dimension (64 / 4 / 2 = 8 data-parallel replicas)
- With TP=4, each GPU handles ~7.5B/4 = 1.875B parameters per stage
- With PP=2, each GPU handles ~15B/4 = 3.75B parameters
- Use bf16 mixed precision and gradient checkpointing to fit in 80GB per GPU
- Use at least 16 micro-batches per PP step to minimize pipeline bubbles
<details>
<summary>Hints</summary>

1. Consider the compute-to-communication ratio
2. TP should stay within a single node (NVLink)
3. PP introduces bubbles — how many micro-batches minimize this?
4. FSDP can replace pure DP

</details>

:::

---

## Resources

- **[PyTorch Distributed Training Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)** _(tutorial)_ — Official PyTorch tutorial on DistributedDataParallel with practical examples.

- **[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** _(paper)_ by Rajbhandari et al. — The ZeRO paper introducing sharded data parallelism for training very large models.

- **[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)** _(paper)_ by Shoeybi et al. — NVIDIA's tensor parallelism approach for efficient large-scale training.

- **[DeepSpeed Documentation](https://www.deepspeed.ai/getting-started/)** _(tool)_ by Microsoft — Comprehensive documentation for DeepSpeed, including ZeRO, pipeline parallelism, and inference optimization.

- **[PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** _(tutorial)_ — Official tutorial for Fully Sharded Data Parallel training in PyTorch.

- **[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)** _(paper)_ by Narayanan et al. — NVIDIA's paper on combining tensor, pipeline, and data parallelism (3D parallelism) for training GPT-3-scale models.
