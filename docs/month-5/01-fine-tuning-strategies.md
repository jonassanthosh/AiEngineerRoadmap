---
sidebar_position: 1
slug: fine-tuning-strategies
title: "Fine-Tuning Strategies"
---


# Fine-Tuning Strategies

:::info[What You'll Learn]
- Full fine-tuning vs. parameter-efficient methods
- LoRA and QLoRA: how low-rank adapters work
- Adapter layers and prefix tuning
- Choosing the right fine-tuning strategy for your use case
:::

:::note[Prerequisites]
[GPT Architecture](/curriculum/month-4/gpt-architecture) and [nanoGPT](/curriculum/month-4/nanogpt) from Month 4.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

Pretrained LLMs are powerful general-purpose models, but they rarely do exactly what you need out of the box. Fine-tuning adapts a pretrained model to your specific task, domain, or style. The question isn't *whether* to fine-tune — it's *how much* of the model to update and *which method* to use.

This lesson covers the full spectrum: from updating every parameter (full fine-tuning) to updating less than 1% of them (LoRA, QLoRA, adapters). By the end, you'll know when to use each approach and how to implement them with the HuggingFace PEFT library.

## Full Fine-Tuning

Full fine-tuning updates **every parameter** in the model. You take a pretrained model, initialize from its weights, and continue training on your task-specific dataset with a lower learning rate.

```
Pretrained Model (all params frozen = ❄️)
         │
    Unfreeze all params 🔥
         │
    Train on your dataset
         │
Task-Specific Model (all params updated)
```

**Advantages:**
- Maximum expressiveness — the model can adapt every layer to your task
- Best possible performance when you have enough data

**Disadvantages:**
- Requires storing a **full copy** of the model per task (a 7B model = ~14 GB in FP16)
- High GPU memory — you need to store the model, gradients, and optimizer states (often 4–8× model size)
- Risk of **catastrophic forgetting** — the model may lose general capabilities
- Slow training, especially for models above 1B parameters

:::info[The Storage Problem]
If you fine-tune a 7B-parameter model for 10 different tasks using full fine-tuning, you need to store 10 separate copies — roughly 140 GB. Parameter-efficient methods solve this by keeping a single base model and storing small task-specific adapters (a few MB each).
:::

```python title="Full fine-tuning with HuggingFace"
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable:,}")  # Same as total — everything is trainable

training_args = TrainingArguments(
    output_dir="./full-ft-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,         # Much lower than pretraining LR
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
)

# trainer = Trainer(model=model, args=training_args, train_dataset=..., ...)
# trainer.train()
```

:::tip[Line-by-Line Walkthrough]
- **`AutoModelForCausalLM.from_pretrained(...)`** — Loads a pretrained language model with all its weights. Think of it as downloading a fully built brain.
- **`sum(p.numel() for p in model.parameters())`** — Counts every single adjustable number inside the model. For a 1B model, that's about one billion numbers.
- **`learning_rate=2e-5`** — Sets how big each learning step is. This is 100× smaller than pretraining because we want gentle nudges, not big rewrites.
- **`gradient_accumulation_steps=4`** — Instead of updating weights after every mini-batch, we stack 4 mini-batches worth of gradients together. This simulates a larger batch without needing more GPU memory.
- **`fp16=True`** — Uses half-precision (16-bit) numbers instead of full 32-bit, cutting memory use roughly in half.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers datasets torch accelerate
```

**Steps:**
1. Save the code to a file, e.g. `full_finetune.py`
2. Open a terminal and run: `python full_finetune.py`
3. Note: You'll need a HuggingFace account and access to the Llama model. Set your token with `huggingface-cli login`.

**Expected output:**
```
Total parameters:     1,235,814,400
Trainable parameters: 1,235,814,400
```
The trainer lines are commented out, so it will just print parameter counts without actually training.

</details>

## Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods freeze most of the pretrained model and only train a small number of additional or selected parameters. The core insight: the weight updates during fine-tuning often lie in a **low-rank subspace** — you don't need to update millions of parameters to adapt the model.

```
                    Full Fine-Tuning          PEFT (LoRA)
Parameters updated: 100%                      0.1–1%
GPU memory:         Very high                 Moderate
Storage per task:   Full model copy           Small adapter (MB)
Risk of forgetting: High                      Low
```

## LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) is the most widely used PEFT method. The idea is elegant: instead of updating a weight matrix \(W\) directly, you add a **low-rank decomposition** to it.

:::note[LoRA Decomposition]
For a pretrained weight matrix \(W_0 \in \mathbb{R}^{d \times k}\), LoRA adds a low-rank update:

:::info[Plain English: What Is This Formula Doing?]
Imagine you bought a pre-built car (the pretrained model). Instead of rebuilding the entire engine to customize it, you bolt on a small performance chip (the LoRA adapter). The original engine stays untouched — you just add a tiny upgrade on top. That's what LoRA does: it keeps the big weight matrix frozen and adds a small, trainable "patch" to it.
:::

\[
W = W_0 + \Delta W = W_0 + BA
\]

**Reading the formula:** *W* is the final weight matrix used during computation. *W₀* is the original pretrained weight matrix (frozen, not changed). *ΔW* is the small update we're learning. *B* and *A* are two small matrices that, when multiplied together, produce *ΔW*. Instead of learning millions of values in *ΔW* directly, we learn the much smaller *B* and *A*.

where:
- \(B \in \mathbb{R}^{d \times r}\) and \(A \in \mathbb{R}^{r \times k}\)
- \(r \ll \min(d, k)\) is the **rank** (typically 8, 16, or 64)
- \(W_0\) is **frozen** — only \(A\) and \(B\) are trained

The forward pass becomes:

:::info[Plain English: What Is This Formula Doing?]
When data flows through a layer, the model does two things at once: (1) multiplies the input by the original frozen weights (the pre-built engine), and (2) multiplies the input by the small LoRA matrices and adds the result. It's like running the original road plus a shortcut side-road at the same time, then merging at the end.
:::

\[
h = W_0 x + BAx
\]

**Reading the formula:** *h* is the layer's output. *W₀x* is the output from the original pretrained layer (unchanged). *BAx* is the extra contribution from the LoRA adapter — the input *x* passes through matrix *A* first, then through matrix *B*. The two results are added together.

\(A\) is initialized with random Gaussian values, \(B\) is initialized to zero — so at the start of training, \(\Delta W = 0\) and the model behaves identically to the pretrained version.
:::

### Why Does LoRA Work?

The key insight from the paper: the weight updates during fine-tuning have **low intrinsic rank**. When you fine-tune GPT-3's 12,288 × 12,288 attention matrices, the actual meaningful update can be captured by a rank-8 approximation. This means the "useful" part of fine-tuning lives in a tiny subspace.

```
            Pretrained Weight W₀           LoRA Decomposition
            ┌──────────────────┐           ┌──┐   ┌──────────────────┐
            │                  │           │  │   │                  │
            │   d × k matrix   │    +      │B │ × │    A (r × k)     │
            │   (frozen)       │           │  │   │                  │
            │                  │           │d×│   └──────────────────┘
            │                  │           │r │
            └──────────────────┘           └──┘
            Parameters: d × k              Parameters: d×r + r×k
            e.g. 4096 × 4096 = 16.7M      e.g. 4096×16 + 16×4096 = 131K
```

With rank \(r = 16\) on a 4096×4096 matrix, LoRA uses **131K parameters** instead of **16.7M** — a 128× reduction.

```python title="LoRA from scratch (simplified)"
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output

# Example: wrap a frozen linear layer
original = nn.Linear(4096, 4096)
lora_layer = LoRALinear(original, rank=16, alpha=32.0)

base_params = sum(p.numel() for p in original.parameters())
lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"Base layer params:     {base_params:,}")      # 16,781,312
print(f"LoRA trainable params: {lora_params:,}")       # 131,072
print(f"Reduction:             {base_params / lora_params:.0f}×")
```

:::tip[Line-by-Line Walkthrough]
- **`self.scaling = alpha / rank`** — The scaling factor controls how much influence the LoRA update has. A higher alpha relative to rank means bigger updates.
- **`self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)`** — Creates the first LoRA matrix (*A*) with small random values. Its shape is `(rank × in_features)` — much smaller than the full weight matrix.
- **`self.lora_B = nn.Parameter(torch.zeros(out_features, rank))`** — Creates the second LoRA matrix (*B*) initialized to all zeros. This means the LoRA update starts as zero — the model initially behaves exactly like the original.
- **`param.requires_grad = False`** — Freezes the original layer's weights so they won't be updated during training. Only *A* and *B* will learn.
- **`lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling`** — The LoRA forward pass: input *x* is multiplied by *A* (transposed), then by *B* (transposed), then scaled. This produces the small "patch" to the original output.
- **`return base_output + lora_output`** — The final output is the original layer's output plus the LoRA patch. At the start of training (when *B* is all zeros), this equals just the base output.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `lora_scratch.py`
2. Run: `python lora_scratch.py`

**Expected output:**
```
Base layer params:     16,781,312
LoRA trainable params: 131,072
Reduction:             128×
```

</details>

### The Alpha Scaling Factor

The `alpha` parameter controls the magnitude of the LoRA update relative to the pretrained weights. The effective update is scaled by \(\alpha / r\). Common practice:

- Set `alpha = 2 × rank` (e.g., rank=16, alpha=32)
- Higher alpha → larger updates → faster adaptation but more risk of instability
- When sweeping hyperparameters, fix alpha and vary rank

:::tip[Which Layers to Apply LoRA To?]
The original paper applies LoRA to the **query and value projection** matrices in attention (\(W_q\) and \(W_v\)). In practice, applying LoRA to **all linear layers** (Q, K, V, output projection, and MLP layers) gives better results with modest additional cost. The PEFT library lets you specify target modules by name.
:::

## QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit quantization, enabling fine-tuning of 65B-parameter models on a single 48GB GPU. The recipe:

1. **Quantize** the base model to 4-bit using NormalFloat4 (NF4) — a data type optimized for normally distributed weights
2. **Add LoRA adapters** in FP16/BF16 on top of the frozen 4-bit weights
3. **Train only the LoRA parameters** — gradients are computed in BF16, but the base model stays in 4-bit

:::info[QLoRA Memory Savings]
A 7B model in FP16 requires ~14 GB. In 4-bit, it shrinks to ~3.5 GB. The LoRA adapters add a few hundred MB. With QLoRA, you can fine-tune a 7B model on a single GPU with 8 GB of VRAM.
:::

```python title="QLoRA fine-tuning with PEFT and bitsandbytes"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — optimal for normal distributions
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants too
)

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for k-bit training (handles gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                       # Rank
    lora_alpha=32,              # Scaling factor
    target_modules=[            # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~6.8M || all params: ~1.2B || trainable%: 0.55%
```

:::tip[Line-by-Line Walkthrough]
- **`BitsAndBytesConfig(load_in_4bit=True, ...)`** — Configures 4-bit quantization. This compresses the model's weights from 16 bits down to 4 bits, slashing memory by ~4×.
- **`bnb_4bit_quant_type="nf4"`** — Uses NormalFloat4, a special 4-bit format tuned for the bell-curve shape of neural network weights. It gives better quality than naive 4-bit rounding.
- **`bnb_4bit_use_double_quant=True`** — "Quantizes the quantization constants" — an extra trick that squeezes out a bit more memory savings with virtually no quality cost.
- **`prepare_model_for_kbit_training(model)`** — Prepares the quantized model for training by enabling gradient checkpointing and fixing data type handling for backward passes.
- **`target_modules=[...]`** — Lists exactly which layers get LoRA adapters. Here we're targeting all the major projection layers in the attention and MLP blocks.
- **`model.print_trainable_parameters()`** — Prints a summary showing what fraction of parameters are trainable — typically less than 1%.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers peft bitsandbytes torch accelerate
```
You also need a CUDA-capable GPU with at least 6 GB VRAM.

**Steps:**
1. Save to `qlora_setup.py`
2. Run: `python qlora_setup.py`

**Expected output:**
```
trainable params: 6,815,744 || all params: 1,242,630,144 || trainable%: 0.5486
```

</details>

## Adapter Methods

LoRA isn't the only PEFT method. Several adapter architectures predate it and remain useful in specific scenarios.

### Houlsby Adapters

Houlsby et al. (2019) insert small **bottleneck modules** inside each Transformer layer. Each adapter has a down-projection, a nonlinearity, and an up-projection:

:::info[Plain English: What Is This Formula Doing?]
Think of an adapter like a tiny detour on a highway. The data (cars) exits the main road, goes through a narrow tunnel (down-projection), passes a checkpoint (nonlinearity), comes out through a wider exit (up-projection), and merges back onto the highway. The narrow tunnel forces the adapter to learn only the most essential adjustments, keeping it lightweight.
:::

\[
\text{Adapter}(x) = x + f(xW_{\text{down}})W_{\text{up}}
\]

**Reading the formula:** *x* is the input to the adapter (the data flowing through the Transformer layer). *W_down* is a matrix that squeezes the data into a smaller dimension (the narrow tunnel). *f* is a nonlinear activation function (like ReLU) that adds flexibility. *W_up* is a matrix that expands the data back to its original size. The whole thing is added back to the original *x* (the residual connection), so the adapter only needs to learn what to *change*, not the entire output.

The bottleneck dimension is typically 64 or 128, so each adapter adds very few parameters.

```
         ┌───────────────┐
    x ──►│ Layer Norm     │
         ├───────────────┤
         │ Down-project   │  d → bottleneck (e.g. 4096 → 64)
         │ Nonlinearity   │  ReLU or GELU
         │ Up-project     │  bottleneck → d (e.g. 64 → 4096)
         └───────┬───────┘
                 │
    x ──────────►+ ──────► output  (residual connection)
```

### Prefix Tuning

Prefix tuning (Li & Liang, 2021) prepends **learnable vectors** to the key and value sequences in every attention layer. The model's parameters stay completely frozen — only the prefix vectors are trained.

```python title="Prefix tuning concept"
import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    """Simplified prefix tuning for a single attention layer."""
    def __init__(self, num_heads: int, head_dim: int, prefix_len: int = 20,
                 num_layers: int = 1):
        super().__init__()
        self.prefix_len = prefix_len
        total_dim = num_heads * head_dim

        # Learnable prefix keys and values for each layer
        self.prefix_keys = nn.Parameter(torch.randn(num_layers, prefix_len, total_dim) * 0.01)
        self.prefix_values = nn.Parameter(torch.randn(num_layers, prefix_len, total_dim) * 0.01)

    def get_prefix(self, layer_idx: int, batch_size: int):
        """Returns prefix K, V for a given layer, expanded for batch."""
        pk = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
        pv = self.prefix_values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
        return pk, pv  # each: (B, prefix_len, d_model)

prefix = PrefixTuning(num_heads=32, head_dim=128, prefix_len=20, num_layers=32)
params = sum(p.numel() for p in prefix.parameters())
print(f"Prefix tuning params: {params:,}")  # Much less than full model
```

:::tip[Line-by-Line Walkthrough]
- **`self.prefix_keys = nn.Parameter(torch.randn(...) * 0.01)`** — Creates learnable "fake" key vectors that will be prepended to the real keys in each attention layer. They start with small random values.
- **`self.prefix_values = nn.Parameter(torch.randn(...) * 0.01)`** — Same idea for value vectors. Together, these prefix keys and values let the model "steer" its attention without changing any real weights.
- **`pk = self.prefix_keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)`** — Picks the prefix for a specific layer, then copies it across all items in the batch. Every example in the batch sees the same learned prefix.
- **`sum(p.numel() for p in prefix.parameters())`** — Counts total learnable parameters. For 32 layers × 20 prefix tokens × 4096 dims × 2 (keys+values), this is much smaller than the full model.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `prefix_tuning.py`
2. Run: `python prefix_tuning.py`

**Expected output:**
```
Prefix tuning params: 5,242,880
```

</details>

### Prompt Tuning

Prompt tuning (Lester et al., 2021) is even simpler: prepend a small number of **learnable embedding vectors** to the input. Unlike prefix tuning, these soft prompts only affect the first layer's input — they don't inject into every layer's attention.

```python title="Prompt tuning concept"
class PromptTuning(nn.Module):
    """Learnable soft prompt prepended to input embeddings."""
    def __init__(self, d_model: int, prompt_length: int = 20):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, d_model) * 0.01)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        # input_embeddings: (B, T, D)
        batch_size = input_embeddings.size(0)
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, D)
        return torch.cat([prompt, input_embeddings], dim=1)  # (B, P+T, D)

prompt_tuner = PromptTuning(d_model=4096, prompt_length=20)
params = sum(p.numel() for p in prompt_tuner.parameters())
print(f"Prompt tuning params: {params:,}")  # 81,920 — extremely lightweight
```

:::tip[Line-by-Line Walkthrough]
- **`self.soft_prompt = nn.Parameter(torch.randn(prompt_length, d_model) * 0.01)`** — Creates a small set of learnable embedding vectors (20 vectors, each of dimension 4096). These are the "soft prompt" that the model will learn to use as steering instructions.
- **`prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)`** — Copies the soft prompt for every item in the batch, since each example needs its own copy.
- **`torch.cat([prompt, input_embeddings], dim=1)`** — Glues the soft prompt vectors in front of the real input embeddings. The model sees 20 learned "tokens" before the actual text.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `prompt_tuning.py` (add `import torch` and `import torch.nn as nn` at the top)
2. Run: `python prompt_tuning.py`

**Expected output:**
```
Prompt tuning params: 81,920
```

</details>

## When to Use Which Approach

| Method | Trainable Params | GPU Memory | Quality | Best For |
|---|---|---|---|---|
| **Full fine-tuning** | 100% | Very high | Highest | Unlimited compute, large dataset, major domain shift |
| **LoRA** | 0.1–1% | Moderate | Near full FT | Most fine-tuning tasks, good default choice |
| **QLoRA** | 0.1–1% | Low | Close to LoRA | Limited GPU memory, prototyping |
| **Adapters** | 1–5% | Moderate | Good | Multi-task learning with shared backbone |
| **Prefix tuning** | <0.1% | Low | Good | When you need many task-specific adaptations |
| **Prompt tuning** | <0.01% | Very low | Moderate | Large models (175B+), simple adaptations |

:::tip[The Practical Default]
For most practitioners in 2024–2026, **QLoRA with rank 16–64** applied to all linear layers is the sweet spot. It works on consumer GPUs, trains fast, and achieves quality within 1–2% of full fine-tuning. Start here and only move to full fine-tuning if QLoRA doesn't meet your quality bar.
:::

## Practical: Fine-Tuning with HuggingFace PEFT

Let's walk through a complete fine-tuning pipeline using the PEFT library and the `trl` (Transformer Reinforcement Learning) library's `SFTTrainer`.

```python title="Complete QLoRA fine-tuning pipeline"
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# --- 1. Load quantized model ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Use Flash Attention if available
)
model = prepare_model_for_kbit_training(model)

# --- 2. Configure LoRA ---
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",   # Apply to every linear layer
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 3. Load dataset ---
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

def format_chat(example):
    """Format conversations into the model's chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_chat)

# --- 4. Training arguments ---
training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=200,
    optim="paged_adamw_8bit",     # Memory-efficient optimizer
    max_grad_norm=1.0,
    gradient_checkpointing=True,  # Trade compute for memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# --- 5. Train ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    max_seq_length=2048,
    packing=True,                  # Pack short sequences together
)

trainer.train()

# --- 6. Save adapter ---
trainer.save_model("./qlora-adapter")
# This only saves the LoRA weights (~50 MB), not the full model
```

:::tip[Line-by-Line Walkthrough]
- **`tokenizer.pad_token = tokenizer.eos_token`** — Sets the padding token to the end-of-sequence token. Many models don't define a pad token by default, and training will crash without one.
- **`attn_implementation="flash_attention_2"`** — Uses Flash Attention for faster, more memory-efficient attention computation during training.
- **`target_modules="all-linear"`** — A convenient shorthand that applies LoRA adapters to every linear layer in the model, not just the attention projections.
- **`tokenizer.apply_chat_template(...)`** — Formats raw message lists into the exact text format the model was pretrained on (with special tokens for `[INST]`, roles, etc.).
- **`optim="paged_adamw_8bit"`** — Uses a memory-efficient 8-bit version of the AdamW optimizer, which saves GPU memory by storing optimizer states in lower precision.
- **`packing=True`** — Packs multiple short training examples into a single sequence to avoid wasting computation on padding tokens.
- **`trainer.save_model("./qlora-adapter")`** — Saves only the tiny LoRA adapter weights (~50 MB), not the full model. You can later load these on top of the base model.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers peft bitsandbytes trl datasets torch accelerate flash-attn
```
You need a CUDA GPU with at least 8 GB VRAM. If Flash Attention isn't available, remove the `attn_implementation` line.

**Steps:**
1. Save to `qlora_pipeline.py`
2. Log into HuggingFace: `huggingface-cli login`
3. Run: `python qlora_pipeline.py`

**Expected output:**
Training progress bars showing decreasing loss over the training steps, followed by the adapter being saved to `./qlora-adapter/`. Training on 5000 examples should take roughly 20–60 minutes depending on your GPU.

</details>

:::warning[Common Fine-Tuning Mistakes]
1. **Learning rate too high.** Pretrained models need gentle updates. Start at 1e-4 to 2e-4 for LoRA, 1e-5 to 2e-5 for full fine-tuning.
2. **Too many epochs.** LLMs overfit fast on small datasets. 1–3 epochs is usually sufficient. Monitor eval loss.
3. **No chat template.** If fine-tuning a chat model, always use the model's chat template to format data. Wrong formatting = garbage output.
4. **Forgetting gradient checkpointing.** This halves memory usage with ~30% training slowdown. Always enable it for large models.
:::

### Loading and Merging LoRA Adapters

After training, you can load the adapter on top of the base model, or merge the adapter weights into the base model for deployment.

```python title="Loading and merging LoRA adapters"
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model + adapter separately
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")

# Option 1: Use with adapter (keeps base + adapter separate)
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Option 2: Merge adapter into base model (for deployment)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
# Now it's a standard model — no PEFT dependency needed at inference
```

:::tip[Line-by-Line Walkthrough]
- **`PeftModel.from_pretrained(base_model, "./qlora-adapter")`** — Loads the small LoRA adapter and attaches it on top of the base model. The adapter file is typically ~50 MB while the base model is ~2 GB+.
- **`model.merge_and_unload()`** — Mathematically merges the LoRA weights into the base model weights (W₀ + BA), producing a single standard model. After merging, inference is exactly as fast as the original model — no LoRA overhead.
- **`merged_model.save_pretrained("./merged-model")`** — Saves the merged model as a standard HuggingFace model. Anyone can load it without knowing PEFT was used.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers peft torch accelerate
```
You need a previously trained LoRA adapter saved at `./qlora-adapter`.

**Steps:**
1. Save to `merge_adapter.py`
2. Run: `python merge_adapter.py`

**Expected output:**
The model will generate a response to "Explain quantum computing" and then save the merged model to `./merged-model/`. The merged directory will contain the full model weights (~2-4 GB) plus tokenizer files.

</details>

---

## Exercises

<ExerciseBlock title="Exercise 1: LoRA Rank Ablation" difficulty="intermediate" hints={["Try ranks: 4, 8, 16, 32, 64, 128", "Use a small model like Llama-3.2-1B and a small dataset", "Track both training loss convergence speed and final eval quality", "Plot trainable parameters vs. evaluation metric for each rank"]}>

Fine-tune the same model on the same dataset with LoRA at ranks 4, 8, 16, 32, 64, and 128. For each rank, record: number of trainable parameters, training time, and final evaluation loss. Plot the trade-off curve. At what rank do returns diminish?

</ExerciseBlock>

<ExerciseBlock title="Exercise 2: Implement LoRA From Scratch" difficulty="advanced" hints={["Subclass nn.Module and wrap nn.Linear", "Initialize A with Kaiming uniform, B with zeros", "Don't forget the scaling factor alpha/r", "Write a function that patches all linear layers in a model"]}>

Implement a complete LoRA wrapper module from scratch (without using the PEFT library). Your implementation should:
1. Wrap any `nn.Linear` layer with a LoRA adapter
2. Support configurable rank and alpha
3. Include a function that walks a model's modules and replaces target layers with LoRA versions
4. Match the output of the PEFT library on the same model and inputs

</ExerciseBlock>

<ExerciseBlock title="Exercise 3: Compare PEFT Methods" difficulty="advanced" hints={["Use PEFT library — it supports LoRA, prefix tuning, prompt tuning, and adapters", "Keep the same model, dataset, and training budget", "Compare: trainable params, training speed, eval loss, inference latency", "For prefix tuning, try prefix lengths of 10, 20, and 50"]}>

Using the HuggingFace PEFT library, fine-tune the same base model on the same dataset using four different methods: LoRA, prefix tuning, prompt tuning, and Houlsby adapters. Compare them on training speed, memory usage, and final quality. Which method would you recommend for (a) a single high-stakes deployment, (b) serving 100 different customers with personalized models?

</ExerciseBlock>

---

## Resources

<ResourceCard title="LoRA: Low-Rank Adaptation of Large Language Models" url="https://arxiv.org/abs/2106.09685" type="paper" author="Hu et al., 2021" description="The original LoRA paper — elegant, well-written, and practical." />

<ResourceCard title="QLoRA: Efficient Finetuning of Quantized LLMs" url="https://arxiv.org/abs/2305.14314" type="paper" author="Dettmers et al., 2023" description="QLoRA enables 65B model fine-tuning on a single GPU. A landmark in efficiency." />

<ResourceCard title="HuggingFace PEFT Library" url="https://huggingface.co/docs/peft" type="tool" description="The go-to library for parameter-efficient fine-tuning — supports LoRA, QLoRA, prefix tuning, prompt tuning, and more." />

<ResourceCard title="Parameter-Efficient Transfer Learning for NLP" url="https://arxiv.org/abs/1902.00751" type="paper" author="Houlsby et al., 2019" description="The original adapter paper that started the PEFT revolution." />

<ResourceCard title="Prefix-Tuning: Optimizing Continuous Prompts for Generation" url="https://arxiv.org/abs/2101.00190" type="paper" author="Li & Liang, 2021" description="Prefix tuning — prepend learnable vectors to attention layers instead of modifying weights." />

<ResourceCard title="Practical Tips for Finetuning LLMs" url="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms" type="tutorial" author="Sebastian Raschka" description="Excellent hands-on guide covering data preparation, hyperparameters, and common pitfalls." />
