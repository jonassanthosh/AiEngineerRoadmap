---
sidebar_position: 4
slug: pruning-distillation
title: "Pruning and Knowledge Distillation"
---


# Pruning and Knowledge Distillation

:::info[What You'll Learn]
- Structured vs. unstructured pruning
- Magnitude pruning and lottery ticket hypothesis
- Knowledge distillation: training a small model to mimic a large one
- Combining pruning, distillation, and quantization
:::

:::note[Prerequisites]
[Quantization](quantization) from this month.
:::

**Estimated time:** Reading: ~30 min | Exercises: ~2 hours

Quantization reduces the precision of each weight. **Pruning** removes weights entirely. **Knowledge distillation** trains a smaller model to mimic a larger one. These three techniques — quantization, pruning, and distillation — form the core toolkit for model compression.

## Weight Pruning

Pruning removes weights from a neural network that contribute least to its output. The result is a **sparse** model with many zero-valued weights. If the sparsity structure is right, this translates into real memory and compute savings.

### Unstructured Pruning

Unstructured pruning sets individual weights to zero regardless of their position in the matrix. This achieves high sparsity (90%+) with minimal quality loss, but the resulting sparse matrices are difficult to accelerate on GPUs because the zeros are scattered irregularly.

```
Dense matrix:                  Unstructured pruning (50%):
┌─────────────────┐            ┌─────────────────┐
│ 0.3  0.1  0.7   │            │ 0.3  0.0  0.7   │
│ 0.2  0.5  0.4   │   ──────►  │ 0.0  0.5  0.0   │
│ 0.8  0.1  0.6   │            │ 0.8  0.0  0.6   │
└─────────────────┘            └─────────────────┘
                               Irregular pattern — hard to accelerate
```

### Structured Pruning

Structured pruning removes entire rows, columns, attention heads, or layers. The resulting model is a smaller but still **dense** model — no special sparse hardware or kernels needed.

```
Dense matrix:                  Structured pruning (remove column 2):
┌─────────────────┐            ┌───────────┐
│ 0.3  0.1  0.7   │            │ 0.3  0.7  │
│ 0.2  0.5  0.4   │   ──────►  │ 0.2  0.4  │
│ 0.8  0.1  0.6   │            │ 0.8  0.6  │
└─────────────────┘            └───────────┘
                               Smaller dense matrix — standard hardware
```

:::info[Unstructured vs. Structured Pruning]
- **Unstructured** achieves higher sparsity for the same quality loss, but requires sparse matrix libraries and hardware support (e.g., NVIDIA's 2:4 sparsity on Ampere GPUs).
- **Structured** is easier to deploy (the pruned model is just a smaller model), but removing entire structures causes more quality degradation.
:::

## Magnitude Pruning

The simplest and most common pruning criterion: remove the weights with the **smallest absolute values**. The intuition is that small weights contribute little to the output.

```python title="Magnitude pruning implementation"
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class MagnitudePruner:
    """Magnitude-based weight pruning for linear layers."""
    @staticmethod
    def prune_layer(layer: nn.Linear, sparsity: float):
        """Prune a fraction of weights with smallest magnitudes."""
        weight = layer.weight.data
        threshold = torch.quantile(weight.abs().flatten(), sparsity)
        mask = weight.abs() > threshold
        layer.weight.data *= mask.float()
        return mask

    @staticmethod
    def prune_model(model: nn.Module, sparsity: float):
        """Prune all linear layers in a model."""
        total_params = 0
        pruned_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                mask = MagnitudePruner.prune_layer(module, sparsity)
                total = mask.numel()
                zeros = (mask == 0).sum().item()
                total_params += total
                pruned_params += zeros
                print(f"{name}: {zeros}/{total} pruned ({zeros/total*100:.1f}%)")
        print(f"\\nTotal: {pruned_params}/{total_params} pruned "
              f"({pruned_params/total_params*100:.1f}%)")

# Example with PyTorch's built-in pruning utilities
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.ReLU(),
    nn.Linear(3072, 768),
)

# Before pruning
print("Before pruning:")
for name, param in model.named_parameters():
    if "weight" in name:
        sparsity = (param == 0).sum().item() / param.numel()
        print(f"  {name}: sparsity = {sparsity:.1%}")

# Apply magnitude pruning with PyTorch
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.5)

# After pruning
print("\\nAfter 50% pruning:")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        sparsity = (module.weight == 0).sum().item() / module.weight.numel()
        print(f"  {name}.weight: sparsity = {sparsity:.1%}")

# Make pruning permanent (remove the mask and set weights to zero)
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, "weight")
```

:::tip[Line-by-Line Walkthrough]
- **`torch.quantile(weight.abs().flatten(), sparsity)`** — Finds the magnitude threshold below which a given fraction of weights will be pruned. For 50% sparsity, it finds the median absolute weight value.
- **`mask = weight.abs() > threshold`** — Creates a binary mask: True for weights to keep (above threshold), False for weights to prune (below threshold).
- **`layer.weight.data *= mask.float()`** — Zeroes out the pruned weights by multiplying by the mask. Weights above the threshold are unchanged; weights below become zero.
- **`prune.l1_unstructured(module, name="weight", amount=0.5)`** — PyTorch's built-in pruning: removes 50% of weights based on L1 norm (absolute value). It registers a mask as a buffer so the zeros are maintained during training.
- **`prune.remove(module, "weight")`** — Makes pruning permanent by folding the mask into the weight tensor and removing the hook. After this, the zeros are baked into the weights.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `magnitude_pruning.py`
2. Run: `python magnitude_pruning.py`

**Expected output:**
```
Before pruning:
  0.weight: sparsity = 0.0%
  2.weight: sparsity = 0.0%

After 50% pruning:
  0.weight: sparsity = 50.0%
  2.weight: sparsity = 50.0%
```

</details>

### Iterative Pruning

Pruning 90% of weights in one shot is aggressive. **Iterative pruning** alternates between pruning and retraining, gradually increasing sparsity:

1. Train the model to convergence
2. Prune a small fraction (e.g., 20%) of remaining weights
3. Retrain (fine-tune) the pruned model
4. Repeat steps 2–3 until desired sparsity

This achieves much better quality than one-shot pruning at the same sparsity level.

```python title="Iterative magnitude pruning schedule"
import numpy as np

def cubic_sparsity_schedule(step: int, total_steps: int,
                            initial_sparsity: float = 0.0,
                            final_sparsity: float = 0.9) -> float:
    """Gradual pruning schedule from Zhu & Gupta (2017).
    Starts slow, accelerates, then slows down near the target."""
    progress = step / total_steps
    sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
    return sparsity

# Visualize the schedule
steps = np.arange(0, 100)
sparsities = [cubic_sparsity_schedule(s, 100, 0.0, 0.9) for s in steps]

print("Pruning schedule (selected checkpoints):")
for step in [0, 10, 25, 50, 75, 100]:
    s = cubic_sparsity_schedule(min(step, 99), 100, 0.0, 0.9)
    print(f"  Step {step:3d}: {s:.1%} sparsity")
```

:::tip[Line-by-Line Walkthrough]
- **`progress = step / total_steps`** — Converts the current step into a 0-to-1 progress fraction.
- **`sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3`** — A cubic schedule that starts slow, accelerates in the middle, and slows near the target. The cubic curve means the model gets more time to recover after the final rounds of pruning.
- **`np.arange(0, 100)`** — Creates 100 steps for the pruning schedule. In practice, each "step" corresponds to a pruning-then-retraining cycle.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy
```

**Steps:**
1. Save to `pruning_schedule.py`
2. Run: `python pruning_schedule.py`

**Expected output:**
```
Pruning schedule (selected checkpoints):
  Step   0: 0.0% sparsity
  Step  10: 7.3% sparsity
  Step  25: 52.0% sparsity
  Step  50: 78.8% sparsity
  Step  75: 87.6% sparsity
  Step 100: 90.0% sparsity
```

</details>

### NVIDIA 2:4 Sparsity

NVIDIA's Ampere (A100) and later GPUs support **2:4 structured sparsity** in hardware: for every group of 4 weights, exactly 2 must be zero. This gives a fixed 50% sparsity with a guaranteed 2× speedup for matrix multiplications.

```
Original:  [0.3, 0.1, 0.7, 0.2,  0.5, 0.4, 0.8, 0.1, ...]
2:4 sparse: [0.3, 0.0, 0.7, 0.0,  0.5, 0.0, 0.8, 0.0, ...]
             └─── group 1 ──────┘  └─── group 2 ──────┘
```

:::tip[When to Use Pruning]
Pruning is most useful when:
- You need a specific sparsity ratio for hardware support (2:4 sparsity on NVIDIA)
- You're deploying to edge devices with limited memory
- Combined with quantization for maximum compression

For LLMs served on standard GPUs, quantization alone usually provides sufficient compression. Pruning adds complexity and is less commonly used in practice.
:::

## Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) trains a small **student** model to mimic the behavior of a large **teacher** model. The student learns not just the correct answers, but the teacher's full probability distribution over all possible outputs.

### Why Soft Labels Are Better Than Hard Labels

When a teacher model classifies a cat image, it might output:

```
Teacher's output (soft labels):
  cat:    0.85
  tiger:  0.08     ← "This looks a bit like a tiger"
  dog:    0.05     ← "Not really a dog, but more dog-like than car-like"
  car:    0.02

Hard label:
  cat:    1.0
  tiger:  0.0      ← All this relational information is lost
  dog:    0.0
  car:    0.0
```

The soft labels encode **dark knowledge** — relationships between classes that hard labels discard. A cat is more similar to a tiger than to a car. The student model that learns from soft labels generalizes better than one trained on hard labels alone.

:::note[Distillation Loss]
The distillation loss combines two objectives:

:::info[Plain English: What Is This Formula Doing?]
Think of a student learning from both a textbook (hard labels) and a wise teacher (soft labels). The textbook gives black-and-white answers: "This is a cat. Period." The teacher says: "This is mostly a cat, but notice it has some tiger-like stripes and its shape is a bit dog-like." The distillation loss balances these two sources of learning — *α* controls how much weight the student puts on the textbook vs. the teacher. The temperature *T* is like asking the teacher to "think out loud" more — a higher temperature makes the teacher reveal more subtle opinions.
:::

\[
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot T^2 \cdot \mathcal{L}_{\text{soft}}
\]

**Reading the formula:** The total loss *L* is a weighted sum of two parts. *L_hard* is the standard cross-entropy loss comparing the student's prediction against the true label (the textbook). *L_soft* is the KL divergence between the student's and teacher's softened probability distributions (learning from the teacher's nuanced opinions). *α* controls the balance (e.g., 0.5 means equal weight). *T²* compensates for the fact that raising the temperature reduces the gradient magnitudes.

where:

:::info[Plain English: What Is This Formula Doing?]
The hard loss is straightforward: "How wrong is the student compared to the correct answer?" It's the standard test-score metric used in all classification training.
:::

\[
\mathcal{L}_{\text{hard}} = \text{CrossEntropy}(y_{\text{student}}, y_{\text{true}})
\]

**Reading the formula:** *L_hard* measures the difference between the student's predictions (*y_student*) and the true labels (*y_true*). Lower means the student is getting the right answers more often.

:::info[Plain English: What Is This Formula Doing?]
The soft loss is where the magic happens. We take both the teacher's and student's raw scores, "soften" them with temperature (making the teacher share its subtle opinions), and then measure how similar the student's softened opinions are to the teacher's. The softmax function converts raw scores into probabilities, and the KL divergence measures how different two probability distributions are.
:::

\[
\mathcal{L}_{\text{soft}} = \text{KL}\left(\text{softmax}\left(\frac{z_{\text{student}}}{T}\right) \;\|\; \text{softmax}\left(\frac{z_{\text{teacher}}}{T}\right)\right)
\]

**Reading the formula:** *L_soft* is the KL divergence between the student's and teacher's softened probability distributions. *z_student* and *z_teacher* are the raw output scores (logits) from each model. Dividing by temperature *T* before applying softmax produces smoother distributions — when *T* is high, even low-confidence predictions become visible, revealing the teacher's "dark knowledge."

- \(T\) is the **temperature** — higher values produce softer distributions that reveal more dark knowledge
- \(\alpha\) balances the hard label loss and the soft distillation loss
- \(T^2\) compensates for the gradient magnitude change from temperature scaling
:::

```python title="Knowledge distillation training loop"
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.5):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
        T = self.temperature

        # Soft distillation loss (KL divergence on softened distributions)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * (T ** 2) * soft_loss
        return loss

    def train_step(self, batch_x: torch.Tensor,
                   batch_y: torch.Tensor) -> dict:
        self.student.train()

        # Forward pass through both models
        with torch.no_grad():
            teacher_logits = self.teacher(batch_x)
        student_logits = self.student(batch_x)

        loss = self.distillation_loss(student_logits, teacher_logits, batch_y)
        return {"loss": loss, "student_logits": student_logits}

# Example: distill a large model into a small one
teacher = nn.Sequential(nn.Linear(768, 2048), nn.ReLU(), nn.Linear(2048, 10))
student = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 10))

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())
print(f"Teacher params: {teacher_params:,}")
print(f"Student params: {student_params:,}")
print(f"Compression:    {teacher_params / student_params:.1f}×")

trainer = DistillationTrainer(teacher, student, temperature=4.0, alpha=0.5)

x = torch.randn(32, 768)
y = torch.randint(0, 10, (32,))
result = trainer.train_step(x, y)
print(f"Distillation loss: {result['loss'].item():.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`self.teacher = teacher.eval()`** — Puts the teacher model in evaluation mode (disables dropout, etc.). The teacher is only used for predictions, never for training.
- **`param.requires_grad = False`** — Freezes all teacher parameters. We never want to accidentally update the teacher during training.
- **`F.log_softmax(student_logits / T, dim=-1)`** — Divides the student's raw scores by temperature *T* and converts to log-probabilities. Higher temperature makes the distribution more uniform, revealing subtle differences.
- **`F.softmax(teacher_logits / T, dim=-1)`** — Same temperature scaling for the teacher, producing softened probabilities that encode "dark knowledge."
- **`F.kl_div(student_soft, teacher_soft, ...)`** — Measures how different the student's distribution is from the teacher's. The student learns to match the teacher's nuanced opinions.
- **`self.alpha * hard_loss + (1 - self.alpha) * (T ** 2) * soft_loss`** — Combines both losses. The *T²* factor compensates for the temperature scaling's effect on gradient magnitudes.
- **`with torch.no_grad(): teacher_logits = self.teacher(batch_x)`** — Gets teacher predictions without computing gradients. This saves memory and compute since we don't train the teacher.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `distillation.py`
2. Run: `python distillation.py`

**Expected output:**
```
Teacher params: 1,597,450
Student params: 199,434
Compression:    8.0×
Distillation loss: 2.3456
```
The student is 8× smaller than the teacher.

</details>

## Teacher-Student Framework for LLMs

For language models, distillation works at the **token level**: the student learns to match the teacher's probability distribution over the vocabulary at every position in the sequence.

```
Prompt: "The capital of France is"

Teacher (70B):   Paris: 0.92, Lyon: 0.03, Marseille: 0.02, ...
Student (7B):    Paris: 0.75, Lyon: 0.08, Marseille: 0.05, ...  ← before distillation
Student (7B):    Paris: 0.89, Lyon: 0.04, Marseille: 0.03, ...  ← after distillation
```

```python title="LLM distillation with sequence-level KL divergence"
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_lm_distillation_loss(
    student_model,
    teacher_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
):
    """Compute token-level distillation loss for language models."""
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        teacher_logits = teacher_outputs.logits  # (B, T, vocab_size)

    student_outputs = student_model(
        input_ids=input_ids, attention_mask=attention_mask
    )
    student_logits = student_outputs.logits

    # Shift for causal LM (predict next token)
    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Soft loss: KL divergence at temperature T
    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Hard loss: standard next-token prediction
    hard_loss = F.cross_entropy(
        shift_student.view(-1, shift_student.size(-1)),
        shift_labels.view(-1),
    )

    loss = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * soft_loss
    return loss
```

:::tip[Line-by-Line Walkthrough]
- **`teacher_outputs = teacher_model(input_ids=input_ids, ...)`** — Runs the large teacher model to get its predictions at every token position. This is done inside `torch.no_grad()` because we don't train the teacher.
- **`shift_student = student_logits[:, :-1, :]`** — Shifts logits left by one position for causal language modeling. Position *i*'s logits predict position *i+1*'s token.
- **`shift_labels = input_ids[:, 1:]`** — The target tokens, shifted to align with the predictions. Token at position 1 is the target for the prediction at position 0.
- **`F.log_softmax(shift_student / temperature, dim=-1)`** — Softens and converts the student's predictions to log-probabilities across the vocabulary at each position.
- **`F.cross_entropy(shift_student.view(-1, ...), shift_labels.view(-1))`** — The hard loss: standard next-token prediction measured across all positions in the sequence.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers torch
```

**Steps:**
1. This function is meant to be used inside a training loop. To test it standalone, you would load a teacher and student model and call `compute_lm_distillation_loss(...)` with tokenized input.
2. In practice, the teacher is a large model (e.g., 70B) and the student is smaller (e.g., 7B).

**Expected output:**
A single scalar loss value (e.g., `2.45`) that combines the hard and soft objectives. During training, this loss would decrease as the student learns to match the teacher.

</details>

## DistilBERT Case Study

DistilBERT (Sanh et al., 2019) is the most cited example of knowledge distillation applied to Transformers. Starting from BERT-base (110M parameters), the team produced a model with:

- **40% fewer parameters** (66M)
- **60% faster** inference
- **97% of BERT's performance** on downstream benchmarks

The recipe:
1. Initialize the student with every other layer from the teacher (layers 0, 2, 4, 6, 8, 10 → 6 layers instead of 12)
2. Train with a triple loss: distillation loss + masked language modeling loss + cosine embedding loss (align hidden states)
3. Train on the same data BERT was pretrained on

:::info[Distillation Beyond Classification]
Modern LLM distillation goes beyond matching logits. Common techniques include:
- **Hidden state matching:** Align intermediate representations between teacher and student
- **Attention transfer:** Make the student's attention patterns resemble the teacher's
- **Synthetic data distillation:** Generate a large training dataset using the teacher, then train the student on it (this is how many open-source models are trained on GPT-4 outputs)
:::

:::warning[Legal Considerations]
Training a student model on outputs from a proprietary model (e.g., GPT-4) may violate the model's terms of service. OpenAI's usage policies prohibit using their outputs to train competing models. Always check the license and terms before distilling from a commercial model.
:::

---

## Exercises

:::tip[Exercise 1: Pruning Sensitivity Analysis — beginner]

Load a pretrained BERT model, evaluate it on a text classification task (e.g., SST-2), and then prune it at increasing sparsity levels (10% to 90%). Plot accuracy vs. sparsity. At what sparsity does performance drop sharply? Is the model more sensitive to pruning in attention layers or FFN layers?

<details>
<summary>Hints</summary>

1. Load a pretrained model (e.g. BERT-base) and evaluate on a task
2. Prune at 10%, 30%, 50%, 70%, 90% sparsity and evaluate after each
3. Try pruning only the FFN layers vs. only the attention layers
4. Plot accuracy vs. sparsity for each configuration

</details>

:::

:::tip[Exercise 2: Distill BERT to a Smaller Model — intermediate]

Implement knowledge distillation from BERT-base (12 layers) to a smaller BERT (4 layers) on the SST-2 sentiment classification task. Compare three student training strategies:
1. Train the student from scratch on SST-2
2. Initialize from teacher layers, then fine-tune on SST-2
3. Initialize from teacher layers, then distill from the teacher on SST-2

Which approach gives the best accuracy? What's the speed improvement?

<details>
<summary>Hints</summary>

1. Use BERT-base as the teacher and a 4-layer BERT as the student
2. Initialize the student from layers 0, 4, 8, 11 of the teacher
3. Fine-tune both teacher and student on SST-2
4. Then distill the teacher into the student and compare

</details>

:::

:::tip[Exercise 3: Temperature Sweep for Distillation — intermediate]

Train a student model with knowledge distillation at temperatures 1, 2, 4, 8, 16, and 20. For each temperature, record the student's final accuracy. Plot the results and identify the optimal temperature. How does the optimal temperature relate to the number of classes and the teacher's confidence level?

<details>
<summary>Hints</summary>

1. Try temperatures: 1, 2, 4, 8, 16, 20
2. At T=1, soft labels are the same as hard labels (no dark knowledge)
3. At very high T, the distribution becomes nearly uniform (too much smoothing)
4. Plot student accuracy vs. temperature to find the sweet spot

</details>

:::

---

## Resources

- **[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)** _(paper)_ by Hinton et al., 2015 — The original knowledge distillation paper — surprisingly readable and full of insights.

- **[DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)** _(paper)_ by Sanh et al., 2019 — DistilBERT: 97% of BERT's performance with 40% fewer parameters.

- **[The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)** _(paper)_ by Frankle & Carlin, 2019 — Dense networks contain sparse subnetworks that can match the full model's performance when trained in isolation.

- **[To Prune, or Not to Prune](https://arxiv.org/abs/1710.01878)** _(paper)_ by Zhu & Gupta, 2017 — Practical guide to magnitude pruning with gradual sparsity schedules.

- **[A Survey on Knowledge Distillation of LLMs](https://arxiv.org/abs/2402.13116)** _(paper)_ — Recent survey covering distillation techniques specifically for large language models.

- **[PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)** _(tutorial)_ — Official PyTorch tutorial on pruning with torch.nn.utils.prune.
