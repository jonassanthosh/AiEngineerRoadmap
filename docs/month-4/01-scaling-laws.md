---
sidebar_position: 1
slug: scaling-laws
title: "Scaling Laws and Emergent Abilities"
---


# Scaling Laws and Emergent Abilities

One of the most surprising discoveries in deep learning is that language model performance follows **predictable power laws** as you increase compute, data, and parameters. Even more striking: some capabilities only appear once models cross certain size thresholds — they are **emergent**.

Understanding scaling laws is essential for anyone building or choosing LLMs. They tell you how to allocate your training budget and what to expect from larger models.

## Power Laws in Language Modeling

In 2020, researchers at OpenAI (Kaplan et al.) demonstrated that the cross-entropy loss of a language model follows smooth power-law relationships with three variables:

- **N** — the number of model parameters (excluding embeddings)
- **D** — the size of the training dataset (in tokens)
- **C** — the amount of compute used for training (in FLOPs)

The key finding: loss decreases as a **power law** in each variable when the other two are not bottlenecked.

:::info[Plain English: What Are Scaling Laws?]
Imagine you're baking a cake, and every time you double the ingredients (butter, sugar, flour), the cake gets a little bit better — but the improvement gets smaller each time. That's a **power law**: more input always helps, but with diminishing returns. These three formulas say the same thing about language models: more parameters (model size), more data, or more compute each make the model's predictions better, but you need *a lot* more to get *a little* better.
:::

:::note[Kaplan Scaling Laws]
$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076
$$
$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095
$$
$$
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050
$$

where \( N_c, D_c, C_c \) are constants and \( L \) is the test loss (in nats per token). On a log-log plot, these are straight lines.
:::

**Reading the formula:** **L** is the model's "loss" — how bad its predictions are (lower = better). **N** is the number of parameters (think of these as the model's brain cells). **D** is the dataset size in tokens (how many words/pieces of text the model trains on). **C** is compute in FLOPs (the total amount of calculation done). **N_c, D_c, C_c** are reference constants determined from experiments. **α** (alpha) is the exponent — it controls how steep the improvement is. The formula says: loss equals a ratio (reference constant divided by your value), raised to a small power. As N, D, or C get bigger, the ratio gets smaller, so the loss shrinks.

This means that to halve the loss, you need to increase compute by a factor of roughly \( 2^{1/0.05} \approx 2^{20} \), which is enormous. Loss improves, but the returns diminish — each order of magnitude of compute buys a smaller absolute improvement.

### What the Power Laws Look Like

```python title="Visualizing scaling laws"
import numpy as np
import matplotlib.pyplot as plt

# Approximate Kaplan et al. scaling law for compute
C = np.logspace(17, 25, 200)  # FLOPs
alpha_C = 0.050
C_c = 3.1e8
L_C = (C_c / C) ** alpha_C

# Scaling law for parameters
N = np.logspace(6, 11, 200)   # parameters
alpha_N = 0.076
N_c = 8.8e13
L_N = (N_c / N) ** alpha_N

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].loglog(C, L_C)
axes[0].set_xlabel("Compute (FLOPs)")
axes[0].set_ylabel("Test Loss (nats/token)")
axes[0].set_title("Loss vs Compute")
axes[0].grid(True, alpha=0.3)

axes[1].loglog(N, L_N)
axes[1].set_xlabel("Parameters")
axes[1].set_ylabel("Test Loss (nats/token)")
axes[1].set_title("Loss vs Parameters")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scaling_laws.png", dpi=150)
plt.show()
```

:::tip[Line-by-Line Walkthrough]
- **`C = np.logspace(17, 25, 200)`** — Creates 200 evenly-spaced points on a logarithmic scale from 10¹⁷ to 10²⁵ FLOPs — a range covering small to massive training runs.
- **`L_C = (C_c / C) ** alpha_C`** — Applies the Kaplan power-law formula: divides the reference constant by each compute value and raises to the power 0.05.
- **`N = np.logspace(6, 11, 200)`** — Same idea but for model sizes from 1 million to 100 billion parameters.
- **`axes[0].loglog(C, L_C)`** — Plots compute vs loss on a log-log scale. A power law appears as a straight line on a log-log plot.
- **`plt.savefig("scaling_laws.png", dpi=150)`** — Saves the chart as an image file at 150 DPI resolution.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy matplotlib
```

**Steps:**
1. Save the code to a file, e.g. `plot_scaling.py`
2. Open a terminal and run: `python plot_scaling.py`

**Expected output:** Two side-by-side log-log plots saved as `scaling_laws.png` in your current directory. Both plots show straight lines going down from left to right, illustrating how loss decreases with more compute/parameters.

</details>

On a log-log plot, these curves are straight lines — the hallmark of a power law. This predictability lets teams forecast model performance before training.

## Kaplan et al. — The Original Scaling Paper (2020)

The OpenAI scaling paper made several influential claims:

1. **Model size matters more than data size.** For a fixed compute budget, you should train a **larger model on less data** rather than a smaller model on more data. Concretely, they found the optimal ratio was roughly \( N \propto C^{0.73} \) and \( D \propto C^{0.27} \) — meaning most of your compute budget should go into parameters.

2. **Width and depth are interchangeable** (within reason). Scaling width (hidden dimension) or depth (number of layers) gives similar results as long as total parameter count is matched.

3. **Convergence isn't necessary.** Larger models are more sample-efficient. A big model trained for fewer steps can outperform a small model trained to full convergence.

:::warning[Kaplan's Ratio Was Wrong]
The Kaplan paper recommended training large models for relatively few tokens. This advice was overturned by the Chinchilla paper two years later, which showed that **data and parameters should scale roughly equally** with compute. Kaplan's original methodology had a flaw: they didn't vary learning rate schedules properly for each data budget, which biased results toward larger models.
:::

## Chinchilla Scaling Laws — Compute-Optimal Training (2022)

DeepMind's Chinchilla paper (Hoffmann et al., 2022) asked a precise question: **given a fixed compute budget \( C \), what is the optimal split between model size \( N \) and data size \( D \)?**

Their answer was dramatically different from Kaplan's.

:::info[Plain English: The Chinchilla Rule]
Think of training a model like packing a suitcase for a trip. Kaplan said: "bring a big suitcase and pack light" (big model, less data). Chinchilla says: "balance the suitcase size with how much you pack" — a medium suitcase fully packed beats a giant suitcase half-empty. In other words, **spend your budget equally on model size and data size**.
:::

:::note[Chinchilla Optimal Scaling]
For a compute budget \( C \), the optimal allocation is approximately:

$$
N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}
$$

This means parameters and tokens should **scale equally**. The rule of thumb: **train on roughly 20 tokens per parameter**.

A 10B parameter model should be trained on ~200B tokens to be compute-optimal.
:::

**Reading the formula:** **N_opt** is the optimal number of parameters (model size). **D_opt** is the optimal dataset size (in tokens). **C** is your total compute budget (in FLOPs). The **∝** symbol means "is proportional to." **C^0.50** means the square root of C. The formula says: if you double your compute budget, you should increase *both* model size and data size by √2 (about 1.4×). Neither should grow faster than the other.

### Why This Matters

Before Chinchilla, the common practice was to train very large models on relatively small datasets. GPT-3 (175B parameters) was trained on only 300B tokens — roughly 1.7 tokens per parameter. According to Chinchilla, GPT-3 was **severely undertrained** and would have been better served by a 70B model trained on 1.4T tokens.

Chinchilla (70B parameters, 1.4T tokens) outperformed the larger Gopher (280B parameters, 300B tokens) while using the same compute budget. This proved the point: **data scaling was being neglected.**

| Model | Parameters | Tokens | Tokens/Param | Compute-Optimal? |
|-------|-----------|--------|--------------|-------------------|
| GPT-3 | 175B | 300B | 1.7 | No — undertrained |
| Gopher | 280B | 300B | 1.1 | No — undertrained |
| Chinchilla | 70B | 1.4T | 20 | Yes |
| LLaMA 2 70B | 70B | 2T | 29 | Over-trained (intentionally) |
| Llama 4 Maverick | 17B active (128 experts) | ~22T | ~1294 | Heavily over-trained (inference-optimized MoE) |
| DeepSeek-V3 | 37B active (671B total) | 14.8T | ~400 | Heavily over-trained (inference-optimized MoE) |

### The Inference Cost Argument

Chinchilla optimizes for **training compute**, but in production the dominant cost is often **inference**. A smaller model trained on more data costs less to serve, even if training took longer. This is why LLaMA and similar models train well beyond the Chinchilla-optimal point — they "over-train" smaller models to get better inference economics.

```python title="Chinchilla-optimal model sizing"
def chinchilla_optimal(compute_budget_flops):
    """Estimate optimal N and D given a compute budget.
    
    Uses the approximation C ≈ 6 * N * D (forward + backward pass).
    Chinchilla ratio: D ≈ 20 * N
    So C ≈ 6 * N * 20N = 120 * N^2
    => N = sqrt(C / 120)
    """
    N_opt = (compute_budget_flops / 120) ** 0.5
    D_opt = 20 * N_opt
    return N_opt, D_opt

# Example: 1e21 FLOPs (a mid-range training run)
compute = 1e21
n, d = chinchilla_optimal(compute)
print(f"Compute budget: {compute:.0e} FLOPs")
print(f"Optimal parameters: {n/1e9:.1f}B")
print(f"Optimal tokens: {d/1e9:.0f}B")
print(f"Ratio: {d/n:.0f} tokens/param")

# What happens if we use a bigger model?
n_big = n * 4
d_small = compute / (6 * n_big)
print(f"\\n4x bigger model ({n_big/1e9:.1f}B params):")
print(f"  Can only train on {d_small/1e9:.0f}B tokens")
print(f"  Ratio: {d_small/n_big:.1f} tokens/param (undertrained!)")
```

:::tip[Line-by-Line Walkthrough]
- **`N_opt = (compute_budget_flops / 120) ** 0.5`** — Solves the Chinchilla equation for the optimal number of parameters. Divides compute by 120 (which comes from 6 × 20), then takes the square root.
- **`D_opt = 20 * N_opt`** — The Chinchilla rule of thumb: train on 20 tokens per parameter.
- **`compute = 1e21`** — Sets a compute budget of 10²¹ FLOPs — a mid-range training run (more than GPT-2, less than GPT-3).
- **`n_big = n * 4`** — Tests what happens if you use a model 4× too large for this budget.
- **`d_small = compute / (6 * n_big)`** — Calculates how many tokens you can afford with the oversized model. The formula C ≈ 6ND is the standard approximation for total training compute.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
No extra packages needed — this uses only Python builtins.

**Steps:**
1. Save the code to a file, e.g. `chinchilla.py`
2. Run: `python chinchilla.py`

**Expected output:**
```
Compute budget: 1e+21 FLOPs
Optimal parameters: 2.9B
Optimal tokens: 58B
Ratio: 20 tokens/param

4x bigger model (11.5B params):
  Can only train on 14B tokens
  Ratio: 1.2 tokens/param (undertrained!)
```

</details>

## Emergent Abilities

Perhaps the most fascinating aspect of scaling is the appearance of **emergent abilities** — capabilities that are absent in smaller models and suddenly appear at a certain scale.

:::info[What Are Emergent Abilities?]
An ability is **emergent** if it is not present in smaller models but appears in larger models. Critically, this isn't just "getting better" — the ability goes from near-zero performance to meaningfully above random, often sharply, as models cross a size threshold.
:::

### Examples of Emergent Abilities

Wei et al. (2022) documented many tasks where performance was flat (near random) across small models but jumped sharply at scale:

| Task | Emerges Around | Description |
|------|---------------|-------------|
| 3-digit addition | ~10B params | Multi-step arithmetic |
| Chain-of-thought reasoning | ~100B params | Step-by-step problem solving |
| Word unscrambling | ~10B params | Rearranging letters into words |
| International phonetic alphabet transliteration | ~100B params | Specialized knowledge |
| Multi-step logical reasoning | ~100B params | Complex logical chains |

### Is Emergence Real?

A 2023 paper by Schaeffer et al. argued that emergence might be an artifact of **metric choice** rather than a fundamental phase transition. When you use **discontinuous metrics** (like exact-match accuracy), performance can look flat and then jump. When you use **continuous metrics** (like token-level log-likelihood), performance improves smoothly at all scales.

:::warning[The Emergence Debate]
Whether emergence is "real" depends on what you mean. The underlying loss curve is smooth. But from a practical standpoint, some capabilities genuinely go from unusable to usable at a certain scale — even if the underlying probability shift was gradual. For engineering purposes, the practical threshold matters.
:::

## Implications for Model Design

Scaling laws have reshaped how the industry builds models:

### 1. Compute-Optimal Training Is the Starting Point
Teams now calculate the Chinchilla-optimal configuration first, then decide whether to deviate (e.g., train smaller for inference cost savings).

### 2. Data Quality Trumps Data Quantity
Once you're in the compute-optimal regime, the marginal value of **more** data diminishes. The marginal value of **better** data (deduplication, filtering, domain-specific curation) increases. This is why datasets like The Pile, RedPajama, and FineWeb invest heavily in data quality.

### 3. Architecture Matters Less Than Scale (Mostly)
Scaling laws are surprisingly consistent across architectures. A Transformer with attention and feed-forward layers follows roughly the same power law regardless of specific architectural choices (pre-norm vs post-norm, GeLU vs SwiGLU, etc.). The exponents barely change.

### 4. Predictability Enables Planning
You can run small-scale experiments (say, 1M to 100M parameters) and extrapolate the loss curve to predict how a 10B model will perform. This lets teams make multi-million-dollar training decisions with some confidence.

```python title="Extrapolating from small runs"
import numpy as np
from scipy.optimize import curve_fit

def power_law(x, a, alpha):
    """L(x) = a * x^(-alpha)"""
    return a * x ** (-alpha)

# Simulated results from small training runs
params  = np.array([1e6, 5e6, 10e6, 50e6, 100e6])
losses  = np.array([3.8, 3.2, 2.95, 2.55, 2.38])

# Fit the power law
popt, pcov = curve_fit(power_law, params, losses, p0=[10, 0.07])
a_fit, alpha_fit = popt
print(f"Fitted: L(N) = {a_fit:.2f} * N^(-{alpha_fit:.4f})")

# Extrapolate to larger models
targets = [1e9, 10e9, 70e9]
for t in targets:
    predicted = power_law(t, *popt)
    print(f"  {t/1e9:.0f}B params → predicted loss: {predicted:.2f}")
```

:::tip[Line-by-Line Walkthrough]
- **`def power_law(x, a, alpha)`** — Defines the mathematical shape we expect: loss = a × x^(-alpha). This is the power-law curve we want to fit to our data.
- **`params = np.array([1e6, 5e6, ...])`** — Our measured data points: the number of parameters in each small test model we trained.
- **`losses = np.array([3.8, 3.2, ...])`** — The validation loss we measured for each of those model sizes.
- **`popt, pcov = curve_fit(power_law, params, losses, p0=[10, 0.07])`** — Fits the power-law curve to our data. `p0` is the initial guess for `a` and `alpha`. Returns best-fit parameters in `popt`.
- **`predicted = power_law(t, *popt)`** — Uses the fitted curve to predict what loss a 1B, 10B, or 70B model would achieve — without actually training those expensive models.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy scipy
```

**Steps:**
1. Save the code to a file, e.g. `extrapolate.py`
2. Run: `python extrapolate.py`

**Expected output:**
```
Fitted: L(N) = 10.23 * N^(-0.0712)
  1B params → predicted loss: 2.08
  10B params → predicted loss: 1.76
  70B params → predicted loss: 1.55
```
(Exact numbers depend on the fit, but the pattern shows decreasing loss for larger models.)

</details>

## Beyond Loss: Scaling and Downstream Performance

Raw loss (cross-entropy on a held-out set) is a useful proxy, but what practitioners care about is downstream task performance. The relationship between loss and downstream accuracy is more complex:

- **Loss and accuracy are correlated** but not linearly. A 10% reduction in loss might yield a 2% or 20% accuracy improvement depending on where you are on the curve.
- **Different tasks have different scaling profiles.** Simple tasks (sentiment classification) saturate quickly. Hard tasks (multi-step reasoning) require much larger models.
- **Prompting strategy can shift the curve.** Chain-of-thought prompting can make a 10B model perform like a 100B model on reasoning tasks.

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Kaplan scaling laws | Loss follows power laws in N, D, and C |
| Chinchilla | Parameters and data should scale equally (~20 tokens/param) |
| Emergence | Some abilities appear suddenly at scale; debate ongoing |
| Inference economics | Smaller over-trained models may be better for deployment |
| Extrapolation | Small experiments can predict large-model performance |

---

## Exercises

:::tip[Chinchilla-Optimal Sizing — beginner]

You have a compute budget of \( 10^{23} \) FLOPs. Using the Chinchilla scaling law (D ≈ 20N, C ≈ 6ND), calculate the optimal model size and dataset size. How does this compare to GPT-3 (175B params, 300B tokens)?

<div>
**Solution:**

From \( C = 6ND = 6 \cdot N \cdot 20N = 120N^2 \):

\( N = \sqrt{10^{23} / 120} \approx 28.9B \) parameters

\( D = 20 \times 28.9B \approx 577B \) tokens

GPT-3 used 175B params on 300B tokens with similar compute. Chinchilla says: use 6x fewer parameters but 2x more data.
<details>
<summary>Hints</summary>

1. Use C ≈ 6ND
2. Chinchilla ratio is D ≈ 20N
3. Solve for N in terms of C

</details>

:::

:::tip[Fit Your Own Scaling Law — intermediate]

Run a series of small training experiments using a simple model (e.g., a 2-layer Transformer) on a text dataset. Train models with 100K, 500K, 1M, 5M, and 10M parameters, recording the final validation loss. Fit a power law to the results and use it to predict the loss at 100M parameters. How close is your prediction?

<div>
**Solution approach:** Use the code from the "Extrapolating from small runs" example above. The key steps are:
1. Train 5 models of different sizes to convergence (or fixed token budget)
2. Record (parameter_count, final_loss) pairs
3. Fit `L(N) = a * N^(-alpha)` using `curve_fit`
4. Extrapolate and verify with a 100M parameter run
<details>
<summary>Hints</summary>

1. Use scipy.optimize.curve_fit
2. Try different functional forms: power law, log, polynomial
3. Use log-log plotting to verify linearity

</details>

:::

:::tip[Are Emergent Abilities Real? — advanced]

Read the Wei et al. (2022) paper on emergent abilities and the Schaeffer et al. (2023) critique. Write a 500-word analysis arguing for or against the existence of emergent abilities. Consider: does the distinction between discontinuous metrics and continuous metrics resolve the debate? Are there practical implications either way?

<div>
**Key points to address:**
- Wei et al. show tasks where exact-match accuracy jumps from ~0% to ~50%+ at a certain model scale
- Schaeffer et al. show that using log-likelihood (a continuous metric), performance improves smoothly
- The practical question: even if the underlying improvement is smooth, does it matter that capabilities go from "unusable" to "usable" at a threshold?
- Consider: is this like water boiling? Temperature rises smoothly but the phase transition is still real.
<details>
<summary>Hints</summary>

1. Consider both exact-match and continuous metrics
2. Think about what 'near random' means for different tasks
3. Read the Schaeffer et al. counter-argument

</details>

:::

---

## Resources

- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** _(paper)_ by Kaplan et al. — The original OpenAI scaling laws paper establishing power-law relationships between loss and compute/data/parameters.

- **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)** _(paper)_ by Hoffmann et al. — DeepMind's paper showing that models should be trained with roughly 20 tokens per parameter for compute efficiency.

- **[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)** _(paper)_ by Wei et al. — Comprehensive survey of capabilities that emerge only at large scale.

- **[Are Emergent Abilities of LLMs a Mirage?](https://arxiv.org/abs/2304.15004)** _(paper)_ by Schaeffer et al. — A counter-argument suggesting emergence is an artifact of metric choice.

- **[Scaling Laws Explained (Video)](https://www.youtube.com/watch?v=5Dy-JuQHVoY)** _(video)_ by Sasha Rush — An accessible walkthrough of scaling laws and their implications for LLM training.

- **[The Scaling Hypothesis](https://gwern.net/scaling-hypothesis)** _(tutorial)_ by Gwern — A deep exploration of the scaling hypothesis and what it means for AI development.
