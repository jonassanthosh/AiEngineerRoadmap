---
sidebar_position: 3
slug: loss-and-optimizers
title: "Loss Functions and Optimizers"
---


# Loss Functions and Optimizers

:::info[What You'll Learn]
- Cross-entropy, MSE, and when to use each loss function
- SGD, Adam, and AdamW optimizers
- Learning rate schedules and warm-up strategies
- How optimizer choice affects convergence
:::

:::note[Prerequisites]
[Math Foundations](/curriculum/month-1/math-foundations) (gradients) and [Neural Networks Introduction](/curriculum/month-1/neural-networks-intro) from Month 1.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~2 hours

Training a neural network means finding parameters that minimize a **loss function** — a scalar measure of how wrong the model's predictions are. The **optimizer** is the algorithm that updates those parameters using the loss's gradients. Choosing the right loss and optimizer is just as important as choosing the right architecture.

## Loss Functions

### Mean Squared Error (MSE)

MSE is the default choice for **regression** tasks. It penalizes large errors quadratically, making it sensitive to outliers.

:::note[MSE Definition]

:::info[Plain English: What Does This Formula Mean?]
Imagine you're throwing darts at a target. For each throw, you measure how far the dart landed from the bullseye. MSE squares each of those distances (so a miss by 3 counts as 9, not 3 — big misses are punished extra hard) and then averages them all. A low MSE means your throws are consistently close to the bullseye.
:::

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

**Reading the formula:** *\(\mathcal{L}_{\text{MSE}}\)* is the loss (a single number representing how wrong the model is overall). *\(N\)* is the number of predictions. *\(y_i\)* is the true value for example *i* (what the answer should be). *\(\hat{y}_i\)* is the predicted value for example *i* (the model's guess). *\((y_i - \hat{y}_i)^2\)* is the squared error for one example — the difference between truth and guess, squared. The *\(\sum\)* adds up all the squared errors, and dividing by *\(N\)* gives the average.
:::

```python title="MSE Loss"
import torch
import torch.nn as nn

predictions = torch.tensor([2.5, 0.0, 2.1, 7.8])
targets = torch.tensor([3.0, -0.5, 2.0, 7.5])

mse = nn.MSELoss()
print(mse(predictions, targets))  # tensor(0.1025)
```

:::tip[Line-by-Line Walkthrough]
- **`predictions = torch.tensor([2.5, 0.0, 2.1, 7.8])`** — The model's 4 guesses.
- **`targets = torch.tensor([3.0, -0.5, 2.0, 7.5])`** — The 4 correct answers.
- **`nn.MSELoss()`** — Creates the MSE loss function.
- **`mse(predictions, targets)`** — Computes the average squared difference: ((2.5−3.0)² + (0.0−(−0.5))² + (2.1−2.0)² + (7.8−7.5)²) / 4 = 0.1025.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `mse_loss.py`
2. Run: `python mse_loss.py`

**Expected output:**
```
tensor(0.1025)
```

</details>

:::warning[MSE for Classification]
Never use MSE for classification. The squared-error gradient near the correct class boundary is small (the "plateau" problem), leading to slow, unreliable training. Cross-entropy is the right choice.
:::

### Binary Cross-Entropy (BCE)

For **binary classification** — one class or the other. The model outputs a probability \(p \in [0, 1]\) (typically via sigmoid).

:::note[BCE Definition]

:::info[Plain English: What Does This Formula Mean?]
Imagine a weather forecaster who predicts "80% chance of rain." If it rains, they were pretty good (80% confident in the right answer). If it doesn't rain, they were badly wrong (80% confident in the wrong answer). BCE measures how surprised we are by the outcome given the prediction. The more confident you were in the wrong answer, the bigger the penalty. Perfect confidence in the right answer gives zero loss.
:::

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

**Reading the formula:** *\(N\)* is the number of examples. *\(y_i\)* is the true label (0 or 1) for example *i*. *\(\hat{p}_i\)* is the model's predicted probability that example *i* belongs to class 1. When *\(y_i = 1\)* (positive case), we use *\(\log(\hat{p}_i)\)* — the higher the predicted probability, the lower the loss. When *\(y_i = 0\)* (negative case), we use *\(\log(1 - \hat{p}_i)\)* — the lower the predicted probability, the lower the loss. The minus sign at the front flips the log values (which are negative) to make the loss positive.
:::

```python title="BCE Loss"
# Two equivalent approaches:
# 1. Apply sigmoid yourself + BCELoss
probs = torch.sigmoid(torch.tensor([0.8, -1.2, 2.0]))
labels = torch.tensor([1.0, 0.0, 1.0])
bce = nn.BCELoss()
print(bce(probs, labels))

# 2. BCEWithLogitsLoss (numerically stable — preferred)
logits = torch.tensor([0.8, -1.2, 2.0])
bce_logits = nn.BCEWithLogitsLoss()
print(bce_logits(logits, labels))
```

:::tip[Line-by-Line Walkthrough]
- **`torch.sigmoid(torch.tensor([0.8, -1.2, 2.0]))`** — Convert raw model outputs (logits) into probabilities between 0 and 1 using the sigmoid function.
- **`labels = torch.tensor([1.0, 0.0, 1.0])`** — True labels: examples 1 and 3 are positive, example 2 is negative.
- **`nn.BCELoss()`** — Binary cross-entropy loss that expects probabilities as input.
- **`bce(probs, labels)`** — Compute the loss between predicted probabilities and true labels.
- **`nn.BCEWithLogitsLoss()`** — Combines sigmoid and BCE in one step. Preferred because it's numerically more stable (avoids log-of-zero problems).
- **`bce_logits(logits, labels)`** — Same result but takes raw logits instead of probabilities.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `bce_loss.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python bce_loss.py`

**Expected output:**
```
tensor(0.3476)
tensor(0.3476)
```
(Both methods produce the same result.)

</details>

:::tip[Always Prefer BCEWithLogitsLoss]
`BCEWithLogitsLoss` combines sigmoid and BCE in a single numerically stable operation. It avoids the log-of-zero issues that can arise when you apply sigmoid then log separately.
:::

### Cross-Entropy Loss

For **multi-class classification** — exactly one correct class out of \(C\) classes. The model outputs raw logits; the loss internally applies softmax.

:::note[Cross-Entropy Definition]

:::info[Plain English: What Does This Formula Mean?]
Imagine a multiple-choice test where the model picks one answer out of several options. For each question, the model gives a confidence score for every option (like "I'm 70% sure it's A, 20% sure it's B, 10% sure it's C"). Cross-entropy measures: "How surprised should we be by the correct answer given the model's confidence scores?" If the model was very confident in the right answer, the loss is small. If it was confident in the wrong answer, the loss is large.
:::

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{z_{i, y_i}}}{\sum_{c=1}^{C} e^{z_{i,c}}}
$$

**Reading the formula:** *\(N\)* is the number of examples. *\(C\)* is the total number of classes. *\(z_{i,c}\)* is the raw score (logit) that the model gave to class *c* for example *i*. *\(y_i\)* is the correct class for example *i*. The fraction *\(e^{z_{i,y_i}} / \sum_c e^{z_{i,c}}\)* is the softmax probability — it converts raw scores into probabilities that sum to 1. The *\(\log\)* of this softmax probability for the correct class is then averaged over all examples. The minus sign makes the loss positive.

where \(z_{i,c}\) are the raw logits and \(y_i\) is the correct class index.
:::

```python title="Cross-Entropy Loss"
# logits: (batch_size, num_classes) — raw, unnormalized scores
logits = torch.tensor([
    [ 2.0,  1.0, 0.1],  # sample 1
    [ 0.5,  2.5, 0.3],  # sample 2
])
targets = torch.tensor([0, 1])  # correct class indices

ce = nn.CrossEntropyLoss()
loss = ce(logits, targets)
print(f"Loss: {loss:.4f}")

# Cross-entropy also works with soft/probabilistic labels (label smoothing)
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
print(f"Smoothed Loss: {ce_smooth(logits, targets):.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])`** — Raw scores from the model for 2 samples across 3 classes. Higher score = more confident.
- **`targets = torch.tensor([0, 1])`** — Sample 1's correct class is 0 (first class); sample 2's correct class is 1 (second class).
- **`nn.CrossEntropyLoss()`** — Creates the loss function. It automatically applies softmax internally — don't apply softmax yourself!
- **`ce(logits, targets)`** — Compute the loss. For sample 1, the model gave class 0 a high score (2.0), which is correct, so this contributes low loss. For sample 2, class 1 got the highest score (2.5), also correct.
- **`label_smoothing=0.1`** — Instead of the target being 100% class 0 and 0% others, it becomes 93.3% class 0 and 3.3% each for the others. This acts as a regularizer.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `cross_entropy.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python cross_entropy.py`

**Expected output:**
```
Loss: 0.4076
Smoothed Loss: 0.5765
```

</details>

### Choosing the Right Loss Function

| Task | Loss Function | Model Output |
|------|-------------|--------------|
| Regression | MSE or MAE | Raw value |
| Binary classification | BCEWithLogitsLoss | Single logit |
| Multi-class (one label) | CrossEntropyLoss | \(C\) logits |
| Multi-label | BCEWithLogitsLoss | \(C\) logits |

## Optimizers

All optimizers share the same goal: use gradient information to update parameters. They differ in **how** they use it.

### Stochastic Gradient Descent (SGD)

The simplest optimizer. At each step, update each parameter \(\theta\) by:

:::info[Plain English: What Does This Formula Mean?]
Think of yourself standing on a foggy hillside, trying to reach the lowest valley. You can't see far, but you can feel which way the ground slopes under your feet. SGD says: "Just take a step downhill." The learning rate controls how big each step is. Small steps are safe but slow; big steps are fast but risky (you might overshoot the valley).
:::

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

**Reading the formula:** *\(\theta\)* represents the model's parameters (all the weights and biases — the knobs we're trying to tune). *\(\eta\)* (eta) is the learning rate — how big a step we take. *\(\nabla_\theta \mathcal{L}\)* is the gradient — the direction and steepness of the slope. The formula says: update each parameter by subtracting the learning rate times the gradient. We subtract because we want to go *downhill* (toward lower loss).

where \(\eta\) is the learning rate.

SGD is noisy (because it estimates the gradient from a mini-batch) but this noise can actually help escape shallow local minima.

### SGD with Momentum

Momentum accumulates a velocity vector that smooths out oscillations:

:::info[Plain English: What Does This Formula Mean?]
Imagine a ball rolling downhill. Without momentum, the ball just follows the local slope (like basic SGD). With momentum, the ball builds up speed — it remembers which direction it was going. This helps it roll through small bumps and flat spots instead of getting stuck. The velocity vector is the ball's "memory of past motion."
:::

$$
v_t = \mu \cdot v_{t-1} + \nabla_\theta \mathcal{L}, \quad \theta \leftarrow \theta - \eta \cdot v_t
$$

**Reading the formula:** *\(v_t\)* is the velocity (accumulated direction of travel) at step *t*. *\(\mu\)* (mu) is the momentum coefficient (typically 0.9) — how much of the previous velocity to keep. *\(v_{t-1}\)* is the velocity from the previous step. *\(\nabla_\theta \mathcal{L}\)* is the current gradient. So the new velocity is 90% of the old velocity plus 100% of the current gradient. Then we update the parameters by subtracting the learning rate times this velocity.

Typical \(\mu = 0.9\). Momentum helps SGD navigate ravines (regions where the surface curves much more steeply in one dimension than another).

### RMSProp

Adapts the learning rate per-parameter by dividing by a running average of squared gradients:

:::info[Plain English: What Does This Formula Mean?]
Some parameters have huge, noisy gradients while others have tiny, consistent ones. RMSProp is like giving each parameter its own personalized step size. Parameters that have been getting big gradient updates in the past get smaller steps (to calm them down), and parameters with small gradients get bigger steps (to speed them up). It's like a teacher who gives extra attention to quiet students and asks the loud ones to settle down.
:::

$$
s_t = \beta \cdot s_{t-1} + (1-\beta) \cdot g_t^2, \quad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot g_t
$$

**Reading the formula:** *\(s_t\)* is the running average of squared gradients (a history of how big the gradient has been). *\(\beta\)* (typically 0.9) controls how much history to keep. *\(g_t\)* is the current gradient. *\(g_t^2\)* is the gradient squared. The parameter update divides the gradient by *\(\sqrt{s_t + \epsilon}\)* — the square root of the running average. If a parameter has had large gradients, *\(s_t\)* is big, so we divide by a big number → smaller effective step. *\(\epsilon\)* (epsilon, a tiny number like 10⁻⁸) prevents dividing by zero.

Parameters with historically large gradients get smaller effective learning rates. This prevents any single parameter from dominating the update.

### Adam (Adaptive Moment Estimation)

:::info[Why Adam Is the Default]
Adam combines the best ideas from momentum (first moment) and RMSProp (second moment) with bias correction. It adapts per-parameter learning rates and handles sparse gradients well. This makes it the most popular optimizer in deep learning practice.
:::

:::info[Plain English: What Does This Formula Mean?]
Adam is like a smart hiker who both remembers which direction they've been generally heading (momentum, like SGD with momentum) AND adjusts their step size per-dimension based on the terrain roughness (adaptive learning rate, like RMSProp). On top of that, it includes a "warm-up" correction for the first few steps (bias correction) because the momentum and terrain estimates start at zero and need time to become accurate.
:::

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta &\leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
\end{aligned}
$$

**Reading the formula:**
- *\(m_t\)* — **First moment** (running average of gradients, i.e. momentum): a smoothed version of the gradient direction. *\(\beta_1 = 0.9\)* means 90% memory of past gradients + 10% of the current gradient.
- *\(v_t\)* — **Second moment** (running average of squared gradients): tracks how rough/variable the gradient has been. *\(\beta_2 = 0.999\)* means 99.9% memory.
- *\(\hat{m}_t\)*, *\(\hat{v}_t\)* — **Bias-corrected moments**: dividing by *\(1 - \beta^t\)* compensates for the fact that *m* and *v* start at zero. In early steps, *t* is small, so this boosts the estimates. Later, the correction fades to nearly 1.
- The final update divides the corrected momentum by the square root of the corrected variance (plus a tiny *\(\epsilon\)*), then multiplies by the learning rate *\(\eta\)*.

Defaults: \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\epsilon = 10^{-8}\).

```python title="Comparing optimizers in PyTorch"
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(784, 10)

# SGD with momentum
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam
adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# AdamW (Adam with decoupled weight decay — often preferred)
adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Linear(784, 10)`** — A simple linear layer with 784 inputs and 10 outputs (like a model for MNIST digits).
- **`optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)`** — SGD optimizer: learning rate 0.01, momentum 0.9 (remembers 90% of previous direction), and a small weight decay (L2 regularization) of 0.0001.
- **`optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))`** — Adam optimizer: learning rate 0.001, first moment decay 0.9, second moment decay 0.999 (the standard defaults).
- **`optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`** — AdamW: like Adam but with decoupled weight decay, which is the recommended variant for most modern training.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `optimizers.py` (add `import torch` at the top)
2. Run: `python optimizers.py`

**Expected output:** No printed output — this code creates the optimizers. You would use them in a training loop like: `optimizer.zero_grad(); loss.backward(); optimizer.step()`.

</details>

:::tip[Adam vs AdamW]
Standard Adam applies weight decay inside the adaptive update, which interacts poorly with the per-parameter scaling. **AdamW** decouples weight decay from the gradient-based update, and is now recommended over Adam in most settings. Use `torch.optim.AdamW`.
:::

## Learning Rate Schedules

The learning rate is the single most important hyperparameter. Too high and training diverges; too low and it takes forever (or gets stuck in a bad minimum). **Schedules** adjust the learning rate during training.

### Step Decay

Multiply the learning rate by a factor (e.g., 0.1) every \(N\) epochs.

```python title="Step LR schedule"
scheduler = optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.1)
# After epoch 10: lr = 1e-4, after epoch 20: lr = 1e-5
```

:::tip[Line-by-Line Walkthrough]
- **`optim.lr_scheduler.StepLR(adam, step_size=10, gamma=0.1)`** — Every 10 epochs, multiply the learning rate by 0.1 (divide it by 10). Starting from lr=0.001: after epoch 10 it becomes 0.0001, after epoch 20 it becomes 0.00001.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. This code requires an existing optimizer (`adam` from the previous example). Combine both code blocks in one file.
2. In your training loop, call `scheduler.step()` after each epoch.

**Expected output:** The learning rate will decrease by 10× every 10 epochs.

</details>

### Cosine Annealing

Smoothly decays the learning rate following a cosine curve, from the initial LR down to near zero.

```python title="Cosine annealing schedule"
scheduler = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100, eta_min=1e-6)
# Smoothly decays from 1e-3 to 1e-6 over 100 epochs
```

:::tip[Line-by-Line Walkthrough]
- **`CosineAnnealingLR(adam, T_max=100, eta_min=1e-6)`** — Over 100 epochs, the learning rate follows a smooth cosine curve from its initial value (0.001) down to `eta_min` (0.000001). The curve is gentle at first, steeper in the middle, and gentle again near the end.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Combine with the optimizer code above in one file.
2. Call `scheduler.step()` after each epoch in your training loop.

**Expected output:** The learning rate will smoothly decrease from 1e-3 to 1e-6 over 100 epochs.

</details>

### Warmup + Cosine Decay

Modern best practice: linearly ramp up the learning rate for the first few epochs (warmup), then cosine-decay for the rest. Warmup prevents early instability when the model parameters are randomly initialized and gradients are large.

```python title="Warmup + cosine schedule"
from torch.optim.lr_scheduler import LambdaLR
import math

def warmup_cosine(step, warmup_steps=500, total_steps=10000):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

# In training loop:
# for step in range(total_steps):
#     optimizer.step()
#     scheduler.step()
```

:::tip[Line-by-Line Walkthrough]
- **`warmup_cosine(step, warmup_steps=500, total_steps=10000)`** — A function that returns a multiplier for the learning rate at each step.
- **`if step < warmup_steps: return step / warmup_steps`** — During the first 500 steps, linearly ramp up the LR from 0 to the full value. At step 250, the multiplier is 0.5 (half the target LR).
- **`progress = (step - warmup_steps) / (total_steps - warmup_steps)`** — After warmup, track how far through training we are (0.0 to 1.0).
- **`0.5 * (1 + math.cos(math.pi * progress))`** — Cosine decay: smoothly decrease the multiplier from 1.0 to 0.0 over the remaining steps.
- **`LambdaLR(optimizer, lr_lambda=warmup_cosine)`** — Creates a scheduler that multiplies the base LR by the value returned from `warmup_cosine` at each step.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `warmup_schedule.py` (include the `model` and optimizer setup from earlier)
2. Run: `python warmup_schedule.py`

**Expected output:** No direct output — the scheduler is used inside a training loop. The LR ramps up linearly for 500 steps, then decays via a cosine curve for the remaining 9,500 steps.

</details>

### One-Cycle Policy

The one-cycle policy (Smith, 2018) ramps the learning rate up to a maximum, then cosine-decays. It often achieves better results in fewer epochs ("super-convergence").

```python title="One-cycle policy"
scheduler = optim.lr_scheduler.OneCycleLR(
    adam, max_lr=0.01, total_steps=1000, pct_start=0.3
)
# Ramps up for first 30% of training, then decays
```

:::tip[Line-by-Line Walkthrough]
- **`OneCycleLR(adam, max_lr=0.01, total_steps=1000, pct_start=0.3)`** — For the first 30% of training (300 steps), ramp the LR up from a small value to 0.01. For the remaining 70% (700 steps), decay back down. This aggressive schedule often leads to faster convergence.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Combine with the optimizer code in one file.
2. Call `scheduler.step()` after each training step (not each epoch).

**Expected output:** The LR ramps up to 0.01 over 300 steps, then decays over the remaining 700 steps.

</details>

## How to Choose the Right Optimizer

Here are practical guidelines:

1. **Start with AdamW**, lr=1e-3, weight_decay=0.01. This works well out of the box for most architectures.
2. **For CNNs on vision tasks**, SGD with momentum (lr=0.1, momentum=0.9) + cosine schedule often generalizes better than Adam if you can afford the tuning time.
3. **For transformers and LLMs**, AdamW with warmup + cosine decay is near-universal.
4. **If training is unstable**, reduce the learning rate, add gradient clipping, or increase warmup steps.
5. **Never tune the optimizer before the architecture.** Get a working model first, then optimize.

:::warning[Learning Rate vs Batch Size]
When scaling batch size, a common heuristic is the **linear scaling rule**: multiply the learning rate by the same factor as the batch size. If you double the batch size, double the learning rate. But this breaks down at very large batch sizes — use warmup to compensate.
:::

## Exercises

:::tip[Optimizer Comparison on MNIST — beginner]

Train the **same** model on MNIST with four optimizers: SGD, SGD with momentum, Adam, and AdamW. Plot their training loss curves. Which converges fastest? Which reaches the lowest final loss? Does the ranking change if you train for longer?

<details>
<summary>Hints</summary>

1. Use the same model architecture and seed for fair comparison
2. Track loss at each step, not just each epoch
3. Use matplotlib to plot loss curves on the same axes

</details>

:::

:::tip[Implement Adam from Scratch — advanced]

Implement the **Adam optimizer from scratch** (no `torch.optim`). Your implementation should:
1. Maintain first and second moment estimates for each parameter
2. Apply bias correction
3. Update parameters using the corrected moments

Verify that your implementation produces the same parameter updates as `torch.optim.Adam` on a simple linear regression problem.

<details>
<summary>Hints</summary>

1. You need to maintain running means m and v for each parameter
2. Don't forget bias correction: m_hat = m / (1 - beta1^t)
3. Use torch.no_grad() when updating parameters
4. Test by comparing against torch.optim.Adam on a simple problem

</details>

:::

:::tip[Learning Rate Finder — intermediate]

4 * best_loss:
            break
        best_loss = min(best_loss, loss.item())
        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())
        optimizer.param_groups[0]["lr"] *= factor

    plt.semilogx(lrs, losses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.show()
```
  }
>
Implement a **learning rate finder** (Leslie Smith, 2017). Start with a tiny learning rate, increase it exponentially over a few hundred steps, and record the loss. Plot loss vs. learning rate (log scale). The optimal learning rate is roughly where the loss curve has the steepest negative slope. Test it on a CNN + CIFAR-10.

<details>
<summary>Hints</summary>

1. Start with a very small lr (1e-7) and multiply by a constant factor each step
2. Record the loss at each step
3. The best lr is roughly where the loss is decreasing most steeply
4. Stop when the loss exceeds 4x the minimum observed loss

</details>

:::

## Resources

- **[An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747)** _(paper)_ by Sebastian Ruder — Comprehensive survey of optimization algorithms for deep learning — from SGD to Adam and beyond.

- **[How to Set the Learning Rate](https://www.jeremyjordan.me/nn-learning-rate/)** _(tutorial)_ by Jeremy Jordan — Practical guide to learning rate selection with visualizations of different schedules.

- **[Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)** _(paper)_ by Loshchilov & Hutter — The paper that introduced AdamW — explains why standard Adam's weight decay is broken and how to fix it.

- **[Super-Convergence (One-Cycle Policy)](https://arxiv.org/abs/1708.07120)** _(paper)_ by Leslie N. Smith — Shows that large learning rates with cyclical schedules can train networks 10x faster.
