---
sidebar_position: 4
slug: regularization
title: "Regularization Techniques"
---


# Regularization Techniques

A model that memorizes the training data but fails on unseen data is useless. **Regularization** is the collection of techniques that fight overfitting — the gap between training performance and test performance. This chapter covers the most important ones, from classical penalty terms to modern architectural tricks.

## Overfitting Revisited

Overfitting happens when a model has enough capacity to fit not just the signal in the training data, but also its noise. The classic symptoms:

- Training loss keeps decreasing.
- Validation loss decreases initially, then starts increasing.
- The gap between training and validation accuracy widens over time.

:::info[The Bias-Variance Tradeoff]
**Bias** is the error from underfitting — the model is too simple to capture the pattern. **Variance** is the error from overfitting — the model is so flexible it fits random noise. Regularization reduces variance at the cost of slightly increased bias, but the net effect on test error is positive.
:::

The fundamental approach to fighting overfitting: either **constrain the model** (make it simpler), **add noise** (make training harder), or **get more data** (reduce the noise-to-signal ratio).

## L1 and L2 Regularization

### L2 Regularization (Weight Decay)

Add the squared magnitude of all weights to the loss:

:::note[L2-Regularized Loss]

:::info[Plain English: What Does This Formula Mean?]
Imagine you're packing a suitcase and you have a weight limit. L2 regularization adds a "penalty fee" for every heavy item you pack — the heavier the item, the bigger the fee (and it grows quadratically, so really heavy items are *much* more expensive). This encourages you to spread your load across many lighter items instead of relying on a few heavy ones. In neural network terms, it discourages any single weight from becoming too large.
:::

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \sum_{i} w_i^2
$$

**Reading the formula:** *\(\mathcal{L}_{\text{reg}}\)* is the total loss the model actually minimizes. *\(\mathcal{L}_{\text{data}}\)* is the original loss (e.g., cross-entropy) measuring prediction quality. *\(\lambda\)* (lambda) is the regularization strength — a knob you set (e.g., 0.0001). Higher = more penalty. *\(w_i\)* is each individual weight in the model. *\(w_i^2\)* is that weight squared. *\(\sum_i w_i^2\)* adds up the squares of *all* weights. The *\(\frac{1}{2}\)* is just a convenience factor that makes the gradient cleaner (it becomes simply *\(\lambda w\)*).

The gradient becomes \(\nabla_w \mathcal{L}_{\text{data}} + \lambda w\), which shrinks weights toward zero at every step. This is why L2 regularization is also called **weight decay**.
:::

L2 penalizes large weights, encouraging the model to distribute information across many features rather than relying heavily on a few.

```python title="L2 regularization in PyTorch"
import torch.optim as optim

# Method 1: weight_decay parameter in the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Method 2: manual L2 penalty (useful for selective regularization)
l2_lambda = 1e-4
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = criterion(model(x), y) + l2_lambda * l2_reg
```

:::tip[Line-by-Line Walkthrough]
- **`optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`** — The easiest way to add L2 regularization: set `weight_decay` in the optimizer. Every update step, each weight gets shrunk by a tiny fraction toward zero.
- **`l2_lambda = 1e-4`** — The regularization strength (how much to penalize large weights).
- **`sum(p.pow(2).sum() for p in model.parameters())`** — Manually compute the L2 penalty: square every parameter, sum them all up. This loops through every tensor of parameters in the model.
- **`criterion(model(x), y) + l2_lambda * l2_reg`** — Add the L2 penalty to the normal loss. The optimizer will then minimize both prediction error AND weight magnitude simultaneously.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. This code assumes you have a `model`, `criterion`, `x`, and `y` defined. Combine with a model definition (e.g., `model = nn.Linear(784, 10)`) in a single file.
2. Run: `python l2_regularization.py`

**Expected output:** No printed output — this code sets up the optimizer with weight decay. You'd use it in a training loop.

</details>

### L1 Regularization (Lasso)

Add the absolute value of weights to the loss:

:::info[Plain English: What Does This Formula Mean?]
L1 regularization is like a strict budget: every weight you use costs a fixed fee, regardless of how big it is. This encourages the model to use as few weights as possible (set the rest to exactly zero). Think of it like Marie Kondo for neural networks — if a weight doesn't "spark joy" (help predictions), it gets thrown out entirely.
:::

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \sum_{i} |w_i|
$$

**Reading the formula:** *\(\mathcal{L}_{\text{data}}\)* is the normal prediction loss. *\(\lambda\)* controls how strongly we penalize. *\(|w_i|\)* is the absolute value of each weight (ignoring whether it's positive or negative). *\(\sum_i |w_i|\)* adds up the absolute values of all weights. Unlike L2 (which uses \(w^2\)), L1 applies a *constant* penalty per unit of weight — this is why it can push weights all the way to exactly zero.

L1 drives some weights exactly to zero, producing **sparse** models. This is useful for feature selection — non-zero weights indicate which features the model actually uses.

```python title="L1 regularization in PyTorch"
l1_lambda = 1e-5
l1_reg = sum(p.abs().sum() for p in model.parameters())
loss = criterion(model(x), y) + l1_lambda * l1_reg
```

:::tip[Line-by-Line Walkthrough]
- **`l1_lambda = 1e-5`** — The L1 regularization strength. Smaller than L2 because L1 penalties are absolute (not squared).
- **`sum(p.abs().sum() for p in model.parameters())`** — Take the absolute value of every parameter, then sum them all up.
- **`criterion(model(x), y) + l1_lambda * l1_reg`** — Add the L1 penalty to the normal loss. During training, the optimizer will try to minimize both prediction error and the total size of all weights.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Combine with a model definition, criterion, and sample data in a single file.
2. Run: `python l1_regularization.py`

**Expected output:** No printed output — this computes the L1-regularized loss. Use it in a training loop.

</details>

:::tip[L1 vs L2 in Practice]
L2 is far more common in deep learning. L1 is useful when you want interpretable sparsity. You can combine both (**Elastic Net**) to get the benefits of each.
:::

## Dropout

Dropout (Srivastava et al., 2014) randomly sets a fraction of neuron activations to zero during training. This prevents neurons from co-adapting — each neuron must be independently useful because it cannot rely on any specific partner being present.

:::info[How Dropout Works]
At training time, each activation is independently zeroed with probability \(p\) (the dropout rate). The remaining activations are scaled by \(\frac{1}{1-p}\) to maintain the expected value ("inverted dropout"). At test time, no dropout is applied — all neurons are active.
:::

```python title="Dropout in a network"
import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),    # 50% of hidden activations zeroed
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLPWithDropout(784, 512, 10)

# CRITICAL: switch between train and eval modes
model.train()   # dropout active
model.eval()    # dropout disabled
```

:::tip[Line-by-Line Walkthrough]
- **`class MLPWithDropout(nn.Module)`** — A multi-layer perceptron (fully connected network) with dropout.
- **`nn.Linear(input_dim, hidden_dim)`** — First fully connected layer: 784 inputs → 512 hidden neurons.
- **`nn.ReLU()`** — Activation function that zeroes out negative values.
- **`nn.Dropout(drop_rate)`** — During training, randomly set 50% of the activations to zero. During evaluation, this does nothing. The remaining activations are scaled up by 2× to compensate.
- **`model.train()`** — Enable training mode: dropout is active, batch norm uses batch statistics. **Critical** to call before training.
- **`model.eval()`** — Enable evaluation mode: dropout is disabled, batch norm uses running statistics. **Critical** to call before validation/testing.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `dropout_example.py`
2. Run: `python dropout_example.py`

**Expected output:** No errors — the model is created and mode-switched. To verify dropout is working, pass the same input twice in `train()` mode (different outputs due to random dropout) vs. `eval()` mode (same outputs).

</details>

:::warning[Forgetting model.eval()]
The single most common regularization bug is forgetting to call `model.eval()` during validation/testing. With dropout still active, your evaluation metrics will be noisy and pessimistic. Always set the mode explicitly.
:::

### Dropout rates by layer type

- **Fully connected layers**: 0.5 is the classic rate; 0.1–0.3 is common in modern architectures.
- **After embeddings** (in NLP): 0.1–0.3.
- **Convolutional layers**: use `Dropout2d` which drops entire channels, or use lower rates (0.05–0.2).
- **Recurrent layers**: apply dropout between layers (not within the recurrence). PyTorch's `nn.LSTM(dropout=0.3)` handles this.

## Batch Normalization

Batch normalization (Ioffe & Szegedy, 2015) normalizes the inputs to each layer to have zero mean and unit variance across the mini-batch, then applies a learnable affine transformation.

:::note[Batch Norm Computation]

:::info[Plain English: What Does This Formula Mean?]
Imagine a classroom where some students shout (large activations) and some whisper (small activations). Batch normalization is like a volume equalizer: it adjusts everyone's volume to a comfortable range, then lets each student choose to be a little louder or quieter through learnable adjustments. This makes it much easier for the next layer to learn, because its inputs are always in a predictable range.
:::

For a mini-batch \(B = {x_1, \ldots, x_m}\):

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

**Reading the formula:** *\(x_i\)* is the activation for one example in the batch. *\(\mu_B\)* is the mean (average) of all activations in the batch. *\(\sigma_B^2\)* is the variance (how spread out the values are). *\(\hat{x}_i\)* is the normalized activation — subtract the mean (center at zero) and divide by the standard deviation (scale to unit spread). *\(\epsilon\)* (epsilon, a tiny number like 10⁻⁵) prevents dividing by zero. *\(\gamma\)* and *\(\beta\)* are learnable parameters — *\(\gamma\)* scales and *\(\beta\)* shifts the normalized value, so the network can learn the optimal range for each feature.
:::

Benefits of batch normalization:

- **Faster training** — allows higher learning rates.
- **Regularization effect** — the batch statistics add noise, acting like a mild regularizer.
- **Reduced sensitivity to initialization**.

```python title="Batch normalization in a CNN"
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

# For 1D data (e.g., after a linear layer):
bn1d = nn.BatchNorm1d(256)

# For sequences / NLP, LayerNorm is preferred:
ln = nn.LayerNorm(256)
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)`** — A conv layer with no bias (`bias=False`), because the following BatchNorm already has its own bias parameter (*\(\beta\)*), making the conv's bias redundant.
- **`nn.BatchNorm2d(out_ch)`** — Normalize each of the `out_ch` feature maps across the batch. Learns a scale (*\(\gamma\)*) and shift (*\(\beta\)*) per channel.
- **`nn.ReLU()`** — Activation after normalization (the standard order: Conv → BatchNorm → ReLU).
- **`nn.BatchNorm1d(256)`** — BatchNorm for 1D data (e.g., output of a linear layer with 256 features).
- **`nn.LayerNorm(256)`** — LayerNorm normalizes across features *within a single sample* (not across the batch). Preferred for NLP and transformers because it's independent of batch size.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `batchnorm_example.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python batchnorm_example.py`

**Expected output:** No errors — the modules are created. Test with: `block = ConvBlock(3, 16); print(block(torch.randn(4, 3, 32, 32)).shape)` → `torch.Size([4, 16, 32, 32])`.

</details>

:::info[BatchNorm vs LayerNorm]
**BatchNorm** normalizes across the batch dimension — works well for CNNs but breaks with small batch sizes or variable-length sequences. **LayerNorm** normalizes across the feature dimension within each sample — batch-size independent, and standard in transformers and NLP.
:::

## Data Augmentation

Instead of constraining the model, increase the effective dataset size by applying random transformations to the training data. The model never sees the exact same image twice.

```python title="Image augmentation pipeline"
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomGrayscale(p=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25),
])

# At test time — no augmentation, only resize and normalize
test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

:::tip[Line-by-Line Walkthrough]
- **`T.Compose([...])`** — Chain multiple transforms into a pipeline applied to each image.
- **`T.RandomResizedCrop(224, scale=(0.8, 1.0))`** — Randomly crop 80–100% of the image and resize to 224×224. Simulates different zoom levels.
- **`T.RandomHorizontalFlip(p=0.5)`** — 50% chance of mirroring the image left-right.
- **`T.RandomRotation(15)`** — Randomly rotate by up to ±15 degrees.
- **`T.ColorJitter(...)`** — Randomly change brightness, contrast, saturation, and hue. Makes the model robust to lighting conditions.
- **`T.RandomGrayscale(p=0.1)`** — 10% chance of converting to grayscale. Forces the model to not rely solely on color.
- **`T.ToTensor()`** — Convert the PIL image to a PyTorch tensor (values from 0–255 become 0.0–1.0).
- **`T.Normalize(...)`** — Subtract the ImageNet mean and divide by the ImageNet std per channel.
- **`T.RandomErasing(p=0.25)`** — 25% chance of erasing a random rectangle in the image (like cutout). Forces the model to not depend on any single region.
- **Test transform** — No randomness! Just resize, center-crop, and normalize. Evaluation must be deterministic.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision pillow
```

**Steps:**
1. Save to `augmentation.py`
2. To test, add: `from PIL import Image; img = Image.new('RGB', (256, 256)); print(train_transform(img).shape)`
3. Run: `python augmentation.py`

**Expected output:**
```
torch.Size([3, 224, 224])
```

</details>

### Common augmentation strategies

| Domain | Techniques |
|--------|-----------|
| Images | Flip, crop, rotate, color jitter, cutout, mixup, CutMix |
| Text | Synonym replacement, random insertion/deletion, back-translation |
| Audio | Time stretching, pitch shifting, noise injection, SpecAugment |

:::tip[Mixup and CutMix]
**Mixup** blends two images and their labels: \(\tilde{x} = \lambda x_i + (1-\lambda) x_j\), \(\tilde{y} = \lambda y_i + (1-\lambda) y_j\). **CutMix** replaces a rectangular region of one image with a patch from another. Both improve generalization significantly and are easy to implement.
:::

```python title="Simple Mixup implementation"
import numpy as np

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.beta(alpha, alpha)`** — Draw a random blending ratio from a Beta distribution. With alpha=0.2, *\(\lambda\)* is usually near 0 or 1 (mostly one image or the other), but sometimes in between.
- **`torch.randperm(batch_size, device=x.device)`** — Create a random permutation of indices — this is how we pick which pairs of images to blend.
- **`lam * x + (1 - lam) * x[index]`** — Blend each image with a randomly shuffled partner. If *\(\lambda\)*=0.7, the result is 70% of the original image + 30% of the partner.
- **`mixup_criterion(criterion, pred, y_a, y_b, lam)`** — The loss is also blended: 70% of the loss for label A + 30% of the loss for label B. This is how the model learns from the mixed input.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy
```

**Steps:**
1. Save to `mixup.py` (add `import torch` at the top)
2. Test with: `x = torch.randn(8, 3, 32, 32); y = torch.randint(0, 10, (8,)); mixed_x, y_a, y_b, lam = mixup_data(x, y); print(mixed_x.shape, lam)`
3. Run: `python mixup.py`

**Expected output:**
```
torch.Size([8, 3, 32, 32]) 0.xxx
```
(where 0.xxx is a random blending ratio)

</details>

## Early Stopping

Monitor validation loss during training and stop when it stops improving. This is arguably the simplest and most effective regularization technique.

```python title="Early stopping implementation"
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# Usage in training loop
early_stop = EarlyStopping(patience=10)
for epoch in range(200):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    early_stop(val_loss, model)
    if early_stop.should_stop:
        print(f"Early stopping at epoch {epoch}")
        model.load_state_dict(torch.load("best_model.pt"))
        break
```

:::tip[Line-by-Line Walkthrough]
- **`patience=7`** — How many epochs to wait without improvement before stopping. Higher patience = more tolerance.
- **`min_delta=0.001`** — The minimum improvement in validation loss that counts as "getting better." Tiny improvements below this threshold are ignored.
- **`self.best_loss = float("inf")`** — Start with "infinite" loss so any real loss is an improvement.
- **`if val_loss < self.best_loss - self.min_delta:`** — If the new validation loss is meaningfully lower than the best we've seen, reset the patience counter and save the model.
- **`torch.save(model.state_dict(), "best_model.pt")`** — Save the model's weights whenever we see a new best validation loss.
- **`self.counter += 1`** — If we didn't improve, increment the "no improvement" counter.
- **`if self.counter >= self.patience: self.should_stop = True`** — If we've gone `patience` epochs without improvement, signal that training should stop.
- **`model.load_state_dict(torch.load("best_model.pt"))`** — After stopping, reload the best model (not the final, potentially overfitting one).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. This code is a reusable class. Include it in your training script alongside your model, data loaders, and training functions.
2. The `train_one_epoch` and `evaluate` functions need to be defined (see examples in other chapters).

**Expected output:**
```
Early stopping at epoch 47
```
(The exact epoch depends on your model and data.)

</details>

## Putting It All Together

In practice, you combine multiple regularization techniques. Here's a typical setup for a CNN:

```python title="A well-regularized CNN"
class RegularizedCNN(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),          # batch norm
            nn.ReLU(),
            nn.Dropout2d(drop_rate),     # spatial dropout
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop_rate),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),             # standard dropout before FC
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Optimizer with weight decay (L2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Data augmentation in the data pipeline
# + Early stopping monitoring val loss
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Conv2d(3, 64, 3, padding=1, bias=False)`** — Conv layer with no bias (BatchNorm provides its own).
- **`nn.BatchNorm2d(64)`** — Normalize feature maps for stable training.
- **`nn.Dropout2d(drop_rate)`** — Spatial dropout: drops entire feature map channels (not individual pixels). Better for convolutional layers because nearby pixels are correlated.
- **`nn.Dropout(0.5)`** — Standard dropout (50%) before the final fully connected layer — the most impactful position for dropout.
- **`nn.Flatten()`** — Reshape the 3D feature maps into a 1D vector.
- **`optim.AdamW(..., weight_decay=0.01)`** — AdamW with weight decay = L2 regularization.
- The comment notes that this model uses **four** regularization techniques simultaneously: batch normalization, spatial dropout, standard dropout, and weight decay (plus data augmentation and early stopping from the training pipeline).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `regularized_cnn.py` (add `import torch; import torch.nn as nn; import torch.optim as optim` at the top)
2. Add: `model = RegularizedCNN(); x = torch.randn(4, 3, 32, 32); print(model(x).shape)` to test.
3. Run: `python regularized_cnn.py`

**Expected output:**
```
torch.Size([4, 10])
```

</details>

## Exercises

:::tip[Quantify the Effect of Dropout — beginner]

Train an MLP on MNIST (or Fashion-MNIST) with different dropout rates: 0, 0.1, 0.3, 0.5, and 0.7. For each, track training accuracy and validation accuracy over 30 epochs. Plot the results. At what dropout rate is the gap between train and val accuracy smallest? Does the best validation accuracy always correspond to the smallest gap?

<details>
<summary>Hints</summary>

1. Train the same architecture with dropout rates 0, 0.1, 0.3, 0.5, and 0.7
2. Plot both training and validation accuracy for each
3. Too much dropout will hurt training accuracy — find the sweet spot

</details>

:::

:::tip[Implement Label Smoothing — intermediate]

**Label smoothing** is a regularization technique that softens the target distribution — instead of training toward hard 0/1 targets, you target \(1 - \epsilon\) for the correct class and \(\epsilon / C\) for the others. Implement label smoothing manually using log-softmax, and verify it matches PyTorch's `CrossEntropyLoss(label_smoothing=0.1)`.

<details>
<summary>Hints</summary>

1. Instead of one-hot targets [0, 0, 1, 0], use [ε/C, ε/C, 1-ε+ε/C, ε/C]
2. ε = 0.1 is a common smoothing factor, C is the number of classes
3. PyTorch's CrossEntropyLoss has a built-in label_smoothing parameter — compare against your manual version

</details>

:::

:::tip[Overfitting Diagnostic Dashboard — advanced]

Build a **training diagnostic dashboard** that plots, in a 2×2 grid: (1) training vs. validation loss, (2) training vs. validation accuracy, (3) the generalization gap over time, and (4) gradient norms per epoch. Train a model with and without regularization and compare the dashboards. What visual signatures indicate overfitting?

<details>
<summary>Hints</summary>

1. Record training loss, validation loss, train accuracy, val accuracy, and gradient norms
2. Use matplotlib subplots to create a multi-panel figure
3. The generalization gap (train_acc - val_acc) over time is a key metric
4. Weight histogram changes across training can reveal if L2 is working

</details>

:::

## Resources

- **[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)** _(paper)_ by Srivastava et al. — The original dropout paper — shows consistent improvements across diverse tasks.

- **[Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)** _(paper)_ by Ioffe & Szegedy — Introduced batch normalization and its dramatic effect on training speed and stability.

- **[Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187)** _(paper)_ by He et al. — Practical guide to all the training tricks (augmentation, LR schedules, label smoothing, mixup) that add up to significant accuracy gains.

- **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)** _(tutorial)_ by Andrej Karpathy — Practical wisdom on the debugging and regularization process — required reading for any practitioner.
