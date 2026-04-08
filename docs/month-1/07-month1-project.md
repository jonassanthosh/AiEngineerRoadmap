---
sidebar_position: 8
title: "Month 1 Project: MNIST Digit Classifier"
slug: month1-project
---


# Month 1 Project: MNIST Digit Classifier

:::info[What You'll Learn]
- Loading and preprocessing the MNIST dataset
- Designing a neural network architecture
- Implementing a full training and evaluation pipeline
- Experimenting with hyperparameters to improve accuracy
:::

**Estimated time:** Reading: ~30 min | Project work: ~8 hours

Congratulations on making it through the foundations! It's time to put everything together in a real project. You'll build a neural network that recognizes handwritten digits from the **MNIST dataset**—the "Hello, World!" of deep learning.

By the end of this project, you'll have a model that achieves **>97% accuracy** on 10,000 test images it has never seen. More importantly, you'll understand every piece of the pipeline: data loading, preprocessing, model architecture, training, evaluation, and experimentation.

## The MNIST Dataset

MNIST contains 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels:

- **60,000** training images
- **10,000** test images

Each pixel is a value from 0 (black) to 255 (white). The task is multi-class classification: given a 28×28 image, predict which digit (0–9) it represents.

:::info[Why MNIST?]
MNIST is the standard first project for a reason: it's small enough to train on a laptop CPU in minutes, complex enough to require a real neural network, and well-studied enough that you can easily compare your results to known benchmarks. State-of-the-art is ~99.8% accuracy (with convolutional networks and data augmentation), so there's room to experiment.
:::

## Step 1: Setup and Data Loading

```python title="Loading MNIST with PyTorch"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transforms: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),           # converts PIL image to tensor, scales to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Inspect the data
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

sample_image, sample_label = train_dataset[0]
print(f"Image shape: {sample_image.shape}")  # (1, 28, 28)
print(f"Label: {sample_label}")
print(f"Pixel range: [{sample_image.min():.4f}, {sample_image.max():.4f}]")

# Class distribution
from collections import Counter
label_counts = Counter(int(train_dataset.targets[i]) for i in range(len(train_dataset)))
for digit in range(10):
    print(f"  Digit {digit}: {label_counts[digit]} samples")
```

:::tip[Line-by-Line Walkthrough]
- **`transforms.Compose([...])`** — Chains multiple image transformations together. Each image passes through them in order.
- **`transforms.ToTensor()`** — Converts a PIL image (pixels 0–255) into a PyTorch tensor (pixels 0.0–1.0). Also rearranges dimensions from (height, width, channels) to (channels, height, width).
- **`transforms.Normalize((0.1307,), (0.3081,))`** — Subtracts the mean (0.1307) and divides by the standard deviation (0.3081) of the MNIST dataset. Centers pixel values around zero for better training.
- **`datasets.MNIST(root='./data', train=True, download=True, transform=transform)`** — Downloads the MNIST training set (60,000 images) into a `./data` folder and applies our transforms automatically.
- **`DataLoader(train_dataset, batch_size=64, shuffle=True)`** — Creates an iterator that serves 64 images at a time in random order. Shuffling prevents the model from memorizing the order of the data.
- **`sample_image.shape`** — Should be `(1, 28, 28)`: 1 color channel (grayscale), 28 pixels high, 28 pixels wide.
- **`Counter(...)`** — Counts how many images belong to each digit class. A balanced dataset has roughly equal numbers for each digit.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save to `load_mnist.py` and run: `python load_mnist.py`
2. On first run, it will download the MNIST dataset (~11 MB) into a `./data` folder.

**Expected output:**
```
Training samples: 60000
Test samples: 10000
Image shape: torch.Size([1, 28, 28])
Label: 5
Pixel range: [-0.4242, 2.8215]
  Digit 0: 5923 samples
  Digit 1: 6742 samples
  ...
  Digit 9: 5949 samples
```

</details>

:::tip[Normalization Matters]
We normalize using MNIST's global mean (0.1307) and standard deviation (0.3081). This centers the data around zero and scales it to roughly unit variance, which helps gradient descent converge faster. These specific values are pre-computed over the entire MNIST training set.
:::

## Step 2: Visualizing the Data

Before building a model, always look at your data. This helps you understand what the model needs to learn and catch potential issues.

```python title="Visualizing MNIST Samples"
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Show some samples as ASCII art (works in terminal)
def ascii_digit(image_tensor):
    """Render a 28x28 tensor as ASCII art."""
    # Unnormalize
    img = image_tensor.squeeze() * 0.3081 + 0.1307
    img = img.numpy()
    chars = ' .:-=+*#%@'
    rows = []
    for y in range(0, 28, 2):  # skip every other row for aspect ratio
        row = ''
        for x in range(28):
            intensity = img[y, x]
            idx = min(int(intensity * len(chars)), len(chars) - 1)
            row += chars[idx]
        rows.append(row)
    return '\\n'.join(rows)

# Display 5 random samples
import random
random.seed(42)
indices = random.sample(range(len(train_dataset)), 5)

for idx in indices:
    image, label = train_dataset[idx]
    print(f"\\n--- Digit: {label} ---")
    print(ascii_digit(image))
```

:::tip[Line-by-Line Walkthrough]
- **`def ascii_digit(image_tensor):`** — Converts a 28×28 image tensor into text art you can view in a terminal (no graphics library needed).
- **`img = image_tensor.squeeze() * 0.3081 + 0.1307`** — Reverses the normalization to get pixel values back to the original 0–1 range. `.squeeze()` removes the channel dimension.
- **`chars = ' .:-=+*#%@'`** — A gradient of characters from light (space) to dark (@). Each pixel's brightness picks a character.
- **`for y in range(0, 28, 2):`** — Skips every other row because terminal characters are taller than they are wide, which would stretch the image vertically.
- **`idx = min(int(intensity * len(chars)), len(chars) - 1)`** — Maps each pixel's brightness to one of the ASCII characters.
- **`random.sample(range(len(train_dataset)), 5)`** — Picks 5 random images from the training set to display.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save to `visualize_mnist.py` and run: `python visualize_mnist.py`

**Expected output:**
```
--- Digit: 7 ---
      .:=+*
       .=*#
      .:*#
     .:*#
    .:*#
...
```
(You'll see 5 handwritten digits rendered as ASCII art in your terminal.)

</details>

## Step 3: Building the Model

We'll build a simple but effective multi-layer perceptron. Later in the bonus challenges, you'll explore ways to improve it.

```python title="MNIST Classifier Architecture"
import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 28x28 → 784
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),  # 10 output classes, no softmax (CrossEntropyLoss handles it)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = MNISTClassifier()

# Inspect the model
print(model)
print(f"\\nParameter count per layer:")
total = 0
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} ({param.numel():,} params)")
    total += param.numel()
print(f"\\nTotal parameters: {total:,}")
```

:::tip[Line-by-Line Walkthrough]
- **`class MNISTClassifier(nn.Module):`** — Defines our digit classifier as a PyTorch model.
- **`nn.Flatten()`** — Converts each 28×28 image into a flat vector of 784 numbers. Like unrolling a grid into a single line.
- **`nn.Linear(784, 256)`** — First hidden layer: takes the 784 pixel values and transforms them into 256 features. Each of the 256 neurons learns to detect a different pattern.
- **`nn.ReLU()`** — Activation function that keeps positive values and zeroes out negatives. Adds non-linearity so the network can learn curved decision boundaries.
- **`nn.Dropout(0.2)`** — During training, randomly turns off 20% of neurons each pass. This prevents the network from relying too much on any single neuron (regularization).
- **`nn.Linear(128, 10)`** — Output layer: 10 neurons, one for each digit (0–9). The neuron with the highest value is the predicted digit.
- **`def forward(self, x):`** — Defines the data flow: flatten the image, then pass through the network.
- **`model.named_parameters()`** — Iterates through every learnable parameter (weight matrices and bias vectors) to count the total number of parameters.
:::

:::info[Architecture Decisions]
**Why flatten?** MNIST images are 28×28 grids, but our fully-connected layers expect 1D vectors. `nn.Flatten()` reshapes (1, 28, 28) → (784,). Convolutional networks (Month 2) can process the 2D structure directly.

**Why no softmax in the output?** `nn.CrossEntropyLoss` in PyTorch expects raw logits, not probabilities. It applies log-softmax internally for numerical stability. This is a common source of bugs—don't add a softmax layer before CrossEntropyLoss.

**Why dropout?** Dropout randomly zeroes 20% of neuron outputs during training, which acts as regularization. It forces the network to not rely on any single neuron and helps prevent overfitting.
:::

## Step 4: Training

```python title="Training Loop"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.network(self.flatten(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        output = model(batch_X)
        loss = criterion(output, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
        predicted = output.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            predicted = output.argmax(dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / total, correct / total

# Training
num_epochs = 10
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
print("-" * 60)

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_loss:9.4f} | {test_acc:7.2%}")

print(f"\\nFinal test accuracy: {test_acc:.2%}")
```

:::tip[Line-by-Line Walkthrough]
- **`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`** — Automatically uses GPU if available, otherwise falls back to CPU.
- **`model = MNISTClassifier().to(device)`** — Creates the model and moves it to the selected device.
- **`nn.CrossEntropyLoss()`** — The loss function for multi-class classification. Combines log-softmax and negative log-likelihood. Expects raw logits (no softmax in the model).
- **`optim.Adam(model.parameters(), lr=0.001)`** — Adam optimizer with a learning rate of 0.001.
- **`def train_epoch(...):`** — Runs one full pass through the training data, processing it in mini-batches.
- **`batch_X, batch_y = batch_X.to(device), batch_y.to(device)`** — Moves each batch to the same device as the model (CPU or GPU).
- **`predicted = output.argmax(dim=1)`** — For each sample, picks the digit with the highest score as the prediction.
- **`correct += (predicted == batch_y).sum().item()`** — Counts how many predictions match the true labels.
- **`def evaluate(...):`** — Same logic but with `model.eval()` and `torch.no_grad()` — no gradient computation or dropout.
- **The training loop** — Runs 10 epochs, printing train/test loss and accuracy after each one.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save the entire code block to `train_mnist.py` and run: `python train_mnist.py`
2. Training takes about 1–3 minutes on CPU.

**Expected output:**
```
Using device: cpu
Epoch | Train Loss | Train Acc | Test Loss | Test Acc
------------------------------------------------------------
    1 |     0.3412 |   89.87%  |    0.1654 |  95.12%
    2 |     0.1398 |   95.82%  |    0.1187 |  96.40%
    ...
   10 |     0.0287 |   99.12%  |    0.0821 |  97.65%

Final test accuracy: 97.65%
```

</details>

You should see the model reach **~97-98% test accuracy** within 10 epochs. That means it correctly classifies roughly 9,700 out of 10,000 handwritten digits it has never seen before.

## Step 5: Detailed Evaluation

Accuracy alone doesn't tell the full story. Let's look at per-class performance and examine what the model gets wrong.

```python title="Detailed Evaluation"
import torch
import torch.nn as nn
import numpy as np

def detailed_evaluation(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Per-class accuracy
    print("Per-class performance:")
    print(f"{'Digit':>5} | {'Correct':>7} | {'Total':>5} | {'Accuracy':>8}")
    print("-" * 35)
    
    for digit in range(10):
        mask = all_labels == digit
        correct = (all_preds[mask] == digit).sum()
        total = mask.sum()
        acc = correct / total
        print(f"{digit:5d} | {correct:7d} | {total:5d} | {acc:7.2%}")
    
    # Confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1
    
    print(f"\\nConfusion Matrix (rows=true, cols=predicted):")
    header = "     " + "".join(f"{i:5d}" for i in range(10))
    print(header)
    for i in range(10):
        row = f"{i:3d}: " + "".join(f"{confusion[i][j]:5d}" for j in range(10))
        print(row)
    
    # Most confused pairs
    print(f"\\nMost common misclassifications:")
    errors = []
    for true in range(10):
        for pred in range(10):
            if true != pred and confusion[true][pred] > 0:
                errors.append((confusion[true][pred], true, pred))
    
    errors.sort(reverse=True)
    for count, true, pred in errors[:10]:
        print(f"  {true} → {pred}: {count} times")
    
    # Confidence analysis
    correct_mask = all_preds == all_labels
    correct_conf = all_probs[np.arange(len(all_preds))[correct_mask], all_preds[correct_mask]]
    wrong_conf = all_probs[np.arange(len(all_preds))[~correct_mask], all_preds[~correct_mask]]
    
    print(f"\\nConfidence analysis:")
    print(f"  Correct predictions — mean confidence: {correct_conf.mean():.4f}")
    print(f"  Wrong predictions   — mean confidence: {wrong_conf.mean():.4f}")
    
    return all_preds, all_labels, all_probs

# Run evaluation (assumes model, test_loader, device are defined from Step 4)
# all_preds, all_labels, all_probs = detailed_evaluation(model, test_loader, device)
print("Run detailed_evaluation(model, test_loader, device) after training!")
```

:::tip[Line-by-Line Walkthrough]
- **`def detailed_evaluation(model, test_loader, device):`** — A comprehensive evaluation function that goes far beyond simple accuracy.
- **`probs = torch.softmax(output, dim=1)`** — Converts raw logits into probabilities (values between 0 and 1 that sum to 1 across the 10 digits).
- **`preds = output.argmax(dim=1)`** — Picks the digit with the highest score for each image.
- **`all_preds.extend(preds.cpu().numpy())`** — Collects all predictions across batches into a single list. `.cpu()` moves data from GPU to CPU if needed.
- **The per-class accuracy loop** — Computes accuracy separately for each digit. Maybe the model is great at 1s but struggles with 8s.
- **`confusion = np.zeros((10, 10), dtype=int)`** — Builds a confusion matrix: a 10×10 table where row = true digit, column = predicted digit. Diagonal entries are correct predictions; off-diagonal are errors.
- **`errors.sort(reverse=True)`** — Sorts the most common mistakes in descending order, so you can focus on the biggest problems first.
- **The confidence analysis** — Compares how confident the model is when it's right vs. when it's wrong. A well-calibrated model should be less confident on its mistakes.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision numpy
```

**Steps:**
1. This function should be called after training (from Step 4). Add it to the same file and call:
   ```python
   all_preds, all_labels, all_probs = detailed_evaluation(model, test_loader, device)
   ```

**Expected output:**
```
Per-class performance:
Digit | Correct | Total | Accuracy
-----------------------------------
    0 |     970 |   980 |  98.98%
    1 |    1125 |  1135 |  99.12%
    ...

Most common misclassifications:
  4 → 9: 12 times
  3 → 5: 8 times
  ...
```

</details>

:::tip[Common Confusions in MNIST]
The most commonly confused pairs are typically: 4↔9, 3↔5, 7↔2, and 3↔8. These make intuitive sense—a hastily written 4 can look like a 9, and a 3 with a flat top can look like a 5. Looking at specific errors helps you understand what the model finds difficult and what improvements might help.
:::

## Step 6: Examining Errors

```python title="Analyzing Misclassified Digits"
import torch
import numpy as np

def find_interesting_errors(model, test_dataset, device, n=10):
    """Find the most confidently wrong predictions."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, label = test_dataset[i]
            output = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()
            
            if pred != label:
                errors.append({
                    'index': i,
                    'true': label,
                    'predicted': pred,
                    'confidence': confidence,
                    'image': image,
                })
    
    # Sort by confidence (most confidently wrong first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Total errors: {len(errors)} / {len(test_dataset)} ({len(errors)/len(test_dataset)*100:.1f}%)")
    print(f"\\nTop {n} most confidently wrong predictions:")
    for e in errors[:n]:
        print(f"  True: {e['true']}, Predicted: {e['predicted']} "
              f"(confidence: {e['confidence']:.2%})")
    
    return errors

# Run after training:
# errors = find_interesting_errors(model, test_dataset, device)
print("Run find_interesting_errors(model, test_dataset, device) after training!")
```

:::tip[Line-by-Line Walkthrough]
- **`def find_interesting_errors(...):`** — Finds predictions where the model was wrong but highly confident — the most interesting (and dangerous) failure cases.
- **`model(image.unsqueeze(0).to(device))`** — Adds a batch dimension (the model expects batches, not single images) and runs inference.
- **`probs = torch.softmax(output, dim=1)`** — Gets the probability distribution over all 10 digits.
- **`confidence = probs[0, pred].item()`** — How confident was the model in its (wrong) prediction? A model that's 95% confident and wrong is more concerning than one that's 55% confident and wrong.
- **`errors.sort(key=lambda x: x['confidence'], reverse=True)`** — Sorts errors so the most confidently wrong predictions come first. These are the cases worth investigating.
- **The summary printout** — Shows total error count and the top 10 most confidently wrong predictions, so you know exactly where the model struggles.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision numpy
```

**Steps:**
1. Call this function after training (from Step 4):
   ```python
   errors = find_interesting_errors(model, test_dataset, device)
   ```

**Expected output:**
```
Total errors: 234 / 10000 (2.3%)

Top 10 most confidently wrong predictions:
  True: 9, Predicted: 4 (confidence: 99.82%)
  True: 2, Predicted: 7 (confidence: 98.91%)
  ...
```

</details>

## Tips for Experimentation

Once your baseline model is working, try these experiments systematically. Change **one thing at a time** and record the results.

### Hyperparameters to Tune

| Parameter | Baseline | Try |
|-----------|----------|-----|
| Learning rate | 0.001 | 0.0001, 0.003, 0.01 |
| Batch size | 64 | 32, 128, 256 |
| Hidden layer size | 256, 128 | 512/256, 128/64 |
| Dropout rate | 0.2 | 0, 0.1, 0.3, 0.5 |
| Optimizer | Adam | SGD+momentum, AdamW |
| Epochs | 10 | 20, 30 (watch for overfitting) |

### Experiment Tracking

```python title="Simple Experiment Tracking"
import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self):
        self.experiments = []
    
    def log(self, name, config, results):
        self.experiments.append({
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results,
        })
    
    def summary(self):
        print(f"{'Name':<20} | {'Test Acc':>8} | {'Params':>8} | {'Config'}")
        print("-" * 80)
        for exp in sorted(self.experiments, key=lambda x: x['results'].get('test_acc', 0), reverse=True):
            print(f"{exp['name']:<20} | {exp['results'].get('test_acc', 0):7.2%} "
                  f"| {exp['results'].get('n_params', 0):>8,} "
                  f"| lr={exp['config'].get('lr')}, bs={exp['config'].get('batch_size')}")
    
    def save(self, path='experiments.json'):
        with open(path, 'w') as f:
            json.dump(self.experiments, f, indent=2)

# Usage
tracker = ExperimentTracker()

# After each experiment:
# tracker.log(
#     name="baseline",
#     config={'lr': 0.001, 'batch_size': 64, 'hidden': [256, 128], 'dropout': 0.2},
#     results={'test_acc': 0.978, 'train_acc': 0.994, 'n_params': 235146, 'epochs': 10}
# )

# View all results:
# tracker.summary()
print("Experiment tracker ready. Use tracker.log() after each training run.")
```

:::tip[Line-by-Line Walkthrough]
- **`class ExperimentTracker:`** — A simple tool for recording and comparing ML experiments. Like a lab notebook for your model runs.
- **`def log(self, name, config, results):`** — Records one experiment: its name, the hyperparameters used (config), and the results (accuracy, etc.), along with a timestamp.
- **`datetime.now().isoformat()`** — Records exactly when the experiment ran, so you can trace your progress over time.
- **`def summary(self):`** — Prints a comparison table of all experiments, sorted by test accuracy (best first). Makes it easy to see which configuration won.
- **`def save(self, path='experiments.json'):`** — Saves all experiment records to a JSON file so you don't lose your results if you restart Python.
- **`tracker = ExperimentTracker()`** — Creates an instance. After each training run, call `tracker.log(...)` to record results, then `tracker.summary()` to compare.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ (uses only built-in modules: `json` and `datetime`).

**Steps:**
1. Save to `experiment_tracker.py` and run: `python experiment_tracker.py`

**Expected output:**
```
Experiment tracker ready. Use tracker.log() after each training run.
```
(The tracker is ready to use. Uncomment the example `tracker.log()` and `tracker.summary()` calls after running actual experiments.)

</details>

:::warning[The Experimenter's Trap]
Don't spend hours tweaking hyperparameters for tiny gains on MNIST. The goal of this project is to understand the end-to-end workflow, not to squeeze out 0.1% more accuracy. Once you hit ~97%, spend your time on the bonus challenges below—they teach more transferable skills.
:::

## Bonus Challenges

These challenges go beyond the basic project and introduce concepts you'll explore in depth during Month 2. Tackle them in order—each one builds on the last.

:::tip[Bonus 1: Learning Rate Scheduling — intermediate]

Implement a learning rate scheduler that reduces the learning rate during training:

1. Start with a higher initial learning rate (0.01)
2. Use `torch.optim.lr_scheduler.ReduceLROnPlateau` to halve the LR when validation loss plateaus
3. Compare the training curve (loss over epochs) with and without scheduling
4. Does it converge to a better final accuracy?

<details>
<summary>Hints</summary>

1. Try torch.optim.lr_scheduler.StepLR or ReduceLROnPlateau.
2. ReduceLROnPlateau lowers LR when the loss stops improving.
3. Start with lr=0.01 and let the scheduler reduce it.

</details>

:::

:::tip[Bonus 2: Data Augmentation — intermediate]

Add data augmentation to the training pipeline:

1. Apply random rotations (±15°), small translations, and slight scaling
2. Train with augmented data for 20 epochs
3. Compare test accuracy to the non-augmented baseline
4. Does augmentation help more when using less training data? (Try training on only 10% of the data with and without augmentation)

<details>
<summary>Hints</summary>

1. Use torchvision.transforms: RandomRotation, RandomAffine, RandomPerspective.
2. Only augment training data, never test data.
3. Subtle augmentation (±10° rotation, small shift) works best for digits.

</details>

:::

:::tip[Bonus 3: Batch Normalization — intermediate]

Add batch normalization to your network:

1. Insert `nn.BatchNorm1d` after each linear layer (before ReLU)
2. Train and compare convergence speed and final accuracy
3. Try using a higher learning rate (0.01) with batch norm—does it still converge?
4. Does batch norm reduce the need for dropout?

<details>
<summary>Hints</summary>

1. Add nn.BatchNorm1d(n_features) after each linear layer, before the activation.
2. Batch norm normalizes activations within each mini-batch.
3. It often allows using higher learning rates.

</details>

:::

:::tip[Bonus 4: From MLP to CNN — advanced]

Replace the MLP with a simple Convolutional Neural Network (CNN):

```python
# Suggested architecture:
# Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool2d(2)
# Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)
# Flatten → Linear(64*7*7, 128) → ReLU → Dropout → Linear(128, 10)
```

:::tip[Line-by-Line Walkthrough]
- **`Conv2d(1, 32, 3, padding=1)`** — A convolutional layer: slides a 3×3 filter across the image, producing 32 feature maps. Unlike a fully-connected layer, it preserves spatial structure (it knows that neighboring pixels are related). `padding=1` keeps the output the same size as the input.
- **`MaxPool2d(2)`** — Shrinks each feature map by half (28×28 → 14×14, then 14×14 → 7×7) by keeping only the maximum value in each 2×2 block. Reduces computation and makes the model tolerant of small shifts in the image.
- **`Linear(64*7*7, 128)`** — After two pooling layers, each of the 64 feature maps is 7×7 pixels. Flattening gives 64 × 7 × 7 = 3,136 values, which are fed into a traditional fully-connected layer.
- **`Linear(128, 10)`** — Final output: 10 neurons, one per digit class.
:::

1. Implement the CNN architecture above
2. Compare parameter count to the MLP
3. Compare test accuracy (you should see improvement)
4. How much faster/slower is training per epoch?

This is a preview of Month 2's deep learning content!

<details>
<summary>Hints</summary>

1. Use nn.Conv2d(1, 32, 3, padding=1) for the first conv layer.
2. Add nn.MaxPool2d(2) after conv+relu blocks to reduce spatial dimensions.
3. After conv layers, flatten and use a linear layer for classification.
4. Don't flatten the input at the beginning — CNNs work on 2D data.

</details>

:::

:::tip[Bonus 5: Fashion-MNIST — intermediate]

Take your best-performing model and apply it to **Fashion-MNIST** (a harder variant where the classes are clothing items instead of digits):

1. Swap `datasets.MNIST` for `datasets.FashionMNIST` (same API)
2. Train with the same hyperparameters
3. Compare accuracy to digit MNIST—which is harder?
4. Which clothing categories are most confused with each other? Why?

<details>
<summary>Hints</summary>

1. Fashion-MNIST is a drop-in replacement: same size, same number of classes.
2. Use datasets.FashionMNIST instead of datasets.MNIST.
3. The classes are: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

</details>

:::

## Project Submission Checklist

When you're done, make sure you can answer these questions:

- [ ] What test accuracy did your baseline model achieve?
- [ ] Which digits does the model confuse most often? Can you explain why?
- [ ] What was the effect of changing the learning rate?
- [ ] What was the effect of changing the network size (wider vs. narrower)?
- [ ] If you tried bonus challenges, which improvement had the biggest impact?
- [ ] How many total parameters does your model have?
- [ ] What is the gap between training and test accuracy? Is the model overfitting?

:::tip[Month 1 Complete!]
If you've made it through all 7 chapters and completed this project, you have a solid foundation in AI and machine learning. You understand the math, can write Python for data science, know how neural networks work from first principles, and have built a real classifier with PyTorch.

In **Month 2**, we'll go deeper: convolutional networks for images, recurrent networks for sequences, modern architectures, transfer learning, and more advanced training techniques. The foundation you've built this month will make everything that follows click into place.
:::

## Resources

- **[MNIST Database](http://yann.lecun.com/exdb/mnist/)** _(tutorial)_ by Yann LeCun — The official MNIST page with dataset details, benchmarks, and historical context.

- **[PyTorch MNIST Example](https://github.com/pytorch/examples/tree/main/mnist)** _(tutorial)_ by PyTorch — Official PyTorch MNIST example. Compare your implementation to the reference.

- **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** _(tool)_ by Zalando Research — A harder drop-in replacement for MNIST using clothing images.

- **[Weights & Biases: Experiment Tracking](https://wandb.ai/site)** _(tool)_ — Professional experiment tracking tool. Free for personal use. Learn it early—you'll use it throughout your career.

- **[Papers With Code: MNIST](https://paperswithcode.com/dataset/mnist)** _(tutorial)_ — Leaderboard of MNIST results with links to papers and code. See where your model ranks!
