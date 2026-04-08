---
sidebar_position: 6
title: "Build Your First Neural Network"
slug: your-first-model
---


# Build Your First Neural Network

:::info[What You'll Learn]
- Building a neural network in PyTorch from scratch
- The training loop: forward pass, loss, backward pass, optimizer step
- Tracking and visualizing training progress
- Debugging common training issues
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

Theory is essential, but understanding really clicks when you build something yourself. In this chapter, you'll implement a neural network **twice**: first from scratch using only NumPy, then using PyTorch. By comparing the two, you'll understand both *what* high-level frameworks do for you and *why* they do it that way.

We'll tackle a binary classification problem: the classic "two moons" dataset.

## The Dataset

```python title="Generating the Two Moons Dataset"
import numpy as np

def make_moons(n_samples=500, noise=0.1, seed=42):
    """Generate two interleaving half-circle datasets."""
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2
    
    # First moon
    theta1 = np.linspace(0, np.pi, n_each)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    
    # Second moon (shifted)
    theta2 = np.linspace(0, np.pi, n_each)
    x2 = 1 - np.cos(theta2)
    y2 = 1 - np.sin(theta2) - 0.5
    
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)])
    
    X += rng.normal(0, noise, X.shape)
    
    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

X, y = make_moons(500, noise=0.15)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class distribution: {np.bincount(y.astype(int))}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"First 5 samples:")
for i in range(5):
    print(f"  x={X[i].round(3)}, y={int(y[i])}")
```

:::tip[Line-by-Line Walkthrough]
- **`def make_moons(n_samples=500, noise=0.1, seed=42):`** — A function that creates the "two moons" dataset: two interleaving half-circles that a straight line can't separate.
- **`theta1 = np.linspace(0, np.pi, n_each)`** — Creates evenly spaced angles from 0 to π (half a circle) for the first moon.
- **`x1 = np.cos(theta1)` / `y1 = np.sin(theta1)`** — Converts angles to x, y coordinates. This traces out the top half-circle.
- **`x2 = 1 - np.cos(theta2)` / `y2 = 1 - np.sin(theta2) - 0.5`** — The second moon: flipped and shifted so it nestles into the first one.
- **`X += rng.normal(0, noise, X.shape)`** — Adds random noise to every point so the moons aren't perfect curves. More noise = harder problem.
- **`idx = rng.permutation(n_samples)`** — Shuffles the data so points from both moons are mixed together.
- **`np.bincount(y.astype(int))`** — Counts how many points belong to each class (should be 250 and 250 for balanced data).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `make_moons.py` and run: `python make_moons.py`

**Expected output:**
```
X shape: (500, 2)
y shape: (500,)
Class distribution: [250 250]
X range: [-0.45, 2.32]
First 5 samples:
  x=[ 0.123  0.987], y=0
  ...
```

</details>

## Part 1: Neural Network from Scratch (NumPy)

We'll build a fully-connected network with one hidden layer: 2 inputs → 32 hidden neurons (ReLU) → 1 output (sigmoid).

### Weight Initialization

```python title="Network Initialization"
import numpy as np

class NumpyNeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        """
        layer_sizes: list of ints, e.g. [2, 32, 1]
        """
        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialization (good for ReLU)
            W = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
        
        self.n_layers = len(self.weights)
    
    def __repr__(self):
        layers = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            layers.append(f"  Layer {i}: {W.shape[0]} → {W.shape[1]} ({W.size + b.size} params)")
        total = sum(W.size + b.size for W, b in zip(self.weights, self.biases))
        return f"NumpyNeuralNetwork(\\n" + "\\n".join(layers) + f"\\n  Total: {total} parameters\\n)"

net = NumpyNeuralNetwork([2, 32, 1])
print(net)
```

:::tip[Line-by-Line Walkthrough]
- **`class NumpyNeuralNetwork:`** — Defines our neural network as a Python class that holds weights, biases, and methods.
- **`layer_sizes: list of ints, e.g. [2, 32, 1]`** — The network shape: 2 inputs, 32 hidden neurons, 1 output.
- **`for i in range(len(layer_sizes) - 1):`** — Loops through each pair of adjacent layers to create the weight matrices connecting them.
- **`fan_in = layer_sizes[i]`** — Number of incoming connections to this layer.
- **`W = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)`** — He initialization: random weights scaled by $\sqrt{2/n_{in}}$. Without this scaling, signals would explode or vanish as they pass through the network.
- **`b = np.zeros((1, fan_out))`** — Biases start at zero. Each neuron gets one bias.
- **`self.weights.append(W)` / `self.biases.append(b)`** — Stores the weights and biases for later use.
- **`__repr__`** — A method that prints a nice summary of the network architecture and parameter count.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `nn_init.py` and run: `python nn_init.py`

**Expected output:**
```
NumpyNeuralNetwork(
  Layer 0: 2 → 32 (96 params)
  Layer 1: 32 → 1 (33 params)
  Total: 129 parameters
)
```

</details>

:::note[He Initialization]
Weights are initialized from $W \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$ where $n_{\text{in}}$ is the number of input connections. This scaling prevents activations from exploding or vanishing as they pass through many layers. For ReLU activations, He initialization (Kaiming initialization) is the standard choice. For sigmoid/tanh, use Xavier initialization: $\sqrt{1/n_{\text{in}}}$.
:::

### Forward Pass, Loss, and Backward Pass

```python title="Complete NumPy Neural Network"
import numpy as np

class NumpyNeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self.weights.append(rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros((1, fan_out)))
        self.n_layers = len(self.weights)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward pass. Stores intermediate values for backprop."""
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(self.n_layers):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i < self.n_layers - 1:
                current = self.relu(z)
            else:
                current = self.sigmoid(z)
            
            self.activations.append(current)
        
        return current
    
    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss."""
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, y_true):
        """Backward pass using chain rule."""
        m = y_true.shape[0]
        self.grad_weights = [None] * self.n_layers
        self.grad_biases = [None] * self.n_layers
        
        # Output layer gradient (BCE + sigmoid simplifies nicely)
        delta = self.activations[-1] - y_true  # dL/dz for output layer
        
        for i in range(self.n_layers - 1, -1, -1):
            self.grad_weights[i] = (1 / m) * self.activations[i].T @ delta
            self.grad_biases[i] = (1 / m) * np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_deriv(self.z_values[i - 1])
    
    def update(self, lr):
        """Gradient descent update."""
        for i in range(self.n_layers):
            self.weights[i] -= lr * self.grad_weights[i]
            self.biases[i] -= lr * self.grad_biases[i]
    
    def train(self, X, y, epochs=1000, lr=0.1, print_every=100):
        """Full training loop."""
        y = y.reshape(-1, 1)
        history = []
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y)
            self.update(lr)
            
            history.append(loss)
            if epoch % print_every == 0:
                acc = np.mean((y_pred > 0.5).astype(float) == y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")
        
        return history
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int).flatten()

# Create dataset
def make_moons(n_samples=500, noise=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2
    theta1 = np.linspace(0, np.pi, n_each)
    theta2 = np.linspace(0, np.pi, n_each)
    X = np.vstack([
        np.column_stack([np.cos(theta1), np.sin(theta1)]),
        np.column_stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)])
    ])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)])
    X += rng.normal(0, noise, X.shape)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

X, y = make_moons(500)

# Split data
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# Train
net = NumpyNeuralNetwork([2, 32, 1])
history = net.train(X_train, y_train, epochs=2000, lr=0.5, print_every=400)

# Evaluate
train_preds = net.predict(X_train)
test_preds = net.predict(X_test)
print(f"\\nFinal train accuracy: {np.mean(train_preds == y_train):.2%}")
print(f"Final test accuracy:  {np.mean(test_preds == y_test):.2%}")
```

:::tip[Line-by-Line Walkthrough]
- **`def relu(self, x):`** — ReLU activation: if a value is positive, keep it; if negative, replace it with 0. Like a gate that only lets positive signals through.
- **`def sigmoid(self, x):`** — Squashes any number into the range (0, 1). Used on the output layer to produce a probability.
- **`def forward(self, X):`** — The forward pass: data enters the network, gets multiplied by weights, passes through activations, and produces a prediction. Like water flowing through pipes.
- **`z = current @ self.weights[i] + self.biases[i]`** — Matrix multiplication plus bias: the core operation of a neural network layer.
- **`def compute_loss(self, y_pred, y_true):`** — Binary Cross-Entropy loss: measures how far the predictions are from the true labels. A probability of 0.9 for a true class-1 example gives low loss; 0.1 gives high loss.
- **`def backward(self, y_true):`** — Backpropagation: works backwards through the network computing how much each weight contributed to the error. Like tracing back through the pipes to find which valves to adjust.
- **`delta = self.activations[-1] - y_true`** — The initial error signal at the output layer.
- **`self.grad_weights[i] = (1 / m) * self.activations[i].T @ delta`** — Computes the gradient (direction of steepest error increase) for each weight.
- **`def update(self, lr):`** — Gradient descent: nudges each weight in the direction that reduces error, by a small step proportional to the learning rate.
- **`history = net.train(X_train, y_train, epochs=2000, lr=0.5)`** — Runs the full training loop 2,000 times, printing progress every 400 steps.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save the entire code block to `numpy_nn.py` and run: `python numpy_nn.py`

**Expected output:**
```
Epoch    0 | Loss: 0.6931 | Accuracy: 50.50%
Epoch  400 | Loss: 0.1052 | Accuracy: 97.25%
Epoch  800 | Loss: 0.0734 | Accuracy: 98.00%
Epoch 1200 | Loss: 0.0603 | Accuracy: 98.25%
Epoch 1600 | Loss: 0.0520 | Accuracy: 98.50%

Final train accuracy: 98.50%
Final test accuracy:  97.00%
```
(Exact values will vary slightly but accuracy should be in the high 90s.)

</details>

:::info[What You Just Built]
That's a complete neural network: forward pass, loss computation, backpropagation, and gradient descent. Every deep learning framework (PyTorch, TensorFlow, JAX) does exactly these steps, but with automatic differentiation, GPU acceleration, and many quality-of-life features. Understanding the raw mechanics helps you debug, optimize, and reason about models.
:::

## Part 2: The Same Network in PyTorch

Now let's rebuild the exact same network using PyTorch. Notice how much simpler it is—but also notice that every step maps directly to something you implemented manually.

### Why PyTorch?

PyTorch provides:
- **Automatic differentiation**: No manual backward pass—PyTorch tracks operations and computes gradients automatically
- **GPU acceleration**: Move tensors to GPU with `.to('cuda')` and training runs 10–100x faster
- **Building blocks**: Pre-built layers, loss functions, optimizers, data loaders
- **Dynamic computation graphs**: Build networks with normal Python control flow

```python title="PyTorch Neural Network"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Same dataset
def make_moons(n_samples=500, noise=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2
    theta1 = np.linspace(0, np.pi, n_each)
    theta2 = np.linspace(0, np.pi, n_each)
    X = np.vstack([
        np.column_stack([np.cos(theta1), np.sin(theta1)]),
        np.column_stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)])
    ])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)])
    X += rng.normal(0, noise, X.shape)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

X_np, y_np = make_moons(500)
X_train = torch.FloatTensor(X_np[:400])
y_train = torch.FloatTensor(y_np[:400]).unsqueeze(1)
X_test = torch.FloatTensor(X_np[400:])
y_test = torch.FloatTensor(y_np[400:]).unsqueeze(1)

# Define the model
class MoonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize
torch.manual_seed(42)
model = MoonClassifier()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model}")
print(f"Total parameters: {total_params}")

# Training loop
for epoch in range(2000):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass + update (PyTorch handles this!)
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute gradients via autograd
    optimizer.step()       # Update parameters
    
    if epoch % 400 == 0:
        with torch.no_grad():
            acc = ((y_pred > 0.5).float() == y_train).float().mean()
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")

# Evaluate
with torch.no_grad():
    train_acc = ((model(X_train) > 0.5).float() == y_train).float().mean()
    test_acc = ((model(X_test) > 0.5).float() == y_test).float().mean()
    print(f"\\nFinal train accuracy: {train_acc:.2%}")
    print(f"Final test accuracy:  {test_acc:.2%}")
```

:::tip[Line-by-Line Walkthrough]
- **`import torch` / `import torch.nn as nn` / `import torch.optim as optim`** — Imports PyTorch's core library, neural network building blocks, and optimizers.
- **`X_train = torch.FloatTensor(X_np[:400])`** — Converts NumPy arrays to PyTorch tensors. Tensors are like NumPy arrays but with automatic gradient tracking and GPU support.
- **`y_train.unsqueeze(1)`** — Adds an extra dimension: changes shape from (400,) to (400, 1) so it matches the model's output shape.
- **`class MoonClassifier(nn.Module):`** — Defines the model by inheriting from PyTorch's base class. Every PyTorch model follows this pattern.
- **`nn.Sequential(...)`** — Chains layers together: data flows through Linear → ReLU → Linear → Sigmoid in order.
- **`nn.Linear(2, 32)`** — A fully connected layer: 2 inputs → 32 outputs. Equivalent to our manual `X @ W + b`.
- **`criterion = nn.BCELoss()`** — Binary Cross-Entropy loss, same as we computed manually but optimized and battle-tested.
- **`optimizer = optim.SGD(model.parameters(), lr=0.5)`** — Stochastic Gradient Descent optimizer that handles the weight updates.
- **`optimizer.zero_grad()`** — Clears leftover gradients from the previous step. PyTorch accumulates gradients by default.
- **`loss.backward()`** — Runs backpropagation automatically — no manual chain rule needed! PyTorch tracks every operation and computes gradients.
- **`optimizer.step()`** — Applies the gradient descent update to all weights.
- **`with torch.no_grad():`** — Disables gradient tracking for evaluation (saves memory and computation).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy
```

**Steps:**
1. Save to `pytorch_nn.py` and run: `python pytorch_nn.py`

**Expected output:**
```
Model: MoonClassifier(...)
Total parameters: 129
Epoch    0 | Loss: 0.7123 | Accuracy: 48.25%
Epoch  400 | Loss: 0.0987 | Accuracy: 97.50%
...

Final train accuracy: 98.25%
Final test accuracy:  97.00%
```

</details>

### Side-by-Side Comparison

| Aspect | NumPy Version | PyTorch Version |
|--------|--------------|-----------------|
| **Forward pass** | Manual matrix multiplication | `model(x)` |
| **Loss** | Implement BCE formula | `nn.BCELoss()` |
| **Gradients** | Derive and code chain rule | `loss.backward()` |
| **Weight update** | `W -= lr * grad` | `optimizer.step()` |
| **GPU support** | No | `model.to('cuda')` |
| **Lines of code** | ~80 | ~30 |
| **Educational value** | Very high | N/A |
| **Production use** | Never | Standard |

:::tip[Always Understand What the Framework Does]
When something goes wrong—and it will—you'll need to reason about gradients, loss landscapes, and numerical stability. The NumPy implementation gives you that foundation. Think of it this way: you should understand manual transmission to be a good driver, even if you normally drive automatic.
:::

## The Training Loop Explained

Every PyTorch training loop follows the same pattern:

```python title="The Canonical Training Loop"
import torch
import torch.nn as nn
import torch.optim as optim

# This is the pattern you'll use for EVERY PyTorch project:

"""
model = YourModel()
criterion = nn.SomeLoss()
optimizer = optim.SomeOptimizer(model.parameters(), lr=...)

for epoch in range(num_epochs):
    # 1. Forward pass: compute predictions
    predictions = model(inputs)
    
    # 2. Compute loss
    loss = criterion(predictions, targets)
    
    # 3. Backward pass: compute gradients
    optimizer.zero_grad()  # MUST zero gradients first (PyTorch accumulates)
    loss.backward()        # Populates .grad for each parameter
    
    # 4. Update parameters
    optimizer.step()       # Applies the optimizer's update rule
    
    # 5. (Optional) Logging, validation, checkpointing
"""

# Why zero_grad() is necessary:
# PyTorch accumulates gradients by default (useful for some advanced cases).
# If you forget zero_grad(), gradients from previous batches add up,
# and your model trains on incorrect gradient information.

# Demonstrating gradient accumulation
x = torch.tensor([2.0], requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Step {i}: x.grad = {x.grad.item()}")
    # Without zero_grad, gradient accumulates: 4, 8, 12 instead of 4, 4, 4

print("\\nWith proper zeroing:")
for i in range(3):
    if x.grad is not None:
        x.grad.zero_()
    y = x ** 2
    y.backward()
    print(f"Step {i}: x.grad = {x.grad.item()}")
```

:::tip[Line-by-Line Walkthrough]
- **The commented block at the top** — Shows the universal pattern every PyTorch training loop follows: forward pass → loss → zero gradients → backward pass → update.
- **`optimizer.zero_grad()`** — Essential step: clears old gradients. If you forget this, gradients accumulate from previous iterations and your model trains on wrong information.
- **`loss.backward()`** — Computes gradients for every parameter in the model by walking backwards through the computation graph (automatic differentiation).
- **`optimizer.step()`** — Applies the optimizer's update rule (e.g., SGD, Adam) to adjust all weights.
- **`x = torch.tensor([2.0], requires_grad=True)`** — Creates a tensor with gradient tracking enabled, so PyTorch will record operations on it.
- **The first loop (without zeroing)** — Shows gradients accumulating: 4, 8, 12 instead of the correct 4, 4, 4. The derivative of x² at x=2 is always 4, but without zeroing, each step adds 4 more.
- **The second loop (with `x.grad.zero_()`)** — Correctly zeroes the gradient each time, getting 4, 4, 4.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `training_loop.py` and run: `python training_loop.py`

**Expected output:**
```
Step 0: x.grad = 4.0
Step 1: x.grad = 8.0
Step 2: x.grad = 12.0

With proper zeroing:
Step 0: x.grad = 4.0
Step 1: x.grad = 4.0
Step 2: x.grad = 4.0
```

</details>

## Adding Mini-Batch Training

Real training uses mini-batches, not the full dataset at once. PyTorch's `DataLoader` handles batching, shuffling, and parallel data loading:

```python title="Training with DataLoader"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Generate data
def make_moons(n_samples=1000, noise=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2
    theta1 = np.linspace(0, np.pi, n_each)
    theta2 = np.linspace(0, np.pi, n_each)
    X = np.vstack([
        np.column_stack([np.cos(theta1), np.sin(theta1)]),
        np.column_stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)])
    ])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)])
    X += rng.normal(0, noise, X.shape)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

X_np, y_np = make_moons(1000)
X_train_t = torch.FloatTensor(X_np[:800])
y_train_t = torch.FloatTensor(y_np[:800]).unsqueeze(1)
X_test_t = torch.FloatTensor(X_np[800:])
y_test_t = torch.FloatTensor(y_np[800:]).unsqueeze(1)

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model with more capacity
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with mini-batches
for epoch in range(100):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t)
            test_pred = model(X_test_t)
            train_acc = ((train_pred > 0.5) == y_train_t).float().mean()
            test_acc = ((test_pred > 0.5) == y_test_t).float().mean()
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss/len(train_loader):.4f} "
                  f"| Train: {train_acc:.2%} | Test: {test_acc:.2%}")

# Final evaluation
model.eval()
with torch.no_grad():
    final_acc = ((model(X_test_t) > 0.5) == y_test_t).float().mean()
    print(f"\\nFinal test accuracy: {final_acc:.2%}")
```

:::tip[Line-by-Line Walkthrough]
- **`from torch.utils.data import TensorDataset, DataLoader`** — Imports PyTorch's data utilities that handle batching and shuffling.
- **`train_dataset = TensorDataset(X_train_t, y_train_t)`** — Wraps tensors into a dataset object that pairs each input with its label.
- **`DataLoader(train_dataset, batch_size=32, shuffle=True)`** — Creates an iterator that serves the data in shuffled mini-batches of 32 samples. Instead of processing all 800 samples at once, we process 32 at a time.
- **`nn.Sequential(...)`** — A deeper model: 2 → 64 → 32 → 1, with ReLU activations between layers.
- **`optim.Adam(model.parameters(), lr=0.001)`** — Adam optimizer: a smarter version of SGD that adapts the learning rate for each parameter individually.
- **`model.train()`** — Tells the model we're in training mode (enables dropout, batch norm updates).
- **`for batch_X, batch_y in train_loader:`** — Loops through the data one mini-batch at a time. Each batch is a random subset of 32 samples.
- **`epoch_loss += loss.item()`** — Accumulates the loss across all batches in one epoch.
- **`model.eval()`** — Switches to evaluation mode (disables dropout, uses running batch norm statistics).
- **`with torch.no_grad():`** — Disables gradient computation during evaluation to save memory and speed things up.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy
```

**Steps:**
1. Save to `dataloader_training.py` and run: `python dataloader_training.py`

**Expected output:**
```
Epoch   0 | Loss: 0.6789 | Train: 63.25% | Test: 64.00%
Epoch  20 | Loss: 0.2451 | Train: 95.75% | Test: 94.50%
Epoch  40 | Loss: 0.0892 | Train: 98.50% | Test: 97.50%
...

Final test accuracy: 99.00%
```

</details>

:::info[model.train() vs. model.eval()]
`model.train()` enables training-specific behavior (dropout, batch normalization running stats updates). `model.eval()` disables them for inference. Always switch modes appropriately—forgetting `model.eval()` during evaluation is a common bug that causes inconsistent results.
:::

## Saving and Loading Models

```python title="Model Checkpointing"
import torch

# Save just the model weights (recommended)
# torch.save(model.state_dict(), 'model_weights.pt')

# Load weights into a new model instance
# new_model = nn.Sequential(...)  # same architecture
# new_model.load_state_dict(torch.load('model_weights.pt'))

# Save everything (model + optimizer state) for resuming training
# checkpoint = {
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss.item(),
# }
# torch.save(checkpoint, 'checkpoint.pt')

# Resume training
# checkpoint = torch.load('checkpoint.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']

print("Model saving patterns:")
print("  torch.save(model.state_dict(), 'weights.pt')  — weights only (recommended)")
print("  torch.save(checkpoint, 'checkpoint.pt')        — full checkpoint for resuming")
```

:::tip[Line-by-Line Walkthrough]
- **`torch.save(model.state_dict(), 'model_weights.pt')`** — Saves just the learned weights to a file. The recommended approach because it's smaller and more portable.
- **`new_model.load_state_dict(torch.load('model_weights.pt'))`** — Loads saved weights into a new model instance. You need to define the same architecture first.
- **The checkpoint dictionary** — Saves everything needed to resume training: the current epoch, model weights, optimizer state (momentum buffers, etc.), and the latest loss.
- **`torch.load('checkpoint.pt')`** — Loads the checkpoint, then you restore both model and optimizer state and continue training from where you left off.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `model_saving.py` and run: `python model_saving.py`

**Expected output:**
```
Model saving patterns:
  torch.save(model.state_dict(), 'weights.pt')  — weights only (recommended)
  torch.save(checkpoint, 'checkpoint.pt')        — full checkpoint for resuming
```
(This code block demonstrates the patterns but the actual save/load lines are commented out. Uncomment them in your own project to save real weights.)

</details>

## Exercises

:::tip[Exercise 1: Add a Hidden Layer to the NumPy Network — intermediate]

Modify the NumPy neural network to have **two** hidden layers: `[2, 32, 16, 1]`.

1. Update the forward pass to compute activations for both hidden layers
2. Update the backward pass to propagate gradients through the extra layer
3. Train on the moons dataset and compare accuracy to the single-hidden-layer version
4. Does adding depth help? Why or why not for this problem?

<details>
<summary>Hints</summary>

1. Change layer_sizes to [2, 32, 16, 1].
2. The backward pass now has one more layer to propagate through.
3. Make sure the delta computation chains correctly through the extra layer.

</details>

:::

:::tip[Exercise 2: Implement Adam Optimizer from Scratch — advanced]

Replace the simple SGD update in the NumPy network with Adam:

:::info[Plain English: What Is Adam?]
Think of regular gradient descent like rolling a ball downhill on a bumpy surface — it can get stuck in small dips or oscillate wildly. Adam is like giving the ball *momentum* (it remembers which direction it's been rolling) and *adaptive speed* (it slows down on steep slopes and speeds up on flat ones). It combines two ideas: (1) a running average of recent gradients (direction memory), and (2) a running average of recent squared gradients (speed control).
:::

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**Reading the formula:** $m_t$ is the momentum (a smoothed average of recent gradients). $g_t$ is the current gradient at step $t$. $\beta_1$ (usually 0.9) controls how much we remember old gradients vs. the current one. $v_t$ is the velocity (a smoothed average of squared gradients), controlled by $\beta_2$ (usually 0.999). Together, $m$ tracks *direction* and $v$ tracks *magnitude*.

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Reading the formula:** $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected versions of $m$ and $v$. In the early steps, $m$ and $v$ are biased toward zero because they start at zero. Dividing by $(1 - \beta^t)$ corrects this so the estimates are accurate from the very first step. As $t$ grows large, this correction fades to 1 (no effect).

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**Reading the formula:** $\theta_t$ represents the model's parameters (weights) at step $t$. $\alpha$ is the learning rate (step size). The fraction $\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ divides the momentum by the root of the velocity — if a parameter has been getting large gradients, $\hat{v}$ is big, so the step shrinks (adaptive!). $\epsilon$ (a tiny number like $10^{-8}$) prevents dividing by zero.

Compare training speed and final accuracy between SGD and Adam on the moons dataset.

<details>
<summary>Hints</summary>

1. Adam tracks two moving averages: m (momentum) and v (squared gradients).
2. Both need bias correction in early steps.
3. Default hyperparameters: beta1=0.9, beta2=0.999, epsilon=1e-8.

</details>

:::

:::tip[Exercise 3: Multi-Class Classification — intermediate]

Extend the PyTorch model to handle **multi-class classification**:

1. Generate a dataset with 3 classes (use `sklearn.datasets.make_blobs` with 3 centers)
2. Build a network with 3 output neurons
3. Use `nn.CrossEntropyLoss()` as the loss function
4. Train and evaluate, reporting per-class accuracy
5. Print the confusion matrix

<details>
<summary>Hints</summary>

1. Use nn.CrossEntropyLoss() — it combines LogSoftmax + NLLLoss.
2. Output layer should have C neurons (one per class), no activation (CrossEntropyLoss handles it).
3. Labels should be integers (class indices), not one-hot encoded.

</details>

:::

:::tip[Exercise 4: Learning Rate Finder — advanced]

Implement a **learning rate finder** (Smith, 2017):

1. Start with a very small learning rate (e.g., $10^{-7}$)
2. For each mini-batch, multiply the learning rate by a constant factor (e.g., 1.1)
3. Record the loss at each step
4. Plot loss vs. learning rate (log scale)
5. The optimal learning rate is where the loss decreases most steeply (before it starts diverging)

Test on the moons dataset and compare the suggested LR to one you found manually.

<details>
<summary>Hints</summary>

1. Start with a very small LR (1e-7) and increase exponentially.
2. Track loss at each step.
3. The best LR is typically where the loss is decreasing fastest.

</details>

:::

## Resources

- **[PyTorch Tutorials](https://pytorch.org/tutorials/)** _(tutorial)_ — Official PyTorch tutorials. Start with 'Learn the Basics' and 'Quickstart'.

- **[Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** _(video)_ by Andrej Karpathy — Build neural networks from scratch, progressing to GPT. The best hands-on video series for understanding deep learning.

- **[Dive into Deep Learning](https://d2l.ai/)** _(book)_ by Zhang, Lipton, Li & Smola — Interactive deep learning textbook with runnable code. Available in PyTorch, TensorFlow, and JAX.

- **[PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)** _(tutorial)_ by Edward Yang — Deep dive into how PyTorch works under the hood. Read after you're comfortable using PyTorch.

---

**Next up**: You're ready for the Month 1 capstone project — building a complete MNIST digit classifier with PyTorch.
