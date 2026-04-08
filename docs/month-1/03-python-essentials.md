---
sidebar_position: 3
title: "Python Essentials"
slug: python-essentials
---


# Python Essentials

:::info[What You'll Learn]
- Setting up Python, virtual environments, and installing packages
- Variables, data types, control flow, and functions
- Lists, dictionaries, tuples, and comprehensions
- Classes, objects, and the basics of OOP in Python
- File I/O, error handling, and writing clean Python
:::

:::note[Prerequisites]
Basic programming experience in any language. If you've written functions and loops in JavaScript, Java, C++, or similar, you're ready.
:::

**Estimated time:** Reading: ~45 min | Exercises: ~4 hours

The entire AI/ML ecosystem runs on Python. Before we touch NumPy or neural networks, you need to be comfortable writing Python itself — setting up environments, working with data structures, writing functions and classes, and reading other people's code. This lesson gets you there.

If you already write Python daily, skim through and try the exercises at the end. If you're coming from another language, work through every section.

## Setting Up Your Environment

### Installing Python

You need Python 3.10 or newer. Check what you have:

```bash
python3 --version
```

If you don't have Python or need a newer version:
- **macOS**: `brew install python` (via [Homebrew](https://brew.sh))
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3 python3-pip python3-venv`
- **Windows**: Download from [python.org](https://www.python.org/downloads/) — check "Add Python to PATH" during installation

### Virtual Environments

Never install packages globally. Virtual environments isolate each project's dependencies so they don't conflict.

```bash
# Create a virtual environment
python3 -m venv myenv

# Activate it
# macOS/Linux:
source myenv/bin/activate

# Windows:
myenv\Scripts\activate

# Your prompt should now show (myenv)
# Install packages inside the environment
pip install numpy pandas matplotlib

# Deactivate when done
deactivate
```

:::tip[Why Virtual Environments Matter]
Project A needs `torch==2.0` and Project B needs `torch==2.3`. Without virtual environments, installing one breaks the other. With them, each project has its own isolated set of packages.
:::

### pip and requirements.txt

pip is Python's package manager. Common commands:

```bash
pip install numpy                # install a package
pip install numpy==1.26.4        # install a specific version
pip install -r requirements.txt  # install from a file
pip freeze > requirements.txt    # save current packages to a file
pip list                         # list installed packages
```

Create a `requirements.txt` for every project. It makes your work reproducible.

## Variables and Data Types

Python is dynamically typed — you don't declare types, but every value has one.

```python title="Variables and Types"
# Integers and floats
learning_rate = 0.001
epochs = 100
batch_size = 32

# Strings
model_name = "gpt-2"
status = 'training'

# Booleans
is_training = True
use_gpu = False

# None — represents "no value"
best_model = None

# Check types
print(type(learning_rate))  # <class 'float'>
print(type(epochs))         # <class 'int'>
print(type(model_name))     # <class 'str'>
print(type(is_training))    # <class 'bool'>
```

### Numbers and Arithmetic

```python title="Arithmetic"
a = 10
b = 3

print(a + b)     # 13    addition
print(a - b)     # 7     subtraction
print(a * b)     # 30    multiplication
print(a / b)     # 3.333 true division (always float)
print(a // b)    # 3     floor division (integer result)
print(a % b)     # 1     modulo (remainder)
print(a ** b)    # 1000  exponentiation

# Augmented assignment
count = 0
count += 1       # same as count = count + 1
count *= 2       # same as count = count * 2
```

### f-Strings

f-strings are Python's most readable way to format output. You'll see them everywhere in ML code.

```python title="f-Strings"
epoch = 15
loss = 0.0342
accuracy = 0.9567

# Embed variables directly in strings
print(f"Epoch {epoch}: loss={loss}, accuracy={accuracy}")

# Format numbers
print(f"Loss: {loss:.4f}")          # 4 decimal places
print(f"Accuracy: {accuracy:.1%}")  # as percentage: 95.7%
print(f"Epoch: {epoch:03d}")        # zero-padded: 015
print(f"Size: {1_500_000:,}")       # with commas: 1,500,000

# Expressions inside f-strings
print(f"2 + 3 = {2 + 3}")
print(f"Model: {model_name.upper()}")
```

## Data Structures

### Lists

Lists are ordered, mutable sequences. You'll use them constantly.

```python title="Lists"
# Create lists
losses = [2.5, 1.8, 1.2, 0.9, 0.5]
layers = ["embedding", "attention", "ffn", "output"]

# Access elements (0-indexed)
print(losses[0])     # 2.5 (first)
print(losses[-1])    # 0.5 (last)
print(layers[1:3])   # ['attention', 'ffn'] (slice: index 1 up to but not 3)

# Modify
losses.append(0.3)           # add to end
layers.insert(1, "norm")     # insert at index 1
removed = losses.pop()       # remove and return last element
print(f"Removed: {removed}")

# Length
print(f"Number of layers: {len(layers)}")

# Check membership
print("attention" in layers)   # True
print("conv" in layers)        # False

# Useful operations
print(f"Min loss: {min(losses)}")
print(f"Max loss: {max(losses)}")
print(f"Sum: {sum(losses)}")
```

### List Comprehensions

Comprehensions are Python's way of building lists concisely. They replace many `for` loops.

```python title="List Comprehensions"
# Basic: [expression for item in iterable]
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition: [expression for item in iterable if condition]
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Transform a list
names = ["alice", "bob", "charlie"]
upper = [name.upper() for name in names]
print(upper)  # ['ALICE', 'BOB', 'CHARLIE']

# Practical ML example: filter out bad results
results = [
    {"model": "A", "accuracy": 0.92},
    {"model": "B", "accuracy": 0.78},
    {"model": "C", "accuracy": 0.95},
    {"model": "D", "accuracy": 0.81},
]
good_models = [r["model"] for r in results if r["accuracy"] > 0.85]
print(f"Good models: {good_models}")  # ['A', 'C']
```

### Dictionaries

Dictionaries map keys to values. They're Python's hash table — O(1) lookup.

```python title="Dictionaries"
# Create a config dictionary (common pattern in ML)
config = {
    "model_name": "transformer",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "hidden_dim": 512,
}

# Access values
print(config["learning_rate"])     # 0.001
print(config.get("dropout", 0.0)) # 0.0 (default if key missing)

# Modify
config["learning_rate"] = 0.0005  # update
config["dropout"] = 0.1           # add new key

# Iterate
for key, value in config.items():
    print(f"  {key}: {value}")

# Dictionary comprehension
squared = {x: x ** 2 for x in range(5)}
print(squared)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Nested dictionaries (common for experiment results)
experiments = {
    "run_1": {"loss": 0.5, "accuracy": 0.89},
    "run_2": {"loss": 0.3, "accuracy": 0.93},
}
print(experiments["run_2"]["accuracy"])  # 0.93
```

### Tuples

Tuples are like lists but immutable (can't be changed after creation). Used for things that shouldn't change, like shapes and coordinates.

```python title="Tuples"
# Create tuples
point = (3, 4)
shape = (224, 224, 3)  # image: height, width, channels

# Unpack
x, y = point
h, w, c = shape
print(f"Image: {h}x{w} with {c} channels")

# Tuples as dictionary keys (lists can't be dict keys)
cache = {}
cache[(128, 64)] = "model_small"
cache[(512, 256)] = "model_large"

# Swap variables
a, b = 1, 2
a, b = b, a  # a=2, b=1
```

## Control Flow

### if / elif / else

```python title="Conditionals"
accuracy = 0.87

if accuracy >= 0.95:
    grade = "excellent"
elif accuracy >= 0.85:
    grade = "good"
elif accuracy >= 0.70:
    grade = "acceptable"
else:
    grade = "needs improvement"

print(f"Accuracy {accuracy:.0%} → {grade}")

# Ternary (inline if)
status = "converged" if accuracy > 0.9 else "training"
print(f"Status: {status}")
```

### for Loops

```python title="Loops"
# Iterate over a list
layers = ["input", "hidden_1", "hidden_2", "output"]
for layer in layers:
    print(f"  Processing: {layer}")

# range() for counting
for i in range(5):
    print(f"  Epoch {i}")  # 0, 1, 2, 3, 4

# enumerate() gives index + value
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer}")

# zip() pairs two lists together
names = ["Alice", "Bob", "Charlie"]
scores = [92, 85, 78]
for name, score in zip(names, scores):
    print(f"  {name}: {score}")

# Loop with dictionary
config = {"lr": 0.01, "epochs": 50, "batch": 32}
for key, value in config.items():
    print(f"  {key} = {value}")
```

### while Loops

```python title="While Loops"
# Train until convergence
loss = 10.0
epoch = 0
while loss > 0.1:
    loss *= 0.7  # simulate decreasing loss
    epoch += 1
    if epoch > 100:
        print("  Stopped: max epochs reached")
        break

print(f"Converged at epoch {epoch}, loss={loss:.4f}")
```

## Functions

Functions are how you organize reusable code. ML code is full of them.

```python title="Functions"
# Basic function
def compute_accuracy(correct, total):
    """Compute classification accuracy."""
    return correct / total

acc = compute_accuracy(92, 100)
print(f"Accuracy: {acc:.1%}")  # 92.0%


# Default arguments
def create_model(hidden_dim=256, dropout=0.1, num_layers=6):
    """Create a model config with sensible defaults."""
    return {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_layers": num_layers,
    }

# Call with defaults
model1 = create_model()
# Override some arguments
model2 = create_model(hidden_dim=512, num_layers=12)
print(f"Model 1: {model1}")
print(f"Model 2: {model2}")


# *args and **kwargs
def log_metrics(epoch, **metrics):
    """Log any set of metrics for a given epoch."""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    print(f"  Epoch {epoch}: {', '.join(parts)}")

log_metrics(1, loss=2.45, accuracy=0.62, lr=0.001)
log_metrics(10, loss=0.34, accuracy=0.91, lr=0.0005)


# Functions as arguments (higher-order functions)
def apply_to_list(func, data):
    """Apply a function to every element."""
    return [func(x) for x in data]

import math
values = [1, 4, 9, 16, 25]
roots = apply_to_list(math.sqrt, values)
print(f"Square roots: {roots}")
```

### Lambda Functions

Short anonymous functions, often used as arguments.

```python title="Lambda Functions"
# Sort a list of tuples by the second element
results = [("model_a", 0.89), ("model_b", 0.95), ("model_c", 0.91)]
results.sort(key=lambda x: x[1], reverse=True)
print(results)  # [('model_b', 0.95), ('model_c', 0.91), ('model_a', 0.89)]

# With map and filter
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
doubled = list(map(lambda x: x * 2, numbers))
print(f"Evens: {evens}")
print(f"Doubled: {doubled}")
```

## Classes

Classes are how PyTorch organizes neural networks. Every model you build will be a class.

```python title="Classes"
class Experiment:
    """Track a machine learning experiment."""

    def __init__(self, name, learning_rate=0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.history = []

    def log(self, epoch, loss, accuracy):
        """Record metrics for one epoch."""
        self.history.append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
        })

    def best_epoch(self):
        """Return the epoch with the lowest loss."""
        return min(self.history, key=lambda x: x["loss"])

    def summary(self):
        """Print a summary of the experiment."""
        best = self.best_epoch()
        print(f"Experiment: {self.name}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Total epochs: {len(self.history)}")
        print(f"  Best epoch: {best['epoch']} "
              f"(loss={best['loss']:.4f}, acc={best['accuracy']:.4f})")

    def __repr__(self):
        return f"Experiment('{self.name}', lr={self.learning_rate})"


# Use the class
exp = Experiment("baseline", learning_rate=0.01)
exp.log(1, loss=2.5, accuracy=0.45)
exp.log(2, loss=1.8, accuracy=0.62)
exp.log(3, loss=0.9, accuracy=0.78)
exp.log(4, loss=0.5, accuracy=0.89)

exp.summary()
print(repr(exp))
```

### Inheritance

PyTorch models use inheritance heavily. You subclass `nn.Module` and override methods.

```python title="Inheritance"
class Model:
    """Base class for all models."""

    def __init__(self, name):
        self.name = name
        self.trained = False

    def train(self, data):
        """Train the model. Subclasses should override this."""
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, x):
        """Make a prediction. Subclasses should override this."""
        raise NotImplementedError("Subclasses must implement predict()")


class LinearModel(Model):
    """A simple linear model."""

    def __init__(self, name, weight=1.0, bias=0.0):
        super().__init__(name)  # call parent's __init__
        self.weight = weight
        self.bias = bias

    def train(self, data):
        print(f"Training {self.name} on {len(data)} samples...")
        self.trained = True

    def predict(self, x):
        return self.weight * x + self.bias


model = LinearModel("my_model", weight=2.5, bias=1.0)
model.train([1, 2, 3, 4, 5])
print(f"predict(3) = {model.predict(3)}")   # 8.5
print(f"predict(10) = {model.predict(10)}") # 26.0
```

:::tip[Why This Matters for PyTorch]
Every PyTorch neural network follows this exact pattern: you create a class that inherits from `nn.Module`, define the layers in `__init__`, and implement the `forward` method. If you understand the `Model` → `LinearModel` pattern above, you'll understand PyTorch models immediately.
:::

## Error Handling

ML code fails in predictable ways: missing files, bad shapes, out-of-memory. Handle errors gracefully.

```python title="Error Handling"
# try / except
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Warning: division by zero, returning 0")
        return 0.0

print(safe_divide(10, 3))   # 3.333...
print(safe_divide(10, 0))   # Warning + 0.0


# Multiple exception types
def load_config(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return None
    except PermissionError:
        print(f"No permission to read: {path}")
        return None


# Raising your own errors
def set_learning_rate(lr):
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")
    if lr > 1:
        raise ValueError(f"Learning rate too large: {lr}")
    return lr
```

## File I/O

Reading and writing files is fundamental — loading datasets, saving models, logging results.

```python title="File I/O"
import json

# Writing and reading text files
with open("training_log.txt", "w") as f:
    for epoch in range(5):
        f.write(f"Epoch {epoch}: loss=0.{9 - epoch}\n")

with open("training_log.txt", "r") as f:
    contents = f.read()
    print(contents)


# JSON — the standard for config files and API responses
config = {
    "model": "transformer",
    "layers": 6,
    "hidden_dim": 512,
    "dropout": 0.1,
}

# Save to JSON
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# Load from JSON
with open("config.json", "r") as f:
    loaded_config = json.load(f)
    print(f"Loaded: {loaded_config}")


# CSV — line by line (for small files; use Pandas for large ones)
import csv

data = [
    ["epoch", "loss", "accuracy"],
    [1, 2.5, 0.45],
    [2, 1.8, 0.62],
    [3, 0.9, 0.78],
]

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

with open("results.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

:::tip[The `with` Statement]
Always use `with open(...) as f:` instead of `f = open(...)`. The `with` block automatically closes the file when you're done, even if an error occurs. This prevents resource leaks.
:::

## Type Hints

Type hints make code more readable and help your editor catch bugs. PyTorch and Hugging Face code uses them extensively.

```python title="Type Hints"
from typing import Optional

def train_model(
    model_name: str,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Train a model and return metrics."""
    print(f"Training {model_name} for {epochs} epochs on {device}")

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")

    return {
        "model": model_name,
        "final_loss": 0.05,
        "final_accuracy": 0.96,
    }

result = train_model("gpt-small", epochs=50, device="cuda")
print(result)
```

Type hints don't change how the code runs — they're documentation that your editor and tools like `mypy` can check.

## Common Patterns in ML Code

These patterns appear in almost every ML codebase. Recognize them now so they don't slow you down later.

### Unpacking and Star Expressions

```python title="Unpacking"
# Unpacking function returns
def get_data_splits(data):
    n = len(data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return data[:train_end], data[train_end:val_end], data[val_end:]

data = list(range(100))
train, val, test = get_data_splits(data)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Star expression: grab the rest
first, *middle, last = [1, 2, 3, 4, 5]
print(f"first={first}, middle={middle}, last={last}")
```

### Context Managers and Decorators

```python title="Context Managers and Decorators"
import time

# Timing decorator — you'll see this in training code
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

@timer
def slow_computation(n):
    total = sum(i ** 2 for i in range(n))
    return total

result = slow_computation(1_000_000)
print(f"Result: {result}")
```

### Generators

Generators produce values one at a time instead of building an entire list in memory. Essential for processing large datasets.

```python title="Generators"
def data_batches(data, batch_size):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

dataset = list(range(25))
for batch in data_batches(dataset, batch_size=8):
    print(f"  Batch ({len(batch)} items): {batch}")
```

## Exercises

:::tip[Exercise 1: Python Basics Warm-Up — beginner]

Write a function `analyze_scores(scores)` that takes a list of exam scores (integers 0–100) and returns a dictionary with:
- `"mean"`: the average score
- `"median"`: the middle score (sort first)
- `"pass_rate"`: fraction of scores >= 60
- `"grade_distribution"`: a dictionary mapping letter grades (A/B/C/D/F) to counts

Test it with `[85, 92, 45, 78, 63, 90, 55, 71, 88, 50]`.

<details>
<summary>Hints</summary>

1. Use `sum(scores) / len(scores)` for mean.
2. Sort the list and take the middle element for median (handle even/odd lengths).
3. Use a dictionary with grade boundaries to count grade distribution.

</details>

:::

:::tip[Exercise 2: Config System — intermediate]

Build a simple configuration system:

1. Write a `Config` class that accepts a dictionary of default values
2. Support dot-notation access (`config.learning_rate`) using `__getattr__`
3. Support updating via `config.update({"learning_rate": 0.01})`
4. Add a `save(path)` method that writes to JSON and a `load(path)` class method that reads from JSON
5. Add a `diff(other_config)` method that returns which keys have different values

<details>
<summary>Hints</summary>

1. Store the dictionary as `self._data` and use `__getattr__` to access its keys.
2. Use `json.dump` and `json.load` for save/load.
3. For `diff`, compare keys and values between `self._data` and `other._data`.

</details>

:::

:::tip[Exercise 3: Data Pipeline — intermediate]

Build a data processing pipeline using generators:

1. Write a generator `read_lines(filepath)` that yields lines from a file one at a time
2. Write a generator `parse_csv_line(lines)` that takes a line generator and yields dictionaries (first line is the header)
3. Write a generator `filter_rows(rows, column, min_value)` that filters rows where `column >= min_value`
4. Chain them together: `filter_rows(parse_csv_line(read_lines("data.csv")), "score", 80)`

Create a test CSV file with at least 20 rows to verify it works.

<details>
<summary>Hints</summary>

1. Use `yield` to make each function a generator.
2. The first line of a CSV is the header — use `next()` to grab it.
3. The pipeline should process one row at a time without loading the whole file.

</details>

:::

:::tip[Exercise 4: Experiment Tracker — advanced]

Build a mini experiment tracker class with these features:

1. `Tracker.__init__(experiment_name)` — initialize with a name
2. `tracker.log(epoch, **metrics)` — log metrics for an epoch
3. `tracker.best(metric, mode="min")` — return the epoch with the best value for a metric (mode is "min" or "max")
4. `tracker.plot_metric(metric)` — print an ASCII bar chart of the metric over epochs
5. `tracker.compare(other_tracker)` — print a side-by-side comparison of final metrics
6. `tracker.save(path)` / `Tracker.load(path)` — serialize to/from JSON

Test with two experiments that have different learning rates.

<details>
<summary>Hints</summary>

1. Store metrics as a list of dictionaries, one per epoch.
2. For the ASCII bar chart, scale values to fit in ~50 characters.
3. Use `json` for serialization — make sure the data is JSON-serializable.

</details>

:::

## Resources

- **[The Official Python Tutorial](https://docs.python.org/3/tutorial/)** _(tutorial)_ — Python's own tutorial. Thorough, authoritative, and free.

- **[Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)** _(book)_ by Al Sweigart — Practical Python for beginners. Free to read online.

- **[Real Python](https://realpython.com/)** _(tutorial)_ — High-quality Python tutorials covering beginner to advanced topics.

- **[Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)** _(book)_ by Jake VanderPlas — Covers Python, NumPy, Pandas, and Matplotlib in depth. Free online.

- **[Google's Python Class](https://developers.google.com/edu/python)** _(course)_ by Google — A free, compact Python course designed for people with some programming experience.

---

**Next up**: Now that you can write Python, we'll learn the libraries that make Python the language of AI: NumPy for numerical computing, Pandas for data manipulation, and Matplotlib for visualization.
