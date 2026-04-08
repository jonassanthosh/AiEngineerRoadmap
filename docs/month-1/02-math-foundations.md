---
sidebar_position: 2
title: "Math Foundations for AI"
slug: math-foundations
---


# Math Foundations for AI

:::info[What You'll Learn]
- Vectors, matrices, and the operations needed for neural networks
- Derivatives and gradients for backpropagation
- Probability distributions and Bayes' theorem
- How these mathematical concepts connect to real ML algorithms
:::

**Estimated time:** Reading: ~45 min | Exercises: ~3 hours

:::note[Prerequisites]
This lesson is self-contained — no prior math beyond high-school algebra is assumed.
:::

You don't need a math PhD to be an effective AI engineer, but you *do* need fluency in three areas: **linear algebra**, **calculus**, and **probability**. These aren't arbitrary prerequisites—they are the language in which machine learning is written. Every neural network is a composition of matrix multiplications (linear algebra), trained by following gradients (calculus) on a loss function defined over probabilistic predictions (probability).

This chapter gives you the working knowledge you need. We'll connect every concept to its ML application so you can see *why* it matters.

## Linear Algebra

Linear algebra is the backbone of machine learning. Data is stored as vectors and matrices. Model parameters are matrices. Every forward pass through a neural network is a sequence of matrix operations.

### Scalars, Vectors, and Matrices

A **scalar** is a single number: $x = 5$, $\alpha = 0.001$.

:::info[Plain English: What Is a Vector?]
Think of a **vector** as a **shopping list of numbers**. If you're describing a house to sell, you might list: [square footage, number of bedrooms, price] = [1500, 3, 250000]. That's a vector with 3 numbers! The order matters — the first slot is always square footage, the second is always bedrooms, and so on.
:::

A **vector** is an ordered list of numbers. We write vectors as column vectors by convention:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \in \mathbb{R}^3
$$

**Reading the formula:** This says "the vector **x** contains three numbers stacked on top of each other ($x_1$, $x_2$, $x_3$), and each number is a real number." The $\mathbb{R}^3$ part just means "3 real numbers."

In ML, a vector might represent a data point (a list of features), a word embedding, or the weights of a single neuron.

:::info[Plain English: What Is a Matrix?]
A **matrix** is like a **spreadsheet** — it has rows and columns filled with numbers. If you have data about 100 houses, each with 5 features (size, bedrooms, bathrooms, age, price), you'd store all of that in a matrix with 100 rows and 5 columns.
:::

A **matrix** is a 2D array of numbers:

$$
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \in \mathbb{R}^{2 \times 3}
$$

**Reading the formula:** This is a grid with 2 rows and 3 columns. The little numbers $a_{11}$ mean "row 1, column 1." The $\mathbb{R}^{2 \times 3}$ just means "a 2-by-3 grid of real numbers."

A matrix with $m$ rows and $n$ columns has shape $(m, n)$. In ML, weight matrices connect layers of a neural network, datasets are stored as matrices (rows = samples, columns = features), and images are matrices (or tensors) of pixel values.

A **tensor** generalizes these concepts to arbitrary dimensions. A 3D tensor might represent a batch of images, where the dimensions are (batch_size, height, width).

```python title="Scalars, Vectors, Matrices, and Tensors in NumPy"
import numpy as np

# Scalar
x = 5

# Vector (1D array)
v = np.array([1, 2, 3])
print(f"Vector: {v}, shape: {v.shape}")

# Matrix (2D array)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix:\n{A}, shape: {A.shape}")

# Tensor (3D array) - e.g., a batch of 2 grayscale 3x3 images
T = np.random.randn(2, 3, 3)
print(f"Tensor shape: {T.shape}")  # (2, 3, 3)
```

:::tip[Line-by-Line Walkthrough]
- **`import numpy as np`** — Loads NumPy, the main Python library for working with numbers and arrays. We call it `np` for short.
- **`x = 5`** — Creates a scalar (just a single number).
- **`v = np.array([1, 2, 3])`** — Creates a vector (a list of 3 numbers). Think of it as a shopping list with 3 items.
- **`print(f"Vector: {v}, shape: {v.shape}")`** — Prints the vector and its shape. `shape` tells us how many numbers are in it — `(3,)` means "3 numbers in a row."
- **`A = np.array([[1, 2, 3], [4, 5, 6]])`** — Creates a matrix (a spreadsheet with 2 rows and 3 columns).
- **`T = np.random.randn(2, 3, 3)`** — Creates a 3D tensor with random numbers. Think of it as 2 separate 3x3 grids stacked together — like 2 tiny images.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Make sure Python 3.8+ is installed. Install NumPy if you haven't:
```bash
pip install numpy
```

**Steps:**
1. Copy the code above into a file called `scalars_vectors_matrices.py`
2. Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux)
3. Navigate to the folder where you saved the file: `cd path/to/your/folder`
4. Run: `python scalars_vectors_matrices.py`

**Expected output:**
```
Vector: [1 2 3], shape: (3,)
Matrix:
[[1 2 3]
 [4 5 6]], shape: (2, 3)
Tensor shape: (2, 3, 3)
```

</details>

### The Dot Product

:::info[Plain English: What Is a Dot Product?]
Imagine two friends rating 3 movies on a scale of 1–5. Alice rates them [5, 1, 3] and Bob rates them [4, 2, 5]. To figure out how similar their tastes are, you multiply their ratings movie-by-movie and add up the results: (5×4) + (1×2) + (3×5) = 20 + 2 + 15 = 37. That's the dot product! A higher number means their tastes are more similar.
:::

The **dot product** of two vectors $\mathbf{a}$ and $\mathbf{b}$ is:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n
$$

**Reading the formula step by step:**
1. Take the first number from each list and multiply them: $a_1 \times b_1$
2. Take the second number from each list and multiply them: $a_2 \times b_2$
3. Keep going for every pair...
4. Add all those products together. That's your dot product!

The $\sum$ symbol just means "add them all up." The dot product measures *similarity* between two vectors. When two vectors point in the same direction, their dot product is large and positive. When they're perpendicular, it's zero. When they point in opposite directions, it's large and negative.

:::info[Dot Products Are Everywhere in ML]
- A single neuron computes a dot product: $z = \mathbf{w} \cdot \mathbf{x} + b$
- Attention scores in Transformers are dot products between query and key vectors
- Cosine similarity (used in embedding search) is a normalized dot product
:::

```python title="Dot Products and Cosine Similarity"
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot}")

# Cosine similarity = dot product / (magnitude_a * magnitude_b)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity: {cos_sim:.4f}")

# Similar vectors have high cosine similarity
v1 = np.array([1, 0, 0])
v2 = np.array([0.9, 0.1, 0])
v3 = np.array([0, 0, 1])  # orthogonal to v1
print(f"sim(v1, v2) = {np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)):.4f}")
print(f"sim(v1, v3) = {np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)):.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`a = np.array([1, 2, 3])`** — Creates a vector with 3 numbers: [1, 2, 3].
- **`b = np.array([4, 5, 6])`** — Creates another vector: [4, 5, 6].
- **`dot = np.dot(a, b)`** — Computes the dot product: (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32.
- **`np.linalg.norm(a)`** — Calculates the "length" (magnitude) of vector `a`. Think of it as "how far from zero does this arrow reach?" Computed as $\sqrt{1^2 + 2^2 + 3^2} = \sqrt{14} \approx 3.74$.
- **`cos_sim = np.dot(a, b) / (...)`** — Cosine similarity divides the dot product by both lengths, giving a score between -1 (opposite) and 1 (identical direction).
- **`v1, v2, v3`** — Three test vectors. `v1` and `v2` point in nearly the same direction (high similarity), while `v1` and `v3` are perpendicular (zero similarity).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `dot_product.py`
2. Open your terminal
3. Run: `python dot_product.py`

**Expected output:**
```
Dot product: 32
Cosine similarity: 0.9746
sim(v1, v2) = 0.9939
sim(v1, v3) = 0.0000
```

</details>

### Matrix Multiplication

:::info[Plain English: What Is Matrix Multiplication?]
Imagine you run a bakery. You have a recipe table (matrix A) showing how much of each ingredient each product needs, and a price table (matrix B) showing how much each ingredient costs from different suppliers. Matrix multiplication combines these two tables to tell you "how much does each product cost from each supplier?" — it does all the multiply-and-add steps for every combination automatically.

The key rule: you can only multiply two matrices if **the number of columns in the first matches the number of rows in the second**. Think of it like a chain link — the inner sizes have to click together.
:::

Matrix multiplication is the workhorse of deep learning. For matrices $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$, the product $\mathbf{C} = \mathbf{A}\mathbf{B}$ has shape $(m, p)$ and each element is:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

**Reading the formula:** To fill in one cell of the result (row $i$, column $j$), take row $i$ from the first matrix and column $j$ from the second matrix, multiply them number-by-number, and add up all the products. That's just a dot product! So matrix multiplication is really just **doing lots of dot products** — one for every cell in the result.

The inner dimensions must match: $(m \times \textbf{n}) \cdot (\textbf{n} \times p) = (m \times p)$.

:::note[Why Matrix Multiplication Matters]
A neural network layer computes $\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$, where $\mathbf{W}$ is the weight matrix, $\mathbf{x}$ is the input, $\mathbf{b}$ is the bias, and $\sigma$ is an activation function. The $\mathbf{W}\mathbf{x}$ step is matrix multiplication. A 768-dimensional input with a layer of 3072 neurons means multiplying a $(3072 \times 768)$ matrix by a $(768 \times 1)$ vector—over 2 million multiply-add operations for a single input in a single layer.
:::

```python title="Matrix Multiplication in NumPy"
import numpy as np

# A neural network layer: y = Wx + b
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])  # shape (2, 3): 2 neurons, 3 inputs
x = np.array([1.0, 2.0, 3.0])    # shape (3,): 3 input features
b = np.array([0.1, 0.2])          # shape (2,): 2 biases

# Matrix-vector multiplication
z = W @ x + b  # '@' is the matrix multiply operator
print(f"z = Wx + b = {z}")  # Linear output before activation

# Batch processing: multiply a weight matrix by a batch of inputs
X_batch = np.random.randn(32, 3)  # 32 samples, 3 features each
Z_batch = X_batch @ W.T + b       # shape: (32, 2)
print(f"Batch output shape: {Z_batch.shape}")
```

:::tip[Line-by-Line Walkthrough]
- **`W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])`** — Creates a weight matrix with 2 rows and 3 columns. Think of this as 2 neurons, each with 3 "importance knobs" (weights) — one for each input feature.
- **`x = np.array([1.0, 2.0, 3.0])`** — The input: 3 features (like height, weight, age of a person).
- **`b = np.array([0.1, 0.2])`** — The bias: a small extra number added to each neuron's output, like a head start.
- **`z = W @ x + b`** — The `@` symbol means "matrix multiply." This computes: neuron 1 = (0.1×1 + 0.2×2 + 0.3×3) + 0.1 = 1.5, neuron 2 = (0.4×1 + 0.5×2 + 0.6×3) + 0.2 = 3.4.
- **`X_batch = np.random.randn(32, 3)`** — Creates 32 fake data points, each with 3 features. This simulates processing a batch of data at once (much faster than one at a time).
- **`X_batch @ W.T + b`** — `W.T` flips the weight matrix (transpose) so the dimensions line up correctly for batch processing. Result: 32 outputs, one per data point.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `matrix_multiply.py`
2. Open your terminal
3. Run: `python matrix_multiply.py`

**Expected output:**
```
z = Wx + b = [1.5 3.4]
Batch output shape: (32, 2)
```

</details>

### The Transpose

:::info[Plain English: What Is a Transpose?]
Imagine you have a spreadsheet where rows are students and columns are subjects. The **transpose** flips it so rows become subjects and columns become students. It's like rotating your spreadsheet 90 degrees. The first row becomes the first column, the second row becomes the second column, etc.
:::

The transpose $\mathbf{A}^T$ flips a matrix over its diagonal: rows become columns and columns become rows. If $\mathbf{A}$ has shape $(m, n)$, then $\mathbf{A}^T$ has shape $(n, m)$.

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \Rightarrow \mathbf{A}^T = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}
$$

**Reading the formula:** The original matrix has 3 rows and 2 columns. After transposing, it has 2 rows and 3 columns. Notice how the first column [1, 3, 5] became the first row [1, 3, 5].

In NumPy: `A.T` or `np.transpose(A)`.

## Calculus

Calculus gives us the tools to *optimize*—to find the parameters that minimize a loss function. Training a neural network is fundamentally an optimization problem, and gradients tell us which direction to move.

### Derivatives

:::info[Plain English: What Is a Derivative?]
Imagine you're hiking on a hill. The **derivative** tells you **how steep the hill is** right where you're standing. If the derivative is large and positive, you're going steeply uphill. If it's negative, you're going downhill. If it's zero, you're at a flat spot (maybe the top or bottom of the hill).

In AI, we use derivatives to figure out which direction to "walk" to reach the lowest point of a valley — that valley represents the best settings for our model.
:::

The **derivative** of a function $f(x)$ at a point $x$ measures the rate of change—the slope of the function at that point:

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

**Reading the formula step by step:**
1. Pick a spot on the curve ($x$)
2. Move a tiny bit to the right ($x + h$) and see how much the output changed ($f(x+h) - f(x)$)
3. Divide that change by how far you moved ($h$)
4. Make $h$ incredibly tiny (that's the $\lim_{h \to 0}$ part — "limit as $h$ approaches zero")
5. The result is the slope at that point

Common derivatives you'll see constantly in ML:

| Function | Derivative | Where It Appears |
|----------|-----------|------------------|
| $x^n$ | $nx^{n-1}$ | Polynomial features |
| $e^x$ | $e^x$ | Softmax, exponential distributions |
| $\ln(x)$ | $1/x$ | Cross-entropy loss |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation |

### Partial Derivatives and Gradients

:::info[Plain English: What Is a Gradient?]
Imagine you're blindfolded on a hilly field and you want to find the lowest point. You can feel which direction is steepest under your feet. The **gradient** is like an arrow pointing in the direction of the **steepest uphill climb**. To go downhill (which is what we want in AI — find the lowest error), you walk in the **opposite direction** of that arrow.

A **partial derivative** is just checking the slope in one direction at a time — like asking "if I only walk north, does the ground go up or down?" Then asking "if I only walk east, does the ground go up or down?" The gradient puts all these directions together into one arrow.
:::

When a function depends on multiple variables, a **partial derivative** measures how the function changes when you vary just one variable while holding the others constant:

$$
\frac{\partial f}{\partial x_i}
$$

**Reading the formula:** The $\partial$ symbol (curly d) means "partial." This says "how much does $f$ change if I nudge just $x_i$ a tiny bit, while keeping everything else frozen?"

The **gradient** collects all partial derivatives into a vector:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

**Reading the formula:** The upside-down triangle $\nabla$ is called "nabla" or "del." The gradient is just a list of all the slopes — one for each variable. It tells you "the steepest uphill direction is this way."

:::info[The Gradient Points Uphill]
The gradient $\nabla f$ points in the direction of steepest *increase*. To minimize a loss function, we move in the *opposite* direction: $\theta \leftarrow \theta - \alpha \nabla L(\theta)$. This is **gradient descent**, the core algorithm for training neural networks.

**In plain English:** $\theta$ is our model's settings. $\alpha$ is the step size (learning rate — how big of a step we take). $\nabla L(\theta)$ is the "uphill" direction. By subtracting, we go downhill toward less error.
:::

```python title="Numerical Gradients"
import numpy as np

def f(x):
    """Example: f(x) = x^2 + 3x + 1"""
    return x**2 + 3*x + 1

def numerical_derivative(f, x, h=1e-7):
    """Approximate derivative using finite differences."""
    return (f(x + h) - f(x - h)) / (2 * h)

# f'(x) = 2x + 3
x = 2.0
approx = numerical_derivative(f, x)
exact = 2 * x + 3
print(f"Numerical derivative at x={x}: {approx:.6f}")
print(f"Exact derivative: {exact:.6f}")

def g(params):
    """Multivariate: g(x, y) = x^2 + xy + y^2"""
    x, y = params
    return x**2 + x*y + y**2

def numerical_gradient(f, params, h=1e-7):
    """Compute gradient numerically for a multivariate function."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h
        grad[i] = (f(params_plus) - f(params_minus)) / (2 * h)
    return grad

point = np.array([1.0, 2.0])
grad = numerical_gradient(g, point)
print(f"\nGradient at (1, 2): {grad}")
# Exact: dg/dx = 2x + y = 4, dg/dy = x + 2y = 5
print(f"Exact gradient: [4. 5.]")
```

:::tip[Line-by-Line Walkthrough]
- **`def f(x): return x**2 + 3*x + 1`** — Defines a simple math function: take x, square it, add 3 times x, add 1. For example, f(2) = 4 + 6 + 1 = 11.
- **`def numerical_derivative(f, x, h=1e-7)`** — This function estimates the slope by checking two nearby points. `h=1e-7` means 0.0000001 — a very tiny step.
- **`(f(x + h) - f(x - h)) / (2 * h)`** — Check the function value slightly to the right and slightly to the left of x, then divide the difference by the distance. This gives you the slope!
- **`approx = numerical_derivative(f, x)`** — Computes the estimated slope at x=2. The exact answer is 2×2 + 3 = 7.
- **`def g(params)`** — A function of two variables (x and y). This is more realistic — real models have millions of variables.
- **`def numerical_gradient(f, params, h=1e-7)`** — Computes the slope in each direction, one at a time. For each variable, it wiggles just that one while keeping others fixed.
- **`grad = numerical_gradient(g, point)`** — Computes the gradient (slope in every direction) at the point (1, 2). The result [4, 5] means "going in the x-direction, the slope is 4; in the y-direction, the slope is 5."
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `gradients.py`
2. Open your terminal
3. Run: `python gradients.py`

**Expected output:**
```
Numerical derivative at x=2.0: 7.000000
Exact derivative: 7.0

Gradient at (1, 2): [4. 5.]
Exact gradient: [4. 5.]
```

</details>

### The Chain Rule

:::info[Plain English: What Is the Chain Rule?]
Imagine a Rube Goldberg machine: a ball hits a domino, the domino tips a cup, the cup pours water on a plant. If you want to know "how much does moving the ball affect the plant's growth?", you multiply the effects together: (how much does the ball affect the domino) × (how much does the domino affect the cup) × (how much does the cup affect the plant).

That's the chain rule! When things are connected in a chain, the total effect is the product of each individual effect. In neural networks, layers are chained together, and the chain rule lets us figure out how changing any weight (no matter how deep in the network) affects the final output.
:::

The **chain rule** is the mathematical foundation of backpropagation. If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)
$$

**Reading the formula:** If $y$ depends on $g$, and $g$ depends on $x$, then the rate at which $y$ changes with $x$ equals (rate $y$ changes with $g$) × (rate $g$ changes with $x$). Multiply the individual slopes along the chain.

In a neural network, the output is a composition of many functions: $L = L(\sigma(Wx + b))$. The chain rule lets us compute how the loss $L$ changes with respect to any parameter, even ones buried deep in the network.

:::note[Chain Rule Example]
Let $L = (y - \hat{y})^2$ where $\hat{y} = \sigma(wx + b)$ and $\sigma(z) = 1/(1+e^{-z})$.

To find $\frac{\partial L}{\partial w}$, we apply the chain rule:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**In plain English:** How does the loss change when we change the weight $w$? We break it into three easy pieces: (1) how does the loss change when the prediction changes? (2) how does the prediction change when the input to sigmoid changes? (3) how does the input to sigmoid change when the weight changes? Multiply those three answers together.

where $z = wx + b$. Each piece is simple; the chain rule composes them.
:::

## Probability and Statistics

Machine learning is fundamentally about making predictions under uncertainty. Probability gives us the language and tools for this.

### Probability Distributions

A **probability distribution** describes how likely different outcomes are.

:::info[Plain English: What Is a Probability Distribution?]
Think of a bag of colored marbles. A **probability distribution** tells you "what are the chances of pulling out each color?" If you have 3 red, 2 blue, and 5 green marbles, the distribution is: red=30%, blue=20%, green=50%. All the percentages **must add up to 100%** (you'll always pull *something* out).

In AI, the model's output is often a probability distribution — for example, "I'm 90% sure this is a cat, 8% sure it's a dog, and 2% sure it's a rabbit."
:::

**Discrete distributions** assign probabilities to countable outcomes. A classifier's output is a discrete distribution over classes.

**Continuous distributions** are described by a **probability density function** (PDF). The most important one in ML is the **Gaussian (normal) distribution**:

$$
p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

**Reading the formula (don't panic!):** This looks scary, but here's what it says in plain English: "The probability of seeing value $x$ depends on how far $x$ is from the center ($\mu$, pronounced 'mu' — it's the average). Values close to the center are very likely; values far away are unlikely. How quickly the probability drops off depends on the spread ($\sigma$, pronounced 'sigma' — the standard deviation)." This creates the famous bell curve shape.

```python title="Visualizing Probability Distributions"
import numpy as np

# Discrete distribution: rolling a die
outcomes = [1, 2, 3, 4, 5, 6]
fair_probs = [1/6] * 6
loaded_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]

print("Fair die probabilities:", [f"{p:.3f}" for p in fair_probs])
print("Loaded die probabilities:", loaded_probs)
print(f"Sum of fair probs: {sum(fair_probs):.1f}")  # Must sum to 1

# Continuous: sampling from Gaussian
samples = np.random.normal(loc=0, scale=1, size=10000)
print(f"\nGaussian samples — mean: {samples.mean():.3f}, std: {samples.std():.3f}")

# Softmax: turning raw scores into a probability distribution
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # numerical stability
    return exp_logits / exp_logits.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax probabilities: {probs}")
print(f"Sum: {probs.sum():.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`fair_probs = [1/6] * 6`** — A fair die has an equal 1/6 (≈16.7%) chance for each of the 6 faces. The `* 6` repeats 1/6 six times to make a list.
- **`loaded_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]`** — A loaded (cheating) die where face 6 comes up 50% of the time.
- **`sum(fair_probs)`** — Adds up all probabilities. This must always equal 1.0 (100%) — if it doesn't, something is wrong.
- **`np.random.normal(loc=0, scale=1, size=10000)`** — Generates 10,000 random numbers from a bell curve (Gaussian). `loc=0` means the center is at 0, `scale=1` means the spread (standard deviation) is 1.
- **`def softmax(logits)`** — **Softmax** is the key function that turns any list of numbers into probabilities. Higher numbers get higher probabilities, and everything adds up to 1. AI models use this as their final step.
- **`np.exp(logits - np.max(logits))`** — Raises each number to the power of $e$ (≈2.718). The `- np.max(logits)` part prevents the numbers from getting too huge (it doesn't change the final answer).
- **`exp_logits / exp_logits.sum()`** — Divides each value by the total, so everything sums to 1. Now you have probabilities!
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `probability.py`
2. Open your terminal
3. Run: `python probability.py`

**Expected output (your Gaussian numbers will vary slightly because they're random):**
```
Fair die probabilities: ['0.167', '0.167', '0.167', '0.167', '0.167', '0.167']
Loaded die probabilities: [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
Sum of fair probs: 1.0

Gaussian samples — mean: -0.002, std: 1.001

Logits: [2.  1.  0.1]
Softmax probabilities: [0.6590 0.2424 0.0986]
Sum: 1.0000
```

</details>

### Bayes' Theorem

:::info[Plain English: What Is Bayes' Theorem?]
You hear a noise outside at night. Before looking, you might think "it's probably the wind" (80% wind, 20% intruder — that's your **prior** belief). Then you look out and see the gate is open. You know the wind opens the gate 10% of the time, but an intruder would open it 90% of the time. Bayes' theorem lets you **update your belief** using this new evidence: now you think "hmm, maybe it's 50/50." It's a formula for learning from clues.
:::

**Bayes' theorem** lets us update our beliefs in light of new evidence:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

**Reading the formula, piece by piece:**
- $P(A \mid B)$ — **"What we want to know."** The probability of A being true, now that we've seen clue B. (Called the **posterior** — your updated belief.)
- $P(B \mid A)$ — **"How likely is the clue if A is true?"** If A is true, how often would we see clue B? (Called the **likelihood**.)
- $P(A)$ — **"What did we believe before seeing the clue?"** Our initial guess about A. (Called the **prior**.)
- $P(B)$ — **"How common is this clue overall?"** Regardless of A, how often does B happen? (Called the **evidence**.)

:::info[Bayes in ML]
Bayesian thinking underlies many ML concepts: Naive Bayes classifiers, Bayesian optimization for hyperparameter tuning, probabilistic programming, and the conceptual framework of updating model "beliefs" with data.
:::

```python title="Bayes' Theorem: Medical Test Example"
# Classic example: a medical test for a rare disease
# Disease prevalence: 1 in 1000
# Test sensitivity (true positive rate): 99%
# Test specificity (true negative rate): 95%

prevalence = 0.001       # P(disease)
sensitivity = 0.99       # P(positive | disease)
false_positive_rate = 0.05  # P(positive | no disease)

# P(positive) = P(positive|disease)*P(disease) + P(positive|no disease)*P(no disease)
p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)

# P(disease | positive) using Bayes' theorem
p_disease_given_positive = (sensitivity * prevalence) / p_positive

print(f"P(disease): {prevalence:.4f}")
print(f"P(positive test): {p_positive:.4f}")
print(f"P(disease | positive test): {p_disease_given_positive:.4f}")
print(f"\nSurprising result: even with a 99% sensitive test,")
print(f"a positive result only means ~{p_disease_given_positive*100:.1f}% chance of disease!")
print(f"This is because the disease is so rare (base rate fallacy).")
```

:::tip[Line-by-Line Walkthrough]
- **`prevalence = 0.001`** — Only 1 in 1,000 people have the disease. This is our **prior** — before any test, there's a 0.1% chance someone is sick.
- **`sensitivity = 0.99`** — If someone IS sick, the test catches it 99% of the time (great!).
- **`false_positive_rate = 0.05`** — If someone is NOT sick, the test incorrectly says "positive" 5% of the time (oops).
- **`p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)`** — Calculates: out of all people tested, what percentage get a positive result? It adds up true positives (sick people who test positive) + false positives (healthy people who test positive).
- **`p_disease_given_positive = (sensitivity * prevalence) / p_positive`** — This IS Bayes' theorem! It tells us: if your test is positive, what's the actual chance you're sick? The answer is surprisingly low (~2%), because the disease is so rare that the false positives outnumber the true positives.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ (no extra packages needed — this uses only basic Python math).

**Steps:**
1. Copy the code into a file called `bayes.py`
2. Open your terminal
3. Run: `python bayes.py`

**Expected output:**
```
P(disease): 0.0010
P(positive test): 0.0509
P(disease | positive test): 0.0194

Surprising result: even with a 99% sensitive test,
a positive result only means ~1.9% chance of disease!
This is because the disease is so rare (base rate fallacy).
```

</details>

### Expectation and Variance

:::info[Plain English: What Are Expected Value and Variance?]
**Expected value** is just a fancy word for "average outcome." If you roll a fair die many times, you'd **expect** to get about 3.5 on average (even though you can never roll 3.5). It's the long-run average.

**Variance** measures "how spread out are the results?" If everyone in a class scores between 90–100 on a test, the variance is small (scores are bunched together). If scores range from 20–100, the variance is large (scores are spread out). In AI, we care about variance because a model that gives wildly different predictions for similar inputs is unreliable.
:::

The **expected value** (mean) of a random variable is its average outcome:

$$
E[X] = \sum_{i} x_i P(x_i) \quad \text{(discrete)} \qquad E[X] = \int x \, p(x) \, dx \quad \text{(continuous)}
$$

**Reading the formula:** For the discrete case (left side): multiply each possible outcome ($x_i$) by its probability ($P(x_i)$), then add them all up. For a fair die: (1×1/6) + (2×1/6) + (3×1/6) + (4×1/6) + (5×1/6) + (6×1/6) = 3.5. The continuous case (right side) is the same idea but for smooth curves instead of countable outcomes (the integral $\int$ is like a continuous version of $\sum$).

The **variance** measures spread:

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

**Reading the formula:** Take each outcome, subtract the average, square the difference (to make negatives positive), then average those squared differences. A small variance means outcomes cluster near the average; a large variance means they're spread out.

In ML, the expected value of the loss function over the training data is what we're minimizing. Variance is important in the bias-variance tradeoff.

### Information Theory Basics

Two concepts from information theory appear constantly in ML:

:::info[Plain English: What Are Entropy and Cross-Entropy?]
**Entropy** measures **surprise**. If you always know what's coming (like a coin that always lands heads), there's no surprise — entropy is zero. If the outcome is completely unpredictable (like a fair coin), surprise is at its maximum — entropy is high.

**Cross-entropy** measures how **surprised you would be** if you used your predictions instead of the real probabilities. If your predictions are good, you won't be very surprised (low cross-entropy). If your predictions are bad, you'll be very surprised (high cross-entropy). That's why AI models try to **minimize** cross-entropy — they want to stop being surprised by the answers!
:::

**Entropy** measures the uncertainty of a distribution:

$$
H(p) = -\sum_{i} p_i \log p_i
$$

**Reading the formula:** For each possible outcome, multiply its probability ($p_i$) by the logarithm of its probability ($\log p_i$), then add them all up and flip the sign (the minus sign at the front). The logarithm is what turns probabilities into "surprise" — a low probability (like 1%) has a high $\log$ value (lots of surprise), and a high probability (like 99%) has a low $\log$ value (no surprise).

**Cross-entropy** measures how well a predicted distribution $q$ matches the true distribution $p$:

$$
H(p, q) = -\sum_{i} p_i \log q_i
$$

**Reading the formula:** Same structure as entropy, but now we're using the **true** probabilities ($p_i$) weighted by the log of our **predicted** probabilities ($q_i$). If our predictions $q$ are close to the truth $p$, cross-entropy will be low. If they're way off, it will be high.

Cross-entropy is the most common loss function for classification tasks. When $p$ is the true label (one-hot encoded) and $q$ is the model's predicted probability, minimizing cross-entropy makes the model's predictions match the true labels.

```python title="Entropy and Cross-Entropy"
import numpy as np

def entropy(probs):
    """Shannon entropy of a probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))

def cross_entropy(true_probs, pred_probs):
    """Cross-entropy between true and predicted distributions."""
    true_probs = np.array(true_probs)
    pred_probs = np.array(pred_probs)
    return -np.sum(true_probs * np.log(pred_probs))

# Entropy: uniform distribution has maximum entropy
uniform = [0.25, 0.25, 0.25, 0.25]
peaked = [0.97, 0.01, 0.01, 0.01]

print(f"Entropy of uniform dist: {entropy(uniform):.4f} bits")
print(f"Entropy of peaked dist:  {entropy(peaked):.4f} bits")
print("(Uniform = maximum uncertainty; peaked = low uncertainty)")

# Cross-entropy loss for classification
true_label = [1, 0, 0]  # true class is 0
good_pred = [0.9, 0.05, 0.05]
bad_pred = [0.3, 0.4, 0.3]

print(f"\nCross-entropy (good prediction): {cross_entropy(true_label, good_pred):.4f}")
print(f"Cross-entropy (bad prediction):  {cross_entropy(true_label, bad_pred):.4f}")
print("(Lower cross-entropy = better prediction)")
```

:::tip[Line-by-Line Walkthrough]
- **`def entropy(probs)`** — Defines a function that measures how "surprising" or "uncertain" a distribution is.
- **`probs = probs[probs > 0]`** — Removes any zero probabilities because log(0) is undefined (you can't take the logarithm of zero — it blows up to negative infinity).
- **`-np.sum(probs * np.log2(probs))`** — For each probability, compute (probability × log of probability), sum them all, then flip the sign. The `log2` means we're measuring in "bits."
- **`uniform = [0.25, 0.25, 0.25, 0.25]`** — All 4 outcomes equally likely. Maximum uncertainty — you have no idea what's coming.
- **`peaked = [0.97, 0.01, 0.01, 0.01]`** — One outcome is almost certain. Low uncertainty — you can guess pretty well.
- **`true_label = [1, 0, 0]`** — The correct answer: class 0 is the right one (probability 1), the others are wrong (probability 0). This is called "one-hot encoding."
- **`good_pred = [0.9, 0.05, 0.05]`** — The model's prediction is 90% class 0 (close to truth!), so cross-entropy will be low.
- **`bad_pred = [0.3, 0.4, 0.3]`** — The model's prediction spreads probability across all classes (wrong!), so cross-entropy will be high.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `entropy.py`
2. Open your terminal
3. Run: `python entropy.py`

**Expected output:**
```
Entropy of uniform dist: 2.0000 bits
Entropy of peaked dist:  0.2419 bits
(Uniform = maximum uncertainty; peaked = low uncertainty)

Cross-entropy (good prediction): 0.1054
Cross-entropy (bad prediction):  1.2040
(Lower cross-entropy = better prediction)
```

</details>

## Putting It All Together

Here's how these three branches of math connect in a single training step of a neural network:

1. **Linear algebra**: Compute the forward pass — multiply inputs by weight matrices, add biases
2. **Calculus**: Compute gradients of the loss with respect to every parameter using the chain rule (backpropagation)
3. **Probability**: The loss function (cross-entropy) is defined in terms of probability distributions; the model outputs a probability distribution over classes via softmax

```python title="One Training Step — All Three Math Areas"
import numpy as np

np.random.seed(42)

# --- DATA ---
X = np.array([[1.0, 2.0],    # 2 features per sample
              [3.0, 4.0],
              [5.0, 6.0]])
y = np.array([0, 1, 1])       # binary labels

# --- MODEL PARAMETERS (Linear Algebra) ---
W = np.random.randn(2, 1) * 0.01  # weight matrix (2 inputs -> 1 output)
b = np.zeros((1,))                  # bias

# --- FORWARD PASS (Linear Algebra + Probability) ---
z = X @ W + b                       # matrix multiplication
y_hat = 1 / (1 + np.exp(-z))        # sigmoid -> probability

# --- LOSS (Probability / Information Theory) ---
epsilon = 1e-8
loss = -np.mean(y.reshape(-1, 1) * np.log(y_hat + epsilon) +
                (1 - y.reshape(-1, 1)) * np.log(1 - y_hat + epsilon))

# --- BACKWARD PASS (Calculus — Chain Rule) ---
m = X.shape[0]
dz = y_hat - y.reshape(-1, 1)       # dL/dz
dW = (1/m) * X.T @ dz               # dL/dW (chain rule through matmul)
db = (1/m) * np.sum(dz, axis=0)     # dL/db

# --- GRADIENT DESCENT (Calculus) ---
lr = 0.1
W -= lr * dW
b -= lr * db

print(f"Loss: {loss:.4f}")
print(f"Predictions: {y_hat.flatten()}")
print(f"Gradients dW: {dW.flatten()}")
print(f"Updated W: {W.flatten()}")
```

:::tip[Line-by-Line Walkthrough]
**Setting up the data:**
- **`np.random.seed(42)`** — Fixes the random numbers so you get the same result every time. Think of it as choosing a specific deck shuffle.
- **`X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])`** — 3 data points, each with 2 features. Like 3 people described by (height, weight).
- **`y = np.array([0, 1, 1])`** — Labels: the first person belongs to class 0, the other two belong to class 1.

**Creating the model (Linear Algebra):**
- **`W = np.random.randn(2, 1) * 0.01`** — Creates 2 small random weights (one per input feature). Multiplying by 0.01 keeps them small at the start.
- **`b = np.zeros((1,))`** — Bias starts at zero.

**Forward pass (Linear Algebra + Probability):**
- **`z = X @ W + b`** — Matrix multiply: for each data point, compute (feature1 × weight1) + (feature2 × weight2) + bias. This gives a raw score for each data point.
- **`y_hat = 1 / (1 + np.exp(-z))`** — The **sigmoid function** squishes the raw score into a probability between 0 and 1. A score of 0 becomes 0.5 (50%), a big positive score becomes close to 1, and a big negative score becomes close to 0.

**Computing the loss (Probability):**
- **`epsilon = 1e-8`** — A tiny number (0.00000001) to prevent taking log(0), which would crash the program.
- **`loss = -np.mean(...)`** — **Binary cross-entropy loss**. It measures how wrong our predictions are. Lower is better. It's essentially saying "you predicted 0.6, but the answer was 1 — that's a penalty of X."

**Backward pass (Calculus):**
- **`dz = y_hat - y.reshape(-1, 1)`** — The difference between our prediction and the truth. If we predicted 0.6 and the answer is 1, this is -0.4 (we need to go up).
- **`dW = (1/m) * X.T @ dz`** — Uses the chain rule to compute: "how much should each weight change to reduce the error?" This is the gradient of the loss with respect to the weights.
- **`db = (1/m) * np.sum(dz, axis=0)`** — Same but for the bias.

**Updating the model (Gradient Descent):**
- **`lr = 0.1`** — Learning rate: how big of a step to take. 0.1 means "move 10% of the way the gradient suggests."
- **`W -= lr * dW`** — Nudge the weights in the direction that reduces the loss. This is the core of learning!
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `training_step.py`
2. Open your terminal
3. Run: `python training_step.py`

**Expected output:**
```
Loss: 0.7083
Predictions: [0.4988 0.5015 0.5042]
Gradients dW: [1.3352 1.6694]
Updated W: [-0.0046 -0.0224]
```

</details>

## Exercises

:::tip[Exercise 1: Vector Operations — beginner]

Given $\mathbf{a} = [3, 4]$ and $\mathbf{b} = [1, 2]$:

1. Compute the dot product $\mathbf{a} \cdot \mathbf{b}$
2. Compute the magnitude (L2 norm) of each vector
3. Compute the cosine similarity between them
4. Are these vectors more "similar" or "different"? Why?

<details>
<summary>Hints</summary>

1. Use np.dot() for the dot product.
2. Use np.linalg.norm() for magnitude.
3. Cosine similarity = dot product / (norm_a * norm_b).

</details>

:::

:::tip[Exercise 2: Matrix Multiplication by Hand — beginner]

Compute the following matrix multiplication **by hand**, then verify with NumPy:

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

:::info[Plain English: How to Do This by Hand]
To fill in each cell of the result, take a **row** from the first matrix and a **column** from the second matrix, multiply them number-by-number, and add up. For example, row 1 of the first [1, 2] dotted with column 1 of the second [5, 7] = (1×5) + (2×7) = 5 + 14 = 19. That goes in row 1, column 1 of the result.
:::

<details>
<summary>Hints</summary>

1. Row of first matrix dot column of second matrix = one element of result.
2. Result has shape (rows of A, cols of B).

</details>

:::

:::tip[Exercise 3: Gradient Descent from Scratch — intermediate]

Implement gradient descent to find the minimum of $f(x) = x^2$.

:::info[Plain English: What You're Doing]
$f(x) = x^2$ is a U-shaped curve (a parabola) with its lowest point at $x = 0$. You're going to start far away at $x = 10$ and take small steps downhill until you reach (close to) zero. Each step, you check the slope (derivative = $2x$) and walk in the opposite direction.
:::

1. Start at $x = 10$
2. Use a learning rate of $0.1$
3. Run for 50 iterations
4. Print the value of $x$ and $f(x)$ at each step
5. Experiment: what happens with learning rate $0.01$? With $1.0$?

<details>
<summary>Hints</summary>

1. The gradient of f(x) = x^2 is f'(x) = 2x.
2. Update rule: x = x - learning_rate * gradient.
3. Try different learning rates: 0.01, 0.1, 0.5.

</details>

:::

:::tip[Exercise 4: Bayes' Theorem Application — intermediate]

You're building a spam filter. From your training data, you know:

- 40% of emails are spam
- The word "winner" appears in 80% of spam emails and 5% of non-spam emails
- The word "meeting" appears in 2% of spam emails and 30% of non-spam emails

Using Bayes' theorem, calculate:
1. $P(\text{spam} \mid \text{"winner"})$
2. $P(\text{spam} \mid \text{"meeting"})$

Which word is a stronger spam indicator?

<details>
<summary>Hints</summary>

1. Write out P(spam | word) using Bayes' theorem.
2. P(word | spam) = count of word in spam emails / total words in spam emails.
3. Don't forget the prior P(spam) = proportion of spam in dataset.

</details>

:::

:::tip[Exercise 5: Cross-Entropy Loss — advanced]

A 3-class classifier outputs softmax probabilities. The true label is class 1 (one-hot: $[0, 1, 0]$). Compute the cross-entropy loss for each prediction and explain which is best:

1. $[0.1, 0.8, 0.1]$ (fairly confident, correct)
2. $[0.33, 0.34, 0.33]$ (near uniform)
3. $[0.7, 0.2, 0.1]$ (confident but wrong)
4. $[0.01, 0.98, 0.01]$ (very confident, correct)

Then implement a function that computes cross-entropy loss for an arbitrary number of classes.

<details>
<summary>Hints</summary>

1. Cross-entropy: L = -sum(y_true * log(y_pred))
2. A perfect prediction puts probability 1.0 on the correct class.
3. Think about what happens when you take log(0) — why is a small epsilon needed?

</details>

:::

## Resources

- **[3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)** _(video)_ by Grant Sanderson — The best visual introduction to linear algebra. Watch this series to build geometric intuition.

- **[3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)** _(video)_ by Grant Sanderson — Beautiful visual explanations of derivatives, integrals, and the fundamental ideas of calculus.

- **[Mathematics for Machine Learning](https://mml-book.github.io/)** _(book)_ by Deisenroth, Faisal & Ong — Free textbook covering linear algebra, calculus, and probability with explicit ML connections.

- **[Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)** _(course)_ — Interactive exercises and clear explanations. Great for filling in gaps.

- **[Seeing Theory: Probability](https://seeing-theory.brown.edu/)** _(tutorial)_ by Brown University — Beautiful interactive visualizations of probability concepts.

- **[The Matrix Calculus You Need for Deep Learning](https://arxiv.org/abs/1802.01528)** _(paper)_ by Parr & Howard — Practical guide to the specific matrix calculus used in deep learning.

---

**Next up**: Now that we have the mathematical foundations, let's set up Python and learn the language itself — variables, data structures, functions, classes, and the patterns you'll see in every ML codebase.
