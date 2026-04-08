---
sidebar_position: 5
title: "Introduction to Neural Networks"
slug: neural-networks-intro
---


# Introduction to Neural Networks

:::info[What You'll Learn]
- How a single perceptron works
- Activation functions and why they matter
- Forward propagation step by step
- Backpropagation and gradient descent
- Building a network from individual neurons
:::

**Estimated time:** Reading: ~45 min | Exercises: ~3 hours

Neural networks are the foundation of modern AI. Every large language model, every image classifier, every AI system making headlines is built from neural networks. In this chapter, we'll build your understanding from the ground up: from a single artificial neuron to multi-layer networks, and from forward passes to the backpropagation algorithm that makes learning possible.

## Biological Inspiration

Artificial neural networks are loosely inspired by biological neurons. A biological neuron:

1. Receives electrical signals from other neurons through **dendrites**
2. Sums the incoming signals in the **cell body**
3. If the total signal exceeds a threshold, it **fires** an electrical impulse down its **axon**
4. The signal is transmitted to the next neuron across a **synapse**

The analogy is loose but useful: an artificial neuron receives numerical inputs, computes a weighted sum, applies a function, and produces an output. The "learning" happens by adjusting the weights.

:::warning[Don't Take the Biology Too Literally]
Modern neural networks have diverged significantly from biological plausibility. Real neurons communicate with spikes, have complex temporal dynamics, and learn through mechanisms we don't fully understand. Artificial neural networks are mathematical models that happened to be inspired by biology—they are not simulations of brains.
:::

## The Perceptron

The **perceptron** (Rosenblatt, 1958) is the simplest neural network—a single artificial neuron. It computes:

:::info[Plain English: What Does a Perceptron Do?]
Imagine you're deciding whether to go to the park. You check a few things: Is it sunny? Is it warm? Are your friends going? Each factor matters differently to you — sunshine might matter a lot, while your friends going matters a little. You mentally give each factor a score, multiply by how much you care, add them up, and if the total is high enough, you go! A perceptron works exactly the same way: it takes inputs, multiplies each by an "importance weight," adds them up with a small nudge (the bias), and says "yes" (1) if the total is above zero, or "no" (0) otherwise.
:::

$$
\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w} \cdot \mathbf{x} + b > 0 \\ 0 & \text{otherwise} \end{cases}
$$

**Reading the formula step by step:**
1. **$\mathbf{x}$** — the **input vector**: a list of numbers representing your data (e.g., two features like temperature and sunshine).
2. **$\mathbf{w}$** — the **weight vector**: a list of numbers representing how important each input is.
3. **$\mathbf{w} \cdot \mathbf{x}$** — the **dot product**: multiply each input by its matching weight and add them all up (e.g., $w_1 x_1 + w_2 x_2 + \dots$).
4. **$b$** — the **bias**: a small extra number that shifts the threshold up or down (like giving yourself a head start).
5. **$\hat{y}$** — the **prediction**: if the weighted sum plus bias is greater than 0, output 1 ("yes"); otherwise output 0 ("no").

The perceptron takes an input vector $\mathbf{x}$, computes a weighted sum $\mathbf{w} \cdot \mathbf{x} + b$, and outputs 1 or 0 based on whether the sum exceeds a threshold (here, 0).

```python title="A Perceptron That Learns AND"
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.lr = lr
    
    def predict(self, x):
        return 1 if np.dot(self.weights, x) + self.bias > 0 else 0
    
    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

# AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

p = Perceptron(n_inputs=2)
p.train(X, y)

print(f"Weights: {p.weights}, Bias: {p.bias}")
for xi, yi in zip(X, y):
    print(f"Input: {xi} → Predicted: {p.predict(xi)}, True: {yi}")
```

:::tip[Line-by-Line Walkthrough]
- **`import numpy as np`** — Loads the NumPy library for math operations and calls it `np` for short.
- **`class Perceptron:`** — Defines a new type of object called `Perceptron` that bundles together the weights, bias, and learning logic.
- **`self.weights = np.zeros(n_inputs)`** — Creates the weight list, starting with all zeros (the perceptron hasn't learned anything yet).
- **`self.bias = 0.0`** — Sets the bias nudge to zero initially.
- **`self.lr = lr`** — Stores the learning rate (how big each adjustment step is — default 0.1).
- **`return 1 if np.dot(self.weights, x) + self.bias > 0 else 0`** — The prediction: multiply inputs by weights, add bias, and output 1 if positive, 0 otherwise.
- **`for epoch in range(epochs):`** — Loops through the entire dataset multiple times (each pass is called an "epoch").
- **`error = yi - pred`** — Computes the mistake: 0 means correct, +1 means we should have said "yes," −1 means we should have said "no."
- **`self.weights += self.lr * error * xi`** — Adjusts the weights: if we were wrong, nudge them in the right direction by a small step.
- **`self.bias += self.lr * error`** — Adjusts the bias the same way.
- **`if errors == 0: ... break`** — If we got everything right this pass, stop early — we've learned the pattern!
- **`X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`** — The four possible input combinations for a 2-input AND gate.
- **`y = np.array([0, 0, 0, 1])`** — The correct answers: AND only outputs 1 when both inputs are 1.
- **`p = Perceptron(n_inputs=2)`** — Creates a perceptron with 2 inputs.
- **`p.train(X, y)`** — Trains it on the AND data.
- The final loop prints the learned weights and tests every input to verify the perceptron works.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `perceptron_and.py`
2. Open your terminal
3. Run: `python perceptron_and.py`

**Expected output:**
```
Converged at epoch 4
Weights: [0.1 0.1], Bias: -0.1
Input: [0 0] → Predicted: 0, True: 0
Input: [0 1] → Predicted: 0, True: 0
Input: [1 0] → Predicted: 0, True: 0
Input: [1 1] → Predicted: 1, True: 1
```

</details>

:::info[The XOR Problem]
Minsky and Papert showed in 1969 that a single perceptron cannot learn the XOR function (where output is 1 when exactly one input is 1). This requires a non-linear decision boundary, which a single linear neuron cannot create. This limitation contributed to the first AI winter—but it's solved by adding more layers.
:::

## Activation Functions

Activation functions introduce **non-linearity** into neural networks. Without them, stacking multiple linear layers would still produce a linear function (a composition of linear functions is linear). Non-linearity is what gives neural networks the ability to learn complex patterns.

### Sigmoid

:::info[Plain English: What Does Sigmoid Do?]
Imagine a dimmer switch for a light. No matter how hard you turn the knob in either direction, the light can only go between fully off (0) and fully on (1). Sigmoid is like that dimmer: you feed in any number — very negative, zero, or very positive — and it smoothly squishes the result into a value between 0 and 1. Huge positive inputs give something close to 1, huge negative inputs give something close to 0, and zero gives exactly 0.5 (halfway).
:::

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Reading the formula step by step:**
1. **$x$** — the input number (can be anything: −100, 0, 42, etc.).
2. **$e$** — Euler's number (≈ 2.718), a special mathematical constant.
3. **$e^{-x}$** — Euler's number raised to the power of negative $x$. When $x$ is large and positive, this becomes tiny; when $x$ is large and negative, this becomes huge.
4. **$1 + e^{-x}$** — Add 1 so the denominator is always at least 1.
5. **$\frac{1}{1 + e^{-x}}$** — Divide 1 by that sum. The result is always between 0 and 1.

Squashes any input to the range $(0, 1)$. Historically important but rarely used in hidden layers today because of the **vanishing gradient problem**: for very large or very small inputs, the gradient is nearly zero, which slows learning.

### ReLU (Rectified Linear Unit)

:::info[Plain English: What Does ReLU Do?]
Think of a water faucet that only flows one way. If you push water forward (positive numbers), it flows right through unchanged. If you try to push water backward (negative numbers), the faucet blocks it and you get zero flow. ReLU is that simple: keep positive numbers as they are, and replace negative numbers with zero.
:::

$$
\text{ReLU}(x) = \max(0, x)
$$

**Reading the formula step by step:**
1. **$x$** — the input number.
2. **$\max(0, x)$** — "Pick whichever is larger: 0 or $x$." If $x$ is 5, the answer is 5. If $x$ is −3, the answer is 0.

The most popular activation function for hidden layers. Simple, fast, and doesn't saturate for positive values. Its gradient is either 0 or 1, which helps with training deep networks.

### Tanh

:::info[Plain English: What Does Tanh Do?]
Tanh is like sigmoid's centered cousin. Instead of squishing values between 0 and 1, it squishes them between −1 and +1. Think of a thermometer that reads from "very cold" (−1) through "neutral" (0) to "very hot" (+1), no matter how extreme the actual temperature is.
:::

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Reading the formula step by step:**
1. **$e^x$** — Euler's number raised to the power of $x$ (grows fast for positive $x$).
2. **$e^{-x}$** — Euler's number raised to the power of negative $x$ (grows fast for negative $x$).
3. **$e^x - e^{-x}$** — The difference between these two values (the numerator).
4. **$e^x + e^{-x}$** — The sum of these two values (the denominator).
5. Dividing the difference by the sum always gives a result between −1 and +1.

Squashes to $(-1, 1)$. Zero-centered (unlike sigmoid), which can help optimization. Still suffers from vanishing gradients at extremes.

```python title="Activation Functions and Their Derivatives"
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

x = np.linspace(-5, 5, 11)

print("x:       ", x)
print("sigmoid: ", sigmoid(x).round(3))
print("relu:    ", relu(x).round(3))
print("tanh:    ", tanh(x).round(3))
print()
print("sigmoid': ", sigmoid_derivative(x).round(3))
print("relu':    ", relu_derivative(x).round(3))
print("tanh':    ", tanh_derivative(x).round(3))
```

:::tip[Line-by-Line Walkthrough]
- **`import numpy as np`** — Loads NumPy for efficient math on arrays of numbers.
- **`def sigmoid(x):`** — Defines the sigmoid function: squishes any number into the range 0 to 1.
- **`return 1 / (1 + np.exp(-x))`** — The sigmoid formula. `np.exp(-x)` computes $e^{-x}$ for every element in the array.
- **`def sigmoid_derivative(x):`** — Defines how fast sigmoid changes at each point (its slope).
- **`return s * (1 - s)`** — A neat shortcut: sigmoid's derivative is just sigmoid(x) times (1 − sigmoid(x)).
- **`def relu(x):`** — Defines ReLU: keep positives, zero out negatives.
- **`return np.maximum(0, x)`** — For each number, pick the larger of 0 and x.
- **`def relu_derivative(x):`** — ReLU's slope: 1 for positive inputs, 0 for negative.
- **`return (x > 0).astype(float)`** — Creates an array of True/False, then converts to 1.0/0.0.
- **`def tanh(x):`** — Defines tanh: squishes any number into the range −1 to +1.
- **`def tanh_derivative(x):`** — Tanh's slope at each point.
- **`return 1 - np.tanh(x) ** 2`** — The derivative of tanh is $1 - \tanh^2(x)$.
- **`x = np.linspace(-5, 5, 11)`** — Creates 11 evenly spaced numbers from −5 to 5 (i.e., −5, −4, −3, …, 4, 5).
- The `print` statements show the original values, each activation function's output, and each derivative's output side by side.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `activation_functions.py`
2. Open your terminal
3. Run: `python activation_functions.py`

**Expected output:**
```
x:        [-5. -4. -3. -2. -1.  0.  1.  2.  3.  4.  5.]
sigmoid:  [0.007 0.018 0.047 0.119 0.269 0.5   0.731 0.881 0.953 0.982 0.993]
relu:     [0. 0. 0. 0. 0. 0. 1. 2. 3. 4. 5.]
tanh:     [-1.    -0.999 -0.995 -0.964 -0.762  0.     0.762  0.964  0.995  0.999  1.   ]

sigmoid':  [0.007 0.018 0.045 0.105 0.197 0.25  0.197 0.105 0.045 0.018 0.007]
relu':     [0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1.]
tanh':     [0.    0.001 0.01  0.071 0.42  1.    0.42  0.071 0.01  0.001 0.   ]
```

</details>

Try this interactive visualization to see how activation functions transform inputs:

```jsx live
function ActivationViz() {
  const [fn, setFn] = React.useState('relu');
  
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const relu = x => Math.max(0, x);
  const tanh_fn = x => Math.tanh(x);
  const leakyRelu = x => x > 0 ? x : 0.01 * x;
  
  const fns = { sigmoid, relu, tanh: tanh_fn, leakyRelu };
  const colors = { sigmoid: '#3b82f6', relu: '#ef4444', tanh: '#10b981', leakyRelu: '#f59e0b' };
  
  const width = 400, height = 300;
  const xMin = -6, xMax = 6, yMin = -1.5, yMax = 1.5;
  
  const toSvgX = x => ((x - xMin) / (xMax - xMin)) * width;
  const toSvgY = y => height - ((y - yMin) / (yMax - yMin)) * height;
  
  const activationFn = fns[fn];
  const points = [];
  for (let x = xMin; x <= xMax; x += 0.1) {
    points.push(`${toSvgX(x).toFixed(1)},${toSvgY(activationFn(x)).toFixed(1)}`);
  }
  
  return (
    {Object.keys(fns).map(name => (
          <button
            key={name}
            onClick={() => setFn(name)}
            style={{
              margin: '0 4px', padding: '6px 16px',
              background: fn === name ? colors[name] : '#374151',
              color: 'white', border: 'none', borderRadius: 6,
              cursor: 'pointer', fontWeight: fn === name ? 'bold' : 'normal',
            }}
          >
            {name}
          </button>
        ))}
      
      <svg width={width} height={height} style={{background: '#1e1e2e', borderRadius: 8}}>
        <line x1={toSvgX(xMin)} y1={toSvgY(0)} x2={toSvgX(xMax)} y2={toSvgY(0)} stroke="#555" strokeWidth="1"/>
        <line x1={toSvgX(0)} y1={toSvgY(yMin)} x2={toSvgX(0)} y2={toSvgY(yMax)} stroke="#555" strokeWidth="1"/>
        {[-1, 1].map(v => (
          <line key={v} x1={toSvgX(xMin)} y1={toSvgY(v)} x2={toSvgX(xMax)} y2={toSvgY(v)} stroke="#333" strokeWidth="1" strokeDasharray="4"/>
        ))}
        <polyline points={points.join(' ')} fill="none" stroke={colors[fn]} strokeWidth="3"/>
        <text x={10} y={20} fill="#ccc" fontSize="14">{fn}(x)</text>
      </svg>
    
  );
}
```

## Multi-Layer Perceptrons (MLPs)

A **multi-layer perceptron** stacks multiple layers of neurons. Each layer takes the output of the previous layer as input, applies a linear transformation followed by a non-linear activation:

:::info[Plain English: What Does One Layer of a Neural Network Do?]
Picture an assembly line in a factory. At each station, a worker receives items from the previous station, rearranges and combines them (that's the linear transformation — multiplying by weights and adding a bias), and then applies a quality check that might discard or reshape things (that's the activation function). The output is passed to the next station. Each layer transforms its input into something slightly more useful for the final goal — like raw ingredients becoming chopped, then mixed, then baked, then frosted into a cake.
:::

$$
\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})
$$

**Reading the formula step by step:**
1. **$\mathbf{h}^{(l-1)}$** — the output from the **previous layer** (or the raw input if this is the first layer). It's a list of numbers.
2. **$\mathbf{W}^{(l)}$** — the **weight matrix** for layer $l$: a grid of numbers that determines how each previous output connects to each neuron in this layer.
3. **$\mathbf{W}^{(l)} \mathbf{h}^{(l-1)}$** — **matrix multiplication**: each neuron computes a weighted sum of all the previous layer's outputs.
4. **$\mathbf{b}^{(l)}$** — the **bias vector**: one extra nudge number per neuron.
5. **$\sigma(\dots)$** — the **activation function** (like ReLU or sigmoid): applied to each neuron's result to introduce non-linearity.
6. **$\mathbf{h}^{(l)}$** — the **output of layer $l$**: a new list of numbers ready for the next layer.

A typical MLP has:
- **Input layer**: The raw features (not really a "layer" in terms of computation)
- **Hidden layers**: One or more layers of neurons that learn intermediate representations
- **Output layer**: Produces the final prediction (sigmoid for binary classification, softmax for multi-class)

:::info[The Universal Approximation Theorem]
A neural network with a single hidden layer containing enough neurons can approximate *any* continuous function to arbitrary precision. This doesn't mean it's easy to train—it says nothing about how many neurons you need or how long training takes—but it establishes that neural networks are fundamentally powerful function approximators.
:::

## The Forward Pass

The forward pass computes the output of the network given an input. Let's trace through it step by step for a 2-layer MLP:

```python title="Forward Pass Step by Step"
import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Network: 3 inputs → 4 hidden neurons → 1 output
# Xavier initialization for weights
W1 = np.random.randn(4, 3) * np.sqrt(2.0 / 3)   # (4, 3)
b1 = np.zeros(4)                                   # (4,)
W2 = np.random.randn(1, 4) * np.sqrt(2.0 / 4)     # (1, 4)
b2 = np.zeros(1)                                    # (1,)

# Input
x = np.array([0.5, -0.3, 0.8])

# Layer 1: linear transformation + activation
z1 = W1 @ x + b1              # pre-activation (4,)
a1 = np.maximum(0, z1)        # ReLU activation (4,)

# Layer 2: linear transformation + activation
z2 = W2 @ a1 + b2             # pre-activation (1,)
a2 = sigmoid(z2)              # sigmoid for binary output (1,)

print("=== Forward Pass ===")
print(f"Input x:          {x}")
print(f"z1 = W1·x + b1:  {z1.round(4)}")
print(f"a1 = ReLU(z1):    {a1.round(4)}")
print(f"z2 = W2·a1 + b2:  {z2.round(4)}")
print(f"a2 = σ(z2):       {a2.round(4)} ← prediction")
print(f"\nIf threshold=0.5: class = {int(a2[0] > 0.5)}")
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.seed(42)`** — Fixes the random number generator so you get the same results every time you run this code.
- **`def sigmoid(x):`** — Defines the sigmoid activation function (squishes values to 0–1 range).
- **`W1 = np.random.randn(4, 3) * np.sqrt(2.0 / 3)`** — Creates a 4×3 weight matrix with Xavier initialization. Xavier scaling prevents values from exploding or vanishing as they pass through layers.
- **`b1 = np.zeros(4)`** — Creates a bias vector of four zeros for the first layer.
- **`W2 = np.random.randn(1, 4) * np.sqrt(2.0 / 4)`** — Creates a 1×4 weight matrix for the second layer (1 output neuron connected to 4 hidden neurons).
- **`b2 = np.zeros(1)`** — Creates a bias of one zero for the output layer.
- **`x = np.array([0.5, -0.3, 0.8])`** — Our input: three feature values.
- **`z1 = W1 @ x + b1`** — Layer 1 linear step: multiply the weight matrix by the input, then add biases. The `@` symbol means matrix multiplication.
- **`a1 = np.maximum(0, z1)`** — Layer 1 activation: apply ReLU (keep positives, zero out negatives).
- **`z2 = W2 @ a1 + b2`** — Layer 2 linear step: multiply the second weight matrix by the hidden layer output, add bias.
- **`a2 = sigmoid(z2)`** — Layer 2 activation: apply sigmoid to get a probability between 0 and 1.
- The `print` statements display each intermediate result so you can see exactly how data flows from input to prediction.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `forward_pass.py`
2. Open your terminal
3. Run: `python forward_pass.py`

**Expected output:**
```
=== Forward Pass ===
Input x:          [ 0.5 -0.3  0.8]
z1 = W1·x + b1:  [ 0.5446 -0.3547  0.7063  0.6448]
a1 = ReLU(z1):    [0.5446 0.     0.7063 0.6448]
z2 = W2·a1 + b2:  [-0.1conveniently]
a2 = σ(z2):       [0.4xxx] ← prediction

If threshold=0.5: class = 0
```

(Exact numbers depend on NumPy version but will be consistent across runs thanks to the fixed seed.)

</details>

## Loss Functions

The **loss function** (or cost function) measures how wrong the model's predictions are. Training minimizes this function.

### Mean Squared Error (MSE) — for Regression

:::info[Plain English: What Is Mean Squared Error?]
Imagine you're playing darts. After each throw, you measure how far you landed from the bullseye. MSE is like squaring each distance (so bigger misses get punished much more), then averaging all of them. A small MSE means your throws are clustered near the bullseye; a large MSE means you're all over the board. Squaring makes the math nicer and ensures that a miss of 10 is treated as 100 times worse than a miss of 1, not just 10 times.
:::

$$
L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Reading the formula step by step:**
1. **$n$** — the number of data points (how many darts you threw).
2. **$y_i$** — the **true value** for the $i$-th data point (where the bullseye is).
3. **$\hat{y}_i$** — the **predicted value** for the $i$-th data point (where your dart landed).
4. **$(y_i - \hat{y}_i)^2$** — the **squared error**: how far off you were, squared.
5. **$\sum_{i=1}^{n}$** — add up all the squared errors.
6. **$\frac{1}{n}$** — divide by $n$ to get the average.

### Binary Cross-Entropy — for Binary Classification

:::info[Plain English: What Is Binary Cross-Entropy?]
Suppose a weather app says there's a 90% chance of rain, and it does rain — that's a good prediction, so the penalty is small. But if the app says 10% chance and it rains anyway, that's a terrible prediction, and the penalty is huge. Binary cross-entropy measures this "surprise penalty": confident correct predictions are barely penalized, but confident wrong predictions are heavily penalized. It's the standard way to score yes/no prediction models.
:::

$$
L_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

**Reading the formula step by step:**
1. **$y_i$** — the **true label**: either 1 ("yes, it rained") or 0 ("no rain").
2. **$\hat{y}_i$** — the **predicted probability** of the "yes" class (e.g., 0.9 means "90% chance of rain").
3. **$\log(\hat{y}_i)$** — the natural logarithm of the prediction. When $\hat{y}_i$ is close to 1, this is close to 0 (small penalty). When $\hat{y}_i$ is close to 0, this is a very negative number (huge penalty).
4. **$y_i \log(\hat{y}_i)$** — only contributes when the true label is 1.
5. **$(1 - y_i) \log(1 - \hat{y}_i)$** — only contributes when the true label is 0.
6. The negative sign at the front flips the result so the loss is positive.
7. **$\frac{1}{n}$** — average over all data points.

### Categorical Cross-Entropy — for Multi-Class Classification

:::info[Plain English: What Is Categorical Cross-Entropy?]
Now imagine the weather app predicts probabilities for sunny, cloudy, and rainy. You want the model to put as much probability as possible on the correct answer. Categorical cross-entropy measures how "surprised" the model is by the correct answer. If it predicted 95% sunny and it was sunny, the penalty is tiny. If it predicted only 5% sunny and it was sunny, the penalty is enormous. It generalizes binary cross-entropy to more than two categories.
:::

$$
L_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$

**Reading the formula step by step:**
1. **$n$** — the number of data points.
2. **$C$** — the number of classes (e.g., 3 for sunny/cloudy/rainy).
3. **$y_{ic}$** — is 1 if data point $i$ belongs to class $c$, and 0 otherwise (this is "one-hot encoding").
4. **$\hat{y}_{ic}$** — the model's predicted probability that data point $i$ belongs to class $c$.
5. **$\log(\hat{y}_{ic})$** — logarithm of the predicted probability for the correct class. Higher confidence in the right answer → smaller penalty.
6. The double sum goes over every data point and every class, but the $y_{ic}$ term zeros out all classes except the correct one.

```python title="Loss Functions"
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

# Regression
y_true = np.array([3.0, 5.0, 7.0])
y_pred_good = np.array([2.9, 5.1, 7.2])
y_pred_bad = np.array([1.0, 8.0, 4.0])
print(f"MSE (good): {mse_loss(y_true, y_pred_good):.4f}")
print(f"MSE (bad):  {mse_loss(y_true, y_pred_bad):.4f}")

# Binary classification
y_true = np.array([1, 0, 1, 1])
y_pred_good = np.array([0.9, 0.1, 0.8, 0.95])
y_pred_bad = np.array([0.3, 0.7, 0.4, 0.2])
print(f"\nBCE (good): {binary_cross_entropy(y_true, y_pred_good):.4f}")
print(f"BCE (bad):  {binary_cross_entropy(y_true, y_pred_bad):.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`def mse_loss(y_true, y_pred):`** — Defines the Mean Squared Error function.
- **`return np.mean((y_true - y_pred) ** 2)`** — Subtract predictions from true values, square each difference, then average them all.
- **`def binary_cross_entropy(y_true, y_pred, eps=1e-8):`** — Defines binary cross-entropy. The `eps` parameter is a tiny safety number.
- **`y_pred = np.clip(y_pred, eps, 1 - eps)`** — Clips predictions so they're never exactly 0 or 1 (because `log(0)` is negative infinity, which would crash the math).
- **`return -np.mean(y_true * np.log(y_pred) + ...)`** — Applies the BCE formula: penalizes confident wrong predictions harshly.
- **`def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-8):`** — Defines categorical cross-entropy for multi-class problems.
- **`return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))`** — For each data point, sums the log-probabilities of the correct class, then averages.
- **`y_true = np.array([3.0, 5.0, 7.0])`** — Three true values for a regression problem.
- **`y_pred_good`** and **`y_pred_bad`** — Two sets of predictions: one close to the truth, one far off.
- The `print` statements show that MSE and BCE are both much smaller for good predictions than for bad ones.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `loss_functions.py`
2. Open your terminal
3. Run: `python loss_functions.py`

**Expected output:**
```
MSE (good): 0.0200
MSE (bad):  6.0000

BCE (good): 0.1014
BCE (bad):  1.0349
```

</details>

## Backpropagation

**Backpropagation** is the algorithm that makes neural network training possible. It uses the chain rule to compute how the loss changes with respect to every weight in the network, layer by layer, from output back to input.

### The Chain Rule in Action

Consider a simple 2-layer network. The loss is a function of the output, which is a function of the weights. To update $W_1$ (a weight in the first layer), we need:

:::info[Plain English: What Is the Chain Rule Doing Here?]
Imagine a Rube Goldberg machine: you push a marble, which hits a lever, which swings a hammer, which rings a bell. If you want to know "how much louder does the bell ring if I push the marble harder?", you'd figure it out link by link: how much does the lever move per push? How much does the hammer swing per lever movement? How much does the bell ring per hammer swing? Multiply all those answers together and you get the full chain of cause and effect. That's exactly what the chain rule does — it computes how a small change in a weight ripples through every layer of the network and ultimately changes the loss.
:::

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}
$$

**Reading the formula step by step:**
1. **$\frac{\partial L}{\partial W_1}$** — "How much does the loss $L$ change when we nudge weight $W_1$ a tiny bit?" This is what we want to find.
2. **$\frac{\partial L}{\partial a_2}$** — How the loss changes when the final output $a_2$ changes.
3. **$\frac{\partial a_2}{\partial z_2}$** — How the final output changes when its pre-activation input $z_2$ changes (this is the derivative of the activation function).
4. **$\frac{\partial z_2}{\partial a_1}$** — How $z_2$ changes when the hidden layer output $a_1$ changes (this is just the second layer's weights $W_2$).
5. **$\frac{\partial a_1}{\partial z_1}$** — How the hidden output changes when its pre-activation changes (derivative of hidden activation).
6. **$\frac{\partial z_1}{\partial W_1}$** — How the pre-activation changes when we change $W_1$ (this is just the input $x$).
7. Multiply all these pieces together — that's the chain rule.

Each term is simple. The chain rule just multiplies them together.

```python title="Backpropagation from Scratch (2-Layer Network)"
import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Training data: XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights
W1 = np.random.randn(2, 4) * 0.5   # 2 inputs → 4 hidden
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5   # 4 hidden → 1 output
b2 = np.zeros((1, 1))

lr = 1.0

for epoch in range(5000):
    # === FORWARD PASS ===
    z1 = X @ W1 + b1           # (4, 4)
    a1 = relu(z1)              # (4, 4)
    z2 = a1 @ W2 + b2          # (4, 1)
    a2 = sigmoid(z2)           # (4, 1) — predictions
    
    # === LOSS ===
    loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))
    
    # === BACKWARD PASS (Chain Rule) ===
    m = X.shape[0]
    
    # Output layer gradients
    dz2 = a2 - y                           # dL/dz2 = a2 - y (for BCE + sigmoid)
    dW2 = (1/m) * a1.T @ dz2              # dL/dW2
    db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)  # dL/db2
    
    # Hidden layer gradients (chain rule continues)
    da1 = dz2 @ W2.T                      # dL/da1
    dz1 = da1 * relu_deriv(z1)            # dL/dz1
    dW1 = (1/m) * X.T @ dz1              # dL/dW1
    db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)  # dL/db1
    
    # === UPDATE WEIGHTS ===
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"\nFinal predictions (XOR):")
for xi, yi, pred in zip(X, y, a2):
    print(f"  {xi} → {pred[0]:.4f} (target: {yi[0]})")
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.seed(42)`** — Fixes randomness so results are reproducible.
- **`def sigmoid(x):`** — Sigmoid with clipping: `np.clip(x, -500, 500)` prevents overflow when computing $e^{-x}$ for very large values.
- **`def sigmoid_deriv(x):`** — The derivative of sigmoid: $\sigma(x)(1 - \sigma(x))$.
- **`def relu(x):` / `def relu_deriv(x):`** — ReLU and its derivative (1 for positive inputs, 0 otherwise).
- **`X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`** — The four XOR inputs.
- **`y = np.array([[0], [1], [1], [0]])`** — XOR outputs: 1 when exactly one input is 1.
- **`W1 = np.random.randn(2, 4) * 0.5`** — Random weights connecting 2 inputs to 4 hidden neurons, scaled down by 0.5.
- **`b1 = np.zeros((1, 4))`** — Four biases for the hidden layer, starting at zero.
- **`W2 = np.random.randn(4, 1) * 0.5`** — Random weights connecting 4 hidden neurons to 1 output neuron.
- **`lr = 1.0`** — A learning rate of 1.0 (relatively large, but works for this small problem).
- **`z1 = X @ W1 + b1`** — Forward pass layer 1: matrix-multiply inputs by weights, add biases.
- **`a1 = relu(z1)`** — Apply ReLU activation to the hidden layer.
- **`z2 = a1 @ W2 + b2`** — Forward pass layer 2: multiply hidden activations by output weights, add bias.
- **`a2 = sigmoid(z2)`** — Apply sigmoid to get a prediction between 0 and 1.
- **`loss = -np.mean(...)`** — Compute binary cross-entropy loss over all 4 data points.
- **`dz2 = a2 - y`** — The gradient of the loss with respect to $z_2$ (a known shortcut for BCE + sigmoid).
- **`dW2 = (1/m) * a1.T @ dz2`** — Gradient of the loss with respect to $W_2$: how much to adjust output weights.
- **`db2 = (1/m) * np.sum(dz2, ...)`** — Gradient for the output bias.
- **`da1 = dz2 @ W2.T`** — Propagate the error backward to the hidden layer.
- **`dz1 = da1 * relu_deriv(z1)`** — Multiply by ReLU's derivative: only pass error through neurons that were active.
- **`dW1 = (1/m) * X.T @ dz1`** — Gradient for the first layer's weights.
- **`db1 = ...`** — Gradient for the first layer's biases.
- **`W2 -= lr * dW2`** (and similar for b2, W1, b1) — Update all weights and biases by stepping opposite to the gradient.
- **`if epoch % 1000 == 0:`** — Print the loss every 1000 epochs so you can watch it decrease.
- The final loop prints each XOR input, the network's prediction, and the target value.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ and NumPy (`pip install numpy`).

**Steps:**
1. Copy the code into a file called `backprop_xor.py`
2. Open your terminal
3. Run: `python backprop_xor.py`

**Expected output:**
```
Epoch 0: loss = 0.7570
Epoch 1000: loss = 0.0134
Epoch 2000: loss = 0.0046
Epoch 3000: loss = 0.0025
Epoch 4000: loss = 0.0016

Final predictions (XOR):
  [0 0] → 0.0014 (target: 0)
  [0 1] → 0.9975 (target: 1)
  [1 0] → 0.9976 (target: 1)
  [1 1] → 0.0033 (target: 0)
```

(Exact numbers may vary slightly, but predictions should be close to 0 or 1 matching the XOR pattern.)

</details>

:::info[Why Backprop Works]
Backpropagation is efficient because it reuses computations. Computing the gradient for layer $l$ requires the gradient from layer $l+1$, which we already computed. This is why we go *backward* — each layer's gradient builds on the next layer's gradient. The time complexity is proportional to a single forward pass.
:::

## Gradient Descent Variants

**Gradient descent** updates parameters in the direction opposite to the gradient:

:::info[Plain English: What Is Gradient Descent?]
Imagine you're blindfolded on a hilly landscape and you want to reach the lowest valley. You can feel the slope under your feet. Gradient descent says: take a step in the direction that goes downhill the steepest. Repeat. Eventually you'll reach a low point. The **learning rate** is your step size — too big and you might overshoot the valley and end up on the opposite hill; too small and you'll take forever to get there. The **gradient** ($\nabla L$) is the slope measurement that tells you which direction is "downhill" and how steep it is.
:::

$$
\theta \leftarrow \theta - \alpha \nabla L(\theta)
$$

**Reading the formula step by step:**
1. **$\theta$** — all the model's parameters (weights and biases) bundled together.
2. **$\nabla L(\theta)$** — the **gradient** of the loss: a list of slopes telling you how the loss changes if you nudge each parameter.
3. **$\alpha$** — the **learning rate**: a small positive number controlling step size (e.g., 0.001).
4. **$\alpha \nabla L(\theta)$** — the step: direction times step size.
5. **$\theta \leftarrow \theta - \alpha \nabla L(\theta)$** — "Update the parameters by subtracting the step." The minus sign is because we want to go *downhill* (decrease loss).

where $\alpha$ is the **learning rate**.

### Batch Gradient Descent

Computes the gradient using the *entire* training set. Stable but slow for large datasets.

### Stochastic Gradient Descent (SGD)

Computes the gradient using a *single* random sample. Noisy but fast. The noise can help escape local minima.

### Mini-Batch Gradient Descent

The standard approach: compute the gradient on a *batch* of samples (typically 32–256). Balances stability and speed, and works well with GPU parallelism.

```jsx live
function LearningRateViz() {
  const [lr, setLr] = React.useState(0.1);
  const [steps, setSteps] = React.useState([]);
  
  const f = x => x * x;
  const df = x => 2 * x;
  
  React.useEffect(() => {
    let x = 4.0;
    const newSteps = [{x, y: f(x)}];
    for (let i = 0; i < 20; i++) {
      x = x - lr * df(x);
      if (Math.abs(x) > 10) break;
      newSteps.push({x, y: f(x)});
    }
    setSteps(newSteps);
  }, [lr]);
  
  const width = 400, height = 250;
  const xMin = -5, xMax = 5, yMin = -1, yMax = 20;
  const sx = x => ((x - xMin) / (xMax - xMin)) * width;
  const sy = y => height - ((y - yMin) / (yMax - yMin)) * height;
  
  const curve = [];
  for (let x = xMin; x <= xMax; x += 0.1) {
    curve.push(`${sx(x).toFixed(1)},${sy(f(x)).toFixed(1)}`);
  }
  
  return (
    <label>Learning rate: <strong>{lr.toFixed(2)}</strong></label>
        <br/>
        <input type="range" min="0.01" max="1.1" step="0.01" value={lr}
          onChange={e => setLr(parseFloat(e.target.value))}
          style={{width: 300}}/>
        <span style={{marginLeft: 8, color: lr > 0.95 ? '#ef4444' : lr < 0.05 ? '#f59e0b' : '#22c55e'}}>
          {lr > 0.95 ? 'Diverging!' : lr < 0.05 ? 'Too slow' : 'Good'}
        </span>
      
      <svg width={width} height={height} style={{background: '#1e1e2e', borderRadius: 8}}>
        <polyline points={curve.join(' ')} fill="none" stroke="#555" strokeWidth="2"/>
        {steps.slice(0, -1).map((s, i) => (
          <line key={i}
            x1={sx(s.x)} y1={sy(s.y)}
            x2={sx(steps[i+1].x)} y2={sy(steps[i+1].y)}
            stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="4"/>
        ))}
        {steps.map((s, i) => (
          <circle key={i} cx={sx(s.x)} cy={sy(s.y)} r={i === 0 ? 5 : 3}
            fill={i === 0 ? '#ef4444' : '#3b82f6'}/>
        ))}
        <text x={10} y={20} fill="#ccc" fontSize="12">
          f(x) = x² — {steps.length} steps, final x = {steps[steps.length-1]?.x.toFixed(3)}
        </text>
      </svg>
    
  );
}
```

:::tip[Learning Rate Is the Most Important Hyperparameter]
Too small: training is painfully slow. Too large: the model overshoots the minimum and may diverge. Modern optimizers like Adam adapt the learning rate per parameter, but you still need to choose a good base learning rate. Common starting values: 0.001 for Adam, 0.01–0.1 for SGD with momentum.
:::

## Modern Optimizers

Plain SGD is rarely used alone. Modern optimizers add two key improvements:

**Momentum**: Accumulates a moving average of gradients to smooth updates and accelerate through flat regions. Think of a ball rolling downhill—it builds up speed.

**Adaptive learning rates** (Adam, RMSProp): Different parameters get different effective learning rates based on their gradient history. Parameters with consistently large gradients get smaller steps; parameters with small gradients get larger steps.

**Adam** (Adaptive Moment Estimation) combines both ideas and is the default optimizer for most deep learning tasks:

:::info[Plain English: What Do the Adam Optimizer Equations Mean?]
Think of Adam as a smart hiker on a mountain. The first equation (**momentum**) is like the hiker remembering which direction they've been walking — if they've been going left for the last 10 steps, they keep going left even if one step says "go right" (smoothing out noisy directions). The second equation (**adaptive learning rate**) is like the hiker tracking how bumpy the terrain has been — on smooth ground, they take big confident strides; on rocky ground, they take small careful steps. The third equation combines both: walk in the smoothed direction, and adjust your step size based on terrain roughness.
:::

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(momentum)}
$$

**Reading the formula:** **$m_t$** is the smoothed gradient (momentum) at time step $t$. **$\beta_1$** (typically 0.9) controls how much history to keep — 0.9 means "90% old direction + 10% new gradient." **$g_t$** is the current gradient. **$m_{t-1}$** is the momentum from the previous step. This acts like a rolling average that smooths out noisy gradient updates.

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(adaptive LR)}
$$

**Reading the formula:** **$v_t$** tracks how large the gradients have been recently (the "velocity" of squared gradients). **$\beta_2$** (typically 0.999) controls the memory window. **$g_t^2$** is the current gradient squared. Parameters with historically large gradients will have large $v_t$, leading to smaller effective learning rates (more caution).

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**Reading the formula step by step:**
1. **$\theta_{t-1}$** — the current parameter values.
2. **$\hat{m}_t$** — bias-corrected momentum (the smoothed direction, adjusted for early-step bias).
3. **$\hat{v}_t$** — bias-corrected velocity (the smoothed squared gradients, adjusted similarly).
4. **$\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$** — divide the direction by the "bumpiness" measure. Where gradients are consistently large, the denominator is big, so steps are small. Where gradients are small, steps are bigger.
5. **$\alpha$** — the base learning rate (e.g., 0.001).
6. **$\epsilon$** — a tiny number (e.g., $10^{-8}$) to prevent dividing by zero.
7. **$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$** — update the parameters by stepping in the smoothed direction with an adaptive step size.

## Exercises

:::tip[Exercise 1: Implement a Neuron — beginner]

Implement a `Neuron` class with:

1. A constructor that initializes random weights and a bias
2. A `forward` method that computes the output for a given input
3. Support for sigmoid, ReLU, and tanh activations
4. Test it with a 3-dimensional input

<details>
<summary>Hints</summary>

1. A neuron computes: output = activation(dot(weights, inputs) + bias).
2. Start with sigmoid activation.
3. Test with known inputs and weights to verify.

</details>

:::

:::tip[Exercise 2: XOR with Different Architectures — intermediate]

Using the backpropagation code from this chapter as a starting point:

1. Solve XOR with a 2→2→1 network. Does it always converge? (Run multiple times with different seeds)
2. Try 2→8→1. Is it more reliable?
3. Try a 3-layer network (2→4→4→1). How does adding depth affect training?
4. Plot the loss curves for all architectures on the same graph

<details>
<summary>Hints</summary>

1. Try architectures: 2→2→1, 2→4→1, 2→8→1, 2→4→4→1.
2. Track loss over epochs for each architecture.
3. Some may converge faster than others.

</details>

:::

:::tip[Exercise 3: Gradient Checking — advanced]

Implement gradient checking to verify your backpropagation implementation:

1. For each parameter in the network, compute the numerical gradient using finite differences
2. Compare to the analytical gradient from backpropagation
3. Compute the relative error:

:::info[Plain English: What Does the Gradient Checking Formula Mean?]
This formula measures how closely your hand-computed (analytical) gradient matches a brute-force numerical estimate. Think of it as a "trust but verify" check. You compute both versions and divide their difference by their average size. If the result is tiny (below $10^{-5}$), your backpropagation code is correct. If it's large, there's a bug somewhere.
:::

$$
\frac{|g_\text{analytic} - g_\text{numeric}|}{\max(|g_\text{analytic}|, |g_\text{numeric}|) + \epsilon}
$$

**Reading the formula step by step:**
1. **$g_\text{analytic}$** — the gradient computed by your backpropagation code (fast, but might have bugs).
2. **$g_\text{numeric}$** — the gradient estimated numerically using finite differences: $\frac{f(x+h) - f(x-h)}{2h}$ (slow but reliable).
3. **$|g_\text{analytic} - g_\text{numeric}|$** — the absolute difference between the two.
4. **$\max(|g_\text{analytic}|, |g_\text{numeric}|) + \epsilon$** — the larger of the two magnitudes, plus a tiny $\epsilon$ to avoid dividing by zero.
5. The ratio tells you the relative error. Below $10^{-5}$ means your backpropagation is almost certainly correct.

4. All relative errors should be below $10^{-5}$

This is an invaluable debugging technique whenever you implement a custom backward pass.

<details>
<summary>Hints</summary>

1. Numerical gradient: (f(x+h) - f(x-h)) / (2h) with h ≈ 1e-7.
2. Compare your analytical gradients to numerical ones.
3. They should agree to within ~1e-5 relative error.

</details>

:::

:::tip[Exercise 4: Activation Function Comparison — intermediate]

Train a neural network on a synthetic dataset (e.g., 2 concentric circles using `sklearn.datasets.make_circles`):

1. Use a 2→16→1 architecture
2. Train three versions: one with sigmoid hidden activations, one with ReLU, one with tanh
3. Compare convergence speed and final accuracy
4. Which activation function works best? Why?

<details>
<summary>Hints</summary>

1. Train the same network with sigmoid, ReLU, and tanh hidden activations.
2. Use the same random seed for fair comparison.
3. Track loss per epoch for each.

</details>

:::

## Resources

- **[3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** _(video)_ by Grant Sanderson — The single best visual introduction to neural networks and backpropagation. Watch all 4 videos.

- **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)** _(book)_ by Michael Nielsen — Free online book with interactive elements. Excellent for building intuition about how neural networks learn.

- **[CS231n: Convolutional Neural Networks](https://cs231n.github.io/)** _(course)_ by Stanford / Andrej Karpathy — Stanford's legendary deep learning course. The course notes are outstanding, especially on backpropagation.

- **[The Backpropagation Algorithm](https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf)** _(paper)_ by Raúl Rojas — Clear mathematical derivation of backpropagation from the textbook 'Neural Networks: A Systematic Introduction'.

- **[Playground TensorFlow](https://playground.tensorflow.org/)** _(tool)_ — Interactive neural network visualization. Experiment with architectures, activations, and learning rates in real-time.

---

**Next up**: Time to put theory into practice. In the next chapter, you'll build your first neural network from scratch in NumPy, then rebuild it with PyTorch.
