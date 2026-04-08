---
sidebar_position: 4
title: "Introduction to Machine Learning"
slug: intro-to-ml
---


# Introduction to Machine Learning

:::info[What You'll Learn]
- Supervised vs. unsupervised vs. reinforcement learning
- The training/validation/test split methodology
- Bias-variance tradeoff and overfitting
- Key ML algorithms: linear regression, decision trees, SVMs
- Evaluation metrics: accuracy, precision, recall, F1
:::

**Estimated time:** Reading: ~40 min | Exercises: ~3 hours

Machine learning is the core technology behind modern AI. This chapter explains what ML is, how it differs from traditional programming, and introduces the fundamental concepts you'll use throughout your career as an AI engineer.

## What Is Machine Learning?

Traditional programming and machine learning solve problems in opposite directions:

**Traditional programming**: You write explicit rules. Given inputs and rules, the program produces outputs.

```
Input + Rules → Output
```

**Machine learning**: You provide examples (inputs and desired outputs). The algorithm *learns* the rules.

```
Input + Output → Rules (a trained model)
```

:::info[Arthur Samuel's Definition (1959)]
Machine learning is the "field of study that gives computers the ability to learn without being explicitly programmed." The key insight: instead of hand-coding rules for every scenario, we let algorithms discover patterns from data.
:::

Consider spam detection. A traditional approach might check for keywords like "FREE" or "WINNER." But spammers constantly adapt. An ML approach trains a model on thousands of labeled emails (spam/not-spam), and the model learns subtle patterns—combinations of words, sender behavior, formatting—that generalize to new spam it has never seen.

```python title="Traditional Programming vs. Machine Learning"
# Traditional approach: hand-coded rules
def is_spam_rules(email_text):
    spam_keywords = ['free', 'winner', 'click here', 'limited offer']
    text_lower = email_text.lower()
    return any(kw in text_lower for kw in spam_keywords)

# ML approach: learn from data (conceptual)
def train_spam_classifier(emails, labels):
    """
    Instead of writing rules, we:
    1. Convert emails to numerical features (e.g., word counts)
    2. Feed features + labels to a learning algorithm
    3. The algorithm finds its own decision boundary
    """
    # In reality, you'd use scikit-learn:
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.naive_bayes import MultinomialNB
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(emails)
    # model = MultinomialNB().fit(X, labels)
    # return model
    pass

# The rule-based approach breaks when spammers write "FR33" or "w1nner"
# The ML approach can learn these patterns if they appear in training data
test_emails = [
    "Congratulations! You are a WINNER!",
    "Meeting tomorrow at 3pm",
    "FR33 V1AGRA - click here!!",
]

for email in test_emails:
    print(f"Rule-based: {'SPAM' if is_spam_rules(email) else 'HAM'} | {email[:50]}")
```

:::tip[Line-by-Line Walkthrough]
- **`def is_spam_rules(email_text):`** — A traditional approach: we manually write a list of suspicious words and check if any appear in the email.
- **`spam_keywords = ['free', 'winner', 'click here', 'limited offer']`** — Our hand-coded rules. If the email contains any of these words, we flag it as spam.
- **`any(kw in text_lower for kw in spam_keywords)`** — Checks whether *any* keyword appears in the lowercased email text. Returns True (spam) or False (not spam).
- **`def train_spam_classifier(emails, labels):`** — The ML approach (conceptual). Instead of writing rules, we'd feed thousands of labeled examples to an algorithm and let it discover the patterns.
- **`test_emails = [...]`** — Three test emails. Notice "FR33 V1AGRA" — the rule-based approach misses this because it doesn't match any keyword exactly, but an ML model could learn to catch leetspeak from training data.
- **The `for` loop** — Runs each test email through the rule-based classifier and prints the result.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ (no additional packages needed).

**Steps:**
1. Save to `spam_rules.py` and run: `python spam_rules.py`

**Expected output:**
```
Rule-based: SPAM | Congratulations! You are a WINNER!
Rule-based: HAM | Meeting tomorrow at 3pm
Rule-based: HAM | FR33 V1AGRA - click here!!
```
(Notice the rule-based system misses the third spam email because it uses obfuscated text.)

</details>

## Types of Machine Learning

### Supervised Learning

In **supervised learning**, the model learns from labeled examples—input-output pairs where you provide the correct answer. The model's job is to learn a mapping from inputs to outputs that generalizes to new, unseen data.

There are two main types of supervised learning:

#### Regression

The target variable is **continuous** (a number). Examples:
- Predicting house prices from features (size, location, bedrooms)
- Forecasting tomorrow's temperature
- Estimating a user's lifetime value

The simplest regression model is **linear regression**: $\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$

```python title="Linear Regression from Scratch"
import numpy as np

np.random.seed(42)

# Generate synthetic data: y = 3x + 7 + noise
X = np.random.uniform(0, 10, 100)
y = 3 * X + 7 + np.random.normal(0, 2, 100)

# Learn w and b using the normal equation: w = (X^T X)^{-1} X^T y
X_design = np.column_stack([X, np.ones(len(X))])  # add bias column
w = np.linalg.lstsq(X_design, y, rcond=None)[0]

print(f"Learned: y = {w[0]:.2f}x + {w[1]:.2f}")
print(f"True:    y = 3.00x + 7.00")

# Predictions
y_pred = X_design @ w
residuals = y - y_pred
mse = np.mean(residuals ** 2)
print(f"Mean Squared Error: {mse:.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`X = np.random.uniform(0, 10, 100)`** — Generates 100 random x-values between 0 and 10. These are our input features.
- **`y = 3 * X + 7 + np.random.normal(0, 2, 100)`** — Creates the "true" relationship: y = 3x + 7, plus some random noise (to simulate real-world imperfection).
- **`X_design = np.column_stack([X, np.ones(len(X))])`** — Builds the "design matrix" by adding a column of 1s. The column of 1s lets the formula learn the intercept (the +7 part).
- **`w = np.linalg.lstsq(X_design, y, rcond=None)[0]`** — The normal equation: finds the best-fit line by solving the linear algebra problem directly. Returns the slope and intercept.
- **`y_pred = X_design @ w`** — Makes predictions using the learned weights. `@` is matrix multiplication.
- **`residuals = y - y_pred`** — The difference between actual and predicted values — how far off we are for each point.
- **`mse = np.mean(residuals ** 2)`** — Mean Squared Error: squares each error (so negatives don't cancel positives), then averages. Smaller is better.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `linear_regression.py` and run: `python linear_regression.py`

**Expected output:**
```
Learned: y = 2.98x + 7.15
True:    y = 3.00x + 7.00
Mean Squared Error: 3.4521
```
(The learned values will be close to the true values but not exact, because of the noise.)

</details>

#### Classification

The target variable is **categorical** (a class label). Examples:
- Email spam or not spam (binary classification)
- Digit recognition: 0–9 (multi-class classification)
- Disease diagnosis from medical images

```python title="Logistic Regression for Classification"
import numpy as np

np.random.seed(42)

# Generate 2D classification data
n_samples = 200
X_class0 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
X_class1 = np.random.randn(n_samples // 2, 2) + np.array([4, 4])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Shuffle
indices = np.random.permutation(n_samples)
X, y = X[indices], y[indices]

# Logistic regression with gradient descent
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

W = np.zeros(2)
b = 0.0
lr = 0.1

for epoch in range(100):
    z = X @ W + b
    y_hat = sigmoid(z)
    
    # Gradients
    dW = (1 / n_samples) * X.T @ (y_hat - y)
    db = (1 / n_samples) * np.sum(y_hat - y)
    
    W -= lr * dW
    b -= lr * db

# Evaluate
predictions = (sigmoid(X @ W + b) > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
print(f"Learned weights: {W}")
print(f"Learned bias: {b:.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`X_class0 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])`** — Generates 100 points centered around (1, 1) for Class 0.
- **`X_class1 = np.random.randn(n_samples // 2, 2) + np.array([4, 4])`** — Generates 100 points centered around (4, 4) for Class 1. The two clusters overlap a bit.
- **`X = np.vstack([X_class0, X_class1])`** — Stacks both groups vertically into one dataset.
- **`y = np.array([0] * 100 + [1] * 100)`** — Creates the labels: 0 for the first 100 points, 1 for the next 100.
- **`np.random.permutation(n_samples)`** — Shuffles the data so the model doesn't just learn "first half = 0, second half = 1."
- **`def sigmoid(z):`** — The S-shaped function that squashes any number into the range (0, 1). Used to turn raw scores into probabilities.
- **`z = X @ W + b`** — Forward pass: computes raw scores for each data point.
- **`dW = (1 / n_samples) * X.T @ (y_hat - y)`** — The gradient: tells us which direction to adjust the weights to reduce error. It's the average of how wrong we were, weighted by the input.
- **`W -= lr * dW`** — Gradient descent update: nudges the weights a small step in the direction that reduces error.
- **`(sigmoid(X @ W + b) > 0.5).astype(int)`** — Final predictions: if the probability is above 50%, predict class 1; otherwise, predict class 0.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `logistic_regression.py` and run: `python logistic_regression.py`

**Expected output:**
```
Accuracy: 99.50%
Learned weights: [0.98 1.02]
Learned bias: -4.8123
```
(Exact values will vary slightly due to random data generation.)

</details>

### Unsupervised Learning

In **unsupervised learning**, there are no labels. The model discovers hidden structure in the data on its own.

#### Clustering

Groups similar data points together. The most common algorithm is **K-Means**:

1. Initialize $K$ cluster centers randomly
2. Assign each data point to its nearest center
3. Update each center to the mean of its assigned points
4. Repeat until convergence

```python title="K-Means Clustering from Scratch"
import numpy as np

np.random.seed(42)

# Generate 3 clusters
cluster_1 = np.random.randn(50, 2) + [2, 2]
cluster_2 = np.random.randn(50, 2) + [8, 2]
cluster_3 = np.random.randn(50, 2) + [5, 7]
X = np.vstack([cluster_1, cluster_2, cluster_3])

def kmeans(X, k, max_iters=100):
    # Initialize centers randomly from data points
    indices = np.random.choice(len(X), k, replace=False)
    centers = X[indices].copy()
    
    for iteration in range(max_iters):
        # Assign each point to nearest center
        distances = np.sqrt(((X[:, np.newaxis] - centers[np.newaxis]) ** 2).sum(axis=2))
        labels = distances.argmin(axis=1)
        
        # Update centers
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centers, new_centers):
            print(f"Converged at iteration {iteration}")
            break
        centers = new_centers
    
    return labels, centers

labels, centers = kmeans(X, k=3)
print(f"Cluster centers:\\n{centers}")
print(f"Points per cluster: {[np.sum(labels == i) for i in range(3)]}")
```

:::tip[Line-by-Line Walkthrough]
- **`cluster_1 = np.random.randn(50, 2) + [2, 2]`** — Creates 50 random 2D points centered around (2, 2). Like scattering 50 dots near one corner of a page.
- **`X = np.vstack([cluster_1, cluster_2, cluster_3])`** — Combines all three groups into a single dataset of 150 points.
- **`def kmeans(X, k, max_iters=100):`** — Implements the K-Means algorithm from scratch.
- **`centers = X[indices].copy()`** — Picks k random data points as initial cluster centers.
- **`distances = np.sqrt(((X[:, np.newaxis] - centers[np.newaxis]) ** 2).sum(axis=2))`** — Computes the distance from every point to every center, all at once using broadcasting. Like measuring how far each student is from each of three teachers in a room.
- **`labels = distances.argmin(axis=1)`** — Assigns each point to its nearest center.
- **`new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])`** — Moves each center to the average position of its assigned points. The centers "drift" toward their clusters.
- **`if np.allclose(centers, new_centers):`** — Checks if the centers stopped moving. If so, we've converged (found the answer).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `kmeans.py` and run: `python kmeans.py`

**Expected output:**
```
Converged at iteration 5
Cluster centers:
[[ 2.05  1.97]
 [ 8.01  2.04]
 [ 5.02  6.98]]
Points per cluster: [50, 50, 50]
```

</details>

#### Dimensionality Reduction

Reduces the number of features while preserving important structure. **Principal Component Analysis (PCA)** finds the directions of maximum variance:

```python title="PCA: Dimensionality Reduction"
import numpy as np

np.random.seed(42)

# 100 samples with 5 features, but only 2 underlying dimensions
t = np.random.randn(100, 2)
A = np.random.randn(2, 5)  # maps 2D -> 5D
X = t @ A + np.random.randn(100, 5) * 0.1  # add noise

print(f"Original shape: {X.shape}")

# PCA from scratch
X_centered = X - X.mean(axis=0)
cov_matrix = (X_centered.T @ X_centered) / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort by descending eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Variance explained by each component
var_explained = eigenvalues / eigenvalues.sum()
print(f"\\nVariance explained by each component:")
for i, ve in enumerate(var_explained):
    print(f"  PC{i+1}: {ve:.4f} ({ve*100:.1f}%)")

# Project to 2D
X_reduced = X_centered @ eigenvectors[:, :2]
print(f"\\nReduced shape: {X_reduced.shape}")
print(f"Top 2 components explain {var_explained[:2].sum()*100:.1f}% of variance")
```

:::tip[Line-by-Line Walkthrough]
- **`t = np.random.randn(100, 2)`** — The "true" 2D data. Our 5D data is secretly just this 2D data stretched into 5 dimensions.
- **`A = np.random.randn(2, 5)`** — A random mapping that stretches 2D into 5D.
- **`X = t @ A + np.random.randn(100, 5) * 0.1`** — Creates the 5D dataset with a tiny bit of noise. Even though it has 5 columns, there are really only 2 meaningful directions.
- **`X_centered = X - X.mean(axis=0)`** — Centers the data by subtracting each column's mean. PCA requires centered data.
- **`cov_matrix = (X_centered.T @ X_centered) / (len(X) - 1)`** — Computes the covariance matrix: a summary of how each pair of features varies together.
- **`eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)`** — Finds the principal directions (eigenvectors) and how much variance each captures (eigenvalues). Think of it as finding the natural axes of the data's shape.
- **`var_explained = eigenvalues / eigenvalues.sum()`** — What fraction of the total information each component captures.
- **`X_reduced = X_centered @ eigenvectors[:, :2]`** — Projects the 5D data back to 2D by keeping only the top 2 principal components.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `pca.py` and run: `python pca.py`

**Expected output:**
```
Original shape: (100, 5)

Variance explained by each component:
  PC1: 0.5123 (51.2%)
  PC2: 0.4801 (48.0%)
  PC3: 0.0032 (0.3%)
  PC4: 0.0025 (0.2%)
  PC5: 0.0019 (0.2%)

Reduced shape: (100, 2)
Top 2 components explain 99.2% of variance
```
(PC1 and PC2 capture nearly all the variance because the data is truly 2-dimensional.)

</details>

:::tip[Other ML Paradigms]
Beyond supervised and unsupervised learning, you'll encounter:
- **Semi-supervised learning**: Mix of labeled and unlabeled data
- **Self-supervised learning**: The model creates its own labels from the data (how LLMs are trained—predict the next word)
- **Reinforcement learning**: An agent learns by interacting with an environment and receiving rewards (how RLHF works)
:::

## Training, Validation, and Test Splits

One of the most important concepts in ML is **generalization**—the model should perform well on data it hasn't seen during training. To measure this, we split our data:

| Split | Purpose | Typical Size |
|-------|---------|-------------|
| **Training set** | Model learns from this data | 70–80% |
| **Validation set** | Used to tune hyperparameters and detect overfitting | 10–15% |
| **Test set** | Final, unbiased evaluation of model performance | 10–15% |

:::warning[Data Leakage]
Never let your model see the test set during training or hyperparameter tuning. If test data leaks into training, your performance metrics will be optimistically biased—the model appears better than it actually is on truly new data. This is one of the most common and dangerous mistakes in ML.
:::

```python title="Splitting Data Properly"
import numpy as np

np.random.seed(42)

# Generate dataset
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Manual split
def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

print(f"Training:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Class distribution should be similar across splits
for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"{name} — class 0: {(labels==0).sum()}, class 1: {(labels==1).sum()}")
```

:::tip[Line-by-Line Walkthrough]
- **`X = np.random.randn(1000, 5)`** — Creates 1,000 data points, each with 5 features.
- **`y = (X[:, 0] + X[:, 1] > 0).astype(int)`** — Creates labels: class 1 if the sum of the first two features is positive, class 0 otherwise. A simple rule the model will try to learn.
- **`def train_val_test_split(X, y, ...)`** — A function that splits data into three non-overlapping groups.
- **`indices = np.random.permutation(n)`** — Shuffles the row indices randomly so we don't accidentally put all class-0 data in one split.
- **`train_idx = indices[:train_end]`** — Takes the first 70% of shuffled indices for training.
- **`val_idx = indices[train_end:val_end]`** — The next 15% for validation (tuning).
- **`test_idx = indices[val_end:]`** — The remaining 15% for final testing.
- **The last loop** — Verifies that the class distribution (how many 0s and 1s) is roughly balanced in each split, which is important for fair evaluation.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `data_split.py` and run: `python data_split.py`

**Expected output:**
```
Training:   700 samples (70%)
Validation: 150 samples (15%)
Test:       150 samples (15%)
Train — class 0: 345, class 1: 355
Val — class 0: 80, class 1: 70
Test — class 0: 72, class 1: 78
```

</details>

## Overfitting and Underfitting

The central challenge of ML is the **bias-variance tradeoff**:

**Underfitting** (high bias): The model is too simple to capture the patterns in the data. It performs poorly on both training and validation data.

**Overfitting** (high variance): The model memorizes the training data, including its noise and quirks. It performs well on training data but poorly on new data.

**Good fit**: The model captures the real patterns without memorizing noise. Good performance on both training and validation data.

```python title="Overfitting vs. Underfitting"
import numpy as np

np.random.seed(42)

# True function: y = sin(x)
X = np.linspace(0, 2 * np.pi, 30)
y = np.sin(X) + np.random.normal(0, 0.2, 30)

# Fit polynomials of different degrees
for degree in [1, 4, 15]:
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    train_mse = np.mean((y - y_pred) ** 2)
    
    # Evaluate on new data
    X_test = np.linspace(0, 2 * np.pi, 100)
    y_test_true = np.sin(X_test)
    y_test_pred = np.polyval(coeffs, X_test)
    test_mse = np.mean((y_test_true - y_test_pred) ** 2)
    
    status = "UNDERFITTING" if degree == 1 else "GOOD FIT" if degree == 4 else "OVERFITTING"
    print(f"Degree {degree:2d}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f} ← {status}")
```

:::tip[Line-by-Line Walkthrough]
- **`X = np.linspace(0, 2 * np.pi, 30)`** — Creates 30 evenly spaced points from 0 to 2π (one full wave of sine).
- **`y = np.sin(X) + np.random.normal(0, 0.2, 30)`** — The true pattern is a sine wave, but we add noise to simulate messy real data.
- **`for degree in [1, 4, 15]:`** — Tests three polynomial fits: a straight line (degree 1), a moderate curve (degree 4), and a very wiggly curve (degree 15).
- **`np.polyfit(X, y, degree)`** — Fits a polynomial of the given degree to the data. Higher degree = more flexibility.
- **`np.polyval(coeffs, X)`** — Evaluates the fitted polynomial at each x-value.
- **`train_mse = np.mean((y - y_pred) ** 2)`** — How well does the model fit the *training* data? Degree 15 will fit training data almost perfectly.
- **`test_mse = np.mean((y_test_true - y_test_pred) ** 2)`** — How well does it generalize to *new* data? Degree 15 will perform terribly here because it memorized noise.
- **The `status` label** — Degree 1 underfits (too simple), degree 4 is the sweet spot, degree 15 overfits (too complex).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `overfitting.py` and run: `python overfitting.py`

**Expected output:**
```
Degree  1: train MSE=0.1521, test MSE=0.2987 ← UNDERFITTING
Degree  4: train MSE=0.0368, test MSE=0.0142 ← GOOD FIT
Degree 15: train MSE=0.0148, test MSE=2.4523 ← OVERFITTING
```
(Degree 15 has low training error but terrible test error — the hallmark of overfitting.)

</details>

:::info[The Goldilocks Principle]
A good model is complex enough to capture real patterns but not so complex that it memorizes noise. This balance is controlled through model architecture, regularization, training duration, and data augmentation.
:::

### Strategies to Combat Overfitting

| Strategy | How It Works |
|----------|-------------|
| **More data** | More examples make it harder to memorize |
| **Regularization** | Penalize model complexity (L1, L2, dropout) |
| **Early stopping** | Stop training when validation loss starts increasing |
| **Cross-validation** | Average performance over multiple train/val splits |
| **Data augmentation** | Create variations of existing data (rotations, crops, etc.) |
| **Simpler model** | Use fewer parameters or a less flexible architecture |

## Evaluation Metrics

Choosing the right evaluation metric is critical. The "best" metric depends on your problem and business context.

### Regression Metrics

**Mean Squared Error (MSE)**: Average of squared errors. Penalizes large errors heavily.

:::info[Plain English: What Is MSE?]
Imagine a teacher grading darts. For each throw, she measures how far from the bullseye it landed, then *squares* that distance (so a throw that's 3 inches off counts as 9, not 3). She adds up all those squared distances and divides by the number of throws to get the average. Squaring makes big misses hurt *much* more than small ones.
:::

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Reading the formula:** $n$ is the number of data points. $y_i$ is the actual (true) value for data point $i$. $\hat{y}_i$ (read "y-hat sub i") is the model's prediction for that same point. $(y_i - \hat{y}_i)$ is the error — how far off the prediction is. Squaring it makes all errors positive and punishes big errors extra hard. $\sum_{i=1}^{n}$ means "add up for all $n$ points." Dividing by $n$ gives the average squared error.

**Mean Absolute Error (MAE)**: Average of absolute errors. More robust to outliers.

:::info[Plain English: What Is MAE?]
Same dart-throwing teacher, but now she just measures the raw distance from the bullseye — no squaring. A throw that's 3 inches off counts as 3. This is simpler and more forgiving of the occasional wild throw, because it doesn't blow up large errors the way squaring does.
:::

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Reading the formula:** Everything is the same as MSE except instead of squaring the error, we take the absolute value $|y_i - \hat{y}_i|$ — meaning we just ignore the sign (negative errors become positive). This gives a straightforward "average distance from the truth" without over-emphasizing outliers.

**R-squared ($R^2$)**: Proportion of variance explained. 1.0 is perfect, 0.0 means the model is no better than predicting the mean.

### Classification Metrics

```python title="Classification Metrics Explained"
import numpy as np

# Simulated binary classification results
y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])

# Confusion matrix components
TP = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
TN = np.sum((y_pred == 0) & (y_true == 0))  # True Negatives
FP = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
FN = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives

print(f"Confusion Matrix:")
print(f"  TP={TP}  FP={FP}")
print(f"  FN={FN}  TN={TN}")

accuracy  = (TP + TN) / len(y_true)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\\nAccuracy:  {accuracy:.2%}  (correct predictions / total)")
print(f"Precision: {precision:.2%}  (of predicted positives, how many are correct?)")
print(f"Recall:    {recall:.2%}  (of actual positives, how many did we find?)")
print(f"F1 Score:  {f1:.2%}  (harmonic mean of precision and recall)")
```

:::tip[Line-by-Line Walkthrough]
- **`y_true` and `y_pred`** — Two arrays: what the correct labels actually are, and what our model predicted.
- **`TP = np.sum((y_pred == 1) & (y_true == 1))`** — True Positives: the model said "yes" and the answer really was "yes." Think of catching real spam.
- **`TN`** — True Negatives: the model said "no" and it really was "no." Correctly letting a real email through.
- **`FP`** — False Positives: the model said "yes" but it was wrong. A real email flagged as spam (annoying!).
- **`FN`** — False Negatives: the model said "no" but it was wrong. Spam that slipped through (dangerous!).
- **`accuracy = (TP + TN) / len(y_true)`** — What fraction of all predictions were correct?
- **`precision = TP / (TP + FP)`** — Of all the times the model said "yes," how often was it right?
- **`recall = TP / (TP + FN)`** — Of all actual positive cases, how many did the model catch?
- **`f1 = 2 * precision * recall / (precision + recall)`** — The F1 score: a balanced combination of precision and recall. If either is low, F1 will be low.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `classification_metrics.py` and run: `python classification_metrics.py`

**Expected output:**
```
Confusion Matrix:
  TP=4  FP=1
  FN=2  TN=3

Accuracy:  70.00%  (correct predictions / total)
Precision: 80.00%  (of predicted positives, how many are correct?)
Recall:    66.67%  (of actual positives, how many did we find?)
F1 Score:  72.73%  (harmonic mean of precision and recall)
```

</details>

:::warning[Accuracy Can Be Misleading]
If 99% of emails are non-spam, a model that always predicts "not spam" has 99% accuracy but catches zero spam. In imbalanced datasets, precision, recall, and F1 are much more informative than accuracy.
:::

### When to Use Which Metric

| Scenario | Metric | Why |
|----------|--------|-----|
| Balanced classification | Accuracy or F1 | All classes equally important |
| Medical diagnosis | **Recall** (sensitivity) | Missing a disease (FN) is dangerous |
| Spam detection | **Precision** | Blocking legit emails (FP) is annoying |
| Imbalanced classes | F1 or AUC-ROC | Accuracy is misleading |
| Regression with outliers | MAE | Less sensitive to extreme values |
| Regression, penalize big errors | MSE/RMSE | Squares amplify large errors |

```python title="Implementing All Metrics"
import numpy as np

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

def classification_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

# Regression example
np.random.seed(42)
y_true_reg = np.random.randn(100) * 10
y_pred_reg = y_true_reg + np.random.randn(100) * 2
print("Regression Metrics:")
for name, val in regression_metrics(y_true_reg, y_pred_reg).items():
    print(f"  {name}: {val:.4f}")

# Classification example
y_true_cls = np.array([1]*50 + [0]*50)
y_pred_cls = np.array([1]*40 + [0]*10 + [1]*5 + [0]*45)
print("\\nClassification Metrics:")
for name, val in classification_metrics(y_true_cls, y_pred_cls).items():
    print(f"  {name}: {val:.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`def regression_metrics(y_true, y_pred):`** — A reusable function that computes all four major regression metrics at once.
- **`rmse = np.sqrt(mse)`** — Root Mean Squared Error: the square root of MSE, which puts the error back in the original units (e.g., dollars instead of dollars²).
- **`ss_res = np.sum((y_true - y_pred) ** 2)`** — Sum of squared residuals: total squared error of the model.
- **`ss_tot = np.sum((y_true - y_true.mean()) ** 2)`** — Total sum of squares: how much the data varies from its own mean.
- **`r2 = 1 - ss_res / ss_tot`** — R-squared: 1.0 means perfect predictions, 0.0 means the model is no better than guessing the average.
- **`def classification_metrics(y_true, y_pred):`** — Same idea for classification: computes accuracy, precision, recall, and F1 in one function.
- **The two example blocks** — Run each function on sample data to demonstrate the output format.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `all_metrics.py` and run: `python all_metrics.py`

**Expected output:**
```
Regression Metrics:
  MSE: 4.1234
  RMSE: 2.0306
  MAE: 1.6012
  R²: 0.9601

Classification Metrics:
  Accuracy: 0.8500
  Precision: 0.8889
  Recall: 0.8000
  F1: 0.8421
```

</details>

## The ML Workflow

Every ML project follows a similar workflow:

1. **Define the problem**: What are you predicting? What data do you have?
2. **Collect and explore data**: Understand distributions, correlations, quality issues
3. **Prepare data**: Clean, transform, encode features, split into train/val/test
4. **Choose a model**: Start simple (linear regression, logistic regression), then try complex models
5. **Train the model**: Feed training data, optimize parameters
6. **Evaluate**: Measure performance on validation set, iterate on model/features/hyperparameters
7. **Test**: Final evaluation on held-out test set
8. **Deploy and monitor**: Serve predictions, track performance over time

:::tip[Start Simple]
Always start with the simplest model that could work. A linear regression or logistic regression baseline takes minutes to implement and gives you a performance floor. You can then measure whether more complex models (neural networks, ensemble methods) are worth the added complexity.
:::

## Exercises

:::tip[Exercise 1: Build a Complete ML Pipeline — intermediate]

Using scikit-learn, build a complete ML pipeline for the Iris dataset:

1. Load and explore the dataset (print shape, feature names, class distribution)
2. Split into train (70%), validation (15%), test (15%)
3. Train a logistic regression classifier
4. Evaluate on validation set (accuracy, per-class precision/recall)
5. Only once satisfied, evaluate on test set and report final metrics
6. Print the confusion matrix

<details>
<summary>Hints</summary>

1. Use sklearn.datasets.load_iris() for a classic dataset.
2. Use sklearn.model_selection.train_test_split for splitting.
3. Try LogisticRegression from sklearn.linear_model.
4. Use sklearn.metrics for evaluation.

</details>

:::

:::tip[Exercise 2: Overfitting Experiment — intermediate]

Demonstrate overfitting empirically:

1. Generate 30 noisy data points from $y = \sin(x) + \varepsilon$
2. Fit polynomial models of degree 1, 3, 5, 10, and 20
3. For each degree, compute training MSE and test MSE (use 100 noise-free test points)
4. Identify which degree overfits, which underfits, and which is the best fit

<details>
<summary>Hints</summary>

1. Generate noisy data with np.sin() plus random noise.
2. Use np.polyfit with degrees 1, 3, 5, 10, 20.
3. Compare training MSE vs. test MSE for each degree.

</details>

:::

:::tip[Exercise 3: Metric Sensitivity Analysis — advanced]

Create four different classification scenarios (each with 100 predictions):

1. **Balanced, good model**: ~90% accuracy with balanced errors
2. **Imbalanced, naive model**: 95% class 0, model always predicts class 0
3. **High precision, low recall**: The model is conservative—rarely predicts positive, but when it does, it's right
4. **High recall, low precision**: The model is aggressive—catches most positives but with many false alarms

For each scenario, compute accuracy, precision, recall, and F1. Discuss which metric is most informative for each.

<details>
<summary>Hints</summary>

1. Create different scenarios: balanced, imbalanced, high-FP, high-FN.
2. Compute all four metrics for each scenario.
3. Think about which metric a doctor vs. email provider would prioritize.

</details>

:::

## Resources

- **[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)** _(tutorial)_ — The definitive guide to classical ML in Python. Every algorithm with examples and theory.

- **[Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)** _(course)_ by Google — Free, practical course covering ML fundamentals with TensorFlow exercises.

- **[An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/)** _(book)_ by James, Witten, Hastie & Tibshirani — Free textbook covering statistical ML concepts with R/Python labs. More accessible than ESL.

- **[StatQuest: Machine Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)** _(video)_ by Josh Starmer — Crystal-clear visual explanations of ML algorithms. Great for building intuition.

- **[Kaggle Learn: Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)** _(course)_ by Kaggle — Hands-on tutorial with real datasets. Complete in a few hours.

---

**Next up**: Now that you understand the ML landscape, it's time to dive into the technology that made the deep learning revolution possible — neural networks.
