---
sidebar_position: 6
slug: month2-project
title: "Project: Sentiment Analysis"
---


# Project: Sentiment Analysis on IMDB Reviews

:::info[What You'll Learn]
- Building an LSTM-based text classifier end to end
- Applying regularization to prevent overfitting
- Using learning rate scheduling for better convergence
- Evaluating model quality with multiple metrics
:::

:::note[Prerequisites]
All of Month 2 lessons 1–5.
:::

**Estimated time:** Reading: ~25 min | Project work: ~8 hours

This capstone project ties together everything from Month 2. You'll build an **LSTM-based sentiment classifier** that reads movie reviews and predicts whether they're positive or negative. Along the way, you'll practice data preprocessing, model design, training with regularization, evaluation, and error analysis.

:::info[What You'll Practice]
- **RNNs/LSTMs** — sequential modeling for text (Chapter 2)
- **Loss functions** — binary cross-entropy for two-class classification (Chapter 3)
- **Optimizers** — AdamW with learning rate scheduling (Chapter 3)
- **Regularization** — dropout, weight decay, early stopping (Chapter 4)
- **Analysis** — confusion matrices, per-class accuracy, failure case inspection
:::

## Overview

**Dataset:** IMDB Movie Reviews — 50,000 reviews labeled as positive or negative (25K train, 25K test).

**Goal:** Build a model that achieves **≥ 85% test accuracy** using an LSTM. Then push for 88%+ with the bonus challenges.

**Timeline:** This project should take 4–6 hours. Don't try to finish it in one sitting — split it across the steps below.

## Step 1: Data Preparation

We'll use HuggingFace's `datasets` library to load the IMDB dataset and build a vocabulary from scratch.

```python title="Load and explore the dataset"
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test:  Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# Inspect a few examples
for i in range(3):
    text = dataset["train"][i]["text"][:200]
    label = "Positive" if dataset["train"][i]["label"] == 1 else "Negative"
    print(f"[{label}] {text}...\\n")

# Check class balance
from collections import Counter
labels = dataset["train"]["label"]
print(Counter(labels))  # Counter({0: 12500, 1: 12500}) — perfectly balanced
```

:::tip[Line-by-Line Walkthrough]
- **`load_dataset("imdb")`** — Download and load the IMDB movie review dataset from HuggingFace. It has 25,000 training and 25,000 test reviews, each labeled as positive (1) or negative (0).
- **`dataset["train"][i]["text"][:200]`** — Access the first 200 characters of review *i* from the training split. Reviews can be very long, so we truncate for display.
- **`"Positive" if dataset["train"][i]["label"] == 1 else "Negative"`** — Convert the numeric label (0 or 1) to a human-readable string.
- **`Counter(labels)`** — Count how many positive and negative reviews there are. The dataset is perfectly balanced (12,500 each), so we don't need to worry about class imbalance.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install datasets
```

**Steps:**
1. Save to `load_imdb.py`
2. Run: `python load_imdb.py`
3. The dataset (~80 MB) will be downloaded automatically on first run.

**Expected output:**
```
DatasetDict({
    train: Dataset({features: ['text', 'label'], num_rows: 25000})
    test: Dataset({features: ['text', 'label'], num_rows: 25000})
})
[Negative] I rented I AM CURIOUS-YELLOW from my video store...
...
Counter({0: 12500, 1: 12500})
```

</details>

### Tokenization and Vocabulary

```python title="Build vocabulary and tokenizer"
import re
from collections import Counter

def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"<br\\s*/?>", " ", text)  # remove HTML tags
    text = re.sub(r"[^a-zA-Z\\s]", "", text)  # remove punctuation
    return text.split()

# Build vocabulary from training data
counter = Counter()
for example in dataset["train"]:
    counter.update(tokenize(example["text"]))

print(f"Unique tokens: {len(counter):,}")

# Keep the most common tokens
VOCAB_SIZE = 25000
PAD_IDX = 0
UNK_IDX = 1

vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
for word, _ in counter.most_common(VOCAB_SIZE - 2):
    vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab):,}")

def encode(text, max_len=256):
    """Convert text to padded/truncated integer sequence."""
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, UNK_IDX) for t in tokens]
    # Pad to max_len
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids
```

:::tip[Line-by-Line Walkthrough]
- **`text.lower()`** — Convert to lowercase so "Good" and "good" are treated as the same word.
- **`re.sub(r"<br\\s*/?>", " ", text)`** — Remove HTML `<br>` tags (common in web-scraped reviews) and replace with spaces.
- **`re.sub(r"[^a-zA-Z\\s]", "", text)`** — Remove all characters except letters and whitespace (strips punctuation, numbers, etc.).
- **`counter.update(tokenize(example["text"]))`** — Count how many times each word appears across all training reviews.
- **`VOCAB_SIZE = 25000`** — Keep only the 25,000 most common words. Rare words are mapped to `<unk>` (unknown).
- **`vocab = {"<pad>": 0, "<unk>": 1}`** — Reserve index 0 for padding (added to make all sequences the same length) and index 1 for unknown words.
- **`counter.most_common(VOCAB_SIZE - 2)`** — Get the 24,998 most frequent words (minus 2 for pad and unk).
- **`tokenize(text)[:max_len]`** — Tokenize and truncate to 256 words max (longer reviews are cut off).
- **`vocab.get(t, UNK_IDX)`** — Look up each token's integer ID. If the word isn't in the vocabulary, use the unknown token ID (1).
- **`ids += [PAD_IDX] * (max_len - len(ids))`** — Pad shorter sequences with zeros so all inputs have the same length.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install datasets
```

**Steps:**
1. Combine with the dataset loading code above in one file.
2. Run: `python build_vocab.py`

**Expected output:**
```
Unique tokens: 69,425
Vocabulary size: 25,000
```
(Exact token count may vary slightly.)

</details>

### Create DataLoaders

```python title="PyTorch Dataset and DataLoader"
import torch
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, max_len=256):
        self.texts = hf_dataset["text"]
        self.labels = hf_dataset["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

MAX_LEN = 256
BATCH_SIZE = 64

train_dataset = IMDBDataset(dataset["train"], max_len=MAX_LEN)
test_dataset = IMDBDataset(dataset["test"], max_len=MAX_LEN)

# Use 10% of training data as validation
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
```

:::tip[Line-by-Line Walkthrough]
- **`class IMDBDataset(Dataset)`** — A custom PyTorch dataset that wraps the HuggingFace dataset. PyTorch's `DataLoader` needs this interface.
- **`__len__`** — Returns the number of examples (25,000 for training).
- **`__getitem__(self, idx)`** — Called when the DataLoader requests example *idx*. Encodes the text to integer IDs and returns it with the label as PyTorch tensors.
- **`torch.tensor(ids, dtype=torch.long)`** — Token IDs must be `long` (64-bit integers) because they'll index into an embedding lookup table.
- **`torch.tensor(self.labels[idx], dtype=torch.float)`** — Labels as floats because `BCEWithLogitsLoss` expects float targets.
- **`random_split(train_dataset, [train_size, val_size])`** — Hold out 10% of training data (2,500 reviews) for validation.
- **`DataLoader(train_data, batch_size=64, shuffle=True)`** — Serve 64 reviews per batch, shuffled each epoch. Shuffling is important during training to avoid learning the order of the data.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets
```

**Steps:**
1. Combine with the vocabulary-building code above.
2. Run: `python create_dataloaders.py`

**Expected output:** No errors. To verify: `print(len(train_data), len(val_data), len(test_dataset))` → `22500 2500 25000`.

</details>

## Step 2: LSTM-Based Model

```python title="Sentiment analysis LSTM"
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # Concatenate final hidden states from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]  # last layer forward
        backward_hidden = hidden[-1]  # last layer backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        return self.fc(self.dropout(combined)).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM(vocab_size=VOCAB_SIZE).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# ~3.7M parameters
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)`** — Lookup table: each of the 25,000 words gets a 128-dimensional vector. `padding_idx=0` ensures the padding token always maps to a zero vector.
- **`nn.LSTM(..., bidirectional=True)`** — Bidirectional LSTM with 2 layers. Reads each review left-to-right AND right-to-left simultaneously, capturing context from both directions.
- **`dropout=dropout`** — Apply dropout between the LSTM layers (only effective when `num_layers > 1`).
- **`nn.Linear(hidden_dim * 2, 1)`** — Single output neuron for binary classification. `hidden_dim * 2` because the bidirectional LSTM doubles the hidden size. The output goes through sigmoid (via `BCEWithLogitsLoss`) to produce a probability.
- **`self.dropout(self.embedding(x))`** — Apply dropout to the embeddings: randomly zero out some word representations during training.
- **`hidden[-2]` / `hidden[-1]`** — Extract the final hidden states from the last LSTM layer: `-2` is the forward direction, `-1` is the backward direction.
- **`torch.cat([forward_hidden, backward_hidden], dim=1)`** — Concatenate into a single 512-dimensional vector that summarizes the entire review from both directions.
- **`.squeeze(1)`** — Remove the extra dimension so the output shape is `(batch_size,)` instead of `(batch_size, 1)`.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Combine with the data preparation code from Step 1.
2. Run: `python sentiment_model.py`

**Expected output:**
```
Total parameters: 3,727,873
```

</details>

:::tip[Architecture Decisions]
- **Bidirectional LSTM**: reads reviews in both directions, capturing context from the start and end of each sentence.
- **2 layers**: adds capacity without excessive complexity. The inter-layer dropout prevents overfitting between stacked LSTMs.
- **Dropout on embeddings**: prevents the model from memorizing specific word patterns.
- **Single output neuron**: binary classification — sigmoid gives a probability.
:::

## Step 3: Training

```python title="Training loop with validation"
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        logits = model(texts)
        loss = criterion(logits, labels)
        total_loss += loss.item() * texts.size(0)
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total
```

:::tip[Line-by-Line Walkthrough]
- **`nn.BCEWithLogitsLoss()`** — Binary cross-entropy with built-in sigmoid. Perfect for our single-output binary classifier.
- **`optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`** — AdamW optimizer with weight decay (L2 regularization). lr=0.001 is a good starting point.
- **`ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)`** — If the validation loss doesn't improve for 2 consecutive epochs, halve the learning rate. This helps fine-tune in the later stages of training.
- **`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`** — Clip gradient norms to 1.0. RNNs are prone to exploding gradients on long sequences; this prevents training instability.
- **`preds = (logits > 0).float()`** — Convert logits to predictions: if the logit is positive (>0), predict positive (1.0); otherwise predict negative (0.0). The threshold of 0 corresponds to 50% probability after sigmoid.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Combine with Steps 1 and 2 in one file.
2. These are helper functions used by the training loop below.

**Expected output:** No output — these functions are called in the next code block.

</details>

```python title="Run training"
NUM_EPOCHS = 15
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_sentiment_model.pt")
        print("  -> Saved best model")
```

:::tip[Line-by-Line Walkthrough]
- **`NUM_EPOCHS = 15`** — Train for 15 epochs. With early stopping via `ReduceLROnPlateau`, training may effectively converge earlier.
- **`best_val_loss = float("inf")`** — Start with "infinite" loss so any real value triggers a save.
- **`scheduler.step(val_loss)`** — Tell the scheduler about the current validation loss. If it hasn't improved for `patience` epochs, the LR will be halved.
- **`optimizer.param_groups[0]["lr"]`** — Print the current learning rate so you can see when the scheduler reduces it.
- **`if val_loss < best_val_loss:`** — Save the model whenever validation loss reaches a new minimum. This is a simple form of early stopping — you keep the best model, not the last one.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets
```

**Steps:**
1. Combine all code from Steps 1–3 into a single file, e.g. `train_sentiment.py`.
2. Run: `python train_sentiment.py`

**Expected output:**
```
Epoch 01 | Train Loss: 0.5832 Acc: 0.6894 | Val Loss: 0.4213 Acc: 0.8124 | LR: 1.00e-03
  -> Saved best model
Epoch 02 | Train Loss: 0.3561 Acc: 0.8467 | Val Loss: 0.3312 Acc: 0.8592 | LR: 1.00e-03
  -> Saved best model
...
Epoch 15 | Train Loss: 0.1234 Acc: 0.9567 | Val Loss: 0.3102 Acc: 0.8784 | LR: 2.50e-04
```
Training takes ~10–30 minutes on CPU, ~2–5 minutes on GPU.

</details>

:::warning[Gradient Clipping]
Notice the `clip_grad_norm_` call in the training loop. RNNs are prone to exploding gradients, especially on long sequences. Clipping the gradient norm to 1.0 prevents training instability without affecting the direction of updates.
:::

## Step 4: Evaluation and Analysis

### Test Set Performance

```python title="Final evaluation"
# Load best model
model.load_state_dict(torch.load("best_sentiment_model.pt"))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`torch.load("best_sentiment_model.pt")`** — Load the saved weights from the best epoch (lowest validation loss).
- **`model.load_state_dict(...)`** — Apply the loaded weights to the model. This replaces the current (potentially overfitting) weights with the best ones.
- **`evaluate(model, test_loader, criterion, device)`** — Run the model on the held-out test set (25,000 reviews it has never seen during training or validation).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Run after training completes (or load a previously saved model).
2. Add to the end of your training script.

**Expected output:**
```
Test Loss: 0.3256 | Test Accuracy: 0.8612
```
(Target: ≥ 85%)

</details>

### Detailed Analysis

```python title="Confusion matrix and classification report"
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        logits = model(texts)
        preds = (logits > 0).float().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

print("Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)
# [[TN, FP],
#  [FN, TP]]
```

:::tip[Line-by-Line Walkthrough]
- **`all_preds, all_labels = [], []`** — Collect all predictions and labels across the entire test set.
- **`(logits > 0).float().cpu()`** — Convert logits to binary predictions (positive/negative) and move to CPU for sklearn.
- **`classification_report(all_labels, all_preds, target_names=...)`** — Generate precision, recall, F1-score for each class. Precision = "of all predicted positives, how many were actually positive?" Recall = "of all actual positives, how many did we correctly find?"
- **`confusion_matrix(all_labels, all_preds)`** — A 2×2 grid: true negatives (top-left), false positives (top-right), false negatives (bottom-left), true positives (bottom-right).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch scikit-learn numpy
```

**Steps:**
1. Add to the end of your training/evaluation script.
2. Run after loading the best model.

**Expected output:**
```
Classification Report:
              precision    recall  f1-score   support
    Negative       0.87      0.85      0.86     12500
    Positive       0.85      0.87      0.86     12500
    accuracy                           0.86     25000

Confusion Matrix:
[[10625  1875]
 [ 1625 10875]]
```

</details>

### Error Analysis

Understanding **why** the model fails is often more valuable than squeezing out an extra percentage point of accuracy.

```python title="Analyze misclassified reviews"
def find_errors(model, dataset, device, n=10):
    """Find and display misclassified examples."""
    model.eval()
    errors = []
    for i in range(len(dataset)):
        ids, label = dataset[i]
        with torch.no_grad():
            logit = model(ids.unsqueeze(0).to(device))
            pred = (logit > 0).float().item()
            confidence = torch.sigmoid(logit).item()

        if pred != label.item():
            errors.append({
                "text": dataset.dataset.texts[i] if hasattr(dataset, 'dataset') else "N/A",
                "true_label": "Positive" if label.item() == 1 else "Negative",
                "predicted": "Positive" if pred == 1 else "Negative",
                "confidence": confidence,
            })
        if len(errors) >= n:
            break
    return errors

errors = find_errors(model, test_dataset, device)
for e in errors:
    print(f"True: {e['true_label']} | Pred: {e['predicted']} ({e['confidence']:.2%})")
    print(f"  {e['text'][:300]}...")
    print()
```

:::tip[Line-by-Line Walkthrough]
- **`ids.unsqueeze(0).to(device)`** — Add a batch dimension (model expects a batch) and move to GPU/CPU.
- **`torch.sigmoid(logit).item()`** — Convert the raw logit to a probability (0 to 1). Values near 0.5 mean the model is uncertain.
- **`if pred != label.item()`** — Only collect examples where the model got it wrong.
- **`dataset.dataset.texts[i]`** — Access the original text from the underlying dataset (through the PyTorch wrapper).
- **`e['text'][:300]`** — Print the first 300 characters of each misclassified review so you can see why the model might have been confused.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets
```

**Steps:**
1. Add to the end of your evaluation script.
2. Run after loading the best model.

**Expected output:**
```
True: Positive | Pred: Negative (42.31%)
  This movie had some truly wonderful moments, but overall I found...

True: Negative | Pred: Positive (67.89%)
  Oh sure, this was just the BEST movie ever made...
```
(The model often fails on sarcastic or mixed-sentiment reviews.)

</details>

:::info[Common Error Patterns]
Look for these patterns in misclassified reviews:
- **Sarcasm and irony**: "Oh sure, this movie was just *wonderful*" — positive words, negative intent.
- **Mixed sentiment**: reviews that praise acting but criticize the plot.
- **Negation**: "not bad" should be positive, but the model may focus on "bad."
- **Very long reviews**: information from the beginning may be lost despite the LSTM's memory.
:::

### Confidence Distribution

```python title="Visualize prediction confidence"
import matplotlib.pyplot as plt

all_probs = []
model.eval()
with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        probs = torch.sigmoid(model(texts)).cpu().numpy()
        all_probs.extend(probs)

all_probs = np.array(all_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correct predictions
correct_mask = all_preds == all_labels
axes[0].hist(all_probs[correct_mask], bins=50, alpha=0.7, color="green")
axes[0].set_title(f"Correct Predictions (n={correct_mask.sum()})")
axes[0].set_xlabel("Predicted Probability (positive)")

# Incorrect predictions
axes[1].hist(all_probs[~correct_mask], bins=50, alpha=0.7, color="red")
axes[1].set_title(f"Incorrect Predictions (n={(~correct_mask).sum()})")
axes[1].set_xlabel("Predicted Probability (positive)")

for ax in axes:
    ax.set_ylabel("Count")
    ax.axvline(0.5, color="black", linestyle="--")
plt.tight_layout()
plt.show()
```

:::tip[Line-by-Line Walkthrough]
- **`torch.sigmoid(model(texts)).cpu().numpy()`** — Get predicted probabilities for every test example.
- **`plt.subplots(1, 2, figsize=(14, 5))`** — Create a side-by-side figure: correct predictions on the left, incorrect on the right.
- **`correct_mask = all_preds == all_labels`** — Boolean mask: `True` for every correct prediction.
- **`axes[0].hist(all_probs[correct_mask], bins=50, ...)`** — Histogram of probabilities for correctly classified reviews. You want to see peaks near 0 (confident negative) and 1 (confident positive).
- **`axes[1].hist(all_probs[~correct_mask], bins=50, ...)`** — Histogram for misclassified reviews. If these cluster near 0.5, the model was uncertain. If near 0 or 1, the model was confidently wrong.
- **`ax.axvline(0.5, ...)`** — Draw a dashed line at 0.5 (the decision boundary).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch matplotlib numpy
```

**Steps:**
1. Add after the confusion matrix code (it uses `all_preds` and `all_labels` from there).
2. Run the script.

**Expected output:** A figure with two histograms. Correct predictions should show peaks near 0 and 1 (high confidence). Incorrect predictions may cluster near 0.5 (uncertain) or show some near 0/1 (confidently wrong).

</details>

:::tip[What to Look For]
If incorrect predictions cluster near 0.5, the model is uncertain — these are borderline cases. If incorrect predictions have high confidence (near 0 or 1), the model is confidently wrong — those are the most interesting failures to analyze.
:::

## Bonus Challenges

These challenges push you beyond the baseline and introduce techniques you'll use heavily in later months.

:::tip[Bonus 1: Pretrained Embeddings — intermediate]

Replace the randomly initialized embedding layer with **pretrained GloVe embeddings** (100-dimensional). Compare test accuracy against the baseline. Does starting from GloVe help? Try both freezing the embeddings (feature extraction) and fine-tuning them. Which works better?

<details>
<summary>Hints</summary>

1. Download GloVe embeddings (glove.6B.100d.txt) from Stanford NLP
2. Create an embedding matrix matching your vocabulary
3. Load it into nn.Embedding with from_pretrained()
4. Try both freezing and fine-tuning the embeddings

</details>

:::

:::tip[Bonus 2: Attention Mechanism — advanced]

Add a **self-attention layer** on top of the LSTM outputs. Instead of only using the final hidden state, compute attention weights over all time steps and use a weighted sum as the review representation. Visualize the attention weights for a few reviews — does the model attend to sentiment-bearing words (e.g., "terrible," "brilliant," "boring")?

<details>
<summary>Hints</summary>

1. Instead of using the final hidden state, compute a weighted sum over all LSTM outputs
2. The attention weights should be a softmax over learned scores
3. A simple attention: score = v^T tanh(W * h_t), then normalize with softmax
4. This lets the model focus on the most sentiment-bearing words

</details>

:::

:::tip[Bonus 3: CNN-LSTM Hybrid — advanced]

Build a **CNN-LSTM hybrid** model. Use 1D convolutions to extract local n-gram features from the embedding sequence, then feed those features into an LSTM for sequential modeling. Compare this architecture against the pure LSTM baseline. Does combining local (CNN) and global (LSTM) features improve accuracy?

<details>
<summary>Hints</summary>

1. Use 1D convolutions (nn.Conv1d) over the embedding sequence to extract n-gram features
2. Feed the conv output into the LSTM instead of raw embeddings
3. The CNN captures local patterns (word pairs/triples); the LSTM captures long-range dependencies
4. Conv1d expects (batch, channels, length) — transpose after embedding

</details>

:::

:::tip[Bonus 4: Compare Against a Transformer — advanced]

As a preview of Month 3, fine-tune a **pretrained DistilBERT** model on the IMDB dataset using HuggingFace Transformers. Compare its test accuracy, training time, and model size against your LSTM. How much does the pretrained transformer improve over the LSTM? What does this tell you about the value of pretraining at scale?

<details>
<summary>Hints</summary>

1. Use HuggingFace's transformers library with a small pretrained model like distilbert-base-uncased
2. Fine-tune for just 2-3 epochs — transformers converge quickly
3. Compare accuracy, training time, model size, and inference speed
4. This preview foreshadows Month 3's deep dive into transformers

</details>

:::

## Checklist

Before you consider this project complete, verify:

- [ ] Data pipeline loads IMDB, tokenizes, and creates DataLoaders
- [ ] LSTM model trains successfully and converges
- [ ] Test accuracy ≥ 85%
- [ ] Confusion matrix and classification report generated
- [ ] At least 5 misclassified examples analyzed
- [ ] Confidence distribution plotted
- [ ] At least one bonus challenge attempted

## Resources

- **[IMDB Dataset](https://huggingface.co/datasets/imdb)** _(tool)_ — The IMDB movie review dataset on HuggingFace — 50K reviews for binary sentiment classification.

- **[Practical Text Classification with PyTorch](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)** _(tutorial)_ by PyTorch — Official PyTorch tutorial on text classification covering vocabulary building and model training.

- **[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)** _(tool)_ by Stanford NLP — Pretrained word embeddings in multiple dimensions. Download glove.6B for the bonus challenge.

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** _(paper)_ by Vaswani et al. — The transformer paper — preview reading for Month 3. After building attention from scratch in Bonus 2, see how the full architecture works.
