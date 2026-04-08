---
sidebar_position: 7
slug: month3-project
title: "Project: Text Classifier with Transformers"
---


# Project: Text Classifier with Transformers

This capstone project brings together everything from Month 3. You'll build a **text classification model** using a Transformer encoder, train it on a real dataset, and compare it against an LSTM baseline. By the end, you'll have hands-on experience with the full pipeline — from data preprocessing to model evaluation.

## Project Overview

**Task:** Sentiment analysis on the IMDB movie review dataset (binary classification: positive/negative).

**What you'll build:**
1. A data pipeline with tokenization and batching
2. An LSTM baseline classifier
3. A Transformer encoder classifier
4. Training loops with proper evaluation
5. Comparison and analysis

:::info[Why a Transformer Encoder?]
Text classification doesn't require generating text — we only need to *understand* it. So we use just the **encoder** portion of the Transformer, followed by a classification head. This is exactly what BERT does (BERT = Bidirectional Encoder Representations from Transformers).
:::

## Step 1: Dataset Preparation

We'll use the IMDB dataset, which contains 50,000 movie reviews (25k train, 25k test), each labeled as positive or negative.

```python title="Loading and preprocessing the IMDB dataset"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# --- Download IMDB dataset using Hugging Face datasets ---
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

print(f"Training samples: {len(train_data)}")
print(f"Test samples:     {len(test_data)}")
print(f"Sample: {train_data[0]['text'][:200]}...")
print(f"Label:  {train_data[0]['label']} (0=negative, 1=positive)")

# --- Text cleaning ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\\s*/?>", " ", text)       # HTML line breaks
    text = re.sub(r"[^a-z0-9\\s]", " ", text)     # keep only alphanumeric
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# --- Build vocabulary ---
def build_vocab(texts: list[str], max_vocab: int = 25000, min_freq: int = 2):
    counter = Counter()
    for text in texts:
        tokens = clean_text(text).split()
        counter.update(tokens)

    # Reserve indices: 0=PAD, 1=UNK
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"Vocabulary size: {len(vocab)}")
    return vocab

vocab = build_vocab(train_data["text"])

# --- Dataset class ---
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx]).split()[:self.max_len]
        indices = [self.vocab.get(t, 1) for t in tokens]  # 1 = UNK

        # Pad to max_len
        padded = indices + [0] * (self.max_len - len(indices))
        return torch.tensor(padded), torch.tensor(self.labels[idx])

# Create datasets and loaders
BATCH_SIZE = 64
MAX_LEN = 256

train_dataset = IMDBDataset(train_data["text"], train_data["label"], vocab, MAX_LEN)
test_dataset = IMDBDataset(test_data["text"], test_data["label"], vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Verify
batch_x, batch_y = next(iter(train_loader))
print(f"Batch input shape:  {batch_x.shape}")   # (64, 256)
print(f"Batch labels shape: {batch_y.shape}")    # (64,)
```

:::tip[Line-by-Line Walkthrough]
- **`load_dataset("imdb")`** — Downloads the IMDB movie review dataset from Hugging Face. It contains 25,000 training and 25,000 test reviews, each labeled positive (1) or negative (0).
- **`clean_text(text)`** — Lowercases text, removes HTML tags (IMDB reviews have `<br/>` tags), strips non-alphanumeric characters, and collapses extra whitespace. Cleans up the raw data so the model sees consistent input.
- **`build_vocab(texts, max_vocab=25000, min_freq=2)`** — Counts how often each word appears, keeps the 25,000 most common words that appear at least twice. Reserves index 0 for padding and 1 for unknown words.
- **`self.vocab.get(t, 1)`** — Looks up each token's index in the vocabulary. If the word isn't in the vocabulary, it gets index 1 (the unknown token). Like looking up a word in a dictionary — if it's not there, you mark it as "unknown."
- **`indices + [0] * (self.max_len - len(indices))`** — Pads shorter sequences with zeros (the padding index) so all sequences in a batch have the same length.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets
```

**Steps:**
1. Save to `imdb_data.py`
2. Run: `python imdb_data.py`
3. The first run downloads the IMDB dataset (~85 MB).

**Expected output:**
```
Training samples: 25000
Test samples:     25000
Sample: Bromwell High is a cartoon comedy. It ran at the same time ...
Label:  1 (0=negative, 1=positive)
Vocabulary size: 23456
Batch input shape:  torch.Size([64, 256])
Batch labels shape: torch.Size([64])
```
(Vocabulary size will vary slightly.)

</details>

:::warning[Sequence Length Trade-off]
IMDB reviews can be 500+ words. Setting `max_len=256` truncates long reviews but keeps training fast. In production, you'd use dynamic padding (pad to the longest sequence in the batch, not a global max) for efficiency.
:::

## Step 2: LSTM Baseline

Before building the Transformer, let's establish a baseline with an LSTM classifier.

```python title="LSTM classifier baseline"
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional doubles hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        mask = (x != 0)  # (B, T) — True where not padding
        embedded = self.dropout(self.embedding(x))            # (B, T, E)
        output, (hidden, _) = self.lstm(embedded)             # output: (B, T, 2H)

        # Use mean pooling over non-padded positions
        mask_expanded = mask.unsqueeze(-1).float()            # (B, T, 1)
        summed = (output * mask_expanded).sum(dim=1)          # (B, 2H)
        lengths = mask.sum(dim=1, keepdim=True).float()       # (B, 1)
        pooled = summed / lengths.clamp(min=1)                # (B, 2H)

        return self.classifier(pooled)

lstm_model = LSTMClassifier(vocab_size=len(vocab))
params = sum(p.numel() for p in lstm_model.parameters())
print(f"LSTM parameters: {params:,}")
```

:::tip[Line-by-Line Walkthrough]
- **`nn.LSTM(..., bidirectional=True)`** — A bidirectional LSTM reads the sequence both forwards and backwards, so each token gets context from the entire review, not just the words before it. This doubles the hidden dimension.
- **`mask = (x != 0)`** — Creates a boolean mask that is True for real words and False for padding. This prevents padding from polluting the results.
- **`(output * mask_expanded).sum(dim=1)`** — Mean pooling: sums the LSTM output across all non-padding positions, then divides by the actual sequence length. This produces one fixed-size vector per review regardless of review length.
- **`self.classifier = nn.Sequential(Linear, ReLU, Dropout, Linear)`** — A small feed-forward classification head that takes the pooled representation and outputs scores for each class (positive/negative).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save together with the data loading code from Step 1 to `lstm_classifier.py`
2. Run: `python lstm_classifier.py`

**Expected output:**
```
LSTM parameters: 1,234,562
```
(Exact count depends on vocabulary size.)

</details>

## Step 3: Transformer Encoder Classifier

Now let's build the main model. We take a Transformer encoder and add a classification head on top.

```python title="Transformer encoder classifier"
import math
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = self._sinusoidal_encoding(max_len, d_model)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        padding_mask = (x == self.pad_idx)    # True = ignore

        # Embed and add positional encoding
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = emb + self.pos_encoding[:, :x.size(1)]
        emb = self.pos_dropout(emb)

        # Transformer encoder
        encoded = self.encoder(emb, src_key_padding_mask=padding_mask)
        encoded = self.encoder_norm(encoded)

        # Pool: mean over non-padding positions
        mask = (~padding_mask).unsqueeze(-1).float()   # (B, T, 1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d_model)

        return self.classifier(pooled)

transformer_model = TransformerClassifier(vocab_size=len(vocab))
params = sum(p.numel() for p in transformer_model.parameters())
print(f"Transformer parameters: {params:,}")
```

:::tip[Line-by-Line Walkthrough]
- **`nn.TransformerEncoderLayer(..., norm_first=True)`** — Uses PyTorch's built-in Transformer encoder layer with pre-norm (normalize before attention/FFN), which trains more stably than the original paper's post-norm.
- **`self._sinusoidal_encoding(max_len, d_model)`** — Pre-computes fixed sinusoidal positional encodings (the same sine/cosine waves from the original Transformer paper). Stored as a non-trainable parameter.
- **`padding_mask = (x == self.pad_idx)`** — PyTorch's Transformer uses `True = ignore`, so padding positions are marked True.
- **`emb = self.embedding(x) * math.sqrt(self.d_model)`** — Scales embeddings by √d_model to balance them against positional encodings.
- **Mean pooling** — Instead of using a special [CLS] token, this averages all non-padding token representations to get a single vector per review. The mask ensures padding tokens don't contribute.
- **`self.classifier = nn.Sequential(Linear, GELU, Dropout, Linear)`** — Classification head using GELU activation (smoother than ReLU, commonly used in modern Transformers).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save together with data loading code to `transformer_classifier.py`
2. Run: `python transformer_classifier.py`

**Expected output:**
```
Transformer parameters: 2,345,678
```
(Exact count depends on vocabulary size.)

</details>

:::tip[Using PyTorch's Built-in Transformer]
Here we use `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` for convenience. You could substitute the from-scratch modules from the previous lesson. Using the built-in version lets us focus on the classification task while knowing the attention implementation is battle-tested.
:::

## Step 4: Training and Evaluation

```python title="Training utilities"
from torch.optim.lr_scheduler import OneCycleLR

def train_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, num_epochs=10, lr=3e-4, device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    return history
```

:::tip[Line-by-Line Walkthrough]
- **`train_epoch(...)`** — Runs one full pass through the training data. For each batch: forward pass → compute loss → backward pass → clip gradients → update weights → step the learning rate scheduler.
- **`clip_grad_norm_(model.parameters(), 1.0)`** — Prevents exploding gradients by capping the total gradient magnitude at 1.0.
- **`OneCycleLR(optimizer, max_lr=lr, ...)`** — A learning rate schedule that warms up from a low LR, peaks at `max_lr`, then gradually decays. This often trains faster than a constant learning rate.
- **`evaluate(...)`** — Runs the model on test data with `@torch.no_grad()` (no gradient computation) to measure accuracy and loss without affecting the model.
- **`history` dictionary** — Tracks training and test metrics across epochs for later plotting.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save with both model classes and data loading code
2. These are utility functions — they're called by the training code below

**Expected output:** No direct output — these functions are used by the training script in the next block.

</details>

```python title="Training both models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_EPOCHS = 8

# --- Train LSTM baseline ---
print("\\n" + "="*50)
print("Training LSTM Baseline")
print("="*50)
lstm_model = LSTMClassifier(vocab_size=len(vocab))
lstm_history = train_model(lstm_model, train_loader, test_loader,
                           num_epochs=NUM_EPOCHS, lr=1e-3, device=device)

# --- Train Transformer ---
print("\\n" + "="*50)
print("Training Transformer Classifier")
print("="*50)
transformer_model = TransformerClassifier(vocab_size=len(vocab))
transformer_history = train_model(transformer_model, train_loader, test_loader,
                                  num_epochs=NUM_EPOCHS, lr=3e-4, device=device)
```

:::tip[Line-by-Line Walkthrough]
- **`torch.device("cuda" if torch.cuda.is_available() else "cpu")`** — Automatically uses GPU if available, otherwise falls back to CPU. GPU training is 5-10× faster for this task.
- **LSTM uses `lr=1e-3`** — LSTMs often benefit from a slightly higher learning rate than Transformers.
- **Transformer uses `lr=3e-4`** — Transformers typically use smaller learning rates (3e-4 is a common starting point, following the BERT convention).
- **`NUM_EPOCHS = 8`** — Both models train for 8 epochs. Each epoch processes all 25,000 training reviews once.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch datasets
```

**Steps:**
1. Save all code (data loading, both models, training utilities, and this block) to `train_classifiers.py`
2. Run: `python train_classifiers.py`
3. Training takes ~15-30 minutes on CPU, ~3-5 minutes on GPU.

**Expected output:**
```
Using device: cpu
==================================================
Training LSTM Baseline
==================================================
Epoch  1/8 | Train Loss: 0.6821 Acc: 0.5534 | Test  Loss: 0.6102 Acc: 0.6780
...
Epoch  8/8 | Train Loss: 0.2134 Acc: 0.9156 | Test  Loss: 0.3456 Acc: 0.8621
==================================================
Training Transformer Classifier
==================================================
Epoch  1/8 | Train Loss: 0.6543 Acc: 0.5892 | Test  Loss: 0.5678 Acc: 0.7123
...
Epoch  8/8 | Train Loss: 0.1876 Acc: 0.9287 | Test  Loss: 0.3210 Acc: 0.8745
```

</details>

## Step 5: Comparing LSTM vs. Transformer

```python title="Visualization and comparison"
import matplotlib.pyplot as plt

def plot_comparison(lstm_hist, transformer_hist):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(lstm_hist["train_loss"], label="LSTM Train", linestyle="--", color="#3b82f6")
    axes[0].plot(lstm_hist["test_loss"], label="LSTM Test", color="#3b82f6")
    axes[0].plot(transformer_hist["train_loss"], label="Transformer Train", linestyle="--", color="#ef4444")
    axes[0].plot(transformer_hist["test_loss"], label="Transformer Test", color="#ef4444")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Test Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(lstm_hist["train_acc"], label="LSTM Train", linestyle="--", color="#3b82f6")
    axes[1].plot(lstm_hist["test_acc"], label="LSTM Test", color="#3b82f6")
    axes[1].plot(transformer_hist["train_acc"], label="Transformer Train", linestyle="--", color="#ef4444")
    axes[1].plot(transformer_hist["test_acc"], label="Transformer Test", color="#ef4444")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lstm_vs_transformer.png", dpi=150)
    plt.show()

    # Summary table
    print("\\n" + "="*55)
    print(f"{'Metric':<25} {'LSTM':>12} {'Transformer':>14}")
    print("="*55)
    print(f"{'Final Test Accuracy':<25} {lstm_hist['test_acc'][-1]:>11.4f} {transformer_hist['test_acc'][-1]:>13.4f}")
    print(f"{'Final Test Loss':<25} {lstm_hist['test_loss'][-1]:>11.4f} {transformer_hist['test_loss'][-1]:>13.4f}")
    print(f"{'Best Test Accuracy':<25} {max(lstm_hist['test_acc']):>11.4f} {max(transformer_hist['test_acc']):>13.4f}")

    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    trans_params = sum(p.numel() for p in transformer_model.parameters())
    print(f"{'Parameters':<25} {lstm_params:>11,} {trans_params:>13,}")
    print("="*55)

plot_comparison(lstm_history, transformer_history)
```

:::tip[Line-by-Line Walkthrough]
- **`fig, axes = plt.subplots(1, 2, ...)`** — Creates two side-by-side plots: one for loss curves, one for accuracy curves. This lets you compare LSTM and Transformer training dynamics at a glance.
- **Dashed lines (`linestyle="--"`)** — Represent training metrics; solid lines represent test metrics. If the training line is much better than the test line, the model is overfitting.
- **Blue for LSTM, Red for Transformer** — Color-coded so you can instantly tell which model is performing better.
- **Summary table** — Prints a clean comparison of final accuracy, best accuracy, and parameter counts for both models.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch matplotlib scikit-learn
```

**Steps:**
1. Append this code to the training script
2. Run after training completes

**Expected output:** A two-panel figure saved as `lstm_vs_transformer.png` showing loss and accuracy curves for both models, plus a printed summary table comparing final metrics.

</details>

:::info[What to Expect]
On IMDB with these model sizes and training budgets:
- **LSTM** typically reaches **85-87%** test accuracy.
- **Transformer** typically reaches **86-88%** test accuracy.

The gap is modest because IMDB reviews are relatively short and the task is simple. The Transformer's advantage becomes much more pronounced on longer sequences and more complex tasks. Pretrained Transformers like BERT reach **93%+** because they leverage massive pretraining data.
:::

## Step 6: Error Analysis

Understanding *where* your model fails is as important as knowing its accuracy.

```python title="Error analysis"
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = F.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(batch_y.tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_preds, all_labels, all_probs

preds, labels, probs = get_predictions(transformer_model, test_loader, device)

# Find confident errors
errors = []
for i, (pred, label, prob) in enumerate(zip(preds, labels, probs)):
    if pred != label:
        confidence = max(prob)
        errors.append((i, pred, label, confidence, test_data[i]["text"][:200]))

# Sort by confidence (most confident errors first)
errors.sort(key=lambda x: x[3], reverse=True)

print("Top 5 Most Confident Errors:")
print("-" * 80)
for idx, pred, label, conf, text in errors[:5]:
    pred_str = "positive" if pred == 1 else "negative"
    true_str = "positive" if label == 1 else "negative"
    print(f"Predicted: {pred_str} (confidence: {conf:.3f}) | True: {true_str}")
    print(f"Text: {text}...")
    print("-" * 80)

# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print("\\nClassification Report (Transformer):")
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
```

:::tip[Line-by-Line Walkthrough]
- **`get_predictions(model, loader, device)`** — Runs the model on the entire test set and collects predictions, true labels, and probability scores for each review.
- **`F.softmax(logits, dim=1)`** — Converts raw model scores into probabilities (summing to 1). A probability of 0.95 for "positive" means the model is very confident.
- **Confident errors** — Finds cases where the model was very sure of its answer but got it wrong. These are the most interesting errors to analyze — they reveal systematic blind spots.
- **`errors.sort(key=lambda x: x[3], reverse=True)`** — Sorts errors by confidence (highest confidence first) so you see the model's worst mistakes first.
- **`classification_report(...)`** — From scikit-learn: prints precision, recall, and F1-score for each class, giving a complete picture of model performance beyond just accuracy.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch scikit-learn datasets
```

**Steps:**
1. Append this code to the training script (after the Transformer has been trained)
2. Run the full script

**Expected output:**
```
Top 5 Most Confident Errors:
--------------------------------------------------------------------------------
Predicted: positive (confidence: 0.987) | True: negative
Text: This movie had such potential but the ending completely ruined...
--------------------------------------------------------------------------------
...

Classification Report (Transformer):
              precision    recall  f1-score   support
    Negative       0.87      0.86      0.87     12500
    Positive       0.86      0.87      0.87     12500
    accuracy                           0.87     25000
```

</details>

## Bonus Challenges

These extend the project beyond the basics. Tackle them to deepen your understanding.

:::tip[Bonus 1: CLS Token Classification — intermediate]

Instead of mean pooling, prepend a special `[CLS]` token to every input and use its representation for classification (the BERT approach). Compare accuracy vs. mean pooling. Which works better? Why might one be preferable?

<details>
<summary>Hints</summary>

1. Add a special [CLS] token (index 2) at the beginning of every input
2. Update vocab to include it
3. Use the Transformer output at position 0 as the sequence representation
4. This is how BERT does classification

</details>

:::

:::tip[Bonus 2: Attention-Based Interpretability — intermediate]

Extract attention weights from the trained Transformer and identify which tokens receive the most attention for positive vs. negative predictions. Create a visualization that highlights important words in a review (like a simpler version of BertViz). Do the highlighted words align with human intuition about sentiment?

<details>
<summary>Hints</summary>

1. Extract attention weights from each layer
2. Average across heads and layers
3. For each review, find the top-10 highest-attention tokens
4. Do high-attention words correspond to sentiment words?

</details>

:::

:::tip[Bonus 3: Pretrained Embeddings — intermediate]

Replace the randomly initialized embedding layer with pretrained GloVe embeddings. Test three configurations: (1) frozen embeddings, (2) fine-tuned embeddings, (3) random initialization. Which performs best, and does the advantage change with training duration?

<details>
<summary>Hints</summary>

1. Load GloVe 100d or 200d vectors
2. Initialize the embedding layer with pretrained weights
3. Try both freezing and fine-tuning the embeddings
4. Compare to random initialization

</details>

:::

:::tip[Bonus 4: Multi-Class Extension — advanced]

Extend your classifier to handle multi-class classification. Use the AG News dataset (4 categories: World, Sports, Business, Sci/Tech). How does the Transformer's advantage over the LSTM change with more classes? Does the Transformer benefit more from the ability to capture cross-topic linguistic patterns?

<details>
<summary>Hints</summary>

1. Use the AG News or Yahoo Answers dataset (4-10 classes)
2. Adjust num_classes in the classifier
3. Consider class imbalance — use weighted cross-entropy if needed
4. Evaluate with per-class F1 scores

</details>

:::

:::tip[Bonus 5: Efficiency Comparison — advanced]

Benchmark training speed and GPU memory usage for the LSTM vs. Transformer across different sequence lengths (128, 256, 512, 1024). At what sequence length does the Transformer's \(O(n^2)\) attention cost dominate? Plot training time per epoch and peak memory usage.

<details>
<summary>Hints</summary>

1. Measure training time per epoch for both models
2. Profile GPU memory usage with torch.cuda.max_memory_allocated()
3. Test with max_len = 128, 256, 512, 1024
4. Plot time and memory vs. sequence length for both models

</details>

:::

---

## Submission Checklist

Before considering this project complete, verify:

- [ ] Data pipeline loads IMDB, tokenizes, builds vocabulary, creates DataLoaders
- [ ] LSTM baseline trains and evaluates successfully
- [ ] Transformer classifier trains and evaluates successfully
- [ ] Both models are compared with loss/accuracy plots
- [ ] Error analysis identifies patterns in misclassifications
- [ ] At least one bonus challenge attempted
- [ ] Code is clean, well-organized, and runs end-to-end

:::tip[Wrapping Up Month 3]
You've gone from raw text preprocessing to building a Transformer from scratch and applying it to a real classification task. In Month 4, you'll build on this foundation by working with **pretrained Transformer models** — BERT, GPT, and the Hugging Face ecosystem — where the real power of these architectures becomes apparent.
:::

---

## Resources

- **[IMDB Dataset](https://huggingface.co/datasets/imdb)** _(tool)_ — The IMDB movie review dataset on Hugging Face — 50k reviews for binary sentiment classification.

- **[Text Classification with Transformers](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)** _(tutorial)_ — Official PyTorch tutorial on using Transformers for sequence modeling.

- **[BertViz](https://github.com/jessevig/bertviz)** _(tool)_ by Jesse Vig — Interactive tool for visualizing attention in Transformer models.

- **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** _(paper)_ by Devlin et al., 2019 — The BERT paper — the Transformer encoder approach to NLP that your project mirrors.

- **[A Survey on Text Classification](https://arxiv.org/abs/2008.00364)** _(paper)_ — Comprehensive survey covering classical and deep learning approaches to text classification.
