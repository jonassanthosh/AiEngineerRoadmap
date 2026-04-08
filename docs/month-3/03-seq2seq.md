---
sidebar_position: 3
slug: seq2seq
title: "Sequence-to-Sequence Models"
---


# Sequence-to-Sequence Models

Many of the most interesting NLP tasks involve transforming one sequence into another: translating English to French, summarizing a paragraph into a sentence, or converting a question into an answer. **Sequence-to-sequence (seq2seq)** models provide the foundational architecture for these tasks — and understanding them is essential before tackling attention and Transformers.

## The Encoder-Decoder Architecture

A seq2seq model has two components:

1. **Encoder** — reads the input sequence token by token and compresses it into a fixed-length vector called the **context vector** (or **thought vector**).
2. **Decoder** — receives the context vector and generates the output sequence token by token.

Both the encoder and decoder are typically **recurrent neural networks** (RNNs, LSTMs, or GRUs).

```
Input: "I love cats"
         ↓
   ┌─────────────┐
   │   Encoder    │  → reads tokens left to right
   │  (LSTM/GRU)  │
   └──────┬──────┘
          │
    context vector (h)
          │
   ┌──────┴──────┐
   │   Decoder    │  → generates tokens left to right
   │  (LSTM/GRU)  │
   └─────────────┘
         ↓
Output: "J'aime les chats"
```

:::info[How the Encoder Works]
The encoder processes the input sequence one token at a time. At each step, it updates its hidden state. The **final hidden state** becomes the context vector — a compressed representation of the entire input meaning.

For an LSTM encoder, the context vector is actually a tuple \((h_T, c_T)\) — both the hidden state and the cell state from the final time step.
:::

```python title="Encoder implementation"
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))   # (B, T, E)
        outputs, (hidden, cell) = self.rnn(embedded)   # hidden: (layers, B, H)
        return hidden, cell
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Embedding(vocab_size, embed_dim)`** — A lookup table that converts word indices into dense vectors. Like giving each word a unique, compact ID card of numbers.
- **`nn.LSTM(embed_dim, hidden_dim, ...)`** — The LSTM reads the embedded words one at a time, updating an internal "memory" (hidden state) at each step. Think of it as reading a sentence word by word and building up understanding.
- **`self.dropout(self.embedding(src))`** — Randomly zeroes out some embedding values during training to prevent overfitting (like randomly covering some words while studying to build robustness).
- **`return hidden, cell`** — Returns the final "memory" of the encoder after reading the entire input. This compressed summary is passed to the decoder. The LSTM has two types of memory: `hidden` (short-term) and `cell` (long-term).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `encoder.py`
2. Add a test: `enc = Encoder(100, 32, 64); print(enc(torch.randint(0, 100, (2, 10)))[0].shape)`
3. Run: `python encoder.py`

**Expected output:**
```
torch.Size([1, 2, 64])
```

</details>

```python title="Decoder implementation"
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_token: (batch_size,) — single token
        input_token = input_token.unsqueeze(1)                  # (B, 1)
        embedded = self.dropout(self.embedding(input_token))    # (B, 1, E)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))             # (B, vocab_size)
        return prediction, hidden, cell
```

:::tip[Line-by-Line Walkthrough]
- **`input_token.unsqueeze(1)`** — The decoder processes one word at a time. `unsqueeze` adds a "time" dimension so the LSTM receives the expected shape (batch, 1 time step, features).
- **`self.rnn(embedded, (hidden, cell))`** — Feeds the current word embedding plus the previous memory state into the LSTM. The LSTM updates its memory and produces an output. It's like the decoder saying "Given what I remember so far and this new word, here's my updated understanding."
- **`self.fc_out(output.squeeze(1))`** — A linear layer that converts the LSTM's output into a score for every word in the vocabulary. The highest score indicates the decoder's best guess for the next word.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save both the Encoder and Decoder classes to `seq2seq_modules.py`
2. Add a test: `dec = Decoder(100, 32, 64); pred, h, c = dec(torch.tensor([1, 2]), torch.randn(1, 2, 64), torch.randn(1, 2, 64)); print(pred.shape)`
3. Run: `python seq2seq_modules.py`

**Expected output:**
```
torch.Size([2, 100])
```

</details>

## Teacher Forcing

During training, the decoder generates one token at a time. At each step, it could use either:
- **Its own previous prediction** (autoregressive), or
- **The actual ground-truth token** from the target sequence.

Using the ground-truth token is called **teacher forcing** — it dramatically speeds up training because errors don't compound.

:::warning[The Teacher Forcing Trade-off]
With 100% teacher forcing, the decoder never learns to recover from its own mistakes. At inference time (when ground-truth tokens aren't available), small errors can cascade. The common solution is **scheduled sampling**: gradually reduce the teacher forcing ratio during training so the model learns to handle its own imperfect outputs.
:::

```python title="Seq2Seq model with teacher forcing"
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src: (B, src_len), trg: (B, trg_len)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        hidden, cell = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = prediction

            # Teacher forcing: use ground truth or predicted token
            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t]       # ground truth
            else:
                input_token = prediction.argmax(dim=1)  # model's prediction

        return outputs
```

:::tip[Line-by-Line Walkthrough]
- **`hidden, cell = self.encoder(src)`** — The encoder reads the entire input sentence and compresses it into a "context vector" (the hidden and cell states). Like reading a letter and summarizing it in your head before passing the summary to a friend.
- **`input_token = trg[:, 0]`** — The decoder starts with the "start of sentence" token — it's like saying "Begin your translation now."
- **`for t in range(1, trg_len):`** — The decoder generates one word at a time in a loop, just like writing a translation word by word.
- **`if random.random() < teacher_forcing_ratio:`** — During training, with 50% probability we give the decoder the correct answer (teacher forcing) and 50% we let it use its own prediction. It's like a student sometimes getting hints and sometimes working independently.
- **`prediction.argmax(dim=1)`** — Picks the word with the highest predicted score as the next input.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the Encoder, Decoder, and Seq2Seq classes together in `seq2seq.py`
2. Add a test:
```python
enc = Encoder(100, 32, 64)
dec = Decoder(100, 32, 64)
model = Seq2Seq(enc, dec)
src = torch.randint(0, 100, (2, 10))
trg = torch.randint(0, 100, (2, 8))
out = model(src, trg)
print(out.shape)  # (2, 8, 100)
```
3. Run: `python seq2seq.py`

**Expected output:**
```
torch.Size([2, 8, 100])
```

</details>

## The Information Bottleneck Problem

The entire input sequence is compressed into a single fixed-length context vector. This works for short sequences, but for long ones, critical information gets lost.

:::info[Why Fixed-Length Context Fails]
Imagine compressing a 100-word paragraph into a single 256-dimensional vector. The encoder must decide *which* information to keep and which to discard — and it must make this decision before it knows what the decoder will need. For machine translation, a sentence like "The old man who lived by the river caught a large fish yesterday" requires the decoder to access specific details (who? where? what? when?) at different generation steps.
:::

This is exactly the problem that **attention mechanisms** solve (next lesson). But first, let's build a working seq2seq model.

## Building a Simple Seq2Seq: Number Reversal

Let's train a seq2seq model on a toy task: reversing sequences of digits. This is simple enough to train quickly but complex enough to validate our architecture.

```python title="Complete seq2seq training on sequence reversal"
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Special tokens
PAD, SOS, EOS = 0, 1, 2
DIGITS_OFFSET = 3

def generate_data(num_samples: int, seq_len: int = 5):
    """Generate (input, reversed output) pairs of digit sequences."""
    pairs = []
    for _ in range(num_samples):
        seq = [random.randint(0, 9) for _ in range(seq_len)]
        src = [SOS] + [d + DIGITS_OFFSET for d in seq] + [EOS]
        trg = [SOS] + [d + DIGITS_OFFSET for d in reversed(seq)] + [EOS]
        pairs.append((src, trg))
    return pairs

# Hyperparameters
VOCAB_SIZE = 13  # PAD, SOS, EOS, 0-9
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_EPOCHS = 50
BATCH_SIZE = 64

# Generate data
train_data = generate_data(5000)
test_data = generate_data(500)

# Create model
encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

def make_batch(data, start, batch_size):
    batch = data[start:start + batch_size]
    src = torch.tensor([p[0] for p in batch])
    trg = torch.tensor([p[1] for p in batch])
    return src, trg

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    random.shuffle(train_data)
    total_loss = 0

    for i in range(0, len(train_data), BATCH_SIZE):
        src, trg = make_batch(train_data, i, BATCH_SIZE)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)

        # Reshape for loss: (B * T, vocab) vs (B * T,)
        output = output[:, 1:].reshape(-1, VOCAB_SIZE)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for src_seq, trg_seq in test_data[:100]:
        src = torch.tensor([src_seq])
        trg = torch.tensor([trg_seq])
        output = model(src, trg, teacher_forcing_ratio=0.0)
        predicted = output.argmax(dim=2)[0, 1:-1].tolist()
        expected = trg_seq[1:-1]
        if predicted == expected:
            correct += 1

print(f"Accuracy: {correct}/100 = {correct}%")
```

:::tip[Line-by-Line Walkthrough]
- **`generate_data(5000)`** — Creates 5,000 training examples. Each example is a random sequence of digits and its reverse. For instance, [3, 7, 1] → [1, 7, 3]. Special tokens SOS (start) and EOS (end) are added to mark sentence boundaries.
- **`CrossEntropyLoss(ignore_index=PAD)`** — Measures how wrong the model's predictions are. `ignore_index=PAD` tells it to skip padding tokens when calculating the loss.
- **`output[:, 1:].reshape(-1, VOCAB_SIZE)`** — We skip the first time step (since it's just the SOS token) and flatten everything for the loss function.
- **`clip_grad_norm_(model.parameters(), 1.0)`** — Prevents exploding gradients by capping gradient magnitudes. Like putting guardrails on a winding road — it keeps training from going off the rails.
- **`teacher_forcing_ratio=0.0`** — During evaluation, no teacher forcing: the model must rely entirely on its own predictions, simulating real-world usage.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save all classes (Encoder, Decoder, Seq2Seq) and this training code together in `seq2seq_train.py`
2. Run: `python seq2seq_train.py`
3. Training takes about 1-2 minutes on CPU.

**Expected output:**
```
Epoch 10/50, Loss: 12.3456
Epoch 20/50, Loss: 4.5678
Epoch 30/50, Loss: 1.2345
Epoch 40/50, Loss: 0.4567
Epoch 50/50, Loss: 0.1234
Accuracy: 85/100 = 85%
```
(Exact numbers will vary. Accuracy should be 70-95%.)

</details>

:::tip[Gradient Clipping]
RNNs are prone to exploding gradients. Always use `clip_grad_norm_` during training. A max norm of 1.0 is a reasonable default.
:::

## Greedy Decoding vs. Beam Search

At inference time, there are two strategies for generating output:

**Greedy decoding** picks the highest-probability token at each step. It's fast but can miss better overall sequences.

**Beam search** keeps track of the top-*k* partial sequences (beams) at each step and expands all of them. It produces higher-quality output at the cost of *k*× more computation.

```python title="Greedy decoding"
def greedy_decode(model, src: torch.Tensor, max_len: int = 20):
    """Generate output sequence using greedy decoding."""
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        input_token = torch.tensor([SOS])
        result = []

        for _ in range(max_len):
            prediction, hidden, cell = model.decoder(input_token, hidden, cell)
            top_token = prediction.argmax(dim=1)
            if top_token.item() == EOS:
                break
            result.append(top_token.item())
            input_token = top_token

    return result

# Test
src = torch.tensor([[SOS, 3+3, 7+3, 1+3, 4+3, 9+3, EOS]])  # 3 7 1 4 9
result = greedy_decode(model, src)
print(f"Input:    {[t - DIGITS_OFFSET for t in src[0, 1:-1].tolist()]}")
print(f"Reversed: {[t - DIGITS_OFFSET for t in result]}")
```

:::tip[Line-by-Line Walkthrough]
- **`hidden, cell = model.encoder(src)`** — Encode the input sequence to get the context vector. The decoder will use this as its starting memory.
- **`input_token = torch.tensor([SOS])`** — Start the decoder with the "start of sentence" token, like saying "ready, go!"
- **`prediction.argmax(dim=1)`** — At each step, the decoder picks the single most likely next word (greedy = always pick the best-looking option right now).
- **`if top_token.item() == EOS: break`** — Stop generating when the model outputs the "end of sentence" token.
- **`input_token = top_token`** — Feed the model's own prediction back as the next input, creating an autoregressive loop.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. This code should be appended to the training script (after the model is trained)
2. Run the full script: `python seq2seq_train.py`

**Expected output:**
```
Input:    [3, 7, 1, 4, 9]
Reversed: [9, 4, 1, 7, 3]
```
(If the model is well-trained, the output should be the input reversed.)

</details>

---

## Exercises

:::tip[Exercise 1: Sequence Sorting — beginner]

Modify the number reversal task to instead **sort** sequences of digits. Train the seq2seq model and report accuracy on a test set. How does performance change as sequence length increases from 5 to 10 to 15?

<details>
<summary>Hints</summary>

1. Reuse the same architecture but change the training data
2. The target is the sorted version of the input sequence
3. Start with short sequences (length 4-5)

</details>

:::

:::tip[Exercise 2: Scheduled Sampling — intermediate]

Implement **scheduled sampling** — a training strategy where the teacher forcing ratio starts at 1.0 and gradually decreases to 0.0 over training. Compare three decay schedules (linear, exponential, inverse sigmoid) and plot training loss curves for each. Which one gives the best test accuracy?

<details>
<summary>Hints</summary>

1. Start with teacher_forcing_ratio=1.0 and decay it each epoch
2. Try linear decay: ratio = 1.0 - (epoch / total_epochs)
3. Try exponential decay: ratio = 0.99 ** epoch
4. Compare final accuracy for each strategy

</details>

:::

:::tip[Exercise 3: Beam Search Decoder — advanced]

Implement **beam search** decoding for the seq2seq model. Compare the output quality (accuracy on the reversal task) of greedy decoding vs. beam search with beam widths of 2, 3, and 5. Does beam search help on this toy task?

<details>
<summary>Hints</summary>

1. Maintain a list of (sequence, score, hidden_state, cell_state) tuples
2. At each step, expand each beam by top-k tokens
3. Prune to keep only the best beam_width candidates
4. Normalize scores by sequence length to avoid preferring short sequences

</details>

:::

---

## Resources

- **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)** _(paper)_ by Sutskever, Vinyals, Le (2014) — The foundational seq2seq paper from Google, applying LSTMs to machine translation.

- **[The Illustrated Seq2Seq](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)** _(tutorial)_ by Jay Alammar — Visual walkthrough of encoder-decoder models with and without attention.

- **[PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)** _(tutorial)_ — Official PyTorch tutorial building a seq2seq model for translation.

- **[Scheduled Sampling for Sequence Prediction](https://arxiv.org/abs/1506.03099)** _(paper)_ by Bengio et al., 2015 — The paper introducing scheduled sampling to bridge training and inference.
