---
sidebar_position: 2
slug: rnns-lstms
title: "RNNs and LSTMs"
---


# Recurrent Neural Networks and LSTMs

:::info[What You'll Learn]
- Why sequential data needs special architectures
- Vanilla RNNs and the vanishing gradient problem
- LSTM gates (forget, input, output) and how they solve it
- GRUs as a simpler alternative
- Bidirectional and stacked RNNs
:::

:::note[Prerequisites]
[Neural Networks Introduction](/curriculum/month-1/neural-networks-intro) from Month 1.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~3 hours

Language, audio, and time series all share a defining property: **order matters**. The sentence "dog bites man" means something very different from "man bites dog." Standard feedforward networks treat inputs as fixed-size vectors with no notion of sequence — they cannot model this ordering. Recurrent Neural Networks (RNNs) were designed precisely for sequential data.

## Why Vanilla Neural Networks Fail on Sequences

A fully connected network receiving a sentence would need to:

1. Fix a maximum input length and pad shorter sequences.
2. Assign separate weights to each position, so the word "good" at position 3 uses entirely different parameters than "good" at position 7.
3. Lose the ability to generalize across positions.

This is both wasteful and brittle. We need an architecture that processes one element at a time, carries forward a **hidden state** that summarizes what it has seen, and shares parameters across time steps.

## RNN Architecture

At each time step \(t\), an RNN takes the current input \(x_t\) and the previous hidden state \(h_{t-1}\), and produces a new hidden state:

:::note[RNN Update Equation]

:::info[Plain English: What Does This Formula Mean?]
Think of an RNN like reading a book one word at a time. You keep a mental "summary" of the story so far (the hidden state). Each time you read a new word, you combine it with your current summary to update your understanding. The formula below describes exactly that: take the new word, mix it with your current summary, and squish the result through a function to get an updated summary.
:::

$$
h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)
$$

**Reading the formula:** *\(h_t\)* is the new hidden state (the updated summary after reading word *t*). *\(x_t\)* is the current input (the word at position *t*). *\(h_{t-1}\)* is the previous hidden state (summary of everything before word *t*). *\(W_{xh}\)* are weights that transform the input word. *\(W_{hh}\)* are weights that transform the previous summary. *\(b_h\)* is a bias term (a small constant for flexibility). *tanh* squishes the result to be between −1 and +1, keeping the numbers from growing too large.

The output at step \(t\) (if needed) is:

:::info[Plain English: What Does This Formula Mean?]
If you need to produce an answer at each step (like predicting the next word), you take your current summary and transform it into a prediction. It's like glancing at your notes and writing down your best guess for what comes next.
:::

$$
y_t = W_{hy}\, h_t + b_y
$$

**Reading the formula:** *\(y_t\)* is the output at time step *t*. *\(W_{hy}\)* are weights that convert the hidden state into an output. *\(h_t\)* is the current hidden state (summary). *\(b_y\)* is a bias term.

The **same** weight matrices \(W_{xh}\), \(W_{hh}\), and \(W_{hy}\) are shared across all time steps.
:::

### Unrolling Through Time

To train an RNN, we "unroll" it across all time steps, creating a computational graph that looks like a very deep feedforward network — one layer per time step. Backpropagation through this unrolled graph is called **Backpropagation Through Time (BPTT)**.

```python title="Manual RNN forward pass"
import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
seq_len = 5
batch_size = 3

W_xh = torch.randn(hidden_size, input_size)
W_hh = torch.randn(hidden_size, hidden_size)
b_h = torch.zeros(hidden_size)

x = torch.randn(seq_len, batch_size, input_size)
h = torch.zeros(batch_size, hidden_size)

outputs = []
for t in range(seq_len):
    h = torch.tanh(x[t] @ W_xh.T + h @ W_hh.T + b_h)
    outputs.append(h)

# Stack outputs: (seq_len, batch_size, hidden_size)
outputs = torch.stack(outputs)
print(outputs.shape)  # torch.Size([5, 3, 20])
```

:::tip[Line-by-Line Walkthrough]
- **`input_size = 10`** — Each input at each time step has 10 features (like a 10-dimensional word embedding).
- **`hidden_size = 20`** — The hidden state (the "memory") will be a vector of 20 numbers.
- **`seq_len = 5`** — We're processing a sequence of 5 time steps (like 5 words).
- **`batch_size = 3`** — We process 3 sequences at the same time for efficiency.
- **`W_xh = torch.randn(hidden_size, input_size)`** — Random weight matrix to transform inputs (20×10).
- **`W_hh = torch.randn(hidden_size, hidden_size)`** — Random weight matrix to transform the previous hidden state (20×20).
- **`b_h = torch.zeros(hidden_size)`** — Bias vector initialized to zeros.
- **`x = torch.randn(seq_len, batch_size, input_size)`** — Random input data: 5 time steps, 3 sequences, 10 features each.
- **`h = torch.zeros(batch_size, hidden_size)`** — Start with a blank memory (all zeros).
- **`for t in range(seq_len):`** — Loop through each time step, one at a time.
- **`h = torch.tanh(x[t] @ W_xh.T + h @ W_hh.T + b_h)`** — The RNN equation: multiply the input by its weights, multiply the old hidden state by its weights, add the bias, and squish through tanh to get the new hidden state.
- **`outputs = torch.stack(outputs)`** — Combine all hidden states into a single tensor.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `manual_rnn.py`
2. Run: `python manual_rnn.py`

**Expected output:**
```
torch.Size([5, 3, 20])
```

</details>

## The Vanishing Gradient Problem

When sequences are long, gradients must flow through many multiplications of \(W_{hh}\). If the largest singular value of \(W_{hh}\) is less than 1, gradients shrink exponentially — this is the **vanishing gradient** problem. The network effectively forgets information from early time steps.

:::warning[Vanishing vs. Exploding Gradients]
If the largest singular value is greater than 1, gradients **explode** instead. Gradient clipping mitigates explosions, but vanishing gradients require architectural changes — which is exactly what LSTMs and GRUs provide.
:::

Concretely, the gradient of the loss at time \(T\) with respect to the hidden state at time \(t\) involves:

:::info[Plain English: What Does This Formula Mean?]
Imagine a game of "telephone" (Chinese whispers) across many people. Each person slightly changes the message. After many people, the original message is completely lost or garbled. The same thing happens with gradients in an RNN: the learning signal from the end of a long sequence has to travel backward through many time steps, and at each step it gets multiplied by a number. If that number is less than 1, the signal shrinks and eventually vanishes (the network "forgets" the early words).
:::

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
$$

**Reading the formula:** *\(\partial h_T / \partial h_t\)* means "how much does the hidden state at the end (step *T*) change if we tweak the hidden state at step *t*?" The *\(\prod\)* (product symbol) means "multiply together" all the intermediate factors. Each factor *\(\partial h_k / \partial h_{k-1}\)* is how one step's hidden state depends on the previous step. If each factor is less than 1, multiplying many of them together gives a number that shrinks toward zero.

Each factor involves \(W_{hh}\) and the derivative of tanh (which saturates near ±1), causing the product to vanish for large \(T - t\).

## LSTM: Long Short-Term Memory

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem by introducing a **cell state** \(c_t\) — a highway that carries information across time steps with minimal interference — and three **gates** that control information flow.

:::info[The Three Gates]
1. **Forget gate** \(f_t\): decides what fraction of the old cell state to keep.
2. **Input gate** \(i_t\): decides what new information to write into the cell state.
3. **Output gate** \(o_t\): decides what part of the cell state to expose as the hidden state.

All gates use sigmoid activations, producing values in \([0, 1]\) that act as soft switches.
:::

:::note[LSTM Equations]

:::info[Plain English: What Does This Formula Mean?]
Think of an LSTM like a person carrying a notebook (the cell state). At each step, they can:
1. **Erase** parts of the notebook they no longer need (forget gate).
2. **Write** new notes (input gate + candidate).
3. **Read** parts of the notebook aloud to share with others (output gate).

The notebook travels through time largely untouched, which is why LSTMs can remember things from hundreds of steps ago — unlike a vanilla RNN where the memory gets rewritten at every step.
:::

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

**Reading the formula:**
- *\(f_t\)* — **Forget gate**: a number between 0 and 1 for each cell. 1 means "keep everything", 0 means "erase everything." *\(\sigma\)* is the sigmoid function that outputs values in [0, 1]. *\([h_{t-1}, x_t]\)* means "glue together the previous summary and the new input."
- *\(i_t\)* — **Input gate**: decides how much of the new candidate to add. Also between 0 and 1.
- *\(\tilde{c}_t\)* — **Candidate cell state**: a proposal for new information to write, squished between −1 and +1 by tanh.
- *\(c_t\)* — **Updated cell state**: erase old stuff (\(f_t \odot c_{t-1}\), where \(\odot\) means multiply element by element) then add new stuff (\(i_t \odot \tilde{c}_t\)).
- *\(o_t\)* — **Output gate**: decides what part of the cell state to reveal.
- *\(h_t\)* — **Hidden state**: the output — the cell state pushed through tanh, then filtered by the output gate.

Here \(\odot\) denotes element-wise multiplication and \([h_{t-1}, x_t]\) is concatenation.
:::

The cell state update \(c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\) is the key: when the forget gate is near 1 and the input gate is near 0, the cell state passes through unchanged, allowing gradients to flow across many time steps.

```python title="LSTM in PyTorch"
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

x = torch.randn(3, 15, 10)  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (3, 15, 20) — hidden state at every time step
print(h_n.shape)     # (2, 3, 20) — final hidden state per layer
print(c_n.shape)     # (2, 3, 20) — final cell state per layer
```

:::tip[Line-by-Line Walkthrough]
- **`nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)`** — Create a 2-layer LSTM: each input has 10 features, the hidden state is 20-dimensional, and `batch_first=True` means the batch dimension comes first in the input tensor.
- **`torch.randn(3, 15, 10)`** — Random input: 3 sequences, each 15 steps long, each step has 10 features.
- **`output, (h_n, c_n) = lstm(x)`** — Run the entire sequence through the LSTM. `output` contains the hidden state at every time step; `h_n` is the final hidden state from each layer; `c_n` is the final cell state (the "notebook") from each layer.
- **`output.shape` → `(3, 15, 20)`** — For each of the 3 sequences, at each of the 15 steps, we get a 20-dimensional hidden state.
- **`h_n.shape` → `(2, 3, 20)`** — Final hidden states: 2 layers × 3 sequences × 20 dimensions.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `lstm_example.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python lstm_example.py`

**Expected output:**
```
torch.Size([3, 15, 20])
torch.Size([2, 3, 20])
torch.Size([2, 3, 20])
```

</details>

## GRUs: Gated Recurrent Units

GRUs (Cho et al., 2014) simplify the LSTM by merging the cell state and hidden state into a single state vector and using only two gates:

- **Reset gate** \(r_t\): controls how much of the previous hidden state to forget when computing the candidate.
- **Update gate** \(z_t\): controls the trade-off between the old state and the new candidate (combining the forget and input gates from the LSTM).

:::note[GRU Equations]

:::info[Plain English: What Does This Formula Mean?]
A GRU is like a simplified version of the LSTM notebook. Instead of a separate notebook and summary, the GRU keeps just one summary. It has two knobs:
1. **Update gate** (\(z_t\)): "How much of my old summary should I keep vs. replace with new info?" (like a sliding scale between 0% and 100% new).
2. **Reset gate** (\(r_t\)): "Should I forget my old summary when creating the new candidate?" (like deciding whether to start fresh or build on existing knowledge).
:::

$$
\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t]) \\
r_t &= \sigma(W_r [h_{t-1}, x_t]) \\
\tilde{h}_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

**Reading the formula:**
- *\(z_t\)* — **Update gate**: a value between 0 and 1. Close to 1 means "mostly use the new candidate"; close to 0 means "mostly keep the old state."
- *\(r_t\)* — **Reset gate**: a value between 0 and 1. Close to 0 means "ignore the old state when creating the candidate" (fresh start); close to 1 means "use the old state fully."
- *\(\tilde{h}_t\)* — **Candidate state**: proposed new summary. The reset gate controls how much of the old summary is mixed in.
- *\(h_t\)* — **Final state**: a blend of the old state and the candidate, controlled by the update gate. *\((1 - z_t)\)* keeps part of the old; *\(z_t\)* brings in the new. *\(\odot\)* means element-wise multiplication.
:::

GRUs have fewer parameters than LSTMs and often perform comparably on many tasks. When in doubt, try both.

```python title="GRU in PyTorch"
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, h_n = gru(torch.randn(3, 15, 10))
print(output.shape)  # (3, 15, 20)
print(h_n.shape)     # (2, 3, 20) — no cell state, just hidden
```

:::tip[Line-by-Line Walkthrough]
- **`nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)`** — Create a 2-layer GRU. Same interface as the LSTM but simpler internally (no separate cell state).
- **`output, h_n = gru(torch.randn(3, 15, 10))`** — Run 3 sequences of 15 steps through the GRU. Notice only `h_n` is returned (no `c_n`), because GRUs don't have a separate cell state.
- **`output.shape` → `(3, 15, 20)`** — Hidden state at every time step for every sequence.
- **`h_n.shape` → `(2, 3, 20)`** — Final hidden states from each of the 2 layers.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `gru_example.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python gru_example.py`

**Expected output:**
```
torch.Size([3, 15, 20])
torch.Size([2, 3, 20])
```

</details>

## Building an RNN for Text Classification

Let's build an LSTM-based classifier for a simple sentiment analysis task. We'll use PyTorch's `nn.Embedding` to convert token indices to dense vectors.

```python title="LSTM Text Classifier"
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token ids
        embedded = self.dropout(self.embedding(x))
        output, (h_n, _) = self.lstm(embedded)

        # Concatenate final forward and backward hidden states
        forward_h = h_n[-2]   # last layer, forward
        backward_h = h_n[-1]  # last layer, backward
        combined = torch.cat([forward_h, backward_h], dim=1)

        return self.fc(self.dropout(combined))


model = TextClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=2)
x = torch.randint(0, 10000, (4, 50))  # batch of 4 sequences, length 50
print(model(x).shape)  # (4, 2)
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Embedding(vocab_size, embed_dim, padding_idx=0)`** — A lookup table: each of the 10,000 words gets its own 128-dimensional vector. Token 0 (padding) always maps to zeros.
- **`nn.LSTM(..., bidirectional=True)`** — Bidirectional LSTM: one reads left-to-right, another reads right-to-left. This lets the model use context from both directions.
- **`nn.Linear(hidden_dim * 2, num_classes)`** — Output layer: `hidden_dim * 2` because bidirectional doubles the hidden size. Maps to 2 class scores (e.g., positive vs. negative).
- **`self.dropout(self.embedding(x))`** — Look up word vectors, then randomly zero out some of them during training (regularization).
- **`h_n[-2]` / `h_n[-1]`** — With bidirectional, the final hidden states alternate: `h_n[-2]` is the forward direction's last hidden state, `h_n[-1]` is the backward direction's.
- **`torch.cat([forward_h, backward_h], dim=1)`** — Glue the forward and backward summaries side by side.
- **`torch.randint(0, 10000, (4, 50))`** — Fake input: 4 sentences, each 50 tokens, with random word IDs between 0 and 9999.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `text_classifier.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python text_classifier.py`

**Expected output:**
```
torch.Size([4, 2])
```

</details>

:::tip[Bidirectional RNNs]
Setting `bidirectional=True` runs two separate LSTMs: one reads left-to-right, the other right-to-left. Their hidden states are concatenated, giving the model context from both directions. This doubles the output dimension but often significantly improves accuracy for classification tasks.
:::

### Handling Variable-Length Sequences

Real text data has varying lengths. PyTorch provides `pack_padded_sequence` and `pad_packed_sequence` to avoid wasting computation on padding tokens.

```python title="Packed sequences"
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Suppose we have sequences of lengths [10, 7, 15, 3]
lengths = torch.tensor([10, 7, 15, 3])
# Sort by length (descending) for packing
sorted_lengths, sort_idx = lengths.sort(descending=True)
x_sorted = x[sort_idx]

embedded = model.embedding(x_sorted)
packed = pack_padded_sequence(embedded, sorted_lengths.cpu(), batch_first=True)
packed_output, (h_n, c_n) = model.lstm(packed)
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(output.shape)  # (4, 15, 512) — padded back to max length
```

:::tip[Line-by-Line Walkthrough]
- **`lengths = torch.tensor([10, 7, 15, 3])`** — Each sequence in the batch has a different true length (before padding).
- **`lengths.sort(descending=True)`** — PyTorch packing requires sequences sorted from longest to shortest.
- **`x[sort_idx]`** — Reorder the batch to match the sorted lengths.
- **`pack_padded_sequence(embedded, sorted_lengths.cpu(), batch_first=True)`** — Pack the padded sequences into a compact format that tells the LSTM to skip padding tokens. This saves computation and avoids polluting the hidden state with meaningless zeros.
- **`model.lstm(packed)`** — Run the LSTM on the packed sequence (it automatically handles variable lengths).
- **`pad_packed_sequence(packed_output, batch_first=True)`** — Unpack back to a padded tensor for downstream use. The output is padded to the length of the longest sequence (15).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. This code depends on the `TextClassifier` model and `x` defined above. Combine them in a single file, e.g. `packed_sequences.py`.
2. Run: `python packed_sequences.py`

**Expected output:**
```
torch.Size([4, 15, 512])
```

</details>

## Exercises

:::tip[Character-Level Language Model — intermediate]

Build a **character-level language model** using an LSTM. Train it on a text file of your choice (e.g., a book from Project Gutenberg). The model should:
1. Take a sequence of characters and predict the next character at each position.
2. Include a `generate` method that samples text autoregressively.

Try generating text after training for different numbers of epochs. How does quality improve?

<details>
<summary>Hints</summary>

1. Encode each character as an integer, then use nn.Embedding
2. Use a single-layer LSTM with hidden_size=128
3. The output at each step predicts the next character — use CrossEntropyLoss
4. Generate text by sampling from the softmax output and feeding back

</details>

:::

:::tip[LSTM vs GRU Comparison — beginner]

Train both an LSTM-based and a GRU-based text classifier on the same dataset. Compare:
- Number of parameters
- Training speed (time per epoch)
- Final accuracy

Which one would you choose for a production system and why?

<details>
<summary>Hints</summary>

1. Use the same hyperparameters (hidden_dim, num_layers, learning rate) for both
2. Track training loss and validation accuracy per epoch
3. Count parameters with sum(p.numel() for p in model.parameters())

</details>

:::

:::tip[Vanishing Gradient Visualization — advanced]

Empirically demonstrate the vanishing gradient problem. Create a vanilla RNN and an LSTM with the same hidden size. Pass a long sequence (100 time steps) through each, compute a loss from the final output, and backpropagate. Plot the gradient norms at each time step. How do they differ?

<details>
<summary>Hints</summary>

1. Use register_backward_hook on the RNN to capture gradients at each time step
2. Compare gradient magnitudes between a vanilla RNN and an LSTM
3. Use sequences of length 100+ to see the effect clearly

</details>

:::

## Resources

- **[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)** _(tutorial)_ by Christopher Olah — The definitive visual guide to LSTM internals. Beautiful diagrams of the gates and cell state.

- **[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)** _(tutorial)_ by Andrej Karpathy — Classic blog post showing what character-level RNNs can learn — from Shakespeare to LaTeX.

- **[Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)** _(paper)_ by Cho et al. — The paper that introduced GRUs and the encoder-decoder architecture for sequence-to-sequence tasks.

- **[Sequence Models (deeplearning.ai)](https://www.coursera.org/learn/nlp-sequence-models)** _(course)_ by Andrew Ng — Coursera course covering RNNs, LSTMs, GRUs, and attention mechanisms with hands-on assignments.
