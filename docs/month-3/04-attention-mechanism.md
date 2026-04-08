---
sidebar_position: 4
slug: attention-mechanism
title: "The Attention Mechanism"
---


# The Attention Mechanism

:::info[What You'll Learn]
- The information bottleneck problem in seq2seq
- How attention computes alignment scores
- Bahdanau (additive) vs. Luong (multiplicative) attention
- Visualizing and interpreting attention weights
:::

:::note[Prerequisites]
[Seq2Seq Models](seq2seq) from this month.
:::

**Estimated time:** Reading: ~35 min | Exercises: ~2 hours

In the previous lesson, we saw how seq2seq models compress an entire input sequence into a single fixed-length context vector. This creates an information bottleneck — long sequences lose critical details. The **attention mechanism** solves this by allowing the decoder to look back at *all* encoder hidden states and dynamically focus on the most relevant parts of the input at each decoding step.

Attention is arguably the single most important idea in modern deep learning. It powers Transformers, which power GPT, BERT, and every major language model today.

## Motivation: Why Attention?

Consider translating "The agreement on the European Economic Area was signed in August 1992" from English to French. When generating the French word for "European", the decoder needs to focus on that specific part of the input — not on "August" or "signed". But with a fixed context vector, the decoder has no way to selectively access different parts of the input.

:::info[The Core Idea of Attention]
Instead of compressing the input into one vector, **keep all encoder hidden states** and let the decoder learn which ones to focus on at each generation step. The decoder computes a weighted sum of encoder hidden states, where the weights reflect "how much attention" each input token deserves for the current output token.
:::

## Bahdanau Attention (Additive)

Bahdanau et al. (2015) introduced the first attention mechanism for seq2seq models. It's called **additive** because it uses a feed-forward neural network to compute attention scores.

### How It Works

At each decoder time step \( t \):

1. Take the decoder's previous hidden state \( s_{t-1} \) and every encoder hidden state \( h_j \).
2. Compute an **alignment score** \( e_{t,j} \) for each encoder state:

:::info[Plain English: What Does This Formula Mean?]
Think of the alignment score as the decoder asking each encoder word: "How relevant are you to what I'm about to say?" It combines what the decoder is thinking about (*s*) with what each input word represents (*h*), passes them through a small neural network, and gets a relevance score. Higher score = more relevant.
:::

\[
e_{t,j} = v^T \tanh(W_s \cdot s_{t-1} + W_h \cdot h_j)
\]

**Reading the formula:** *e_{t,j}* is the alignment score between decoder step *t* and encoder position *j*. *s_{t-1}* is what the decoder was thinking at the previous step. *h_j* is the encoder's representation of input word *j*. *W_s* and *W_h* are learned weight matrices that project both into the same space. *tanh* squashes the result to be between -1 and 1. *v^T* converts the result into a single number (the score).

3. Normalize scores with softmax to get **attention weights** \( \alpha_{t,j} \):

:::info[Plain English: What Does This Formula Mean?]
Softmax is like turning a set of scores into percentages that add up to 100%. If one input word scored much higher than the rest, it gets most of the attention (e.g., 80%), while the others share the remaining 20%.
:::

\[
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}
\]

**Reading the formula:** *α_{t,j}* is the attention weight — what fraction of the decoder's focus goes to input word *j* at step *t*. *exp(e_{t,j})* makes the score positive. Dividing by the sum of all scores ensures they add up to 1 (like a probability distribution). *T_x* is the length of the input sequence.

4. Compute the **context vector** as a weighted sum of encoder states:

:::info[Plain English: What Does This Formula Mean?]
The context vector is like a custom summary of the input, tailored for the current output word. If the decoder needs to translate "cat," it focuses heavily on the input word "cat" and barely pays attention to "the" or "sat." The result is a blended representation, weighted by relevance.
:::

\[
c_t = \sum_{j=1}^{T_x} \alpha_{t,j} \cdot h_j
\]

**Reading the formula:** *c_t* is the context vector at decoder step *t*. It's a weighted average of all encoder hidden states *h_j*, where the weights *α_{t,j}* determine how much each input word contributes. A word with a high attention weight contributes a lot; a word with low weight contributes almost nothing.

5. Concatenate the context vector with the decoder's input and pass it through the decoder RNN.

```python title="Bahdanau (additive) attention"
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: (B, decoder_dim)
        # encoder_outputs: (B, src_len, encoder_dim)

        # Project both to attention space
        encoder_proj = self.W_encoder(encoder_outputs)                 # (B, T, A)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)     # (B, 1, A)

        # Additive score with tanh activation
        energy = torch.tanh(encoder_proj + decoder_proj)               # (B, T, A)
        scores = self.v(energy).squeeze(-1)                            # (B, T)

        # Normalize to get attention weights
        attn_weights = F.softmax(scores, dim=1)                        # (B, T)

        # Weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)                # (B, encoder_dim)

        return context, attn_weights
```

:::tip[Line-by-Line Walkthrough]
- **`W_encoder` / `W_decoder`** — Two learned projection matrices that bring encoder and decoder representations into the same "comparison space." Like translating two different languages into a common language so they can be compared.
- **`encoder_proj + decoder_proj`** — Adds the projected encoder states and decoder state together. The `unsqueeze(1)` on the decoder lets it broadcast across all encoder positions (one decoder state compared to many encoder states).
- **`torch.tanh(...)`** — Squashes the combined values into the range [-1, 1], keeping things numerically stable.
- **`self.v(energy).squeeze(-1)`** — Reduces each attention-dim vector to a single score. Each input word now has one number saying "how relevant am I?"
- **`F.softmax(scores, dim=1)`** — Turns the raw scores into a probability distribution (all positive, sums to 1).
- **`torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)`** — Weighted sum: multiplies each encoder state by its attention weight and adds them up. The result is a single context vector tailored to the current decoder step.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `bahdanau_attention.py`
2. Add a test:
```python
attn = BahdanauAttention(64, 64, 32)
ctx, weights = attn(torch.randn(2, 64), torch.randn(2, 10, 64))
print(f"Context: {ctx.shape}, Weights: {weights.shape}")
```
3. Run: `python bahdanau_attention.py`

**Expected output:**
```
Context: torch.Size([2, 64]), Weights: torch.Size([2, 10])
```

</details>

## Luong Attention (Multiplicative)

Luong et al. (2015) proposed a simpler variant that computes alignment scores using a dot product — hence **multiplicative** attention.

### Three Score Functions

Luong defined three ways to compute the alignment score:

| Type | Formula | Notes |
|------|---------|-------|
| **Dot** | \( e_{t,j} = s_t^T h_j \) | Simplest, requires same dimensions |
| **General** | \( e_{t,j} = s_t^T W h_j \) | Learned projection, flexible dimensions |
| **Concat** | \( e_{t,j} = v^T \tanh(W[s_t; h_j]) \) | Similar to Bahdanau |

```python title="Luong attention (multiplicative)"
class LuongAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, method: str = "general"):
        super().__init__()
        self.method = method

        if method == "general":
            self.W = nn.Linear(encoder_dim, decoder_dim, bias=False)
        elif method == "concat":
            self.W = nn.Linear(encoder_dim + decoder_dim, decoder_dim, bias=False)
            self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: (B, decoder_dim)
        # encoder_outputs: (B, src_len, encoder_dim)

        if self.method == "dot":
            scores = torch.bmm(encoder_outputs,
                                decoder_hidden.unsqueeze(2)).squeeze(2)    # (B, T)

        elif self.method == "general":
            projected = self.W(encoder_outputs)                            # (B, T, D)
            scores = torch.bmm(projected,
                                decoder_hidden.unsqueeze(2)).squeeze(2)    # (B, T)

        elif self.method == "concat":
            decoder_expanded = decoder_hidden.unsqueeze(1).expand_as(
                encoder_outputs[:, :, :decoder_hidden.size(1)])
            combined = torch.cat([encoder_outputs, decoder_expanded], dim=2)
            scores = self.v(torch.tanh(self.W(combined))).squeeze(2)       # (B, T)

        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)

        return context, attn_weights
```

:::tip[Line-by-Line Walkthrough]
- **`self.method`** — Stores which scoring method to use: "dot" (simplest), "general" (with a learned transformation), or "concat" (similar to Bahdanau).
- **Dot method:** `torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2))` — Directly computes dot products between the decoder state and each encoder state. Like checking how much two arrows point in the same direction. Requires encoder and decoder dimensions to match.
- **General method:** `self.W(encoder_outputs)` — First transforms the encoder outputs through a learned matrix, then computes dot products. More flexible because it can compare vectors of different sizes.
- **Concat method:** Concatenates decoder and encoder states side by side, then passes through a neural network — similar to Bahdanau's approach.
- **`F.softmax(scores, dim=1)`** — Normalizes scores into attention weights (probabilities that sum to 1).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `luong_attention.py` (include the `import` statements from above)
2. Add a test:
```python
attn = LuongAttention(64, 64, method="general")
ctx, weights = attn(torch.randn(2, 64), torch.randn(2, 10, 64))
print(f"Context: {ctx.shape}, Weights: {weights.shape}")
```
3. Run: `python luong_attention.py`

**Expected output:**
```
Context: torch.Size([2, 64]), Weights: torch.Size([2, 10])
```

</details>

:::info[Bahdanau vs. Luong: Key Differences]
- **Bahdanau** uses the *previous* decoder state \(s_{t-1}\) to compute attention, then feeds the context into the current RNN step.
- **Luong** uses the *current* decoder state \(s_t\) (computed first without attention), then combines it with the context vector.
- **Luong dot** is the simplest and fastest but requires encoder and decoder to have the same hidden size.
- In practice, the differences are small. Luong's general attention is the most common choice.
:::

## Attention Weights Visualization

One of the most beautiful properties of attention is **interpretability**. By visualizing the attention weight matrix, you can see exactly which input tokens the decoder focused on when generating each output token.

```python title="Visualizing attention weights"
import matplotlib.pyplot as plt
import numpy as np

def plot_attention(attention_weights: np.ndarray,
                   source_tokens: list[str],
                   target_tokens: list[str],
                   title: str = "Attention Weights"):
    """Visualize attention as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(target_tokens, fontsize=11)

    ax.set_xlabel("Source (Input)")
    ax.set_ylabel("Target (Output)")
    ax.set_title(title)

    # Annotate each cell with the weight value
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            val = attention_weights[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()

# Simulated attention weights for English→French translation
source = ["The", "cat", "sat", "on", "the", "mat"]
target = ["Le", "chat", "était", "assis", "sur", "le", "tapis"]

np.random.seed(42)
weights = np.random.dirichlet(np.ones(len(source)), size=len(target))
# In a real model, "chat" would attend strongly to "cat", etc.
plot_attention(weights, source, target)
```

:::tip[Line-by-Line Walkthrough]
- **`ax.imshow(attention_weights, cmap="YlOrRd", ...)`** — Displays the attention weights as a colored grid (heatmap). Brighter/redder cells mean higher attention. Each row is an output word, each column is an input word.
- **`np.random.dirichlet(...)`** — Generates random attention weights where each row sums to 1 (simulating softmax output). In a real model these wouldn't be random.
- **`ax.text(j, i, f"{val:.2f}", ...)`** — Prints the actual weight value in each cell so you can read exact numbers, not just colors.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install matplotlib numpy
```

**Steps:**
1. Save to `attention_viz.py`
2. Run: `python attention_viz.py`
3. A heatmap window appears and `attention_heatmap.png` is saved.

**Expected output:** A 7×6 color-coded heatmap showing (random) attention weights between French and English words. Each cell shows a number between 0 and 1.

</details>

In a properly trained translation model, you'd see a roughly diagonal pattern — "Le" attends to "The", "chat" to "cat", etc. — with interesting deviations for word reordering between languages.

## The Math Behind Attention: Q, K, V

The attention mechanism can be generalized into a framework of **Queries**, **Keys**, and **Values**:

- **Query (Q)**: "What am I looking for?" — the decoder hidden state
- **Key (K)**: "What do I contain?" — each encoder hidden state
- **Value (V)**: "What information do I provide?" — also the encoder hidden states (often same as keys)

:::note[Generalized Attention]

:::info[Plain English: What Does This Formula Mean?]
Imagine you're at a library looking for information (that's your **query**). Each book on the shelf has a title (**key**) and contents (**value**). You compare your query to each title to decide which books are most relevant, then you read mostly from the best-matching books. Attention works the same way: the query asks "what am I looking for?", the keys answer "what do I contain?", and the values provide the actual information. The result is a smart blend of the most relevant information.
:::

Given a query \(\mathbf{q}\), a set of keys \(\mathbf{K} = [k_1, \ldots, k_n]\), and values \(\mathbf{V} = [v_1, \ldots, v_n]\):

\[
\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_{i=1}^{n} \alpha_i \mathbf{v}_i
\]

**Reading the formula:** The output is a weighted sum of all value vectors *v_i*. Each *α_i* is a weight between 0 and 1 (and all weights sum to 1). If *α_3* is 0.7 and the rest are small, the output is mostly *v_3* — the third value vector dominates.

where

\[
\alpha_i = \text{softmax}\left(\text{score}(\mathbf{q}, \mathbf{k}_i)\right) = \frac{\exp(\text{score}(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^{n}\exp(\text{score}(\mathbf{q}, \mathbf{k}_j))}
\]

**Reading the formula:** *α_i* is the attention weight for value *i*. *score(q, k_i)* is how well query *q* matches key *k_i* (higher = better match). The *exp* and division by the sum (softmax) converts raw scores into a probability distribution — ensuring all weights are positive and sum to 1.

The score function can be dot product, general (bilinear), or additive — corresponding to the variants we've seen above.
:::

This Q/K/V abstraction is exactly what Transformers use. In the next lesson, you'll see how **self-attention** applies this framework with the query, key, and value all coming from the same sequence.

```python title="Q, K, V attention in code"
def qkv_attention(query: torch.Tensor, keys: torch.Tensor,
                   values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    General Q, K, V attention.
    query:  (B, 1, D)  — single query
    keys:   (B, T, D)  — T key vectors
    values: (B, T, D)  — T value vectors
    """
    d_k = keys.shape[-1]

    # Scaled dot-product scores
    scores = torch.bmm(query, keys.transpose(1, 2)) / (d_k ** 0.5)  # (B, 1, T)
    weights = F.softmax(scores, dim=-1)                               # (B, 1, T)
    context = torch.bmm(weights, values)                              # (B, 1, D)

    return context.squeeze(1), weights.squeeze(1)

# Example
B, T, D = 2, 10, 64
query = torch.randn(B, 1, D)
keys = torch.randn(B, T, D)
values = torch.randn(B, T, D)

ctx, w = qkv_attention(query, keys, values)
print(f"Context shape: {ctx.shape}")      # (2, 64)
print(f"Weights shape: {w.shape}")        # (2, 10)
print(f"Weights sum:   {w.sum(dim=1)}")   # [1.0, 1.0]
```

:::tip[Line-by-Line Walkthrough]
- **`torch.bmm(query, keys.transpose(1, 2))`** — Computes dot products between the query and every key. The result is a score for each key, telling us "how well does this key match the query?"
- **`/ (d_k ** 0.5)`** — Divides by the square root of the key dimension. Without this, high-dimensional dot products produce very large numbers, pushing softmax to produce extreme (near 0 or 1) weights. This scaling keeps gradients healthy.
- **`F.softmax(scores, dim=-1)`** — Converts raw scores into attention weights (probabilities). The highest-scoring key gets the most weight.
- **`torch.bmm(weights, values)`** — Computes the weighted sum of values using the attention weights. This is the final "attended" output.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save to `qkv_attention.py` (include `import torch` and `import torch.nn.functional as F`)
2. Run: `python qkv_attention.py`

**Expected output:**
```
Context shape: torch.Size([2, 64])
Weights shape: torch.Size([2, 10])
Weights sum:   tensor([1.0000, 1.0000])
```

</details>

Notice the **scaling factor** \(\sqrt{d_k}\) — this prevents dot products from growing too large in high dimensions, which would push softmax into regions with vanishingly small gradients. This is **scaled dot-product attention**, the exact formulation used in Transformers.

## Putting It Together: Seq2Seq with Attention

```python title="Decoder with attention"
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim, hidden_dim, hidden_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor):
        # input_token: (B,)
        # hidden: (1, B, H)
        # encoder_outputs: (B, T, H)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # (B, 1, E)

        context, attn_weights = self.attention(
            hidden.squeeze(0), encoder_outputs)                           # (B, H), (B, T)

        # Concatenate embedded input with context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)    # (B, 1, E+H)
        output, hidden = self.rnn(rnn_input, hidden)

        # Predict next token
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )

        return prediction, hidden, attn_weights
```

:::tip[Line-by-Line Walkthrough]
- **`self.attention(hidden.squeeze(0), encoder_outputs)`** — Before the decoder generates the next word, it first "looks back" at the entire input through attention. The decoder's current state (what it's thinking about) is compared to every encoder state to compute a context vector.
- **`torch.cat([embedded, context.unsqueeze(1)], dim=2)`** — Glues together the current word embedding and the attention context vector. The GRU gets both "what word I just saw" and "what parts of the input are relevant right now."
- **`self.fc_out(torch.cat([output, context, embedded], dim=1))`** — The final prediction uses three pieces of information: the GRU output, the attention context, and the embedded input — giving the model multiple perspectives to make a better guess.
- **`return prediction, hidden, attn_weights`** — Returns the word prediction, updated hidden state, and the attention weights (useful for visualization).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save `BahdanauAttention` and `AttentionDecoder` together in `attention_decoder.py`
2. Add a test:
```python
dec = AttentionDecoder(100, 32, 64)
pred, h, w = dec(torch.tensor([1, 2]), torch.randn(1, 2, 64), torch.randn(2, 10, 64))
print(f"Prediction: {pred.shape}, Weights: {w.shape}")
```
3. Run: `python attention_decoder.py`

**Expected output:**
```
Prediction: torch.Size([2, 100]), Weights: torch.Size([2, 10])
```

</details>

---

## Exercises

:::tip[Exercise 1: Attention Visualization — beginner]

Train a seq2seq model with Bahdanau attention on the number reversal task from the previous lesson. At inference time, collect the attention weights at each decoding step and visualize them as a heatmap. What pattern do you expect to see for sequence reversal?

<details>
<summary>Hints</summary>

1. Store attention weights from each decoder step in a list
2. Stack them into a matrix of shape (trg_len, src_len)
3. Use the plot_attention function from above

</details>

:::

:::tip[Exercise 2: Comparing Attention Variants — intermediate]

Implement all four attention scoring functions (Bahdanau additive, Luong dot, Luong general, Luong concat) and compare them on a small translation dataset (e.g., English→Pig Latin or English→number sequences). Which converges fastest? Which achieves the highest accuracy?

<details>
<summary>Hints</summary>

1. Implement Luong dot, general, and concat alongside Bahdanau
2. Train all four on the same data with the same hyperparameters
3. Compare convergence speed (loss vs. epoch) and final accuracy
4. Measure wall-clock time per epoch

</details>

:::

:::tip[Exercise 3: Attention Is Not Explanation — advanced]

Attention weights are often interpreted as "importance scores," but this can be misleading. Design an experiment: train a sentiment classifier with attention, then (a) randomly shuffle attention weights and measure accuracy change, and (b) find alternative attention distributions that produce the same prediction. Write up your findings on whether attention weights reliably indicate feature importance.

<details>
<summary>Hints</summary>

1. Read the Jain & Wallace (2019) paper 'Attention is not Explanation'
2. Randomly permute attention weights and measure prediction change
3. Train an alternative attention distribution that produces the same output
4. Consider: does high attention weight mean a token was important for the prediction?

</details>

:::

---

## Resources

- **[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)** _(paper)_ by Bahdanau, Cho, Bengio (2015) — The paper that introduced attention for seq2seq — one of the most cited papers in deep learning.

- **[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)** _(paper)_ by Luong, Pham, Manning (2015) — Introduced multiplicative attention variants (dot, general, concat).

- **[Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)** _(tutorial)_ by Lilian Weng — Comprehensive overview of attention mechanisms with clear math and diagrams.

- **[The Illustrated Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)** _(tutorial)_ by Jay Alammar — Visual guide to how attention works in encoder-decoder models.

- **[Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)** _(tutorial)_ by Chris Olah & Shan Carter — Interactive Distill article exploring attention and other neural network augmentations.
