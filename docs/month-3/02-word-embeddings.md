---
sidebar_position: 2
slug: word-embeddings
title: "Word Embeddings"
---


# Word Embeddings

In the previous lesson we turned text into tokens. But models need **numbers**, not strings. How do we represent words as vectors in a way that captures their meaning? Word embeddings are the answer — and they're one of the most important ideas in modern NLP.

## The Problem: Representing Words as Numbers

### One-Hot Encoding

The simplest approach is to assign each word a unique index and represent it as a binary vector with a 1 at that index and 0s everywhere else.

```python title="One-hot encoding"
import numpy as np

vocab = ["king", "queen", "man", "woman", "child"]
word_to_idx = {w: i for i, w in enumerate(vocab)}

def one_hot(word: str) -> np.ndarray:
    vec = np.zeros(len(vocab))
    vec[word_to_idx[word]] = 1.0
    return vec

king = one_hot("king")
queen = one_hot("queen")

print(f"king:  {king}")
print(f"queen: {queen}")
print(f"Dot product (king · queen): {np.dot(king, queen)}")
# Dot product = 0.0 — no relationship captured at all
```

:::tip[Line-by-Line Walkthrough]
- **`word_to_idx = {w: i for i, w in enumerate(vocab)}`** — Creates a dictionary mapping each word to a number: "king"→0, "queen"→1, etc. Think of it like assigning each word a locker number.
- **`np.zeros(len(vocab))`** — Creates a vector of all zeros, as long as the vocabulary. Each word gets its own slot.
- **`vec[word_to_idx[word]] = 1.0`** — Puts a 1 in the slot for this word and leaves everything else as 0. So "king" = [1, 0, 0, 0, 0] and "queen" = [0, 1, 0, 0, 0].
- **`np.dot(king, queen)`** — The dot product measures similarity. Since king and queen have 1s in different positions, the result is 0 — the encoding treats them as completely unrelated.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy
```

**Steps:**
1. Save to `one_hot.py`
2. Run: `python one_hot.py`

**Expected output:**
```
king:  [1. 0. 0. 0. 0.]
queen: [0. 1. 0. 0. 0.]
Dot product (king · queen): 0.0
```

</details>

:::warning[One-Hot Encoding Failures]
One-hot vectors have three critical problems:
1. **No semantic similarity** — "king" and "queen" are as distant as "king" and "refrigerator".
2. **Massive dimensionality** — a 50,000-word vocabulary requires 50,000-dimensional vectors.
3. **No generalization** — learning about "king" tells the model nothing about "queen".
:::

## Word2Vec

In 2013, Mikolov et al. at Google introduced **Word2Vec**, which learns dense, low-dimensional vectors (typically 100–300 dimensions) from large text corpora. The key insight: **a word is characterized by the company it keeps** (the distributional hypothesis).

Word2Vec comes in two flavors:

### CBOW (Continuous Bag of Words)

CBOW predicts the **center word** from its surrounding context words. Given the context ["the", "cat", "on", "the", "mat"], predict "sat".

### Skip-gram

Skip-gram does the reverse: given the **center word**, predict the surrounding context. Given "sat", predict ["the", "cat", "on", "the", "mat"].

:::info[Skip-gram vs CBOW]
- **CBOW** is faster to train and works better for frequent words.
- **Skip-gram** works better with small datasets and for rare words.
- In practice, Skip-gram with negative sampling is the most popular choice.
:::

```python title="Training Word2Vec with Gensim"
from gensim.models import Word2Vec

# Simple training corpus (in practice, use millions of sentences)
sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "kingdom"],
    ["a", "man", "and", "a", "woman", "walked"],
    ["the", "prince", "is", "the", "son", "of", "the", "king"],
    ["the", "princess", "is", "the", "daughter", "of", "the", "queen"],
    ["a", "boy", "becomes", "a", "man"],
    ["a", "girl", "becomes", "a", "woman"],
    ["the", "king", "and", "queen", "have", "a", "son"],
    ["the", "man", "walked", "to", "the", "kingdom"],
    ["the", "woman", "walked", "to", "the", "kingdom"],
]

model = Word2Vec(
    sentences,
    vector_size=50,    # embedding dimension
    window=3,          # context window size
    min_count=1,       # minimum word frequency
    sg=1,              # 1 = Skip-gram, 0 = CBOW
    epochs=200,        # more epochs for small data
)

# Find most similar words
print("Most similar to 'king':")
for word, score in model.wv.most_similar("king", topn=5):
    print(f"  {word}: {score:.3f}")
```

:::tip[Line-by-Line Walkthrough]
- **`Word2Vec(sentences, ...)`** — Trains a Word2Vec model on our small corpus. Think of it as teaching the model which words tend to appear near each other.
- **`vector_size=50`** — Each word will be represented as a list of 50 numbers (instead of one-hot's thousands of numbers). These 50 numbers capture the word's meaning.
- **`window=3`** — When learning, the model looks 3 words to the left and right of each word. Like reading a sentence and noticing which words are near each other.
- **`sg=1`** — Uses the Skip-gram algorithm: given a word, predict its neighbors (as opposed to CBOW which does the reverse).
- **`model.wv.most_similar("king", topn=5)`** — Finds the 5 words whose vectors are closest to "king" in the learned space.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install gensim
```

**Steps:**
1. Save to `train_word2vec.py`
2. Run: `python train_word2vec.py`

**Expected output:** A list of the 5 words most similar to "king" with similarity scores. With this tiny corpus, results will vary — you'll likely see "queen" and "kingdom" near the top.

</details>

### The Famous Analogy: king - man + woman ≈ queen

Word2Vec embeddings capture **linear relationships** between concepts. The vector arithmetic `king - man + woman` produces a vector closest to `queen`.

```python title="Word embedding arithmetic"
# With a properly trained model on large data, this works:
# result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
# → [('queen', 0.87)]

# Let's demonstrate with pretrained vectors
import gensim.downloader as api

# Download pretrained GloVe vectors (this may take a moment)
glove = api.load("glove-wiki-gigaword-50")

result = glove.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
print("king - man + woman =")
for word, score in result:
    print(f"  {word}: {score:.4f}")
# Expected: queen, monarch, princess
```

:::tip[Line-by-Line Walkthrough]
- **`api.load("glove-wiki-gigaword-50")`** — Downloads pretrained GloVe word vectors trained on Wikipedia and Gigaword (billions of words). Each word is a 50-dimensional vector. The first download is ~66 MB.
- **`glove.most_similar(positive=["king", "woman"], negative=["man"], topn=3)`** — Performs vector arithmetic: takes the vector for "king," adds "woman," subtracts "man," and finds the 3 closest words to the result. It's like asking: "What is to woman as king is to man?"
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install gensim
```

**Steps:**
1. Save to `word_arithmetic.py`
2. Run: `python word_arithmetic.py`
3. The first run downloads GloVe vectors (~66 MB).

**Expected output:**
```
king - man + woman =
  queen: 0.8524
  monarch: 0.7140
  princess: 0.7104
```

</details>

:::note[Skip-gram Objective]

:::info[Plain English: What Does This Formula Mean?]
Imagine you're reading a book and you cover up one word. Skip-gram is like a game: given the uncovered word, can you guess which words are nearby? The formula below measures how good the model is at this guessing game across the entire text. A higher score means the model has learned that "king" tends to appear near "crown" and "throne," not near "bicycle" and "pizza."
:::

The Skip-gram model maximizes the probability of context words given a center word. For a center word \( w_t \) and window size \( c \):

\[
J(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, \, j \neq 0} \log P(w_{t+j} \mid w_t)
\]

**Reading the formula:** *J(θ)* is the model's overall score (higher = better). *T* is the total number of words in the text. *w_t* is the center word at position *t*. *j* ranges from *-c* to *c* (skipping 0), meaning we look at *c* words to the left and *c* words to the right. *P(w_{t+j} | w_t)* is the probability the model assigns to the neighbor word *w_{t+j}* given the center word *w_t*. The *log* turns multiplication into addition, making the math easier.

where \( P(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})} \)

**Reading the formula:** *P(w_O | w_I)* is the probability of seeing output word *w_O* near input word *w_I*. *v'_{w_O}* and *v_{w_I}* are the vector representations (embeddings) of the output and input words. The dot product *v' · v* measures how similar two vectors are. The *exp(...)* makes everything positive, and dividing by the sum over all *V* vocabulary words turns it into a probability that adds up to 1.

The denominator sums over the entire vocabulary — this is expensive, which is why **negative sampling** approximates it by only updating a small number of "negative" (random) words.
:::

## GloVe: Global Vectors

**GloVe** (Pennington et al., 2014) takes a different approach. Instead of predicting context words from a sliding window, GloVe builds a **co-occurrence matrix** of the entire corpus and then factorizes it.

:::info[GloVe's Key Idea]

:::info[Plain English: What Does This Formula Mean?]
Imagine you have a giant table showing how often every pair of words appears together in the same sentence. GloVe tries to learn word vectors so that the dot product of two word vectors matches how often those words actually appeared together. It's like arranging words on a map so that words that hang out together (like "ice" and "cream") end up close, while words that rarely co-occur (like "ice" and "volcano") end up far apart.
:::

If words \(i\) and \(j\) co-occur frequently, their dot product should be high. GloVe optimizes:

\[
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^T \mathbf{w}_j + b_i + b_j - \log X_{ij} \right)^2
\]

**Reading the formula:** *J* is the total error we want to minimize. *V* is the vocabulary size. *X_{ij}* is the number of times word *i* and word *j* appeared near each other. *w_i* and *w_j* are the learned vectors for words *i* and *j*. *b_i* and *b_j* are bias terms (small adjustments for each word). *log X_{ij}* is the target — the logarithm of the co-occurrence count. *f(X_{ij})* is a weighting function that reduces the influence of extremely common pairs (like "the" + "of") so they don't dominate learning. The whole thing is squared so the model is penalized equally for over- or under-estimating.

where \( X_{ij} \) is the co-occurrence count and \( f \) is a weighting function that prevents very common pairs from dominating.
:::

In practice, Word2Vec and GloVe produce embeddings of similar quality. GloVe is often preferred because the pretrained vectors from Stanford are high quality and widely available.

## Using Pretrained Embeddings

Training embeddings from scratch requires massive corpora. For most tasks, you should start with pretrained embeddings and optionally fine-tune them.

```python title="Loading pretrained embeddings into PyTorch"
import torch
import torch.nn as nn
import numpy as np

# Simulating loading GloVe vectors into a PyTorch embedding layer
# In practice, load from a file like glove.6B.100d.txt

vocab = ["<pad>", "<unk>", "the", "cat", "sat", "on", "mat"]
embedding_dim = 100
vocab_size = len(vocab)

# Create embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

# In real code, you'd load pretrained vectors:
# pretrained_weights = load_glove_vectors("glove.6B.100d.txt", vocab)
# embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))

# Optionally freeze embeddings during training
# embedding.weight.requires_grad = False

# Look up embeddings for a sentence
word_to_idx = {w: i for i, w in enumerate(vocab)}
sentence = ["the", "cat", "sat", "on", "the", "mat"]
indices = torch.tensor([word_to_idx[w] for w in sentence])

embedded = embedding(indices)
print(f"Input shape:  {indices.shape}")       # [6]
print(f"Output shape: {embedded.shape}")      # [6, 100]
```

:::tip[Line-by-Line Walkthrough]
- **`nn.Embedding(vocab_size, embedding_dim, padding_idx=0)`** — Creates a lookup table: each word gets its own row of 100 numbers. Think of it as a dictionary where you look up a word and get back a list of numbers describing it. `padding_idx=0` means word index 0 (our padding token) will always return zeros.
- **`torch.tensor([word_to_idx[w] for w in sentence])`** — Converts each word in the sentence to its index number, like looking up locker numbers.
- **`embedding(indices)`** — Looks up the embedding for each word index. Input is 6 numbers, output is 6 rows of 100 numbers each — each word now has a rich numerical representation.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch numpy
```

**Steps:**
1. Save to `pytorch_embeddings.py`
2. Run: `python pytorch_embeddings.py`

**Expected output:**
```
Input shape:  torch.Size([6])
Output shape: torch.Size([6, 100])
```

</details>

```python title="Loading GloVe from file"
def load_glove(filepath: str, vocab: dict[str, int], dim: int = 100) -> np.ndarray:
    """Load GloVe vectors for words in vocab."""
    embeddings = np.random.randn(len(vocab), dim) * 0.01  # random init for OOV

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[vocab[word]] = vector

    found = sum(1 for w in vocab if w in vocab)
    print(f"Loaded {found}/{len(vocab)} words from GloVe")
    return embeddings

# Usage:
# glove_matrix = load_glove("glove.6B.100d.txt", word_to_idx, dim=100)
# embedding.weight.data.copy_(torch.from_numpy(glove_matrix))
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.randn(len(vocab), dim) * 0.01`** — Starts with tiny random vectors for every word. Words we find in GloVe will be overwritten; words not in GloVe keep these small random values (better than all zeros).
- **`for line in f:`** — Reads the GloVe file line by line. Each line has a word followed by its vector values (e.g., "king 0.50 -0.32 0.18 ...").
- **`if word in vocab:`** — Only loads vectors for words we actually need, saving memory.
- **`embeddings[vocab[word]] = vector`** — Places the pretrained vector into the correct row of our embedding matrix.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy
```
You also need to download GloVe vectors from [nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) — the `glove.6B.zip` file (~822 MB).

**Steps:**
1. Download and unzip `glove.6B.zip`
2. Uncomment the usage lines and set the correct file path
3. Run the script

**Expected output:**
```
Loaded 7/7 words from GloVe
```

</details>

:::tip[Freeze or Fine-tune?]
- **Freeze** pretrained embeddings when you have limited training data — this prevents overfitting.
- **Fine-tune** when you have enough data and your domain vocabulary differs significantly from the pretraining corpus (e.g., medical text, legal documents).
:::

## Visualizing Embeddings

High-dimensional embeddings can be projected to 2D or 3D using **t-SNE** or **PCA** for visualization. Semantically similar words should cluster together.

```python title="Visualizing embeddings with t-SNE"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Words to visualize (in practice, use pretrained vectors)
words = [
    "king", "queen", "prince", "princess",
    "man", "woman", "boy", "girl",
    "dog", "cat", "fish", "bird",
    "car", "bus", "train", "bicycle",
]

# Simulate embeddings (replace with real vectors)
# vectors = np.array([glove[w] for w in words])
np.random.seed(42)
vectors = np.random.randn(len(words), 50)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced = tsne.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)

for i, word in enumerate(words):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]),
                 fontsize=12, ha='center', va='bottom')

plt.title("Word Embeddings Visualized with t-SNE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_tsne.png", dpi=150)
plt.show()
```

:::tip[Line-by-Line Walkthrough]
- **`TSNE(n_components=2, ...)`** — Creates a t-SNE reducer that squishes 50-dimensional vectors down to 2 dimensions for plotting. It tries to keep similar words close together on the 2D map.
- **`tsne.fit_transform(vectors)`** — Runs the t-SNE algorithm on our word vectors and returns 2D coordinates for each word.
- **`plt.annotate(word, ...)`** — Labels each dot on the scatter plot with its word, so you can see which words cluster together.
- **`plt.savefig("embeddings_tsne.png", dpi=150)`** — Saves the plot as a high-resolution image file.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install numpy matplotlib scikit-learn
```

**Steps:**
1. Save to `visualize_embeddings.py`
2. Run: `python visualize_embeddings.py`
3. A plot window will appear and `embeddings_tsne.png` will be saved.

**Expected output:** A scatter plot with 16 labeled word dots. With random vectors the clusters won't be meaningful, but with real GloVe vectors you'd see royalty words, animal words, and vehicle words grouping together.

</details>

With real pretrained embeddings, you'd see clusters for royalty, gender, animals, and vehicles.

## Limitations of Static Embeddings

Word2Vec and GloVe produce a **single vector per word**, regardless of context. The word "bank" gets the same embedding whether it means a financial institution or the side of a river.

:::info[From Static to Contextual]
This limitation is what motivated **contextual embeddings** — models like ELMo, BERT, and GPT that produce different representations for the same word depending on its surrounding context. You'll encounter these in later lessons.
:::

---

## Exercises

:::tip[Exercise 1: Exploring Word Relationships — beginner]

Using pretrained GloVe vectors from Gensim, find:
1. The 5 words most similar to "computer"
2. The similarity score between "happy" and "sad"
3. The odd one out from ["breakfast", "lunch", "dinner", "programming"]
4. Three word analogies beyond the classic king/queen example

<details>
<summary>Hints</summary>

1. Use gensim.downloader.load('glove-wiki-gigaword-50')
2. Try .most_similar() and .similarity() methods

</details>

:::

:::tip[Exercise 2: Embedding Visualization Dashboard — intermediate]

Create a visualization of 40 words from 5 categories (countries, colors, animals, professions, emotions) using pretrained GloVe embeddings. Project them to 2D with both PCA and t-SNE. Do semantically similar words cluster together? Which projection method preserves clusters better?

<details>
<summary>Hints</summary>

1. Pick 30-40 words from 4-5 semantic categories
2. Try both PCA and t-SNE — how do results differ?
3. Color-code words by category in your plot

</details>

:::

:::tip[Exercise 3: Word2Vec from Scratch — advanced]

Implement the Skip-gram model with negative sampling in PyTorch. Train it on a small text corpus (e.g., a few Wikipedia articles) and verify that the learned embeddings capture word similarities. Compare your results to pretrained GloVe vectors.

<details>
<summary>Hints</summary>

1. Implement Skip-gram with negative sampling
2. Use two embedding matrices: one for center words, one for context words
3. Sample negative words proportional to word frequency raised to the 3/4 power
4. Use binary cross-entropy loss

</details>

:::

---

## Resources

- **[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)** _(paper)_ by Mikolov et al., 2013 — The original Word2Vec paper introducing CBOW and Skip-gram.

- **[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)** _(paper)_ by Pennington et al., 2014 — The GloVe paper and pretrained vectors.

- **[The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)** _(tutorial)_ by Jay Alammar — Beautifully illustrated guide to Word2Vec internals.

- **[Stanford CS224N: Word Vectors](https://www.youtube.com/watch?v=8rXD5-xhemo)** _(video)_ by Christopher Manning — Stanford lecture on word embeddings from the NLP with Deep Learning course.

- **[Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)** _(tutorial)_ by Radim Řehůřek — Hands-on tutorial for training and using Word2Vec with Gensim.
