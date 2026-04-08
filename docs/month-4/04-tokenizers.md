---
sidebar_position: 4
slug: tokenizers
title: "Tokenizers: BPE, WordPiece, and SentencePiece"
---


# Tokenizers: BPE, WordPiece, and SentencePiece

Tokenization is the first step in any language model pipeline and one of the most consequential. The choice of tokenizer affects model performance, multilingual capability, arithmetic ability, and even how many API dollars you spend. Yet it's often glossed over. In this lesson, we'll build a deep understanding of how modern tokenizers work, implement one from scratch, and explore the tradeoffs involved.

## Why Tokenization Matters

Language models don't see text — they see sequences of integers. A tokenizer converts raw text into a sequence of token IDs and back. The way you split text into tokens determines:

1. **Vocabulary size.** Larger vocabularies mean the embedding table consumes more parameters, but each token carries more information. Smaller vocabularies mean fewer parameters but longer sequences.

2. **Sequence length.** A sentence tokenized into characters might be 50 tokens long. With subword tokenization, it might be 15 tokens. Since Transformers have quadratic attention cost in sequence length, shorter is better.

3. **Handling of rare words.** Word-level tokenizers can't handle words they've never seen. Subword tokenizers can break unknown words into known pieces: `"unhappiness"` → `["un", "happiness"]` or `["un", "hap", "pi", "ness"]`.

4. **Multilingual support.** Languages with large character sets (Chinese, Japanese) or agglutinative morphology (Turkish, Finnish) need tokenizers that handle diverse scripts efficiently.

:::info[The Tokenization Sweet Spot]
Character-level tokenization is too fine-grained (sequences too long, each token has little meaning). Word-level tokenization is too coarse (can't handle new words, huge vocabulary). **Subword tokenization** hits the sweet spot: common words stay whole, rare words are split into meaningful pieces, and vocabulary stays manageable.
:::

## Byte-Pair Encoding (BPE)

BPE is the most widely used tokenization algorithm. GPT-2, GPT-3, GPT-4, LLaMA, and most modern LLMs use variants of BPE. It was originally a data compression algorithm, adapted for NLP by Sennrich et al. in 2016.

### The BPE Algorithm, Step by Step

**Training phase** (building the vocabulary):

1. Start with a base vocabulary of all individual characters (or bytes)
2. Count the frequency of every adjacent pair of tokens in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 for a desired number of merges (this determines vocabulary size)

**Encoding phase** (tokenizing new text):

1. Split the text into individual characters
2. Apply the learned merges in the same order they were learned during training
3. The result is a sequence of subword tokens

### Walkthrough Example

Let's trace BPE on a tiny corpus:

```
Corpus: "low low low low low lowest lowest newer newer newer wider"

Step 0 — Character vocabulary:
  {l, o, w, e, s, t, n, r, i, d, ' '}
  
  Frequencies of words:
  "low"    → 5
  "lowest" → 2
  "newer"  → 3
  "wider"  → 1

Step 1 — Split into characters:
  l o w (×5), l o w e s t (×2), n e w e r (×3), w i d e r (×1)
  
Step 2 — Count all adjacent pairs:
  (l, o): 7  ← most frequent
  (o, w): 7
  (w, e): 5
  (e, s): 2
  (s, t): 2
  (n, e): 3
  (e, r): 4
  (w, i): 1
  (i, d): 1
  (d, e): 1

Step 3 — Merge (l, o) → "lo":
  lo w (×5), lo w e s t (×2), n e w e r (×3), w i d e r (×1)

Step 4 — Recount pairs, merge (lo, w) → "low":
  low (×5), low e s t (×2), n e w e r (×3), w i d e r (×1)

Step 5 — Recount, merge (e, r) → "er":
  low (×5), low e s t (×2), n e w er (×3), w i d er (×1)

Step 6 — Merge (n, e) → "ne":
  low (×5), low e s t (×2), ne w er (×3), w i d er (×1)

...and so on until we reach the desired vocabulary size.
```

### BPE From Scratch

```python title="Building a BPE tokenizer from scratch"
from collections import Counter, defaultdict
import re

class SimpleBPE:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = []  # ordered list of (pair, merged_token)
        self.vocab = set()

    def _get_pairs(self, word_freqs):
        """Count frequency of all adjacent symbol pairs across the corpus."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def train(self, corpus):
        """Learn BPE merges from a text corpus."""
        # Tokenize into words and count frequencies
        words = corpus.split()
        word_counts = Counter(words)

        # Initialize: split each word into characters with spaces
        # Add a special end-of-word marker
        word_freqs = {}
        for word, count in word_counts.items():
            chars = " ".join(list(word)) + " </w>"
            word_freqs[chars] = count

        # Build initial vocabulary from characters
        self.vocab = set()
        for word in word_freqs:
            for symbol in word.split():
                self.vocab.add(symbol)

        for i in range(self.num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self.vocab.add(merged)

            # Apply merge to all words
            new_word_freqs = {}
            pattern = re.escape(best_pair[0]) + r" " + re.escape(best_pair[1])
            for word, freq in word_freqs.items():
                new_word = re.sub(pattern, merged, word)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

            if (i + 1) % 20 == 0:
                print(f"Merge {i+1}: {best_pair} → {merged}")

        print(f"\\nVocabulary size: {len(self.vocab)}")

    def tokenize(self, text):
        """Tokenize a string using learned merges."""
        tokens = []
        for word in text.split():
            word = " ".join(list(word)) + " </w>"

            for pair in self.merges:
                pattern = re.escape(pair[0]) + r" " + re.escape(pair[1])
                word = re.sub(pattern, pair[0] + pair[1], word)

            tokens.extend(word.split())
        return tokens


# Train on a small corpus
corpus = ("low " * 5 + "lowest " * 2 + "newer " * 3 + "wider " * 1) * 10
bpe = SimpleBPE(num_merges=30)
bpe.train(corpus)

# Tokenize new text
test = "lowest newest widest"
tokens = bpe.tokenize(test)
print(f"\\nTokenized '{test}': {tokens}")
```

:::tip[Line-by-Line Walkthrough]
- **`self.merges = []`** — Stores the ordered list of merge operations learned during training. Order matters because merges are applied sequentially.
- **`pairs[(symbols[i], symbols[i + 1])] += freq`** — Counts how often each adjacent pair appears across the entire corpus, weighted by word frequency.
- **`best_pair = pairs.most_common(1)[0][0]`** — Finds the most frequent adjacent pair — this is the one BPE merges next.
- **`merged = best_pair[0] + best_pair[1]`** — Creates the new token by concatenating the two pieces (e.g., "l" + "o" = "lo").
- **`pattern = re.escape(best_pair[0]) + r" " + re.escape(best_pair[1])`** — Builds a regex pattern to find the pair in our space-separated representation, then replaces it with the merged token.
- **`for pair in self.merges:`** — During tokenization, applies each learned merge in the exact same order they were learned. This deterministic ordering ensures consistent tokenization.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
No extra packages needed — this uses only Python builtins (`collections`, `re`).

**Steps:**
1. Save the code to `bpe_from_scratch.py`
2. Run: `python bpe_from_scratch.py`

**Expected output:**
```
Merge 20: ('low', 'est</w>') → lowest</w>

Vocabulary size: 42
Tokenized 'lowest newest widest': ['lowest</w>', 'new', 'est</w>', 'wid', 'est</w>']
```
(Exact output depends on merge order. "lowest" becomes one token because it was common. "newest" and "widest" get split because they're rarer.)

</details>

## WordPiece (BERT's Tokenizer)

WordPiece is very similar to BPE but uses a different merge criterion. Instead of merging the most **frequent** pair, WordPiece merges the pair that maximizes the **likelihood of the training data** — effectively choosing pairs where the merged token's frequency is high relative to the individual token frequencies.

:::info[Plain English: How Does WordPiece Choose What to Merge?]
Imagine you're at a party and you notice that "pea" and "nut" always appear together — you see "peanut" way more often than you'd expect if "pea" and "nut" just happened to be neighbors by chance. WordPiece uses the same logic: it merges two pieces that appear together surprisingly often relative to how common each piece is individually. Frequent pairs where both parts are also individually common (like "t" and "h") score lower than rare-but-inseparable pairs.
:::

:::note[WordPiece Merge Criterion]
WordPiece scores each candidate pair \( (a, b) \) as:

$$
\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}
$$

This favors merging pairs where the combination is disproportionately common — a signal that they form a meaningful unit.
:::

**Reading the formula:** **score(a, b)** is how strongly we want to merge piece "a" and piece "b" into one token. **freq(ab)** is how many times "a" and "b" appear next to each other in the training text. **freq(a)** and **freq(b)** are how many times each piece appears individually (anywhere). The formula divides the pair frequency by the product of individual frequencies — this is high when the pair appears together *more than you'd expect by chance*.

### WordPiece Encoding Differences

WordPiece marks continuation tokens with `##`. When a word is split, all pieces except the first are prefixed with `##`:

```
"unhappiness" → ["un", "##hap", "##pi", "##ness"]
"embedding"   → ["em", "##bed", "##ding"]
```

This is purely a convention — it lets you reconstruct the original text by joining tokens and removing `##` prefixes.

```python title="WordPiece tokenization with BERT"
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = [
    "Hello, world!",
    "Transformers are revolutionizing NLP.",
    "antidisestablishmentarianism",
    "The café served crème brûlée.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"Text:   {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Count:  {len(tokens)} tokens")
    print()
```

:::tip[Line-by-Line Walkthrough]
- **`BertTokenizer.from_pretrained('bert-base-uncased')`** — Loads BERT's WordPiece tokenizer. "uncased" means it lowercases everything first.
- **`tokenizer.tokenize(text)`** — Splits text into subword tokens. Words BERT doesn't recognize get broken into pieces prefixed with `##`.
- **`tokenizer.encode(text)`** — Goes further: converts tokens to integer IDs and adds special tokens (`[CLS]` at the start, `[SEP]` at the end).
- The loop tests diverse inputs (English, long words, code, accented characters) to show how WordPiece handles different text types.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers
```

**Steps:**
1. Save the code to `wordpiece_demo.py`
2. Run: `python wordpiece_demo.py`
3. First run downloads the BERT tokenizer files (~230 KB).

**Expected output:**
```
Text:   Hello, world!
Tokens: ['hello', ',', 'world', '!']
IDs:    [101, 7592, 1010, 2088, 999, 102]
Count:  4 tokens

Text:   antidisestablishmentarianism
Tokens: ['anti', '##dis', '##est', '##ab', '##lish', '##ment', '##aria', '##nis', '##m']
...
```

</details>

## SentencePiece

SentencePiece (Kudo & Richardson, 2018) solves a practical problem: most tokenizers assume whitespace-separated words, which doesn't work for languages like Chinese, Japanese, or Thai that don't use spaces between words.

SentencePiece treats the input as a **raw byte stream** — it doesn't assume whitespace means word boundaries. It replaces spaces with a special character `▁` (Unicode character U+2581) and applies BPE or Unigram segmentation directly on the raw text.

:::info[Language-Agnostic Tokenization]
SentencePiece is **language-agnostic** — it works identically for English, Chinese, Arabic, or any other language. This made it the standard tokenizer for multilingual models like mBART, XLM-R, and LLaMA.
:::

### Unigram Model (SentencePiece Alternative to BPE)

SentencePiece also implements the **Unigram** language model tokenizer, which works oppositely from BPE:

- **BPE** starts with characters and iteratively **merges** (bottom-up)
- **Unigram** starts with a large vocabulary and iteratively **prunes** (top-down)

The Unigram model assigns a probability to each token and finds the segmentation that maximizes the total probability. This means the same word can be tokenized differently depending on context and probability distributions.

```python title="SentencePiece with Llama 4 tokenizer"
from transformers import AutoTokenizer

# LLaMA uses SentencePiece with BPE
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-4-Scout-17B-16E')

texts = [
    "Hello world",
    "The Transformer architecture was introduced in 2017.",
    "こんにちは世界",  # Japanese
    "def hello(): print('world')",
    "   lots   of   spaces   ",
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"Text:   {repr(text)}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(tokens)}")
    print()
```

:::tip[Line-by-Line Walkthrough]
- **`AutoTokenizer.from_pretrained('meta-llama/Llama-4-Scout-17B-16E')`** — Loads the tokenizer for LLaMA 4, which uses SentencePiece with BPE. `AutoTokenizer` automatically detects the right tokenizer class.
- **`tokenizer.tokenize(text)`** — Splits text into subword tokens. Notice the `▁` character — SentencePiece uses this to represent spaces, since it treats text as a raw stream (no pre-splitting on whitespace).
- The test inputs include English, Japanese, code, and extra spaces to demonstrate how SentencePiece is language-agnostic — it handles all of them without special rules.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers sentencepiece protobuf
```
You may also need to authenticate with Hugging Face if the model is gated: `huggingface-cli login`

**Steps:**
1. Save the code to `sentencepiece_demo.py`
2. Run: `python sentencepiece_demo.py`

**Expected output:**
```
Text:   'Hello world'
Tokens: ['▁Hello', '▁world']
Count:  2

Text:   'こんにちは世界'
Tokens: ['▁', 'こんにちは', '世界']
Count:  3
```
(Exact tokenization varies by model version.)

</details>

## Tiktoken and Modern Tokenizers

OpenAI's **tiktoken** is a fast BPE tokenizer used by GPT-3.5, GPT-4, and the OpenAI API. It's a byte-level BPE tokenizer implemented in Rust with Python bindings, making it extremely fast.

There are two main tiktoken encodings:
- **`cl100k_base`** — used by GPT-4 and GPT-3.5-turbo (~100K vocab)
- **`o200k_base`** — used by GPT-4o, o1, o3, and newer models (~200K vocab, better multilingual compression)

```python title="Using tiktoken"
import tiktoken

# cl100k_base is used by GPT-4 and GPT-3.5-turbo
# For GPT-4o and newer models, use o200k_base instead
enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenization is the unsung hero of NLP!"
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode each token individually to see the pieces
print("\\nTokens breakdown:")
for tid in tokens:
    decoded = enc.decode([tid])
    print(f"  {tid:6d} → {repr(decoded)}")

# Compare token counts across different types of text
examples = {
    "English prose": "The quick brown fox jumps over the lazy dog.",
    "Python code": "def factorial(n):\\n    return 1 if n <= 1 else n * factorial(n-1)",
    "JSON": '{"name": "Alice", "age": 30, "city": "NYC"}',
    "Math": "∫₀^∞ e^(-x²) dx = √π/2",
    "Repetitive": "buffalo " * 8,
}

print("\\nToken counts by content type:")
for label, text in examples.items():
    count = len(enc.encode(text))
    ratio = len(text) / count
    print(f"  {label:20s}: {count:3d} tokens ({ratio:.1f} chars/token)")
```

:::tip[Line-by-Line Walkthrough]
- **`enc = tiktoken.get_encoding("cl100k_base")`** — Loads the `cl100k_base` encoding, which is used by GPT-4 and GPT-3.5-turbo. This has ~100K tokens in its vocabulary.
- **`tokens = enc.encode(text)`** — Converts text into a list of integer token IDs.
- **`enc.decode([tid])`** — Converts a single token ID back into its text representation. Useful for seeing exactly how the text was split.
- **The `examples` dict** — Tests different content types to show that the tokenizer's efficiency varies. English prose compresses well (~4 chars/token), while math symbols and emojis compress poorly (~1-2 chars/token).
- **`ratio = len(text) / count`** — Computes the average characters per token. Higher is better — it means each token carries more information.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tiktoken
```

**Steps:**
1. Save the code to `tiktoken_demo.py`
2. Run: `python tiktoken_demo.py`

**Expected output:**
```
Text: Tokenization is the unsung hero of NLP!
Token IDs: [3947, 2065, ...]
Token count: 9

Token counts by content type:
  English prose       :   9 tokens (4.9 chars/token)
  Python code         :  18 tokens (3.5 chars/token)
  JSON                :  14 tokens (3.1 chars/token)
  Math                :  12 tokens (2.0 chars/token)
  Repetitive          :  16 tokens (4.0 chars/token)
```

</details>

### Byte-Level BPE

Modern tokenizers (GPT-2, GPT-4, LLaMA) use **byte-level** BPE. Instead of starting with Unicode characters, they start with the 256 individual bytes. This guarantees that any input can be tokenized — even binary data, emojis, or unknown scripts — because every byte is a valid token.

The tradeoff: a single Unicode character might be split into multiple byte tokens. For example, the emoji 🎉 is 4 bytes in UTF-8, so it could be 1–4 tokens depending on whether the tokenizer learned to merge those bytes.

:::warning[Tokenization Affects Arithmetic]
LLMs are notoriously bad at arithmetic, and tokenization is partly to blame. The number `42137` might be tokenized as `["421", "37"]` — the digit boundaries don't align with the token boundaries, making it harder for the model to learn positional arithmetic. This is why some models use character-level or digit-level tokenization for numbers.
:::

## Vocabulary Size Tradeoffs

| Vocab Size | Pros | Cons |
|-----------|------|------|
| Small (8K–16K) | Smaller embedding table, more regularization | Longer sequences, each token less meaningful |
| Medium (32K–64K) | Good balance, standard for most models | — |
| Large (100K–256K) | Short sequences, better multilingual coverage | Huge embedding table, sparse usage of rare tokens |

**GPT-2:** 50,257 tokens. **LLaMA:** 32,000 tokens. **GPT-4:** ~100,000 tokens. **Gemini:** 256,000 tokens.

The trend is toward larger vocabularies, especially for multilingual models. A larger vocabulary compresses non-English text more efficiently, reducing the "multilingual tax" where non-English text takes disproportionately more tokens.

```python title="Measuring the multilingual tax"
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

sentences = {
    "English": "Artificial intelligence is transforming the world.",
    "Spanish": "La inteligencia artificial está transformando el mundo.",
    "German": "Künstliche Intelligenz transformiert die Welt.",
    "Chinese": "人工智能正在改变世界。",
    "Arabic": "الذكاء الاصطناعي يغير العالم.",
    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
    "Japanese": "人工知能は世界を変えている。",
}

print(f"{'Language':<12} {'Chars':<8} {'Tokens':<8} {'Chars/Token':<12}")
print("-" * 44)
for lang, text in sentences.items():
    n_chars = len(text)
    n_tokens = len(enc.encode(text))
    ratio = n_chars / n_tokens
    print(f"{lang:<12} {n_chars:<8} {n_tokens:<8} {ratio:<12.2f}")
```

:::tip[Line-by-Line Walkthrough]
- **`sentences = { "English": "...", "Chinese": "...", ... }`** — The same sentence (roughly "Artificial intelligence is transforming the world") in seven languages. This lets us compare tokenization efficiency directly.
- **`n_tokens = len(enc.encode(text))`** — Counts how many tokens each language's version requires.
- **`ratio = n_chars / n_tokens`** — Characters per token. English typically gets 3-4 chars/token. Languages like Chinese, Arabic, or Hindi often get only 1-2 chars/token — this is the "multilingual tax" where non-English text takes more tokens (and costs more in API usage).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tiktoken
```

**Steps:**
1. Save the code to `multilingual_tax.py`
2. Run: `python multilingual_tax.py`

**Expected output:**
```
Language     Chars    Tokens   Chars/Token
--------------------------------------------
English      47       10       4.70
Spanish      53       13       4.08
German       44       11       4.00
Chinese      10       10       1.00
Arabic       28       18       1.56
Hindi        37       30       1.23
Japanese     12       10       1.20
```
(Exact numbers vary. The key insight: non-Latin scripts use significantly more tokens for the same meaning.)

</details>

## Special Tokens

All tokenizers include special tokens that serve specific purposes:

| Token | Used By | Purpose |
|-------|---------|---------|
| `[CLS]` | BERT | Sequence-level representation |
| `[SEP]` | BERT | Segment separator |
| `[MASK]` | BERT | MLM placeholder |
| `[PAD]` | Most models | Padding for batch alignment |
| `<s>`, `</s>` | LLaMA, T5 | Beginning/end of sequence |
| `<|endoftext|>` | GPT-2/3 | Document separator |
| `<|im_start|>`, `<|im_end|>` | ChatGPT | Chat message delimiters |

These tokens have reserved IDs and are never produced by the merging algorithm. They're added to the vocabulary separately.

## Summary

| Algorithm | Used By | Key Feature |
|-----------|---------|-------------|
| BPE | GPT-2/3/4, LLaMA | Frequency-based merging, most common |
| WordPiece | BERT, DistilBERT | Likelihood-based merging, `##` prefixes |
| SentencePiece | LLaMA, T5, mBART | Language-agnostic, no whitespace assumption |
| Unigram | Part of SentencePiece | Top-down pruning, probabilistic segmentation |
| Tiktoken | OpenAI models | Fast Rust-based BPE implementation |

---

## Exercises

:::tip[Token Count Estimation — beginner]

Without running code, estimate the number of tokens for each of these strings using GPT-4's tokenizer (`cl100k_base`). Then check your answers with tiktoken.

1. `"Hello, world!"`
2. `"antidisestablishmentarianism"`
3. `"def f(x): return x * 2"`
4. `"12345678"`
5. `"🎉🎊🎈🎁"`

<div>
**Approximate answers:**
1. 4 tokens (`Hello`, `,`, ` world`, `!`)
2. 6 tokens (splits into subwords)
3. 9 tokens (code tokenizes less efficiently)
4. 3 tokens (numbers are often split into 2-4 digit chunks)
5. 4 tokens (one per emoji, since each is a known byte sequence)
<details>
<summary>Hints</summary>

1. Use tiktoken to check your guesses
2. English prose is typically 3-4 characters per token
3. Code has lower chars-per-token ratios
4. Numbers are tokenized unpredictably

</details>

:::

:::tip[Train a BPE Tokenizer — intermediate]

Using the HuggingFace `tokenizers` library, train a BPE tokenizer from scratch on a text corpus of your choice (e.g., a book from Project Gutenberg). Compare how the tokenizer's behavior changes with vocabulary sizes of 1000, 5000, and 30000.

<div>
**Solution:**

```python
from tokenizers import ByteLevelBPETokenizer

for vocab_size in [1000, 5000, 30000]:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["corpus.txt"], vocab_size=vocab_size,
                    min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
    
    output = tokenizer.encode("The quick brown fox jumps over the lazy dog")
    print(f"Vocab \{vocab_size\}: \{len(output.tokens)\} tokens → \{output.tokens\}")
```

:::tip[Line-by-Line Walkthrough]
- **`ByteLevelBPETokenizer()`** — Creates a fresh byte-level BPE tokenizer with no learned merges (starting from 256 byte tokens).
- **`tokenizer.train(files=["corpus.txt"], vocab_size=vocab_size, min_frequency=2, ...)`** — Learns BPE merges from the text file until the vocabulary reaches the target size. `min_frequency=2` ensures a pair must appear at least twice to be merged.
- **`output = tokenizer.encode("The quick brown fox...")`** — Tokenizes a test sentence with the freshly trained tokenizer. Smaller vocabularies produce more tokens (finer splits); larger vocabularies produce fewer tokens (coarser splits).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tokenizers
```

**Steps:**
1. Prepare a text file named `corpus.txt` (at least 1 MB of text — e.g., a Project Gutenberg book).
2. Save the code to `train_bpe.py`
3. Run: `python train_bpe.py`

**Expected output:**
```
Vocab 1000: 14 tokens → ['The', 'Ġqu', 'ick', 'Ġbr', 'own', ...]
Vocab 5000: 11 tokens → ['The', 'Ġquick', 'Ġbrown', ...]
Vocab 30000: 9 tokens → ['The', 'Ġquick', 'Ġbrown', 'Ġfox', ...]
```
(Larger vocabularies compress text into fewer tokens.)

</details>

<details>
<summary>Hints</summary>

1. Use the HuggingFace tokenizers library
2. Start with a ByteLevelBPETokenizer
3. Train on a corpus of at least 1MB
4. Experiment with different vocab sizes: 1000, 5000, 30000

</details>

:::

:::tip[The Tokenization Arithmetic Problem — advanced]

Investigate why LLMs struggle with arithmetic by analyzing how GPT-4's tokenizer handles numbers. Tokenize the numbers 1 through 100000, and for each, check whether the tokenization aligns with digit boundaries. What fraction of multi-digit numbers are tokenized as a single token? What patterns do you see?

Then, propose and implement a "digit-aware" tokenization scheme that always keeps individual digits as separate tokens while still using BPE for text.

<div>
**Solution approach:**

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

single_token = 0
multi_token = 0
for n in range(1, 100001):
    tokens = enc.encode(str(n))
    if len(tokens) == 1:
        single_token += 1
    else:
        multi_token += 1

print(f"Single-token numbers: \{single_token\}")
print(f"Multi-token numbers: \{multi_token\}")
```

:::tip[Line-by-Line Walkthrough]
- **`enc = tiktoken.get_encoding("cl100k_base")`** — Loads GPT-4's tokenizer so we can see how it handles numbers.
- **`for n in range(1, 100001):`** — Loops through every number from 1 to 100,000, tokenizing each one.
- **`tokens = enc.encode(str(n))`** — Converts the number to a string and tokenizes it. If a number like "42137" becomes `["421", "37"]`, it took 2 tokens — meaning the split doesn't align with digit boundaries.
- **`if len(tokens) == 1: single_token += 1`** — Counts how many numbers fit entirely in one token vs. being split across multiple tokens.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tiktoken
```

**Steps:**
1. Save the code to `number_tokens.py`
2. Run: `python number_tokens.py`

**Expected output:**
```
Single-token numbers: ~2700
Multi-token numbers: ~97300
```
(Most numbers above ~999 get split across multiple tokens at unpredictable boundaries.)

</details>

You'll find that most 1-3 digit numbers are single tokens, but larger numbers get split at unpredictable boundaries. A digit-aware scheme would pre-process numbers to insert spaces between digits before tokenization, or use a separate digit vocabulary.
<details>
<summary>Hints</summary>

1. Tokenize numbers of varying lengths
2. Compare token boundaries with digit boundaries
3. Consider: would character-level tokenization of digits help?
4. Think about how addition requires aligned digit columns

</details>

:::

---

## Resources

- **[Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909)** _(paper)_ by Sennrich et al. — The paper that introduced BPE for neural NLP, now the dominant tokenization approach.

- **[SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)** _(paper)_ by Kudo & Richardson — Language-agnostic tokenization that handles any script without pre-tokenization.

- **[HuggingFace Tokenizers Library](https://github.com/huggingface/tokenizers)** _(tool)_ — Fast Rust-based tokenizer library with BPE, WordPiece, and Unigram implementations.

- **[tiktoken](https://github.com/openai/tiktoken)** _(tool)_ by OpenAI — OpenAI's fast BPE tokenizer used by GPT-3.5 and GPT-4.

- **[Let's Build the GPT Tokenizer (Video)](https://www.youtube.com/watch?v=zduSFxRajkE)** _(video)_ by Andrej Karpathy — A 2-hour deep dive into building a BPE tokenizer from scratch — the definitive educational resource.

- **[A Programmer's Introduction to Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)** _(tutorial)_ by Nathan Reed — Essential background on Unicode, UTF-8, and byte representations that underlie modern tokenizers.
