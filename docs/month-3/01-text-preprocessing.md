---
sidebar_position: 1
slug: text-preprocessing
title: "Text Preprocessing"
---


# Text Preprocessing

:::info[What You'll Learn]
- Tokenization strategies (word, subword, character)
- Text cleaning and normalization techniques
- Building vocabularies and handling unknown tokens
- Preparing text data for neural network input
:::

:::note[Prerequisites]
[Python for AI](/curriculum/month-1/python-for-ai) and [Neural Networks Introduction](/curriculum/month-1/neural-networks-intro) from Month 1.
:::

**Estimated time:** Reading: ~30 min | Exercises: ~2 hours

Before any NLP model can understand text, that text must be transformed from raw strings into structured numerical representations. Text preprocessing is the critical first step in every NLP pipeline — and getting it right has a disproportionate effect on model performance.

In this lesson, you'll learn the core preprocessing techniques that underpin all modern NLP work, from classical bag-of-words models to the tokenizers used by GPT and BERT.

## Why Preprocessing Matters

Raw text is noisy. Consider a single tweet: it might contain uppercase letters, hashtags, URLs, emojis, misspellings, and slang. A model that receives raw text must learn to handle all of this variation — or we can reduce the variation upfront so the model can focus on semantics.

:::info[The Preprocessing Trade-off]
Every preprocessing step discards information. Lowercasing removes the distinction between "Apple" (company) and "apple" (fruit). Removing stop words can destroy negation ("not good" → "good"). Always consider what information your task actually needs.
:::

## Tokenization

**Tokenization** splits text into smaller units called **tokens**. The granularity of these tokens profoundly affects what your model can learn.

### Word-Level Tokenization

The simplest approach: split on whitespace and punctuation.

```python title="Word-level tokenization"
import re

def word_tokenize(text: str) -> list[str]:
    """Split text into word tokens using regex."""
    return re.findall(r"\\b\\w+\\b", text.lower())

text = "The cat sat on the mat. The cat ate the hat!"
tokens = word_tokenize(text)
print(tokens)
# ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the', 'cat', 'ate', 'the', 'hat']
print(f"Vocabulary size from this sentence: {len(set(tokens))}")
# Vocabulary size from this sentence: 7
```

:::tip[Line-by-Line Walkthrough]
- **`re.findall(r"\\b\\w+\\b", text.lower())`** — Converts the text to lowercase, then uses a regex to find all "word" sequences (letters, digits, underscores). The `\\b` markers mean "word boundary," so punctuation gets dropped automatically.
- **`set(tokens)`** — Converts the list of tokens to a set (removing duplicates) so we can count how many *unique* words there are.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.10+ (no extra packages needed — `re` is built-in).

**Steps:**
1. Save the code to a file, e.g. `word_tokenize.py`
2. Open a terminal and run: `python word_tokenize.py`

**Expected output:**
```
['the', 'cat', 'sat', 'on', 'the', 'mat', 'the', 'cat', 'ate', 'the', 'hat']
Vocabulary size from this sentence: 7
```

</details>

**Pros:** Intuitive, easy to implement.
**Cons:** Huge vocabularies for large corpora; can't handle out-of-vocabulary (OOV) words; struggles with morphology ("running", "runs", "ran" are all separate tokens).

### Character-Level Tokenization

Split every character as its own token. The vocabulary is tiny (just the alphabet plus punctuation), but sequences become very long.

```python title="Character-level tokenization"
text = "Hello, NLP!"
char_tokens = list(text)
print(char_tokens)
# ['H', 'e', 'l', 'l', 'o', ',', ' ', 'N', 'L', 'P', '!']
print(f"Vocabulary: {sorted(set(char_tokens))}")
# Vocabulary: [' ', '!', ',', 'H', 'L', 'N', 'P', 'e', 'l', 'o']
```

:::tip[Line-by-Line Walkthrough]
- **`list(text)`** — Python's `list()` on a string splits it into individual characters, so each letter, space, and punctuation mark becomes its own token.
- **`sorted(set(char_tokens))`** — `set()` removes duplicates, `sorted()` puts them in order so the vocabulary is easy to read.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.10+ (no extra packages needed).

**Steps:**
1. Save to `char_tokenize.py`
2. Run: `python char_tokenize.py`

**Expected output:**
```
['H', 'e', 'l', 'l', 'o', ',', ' ', 'N', 'L', 'P', '!']
Vocabulary: [' ', '!', ',', 'H', 'L', 'N', 'P', 'e', 'l', 'o']
```

</details>

**Pros:** No OOV problem; tiny vocabulary.
**Cons:** Sequences are very long; the model must learn to compose characters into meaning, which requires more capacity.

### Subword Tokenization

Modern NLP uses subword methods that balance vocabulary size with sequence length. The two dominant algorithms are **Byte Pair Encoding (BPE)** and **WordPiece**.

:::info[Byte Pair Encoding (BPE)]
BPE starts with individual characters and iteratively merges the most frequent adjacent pair. After enough merges, common words become single tokens while rare words are split into known subwords. GPT-2/3/4 use BPE. BERT uses a variant called WordPiece.
:::

```python title="Subword tokenization with Hugging Face tokenizers"
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Train a BPE tokenizer from scratch on a small corpus
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=500,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
)

corpus = [
    "the cat sat on the mat",
    "the cat ate the hat",
    "transformers are powerful models for NLP",
    "natural language processing is fascinating",
]
tokenizer.train_from_iterator(corpus, trainer)

encoded = tokenizer.encode("transformers process language naturally")
print("Tokens:", encoded.tokens)
print("IDs:   ", encoded.ids)
```

:::tip[Line-by-Line Walkthrough]
- **`Tokenizer(models.BPE())`** — Creates a brand-new tokenizer that will use the Byte Pair Encoding algorithm (the same one GPT uses).
- **`pre_tokenizers.Whitespace()`** — Tells the tokenizer to first split on spaces before applying BPE merges.
- **`BpeTrainer(vocab_size=500, ...)`** — Configures training: build a vocabulary of up to 500 tokens, and reserve four special tokens for padding, unknowns, and sentence markers.
- **`tokenizer.train_from_iterator(corpus, trainer)`** — Feeds our small corpus to the trainer, which learns which character pairs to merge most frequently.
- **`tokenizer.encode(...)`** — Converts a new sentence into tokens and numerical IDs using the learned merges.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install tokenizers
```

**Steps:**
1. Save to `bpe_tokenizer.py`
2. Run: `python bpe_tokenizer.py`

**Expected output:** A list of subword tokens and their numeric IDs. Exact tokens depend on what BPE learned, but you'll see words split into subwords (e.g., "naturally" might become "natural" + "ly").

</details>

## Stemming and Lemmatization

Both techniques reduce words to a common base form, but they differ in approach.

**Stemming** chops off suffixes using crude rules. It's fast but often produces non-words.

**Lemmatization** uses a dictionary and morphological analysis to return the actual root word (lemma).

```python title="Stemming vs. lemmatization"
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "better", "geese", "studies", "wolves", "happily"]

print(f"{'Word':<12} {'Stem':<12} {'Lemma':<12}")
print("-" * 36)
for w in words:
    print(f"{w:<12} {stemmer.stem(w):<12} {lemmatizer.lemmatize(w):<12}")

# Word         Stem         Lemma
# running      run          running      (lemmatizer needs POS tag for verbs)
# better       better       better
# geese        gees         goose
# studies      studi        study
# wolves       wolv         wolf
# happily      happili      happily
```

:::tip[Line-by-Line Walkthrough]
- **`PorterStemmer()`** — Creates a stemmer that chops endings off words using fixed rules (like removing "-ing", "-ed", "-ly"). Fast but crude — "studies" becomes "studi," which isn't a real word.
- **`WordNetLemmatizer()`** — Creates a lemmatizer that uses a dictionary to find the proper root word. "geese" correctly becomes "goose."
- **`nltk.download('wordnet', quiet=True)`** — Downloads the WordNet dictionary that the lemmatizer needs. `quiet=True` suppresses progress messages.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install nltk
```

**Steps:**
1. Save to `stem_lemma.py`
2. Run: `python stem_lemma.py`
3. The first run downloads the WordNet data automatically.

**Expected output:**
```
Word         Stem         Lemma
------------------------------------
running      run          running
better       better       better
geese        gees         goose
studies      studi        study
wolves       wolv         wolf
happily      happili      happily
```

</details>

:::warning[Lemmatization requires POS tags]
`WordNetLemmatizer.lemmatize("running")` returns "running" because it defaults to noun. Pass `pos='v'` to get "run". Always provide part-of-speech tags for accurate lemmatization.
:::

## Stop Words, Lowercasing, and Cleaning

### Stop Words

Stop words are extremely common words ("the", "is", "at") that carry little semantic weight. Removing them reduces noise for tasks like topic modeling or search, but **keep them** for tasks sensitive to word order (machine translation, question answering).

```python title="Stop word removal"
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

text = "This is a sample sentence showing off the stop words filtration"
tokens = text.lower().split()
filtered = [t for t in tokens if t not in stop_words]

print(f"Original:  {tokens}")
print(f"Filtered:  {filtered}")
# Filtered:  ['sample', 'sentence', 'showing', 'stop', 'words', 'filtration']
```

:::tip[Line-by-Line Walkthrough]
- **`stopwords.words('english')`** — Loads a built-in list of ~180 common English words ("the", "is", "at", "a", etc.) that usually don't carry meaning on their own.
- **`set(...)`** — Wraps the list in a set for fast lookup (checking "is this word a stop word?" is instant with a set, slow with a list).
- **`[t for t in tokens if t not in stop_words]`** — Keeps only the words that are *not* in the stop word list — the meaningful, content-carrying words.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install nltk
```

**Steps:**
1. Save to `stopwords_demo.py`
2. Run: `python stopwords_demo.py`

**Expected output:**
```
Original:  ['this', 'is', 'a', 'sample', 'sentence', 'showing', 'off', 'the', 'stop', 'words', 'filtration']
Filtered:  ['sample', 'sentence', 'showing', 'stop', 'words', 'filtration']
```

</details>

### Text Cleaning

Real-world text often contains HTML tags, URLs, special characters, and inconsistent whitespace. A cleaning function normalizes all of this.

```python title="Text cleaning pipeline"
import re
import unicodedata

def clean_text(text: str) -> str:
    """Normalize and clean raw text."""
    # Normalize unicode (e.g., accented characters)
    text = unicodedata.normalize("NFKD", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\\S+@\\S+\\.\\S+", "", text)
    # Keep only letters, numbers, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\\s.,!?'-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\\s+", " ", text).strip()
    return text

raw = "<p>Check out https://example.com! Email me@test.com — it's    great™</p>"
print(clean_text(raw))
# "Check out Email it's great"
```

:::tip[Line-by-Line Walkthrough]
- **`unicodedata.normalize("NFKD", text)`** — Converts fancy Unicode characters (like "™" or accented letters) into their plain equivalents. Think of it as translating special characters into "normal" ones.
- **`re.sub(r"<[^>]+>", " ", text)`** — Strips HTML tags like `<p>` and `</p>` by matching anything between angle brackets.
- **`re.sub(r"https?://\\S+|www\\.\\S+", "", text)`** — Removes URLs starting with "http://" or "www.".
- **`re.sub(r"\\s+", " ", text).strip()`** — Collapses multiple spaces/tabs/newlines into a single space, then trims whitespace from both ends.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.10+ (no extra packages — `re` and `unicodedata` are built-in).

**Steps:**
1. Save to `clean_text.py`
2. Run: `python clean_text.py`

**Expected output:**
```
Check out Email it's great
```

</details>

## Building a Complete Text Pipeline

In practice, you chain these steps together. Here's a reusable pipeline class.

```python title="Full text preprocessing pipeline"
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPipeline:
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\\S+", "", text)
        text = re.sub(r"[^a-zA-Z\\s]", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        return tokens

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(self.clean(text))

pipeline = TextPipeline()
sample = "<h1>The Transformers are revolutionizing NLP!</h1> Visit https://arxiv.org"
print(pipeline(sample))
# ['transformer', 'revolutionizing', 'nlp']
```

:::tip[Line-by-Line Walkthrough]
- **`class TextPipeline`** — A reusable pipeline that chains all preprocessing steps together. You configure which steps to apply when you create it.
- **`self.stop_words = set(stopwords.words('english'))`** — Loads the stop word list once during initialization so you don't reload it for every text.
- **`clean()` method** — Strips HTML, URLs, non-letter characters, and extra whitespace. Think of it as "scrubbing the dirt off" the raw text.
- **`tokenize()` method** — Splits the cleaned text into words, optionally lowercases, removes stop words, lemmatizes, and filters out very short tokens.
- **`__call__`** — Makes the pipeline callable like a function: `pipeline(text)` runs `clean` then `tokenize`.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install nltk
```

**Steps:**
1. Save to `text_pipeline.py`
2. Run: `python text_pipeline.py`

**Expected output:**
```
['transformer', 'revolutionizing', 'nlp']
```

</details>

:::tip[When to use what]
- **Search / IR:** Aggressive preprocessing — lowercase, remove stop words, stem.
- **Text classification:** Moderate — lowercase, light cleaning, maybe remove stop words.
- **Machine translation / generation:** Minimal — don't remove stop words, don't stem. Use subword tokenization.
- **Transformer fine-tuning:** Use the model's built-in tokenizer. Don't apply your own preprocessing.
:::

## Tokenization for Transformers

When working with pretrained models like BERT or GPT, always use the model's own tokenizer. These tokenizers handle subword splitting, special tokens, and padding.

```python title="Using a pretrained BERT tokenizer"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Transformers revolutionized natural language processing."
encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

print("Input IDs:", encoded["input_ids"])
print("Tokens:   ", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
print("Decoded:  ", tokenizer.decode(encoded["input_ids"][0]))
# Tokens: ['[CLS]', 'transformers', 'revolution', '##ized', 'natural', 'language', 'processing', '.', '[SEP]']
```

:::tip[Line-by-Line Walkthrough]
- **`AutoTokenizer.from_pretrained("bert-base-uncased")`** — Downloads BERT's own tokenizer (the same one the model was trained with). "uncased" means it lowercases everything.
- **`tokenizer(text, return_tensors="pt", ...)`** — Converts the sentence into numerical IDs that BERT understands, returned as a PyTorch tensor. It also adds special `[CLS]` (start) and `[SEP]` (end) tokens automatically.
- **`convert_ids_to_tokens(...)`** — Converts the numerical IDs back to human-readable tokens so you can see exactly how BERT split the words. Notice "revolutionized" becomes "revolution" + "##ized" (the `##` prefix means "continuation of the previous word").
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers torch
```

**Steps:**
1. Save to `bert_tokenizer.py`
2. Run: `python bert_tokenizer.py`
3. The first run downloads the BERT tokenizer files (~230 KB).

**Expected output:**
```
Input IDs: tensor([[ 101, 19081,  4329, 3550,  3019,  2653,  6364,  1012,   102]])
Tokens:    ['[CLS]', 'transformers', 'revolution', '##ized', 'natural', 'language', 'processing', '.', '[SEP]']
Decoded:   [CLS] transformers revolutionized natural language processing. [SEP]
```

</details>

Notice how "revolutionized" is split into "revolution" + "##ized" — that's WordPiece subword tokenization in action.

---

## Exercises

<ExerciseBlock title="Exercise 1: Custom Tokenizer Comparison" difficulty="beginner" hints={["Use Python's split(), re.findall(), and list() for the three approaches", "Count unique tokens with set()"]}>

Write a function that takes a sentence and returns three tokenizations: word-level, character-level, and whitespace-only. Compare the vocabulary sizes for the sentence: *"The quick brown fox jumps over the lazy dog. The fox was very quick!"*

</ExerciseBlock>

<ExerciseBlock title="Exercise 2: Preprocessing Impact Analysis" difficulty="intermediate" hints={["Try scikit-learn's CountVectorizer with different preprocessing", "Measure vocabulary size and document similarity before/after each step", "Use cosine similarity from sklearn.metrics.pairwise"]}>

Take 5 news headlines and measure how cosine similarity (using bag-of-words vectors) changes as you apply each preprocessing step incrementally: (1) lowercasing, (2) removing punctuation, (3) removing stop words, (4) lemmatization. Which step has the biggest impact on the similarity matrix?

</ExerciseBlock>

<ExerciseBlock title="Exercise 3: Build a BPE Tokenizer" difficulty="advanced" hints={["Start with character-level tokens", "Count all adjacent pairs in the corpus", "Merge the most frequent pair and repeat", "Track your merge rules — they define the tokenizer"]}>

Implement the Byte Pair Encoding algorithm from scratch. Start with a small corpus, represent each word as a sequence of characters plus an end-of-word marker `</w>`, and iteratively merge the most frequent pair. Run 10 merge operations and print the resulting vocabulary.

</ExerciseBlock>

---

## Resources

<ResourceCard title="NLTK Documentation" url="https://www.nltk.org/" type="tool" description="The classic Python NLP library — extensive tokenizers, stemmers, and corpora." />

<ResourceCard title="Hugging Face Tokenizers" url="https://huggingface.co/docs/tokenizers/" type="tool" description="Fast, production-ready tokenizers including BPE, WordPiece, and Unigram." />

<ResourceCard title="A Visual Guide to Tokenization" url="https://www.youtube.com/watch?v=zduSFxRajkE" type="video" author="Andrej Karpathy" description="Karpathy's deep dive into tokenization and why it matters for LLMs." />

<ResourceCard title="Neural Machine Translation of Rare Words with Subword Units" url="https://arxiv.org/abs/1508.07909" type="paper" author="Sennrich et al., 2016" description="The paper that introduced BPE for NLP." />
