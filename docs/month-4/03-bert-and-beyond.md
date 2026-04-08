---
sidebar_position: 3
slug: bert-and-beyond
title: "BERT, T5, and Encoder-Decoder Models"
---


# BERT, T5, and Encoder-Decoder Models

GPT showed the power of decoder-only models with next-token prediction. But that's not the only way to pretrain a Transformer. **BERT** uses an encoder-only architecture with masked language modeling, and **T5** uses a full encoder-decoder with a text-to-text framework. Understanding all three paradigms — encoder-only, decoder-only, and encoder-decoder — is essential for choosing the right model for a given task.

## Masked Language Modeling (BERT)

BERT (Bidirectional Encoder Representations from Transformers), published by Google in 2018, took a fundamentally different approach from GPT.

:::info[BERT's Key Insight]
GPT reads text left-to-right — each token can only see what came before it. BERT reads text **bidirectionally** — each token can attend to tokens both before and after it. This allows BERT to build richer representations of each token because it uses full context from both directions.
:::

The tradeoff: because BERT sees all tokens simultaneously, it can't be used directly for text generation (there's no causal direction). Instead, BERT excels at **understanding** tasks — classification, named entity recognition, question answering, and semantic similarity.

### How MLM Works

During pre-training, BERT randomly selects 15% of tokens in each sequence and applies one of three transformations:

- **80%** of the time: replace with `[MASK]`
- **10%** of the time: replace with a random token
- **10%** of the time: keep the original token

The model's task is to predict the original token at each masked position.

```python title="Masked language modeling in action"
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Find the position of [MASK]
mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
mask_logits = logits[0, mask_idx, :]
top5 = torch.topk(mask_logits, 5, dim=-1)

print(f"Input: {text}")
print("\\nTop 5 predictions for [MASK]:")
for score, idx in zip(top5.values[0], top5.indices[0]):
    token = tokenizer.decode(idx)
    print(f"  {token:15s} (score: {score:.2f})")
```

:::tip[Line-by-Line Walkthrough]
- **`BertTokenizer.from_pretrained('bert-base-uncased')`** — Downloads and loads BERT's tokenizer (converts text to token IDs and back).
- **`BertForMaskedLM.from_pretrained('bert-base-uncased')`** — Downloads the pre-trained BERT model configured for the fill-in-the-blank (masked language modeling) task.
- **`text = "The capital of France is [MASK]."`** — Our input sentence with one word blanked out using the special `[MASK]` token. BERT will try to guess what goes there.
- **`inputs = tokenizer(text, return_tensors='pt')`** — Converts the text into token IDs (numbers) that the model can process. `'pt'` means return PyTorch tensors.
- **`with torch.no_grad():`** — Tells PyTorch we're only predicting, not training, so skip tracking gradients (saves memory and time).
- **`mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(...)`** — Finds which position in the token sequence corresponds to `[MASK]`.
- **`top5 = torch.topk(mask_logits, 5, dim=-1)`** — Gets the 5 highest-scoring predictions for the masked position.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch transformers
```

**Steps:**
1. Save the code to a file, e.g. `bert_mlm.py`
2. Run: `python bert_mlm.py`
3. The first run will download the BERT model (~440 MB).

**Expected output:**
```
Input: The capital of France is [MASK].
Top 5 predictions for [MASK]:
  paris           (score: 18.42)
  lyon            (score: 13.87)
  lille           (score: 12.45)
  toulouse        (score: 11.89)
  marseille       (score: 11.52)
```

</details>

### Why the 80/10/10 Split?

If BERT always replaced with `[MASK]`, it would learn that only `[MASK]` positions need predictions, and the model would never see `[MASK]` during fine-tuning (since real text doesn't contain mask tokens). The random replacement and keep-original strategies force the model to maintain good representations for **all** positions, not just masked ones.

## Next Sentence Prediction (NSP)

BERT's original pre-training included a second objective: **Next Sentence Prediction**. Given two segments A and B, the model predicts whether B actually follows A in the original document, or is a random sentence.

```
Input:  [CLS] The cat sat on the mat [SEP] It was very comfortable [SEP]
Label:  IsNext

Input:  [CLS] The cat sat on the mat [SEP] Stock prices rose today [SEP]
Label:  NotNext
```

:::warning[NSP Is Probably Not Helpful]
Later research (RoBERTa, ALBERT) showed that NSP doesn't actually improve downstream performance and may even hurt it. The likely explanation: the model can solve NSP by simply detecting **topic mismatch** rather than learning true coherence. RoBERTa dropped NSP entirely and achieved better results. Most modern encoder models skip this objective.
:::

### BERT Architecture Details

| Property | BERT-Base | BERT-Large |
|----------|-----------|------------|
| Layers | 12 | 24 |
| Hidden dim | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |
| Context length | 512 | 512 |

BERT uses the original Transformer encoder with **post-norm** (LayerNorm after each residual connection). It adds special tokens:

- `[CLS]`: Prepended to every input. Its final hidden state is used as the sequence representation for classification tasks.
- `[SEP]`: Separates segments (e.g., question and passage in QA).
- `[MASK]`: Placeholder for masked tokens during MLM.

```python title="BERT for text classification (fine-tuning)"
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
import torch

# Load pre-trained BERT with a classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# The classification head is a single linear layer on top of [CLS]
# Architecture: BERT encoder → [CLS] hidden state → Linear(768, 2) → softmax

# Fine-tuning example (conceptual)
text = "This movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
labels = torch.tensor([1])  # positive sentiment

outputs = model(**inputs, labels=labels)
print(f"Loss: {outputs.loss:.4f}")
print(f"Logits: {outputs.logits}")  # shape: (1, 2)
```

:::tip[Line-by-Line Walkthrough]
- **`BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`** — Loads BERT with a classification head on top. `num_labels=2` means binary classification (e.g., positive vs negative).
- **`inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)`** — Converts the text into token IDs, adding padding if needed and truncating if the text is too long.
- **`labels = torch.tensor([1])`** — The ground-truth label: 1 = positive sentiment.
- **`outputs = model(**inputs, labels=labels)`** — Runs the model. Because `labels` is provided, it also computes the cross-entropy loss automatically.
- **`outputs.logits`** — The model's raw scores for each class (before softmax). Shape is (1, 2) — one score for "negative" and one for "positive."
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch transformers
```

**Steps:**
1. Save the code to `bert_classify.py`
2. Run: `python bert_classify.py`
3. First run downloads the model (~440 MB).

**Expected output:**
```
Loss: 0.5831
Logits: tensor([[ 0.1234, -0.0567]], grad_fn=<AddmmBackward0>)
```
(Exact numbers vary — this is an untrained classification head, so the loss and logits are essentially random.)

</details>

## T5: The Text-to-Text Framework

T5 (Text-to-Text Transfer Transformer), published by Google in 2020, proposed a radical simplification: **cast every NLP task as a text-to-text problem**. Classification, summarization, translation, question answering — all are treated as generating text from text.

:::info[Text-to-Text Unification]
Instead of adding task-specific heads (a linear layer for classification, a span extractor for QA, etc.), T5 uses a single encoder-decoder model for everything. The task is specified via a **text prefix**:

- **Translation:** `"translate English to German: That is good"` → `"Das ist gut"`
- **Summarization:** `"summarize: long article text..."` → `"short summary"`
- **Classification:** `"sst2 sentence: This movie is great"` → `"positive"`
- **QA:** `"question: Who wrote Hamlet? context: William Shakespeare wrote..."` → `"William Shakespeare"`
:::

### T5 Architecture

T5 uses the original **encoder-decoder** Transformer architecture (unlike GPT's decoder-only or BERT's encoder-only). Key design choices:

- **Relative position embeddings** instead of absolute (using a learned bias added to attention scores)
- **Pre-norm** (LayerNorm before each sub-layer, like GPT-2)
- **No bias terms** in the dense layers
- **Span corruption** for pre-training (described below)

### Span Corruption Pre-Training

Instead of masking individual tokens (like BERT), T5 masks **contiguous spans** and replaces them with sentinel tokens:

```
Input:   "Thank you for inviting me to your party last week"
Corrupt: "Thank you <X> me to your <Y> last week"
Target:  "<X> for inviting <Y> party"
```

The sentinel tokens (`<X>`, `<Y>`, etc.) are unique per span. The model must predict the content of each span in the target sequence.

```python title="T5 for multiple tasks"
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

tasks = [
    "translate English to French: The house is wonderful.",
    "summarize: State authorities dispatched emergency crews Tuesday to "
    "survey the damage after a series of powerful thunderstorms swept "
    "through several southern states, killing at least five people.",
    "sst2 sentence: This is the best movie I have ever seen.",
]

for task_input in tasks:
    inputs = tokenizer(task_input, return_tensors='pt', max_length=512,
                       truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=64)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input:  {task_input[:60]}...")
    print(f"Output: {result}")
    print()
```

:::tip[Line-by-Line Walkthrough]
- **`T5ForConditionalGeneration.from_pretrained('t5-small')`** — Loads the small T5 model (60M parameters), which can handle translation, summarization, and classification all through the same text-to-text interface.
- **`tasks = [...]`** — Three different NLP tasks, each expressed as a text prompt with a prefix ("translate", "summarize", "sst2 sentence") that tells T5 what to do.
- **`inputs = tokenizer(task_input, ..., max_length=512, truncation=True)`** — Converts the text prompt to token IDs, cutting off at 512 tokens if the input is too long.
- **`outputs = model.generate(**inputs, max_new_tokens=64)`** — Generates up to 64 new tokens as the answer. T5 uses beam search by default for generation.
- **`tokenizer.decode(outputs[0], skip_special_tokens=True)`** — Converts the generated token IDs back into readable text, stripping out special tokens like `</s>`.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch transformers sentencepiece
```

**Steps:**
1. Save the code to `t5_tasks.py`
2. Run: `python t5_tasks.py`
3. First run downloads the T5-small model (~242 MB).

**Expected output:**
```
Input:  translate English to French: The house is wonderful....
Output: La maison est merveilleuse.

Input:  summarize: State authorities dispatched emergency crews Tu...
Output: at least five people were killed in a series of thunderstorms.

Input:  sst2 sentence: This is the best movie I have ever seen....
Output: positive
```

</details>

### T5 Model Sizes

| Variant | Parameters | Layers (enc + dec) | d_model |
|---------|-----------|-------------------|---------|
| T5-Small | 60M | 6 + 6 | 512 |
| T5-Base | 220M | 12 + 12 | 768 |
| T5-Large | 770M | 24 + 24 | 1024 |
| T5-3B | 3B | 24 + 24 | 1024 |
| T5-11B | 11B | 24 + 24 | 1024 |

## Architecture Comparison

This is the most important table in this lesson. Understanding these three paradigms is crucial for choosing the right model.

| Property | Encoder-Only (BERT) | Decoder-Only (GPT) | Encoder-Decoder (T5) |
|----------|--------------------|--------------------|---------------------|
| **Attention** | Bidirectional (full) | Causal (left-to-right) | Encoder: bidirectional; Decoder: causal + cross-attention |
| **Pre-training** | Masked language modeling | Next-token prediction | Span corruption / denoising |
| **Strengths** | Understanding, classification, retrieval | Generation, in-context learning | Conditional generation, seq-to-seq |
| **Weaknesses** | Cannot generate text naturally | Cannot use future context | More complex, harder to scale |
| **Key models** | BERT, RoBERTa, DeBERTa, ALBERT | GPT-2/3/4, LLaMA, Mistral | T5, BART, Flan-T5, UL2 |
| **Best for** | Embeddings, search, NER, classification | Chatbots, code gen, general-purpose | Translation, summarization, structured tasks |
| **Scaling trend** | Mostly plateaued at ~1B params | Dominant paradigm at scale (>10B) | Used in specialized systems, not at frontier scale |

:::tip[The Industry Has Converged on Decoder-Only]
As of 2026, nearly all frontier LLMs are **decoder-only**: GPT-4o, Claude 4, Llama 4, Gemini 2, DeepSeek-V3, Mistral, Qwen, Gemma. The reason is pragmatic — decoder-only models are simpler to train, easier to scale, and in-context learning has largely replaced the need for task-specific architectures.

Encoder models (BERT family) are still widely used for **embeddings** and **classification** in production. Encoder-decoder models are increasingly rare at the frontier but remain useful for specific tasks like translation.
:::

## When to Use Which Architecture

### Use Encoder-Only (BERT-like) When:
- You need **dense embeddings** for similarity search or retrieval
- You're doing **token classification** (NER, POS tagging)
- You're doing **sequence classification** (sentiment, spam detection)
- You need **bidirectional context** for understanding
- Latency and model size are constrained (BERT-base is only 110M params)

### Use Decoder-Only (GPT-like) When:
- You need **text generation** (chat, completion, creative writing)
- You want **few-shot** or **zero-shot** capabilities via prompting
- You're building a **general-purpose** system
- You're working at **large scale** (>1B parameters)
- You want to leverage the largest pre-trained models

### Use Encoder-Decoder (T5-like) When:
- You have a clear **input → output** transformation
- You're doing **translation** or **summarization**
- You want to **fine-tune** on a specific task with structured I/O
- You need the encoder to build a deep representation of the input before generating

```python title="Using each architecture for sentiment analysis"
# --- Approach 1: BERT (encoder-only) ---
# Fine-tune a classification head on labeled data
from transformers import pipeline

bert_classifier = pipeline("sentiment-analysis",
                           model="nlptown/bert-base-multilingual-uncased-sentiment")
print("BERT:", bert_classifier("This restaurant was fantastic!"))

# --- Approach 2: GPT (decoder-only) ---
# Use zero-shot prompting — no fine-tuning needed
from transformers import pipeline

gpt_generator = pipeline("text-generation", model="gpt2")
prompt = """Classify the sentiment as positive or negative.
Text: This restaurant was fantastic!
Sentiment:"""
result = gpt_generator(prompt, max_new_tokens=5)
print("GPT:", result[0]['generated_text'].split("Sentiment:")[-1].strip())

# --- Approach 3: T5 (encoder-decoder) ---
# Use the text-to-text format
from transformers import pipeline

t5_pipe = pipeline("text2text-generation", model="t5-small")
print("T5:", t5_pipe("sst2 sentence: This restaurant was fantastic!"))
```

:::tip[Line-by-Line Walkthrough]
- **`pipeline("sentiment-analysis", model="nlptown/...")`** — Loads a BERT model already fine-tuned for sentiment analysis. BERT excels here because it was designed for classification tasks.
- **`pipeline("text-generation", model="gpt2")`** — Loads GPT-2 for text generation. Instead of a specialized classification head, GPT-2 approaches sentiment analysis through prompting — you describe the task in plain text.
- **`prompt = """Classify the sentiment..."""`** — The zero-shot prompt tells GPT-2 what to do. GPT-2 will try to continue the text after "Sentiment:" with words like "positive" or "negative."
- **`pipeline("text2text-generation", model="t5-small")`** — T5 uses its text-to-text format with the `"sst2 sentence:"` prefix to indicate this is a sentiment classification task.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch transformers sentencepiece
```

**Steps:**
1. Save the code to `sentiment_comparison.py`
2. Run: `python sentiment_comparison.py`
3. First run downloads three models (BERT, GPT-2, T5-small).

**Expected output:**
```
BERT: [{'label': '5 stars', 'score': 0.73}]
GPT:  positive
T5:  [{'generated_text': 'positive'}]
```
(BERT gives the most reliable result because it was fine-tuned specifically for this task.)

</details>

## Variants and Descendants

### RoBERTa (2019)
Facebook's "Robustly Optimized BERT" showed that BERT was severely undertrained. By removing NSP, training longer with larger batches, and using dynamic masking (different mask each epoch), RoBERTa significantly outperformed BERT on all benchmarks.

### DeBERTa (2021)
Microsoft's DeBERTa (Decoupled attention with Enhanced Mask decoder) improved on BERT/RoBERTa by:
- **Disentangled attention**: separate content and position embeddings that interact through two separate attention matrices
- An enhanced mask decoder for better MLM pre-training

DeBERTa-v3 remains one of the best encoder models for classification and NLU tasks.

### BART (2019)
Facebook's BART is an encoder-decoder model that uses a **denoising** pre-training objective. The input is corrupted (token masking, deletion, permutation, rotation, span infilling) and the decoder reconstructs the original. It's particularly good at summarization and generation tasks.

### Flan-T5 (2022)
Google's Flan-T5 is T5 fine-tuned on a large collection of tasks with **instruction tuning**. This dramatically improves zero-shot and few-shot performance, bridging the gap between encoder-decoder models and the prompting abilities of decoder-only models.

## Summary

| Model | Architecture | Pre-training | Best Use Case |
|-------|-------------|-------------|---------------|
| BERT | Encoder-only | MLM + NSP | Embeddings, classification |
| RoBERTa | Encoder-only | MLM (improved) | Same as BERT, better performance |
| DeBERTa | Encoder-only | MLM (disentangled) | SOTA classification and NLU |
| GPT | Decoder-only | Next-token prediction | Generation, chat, reasoning |
| T5 | Encoder-decoder | Span corruption | Translation, summarization |
| BART | Encoder-decoder | Denoising | Summarization, generation |
| Flan-T5 | Encoder-decoder | Span corruption + instruction tuning | Zero-shot structured tasks |

---

## Exercises

:::tip[BERT vs GPT Representations — beginner]

Generate sentence embeddings using both BERT (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and GPT-2 for the following sentence pairs. Compare cosine similarities. Which model produces better similarity scores and why?

1. "The cat sat on the mat" / "A feline rested on a rug"
2. "The stock market crashed" / "The cat sat on the mat"
3. "I love programming" / "Coding is my passion"

<div>
**Solution approach:**
BERT-based models produce significantly better similarity scores because they use bidirectional context and are often specifically trained for semantic similarity. GPT-2's left-to-right attention means the embedding of any token doesn't incorporate information from tokens after it, making pooled representations less meaningful for similarity tasks. This is why encoder models dominate the embedding/retrieval space.
<details>
<summary>Hints</summary>

1. Use model.encode() from sentence-transformers for BERT
2. For GPT, you'll need to mean-pool the hidden states
3. Compare the similarity scores for clearly related/unrelated sentences

</details>

:::

:::tip[T5 Task Reformulation — intermediate]

Reformulate these tasks as text-to-text problems suitable for T5:

1. Named Entity Recognition: identify person names in "Barack Obama visited Angela Merkel in Berlin"
2. Grammatical error correction: fix "She don't likes swimming"
3. Textual entailment: Does "All cats are mammals" entail "My cat Whiskers is a mammal"?

Write the input and expected output for each.

<div>
**Solution:**

1. **NER:** Input: `"ner: Barack Obama visited Angela Merkel in Berlin"` → Output: `"Barack Obama: person, Angela Merkel: person, Berlin: location"`

2. **GEC:** Input: `"grammar: She don't likes swimming"` → Output: `"She doesn't like swimming"`

3. **Entailment:** Input: `"mnli premise: All cats are mammals. hypothesis: My cat Whiskers is a mammal."` → Output: `"entailment"`
<details>
<summary>Hints</summary>

1. All tasks need an input text and output text
2. Use descriptive prefixes
3. Think about how to represent structured outputs as text

</details>

:::

:::tip[Build a Simple MLM Training Loop — advanced]

Implement a masked language modeling training loop from scratch using PyTorch. Use a small Transformer encoder (4 layers, d_model=256) and train on a small text corpus. Don't use HuggingFace's `DataCollatorForLanguageModeling` — implement the masking logic yourself.

<div>
**Solution sketch:**

```python
def create_mlm_batch(input_ids, vocab_size, mask_token_id,
                     mask_prob=0.15):
    labels = input_ids.clone()
    # Random mask positions
    mask = torch.bernoulli(torch.full(input_ids.shape, mask_prob)).bool()
    # Don't mask special tokens (positions 0 and -1)
    mask[:, 0] = False
    
    labels[~mask] = -100  # only compute loss on masked tokens
    
    rand = torch.rand(input_ids.shape)
    # 80% → [MASK]
    input_ids[mask & (rand < 0.8)] = mask_token_id
    # 10% → random token
    random_tokens = torch.randint(0, vocab_size, input_ids.shape)
    input_ids[mask & (rand >= 0.8) & (rand < 0.9)] = \
        random_tokens[mask & (rand >= 0.8) & (rand < 0.9)]
    # 10% → unchanged
    
    return input_ids, labels
```

:::tip[Line-by-Line Walkthrough]
- **`labels = input_ids.clone()`** — Makes a copy of the original tokens. These will serve as the "answer key" — the model tries to predict these at masked positions.
- **`mask = torch.bernoulli(torch.full(input_ids.shape, mask_prob)).bool()`** — Randomly selects ~15% of token positions by flipping a biased coin at each position (15% chance of True).
- **`labels[~mask] = -100`** — Sets non-masked positions to −100 in the labels. PyTorch's cross-entropy loss ignores positions with this value, so the model is only graded on masked tokens.
- **`input_ids[mask & (rand < 0.8)] = mask_token_id`** — For 80% of masked positions, replaces the token with the special `[MASK]` token.
- **`input_ids[mask & (rand >= 0.8) & (rand < 0.9)] = random_tokens[...]`** — For 10% of masked positions, replaces with a random vocabulary token. This teaches the model to handle noisy input.
- The remaining 10% of masked positions are left unchanged — the model must still predict them, which teaches it to maintain good representations even for visible tokens.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Add `import torch` at the top of your script.
2. Create dummy input: `input_ids = torch.randint(0, 30000, (2, 128))` and call `create_mlm_batch(input_ids, vocab_size=30522, mask_token_id=103)`.

**Expected output:** Two tensors — modified `input_ids` (with ~15% replaced by `[MASK]`, random tokens, or unchanged) and `labels` (original tokens at masked positions, −100 everywhere else).

</details>

<details>
<summary>Hints</summary>

1. Randomly select 15% of tokens
2. Apply the 80/10/10 masking strategy
3. Only compute loss on masked positions
4. Use cross-entropy loss with ignore_index=-100

</details>

:::

---

## Resources

- **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** _(paper)_ by Devlin et al. — The original BERT paper introducing masked language modeling and bidirectional pre-training.

- **[Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)** _(paper)_ by Raffel et al. — The T5 paper — a systematic study of transfer learning that introduced the text-to-text framework.

- **[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)** _(paper)_ by Liu et al. — Showed that BERT was significantly undertrained and established stronger baselines.

- **[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)** _(tutorial)_ by Jay Alammar — Visual walkthrough of BERT, ELMo, and transfer learning in NLP.

- **[HuggingFace Transformers Course: Chapter on Models](https://huggingface.co/learn/nlp-course/chapter1/4)** _(course)_ by HuggingFace — Hands-on introduction to encoder, decoder, and encoder-decoder models with code examples.

- **[BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)** _(paper)_ by Lewis et al. — Facebook's encoder-decoder model combining bidirectional encoding with autoregressive decoding.
