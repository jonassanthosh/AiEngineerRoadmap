---
slug: essential-research-papers
title: "The Essential AI Research Papers Reading List"
authors: [default]
tags: [resources, papers, research]
---

The papers that defined modern AI, ordered so each one builds on the last. Read these and you'll understand how we got from neural networks to GPT-4.

<!-- truncate -->

## Foundational Papers

### Attention & Transformers

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017) — The paper that started it all. Introduced the Transformer architecture.
- **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** (Devlin et al., 2018) — Masked language modeling and bidirectional pretraining.
- **[Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** (Radford et al., 2019) — Showed that scale + pretraining = emergent capabilities.
- **[Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)** (Brown et al., 2020) — In-context learning at scale.

### Scaling & Training

- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** (Kaplan et al., 2020) — Power-law relationships between compute, data, and performance.
- **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)** (Hoffmann et al., 2022) — Showed most LLMs were undertrained on data relative to their parameter count.
- **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)** (Touvron et al., 2023) — Open-weight models matching proprietary performance.

---

## Alignment & Fine-Tuning

- **[Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)** (Ouyang et al., 2022) — RLHF applied to GPT-3, the blueprint for ChatGPT.
- **[Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)** (Rafailov et al., 2023) — RLHF without a reward model.
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** (Bai et al., 2022) — Anthropic's approach to self-supervised alignment.
- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al., 2021) — Parameter-efficient fine-tuning that changed everything.
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** (Dettmers et al., 2023) — Fine-tune a 65B model on a single GPU.

---

## Inference & Efficiency

- **[FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022) — IO-aware attention that made long contexts practical.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** (Dao, 2023) — Faster, better parallelism.
- **[GQA: Training Generalized Multi-Query Transformers](https://arxiv.org/abs/2305.13245)** (Ainslie et al., 2023) — Grouped-query attention used in Llama 2/3/4.
- **[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)** (Kwon et al., 2023) — Virtual memory for KV-cache. The foundation of modern LLM serving.
- **[GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)** (Frantar et al., 2022) — One-shot weight quantization to 3-4 bits.
- **[AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)** (Lin et al., 2023) — Protecting salient weights during quantization.

---

## Architecture Innovations

- **[Mixture of Experts (Switch Transformer)](https://arxiv.org/abs/2101.03961)** (Fedus et al., 2021) — Sparse expert routing for scaling.
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)** (Jiang et al., 2024) — MoE done right at Mistral.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** (Gu & Dao, 2023) — The leading alternative to Transformer attention.
- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)** (Su et al., 2021) — RoPE, now used in almost every modern LLM.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** (DeepSeek AI, 2024) — Open model matching frontier performance, MoE + MLA attention.

---

## Reasoning & Agents

- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)** (Wei et al., 2022) — "Let's think step by step" unlocks reasoning in LLMs.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)** (Yao et al., 2023) — Deliberate problem-solving with LLMs.
- **[ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)** (Yao et al., 2022) — LLMs as reasoning agents.
- **[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)** (Schick et al., 2023) — LLMs learning to call APIs.
- **[DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)** (DeepSeek AI, 2025) — Reasoning via reinforcement learning without supervised fine-tuning.

---

## Evaluation

- **[MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)** (Hendrycks et al., 2020) — The standard knowledge benchmark.
- **[HumanEval: Evaluating Code Generation](https://arxiv.org/abs/2107.03374)** (Chen et al., 2021) — Code generation benchmark from OpenAI.
- **[Chatbot Arena](https://lmarena.ai/)** — Live human preference rankings of LLMs.

---

## How to Use This List

**If you're in Month 1-2:** Read the Transformer paper with Jay Alammar's illustrated guide side by side.

**If you're in Month 3-4:** Read GPT-2, GPT-3, BERT, and the scaling laws papers. Then implement nanoGPT.

**If you're in Month 5:** Read LoRA, DPO, FlashAttention, and vLLM. These are your daily tools.

**If you're in Month 6:** Read everything in Architecture Innovations and Reasoning. Then pick one paper and reimplement it.
