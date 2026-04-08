---
sidebar_position: 6
slug: career-portfolio
title: "Building Your AI Engineering Career"
---


# Building Your AI Engineering Career

:::info[What You'll Learn]
- AI engineering roles and how they differ (ML engineer, LLM engineer, research engineer, infra)
- Building a portfolio that demonstrates real skills
- Technical interview preparation
- Navigating the AI job market
:::

:::note[Prerequisites]
None specific — this lesson is accessible at any point, but most useful after completing the full curriculum.
:::

**Estimated time:** Reading: ~25 min | Exercises: ~4 hours

You've spent six months building deep technical skills in AI and LLM engineering. This lesson helps you translate those skills into a career — whether you're entering the field for the first time, transitioning from another engineering role, or leveling up within AI.

## AI Engineering Roles

The AI job market has evolved rapidly. Here are the main roles and what differentiates them.

### Research Engineer

**Focus:** Implementing and iterating on novel research ideas.

**Day-to-day:**
- Reading and implementing papers
- Running experiments on large GPU clusters
- Analyzing results and proposing modifications
- Writing code that may become a paper or production system

**Key skills:** Deep understanding of ML theory, strong PyTorch skills, ability to read and implement papers quickly, comfort with ambiguity (most experiments fail).

**Typical employers:** Research labs (DeepMind, FAIR, Anthropic, OpenAI), university labs, research teams at large tech companies.

### ML Engineer

**Focus:** Building and maintaining production ML systems.

**Day-to-day:**
- Training and fine-tuning models
- Building data pipelines
- Optimizing model serving (latency, throughput, cost)
- Monitoring model performance in production
- A/B testing model improvements

**Key skills:** Strong software engineering, MLOps (experiment tracking, CI/CD, monitoring), distributed training, model optimization.

**Typical employers:** Any company using ML in production — tech companies, startups, finance, healthcare.

### LLM/AI Engineer

**Focus:** Building applications powered by large language models.

**Day-to-day:**
- Designing prompts and RAG pipelines
- Fine-tuning models for specific tasks
- Building evaluation pipelines
- Integrating LLMs with existing systems
- Managing cost and latency tradeoffs

**Key skills:** Prompt engineering, RAG architecture, fine-tuning (LoRA, QLoRA), evaluation methodology, API design.

**Typical employers:** AI-native startups, product teams at tech companies, consulting firms.

### AI Infrastructure Engineer

**Focus:** Building the platform that other teams use to train and deploy models.

**Day-to-day:**
- Managing GPU clusters and training infrastructure
- Building and maintaining serving systems (vLLM, TensorRT)
- Implementing distributed training frameworks
- Optimizing hardware utilization
- Building internal ML platforms

**Key skills:** Systems engineering, CUDA/GPU programming, distributed systems, Kubernetes, networking.

**Typical employers:** Cloud providers, AI labs, large tech companies with significant ML workloads.

:::tip[The Lines Are Blurring]
In practice, especially at startups, you'll wear multiple hats. An "AI engineer" might do fine-tuning, build RAG pipelines, optimize inference, *and* deploy to production. The most valuable engineers are those who can work across the stack. Don't pigeonhole yourself into one category too early.
:::

## Building a Portfolio

Your portfolio is the most powerful signal in an AI job search. It shows what you can actually build — not just what you've read about.

### GitHub Projects

Your GitHub profile should demonstrate range and depth. Here's a tiered approach:

**Tier 1: Showcase projects (2-3)**
These are polished, well-documented projects that demonstrate significant capability.
- A model you trained from scratch with documented results
- An end-to-end application (RAG system, fine-tuned model API, agent framework)
- An implementation of a paper with clear comparison to the original results

**Tier 2: Learning projects (5-10)**
These show breadth and curiosity.
- Implementations of key algorithms from this course
- Exploratory notebooks with analysis
- Contributions to open-source projects

**Tier 3: Utility projects**
Tools you've built that others might find useful.
- Evaluation frameworks
- Data processing pipelines
- CLI tools for model interaction

```python title="Example: README structure for a portfolio project"
readme_template = """
# Project Title

One-line description of what this does and why it matters.

## Results

| Model | Accuracy | Latency (ms) | Memory (GB) |
|-------|----------|-------------|-------------|
| Baseline (GPT-2) | 72.3% | 45 | 1.2 |
| **Ours** | **78.1%** | **32** | **0.8** |

> Put results FIRST. Reviewers will decide whether to read further
> based on whether the results are interesting.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
python evaluate.py --checkpoint outputs/best.pt
```

## Architecture

[Diagram or brief description of the approach]

## Key Findings

- Finding 1: Brief description with supporting data
- Finding 2: Brief description with supporting data
- Finding 3: Brief description with supporting data

## Reproduction

Detailed instructions for reproducing results, including:
- Hardware requirements
- Training time
- Data preparation steps

## Citation

If you built on someone else's work, cite it properly.
"""
print(readme_template)
```

:::tip[Line-by-Line Walkthrough]
- **`## Results` (at the top)** — Results go FIRST in a good README. Reviewers decide whether to read further based on whether your results are interesting — don't bury the lede.
- **`| Baseline (GPT-2) | 72.3% | ... | **Ours** | **78.1%** |`** — Bold formatting highlights your model's improvements over the baseline. Clear numbers tell the story immediately.
- **`## Quick Start`** — Three commands (install, train, evaluate) that let anyone reproduce your work in minutes. If it takes more than 5 minutes to get running, you'll lose most readers.
- **`## Key Findings`** — Bullet points with supporting data. Each finding should be a sentence with a number: "RMSNorm reduced training time by 12% with no perplexity change."
- **`## Reproduction`** — Hardware requirements, training time, and data prep steps. Without this, nobody can verify your results.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
No external packages needed — this is a template string.

**Steps:**
1. Save the code to `generate_readme.py`
2. Run: `python generate_readme.py`
3. Copy the output into your project's `README.md` and fill in the details

**Expected output:**
Prints a complete README template that you customize with your project's specific results, architecture description, and reproduction instructions.

</details>

### Technical Blog

Writing about what you learn serves multiple purposes: it deepens your understanding, builds your reputation, and creates searchable artifacts that recruiters find.

**What to write about:**
- Paper implementations with your own insights
- Debugging war stories (these are gold — everyone has them, few write them down)
- Performance optimization case studies with numbers
- Comparisons between different approaches to the same problem
- Tutorials that teach something you struggled with

**Where to publish:**
- Your own blog (GitHub Pages, Hugo, or Docusaurus — meta!)
- Medium or Substack for broader reach
- Company engineering blog if applicable

:::info[Write for Your Past Self]
The best technical blog posts answer a question you yourself struggled with. If you spent 3 hours debugging a CUDA memory issue, write a 15-minute post about the solution. Your past self would have been grateful — and so will thousands of others who Google the same error.
:::

### Contributing to Papers

You don't need to be at a research lab to contribute to papers.

- **Reproduce existing papers** and publish your results (even negative results are valuable).
- **Extend a paper** with a small experiment the authors didn't try.
- **Collaborate** with academic researchers who need engineering help.
- **Submit to workshops** at major conferences (NeurIPS, ICML, ACL workshops have lower bars than main conferences).

## Contributing to Open Source

Open-source contributions are the strongest signal of practical engineering ability. Here's how to get started meaningfully.

### Where to Contribute

| Project | What It Does | Good First Issues |
|---------|-------------|-------------------|
| **HuggingFace Transformers** | Model library | Adding new models, fixing tokenizers, improving docs |
| **vLLM** | LLM serving engine | Performance optimization, new model support, testing |
| **LitGPT** | LLM training/fine-tuning | Data loading, new features, benchmarks |
| **lm-evaluation-harness** | Benchmarking | Adding new tasks, fixing edge cases |
| **llama.cpp** | CPU/GPU inference | Optimizations, quantization formats, model support |

### How to Contribute Effectively

1. **Start by using the project.** File bug reports. Answer questions in issues. This builds context and reputation.
2. **Read the contributing guide.** Every major project has one. Follow it exactly.
3. **Pick issues labeled "good first issue" or "help wanted."**
4. **Start small.** Documentation fixes and test additions are valid first contributions.
5. **Communicate early.** Comment on the issue before you start coding. Describe your approach and ask for feedback.

```python title="Example: Adding a benchmark task to lm-evaluation-harness"
# File: lm_eval/tasks/my_custom_task.yaml
# This is the format for adding a new evaluation task
task_config = """
task: my_domain_knowledge
dataset_path: my_org/my_dataset
dataset_name: null
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Question: {{question}}\\nAnswer:"
doc_to_target: "{{answer}}"
doc_to_choice: "{{choices}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
  description: "Custom domain knowledge benchmark"
"""
print(task_config)

# After creating the YAML, test it:
# lm_eval --model hf --model_args pretrained=gpt2 --tasks my_domain_knowledge --num_fewshot 5
```

:::tip[Line-by-Line Walkthrough]
- **`task: my_domain_knowledge`** — The unique name for your benchmark task. This is what you pass to `--tasks` on the command line.
- **`dataset_path: my_org/my_dataset`** — Points to a HuggingFace dataset. You can upload your own or use any existing one.
- **`output_type: multiple_choice`** — Tells lm-eval this is a multiple choice task (other options: `generate_until`, `loglikelihood`).
- **`doc_to_text: "Question: {{question}}\\nAnswer:"`** — Jinja template that formats each example into a prompt. `{{question}}` is filled from the dataset.
- **`doc_to_choice: "{{choices}}"`** — The list of answer options from the dataset.
- **`metric_list`** — Which metrics to compute: `acc` (raw accuracy) and `acc_norm` (length-normalized accuracy, which avoids bias toward shorter answers).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install lm-eval
```

**Steps:**
1. Save the YAML configuration to `lm_eval/tasks/my_custom_task.yaml` (inside the lm-eval installation)
2. Upload your dataset to HuggingFace Hub or reference an existing one
3. Test:
```bash
lm_eval --model hf --model_args pretrained=gpt2 --tasks my_domain_knowledge --num_fewshot 5
```

**Expected output:**
A results table showing accuracy and normalized accuracy for your custom benchmark, similar to how standard benchmarks are reported.

</details>

## Interview Preparation

AI engineering interviews typically cover three areas: system design, ML theory, and coding.

### System Design

You'll be asked to design an ML system end-to-end. Practice these scenarios:

1. **Design a RAG-based Q&A system** — embedding pipeline, retrieval, re-ranking, generation, evaluation
2. **Design a model serving platform** — batching, caching, auto-scaling, A/B testing, monitoring
3. **Design a training pipeline** — data ingestion, preprocessing, distributed training, experiment tracking, model registry
4. **Design a content moderation system** — classifiers, human-in-the-loop, feedback loops, latency requirements

**Framework for answering:**
1. Clarify requirements (scale, latency, accuracy targets)
2. Propose a high-level architecture
3. Deep-dive into 2-3 components
4. Discuss tradeoffs and alternatives
5. Address failure modes and monitoring

### ML Theory

Expect questions on:
- Attention mechanism (compute shapes, complexity, variants)
- Training dynamics (learning rate schedules, gradient issues, loss functions)
- Fine-tuning methods (LoRA, full fine-tuning, RLHF, DPO)
- Evaluation (metrics, benchmarks, contamination)
- Scaling laws (Chinchilla, compute-optimal training)

```python title="Common interview questions: quick implementations"
import torch
import torch.nn.functional as F

# Q: "Implement scaled dot-product attention"
def attention(Q, K, V, mask=None):
    # Q, K, V: [B, H, T, D]
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# Q: "Implement beam search"
def beam_search(model, input_ids, beam_width=3, max_length=50):
    # Each beam: (log_prob, token_ids)
    beams = [(0.0, input_ids.tolist())]

    for _ in range(max_length):
        candidates = []
        for score, tokens in beams:
            if tokens[-1] == 2:  # EOS token
                candidates.append((score, tokens))
                continue
            input_tensor = torch.tensor([tokens])
            with torch.no_grad():
                logits = model(input_tensor)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            top_k = log_probs.topk(beam_width)
            for lp, idx in zip(top_k.values, top_k.indices):
                candidates.append((score + lp.item(), tokens + [idx.item()]))

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[1][-1] == 2 for b in beams):
            break

    return beams[0][1]

# Q: "What's the difference between LoRA and full fine-tuning?"
# LoRA freezes the pretrained weights and adds low-rank update matrices:
# W' = W + BA where B is [d, r] and A is [r, d], with r << d
# This reduces trainable parameters from d² to 2dr
# Example: d=4096, r=16 → 4096² = 16.7M vs 2×4096×16 = 131K (128x reduction)
```

:::tip[Line-by-Line Walkthrough]
- **`scores = Q @ K.transpose(-2, -1) / d_k ** 0.5`** — The core of attention: dot product of queries and keys, scaled down by the square root of the head dimension to prevent huge values that would saturate softmax.
- **`scores.masked_fill(mask == 0, float("-inf"))`** — Applies the causal mask: future positions become negative infinity, so after softmax they become zero — preventing the model from "seeing the future."
- **`weights = F.softmax(scores, dim=-1)`** — Converts raw scores into a probability distribution over all positions.
- **`beams = [(0.0, input_ids.tolist())]`** — Beam search starts with a single beam at log-probability 0.
- **`candidates.append((score + lp.item(), tokens + [idx.item()]))`** — Each beam expands into `beam_width` candidates by appending each top-scoring next token.
- **`beams = sorted(candidates, ...)[:beam_width]`** — Keeps only the top beams by total score, pruning the rest.
- **`W' = W + BA` (in comments)** — LoRA's key insight: instead of updating a huge d×d weight matrix, learn two small matrices B (d×r) and A (r×d) where r is tiny (e.g., 16). This reduces trainable parameters by 128x.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `interview_prep.py`
2. Test the attention function:
```python
Q = K = V = torch.randn(1, 4, 16, 32)
out = attention(Q, K, V)
print(f"Attention output: {out.shape}")  # [1, 4, 16, 32]
```
3. Beam search requires a trained model to produce meaningful output

**Expected output:**
```
Attention output: torch.Size([1, 4, 16, 32])
```

</details>

### Coding

AI coding interviews vary. Some are standard algorithms (LeetCode-style), others are ML-specific:
- Implement a component from scratch (attention, tokenizer, loss function)
- Debug a training loop
- Optimize a data pipeline
- Write evaluation code

:::tip[The Most Important Interview Skill]
Communicate your thinking out loud. Interviewers care more about your problem-solving process than the final answer. State your assumptions, explain tradeoffs, and ask clarifying questions. The best candidates sound like collaborative engineering partners, not coding contest participants.
:::

## Staying Current

AI moves faster than any other field in technology. Here's how to keep up without drowning.

### Daily (15 minutes)
- Skim Twitter/X for trending AI topics — follow key researchers and practitioners
- Check the top 3-5 papers on Hugging Face's daily papers digest

### Weekly (2 hours)
- Read 1-2 papers deeply (using the three-pass method from Lesson 1)
- Listen to an AI podcast (Gradient Dissent, Latent Space, The TWIML AI Podcast)
- Check release notes for key libraries (Transformers, vLLM, PyTorch)

### Monthly
- Attend a local or virtual meetup
- Write a blog post or build a small project exploring a new technique
- Review your learning goals and adjust

### Key Conferences

| Conference | Focus | Timing |
|-----------|-------|--------|
| **NeurIPS** | Broad ML | December |
| **ICML** | Broad ML | July |
| **ICLR** | Representation Learning | May |
| **ACL/EMNLP** | NLP | Varies |
| **CVPR** | Computer Vision | June |

:::warning[Avoid Hype-Driven Learning]
Don't chase every new model release or framework. Focus on fundamentals that compound: understanding attention mechanisms is more valuable than knowing the API of this month's hottest wrapper library. When you see a new technique, ask: "Is this a fundamental advance, or just a new way to package existing ideas?"
:::

:::tip[Exercise 1: Portfolio Project Plan — beginner]

Design a portfolio project:

1. Choose a problem that interests you and has available data.
2. Write a 1-page project plan: problem statement, approach, data, evaluation metrics, timeline.
3. Identify which skills from this course the project demonstrates.
4. Write the README *before* you start coding (this forces clear thinking about goals and results).
5. Build it.

<details>
<summary>Hints</summary>

1. Pick a project that demonstrates multiple skills (training, evaluation, deployment)
2. The project should be completable in 2-4 weeks
3. Include a clear metric you're optimizing for

</details>

:::

:::tip[Exercise 2: Open Source Contribution — intermediate]

Make your first meaningful open-source contribution:

1. Choose one project from the table above (or any ML/AI project you use).
2. Clone it, set up the dev environment, and run the test suite.
3. Find an issue to work on — start with documentation, tests, or bug fixes.
4. Submit a pull request following the project's contribution guidelines.
5. Respond to reviewer feedback and iterate until merged.

<details>
<summary>Hints</summary>

1. Start by looking at 'good first issue' labels on GitHub
2. Documentation improvements count and are often the easiest entry point
3. Read the CONTRIBUTING.md file before doing anything

</details>

:::

:::tip[Exercise 3: Mock System Design — advanced]

Practice a system design interview:

1. Set a 45-minute timer.
2. Design one of these systems: (a) a real-time content moderation pipeline for a social media platform, (b) a multi-tenant LLM serving platform with per-customer fine-tuned models, or (c) a document processing pipeline that extracts structured data from PDFs using LLMs.
3. Write up your design with diagrams, component descriptions, and tradeoff analysis.
4. Have a peer review your design and ask follow-up questions.

<details>
<summary>Hints</summary>

1. Set a 45-minute timer
2. Start by clarifying requirements before jumping into architecture
3. Draw a diagram — even on paper — before describing components

</details>

:::

## Resources

- **[Chip Huyen's ML Interviews Book](https://huyenchip.com/ml-interviews-book/)** _(book)_ by Chip Huyen — Comprehensive guide to ML interviews — covers all major companies and question types.

- **[Latent Space Podcast](https://www.latent.space/podcast)** _(course)_ by swyx & Alessio — AI engineering podcast covering the latest developments — great for staying current.

- **[Hugging Face Daily Papers](https://huggingface.co/papers)** _(tool)_ — Curated daily digest of the most impactful new ML papers with community discussion.

- **[MLOps Community](https://mlops.community/)** _(course)_ — Community of ML engineers sharing production ML knowledge — meetups, Slack, newsletter.

- **[Full Stack Deep Learning](https://fullstackdeeplearning.com/)** _(course)_ by Pieter Abbeel et al. — Course covering the full ML lifecycle: project planning, data management, training, deployment.

- **[awesome-production-machine-learning](https://github.com/EthicalML/awesome-production-machine-learning)** _(tool)_ — Curated list of production ML tools, frameworks, and resources.
