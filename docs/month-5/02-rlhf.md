---
sidebar_position: 2
slug: rlhf
title: "RLHF: Reinforcement Learning from Human Feedback"
---


# RLHF: Reinforcement Learning from Human Feedback

A language model trained on internet text can generate fluent text — but it can also produce harmful, dishonest, or unhelpful responses. **RLHF** is the training technique that transforms a raw language model into an assistant that follows instructions, refuses harmful requests, and provides genuinely useful answers. When OpenAI released ChatGPT in late 2022, it demonstrated the power of this approach to a global audience — and RLHF has since become a foundational ingredient behind ChatGPT, Claude, and every modern chat model.

## The Alignment Problem

A pretrained LLM has one objective: predict the next token. This objective doesn't distinguish between helpful and harmful text, truth and falsehood, or clear and confusing explanations. The model faithfully reflects its training data, including all the biases, toxicity, and misinformation it contains.

:::info[Why Next-Token Prediction Isn't Enough]
Ask a pure pretrained model "How do I make a cake?" and it might:
- Continue with a recipe (helpful)
- Continue with a story about someone making a cake (creative, but not what you wanted)
- Generate a rambling, poorly structured response (unhelpful)
- Produce toxic content if the prompt is adversarial (harmful)

The model doesn't have an objective for being **helpful, harmless, and honest** — the "HHH" criteria. RLHF adds that objective.
:::

## The Three-Stage RLHF Pipeline

The standard RLHF pipeline, as described in the InstructGPT paper (Ouyang et al., 2022), has three stages:

```
Stage 1: SFT                Stage 2: Reward Model        Stage 3: PPO
┌────────────────┐          ┌────────────────┐           ┌────────────────┐
│ Pretrained LLM │          │ Human ranks    │           │ RL fine-tunes  │
│       +        │ ──────►  │ model outputs  │ ──────►   │ SFT model to   │
│ Human-written  │          │       ↓        │           │ maximize reward│
│ demonstrations │          │ Train reward   │           │ model score    │
│       ↓        │          │ model          │           │                │
│ Supervised     │          │                │           │ (with KL       │
│ fine-tuning    │          │                │           │  penalty)      │
└────────────────┘          └────────────────┘           └────────────────┘
```

## Stage 1: Supervised Fine-Tuning (SFT)

SFT is the simplest stage: take the pretrained model and fine-tune it on a dataset of **(prompt, ideal response)** pairs written by human annotators.

```python title="SFT training with TRL"
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# SFT datasets contain instruction-response pairs
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

def format_instruction(example):
    messages = example["messages"]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_instruction)

sft_config = SFTConfig(
    output_dir="./sft-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True,
    max_seq_length=2048,
    packing=True,
    logging_steps=25,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
# trainer.train()
```

:::tip[Line-by-Line Walkthrough]
- **`AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")`** — Loads the pretrained model with automatic data type selection and GPU placement.
- **`tokenizer.pad_token = tokenizer.eos_token`** — Many models don't have a dedicated padding token. This sets the end-of-sequence token to double as the pad token, which is needed for batched training.
- **`load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")`** — Loads 5,000 examples from the UltraChat dataset, which contains multi-turn conversations.
- **`tokenizer.apply_chat_template(messages, tokenize=False)`** — Converts a list of message dicts (role + content) into the exact text format the model expects, including special tokens.
- **`packing=True`** — Concatenates short examples into longer sequences to fill up the context window efficiently, reducing wasted computation on padding.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers trl datasets torch accelerate
```

**Steps:**
1. Save to `sft_training.py`
2. Log into HuggingFace: `huggingface-cli login`
3. Run: `python sft_training.py`

**Expected output:**
The code loads the model and prepares the dataset. The actual `trainer.train()` call is commented out. If uncommented, it will show training progress with loss decreasing over steps.

</details>

After SFT, the model follows instructions reasonably well, but its responses are often verbose, inconsistent in quality, or subtly off. Stages 2 and 3 refine behavior further using human preferences.

## Stage 2: Reward Model Training

The reward model learns to score model outputs based on human preferences. Human annotators are shown a prompt and two (or more) model responses, and they rank them from best to worst. The reward model is then trained to assign higher scores to preferred responses.

:::note[Bradley-Terry Preference Model]
Given a prompt \(x\) and two responses \(y_w\) (preferred) and \(y_l\) (rejected), the reward model \(r_\theta\) is trained with:

:::info[Plain English: What Is This Formula Doing?]
Imagine you're teaching a food critic (the reward model) what "good" and "bad" food taste like. You give them two dishes at a time and tell them which one people prefer. Over many tastings, the critic learns to score dishes on their own. This formula says: "adjust the critic's scoring so that the preferred dish always scores higher than the rejected dish."
:::

\[
\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]
\]

**Reading the formula:** *L(θ)* is the loss we're trying to minimize (lower is better). *E* means "average over all examples." *x* is a prompt, *y_w* is the response humans preferred (the "winner"), and *y_l* is the response humans rejected (the "loser"). *r_θ(x, y)* is the reward model's score for response *y* given prompt *x*. *σ* is the sigmoid function that squashes values between 0 and 1. The loss pushes *r_θ(x, y_w)* to be higher than *r_θ(x, y_l)* — making the preferred response score higher.

This is the **Bradley-Terry model** — the probability that \(y_w\) is preferred over \(y_l\) is modeled as:

:::info[Plain English: What Is This Formula Doing?]
This formula says: "the probability that response A beats response B equals the sigmoid of their score difference." If A scores way higher than B, the probability is near 1 (almost certain A is better). If they're tied, the probability is 0.5 (a coin flip). It's the same math used to rank chess players — bigger rating gap means more predictable outcomes.
:::

\[
P(y_w \succ y_l \mid x) = \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right)
\]

**Reading the formula:** *P(y_w ≻ y_l | x)* is the probability that response *y_w* is preferred over *y_l* given prompt *x*. *σ(...)* is the sigmoid function. *r_θ(x, y_w) - r_θ(x, y_l)* is the score difference — the bigger the gap, the more confident we are that *y_w* is better.

where \(\sigma\) is the sigmoid function. The loss maximizes the probability of the human-preferred ranking.
:::

The reward model is typically initialized from the SFT model (or a similar pretrained model), with the language modeling head replaced by a scalar output head.

```python title="Reward model training with TRL"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# Reward model: same architecture as base LLM, but with a scalar head
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    num_labels=1,                          # Single scalar reward output
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Preference dataset: each example has (prompt, chosen, rejected)
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

# The dataset has 'chosen' and 'rejected' text completions
print(f"Sample chosen (first 200 chars):  {dataset[0]['chosen'][:200]}")
print(f"Sample rejected (first 200 chars): {dataset[0]['rejected'][:200]}")

reward_config = RewardConfig(
    output_dir="./reward-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=25,
    max_length=512,
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
# trainer.train()
```

:::tip[Line-by-Line Walkthrough]
- **`AutoModelForSequenceClassification.from_pretrained(..., num_labels=1)`** — Loads the same LLM architecture but replaces the language modeling head (which predicts next tokens) with a single-number output head (which produces a reward score).
- **`load_dataset("Anthropic/hh-rlhf", split="train[:5000]")`** — Loads 5,000 human preference examples from Anthropic's dataset. Each example contains a "chosen" (preferred) and "rejected" response to the same prompt.
- **`learning_rate=1e-5`** — A very low learning rate. The reward model needs to learn subtle distinctions between good and bad responses without forgetting the language understanding it inherited from the base model.
- **`max_length=512`** — Truncates sequences to 512 tokens. Reward models don't need to see the full response to learn quality signals.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers trl datasets torch accelerate
```

**Steps:**
1. Save to `reward_model.py`
2. Run: `python reward_model.py`

**Expected output:**
Prints two sample responses (chosen and rejected) from the Anthropic dataset. The actual training is commented out. If uncommented, training progress with loss values will appear.

</details>

:::warning[Reward Model Challenges]
- **Label noise:** Humans disagree. Inter-annotator agreement on preference labels is typically 60–75%. The reward model is trained on this noisy signal.
- **Reward hacking:** The policy model may find inputs that score high on the reward model without being genuinely good (e.g., excessively verbose responses that "sound" confident).
- **Distribution shift:** The reward model was trained on SFT model outputs. As PPO modifies the policy, its outputs drift from the training distribution, and the reward model's scores become unreliable.
:::

## Stage 3: PPO Optimization

With a trained reward model, we use **Proximal Policy Optimization (PPO)** — a reinforcement learning algorithm — to fine-tune the SFT model to maximize reward while staying close to the original SFT policy.

:::note[RLHF Objective with KL Penalty]
The optimization objective is:

:::info[Plain English: What Is This Formula Doing?]
Think of training a dog with treats and a leash. The reward model is the treat — it tells the model "good job" for helpful responses. But the KL penalty is the leash — it prevents the model from wandering too far from its original behavior. Without the leash, the dog might learn weird tricks just to get treats (reward hacking). The balance between treat and leash determines whether you get a well-behaved assistant or a sycophantic one.
:::

\[
\max_{\pi_\theta} \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}\left(\pi_\theta(\cdot|x) \;\|\; \pi_{\text{ref}}(\cdot|x)\right) \right]
\]

**Reading the formula:** We want to *maximize* this expression. *π_θ* is the language model (the "policy") we're training. *x* is a prompt drawn from a dataset *D*, and *y* is a response generated by *π_θ*. *r_ϕ(x, y)* is the reward model's score for that response (higher = better). *D_KL(...)* is the KL divergence — a measure of how different our model has become from the reference model *π_ref* (the original SFT model). *β* controls how strongly we penalize drifting. In plain terms: "generate responses that get high rewards, but don't stray too far from your starting behavior."

where:
- \(\pi_\theta\) is the policy being optimized (the language model)
- \(\pi_{\text{ref}}\) is the reference policy (the SFT model, frozen)
- \(r_\phi\) is the reward model
- \(\beta\) controls the KL penalty strength

The **KL divergence penalty** prevents the model from drifting too far from the SFT policy. Without it, the model would exploit reward model weaknesses (reward hacking).
:::

```python title="PPO training with TRL"
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

# Load the SFT model with a value head (for PPO's value function)
model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-output")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-output")
tokenizer = AutoTokenizer.from_pretrained("./sft-output")
tokenizer.pad_token = tokenizer.eos_token

# Load the trained reward model
from transformers import pipeline
reward_pipe = pipeline(
    "text-classification",
    model="./reward-model",
    device_map="auto",
    function_to_apply="none",  # Return raw logits as reward scores
)

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,               # PPO update epochs per batch
    kl_penalty="kl",
    init_kl_coef=0.2,           # β — KL penalty coefficient
    target=6.0,                 # Target KL divergence
    log_with="wandb",
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# PPO training loop (simplified)
prompts = ["Explain photosynthesis.", "What causes rain?", "How do computers work?"]

for prompt_text in prompts:
    # 1. Generate response from current policy
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.pretrained_model.device)
    response_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # 2. Score with reward model
    reward_output = reward_pipe(response_text)
    reward = torch.tensor([reward_output[0]["score"]])

    # 3. PPO update
    query_tensors = inputs["input_ids"][0]
    response_tensors = response_ids[0][len(inputs["input_ids"][0]):]
    stats = ppo_trainer.step([query_tensors], [response_tensors], [reward])
    print(f"Reward: {reward.item():.3f} | KL: {stats['ppo/mean_kl']:.3f}")
```

:::tip[Line-by-Line Walkthrough]
- **`AutoModelForCausalLMWithValueHead.from_pretrained("./sft-output")`** — Loads the SFT model with an extra "value head" — a small network that estimates how good a partial response will be. PPO needs this to compute its advantage estimates.
- **`ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-output")`** — Loads a second copy of the same model that stays frozen. This is the "reference" used to compute KL divergence — how far the policy has drifted.
- **`function_to_apply="none"`** — Tells the pipeline to return raw logit scores instead of applying softmax. We want the raw reward score, not a probability.
- **`init_kl_coef=0.2`** — The initial strength of the KL penalty (β). Higher values keep the model closer to its SFT behavior; lower values allow more aggressive optimization toward the reward.
- **`model.generate(..., do_sample=True, temperature=0.7)`** — Generates a response using sampling (not greedy). The temperature controls randomness — PPO needs stochastic outputs to explore different responses.
- **`ppo_trainer.step([query_tensors], [response_tensors], [reward])`** — The core PPO update: given the prompt, the generated response, and its reward score, update the model's weights to make high-reward responses more likely (while staying close to the reference).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install trl transformers torch accelerate wandb
```
You need a trained SFT model at `./sft-output` and a trained reward model at `./reward-model` from the previous stages.

**Steps:**
1. Save to `ppo_training.py`
2. Run: `python ppo_training.py`

**Expected output:**
```
Reward: 0.432 | KL: 0.012
Reward: 0.567 | KL: 0.018
Reward: 0.612 | KL: 0.025
```
Reward should generally increase while KL divergence stays moderate. If KL spikes, the model is drifting too fast.

</details>

## InstructGPT and ChatGPT's Training Pipeline (Historical Context)

OpenAI's InstructGPT paper (2022) established the canonical three-stage RLHF recipe and remains an important reference for understanding how alignment training works. The original pipeline:

1. **Pretrain** GPT-3 on internet text (175B parameters)
2. **SFT** on ~13K human-written demonstrations
3. **Reward model** trained on ~33K comparisons (labelers ranked 4–9 outputs per prompt)
4. **PPO** optimized on ~31K prompts

The result: InstructGPT (1.3B parameters with RLHF) was preferred by humans over GPT-3 (175B parameters without RLHF). A smaller, aligned model beat a much larger, unaligned one.

ChatGPT (released November 2022) extended this approach with more data, more RLHF iterations, and a conversational format. GPT-4 (2023) used a similar but more sophisticated pipeline.

Since then, the alignment landscape has evolved significantly. **DPO** (covered below) largely replaced PPO for open-source alignment due to its simplicity and stability. **ORPO** merged the SFT and preference stages into one. **Constitutional AI** (Anthropic, 2022–2023) introduced AI-generated feedback to reduce reliance on human labelers. Most recently, **DeepSeek-R1** (2025) demonstrated that reinforcement learning can elicit strong chain-of-thought reasoning capabilities, using rule-based rewards rather than learned reward models. Today's alignment pipelines look quite different from the original InstructGPT recipe, but its three-stage conceptual framework remains foundational.

:::info[The RLHF Tax]
RLHF is expensive and complex:
- **Data cost:** Thousands of human preference labels ($5–15 per comparison)
- **Compute cost:** Three separate training stages, with PPO being the most GPU-intensive
- **Infrastructure:** PPO requires running four models simultaneously (policy, reference, reward, value head)
- **Instability:** PPO is notoriously sensitive to hyperparameters

This cost is what motivated the search for simpler alternatives — enter DPO.
:::

## DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) eliminates the reward model and PPO entirely. Instead of training a reward model and then using RL to optimize against it, DPO directly optimizes the language model on preference data.

:::note[DPO Loss Function]
The DPO loss reparameterizes the RLHF objective. The key insight: the optimal policy under the RLHF objective has a closed-form relationship with the reward function. We can solve for the reward implicitly and optimize preferences directly:

:::info[Plain English: What Is This Formula Doing?]
Imagine you're training a chef by showing them pairs of dishes and telling them which one diners preferred. Instead of first training a separate food critic (reward model) and then having the chef try to please the critic (PPO), DPO skips the middleman. The chef directly learns from the preference feedback: "make your cooking more like the winning dish and less like the losing dish." The reference model acts as a baseline — "don't forget your fundamentals while adapting to preferences."
:::

\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
\]

**Reading the formula:** *L_DPO* is the loss we're minimizing. *π_θ* is our model being trained. *π_ref* is the frozen reference model (the SFT model). *x* is a prompt, *y_w* is the preferred response, *y_l* is the rejected response. The ratio *π_θ(y_w|x) / π_ref(y_w|x)* measures how much more likely our model makes the preferred response compared to the reference. Similarly for the rejected response. *β* controls the strength of the preference signal. *σ* is the sigmoid. In words: "increase the probability of preferred responses and decrease the probability of rejected responses, relative to where the reference model started."

In words: increase the probability of preferred responses and decrease the probability of rejected responses, relative to the reference model.
:::

### DPO vs. RLHF

```
RLHF Pipeline:                         DPO Pipeline:
┌──────────┐                            ┌──────────┐
│ SFT      │                            │ SFT      │
└─────┬────┘                            └─────┬────┘
      │                                       │
┌─────▼────┐                            ┌─────▼─────────┐
│ Train    │                            │ Train directly │
│ Reward   │                            │ on preferences │
│ Model    │                            │ (single step)  │
└─────┬────┘                            └───────────────┘
      │
┌─────▼────┐
│ PPO with │
│ Reward   │
│ Model    │
└──────────┘

3 stages, 4 models                     2 stages, 2 models
```

DPO is simpler, more stable, and cheaper to train. Its quality matches or exceeds RLHF-PPO in most benchmarks. Most open-source models in 2024–2026 use DPO or its variants.

```python title="DPO training with TRL"
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import LoraConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load preference dataset
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train[:5000]")

# DPO datasets need: prompt, chosen, rejected
def format_dpo(example):
    return {
        "prompt": example["instruction"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(format_dpo)

# Optional: use LoRA for memory efficiency
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6,          # DPO uses very low learning rates
    beta=0.1,                    # KL penalty strength
    bf16=True,
    logging_steps=25,
    max_length=1024,
    max_prompt_length=512,
    gradient_checkpointing=True,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
# trainer.train()
```

:::tip[Line-by-Line Walkthrough]
- **`load_dataset("argilla/ultrafeedback-binarized-preferences", ...)`** — Loads a preference dataset where each example has a prompt, a "chosen" (preferred) response, and a "rejected" response.
- **`format_dpo(example)`** — Reshapes the dataset into the three fields DPO expects: `prompt`, `chosen`, and `rejected`.
- **`learning_rate=5e-6`** — DPO uses a very low learning rate (lower than SFT). The model only needs subtle adjustments to prefer better responses.
- **`beta=0.1`** — The KL penalty strength. Higher beta means the model stays closer to the reference and changes less; lower beta allows more aggressive preference optimization.
- **`max_prompt_length=512`** — Limits how many tokens of the prompt are kept. This saves memory — the model doesn't need the full prompt to learn preferences.
- **`peft_config=peft_config`** — Passes a LoRA configuration so DPO trains only the adapter, not the full model. This makes DPO feasible on consumer GPUs.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers trl datasets peft torch accelerate
```

**Steps:**
1. Save to `dpo_training.py`
2. Run: `python dpo_training.py`

**Expected output:**
Dataset loading and preparation output. The actual training is commented out. If uncommented, you'll see training progress with DPO loss decreasing. A full run on 5,000 examples takes roughly 30–90 minutes on a single GPU.

</details>

### Beyond DPO: Recent Variants

The alignment landscape evolves rapidly. Notable DPO variants:

- **IPO (Identity Preference Optimization):** Adds regularization to prevent overfitting to the preference data
- **KTO (Kahneman-Tversky Optimization):** Doesn't require paired preferences — just labels of "good" or "bad" on individual responses
- **ORPO (Odds Ratio Preference Optimization):** Combines SFT and preference optimization into a single stage
- **SimPO:** Simplifies DPO by using sequence-level log probabilities without a reference model

:::tip[Practical Recommendation]
For most teams building aligned models in 2025–2026:
1. **SFT** on high-quality instruction data (start with a strong instruct model)
2. **DPO** on preference data (much simpler than PPO)
3. Use **LoRA/QLoRA** for both stages to keep costs manageable

This two-stage pipeline achieves 90%+ of the quality of a full RLHF pipeline at a fraction of the cost.
:::

---

## Exercises

:::tip[Exercise 1: Preference Data Analysis — beginner]

Load a preference dataset and analyze it quantitatively. Compare the chosen vs. rejected responses on: average length, vocabulary richness, reading level, and sentiment. Do you find systematic patterns? Are longer responses always preferred? What biases might the reward model learn from this data?

<details>
<summary>Hints</summary>

1. Load Anthropic/hh-rlhf or argilla/ultrafeedback-binarized-preferences
2. Compare average response lengths between chosen and rejected
3. Look at the vocabulary and sentiment differences
4. Check if longer responses are always preferred

</details>

:::

:::tip[Exercise 2: Build a Simple Reward Model — intermediate]

Train a simple reward model on the Anthropic HH-RLHF dataset. Use a pretrained DeBERTa-v3-base model with a classification head. Evaluate it on held-out data and compute its agreement with human preferences. What accuracy can you achieve? How does it compare to random (50%)?

<details>
<summary>Hints</summary>

1. Use a pretrained encoder model (e.g. DeBERTa) with a classification head
2. The input is the concatenation of prompt + response
3. Train as binary classification: chosen=1, rejected=0
4. Evaluate with accuracy on held-out preference pairs

</details>

:::

:::tip[Exercise 3: DPO Fine-Tuning — advanced]

Fine-tune a small language model using DPO. Use QLoRA to keep it feasible on consumer hardware. Experiment with different β values and evaluate the results by generating responses to a fixed set of prompts. Can you observe the model becoming more helpful and less verbose? What happens with very high or very low β?

<details>
<summary>Hints</summary>

1. Start from a small instruct model (1B–3B parameters)
2. Use QLoRA to keep memory manageable
3. Try different β values: 0.05, 0.1, 0.2, 0.5
4. Evaluate by generating responses and comparing quality before/after DPO

</details>

:::

---

## Resources

- **[Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)** _(paper)_ by Ouyang et al., 2022 — The InstructGPT paper — the foundational RLHF recipe used by ChatGPT.

- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)** _(paper)_ by Rafailov et al., 2023 — DPO — the simpler alternative to PPO that became the standard for open-source alignment.

- **[TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl)** _(tool)_ — HuggingFace library for SFT, reward modeling, PPO, and DPO — the practical toolkit for RLHF.

- **[Illustrating RLHF](https://huggingface.co/blog/rlhf)** _(tutorial)_ by HuggingFace — Clear visual explanation of the RLHF pipeline with code examples.

- **[RLHF: Reinforcement Learning from Human Feedback (Chip Huyen)](https://huyenchip.com/2023/05/02/rlhf.html)** _(tutorial)_ by Chip Huyen — In-depth blog post covering RLHF's history, mechanics, and practical challenges.

- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)** _(paper)_ by Schulman et al., 2017 — The PPO paper — the RL algorithm used in RLHF's third stage.
