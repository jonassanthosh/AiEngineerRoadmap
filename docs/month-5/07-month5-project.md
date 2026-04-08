---
sidebar_position: 7
slug: month5-project
title: "Project: Fine-Tune and Deploy an LLM"
---


# Project: Fine-Tune and Deploy an LLM

:::info[What You'll Learn]
- Fine-tuning a base model with QLoRA on custom data
- Merging and quantizing the fine-tuned model
- Deploying as a REST API with vLLM or Ollama
- Building an evaluation pipeline to measure model quality
:::

:::note[Prerequisites]
All of Month 5 lessons 1–6.
:::

**Estimated time:** Reading: ~25 min | Project work: ~10 hours

This capstone project integrates everything from Month 5. You'll take a base model, fine-tune it with LoRA on a custom instruction dataset, quantize the result, deploy it as an API, and evaluate the final model. This is the end-to-end workflow used by teams shipping LLM-powered products.

## Project Overview

**Goal:** Build and deploy a domain-specific instruction-following LLM.

**Pipeline:**
```
Base Model (Llama/Mistral)
    │
    ▼
1. Prepare instruction dataset
    │
    ▼
2. Fine-tune with QLoRA
    │
    ▼
3. Merge LoRA adapter
    │
    ▼
4. Quantize merged model (INT4)
    │
    ▼
5. Deploy with vLLM or Ollama
    │
    ▼
6. Build REST API
    │
    ▼
7. Evaluate model quality
```

:::info[Choose Your Domain]
Pick a domain you're interested in. Good options:
- **Code assistant** — fine-tune on code instruction data (e.g., Python coding exercises)
- **Medical Q&A** — fine-tune on medical question-answer pairs
- **Legal assistant** — fine-tune on legal document analysis
- **Customer support** — fine-tune on support ticket conversations
- **Creative writing** — fine-tune on story writing prompts and completions

The steps are the same regardless of domain. We'll use a general instruction dataset in the examples, but you should substitute your domain-specific data.
:::

## Step 1: Prepare the Instruction Dataset

The quality of your fine-tuning dataset matters more than its size. A few thousand high-quality examples beat tens of thousands of noisy ones.

```python title="Dataset preparation"
from datasets import load_dataset, Dataset
import json

# Option 1: Use an existing instruction dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"\\nSample:\\n{json.dumps(dataset[0], indent=2)}")

# Option 2: Create your own dataset from a JSONL file
# Each line should have instruction, input (optional), and output fields
sample_data = [
    {
        "instruction": "Explain the concept of dependency injection.",
        "input": "",
        "output": "Dependency injection is a design pattern where an object receives its dependencies from external sources rather than creating them internally. Instead of a class instantiating its own dependencies, they are 'injected' through constructor parameters, setter methods, or interfaces. This makes code more modular, testable, and loosely coupled."
    },
    {
        "instruction": "Convert this Python function to use a list comprehension.",
        "input": "def get_evens(numbers):\\n    result = []\\n    for n in numbers:\\n        if n % 2 == 0:\\n            result.append(n)\\n    return result",
        "output": "def get_evens(numbers):\\n    return [n for n in numbers if n % 2 == 0]"
    },
]

# Format into chat template
def format_instruction(example):
    """Convert instruction/input/output to chat format."""
    if example.get("input", "").strip():
        user_message = f"{example['instruction']}\\n\\n{example['input']}"
    else:
        user_message = example["instruction"]

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"messages": messages}

formatted = dataset.map(format_instruction)

# Split into train/eval
split = formatted.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train: {len(train_dataset)} examples")
print(f"Eval:  {len(eval_dataset)} examples")
print(f"\\nFormatted sample messages:")
for msg in train_dataset[0]["messages"]:
    print(f"  [{msg['role']}]: {msg['content'][:100]}...")
```

:::tip[Line-by-Line Walkthrough]
- **`load_dataset("yahma/alpaca-cleaned", split="train")`** — Loads the Alpaca Cleaned dataset, a popular instruction-following dataset with ~52K examples that have been deduplicated and cleaned.
- **`format_instruction(example)`** — Converts raw instruction/input/output format into a chat-style message list that matches what modern chat models expect.
- **`if example.get("input", "").strip():`** — Some instructions have additional context/input (e.g., code to refactor). This concatenates it with the instruction when present.
- **`formatted.train_test_split(test_size=0.05, seed=42)`** — Reserves 5% of the data for evaluation. The seed ensures reproducibility so you get the same split every time.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install datasets
```

**Steps:**
1. Save to `prepare_data.py`
2. Run: `python prepare_data.py`

**Expected output:**
```
Dataset size: 51760
Columns: ['instruction', 'input', 'output']

Sample:
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet..."
}
Train: 49172 examples
Eval:  2588 examples
```

</details>

:::warning[Dataset Quality Checklist]
Before fine-tuning, verify your dataset:
- **No duplicates.** Deduplicate by instruction text.
- **Consistent format.** Every example follows the same structure.
- **Correct outputs.** Spot-check at least 50 random examples for accuracy.
- **Appropriate length.** Filter out very short (<10 token) or very long (>2048 token) responses.
- **No data leakage.** Your eval set shouldn't overlap with training data.
:::

## Step 2: Fine-Tune with QLoRA

```python title="QLoRA fine-tuning"
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Model & Tokenizer ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# Alternative: "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4-bit Quantization ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./ft-output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",  # Set to "wandb" for experiment tracking
)

# --- Format function ---
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )

# --- Train ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,
    packing=True,
)

print("Starting training...")
trainer.train()
print("Training complete!")

# Save the LoRA adapter
trainer.save_model("./ft-adapter")
tokenizer.save_pretrained("./ft-adapter")
```

:::tip[Line-by-Line Walkthrough]
- **`tokenizer.padding_side = "right"`** — Ensures padding tokens are added to the right of sequences. Left-padding can cause issues with some models during training.
- **`bnb_4bit_use_double_quant=True`** — Quantizes the quantization constants themselves, squeezing out a bit more memory savings at negligible quality cost.
- **`target_modules="all-linear"`** — Applies LoRA adapters to every linear layer in the model. This gives better quality than targeting only attention layers.
- **`eval_strategy="steps"` / `eval_steps=200`** — Evaluates on the held-out set every 200 training steps. This lets you monitor for overfitting.
- **`load_best_model_at_end=True`** — Automatically loads the checkpoint with the lowest eval loss at the end of training. Prevents using an overfitted final checkpoint.
- **`save_total_limit=3`** — Keeps only the 3 most recent checkpoints to avoid filling up disk space.
- **`packing=True`** — Concatenates short examples into longer sequences to maximize GPU utilization and reduce wasted padding compute.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers peft bitsandbytes trl datasets torch accelerate flash-attn
```
You need a CUDA GPU with at least 8 GB VRAM.

**Steps:**
1. First run the dataset preparation code from Step 1
2. Save this code to `qlora_finetune.py`
3. Run: `python qlora_finetune.py`

**Expected output:**
Training progress bars showing train loss and eval loss decreasing. Final output:
```
trainable params: 13,631,488 || all params: 1,249,445,632 || trainable%: 1.09
Starting training...
{'loss': 2.1, 'learning_rate': 0.0002, 'epoch': 0.1}
...
Training complete!
```
Training on the full Alpaca dataset takes 1–3 hours on a single A100 GPU.

</details>

### Monitoring Training

```python title="Training monitoring and early stopping"
# If using Weights & Biases for experiment tracking:
# pip install wandb
# wandb login
# Then set report_to="wandb" in TrainingArguments

# Manual training curve visualization
import matplotlib.pyplot as plt

def plot_training_logs(trainer):
    """Plot training and eval loss from trainer logs."""
    logs = trainer.state.log_history

    train_steps = [l["step"] for l in logs if "loss" in l]
    train_loss = [l["loss"] for l in logs if "loss" in l]
    eval_steps = [l["step"] for l in logs if "eval_loss" in l]
    eval_loss = [l["eval_loss"] for l in logs if "eval_loss" in l]

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, label="Train Loss", alpha=0.7)
    plt.plot(eval_steps, eval_loss, label="Eval Loss", marker="o", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Fine-Tuning Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()

    # Check for overfitting
    if len(eval_loss) >= 2:
        if eval_loss[-1] > eval_loss[-2]:
            print("⚠️  Eval loss is increasing — potential overfitting!")
        else:
            print("✓ Eval loss is still decreasing")

plot_training_logs(trainer)
```

:::tip[Line-by-Line Walkthrough]
- **`trainer.state.log_history`** — Accesses the trainer's internal log of all metrics recorded during training. Each entry is a dict with keys like `"loss"`, `"eval_loss"`, `"step"`, etc.
- **`[l["step"] for l in logs if "loss" in l]`** — Filters for entries that contain training loss (not eval loss) and extracts the step numbers.
- **`plt.plot(eval_steps, eval_loss, marker="o")`** — Plots eval loss with circle markers. If this line starts going up while train loss continues down, you're overfitting.
- **`if eval_loss[-1] > eval_loss[-2]:`** — A simple overfitting check: if the most recent eval loss is higher than the previous one, the model may be starting to memorize the training data.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install matplotlib
```

**Steps:**
1. Run this code after the training from Step 2 completes (the `trainer` object must be available)
2. It will generate and save a plot as `training_curve.png`

**Expected output:**
A line plot showing training loss decreasing smoothly and eval loss decreasing then potentially flattening. A printed message indicating whether overfitting is detected.

</details>

## Step 3: Merge the LoRA Adapter

After training, merge the LoRA weights into the base model to create a single, standalone model.

```python title="Merging LoRA adapter into base model"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model in FP16 (not quantized — we need full precision for merging)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, "./ft-adapter")

# Merge adapter into base model
print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

# Save the merged model
output_dir = "./ft-merged"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Merged model saved to {output_dir}")

# Quick test
inputs = tokenizer("Explain what a REST API is:", return_tensors="pt").to(merged_model.device)
outputs = merged_model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print("\\nTest output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

:::tip[Line-by-Line Walkthrough]
- **`torch_dtype=torch.float16`** — Loads the base model in FP16, not quantized. Merging requires full-precision weights because you're adding the LoRA matrices into the base weights mathematically.
- **`PeftModel.from_pretrained(base_model, "./ft-adapter")`** — Loads the trained LoRA adapter (~50 MB) and attaches it to the base model.
- **`model.merge_and_unload()`** — Performs the math: W_merged = W_base + B × A × (alpha/rank) for every adapted layer. After merging, the model has no LoRA overhead — it's a standard model.
- **`merged_model.save_pretrained(output_dir)`** — Saves the merged model. This is now a standalone model that doesn't need the PEFT library to run.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install transformers peft torch accelerate
```
You need the trained LoRA adapter from Step 2 at `./ft-adapter`.

**Steps:**
1. Save to `merge.py`
2. Run: `python merge.py`

**Expected output:**
```
Merging LoRA weights...
Merged model saved to ./ft-merged

Test output:
Explain what a REST API is: A REST API (Representational State Transfer Application Programming Interface) is...
```

</details>

## Step 4: Quantize the Fine-Tuned Model

Quantize the merged model to INT4 for efficient deployment.

```python title="Quantizing the merged model"
# Option A: Quantize with AutoGPTQ (best quality)
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

model_dir = "./ft-merged"
quant_dir = "./ft-quantized-gptq"

tokenizer = AutoTokenizer.from_pretrained(model_dir)

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
)

model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config=quantize_config)

# Calibration data
calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:128]")
calib_examples = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calib_data["text"] if len(text) > 100
][:128]

model.quantize(calib_examples)
model.save_quantized(quant_dir)
tokenizer.save_pretrained(quant_dir)
print(f"GPTQ-quantized model saved to {quant_dir}")

# Option B: Convert to GGUF for Ollama (using llama.cpp)
# python convert_hf_to_gguf.py ./ft-merged --outfile ./ft-merged.gguf --outtype q4_k_m
# This creates a GGUF file compatible with Ollama and llama.cpp
```

:::tip[Line-by-Line Walkthrough]
- **`BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)`** — Configures 4-bit GPTQ quantization with 128-element groups and descending activation order for best quality.
- **`AutoGPTQForCausalLM.from_pretrained(model_dir, ...)`** — Loads the merged FP16 model for quantization. This is the model you trained and merged in previous steps.
- **`calib_data = load_dataset("wikitext", ...)`** — Loads a small calibration dataset. GPTQ needs real input data to measure how quantization affects outputs and to make compensatory adjustments.
- **`model.quantize(calib_examples)`** — Runs GPTQ quantization: processes each layer, quantizes weights to 4-bit, and adjusts remaining weights to minimize output error.
- **`convert_hf_to_gguf.py`** — The GGUF conversion (Option B) converts the model to a format that Ollama and llama.cpp can run, including on CPUs and Apple Silicon.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install auto-gptq transformers datasets torch
```

**Steps:**
1. Ensure the merged model exists at `./ft-merged` (from Step 3)
2. Save to `quantize.py`
3. Run: `python quantize.py`

**Expected output:**
```
GPTQ-quantized model saved to ./ft-quantized-gptq
```
The quantized model directory will be ~3-4× smaller than the FP16 merged model.

</details>

## Step 5: Deploy with vLLM or Ollama

```python title="Deployment options"
# ========================================
# Option A: Deploy with vLLM
# ========================================
# For GPTQ-quantized models:
# python -m vllm.entrypoints.openai.api_server \\
#     --model ./ft-quantized-gptq \\
#     --quantization gptq \\
#     --dtype float16 \\
#     --max-model-len 4096 \\
#     --port 8000

# Programmatic vLLM usage:
from vllm import LLM, SamplingParams

llm = LLM(
    model="./ft-quantized-gptq",
    quantization="gptq",
    dtype="float16",
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

test_prompts = [
    "Explain the SOLID principles in software engineering.",
    "What is the difference between a stack and a queue?",
    "Write a Python function to find the longest palindromic substring.",
]

outputs = llm.generate(test_prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
    print("-" * 60)

# ========================================
# Option B: Deploy with Ollama
# ========================================
# 1. Convert to GGUF (if not already done)
# 2. Create a Modelfile:
#
#    FROM ./ft-merged.gguf
#    PARAMETER temperature 0.7
#    PARAMETER top_p 0.9
#    SYSTEM "You are a helpful AI assistant specialized in software engineering."
#
# 3. Create the Ollama model:
#    ollama create my-ft-model -f Modelfile
#
# 4. Run it:
#    ollama run my-ft-model "Explain dependency injection"
#
# 5. Use via API:
import requests

def query_ollama(prompt: str, model: str = "my-ft-model"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9},
    })
    return response.json()["response"]

# result = query_ollama("Explain the observer pattern in software engineering.")
```

:::tip[Line-by-Line Walkthrough]
- **`LLM(model="./ft-quantized-gptq", quantization="gptq", ...)`** — Loads your fine-tuned, quantized model into vLLM's engine. The `quantization="gptq"` flag tells vLLM to use GPTQ-aware kernels for fast inference.
- **`llm.generate(test_prompts, sampling_params)`** — Generates responses for all test prompts using vLLM's continuous batching. All three prompts are processed efficiently in parallel.
- **`FROM ./ft-merged.gguf`** — In the Ollama Modelfile, this specifies the GGUF model file to use. The SYSTEM line sets a default system prompt.
- **`query_ollama(prompt, ...)`** — Sends a generation request to Ollama's REST API. Setting `"stream": False` waits for the complete response.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites (vLLM):**
```bash
pip install vllm
```

**Prerequisites (Ollama):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
# Plus convert your model to GGUF format
```

**Steps (vLLM):**
1. Save to `deploy_vllm.py`
2. Run: `python deploy_vllm.py`

**Steps (Ollama):**
1. Create the Modelfile as shown
2. Run: `ollama create my-ft-model -f Modelfile`
3. Test: `ollama run my-ft-model "Your question"`

**Expected output:**
Generated responses for each test prompt, demonstrating that your fine-tuned model works end-to-end.

</details>

## Step 6: Build a Simple API

```python title="Production-ready API for your fine-tuned model"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import json
import time
import uuid

app = FastAPI(title="Fine-Tuned LLM API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(default=512, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class CompletionResponse(BaseModel):
    id: str
    choices: list[dict]
    usage: dict
    model: str

# --- Initialize Model ---
# Using vLLM's async engine for production
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine_args = AsyncEngineArgs(
    model="./ft-quantized-gptq",
    quantization="gptq",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/v1/chat/completions")
async def chat_completions(request: CompletionRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"

    # Build prompt from messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./ft-quantized-gptq")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=0.9,
    )

    if request.stream:
        async def stream_response():
            async for output in engine.generate(prompt, sampling_params, request_id):
                token = output.outputs[0].text
                chunk = {
                    "id": request_id,
                    "choices": [{"delta": {"content": token}, "index": 0}],
                }
                yield f"data: {json.dumps(chunk)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    # Non-streaming
    final_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        final_output = output

    response_text = final_output.outputs[0].text
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = len(final_output.outputs[0].token_ids)

    return CompletionResponse(
        id=request_id,
        model="ft-model",
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": final_output.outputs[0].finish_reason,
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "ft-model"}

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

:::tip[Line-by-Line Walkthrough]
- **`CORSMiddleware`** — Allows cross-origin requests so web frontends on different domains can call your API.
- **`AsyncLLMEngine.from_engine_args(engine_args)`** — Creates an asynchronous vLLM engine for production serving. The async engine handles concurrent requests efficiently without blocking.
- **`tokenizer.apply_chat_template(messages, ..., add_generation_prompt=True)`** — Converts chat messages into the model's expected format and adds the generation prompt marker so the model knows to start responding.
- **`async for output in engine.generate(...):`** — vLLM's async generator yields outputs as they're produced. For streaming, each yield contains new tokens. For non-streaming, we just take the final output.
- **`StreamingResponse(stream_response(), media_type="text/event-stream")`** — Returns a Server-Sent Events (SSE) stream. The client receives tokens as they're generated, enabling real-time typewriter-style display.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install fastapi uvicorn vllm transformers
```

**Steps:**
1. Save to `api.py`
2. Run: `uvicorn api:app --host 0.0.0.0 --port 8000`
3. Test with curl:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain dependency injection"}]}'
```

**Expected output:**
A JSON response containing the model's answer, token counts, and a unique request ID.

</details>

## Step 7: Evaluate the Model

Evaluation is critical. You need to know whether your fine-tuned model actually improves over the base model.

```python title="Model evaluation"
import json
from openai import OpenAI

# Connect to your deployed model (or the vLLM server)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# --- 1. Automated Evaluation ---
eval_prompts = [
    {
        "prompt": "Explain the difference between a process and a thread.",
        "criteria": ["mentions concurrency", "mentions shared memory", "mentions isolation"],
    },
    {
        "prompt": "Write a Python function to reverse a linked list.",
        "criteria": ["includes function definition", "handles edge cases", "correct algorithm"],
    },
    {
        "prompt": "What is the CAP theorem?",
        "criteria": ["mentions consistency", "mentions availability", "mentions partition tolerance"],
    },
]

def evaluate_response(response: str, criteria: list[str]) -> dict:
    """Simple keyword-based evaluation."""
    scores = {}
    for criterion in criteria:
        keywords = criterion.lower().split()
        scores[criterion] = any(kw in response.lower() for kw in keywords)
    return scores

results = []
for item in eval_prompts:
    response = client.chat.completions.create(
        model="ft-model",
        messages=[{"role": "user", "content": item["prompt"]}],
        max_tokens=512,
        temperature=0.3,
    )
    text = response.choices[0].message.content
    scores = evaluate_response(text, item["criteria"])
    results.append({
        "prompt": item["prompt"],
        "response": text[:200],
        "scores": scores,
        "pass_rate": sum(scores.values()) / len(scores),
    })

# Summary
print("Evaluation Results:")
print("-" * 60)
for r in results:
    print(f"Prompt: {r['prompt'][:60]}...")
    print(f"Pass rate: {r['pass_rate']:.0%}")
    for criterion, passed in r['scores'].items():
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
    print()

avg_pass = sum(r["pass_rate"] for r in results) / len(results)
print(f"Overall pass rate: {avg_pass:.0%}")

# --- 2. Perplexity Evaluation ---
def evaluate_perplexity_on_dataset(model_path, eval_texts, max_length=512):
    """Evaluate model perplexity on held-out text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    total_loss = 0
    total_tokens = 0

    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                          truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

# Compare base vs fine-tuned
# base_ppl = evaluate_perplexity_on_dataset("meta-llama/Llama-3.2-1B-Instruct", eval_texts)
# ft_ppl = evaluate_perplexity_on_dataset("./ft-merged", eval_texts)
# print(f"Base model perplexity: {base_ppl:.2f}")
# print(f"Fine-tuned perplexity: {ft_ppl:.2f}")

# --- 3. Side-by-Side Comparison ---
def compare_models(prompts, base_url, ft_url):
    """Generate responses from both models for comparison."""
    base_client = OpenAI(base_url=base_url, api_key="dummy")
    ft_client = OpenAI(base_url=ft_url, api_key="dummy")

    for prompt in prompts:
        base_resp = base_client.chat.completions.create(
            model="base", messages=[{"role": "user", "content": prompt}],
            max_tokens=256, temperature=0.3,
        )
        ft_resp = ft_client.chat.completions.create(
            model="ft", messages=[{"role": "user", "content": prompt}],
            max_tokens=256, temperature=0.3,
        )
        print(f"Prompt: {prompt}")
        print(f"\\nBase model: {base_resp.choices[0].message.content[:300]}")
        print(f"\\nFine-tuned: {ft_resp.choices[0].message.content[:300]}")
        print("=" * 60)
```

:::tip[Line-by-Line Walkthrough]
- **`OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")`** — Uses the OpenAI Python client to connect to your local vLLM server. Since vLLM exposes an OpenAI-compatible API, any OpenAI client library works.
- **`evaluate_response(response, criteria)`** — A simple keyword-based evaluator that checks if the response mentions expected concepts. In production, you'd use an LLM-as-judge for more nuanced evaluation.
- **`temperature=0.3`** — Uses low temperature for evaluation so responses are consistent and reproducible across runs.
- **`sum(scores.values()) / len(scores)`** — Computes the fraction of criteria the response satisfies. A pass rate of 1.0 means all criteria were met.
- **`evaluate_perplexity_on_dataset(...)`** — Measures perplexity on held-out text. Lower perplexity means the model predicts the text better. Comparing base vs. fine-tuned perplexity shows whether fine-tuning improved the model on your domain.
- **`compare_models(prompts, base_url, ft_url)`** — Generates side-by-side responses from both models for human comparison. This is the most informative evaluation method.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install openai transformers torch
```
You need your fine-tuned model deployed and running (from Step 5 or 6).

**Steps:**
1. Ensure your API is running at `http://localhost:8000`
2. Save to `evaluate.py`
3. Run: `python evaluate.py`

**Expected output:**
```
Evaluation Results:
------------------------------------------------------------
Prompt: Explain the difference between a process and a thre...
Pass rate: 100%
  ✓ mentions concurrency
  ✓ mentions shared memory
  ✓ mentions isolation

Overall pass rate: 89%
```

</details>

:::tip[LLM-as-Judge Evaluation]
For more nuanced evaluation, use a strong LLM (e.g., GPT-4o, Claude 4, or Llama 4 Maverick) as a judge. Provide the prompt, the model's response, and evaluation criteria, and ask the judge model to score on a 1–5 scale. This correlates well with human evaluation and scales much better.
:::

---

## Bonus Challenges

:::tip[Bonus 1: DPO Alignment — advanced]

After SFT, apply DPO to further align your model. Generate response pairs from the SFT model, create preference labels (manually or with an LLM judge), and run DPO training. Compare the DPO-aligned model against the SFT-only version on helpfulness, accuracy, and verbosity.

<details>
<summary>Hints</summary>

1. Generate paired responses from your fine-tuned model
2. Have the model generate at temperature 1.0 to get diverse outputs
3. Manually label 200-500 pairs as preferred/rejected, or use an LLM judge
4. Apply DPO with a low learning rate (5e-7 to 5e-6)

</details>

:::

:::tip[Bonus 2: Multi-Model Routing — advanced]

Deploy two versions of your model (e.g., 1B quantized and 7B quantized). Build a router that classifies incoming queries as simple or complex and routes them to the appropriate model. Measure the cost savings and quality impact of routing compared to always using the larger model.

<details>
<summary>Hints</summary>

1. Deploy two models: a small fast one and a larger capable one
2. Build a classifier that predicts query complexity (0-1)
3. Route simple queries to the small model, complex ones to the large model
4. Measure: cost savings, latency improvement, quality impact

</details>

:::

:::tip[Bonus 3: Continuous Improvement Pipeline — advanced]

Build a feedback loop for your deployed model:
1. Log all requests and responses
2. Implement a thumbs-up/thumbs-down feedback mechanism
3. Periodically sample low-rated responses and create improved versions
4. Retrain the model on the expanded dataset
5. A/B test the new model against the current one before rolling it out

<details>
<summary>Hints</summary>

1. Log all production requests and responses to a database
2. Periodically sample responses and collect human feedback
3. Use the feedback to create new training data
4. Retrain the model on the combined original + feedback data

</details>

:::

---

## Submission Checklist

Before considering this project complete, verify:

- [ ] Instruction dataset is prepared and validated (format, quality, no duplicates)
- [ ] QLoRA fine-tuning runs successfully with decreasing training loss
- [ ] LoRA adapter is merged into the base model
- [ ] Merged model is quantized to INT4 (GPTQ, AWQ, or GGUF)
- [ ] Model is deployed and serving via vLLM, TGI, or Ollama
- [ ] REST API wraps the deployed model with proper request/response handling
- [ ] Evaluation shows improvement over the base model on your target task
- [ ] At least one bonus challenge attempted

:::info[Wrapping Up Month 5]
You've gone from understanding fine-tuning theory to deploying a production-ready fine-tuned LLM. You can now:
- Choose the right fine-tuning strategy (full FT, LoRA, QLoRA) for any situation
- Apply RLHF/DPO for alignment
- Quantize models for efficient deployment
- Optimize inference with KV-caching, Flash Attention, and continuous batching
- Deploy and serve models with modern frameworks
- Evaluate model quality systematically

In Month 6, you'll build on these skills to create **complete AI-powered applications** — integrating LLMs with retrieval (RAG), agents, tool use, and multi-modal systems.
:::

---

## Resources

- **[HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)** _(tool)_ — Complete documentation for parameter-efficient fine-tuning with LoRA, QLoRA, and adapters.

- **[TRL Documentation](https://huggingface.co/docs/trl)** _(tool)_ — HuggingFace's library for SFT, DPO, and RLHF — the toolkit used throughout this project.

- **[vLLM Documentation](https://docs.vllm.ai)** _(tool)_ — Production LLM serving with PagedAttention and continuous batching.

- **[Ollama](https://ollama.ai)** _(tool)_ — The easiest way to run and deploy LLMs locally.

- **[Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)** _(tutorial)_ by Stanford CRFM — Historical reference (2023): How Stanford fine-tuned the original LLaMA on 52K instruction-following demonstrations for under $600. By 2026, fine-tuning newer open models (e.g., Llama 4, Qwen 3) achieves far better results at even lower cost thanks to improved tooling, cheaper compute, and more efficient training methods like QLoRA.

- **[LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)** _(tool)_ by EleutherAI — Standardized evaluation framework for language models — supports 200+ benchmarks.
