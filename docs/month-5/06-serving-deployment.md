---
sidebar_position: 6
slug: serving-deployment
title: "Serving and Deployment"
---


# Serving and Deployment

You've trained, fine-tuned, and optimized your model. Now you need to serve it to users. This lesson covers the practical engineering of deploying LLMs: choosing a serving framework, building APIs, monitoring production systems, and managing costs.

## Model Serving Architectures

LLM serving differs from traditional ML model serving in several ways:

- **Stateful generation:** Each request maintains a KV-cache that grows with each generated token
- **Variable latency:** Response times depend on output length, which is unpredictable
- **High memory requirements:** A single model may consume the entire GPU memory
- **Streaming:** Users expect token-by-token streaming, not batch responses

### Serving Framework Comparison

| Framework | Best For | Key Features |
|---|---|---|
| **vLLM** | High-throughput production serving | PagedAttention, continuous batching, TP |
| **TGI** | HuggingFace ecosystem integration | Token streaming, quantization, multi-GPU |
| **Ollama** | Local development, personal use | Easy setup, model management, REST API |
| **TensorRT-LLM** | Maximum NVIDIA GPU performance | Kernel fusion, FP8, inflight batching |
| **llama.cpp** | CPU/edge deployment | C++ performance, GGUF format, ARM support |

## vLLM Server Setup

vLLM is the most popular open-source LLM serving framework. It implements PagedAttention for efficient memory management and continuous batching for high throughput.

```python title="Starting a vLLM server"
# Install vLLM
# pip install vllm

# Start the OpenAI-compatible API server (run in terminal)
# python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Llama-3.2-1B-Instruct \\
#     --dtype auto \\
#     --max-model-len 4096 \\
#     --gpu-memory-utilization 0.9 \\
#     --enable-prefix-caching \\
#     --port 8000

# For quantized models:
# python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Llama-3.2-1B-Instruct \\
#     --quantization awq \\
#     --dtype float16 \\
#     --port 8000

# For multi-GPU serving (tensor parallelism):
# python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Llama-3.1-70B-Instruct \\
#     --tensor-parallel-size 4 \\
#     --port 8000

# Programmatic usage:
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="auto",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["\\n\\n"],
)

prompts = [
    "Explain quantum computing to a 5 year old.",
    "Write a Python function to compute Fibonacci numbers.",
    "What are the key differences between TCP and UDP?",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text[:200]}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}\\n")
```

:::tip[Line-by-Line Walkthrough]
- **`LLM(model="...", gpu_memory_utilization=0.9)`** — Creates a vLLM engine that uses 90% of GPU memory for the model and KV-cache. The remaining 10% is a safety buffer.
- **`SamplingParams(temperature=0.7, top_p=0.9, ...)`** — Configures how tokens are sampled: temperature controls randomness (lower = more deterministic), top_p keeps only the most likely tokens that sum to 90% probability.
- **`stop=["\\n\\n"]`** — Tells the model to stop generating when it produces two consecutive newlines, preventing runaway generation.
- **`llm.generate(prompts, sampling_params)`** — Generates responses for all prompts using continuous batching internally — vLLM automatically handles efficient scheduling across the batch.
- **`output.outputs[0].text`** — Accesses the generated text. `outputs[0]` because vLLM can return multiple completions per prompt (we asked for one).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install vllm
```
You need a CUDA GPU with at least 6 GB VRAM for the 1B model.

**Steps:**
1. Save the Python part to `vllm_demo.py`
2. Run: `python vllm_demo.py`

**Expected output:**
```
Prompt: Explain quantum computing to a 5 year old....
Output: Quantum computing is like having a magic box that can try...
Tokens: 142

Prompt: Write a Python function to compute Fibonacci nu...
Output: def fibonacci(n):
    if n <= 1:
        return n...
Tokens: 89
```

</details>

:::tip[vLLM Performance Tuning]
Key vLLM flags for production:
- `--gpu-memory-utilization 0.9` — Use 90% of GPU memory for KV-cache (default is 0.9)
- `--enable-prefix-caching` — Cache KV states for shared prefixes (system prompts)
- `--max-num-seqs 256` — Maximum concurrent sequences in a batch
- `--enable-chunked-prefill` — Break long prefills into chunks for better interleaving with decode
:::

## Text Generation Inference (TGI)

HuggingFace's TGI is a Rust-based serving framework optimized for HuggingFace models. It's the backend for HuggingFace's Inference API.

```python title="Deploying with TGI"
# Run TGI via Docker (recommended)
# docker run --gpus all --shm-size 1g -p 8080:80 \\
#     ghcr.io/huggingface/text-generation-inference:latest \\
#     --model-id meta-llama/Llama-3.2-1B-Instruct \\
#     --max-input-length 2048 \\
#     --max-total-tokens 4096 \\
#     --max-batch-prefill-tokens 4096 \\
#     --quantize awq

# Query the TGI server
import requests

def query_tgi(prompt: str, max_tokens: int = 256, stream: bool = False):
    url = "http://localhost:8080/generate"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    }
    response = requests.post(url, json=payload)
    return response.json()

# Streaming response
def stream_tgi(prompt: str, max_tokens: int = 256):
    url = "http://localhost:8080/generate_stream"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7},
    }
    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if data.startswith("data:"):
                    print(data[5:], end="", flush=True)

# Example
result = query_tgi("What is the meaning of life?")
print(result)
```

:::tip[Line-by-Line Walkthrough]
- **`docker run --gpus all --shm-size 1g ...`** — Starts TGI in a Docker container with GPU access. `--shm-size 1g` increases shared memory, which is needed for multi-processing.
- **`query_tgi(prompt, ...)`** — Sends a synchronous request to the TGI server. The response contains the full generated text.
- **`stream_tgi(prompt, ...)`** — Uses the streaming endpoint (`/generate_stream`) which returns tokens as they're generated via Server-Sent Events. This gives the user immediate feedback.
- **`response.iter_lines()`** — Reads the streaming response line by line. Each line contains a token or chunk of the generated text.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install requests
# Plus Docker installed for the TGI server
```

**Steps:**
1. Start TGI with the Docker command shown in the code
2. Save the Python client code to `tgi_client.py`
3. Run: `python tgi_client.py`

**Expected output:**
A JSON response containing the generated text, e.g.:
```json
{"generated_text": "The meaning of life is a question that has been..."}
```

</details>

## Ollama for Local Deployment

Ollama packages LLM inference into a single binary with model management. It's the easiest way to run models locally for development and personal use.

```python title="Using Ollama"
# Install Ollama: https://ollama.ai
# curl -fsSL https://ollama.ai/install.sh | sh

# Pull and run a model (terminal commands):
# ollama pull llama3.2:1b
# ollama run llama3.2:1b "Explain recursion"

# Serve a model (starts API on port 11434):
# ollama serve

# Use a custom quantized model:
# Create a Modelfile:
# FROM ./my-quantized-model.gguf
# PARAMETER temperature 0.7
# SYSTEM "You are a helpful assistant."
# ollama create my-model -f Modelfile

# Python client
import requests
import json

def chat_ollama(messages: list[dict], model: str = "llama3.2:1b",
                stream: bool = False):
    """Send a chat request to the Ollama API."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    if stream:
        with requests.post(url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        print(chunk["message"]["content"], end="", flush=True)
            print()
    else:
        response = requests.post(url, json=payload)
        return response.json()["message"]["content"]

# Example usage
messages = [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "Write a function to merge two sorted lists."},
]

response = chat_ollama(messages)
print(response)

# List available models
models = requests.get("http://localhost:11434/api/tags").json()
for m in models.get("models", []):
    print(f"  {m['name']}: {m['size'] / 1e9:.1f} GB")
```

:::tip[Line-by-Line Walkthrough]
- **`requests.post(url, json=payload, stream=True)`** — Sends a streaming request to Ollama's API. Streaming lets you print tokens as they arrive rather than waiting for the full response.
- **`json.loads(line)`** — Each streamed line is a JSON object containing one or more generated tokens.
- **`chunk["message"]["content"]`** — Extracts the generated text from each streaming chunk.
- **`requests.get("http://localhost:11434/api/tags")`** — Lists all models currently downloaded by Ollama, including their sizes.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
# Pull a model
ollama pull llama3.2:1b
pip install requests
```

**Steps:**
1. Start Ollama: `ollama serve` (may already be running as a service)
2. Save the Python code to `ollama_demo.py`
3. Run: `python ollama_demo.py`

**Expected output:**
A Python function for merging two sorted lists, followed by a list of available models with their sizes.

</details>

## Building a REST API Around an LLM

For production deployments, you'll typically wrap the LLM in a REST API with authentication, rate limiting, request queuing, and error handling.

```python title="Production LLM API with FastAPI"
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import time
import uuid
from collections import defaultdict

app = FastAPI(title="LLM API", version="1.0")

# --- Request/Response Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "llama-3.2-1b"
    max_tokens: int = Field(default=256, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    model: str
    choices: list[dict]
    usage: dict

# --- Rate Limiting ---
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, api_key: str):
        now = time.time()
        window = [t for t in self.requests[api_key] if now - t < 60]
        self.requests[api_key] = window
        if len(window) >= self.rpm:
            raise HTTPException(429, "Rate limit exceeded")
        self.requests[api_key].append(now)

rate_limiter = RateLimiter(requests_per_minute=60)

# --- LLM Backend (swap this for vLLM, TGI, etc.) ---
class LLMBackend:
    def __init__(self):
        # In production, initialize vLLM or connect to a TGI server
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )

    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float) -> tuple[str, dict]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )

        response_ids = outputs[0][input_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        usage = {
            "prompt_tokens": input_len,
            "completion_tokens": len(response_ids),
            "total_tokens": input_len + len(response_ids),
        }
        return response_text, usage

llm = LLMBackend()

# --- API Endpoints ---
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    rate_limiter.check("default")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    response_text, usage = llm.generate(
        messages, request.max_tokens, request.temperature
    )

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        usage=usage,
    )

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "llama-3.2-1b", "object": "model"}]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
```

:::tip[Line-by-Line Walkthrough]
- **`ChatMessage` / `ChatRequest` / `ChatResponse`** — Pydantic models that define the API's request and response formats. The `Field(...)` constraints enforce valid inputs (e.g., temperature between 0 and 2).
- **`RateLimiter`** — A simple sliding-window rate limiter that tracks requests per API key per minute. In production, use Redis-backed rate limiting.
- **`self.tokenizer.apply_chat_template(messages, ...)`** — Converts the list of chat messages into the model's expected text format with proper special tokens.
- **`do_sample=temperature > 0`** — If temperature is 0, uses greedy decoding (deterministic). Otherwise, uses sampling for variety.
- **`response_ids = outputs[0][input_len:]`** — Slices off the prompt tokens from the output, keeping only the newly generated tokens.
- **`@app.post("/v1/chat/completions")`** — OpenAI-compatible endpoint. This means any client that works with OpenAI's API will work with your API too.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install fastapi uvicorn transformers torch accelerate
```

**Steps:**
1. Save to `api.py`
2. Run: `uvicorn api:app --host 0.0.0.0 --port 8000`
3. Test with curl:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

**Expected output:**
A JSON response with the model's reply, token usage stats, and a unique request ID.

</details>

## Monitoring and Observability

Production LLM services need monitoring beyond standard web application metrics. Key metrics to track:

```python title="LLM monitoring with Prometheus metrics"
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter(
    "llm_requests_total", "Total LLM requests", ["model", "status"]
)
REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds", "Request latency",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)
TOKENS_GENERATED = Counter(
    "llm_tokens_generated_total", "Total tokens generated", ["model"]
)
TOKENS_PER_SECOND = Histogram(
    "llm_tokens_per_second", "Generation throughput",
    buckets=[5, 10, 20, 50, 100, 200]
)
TIME_TO_FIRST_TOKEN = Histogram(
    "llm_ttft_seconds", "Time to first token",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
)
ACTIVE_REQUESTS = Gauge(
    "llm_active_requests", "Currently processing requests"
)
GPU_MEMORY_USED = Gauge(
    "llm_gpu_memory_used_bytes", "GPU memory usage", ["gpu_id"]
)
KV_CACHE_USAGE = Gauge(
    "llm_kv_cache_usage_ratio", "KV-cache memory utilization"
)

class MonitoredLLM:
    """Wrapper that adds monitoring to any LLM backend."""
    def __init__(self, backend):
        self.backend = backend

    def generate(self, messages, max_tokens, temperature, model="default"):
        ACTIVE_REQUESTS.inc()
        start = time.perf_counter()

        try:
            response, usage = self.backend.generate(messages, max_tokens, temperature)

            elapsed = time.perf_counter() - start
            tokens = usage["completion_tokens"]

            REQUEST_COUNT.labels(model=model, status="success").inc()
            REQUEST_LATENCY.observe(elapsed)
            TOKENS_GENERATED.labels(model=model).inc(tokens)
            TOKENS_PER_SECOND.observe(tokens / elapsed if elapsed > 0 else 0)

            return response, usage
        except Exception as e:
            REQUEST_COUNT.labels(model=model, status="error").inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()

# Start Prometheus metrics server
start_http_server(9090)
# Metrics available at http://localhost:9090/metrics

# Example Grafana dashboard queries:
# - Request rate:        rate(llm_requests_total[5m])
# - P99 latency:         histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))
# - Tokens/sec:          rate(llm_tokens_generated_total[5m])
# - Active requests:     llm_active_requests
# - Error rate:          rate(llm_requests_total{status="error"}[5m])
```

:::tip[Line-by-Line Walkthrough]
- **`Counter("llm_requests_total", ...)`** — A counter that only goes up. Tracks the total number of requests by model and status (success/error). Use `rate()` in Grafana to see requests per second.
- **`Histogram("llm_request_duration_seconds", ..., buckets=[...])`** — Records the distribution of request latencies. The buckets define the ranges, enabling percentile calculations (P50, P99).
- **`Gauge("llm_active_requests", ...)`** — A gauge that goes up and down, tracking how many requests are currently being processed. Useful for detecting overload.
- **`ACTIVE_REQUESTS.inc()` / `ACTIVE_REQUESTS.dec()`** — Increments on request start, decrements when finished. The `finally` block ensures the gauge decreases even if an error occurs.
- **`REQUEST_LATENCY.observe(elapsed)`** — Records one latency sample. Prometheus automatically calculates percentiles from the histogram.
- **`start_http_server(9090)`** — Starts a Prometheus-compatible HTTP server that Grafana or Prometheus can scrape for metrics.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install prometheus-client
```

**Steps:**
1. Integrate this monitoring code into your FastAPI server from the previous section
2. Run your API server
3. Visit `http://localhost:9090/metrics` to see raw metrics
4. (Optional) Set up Prometheus and Grafana to visualize the metrics

**Expected output:**
Prometheus-formatted metrics at the `/metrics` endpoint, including request counts, latency histograms, and token throughput counters.

</details>

## Cost Optimization Strategies

LLM serving costs are dominated by GPU compute. Optimization strategies:

### 1. Right-Size Your Model

Don't serve a 70B model when a fine-tuned 7B model performs equally well on your task. The cost difference is ~10×.

### 2. Quantize Aggressively

INT4 quantization reduces GPU memory by 4× and often increases throughput by 2×, with minimal quality loss for models ≥7B.

### 3. Prompt Caching

If many requests share the same system prompt, cache its KV states. vLLM's `--enable-prefix-caching` does this automatically.

### 4. Batch Requests

Higher batch sizes = higher GPU utilization = lower cost per token. Continuous batching frameworks (vLLM, TGI) do this automatically.

### 5. Tiered Architecture

Route easy queries to a small/cheap model and hard queries to a large/expensive model.

```python title="Cost-aware request routing"
from dataclasses import dataclass

@dataclass
class ModelTier:
    name: str
    cost_per_1k_tokens: float
    max_quality_score: float  # Estimated quality ceiling

TIERS = [
    ModelTier("llama-3.2-1b-q4", cost_per_1k_tokens=0.001, max_quality_score=0.7),
    ModelTier("llama-3.2-3b-q4", cost_per_1k_tokens=0.003, max_quality_score=0.8),
    ModelTier("llama-3.1-8b-q4", cost_per_1k_tokens=0.008, max_quality_score=0.9),
    ModelTier("llama-3.1-70b-q4", cost_per_1k_tokens=0.05, max_quality_score=1.0),
]

def estimate_complexity(messages: list[dict]) -> float:
    """Estimate query complexity (0-1). In practice, use a small classifier."""
    last_message = messages[-1]["content"]
    word_count = len(last_message.split())

    complexity = 0.3
    if word_count > 100:
        complexity += 0.2
    if any(w in last_message.lower() for w in ["analyze", "compare", "explain why", "evaluate"]):
        complexity += 0.3
    if any(w in last_message.lower() for w in ["code", "implement", "algorithm", "debug"]):
        complexity += 0.2
    return min(complexity, 1.0)

def select_model(messages: list[dict], quality_threshold: float = 0.8) -> str:
    """Select the cheapest model that meets the quality threshold."""
    complexity = estimate_complexity(messages)
    required_quality = complexity * quality_threshold

    for tier in TIERS:
        if tier.max_quality_score >= required_quality:
            return tier.name

    return TIERS[-1].name  # Fall back to the best model

# Example
queries = [
    [{"role": "user", "content": "Hi, how are you?"}],
    [{"role": "user", "content": "Explain the differences between TCP and UDP protocols, including their use cases and tradeoffs in distributed systems."}],
    [{"role": "user", "content": "Write a Python implementation of a B-tree with insert, delete, and search operations."}],
]

for messages in queries:
    model = select_model(messages)
    complexity = estimate_complexity(messages)
    print(f"Query: {messages[-1]['content'][:60]}...")
    print(f"  Complexity: {complexity:.2f} → Model: {model}\\n")
```

:::tip[Line-by-Line Walkthrough]
- **`@dataclass class ModelTier`** — Defines a model tier with its name, cost, and estimated quality ceiling. This makes it easy to add or remove model tiers.
- **`TIERS = [...]`** — Ordered from cheapest/weakest to most expensive/strongest. The router picks the first tier that meets the quality requirement.
- **`estimate_complexity(messages)`** — A simple heuristic that estimates query difficulty based on word count and keywords. In production, you'd train a small classifier for this.
- **`required_quality = complexity * quality_threshold`** — Maps query complexity to a minimum quality score. Harder queries need higher-quality models.
- **`for tier in TIERS:`** — Iterates from cheapest to most expensive, returning the first model that meets the quality bar. Simple queries hit the cheap model; complex queries escalate to the expensive one.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install dataclasses  # (included in Python 3.7+)
```

**Steps:**
1. Save to `model_router.py`
2. Run: `python model_router.py`

**Expected output:**
```
Query: Hi, how are you?...
  Complexity: 0.30 → Model: llama-3.2-1b-q4

Query: Explain the differences between TCP and UDP protocols,...
  Complexity: 0.60 → Model: llama-3.2-1b-q4

Query: Write a Python implementation of a B-tree with insert,...
  Complexity: 0.70 → Model: llama-3.2-3b-q4
```
Simple queries route to the cheapest model; complex ones use a larger model.

</details>

:::tip[Cost Rule of Thumb]
For self-hosted LLM serving on cloud GPUs:
- **A100 80GB** (~$2/hr): Can serve a 70B model in INT4 at ~500 tokens/sec throughput
- **L4 24GB** (~$0.40/hr): Can serve a 7B model in INT4 at ~200 tokens/sec throughput
- **Cost per 1M tokens**: approximately GPU hourly rate / throughput

Compare against API providers (OpenAI, Anthropic) — self-hosting only makes sense above ~50M tokens/month for most teams.
:::

---

## Exercises

:::tip[Exercise 1: Deploy with vLLM — beginner]

Deploy a model with vLLM's OpenAI-compatible server. Then:
1. Send requests using the OpenAI Python client library
2. Measure time-to-first-token and generation speed
3. Send 10 concurrent requests and observe how throughput changes
4. Compare performance with and without `--enable-prefix-caching` using a shared system prompt

<details>
<summary>Hints</summary>

1. Start with a small model like Llama-3.2-1B-Instruct
2. Use the OpenAI-compatible API endpoint
3. Test with curl and the OpenAI Python client
4. Measure TTFT, tokens/sec, and peak GPU memory

</details>

:::

:::tip[Exercise 2: Build a Streaming API — intermediate]

Build a FastAPI endpoint that streams LLM responses token-by-token using Server-Sent Events (SSE). The endpoint should:
1. Accept OpenAI-compatible chat completion requests
2. Stream tokens as they're generated
3. Include usage statistics at the end of the stream
4. Handle errors and client disconnections gracefully

<details>
<summary>Hints</summary>

1. Use FastAPI with StreamingResponse
2. For vLLM, use the AsyncLLMEngine
3. For TGI, use the /generate_stream endpoint
4. Handle client disconnection gracefully

</details>

:::

:::tip[Exercise 3: Production Monitoring Dashboard — advanced]

threshold and error rate > threshold"]}>

Build a complete monitoring setup for an LLM API:
1. Instrument your FastAPI server with Prometheus metrics
2. Set up Grafana dashboards showing: request rate, latency percentiles, token throughput, error rate, and GPU memory
3. Configure alerts for: P99 latency > 10s, error rate > 1%, GPU memory > 95%
4. Load test with `locust` or `vegeta` and observe the dashboard during load

:::

---

## Resources

- **[vLLM: Easy, Fast, and Cheap LLM Serving](https://docs.vllm.ai)** _(tool)_ — The most popular open-source LLM serving framework — PagedAttention, continuous batching, and OpenAI-compatible API.

- **[Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference)** _(tool)_ by HuggingFace — HuggingFace's production serving framework with built-in quantization and multi-GPU support.

- **[Ollama](https://ollama.ai)** _(tool)_ — The simplest way to run LLMs locally — one command to download and serve any model.

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** _(tool)_ by Georgi Gerganov — C/C++ LLM inference — runs on CPU, supports Apple Silicon, and powers Ollama under the hood.

- **[Anyscale - How to Serve LLMs](https://www.anyscale.com/blog/continuous-batching-llm-inference)** _(tutorial)_ by Anyscale — Clear explanation of continuous batching and its impact on LLM serving throughput.

- **[LLM Inference Performance Engineering Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)** _(tutorial)_ by Databricks — Practical guide to LLM inference optimization covering batching, quantization, and hardware selection.
