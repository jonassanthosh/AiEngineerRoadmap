---
sidebar_position: 1
title: "What Is Artificial Intelligence?"
slug: what-is-ai
---


# What Is Artificial Intelligence?

:::info[What You'll Learn]
- The history of AI from Turing to foundation models
- AI winters and their lessons for modern practitioners
- Types of AI: narrow vs. general vs. superintelligence
- The difference between AI, ML, deep learning, and generative AI
- What AI engineering means as a career
:::

**Estimated time:** Reading: ~30 min | Exercises: ~2 hours

Artificial intelligence is one of the most transformative technologies in human history—and also one of the most misunderstood. Before we write a single line of code, we need to understand what AI actually *is*, where it came from, and where it's heading. This chapter gives you the conceptual foundation that separates an AI engineer from someone who merely calls APIs.

## A Brief History of AI

### The Birth of an Idea (1940s–1960s)

The dream of thinking machines predates computers themselves. In 1950, Alan Turing published *Computing Machinery and Intelligence*, proposing the now-famous **Turing Test**: if a machine can converse with a human evaluator so convincingly that the evaluator cannot reliably distinguish it from another human, the machine should be considered intelligent.

The field was formally born at the **Dartmouth Conference** in 1956, where John McCarthy coined the term "artificial intelligence." The early pioneers—McCarthy, Marvin Minsky, Allen Newell, Herbert Simon—believed that general-purpose intelligence was within reach in a generation. Early successes fed this optimism: the Logic Theorist (1956) proved mathematical theorems, and ELIZA (1966) simulated conversation convincingly enough to fool some users.

:::info[The Turing Test]
Turing's original formulation is the "imitation game": a human judge communicates via text with both a human and a machine. If the judge cannot reliably tell which is which, the machine passes the test. Modern LLMs have arguably passed versions of this test, but whether they are truly "intelligent" remains hotly debated.
:::

### Expert Systems and the Knowledge Era (1970s–1980s)

When the early optimism faded, AI research shifted toward **expert systems**—programs that encoded domain knowledge as if-then rules. MYCIN (1976) diagnosed bacterial infections. XCON (1980) configured computer orders at DEC, saving the company millions.

Expert systems were commercially successful but brittle. They couldn't learn from data, couldn't handle novel situations, and required painstaking manual knowledge engineering. Every new domain meant building a new system from scratch.

### The AI Winters

The gap between promises and reality led to two periods of collapsed funding and public disillusionment:

| Period | Trigger | Consequence |
|--------|---------|-------------|
| **First AI Winter** (1974–1980) | Minsky & Papert's critique of perceptrons; failure to deliver on machine translation promises | UK and US government funding slashed |
| **Second AI Winter** (1987–1993) | Collapse of the Lisp machine market; expert systems proved too expensive to maintain | Corporate AI budgets evaporated |

These winters are worth understanding because they teach a crucial lesson: **hype cycles are dangerous**. As AI engineers, we need to be honest about what our systems can and cannot do.

### The Statistical Turn and Machine Learning (1990s–2000s)

As expert systems waned, a quieter revolution was underway. Researchers started letting machines *learn from data* instead of programming rules by hand. Key developments:

- **Support Vector Machines** (Vapnik, 1995) brought rigorous statistical learning theory
- **Random Forests** (Breiman, 2001) showed that ensembles of weak learners could be powerful
- **IBM Deep Blue** (1997) defeated world chess champion Garry Kasparov
- **Statistical NLP** replaced hand-written grammar rules with probabilistic models

### The Deep Learning Revolution (2010s)

The modern era of AI was ignited by three converging forces:

1. **Data**: The internet produced massive labeled datasets like ImageNet (14 million images)
2. **Compute**: GPUs turned out to be excellent for the matrix math underlying neural networks
3. **Algorithms**: Researchers rediscovered and improved neural network techniques

The watershed moment was **2012**, when Alex Krizhevsky's deep convolutional neural network (AlexNet) won the ImageNet competition by a staggering margin. Suddenly, deep learning wasn't a curiosity—it was the state of the art.

Milestones came rapidly:

- **2014**: GANs (Goodfellow) could generate realistic images
- **2016**: AlphaGo (DeepMind) defeated world Go champion Lee Sedol
- **2017**: *Attention Is All You Need* introduced the Transformer architecture
- **2018**: BERT showed that pre-training on massive text corpora produced powerful language representations
- **2020**: GPT-3 demonstrated that scaling language models produced emergent capabilities
- **2023**: GPT-4 raised the bar for reasoning; ChatGPT reached 100 million users faster than any product in history; open-source LLMs exploded (Llama 2, Mistral 7B)
- **2024**: Claude 3, Gemini 1.5, and Llama 3 pushed quality higher; mixture-of-experts went mainstream (Mixtral); reasoning models emerged (OpenAI o1); video generation arrived (Sora)
- **2025**: Llama 4 and DeepSeek-V3/R1 showed open models matching closed models on most benchmarks; on-device AI became practical; AI coding assistants became standard developer tools
- **2026**: AI engineering solidified as a distinct career path; multi-modal models became the default; reasoning-augmented generation became a standard pattern

### The Modern Era: Foundation Models (2020s–Present)

We are now in the era of **foundation models**—large models pre-trained on vast datasets that can be adapted to countless downstream tasks. What started with GPT-3 in 2020 has evolved into a rich ecosystem: closed-source leaders (GPT-4, Claude, Gemini) compete alongside increasingly capable open-weight models (Llama 4, DeepSeek-V3, Mistral). Models are now natively multi-modal, reasoning-capable, and efficient enough to run on consumer hardware. The gap between open and closed models has narrowed dramatically, and AI engineering—building reliable products on top of these models—has emerged as a discipline in its own right.

:::tip[Why History Matters for Engineers]
Understanding AI's history isn't academic nostalgia. The patterns repeat: overpromising leads to winters, narrow benchmarks don't equal general capability, and the most impactful breakthroughs often come from recombining old ideas with new resources. As an AI engineer, this perspective helps you evaluate new claims critically.
:::

## Types of AI

AI researchers and philosophers distinguish between three levels of artificial intelligence:

### Narrow AI (Weak AI)

**Narrow AI** systems are designed and trained for a specific task. Every AI system deployed in production today is narrow AI:

- A spam classifier that labels emails
- A recommendation engine that suggests movies
- GPT-4 generating text
- A self-driving car's perception system

Even the most capable LLMs are narrow AI. They excel at text-based tasks but cannot independently learn to play chess, drive a car, or manipulate physical objects without explicit training for those domains.

### Artificial General Intelligence (AGI)

**AGI** refers to a hypothetical system with human-level cognitive abilities across all domains. An AGI could learn any intellectual task a human can, transfer knowledge between domains, and operate autonomously in open-ended environments.

No AGI system exists today. Estimates for when (or whether) it will arrive range from "within a decade" to "never." This is one of the most debated questions in the field.

### Artificial Superintelligence (ASI)

**ASI** is a hypothetical system that surpasses human intelligence in every dimension—scientific creativity, social intelligence, general wisdom. This concept lives firmly in the realm of philosophy and speculation, but it motivates important work in **AI safety and alignment**.

:::warning[Don't Confuse Capability with Generality]
Modern LLMs can appear general-purpose because natural language is a flexible interface. But "can answer questions about many topics" is not the same as "can learn arbitrary new skills autonomously." Keep this distinction clear when evaluating AI claims.
:::

## The Current AI Landscape

### Large Language Models (LLMs)

LLMs are the technology driving the current AI boom. Built on the Transformer architecture, they are trained to predict the next token in a sequence over trillions of tokens of text. The landscape in 2026 is defined by intense competition between closed and open models, native multi-modality, and built-in reasoning capabilities. Key players:

- **OpenAI**: GPT-4o, o1/o3 reasoning models, Sora (video generation)
- **Anthropic**: Claude 4 family with extended thinking
- **Google**: Gemini 2 family with native multi-modality
- **Meta**: Llama 4 open-weight models, competitive with closed alternatives
- **DeepSeek**: V3/R1 models proving open-source can match frontier performance
- **Mistral, Cohere, and others**: Specialized, efficient, and domain-specific models

### Computer Vision

Vision is no longer a separate discipline—it's a built-in capability. Modern foundation models are natively multi-modal: they process images, video, and audio alongside text. Standalone vision models still matter for latency-sensitive applications like autonomous driving and medical imaging, but the default path for most AI engineers is to use multi-modal foundation models.

### Robotics and Embodied AI

The frontier of applying foundation models to physical systems. Companies like Figure, Tesla, and Boston Dynamics are building robots that combine LLM reasoning with physical manipulation.

### AI Infrastructure

A massive ecosystem supports AI development: cloud GPU providers (AWS, GCP, Azure), specialized hardware (NVIDIA H200s/B200s, Google TPUs), model serving frameworks (vLLM, SGLang), observability platforms, vector databases, and on-device inference runtimes. The infrastructure layer has matured rapidly, making it feasible to run capable models on everything from data-center clusters to laptops and phones.

## Where AI Engineering Fits In

AI engineering is distinct from AI research. Here's how the roles compare:

| Role | Focus | Typical Output |
|------|-------|----------------|
| **AI Researcher** | Advancing the state of the art | Papers, new architectures, theoretical insights |
| **ML Engineer** | Building and deploying ML models | Training pipelines, model serving infrastructure |
| **AI Engineer** | Building products with AI capabilities | Applications, APIs, integrations using pre-trained models |
| **Data Scientist** | Extracting insights from data | Analyses, dashboards, statistical models |

**AI engineering** sits at the intersection of software engineering and machine learning. As an AI engineer, you will:

- Understand how models work well enough to use them effectively
- Design systems that integrate AI capabilities into products
- Evaluate and select models for specific use cases
- Fine-tune and optimize models for production
- Build reliable, scalable AI-powered applications
- Handle the messy realities: latency, cost, hallucinations, safety

:::info[The T-Shaped AI Engineer]
The most effective AI engineers are T-shaped: deep expertise in building AI-powered software, with broad understanding across ML fundamentals, infrastructure, and the specific domain they work in. This curriculum is designed to build that T shape.
:::

## The AI Engineering Stack

Modern AI engineering involves multiple layers:

```python title="The AI Engineering Stack (Conceptual)"
ai_engineering_stack = {
    "Application Layer": [
        "User interfaces",
        "API endpoints",
        "Business logic",
    ],
    "Orchestration Layer": [
        "Prompt engineering",
        "Agent frameworks (LangChain, LlamaIndex)",
        "RAG pipelines",
        "Evaluation and monitoring",
    ],
    "Model Layer": [
        "Foundation models (GPT-4, Claude, Llama)",
        "Fine-tuned models",
        "Embedding models",
    ],
    "Infrastructure Layer": [
        "GPU compute (cloud or on-prem)",
        "Vector databases",
        "Model serving (vLLM, TGI)",
        "MLOps and observability",
    ],
    "Data Layer": [
        "Training data curation",
        "Evaluation datasets",
        "Knowledge bases",
    ],
}

for layer, components in ai_engineering_stack.items():
    print(f"\\n{layer}:")
    for component in components:
        print(f"  - {component}")
```

:::tip[Line-by-Line Walkthrough]
- **`ai_engineering_stack = { ... }`** — Creates a Python dictionary where each key is the name of a layer in the AI stack and each value is a list of components that belong to that layer. Think of it like a filing cabinet: each drawer (layer) holds a collection of folders (components).
- **`for layer, components in ai_engineering_stack.items():`** — Loops through the dictionary one drawer at a time, giving us the drawer label (`layer`) and the list of folders inside it (`components`).
- **`print(f"\\n{layer}:")`** — Prints the name of the current layer with a blank line before it, acting as a section header.
- **`for component in components:`** — Loops through each individual component inside the current layer.
- **`print(f"  - {component}")`** — Prints each component as a bulleted list item, indented under its layer.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Python 3.8+ (no additional packages needed — this uses only built-in features).

**Steps:**
1. Save the code to a file, e.g. `ai_stack.py`
2. Open a terminal and navigate to the folder where you saved it
3. Run: `python ai_stack.py`

**Expected output:**
```
Application Layer:
  - User interfaces
  - API endpoints
  - Business logic

Orchestration Layer:
  - Prompt engineering
  - Agent frameworks (LangChain, LlamaIndex)
  - RAG pipelines
  - Evaluation and monitoring
...
```
(Each layer prints its name followed by its components as a bulleted list.)

</details>

## Exercises

<ExerciseBlock title="Exercise 1: AI System Taxonomy" difficulty="beginner" hints={["Think about what input each system takes and what output it produces.", "Consider whether the system needs to generalize across domains or only works within one."]}>

Classify each of the following as narrow AI, and explain what specific task each is optimized for:

1. Google Translate
2. Tesla Autopilot
3. ChatGPT
4. A hospital's diagnostic imaging system
5. Spotify's recommendation engine

For each one, describe a task *outside* its domain that it could not perform without retraining.

</ExerciseBlock>

<ExerciseBlock title="Exercise 2: Turing Test Debate" difficulty="intermediate" hints={["Consider what 'intelligence' means—is it about behavior or understanding?", "Look up the Chinese Room argument by John Searle.", "Think about whether passing the Turing Test requires consciousness."]}>

Write a 300-word essay arguing either *for* or *against* the following claim:

> "Modern LLMs like GPT-4 have effectively passed the Turing Test, and therefore should be considered intelligent."

Support your argument with specific examples and address at least one counterargument.

</ExerciseBlock>

<ExerciseBlock title="Exercise 3: AI Timeline Research" difficulty="beginner" hints={["Use Wikipedia, arxiv.org, and the resources below as starting points.", "Focus on why each milestone mattered, not just what happened."]}>

Create a timeline of the 10 most important events in AI history from 1950 to today. For each event, write 2-3 sentences explaining its significance and lasting impact on the field.

</ExerciseBlock>

<ExerciseBlock title="Exercise 4: AI Engineering vs. ML Research" difficulty="intermediate" hints={["Think about the different skills, tools, and day-to-day activities.", "Consider what 'success' looks like in each role."]}>

You're advising a computer science graduate who wants to work in AI. They ask: "Should I become an AI researcher or an AI engineer?" Write a comparison covering:

1. Day-to-day work
2. Required skills
3. Career trajectory
4. Impact and job satisfaction

Be specific—don't just list vague differences.

</ExerciseBlock>

## Resources

<ResourceCard title="Computing Machinery and Intelligence" url="https://academic.oup.com/mind/article/LIX/236/433/986238" type="paper" author="Alan Turing (1950)" description="The paper that started it all. Turing's original proposal for the imitation game and his arguments for machine intelligence." />

<ResourceCard title="Attention Is All You Need" url="https://arxiv.org/abs/1706.03762" type="paper" author="Vaswani et al. (2017)" description="The paper that introduced the Transformer architecture—the foundation of every modern LLM." />

<ResourceCard title="AI: A Modern Approach" url="http://aima.cs.berkeley.edu/" type="book" author="Stuart Russell & Peter Norvig" description="The definitive AI textbook. Comprehensive coverage from search algorithms to deep learning." />

<ResourceCard title="3Blue1Brown: Neural Networks" url="https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi" type="video" author="Grant Sanderson" description="Beautiful visual explanations of how neural networks learn. Essential viewing." />

<ResourceCard title="The AI Revolution: The Road to Superintelligence" url="https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html" type="tutorial" author="Tim Urban (Wait But Why)" description="A long-form, accessible exploration of where AI is headed and why it matters." />

<ResourceCard title="Stanford CS229: Machine Learning" url="https://cs229.stanford.edu/" type="course" author="Andrew Ng" description="The classic Stanford ML course. Rigorous mathematical foundations with practical applications." />

---

**Next up**: Before we can understand how AI systems learn, we need the mathematical language they speak. In the next chapter, we'll cover the essential math foundations: linear algebra, calculus, and probability.
