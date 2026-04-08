# AI Engineering Academy

**From beginner to AI engineer in 6 months.**

A structured, open curriculum that takes you from "What is AI?" to building and optimizing your own large language models — with code at every step.

## Curriculum Overview

| Month | Topic | What You'll Build |
|-------|-------|-------------------|
| **1** | AI & ML Foundations | Your first neural network from scratch |
| **2** | Deep Learning | CNNs, RNNs, transfer learning projects |
| **3** | NLP & Transformers | A Transformer from the ground up |
| **4** | Large Language Models | GPT-style language model (nanoGPT) |
| **5** | LLM Optimization | Fine-tune, quantize, and deploy an LLM |
| **6** | Building New Models | Design, train, evaluate, and ship a model |

Each month includes lessons with interactive examples, hands-on exercises, a capstone project, and curated resources (papers, videos, courses).

## Getting Started

### Prerequisites

- Basic programming experience (Python preferred)
- High-school level math (algebra, basic statistics)
- A computer with Python 3.10+ installed
- Curiosity and willingness to learn

### Local Development

```bash
npm install
npm start
```

This starts a local development server at `http://localhost:3000`. Most changes are reflected live without restarting.

### Build

```bash
npm run build
```

Generates a production-ready static site into the `build` directory.

### Type Checking

```bash
npm run typecheck
```

## Tech Stack

- [Docusaurus 3](https://docusaurus.io/) — static site generator
- [React 19](https://react.dev/) — UI components
- [KaTeX](https://katex.org/) — math rendering in lessons
- [MDX](https://mdxjs.com/) — interactive Markdown

## Project Structure

```
ai-engineering-academy/
├── docs/                  # Curriculum content (Markdown)
│   ├── month-1/           #   AI & ML Foundations
│   ├── month-2/           #   Deep Learning
│   ├── month-3/           #   NLP & Transformers
│   ├── month-4/           #   Large Language Models
│   ├── month-5/           #   LLM Optimization
│   └── month-6/           #   Building New Models
├── blog/                  # Resource posts (videos, papers, blogs)
├── src/
│   ├── components/        # Custom React components
│   ├── css/               # Global styles
│   └── pages/             # Custom pages (homepage)
├── static/img/            # Images and assets
├── docusaurus.config.ts   # Site configuration
├── sidebars.ts            # Curriculum sidebar structure
└── package.json
```

## Contributing

Contributions are welcome! Whether it's fixing a typo, improving an explanation, or suggesting a new resource:

1. Fork the repository
2. Create a branch for your change
3. Make your edits
4. Open a pull request with a clear description of what you changed and why

## License

This project is open source. See the repository for license details.
