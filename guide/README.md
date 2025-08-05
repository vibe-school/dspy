# DSPy Zero-to-Hero Tutorial Website

This directory contains the source code for the DSPy tutorial website built with Docusaurus.

## Quick Start

### Prerequisites

- Node.js 18.0 or higher
- npm or yarn

### Installation

```bash
cd dspy-tutorial-site
npm install
```

### Running Locally

```bash
npm start
```

This will start the development server at `http://localhost:3000`.

### Building

```bash
npm run build
```

This creates a production build in the `build/` directory.

### Deployment

The site is configured for GitHub Pages deployment:

```bash
npm run deploy
```

## Project Structure

```
guide/
├── converted-scripts/      # Python scripts extracted from notebooks
│   ├── level1/            # Basic generation examples
│   ├── level2/            # Multi-interaction patterns
│   ├── level3/            # Evaluation examples
│   ├── level4/            # Tool-calling agents
│   ├── level5/            # RAG implementations
│   └── optimizers/        # DSPy optimizer examples
└── dspy-tutorial-site/    # Docusaurus website
    ├── docs/              # Tutorial content
    │   ├── intro.md       # Welcome page
    │   ├── setup.md       # Environment setup
    │   ├── module1/       # DSPy fundamentals
    │   ├── module2/       # Atomic prompts
    │   ├── module3/       # Multi-interaction
    │   ├── module4/       # Evaluation
    │   ├── module5/       # Tools
    │   ├── module6/       # RAG
    │   ├── module7/       # Optimizers
    │   └── module8/       # Applications
    ├── src/               # React components
    └── docusaurus.config.ts

```

## Tutorial Content

The tutorial is organized into 8 modules:

1. **DSPy Fundamentals** - Core concepts and workflow
2. **Atomic Prompts** - Basic prompting patterns
3. **Multi-Interaction** - Complex interaction patterns
4. **Evaluation & Metrics** - Measuring performance
5. **Tool-Calling Agents** - Building agents with tools
6. **RAG Systems** - Retrieval-augmented generation
7. **DSPy Optimizers** - Automatic optimization techniques
8. **Practical Applications** - Production best practices

## Running Code Examples

All code examples are in `converted-scripts/`. To run them:

```bash
cd converted-scripts/level1
python 01_basic_generation.py
```

Make sure you have the required environment variables set:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` (optional)
- `TAVILY_API_KEY` (for Level 4 & 5)

## Contributing

1. Tutorial content is in Markdown files in `docs/`
2. Code examples should be tested and placed in `converted-scripts/`
3. Follow the existing structure and style
4. Test locally before submitting PRs

## Resources

- [DSPy Official Docs](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Original YouTube Course](https://youtu.be/5Bym0ffALaU)
- [AVB's Patreon](https://www.patreon.com/NeuralBreakdownwithAVB)

## License

This tutorial is based on the work by AVB and the DSPy team at Stanford NLP.