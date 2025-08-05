# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## 📊 Progress Tracking

### Documentation Status

| Module | Topic | Docs Created | Sample App | Tested | Notes |
|--------|-------|:------------:|:----------:|:------:|-------|
| Module 1 | DSPy Fundamentals | ✅ | ⬜ | ⬜ | Core concepts documented |
| Module 2 | Atomic Prompts | ⬜ | ⬜ | ⬜ | basic-generation.md created |
| Module 3 | Multi-Interaction | ⬜ | ⬜ | ⬜ | Sequential, iterative, branching |
| Module 4 | Evaluation & Metrics | ⬜ | ⬜ | ⬜ | MLflow integration needed |
| Module 5 | Tool-Calling Agents | ⬜ | ⬜ | ⬜ | Tavily search examples |
| Module 6 | RAG Systems | ⬜ | ⬜ | ⬜ | Vector, BM25, HyDE |
| Module 7 | DSPy Optimizers | ⬜ | ⬜ | ⬜ | All optimizer types |
| Module 8 | Practical Applications | ⬜ | ⬜ | ⬜ | Production patterns |

### Sub-module Documentation Progress

| Module | Sub-topics | Status |
|--------|------------|--------|
| Module 2 | constraints-and-context | ⬜ Not started |
| Module 2 | few-shot-examples | ⬜ Not started |
| Module 2 | system-prompts | ⬜ Not started |
| Module 2 | structured-outputs | ⬜ Not started |
| Module 2 | dspy-vs-traditional | ⬜ Not started |
| Module 3 | sequential-flows | ⬜ Not started |
| Module 3 | iterative-refinement | ⬜ Not started |
| Module 3 | conditional-branching | ⬜ Not started |
| Module 3 | multiple-outputs | ⬜ Not started |
| Module 3 | self-reflection | ⬜ Not started |

### Sample Applications Status

| Application | Description | Status | Location |
|-------------|-------------|:------:|----------|
| Basic Sentiment Classifier | Simple DSPy sentiment analysis | ⬜ | `/examples/sentiment` |
| Multi-step QA System | Chain of thought Q&A | ⬜ | `/examples/qa-system` |
| Tool-calling Agent | Tavily search integration | ⬜ | `/examples/search-agent` |
| RAG Application | Document Q&A with retrieval | ⬜ | `/examples/rag-demo` |
| Optimized Classifier | Bootstrap few-shot example | ⬜ | `/examples/optimized` |

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
