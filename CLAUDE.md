# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two main DSPy tutorial projects:
1. **context-engineering-dspy**: A comprehensive 5-level tutorial from AVB's YouTube course on Context Engineering with DSPy
2. **dspy-breakdown**: An in-depth notebook exploring DSPy framework concepts and optimizers

The goal is to convert these tutorials into a Zero-to-Hero DSPy Tutorial Website with structured levels of complexity and real-world examples.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r pyproject.toml
```

### Running Tutorial Scripts
```bash
# Navigate to specific level and run scripts
cd context-engineering-dspy/level2_multi_interaction
uv run t1_sequence.py

# Or run any Python script
uv run python script_name.py
```

### MLflow for Evaluation (Level 3)
```bash
# Start MLflow server for experiment tracking
uv run mlflow server --backend-store-uri sqlite:///mydb.sqlite --port 5000
# Access dashboard at localhost:5000
```

### Data Preparation for RAG (Level 5)
```bash
# Download dataset from https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes
# Extract to level5_rags/data/
cd context-engineering-dspy/level5_rags
uv run vector_embedding.py
```

## Code Architecture

### DSPy Core Concepts
- **Signatures**: Define input/output specifications for LLM operations
- **Modules**: Composable units that process signatures (Predict, ChainOfThought, etc.)
- **Optimizers**: Automatic prompt tuning and optimization strategies
- **Metrics**: Evaluation functions to measure program performance

### Tutorial Structure

#### Level 1: Atomic Prompts
- Basic prompting techniques
- System prompts and few-shot examples
- Introduction to DSPy signatures

#### Level 2: Multi-Interaction
- Sequential flows (t1_sequence.py)
- Iterative refinement (t2_iterative_refinement.py)
- Conditional branching (t3_conditional_branch.py)
- Multiple outputs and refinement
- Self-reflection patterns (t4_reflection.py)

#### Level 3: Evaluation
- Reflection-based evaluation
- Pairwise ELO comparisons
- MLflow integration for experiment tracking

#### Level 4: Tools
- Tool-calling agents
- Integration with external APIs (Tavily search)
- Modular tool design

#### Level 5: RAGs
- Basic embedding-based retrieval
- BM25 retrieval
- HyDE (Hypothetical Document Embeddings)
- Rank fusion techniques
- Annoy vector indexing

### Key Dependencies
- **dspy**: Core framework (v2.6.27+)
- **google-genai**: Google Gemini integration
- **openai**: OpenAI API support
- **tavily-python**: Web search functionality
- **mlflow**: Experiment tracking
- **annoy**: Approximate nearest neighbor search
- **rank-bm25**: BM25 retrieval

### Environment Variables Required
```bash
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here  # For Level 4 & 5 web search
```

## Common Patterns

### Creating a DSPy Module
```python
class CustomModule(dspy.Module):
    def __init__(self):
        self.step1 = dspy.Predict(Signature1)
        self.step2 = dspy.Predict(Signature2)
    
    def forward(self, input):
        result1 = self.step1(input=input)
        result2 = self.step2(prev=result1)
        return result2
```

### Using Pydantic Models for Structured Output
```python
class StructuredOutput(BaseModel):
    field1: str
    field2: List[str]

class MySignature(dspy.Signature):
    input: str = dspy.InputField()
    output: StructuredOutput = dspy.OutputField()
```

## File Naming Conventions
- Tutorial scripts: `t{number}_{description}.py` (e.g., t1_sequence.py)
- Utility modules: `{function}_utils.py` (e.g., print_utils.py)
- Main entry points: `main.py`
- Data preparation: `prepare_data.py`