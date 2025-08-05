# DSPy Classification Gist Sources

This directory contains recent DSPy classification examples and patterns collected from various GitHub gists and repositories (2024-2025).

## Overview

DSPy (Declarative Self-improving Python) is a framework for programming—not prompting—language models. These examples demonstrate various classification patterns using DSPy's modular approach.

## Directory Structure

```
gist_sources/
├── classification/           # Classification examples
│   ├── sentiment_classification.py    # Sentiment and emotion classification
│   ├── email_classification.py        # Multi-step email processing pipeline
│   └── csv_classification.py          # CSV-based classification with optimization
└── utilities/               # Helper utilities
    └── signature_utils.py   # Utilities for creating and managing signatures
```

## Classification Examples

### 1. Sentiment Classification (`sentiment_classification.py`)

Basic and advanced sentiment analysis patterns:
- **Basic Sentiment**: Classify text as positive, negative, or neutral
- **Emotion Classification**: Identify emotions (joy, sadness, anger, etc.)
- **Sentiment with Confidence**: Include confidence scores
- **Aspect-Based Sentiment**: Extract aspects and their sentiments

```python
# Example usage
classifier = SentimentClassifier()
result = classifier(text="I love this product!")
# Output: sentiment='positive'
```

### 2. Email Classification Pipeline (`email_classification.py`)

Complex multi-step email processing based on GitHub issue #164:
- **Email Summarization**: Handle long email threads
- **Category Classification**: Classify into predefined categories
- **Information Extraction**: Extract category-specific details

Pipeline flow:
1. Summarize long emails
2. Classify into categories (refund, technical support, etc.)
3. Extract relevant information based on category

```python
# Example usage
pipeline = EmailProcessingPipeline()
result = pipeline(email_text=email_content)
# Output: category, confidence, extracted details
```

### 3. CSV-Based Classification (`csv_classification.py`)

Production-ready classification with data loading and optimization:
- **DataLoader**: Load classification data from CSV files
- **Optimization**: Use BootstrapFewShot or MIPRO optimizers
- **Evaluation**: Built-in metrics and evaluation framework

Features:
- Train/test split utilities
- Multiple optimization strategies
- Custom evaluation metrics
- Sample data generation

```python
# Example workflow
examples = DataLoader.load_csv('data.csv')
trainset, testset = DataLoader.split_data(examples)
optimizer = OptimizedCSVClassifier(trainset)
optimized_model = optimizer.optimize_with_bootstrap()
```

## Utilities

### Signature Utilities (`signature_utils.py`)

Helper functions for working with DSPy signatures:

1. **Pydantic to DSPy**: Convert Pydantic models to DSPy signatures
2. **Create Classification Signatures**: Generate classification signatures dynamically
3. **Combine Signatures**: Merge multiple signatures
4. **Inspect Signatures**: Analyze signature structure

```python
# Create a classification signature dynamically
sig = create_classification_signature(
    categories=['spam', 'ham'],
    include_confidence=True
)
```

## Key DSPy Concepts Used

1. **Signatures**: Define input/output specifications
2. **Modules**: Composable units (Predict, ChainOfThought)
3. **Optimizers**: BootstrapFewShot, MIPRO for automatic optimization
4. **Metrics**: Custom evaluation functions

## Requirements

```python
dspy>=2.6.27
pandas  # For CSV handling
pydantic>=2.0  # For structured outputs
```

## Usage Tips

1. **Start Simple**: Begin with basic signatures and add complexity
2. **Use Type Hints**: Leverage Literal types for constrained outputs
3. **Optimize Iteratively**: Start with BootstrapFewShot, try MIPRO for better results
4. **Evaluate Thoroughly**: Use custom metrics relevant to your task

## References

- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [DSPy Documentation](https://dspy.ai/)
- [Context Engineering with DSPy Course](https://github.com/avbiswas/context-engineering-dspy)
- Original gist sources from 2024-2025 GitHub discussions

## Contributing

Feel free to add more classification examples or improve existing ones. Each example should:
1. Include clear documentation
2. Show practical use cases
3. Demonstrate DSPy best practices
4. Be runnable with minimal setup