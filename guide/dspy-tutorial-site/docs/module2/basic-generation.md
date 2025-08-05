---
sidebar_position: 1
---

# Basic Generation with DSPy

Welcome to Module 2! Let's explore the fundamentals of text generation using DSPy, starting from simple prompts and building up to more sophisticated patterns.

## Module Processing Flow

Understanding how DSPy processes your requests is key to mastering the framework:

```
┌─────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Input  │────▶│  Signature   │────▶│   Module    │────▶│  Output  │
└─────────┘     └──────────────┘     └─────────────┘     └──────────┘
                       │                      │
                  Define I/O              Process with
                  & Constraints           LLM Strategy
```

## Simple Text Generation

Let's start with the most basic form of generation:

```python
import dspy

# Configure your LLM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Method 1: String signature
generate = dspy.Predict("topic -> text")
result = generate(topic="artificial intelligence")
print(result.text)

# Method 2: Class-based signature for more control
class TextGeneration(dspy.Signature):
    """Generate text about a given topic."""
    topic: str = dspy.InputField(desc="The topic to write about")
    text: str = dspy.OutputField(desc="Generated text about the topic")

generate = dspy.Predict(TextGeneration)
result = generate(topic="quantum computing")
print(result.text)
```

## Adding Constraints and Context

DSPy allows you to add constraints to guide generation:

```python
class ConstrainedGeneration(dspy.Signature):
    """Generate text with specific constraints."""
    topic: str = dspy.InputField()
    max_words: int = dspy.InputField(desc="Maximum number of words")
    tone: str = dspy.InputField(desc="Tone of the text (formal/casual/humorous)")
    text: str = dspy.OutputField(desc="Generated text following all constraints")

constrained_gen = dspy.Predict(ConstrainedGeneration)

result = constrained_gen(
    topic="machine learning",
    max_words=50,
    tone="casual"
)
print(result.text)
```

## Working with Multiple Outputs

Generate multiple pieces of content in one call:

```python
class MultiOutput(dspy.Signature):
    """Generate multiple related outputs."""
    topic: str = dspy.InputField()
    title: str = dspy.OutputField(desc="A catchy title")
    summary: str = dspy.OutputField(desc="Brief summary")
    main_points: list[str] = dspy.OutputField(desc="Key points")
    conclusion: str = dspy.OutputField(desc="Concluding statement")

multi_gen = dspy.Predict(MultiOutput)
result = multi_gen(topic="renewable energy")

print(f"Title: {result.title}")
print(f"Summary: {result.summary}")
print(f"Main Points: {result.main_points}")
print(f"Conclusion: {result.conclusion}")
```

## Using Chain of Thought

For complex generation tasks, Chain of Thought helps the model reason through its response:

```python
class ComplexGeneration(dspy.Signature):
    """Generate a detailed explanation with reasoning."""
    question: str = dspy.InputField()
    explanation: str = dspy.OutputField()

# Using ChainOfThought adds reasoning
cot_gen = dspy.ChainOfThought(ComplexGeneration)
result = cot_gen(question="How do neural networks learn?")

print(f"Reasoning: {result.reasoning}")  # CoT adds this field
print(f"Explanation: {result.explanation}")
```

## Structured Output with Pydantic

For type-safe outputs, integrate Pydantic models:

```python
from pydantic import BaseModel
from typing import List

class BlogPost(BaseModel):
    title: str
    introduction: str
    sections: List[str]
    tags: List[str]
    word_count: int

class BlogGeneration(dspy.Signature):
    """Generate a structured blog post."""
    topic: str = dspy.InputField()
    target_audience: str = dspy.InputField()
    post: BlogPost = dspy.OutputField()

blog_gen = dspy.Predict(BlogGeneration)
result = blog_gen(
    topic="Introduction to DSPy",
    target_audience="Python developers"
)

# Access structured data
print(f"Title: {result.post.title}")
print(f"Tags: {', '.join(result.post.tags)}")
print(f"Word count: {result.post.word_count}")
```

## Controlling Generation Parameters

Fine-tune generation behavior:

```python
class CreativeWriting(dspy.Signature):
    """Generate creative text with specific parameters."""
    prompt: str = dspy.InputField()
    genre: str = dspy.InputField()
    story: str = dspy.OutputField()

# Configure generation parameters
creative_gen = dspy.Predict(
    CreativeWriting,
    temperature=0.9,  # Higher = more creative
    max_tokens=500
)

result = creative_gen(
    prompt="A robot discovers emotions",
    genre="science fiction"
)
```

## Common Patterns and Best Practices

### 1. Validation in Signatures

```python
class ValidatedGeneration(dspy.Signature):
    """Generate text with built-in validation."""
    topic: str = dspy.InputField()
    min_length: int = dspy.InputField()
    text: str = dspy.OutputField(
        desc="Text that must be at least min_length characters"
    )
    
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        # Add custom validation
        if len(result.text) < kwargs['min_length']:
            raise ValueError("Generated text too short")
        return result
```

### 2. Contextual Generation

```python
class ContextualGeneration(dspy.Signature):
    """Generate text based on context."""
    context: str = dspy.InputField(desc="Background information")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Answer based on context")

# Use for Q&A, summarization, etc.
contextual = dspy.Predict(ContextualGeneration)
result = contextual(
    context="DSPy is a framework for programming LLMs...",
    question="What is DSPy?"
)
```

### 3. Iterative Refinement

```python
class RefineText(dspy.Signature):
    """Refine and improve existing text."""
    original_text: str = dspy.InputField()
    feedback: str = dspy.InputField()
    refined_text: str = dspy.OutputField()

refiner = dspy.Predict(RefineText)

# Initial generation
text = "DSPy is great for AI."

# Refine based on feedback
refined = refiner(
    original_text=text,
    feedback="Make it more detailed and professional"
)
print(refined.refined_text)
```

## Hands-On Exercise

Try building a content generation pipeline:

```python
# Exercise: Build a social media content generator
# 1. Generate post text based on topic
# 2. Create relevant hashtags
# 3. Suggest best posting time
# 4. Generate an engaging caption

class SocialMediaContent(dspy.Signature):
    """Generate complete social media content."""
    topic: str = dspy.InputField()
    platform: str = dspy.InputField(desc="twitter/instagram/linkedin")
    post_text: str = dspy.OutputField()
    hashtags: list[str] = dspy.OutputField()
    best_time: str = dspy.OutputField()
    caption: str = dspy.OutputField()

# Your implementation here
social_gen = dspy.ChainOfThought(SocialMediaContent)
content = social_gen(
    topic="DSPy tutorial launch",
    platform="twitter"
)
```

## Common Pitfalls

1. **Over-constraining outputs**: Too many constraints can lead to poor results
2. **Ignoring token limits**: Always consider model token limits
3. **Not validating outputs**: Add validation for critical applications
4. **Using Predict when ChainOfThought is better**: Complex tasks benefit from reasoning

## Summary

In this module, you learned:
- ✅ Basic text generation with DSPy
- ✅ Adding constraints and context
- ✅ Working with structured outputs
- ✅ Using Chain of Thought for complex generation
- ✅ Best practices for generation tasks

Next, we'll explore more advanced patterns like few-shot learning and prompt optimization.

---

<div style={{textAlign: 'center', marginTop: '2rem'}}>
  <a className="button button--secondary button--lg" href="/dspy/tutorial/module1/fundamentals">
    ← Back to Fundamentals
  </a>
  <span style={{margin: '0 1rem'}}></span>
  <a className="button button--primary button--lg" href="/dspy/tutorial/module2/few-shot-examples">
    Continue to Few-Shot Examples →
  </a>
</div>