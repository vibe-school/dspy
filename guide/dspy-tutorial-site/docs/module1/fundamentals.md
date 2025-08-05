---
sidebar_position: 1
---

# DSPy Fundamentals

Welcome to Module 1! Let's understand what makes DSPy different from traditional prompting and learn the core concepts.

## Traditional Prompting vs DSPy

### The Problem with Traditional Prompting

When working with LLMs traditionally, we spend countless hours:
- Crafting the perfect prompt
- Adding examples manually
- Tweaking wording for better results
- Managing prompt versions
- Dealing with model-specific quirks

```python
# Traditional approach - manual prompt engineering
prompt = """You are a helpful assistant that classifies sentiment.
Given a tweet, classify it as positive, negative, or neutral.

Examples:
Tweet: "I love this product!" -> positive
Tweet: "This is terrible" -> negative
Tweet: "It's okay I guess" -> neutral

Tweet: "Best day ever!"
Classification:"""

response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

### The DSPy Approach

DSPy transforms this into:

```python
import dspy

# Define what you want, not how to ask for it
class SentimentClassification(dspy.Signature):
    """Classify the sentiment of a tweet."""
    tweet: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")

# Create a predictor
classify = dspy.Predict(SentimentClassification)

# Use it
result = classify(tweet="Best day ever!")
print(result.sentiment)  # "positive"
```

## Core Concepts

DSPy is built on four fundamental components that work together to create powerful LLM applications:

```
                            ┌─────────────────┐
                            │   DSPy Program  │
                            └────────┬────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
   ┌────▼─────┐              ┌──────▼──────┐            ┌────────▼────────┐
   │Signatures│              │   Modules   │            │   Optimizers    │
   └────┬─────┘              └──────┬──────┘            └────────┬────────┘
        │                           │                             │
   Input/Output              Predict, CoT,                 BootstrapFewShot,
   Specifications            ReAct, etc.                   COPRO, MIPROv2
```

### 1. Signatures

Signatures define the input/output behavior of your LLM calls. Think of them as function signatures for LLMs.

```python
# Simple string format
simple_sig = dspy.Predict("question -> answer")

# Multiple inputs/outputs
multi_sig = dspy.Predict("context, question -> answer, confidence")

# Class-based with types and descriptions
class QASignature(dspy.Signature):
    """Answer questions based on context."""
    context: str = dspy.InputField(desc="Relevant background information")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Detailed answer")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
```

### 2. Modules

Modules are building blocks that process signatures. They implement different prompting strategies.

```python
# Predict - Basic prediction
predict = dspy.Predict(SentimentClassification)

# ChainOfThought - Adds reasoning step
cot = dspy.ChainOfThought(SentimentClassification)

# ProgramOfThought - Generates executable code
pot = dspy.ProgramOfThought(MathProblem)

# ReAct - Reasoning + Acting with tools
react = dspy.ReAct(ResearchQuestion, tools=[search_tool])
```

### 3. Metrics

Metrics measure how well your program performs. They're functions that return a score.

```python
def exact_match(example, prediction, trace=None):
    """Check if prediction matches expected output exactly."""
    return example.answer.lower() == prediction.answer.lower()

def quality_metric(example, prediction, trace=None):
    """More complex metric using another LLM call."""
    assessment = dspy.Predict("text, criteria -> score")(
        text=prediction.answer,
        criteria="Is this answer accurate and helpful?"
    )
    return float(assessment.score) > 0.8
```

### 4. Optimizers

Optimizers automatically improve your program by finding better prompts, examples, or instructions.

```python
from dspy.teleprompt import BootstrapFewShot

# Create optimizer
optimizer = BootstrapFewShot(metric=exact_match)

# Compile your program with training data
optimized_program = optimizer.compile(
    student=predict,
    trainset=training_examples
)
```

## Your First DSPy Program

Let's build a complete example:

```python
import dspy
import os

# 1. Configure DSPy with your LLM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 2. Define your task signature
class JokeGeneration(dspy.Signature):
    """Generate a joke based on a topic."""
    topic: str = dspy.InputField(desc="The topic for the joke")
    setup: str = dspy.OutputField(desc="The setup of the joke")
    punchline: str = dspy.OutputField(desc="The punchline")

# 3. Create a module
joke_generator = dspy.ChainOfThought(JokeGeneration)

# 4. Use it!
result = joke_generator(topic="artificial intelligence")
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")
print(f"Reasoning: {result.reasoning}")  # ChainOfThought adds this
```

## DSPy Workflow

The typical DSPy workflow consists of 4 steps:

1. **Define**: Create signatures for your tasks
2. **Build**: Compose modules into programs  
3. **Optimize**: Use metrics and optimizers to improve
4. **Deploy**: Use your optimized program in production

```python
# Complete workflow example
# 1. Define
class TextSummary(dspy.Signature):
    """Summarize long text concisely."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

# 2. Build
summarizer = dspy.ChainOfThought(TextSummary)

# 3. Optimize (with training data)
from dspy.teleprompt import BootstrapFewShot

def summary_quality(example, pred, trace=None):
    # Check if summary captures key points
    return len(pred.summary.split()) < 50  # Simple length check

optimizer = BootstrapFewShot(metric=summary_quality, max_bootstrapped_demos=3)
optimized_summarizer = optimizer.compile(summarizer, trainset=train_examples)

# 4. Deploy
production_summary = optimized_summarizer(text="Your long document here...")
```

## Key Advantages

1. **Modularity**: Build complex programs from simple components
2. **Optimization**: Automatically improve performance with data
3. **Portability**: Switch between LLMs without rewriting prompts
4. **Maintainability**: Version and test your LLM programs systematically
5. **Interpretability**: Understand why your program makes decisions

## Hands-On Exercise

Try this exercise to solidify your understanding:

```python
# Exercise: Create a translation program
# 1. Define a signature for translation
# 2. Use ChainOfThought module
# 3. Test with different languages

# Your code here:
class Translation(dspy.Signature):
    """Translate text between languages."""
    text: str = dspy.InputField()
    source_language: str = dspy.InputField()
    target_language: str = dspy.OutputField()
    translation: str = dspy.OutputField()

translator = dspy.ChainOfThought(Translation)
result = translator(
    text="Hello world",
    source_language="English",
    target_language="Spanish"
)
print(result.translation)
```

## Common Pitfalls

1. **Over-complicating signatures**: Start simple, add complexity as needed
2. **Ignoring metrics**: Good metrics are crucial for optimization
3. **Not using traces**: Traces help debug and understand your program
4. **Skipping optimization**: Even basic optimization significantly improves results

## Summary

In this module, you learned:
- ✅ How DSPy differs from traditional prompting
- ✅ The four core concepts: Signatures, Modules, Metrics, Optimizers
- ✅ The DSPy workflow: Define, Build, Optimize, Deploy
- ✅ How to create your first DSPy program

Next, we'll dive deeper into atomic prompts and basic DSPy usage patterns.

---

<div style={{textAlign: 'center', marginTop: '2rem'}}>
  <a className="button button--secondary button--lg" href="/tutorial/setup">
    ← Back to Setup
  </a>
  <span style={{margin: '0 1rem'}}></span>
  <a className="button button--primary button--lg" href="/tutorial/module2/atomic-prompts">
    Continue to Module 2 →
  </a>
</div>