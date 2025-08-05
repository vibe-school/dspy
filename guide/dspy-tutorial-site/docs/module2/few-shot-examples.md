---
sidebar_position: 2
---

# Few-Shot Learning with DSPy

Few-shot learning is a powerful technique where you provide examples to guide the model's behavior. DSPy takes this concept and supercharges it with automatic example selection and optimization.

## Traditional Few-Shot vs DSPy

### The Traditional Approach

In traditional prompting, you manually craft and maintain examples:

```python
# Traditional few-shot prompt (tedious and static)
prompt = """
Classify the sentiment of these tweets:

Example 1:
Tweet: "I love this product! Best purchase ever!"
Sentiment: positive

Example 2:
Tweet: "Completely disappointed. Waste of money."
Sentiment: negative

Example 3:
Tweet: "It's okay, nothing special."
Sentiment: neutral

Tweet: "This exceeded all my expectations!"
Sentiment:"""
```

### The DSPy Approach

DSPy automates example selection and optimization:

```python
import dspy

# Define the task
class SentimentClassification(dspy.Signature):
    """Classify tweet sentiment as positive, negative, or neutral."""
    tweet: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")

# DSPy automatically handles examples through optimization
classifier = dspy.Predict(SentimentClassification)
```

## Manual Few-Shot Examples in DSPy

While DSPy excels at automatic optimization, you can still provide manual examples:

```python
# Method 1: Examples in the signature
class JokeGeneration(dspy.Signature):
    """Generate jokes following the style of these examples:
    
    Topic: AI -> Setup: Why did the AI go to therapy?
    Punchline: It had too many deep learning issues!
    
    Topic: Python -> Setup: Why do Python programmers prefer dark mode?
    Punchline: Because light attracts bugs!
    """
    topic: str = dspy.InputField()
    setup: str = dspy.OutputField()
    punchline: str = dspy.OutputField()

# Method 2: Using dspy.Example
joke_examples = [
    dspy.Example(
        topic="AI",
        setup="Why did the AI go to therapy?",
        punchline="It had too many deep learning issues!"
    ),
    dspy.Example(
        topic="Python",
        setup="Why do Python programmers prefer dark mode?",
        punchline="Because light attracts bugs!"
    )
]
```

## Automatic Few-Shot Optimization

This is where DSPy truly shines - automatically finding the best examples:

### BootstrapFewShot Optimizer

```python
from dspy.teleprompt import BootstrapFewShot

# Create training data
trainset = [
    dspy.Example(tweet="Amazing service!", sentiment="positive"),
    dspy.Example(tweet="Terrible experience", sentiment="negative"),
    dspy.Example(tweet="It's fine", sentiment="neutral"),
    # ... more examples
]

# Define metric
def sentiment_accuracy(example, pred, trace=None):
    return example.sentiment.lower() == pred.sentiment.lower()

# Create optimizer
optimizer = BootstrapFewShot(
    metric=sentiment_accuracy,
    max_bootstrapped_demos=3  # Number of examples to include
)

# Compile with automatic example selection
optimized_classifier = optimizer.compile(
    student=dspy.Predict(SentimentClassification),
    trainset=trainset
)

# Now it includes the best examples automatically
result = optimized_classifier(tweet="This product is fantastic!")
print(result.sentiment)  # "positive"
```

## Advanced Few-Shot Patterns

### 1. Dynamic Example Selection

Select examples based on the input:

```python
class DynamicExampleSelector(dspy.Module):
    def __init__(self, examples_by_category):
        self.examples_by_category = examples_by_category
        self.classifier = dspy.ChainOfThought(SentimentClassification)
    
    def forward(self, tweet, category=None):
        # Select relevant examples
        if category and category in self.examples_by_category:
            examples = self.examples_by_category[category]
            # Include examples in context
            context = "\n".join([
                f"Tweet: {ex.tweet} -> Sentiment: {ex.sentiment}"
                for ex in examples[:3]
            ])
            return self.classifier(tweet=f"{context}\n\nTweet: {tweet}")
        return self.classifier(tweet=tweet)

# Usage
selector = DynamicExampleSelector({
    "product": product_examples,
    "service": service_examples,
    "general": general_examples
})
```

### 2. Example Quality Filtering

Ensure only high-quality examples are used:

```python
class QualityFilteredFewShot(dspy.Module):
    def __init__(self, signature, examples, quality_threshold=0.8):
        self.predictor = dspy.Predict(signature)
        self.quality_checker = dspy.Predict("example, prediction -> quality_score")
        self.filtered_examples = self._filter_examples(examples, quality_threshold)
    
    def _filter_examples(self, examples, threshold):
        filtered = []
        for ex in examples:
            # Test example quality
            pred = self.predictor(**ex.inputs())
            quality = self.quality_checker(
                example=str(ex),
                prediction=str(pred)
            )
            if float(quality.quality_score) >= threshold:
                filtered.append(ex)
        return filtered
```

### 3. Retrieval-Augmented Few-Shot

Dynamically retrieve relevant examples:

```python
class RetrievalAugmentedFewShot(dspy.Module):
    def __init__(self, signature, example_bank):
        self.classifier = dspy.Predict(signature)
        self.example_bank = example_bank  # Vector store of examples
        self.retriever = dspy.Predict("query -> relevant_examples")
    
    def forward(self, **kwargs):
        # Retrieve relevant examples
        query = str(kwargs)
        relevant = self.retriever(query=query)
        
        # Create few-shot prompt with retrieved examples
        context = self._format_examples(relevant.relevant_examples)
        
        # Add context to inputs
        enhanced_inputs = {**kwargs, "examples": context}
        return self.classifier(**enhanced_inputs)
```

## Practical Example: Multi-Class Classification

Let's build a complete few-shot classifier for customer support tickets:

```python
# Define the classification task
class TicketClassification(dspy.Signature):
    """Classify customer support tickets into categories."""
    ticket_text: str = dspy.InputField()
    category: str = dspy.OutputField(
        desc="billing, technical, account, or general"
    )
    priority: str = dspy.OutputField(desc="high, medium, or low")
    
# Create training examples
train_tickets = [
    dspy.Example(
        ticket_text="My credit card was charged twice",
        category="billing",
        priority="high"
    ),
    dspy.Example(
        ticket_text="How do I reset my password?",
        category="account",
        priority="medium"
    ),
    dspy.Example(
        ticket_text="The app crashes when I open it",
        category="technical",
        priority="high"
    ),
    # ... more examples
]

# Define evaluation metric
def classification_accuracy(example, pred, trace=None):
    category_match = example.category == pred.category
    priority_match = example.priority == pred.priority
    return (category_match and priority_match)

# Optimize with few-shot learning
optimizer = BootstrapFewShot(
    metric=classification_accuracy,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)

# Compile the classifier
optimized_classifier = optimizer.compile(
    student=dspy.ChainOfThought(TicketClassification),
    trainset=train_tickets
)

# Use the optimized classifier
new_ticket = "I can't log into my account and it's urgent"
result = optimized_classifier(ticket_text=new_ticket)
print(f"Category: {result.category}")  # "account"
print(f"Priority: {result.priority}")   # "high"
```

## Best Practices for Few-Shot Learning

### 1. Example Diversity

Ensure your examples cover different cases:

```python
def create_diverse_examples():
    return [
        # Positive examples
        dspy.Example(text="Absolutely love it!", sentiment="positive"),
        dspy.Example(text="Best decision ever", sentiment="positive"),
        
        # Negative examples  
        dspy.Example(text="Complete disaster", sentiment="negative"),
        dspy.Example(text="Waste of time", sentiment="negative"),
        
        # Neutral examples
        dspy.Example(text="It's okay", sentiment="neutral"),
        dspy.Example(text="Nothing special", sentiment="neutral"),
        
        # Edge cases
        dspy.Example(text="Not bad, but not great", sentiment="neutral"),
        dspy.Example(text="😍😍😍", sentiment="positive"),
    ]
```

### 2. Example Format Consistency

Keep examples consistent:

```python
class ConsistentExampleFormatter:
    @staticmethod
    def format_example(example):
        """Ensure consistent example format"""
        return dspy.Example(
            input_text=example.get("text", "").strip(),
            output_label=example.get("label", "").lower(),
            metadata={
                "source": example.get("source", "unknown"),
                "confidence": example.get("confidence", 1.0)
            }
        )
```

### 3. Progressive Example Complexity

Start simple and increase complexity:

```python
# Level 1: Basic examples
basic_examples = [
    dspy.Example(math="2+2", answer="4"),
    dspy.Example(math="5*3", answer="15")
]

# Level 2: Intermediate examples
intermediate_examples = [
    dspy.Example(math="(10+5)*2", answer="30"),
    dspy.Example(math="100/4-5", answer="20")
]

# Level 3: Complex examples
complex_examples = [
    dspy.Example(math="sqrt(16)+3^2", answer="13"),
    dspy.Example(math="log(100,10)*5", answer="10")
]
```

## Hands-On Exercise

Build a few-shot email classifier:

```python
# Exercise: Create an email priority classifier
# 1. Define signature for email classification
# 2. Create diverse training examples
# 3. Implement custom metric
# 4. Optimize with BootstrapFewShot
# 5. Test on new emails

class EmailPriority(dspy.Signature):
    """Classify email priority based on content."""
    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    sender: str = dspy.InputField()
    priority: str = dspy.OutputField(desc="urgent, high, normal, or low")

# Your implementation here...
```

## Common Pitfalls

1. **Too Many Examples**: More isn't always better - quality over quantity
2. **Biased Examples**: Ensure balanced representation across classes
3. **Overfitting to Examples**: Test with diverse inputs
4. **Ignoring Example Order**: DSPy optimizers handle this, but be aware when doing manual selection

## Summary

In this module, you learned:
- ✅ The difference between traditional few-shot and DSPy approaches
- ✅ How to use manual examples in DSPy
- ✅ Automatic few-shot optimization with BootstrapFewShot
- ✅ Advanced patterns like dynamic selection and retrieval
- ✅ Best practices for example creation and management

Next, we'll explore how system prompts and instructions work in DSPy.

---

<div style={{textAlign: 'center', marginTop: '2rem'}}>
  <a className="button button--secondary button--lg" href="/dspy/tutorial/module2/basic-generation">
    ← Back to Basic Generation
  </a>
  <span style={{margin: '0 1rem'}}></span>
  <a className="button button--primary button--lg" href="/dspy/tutorial/module2/system-prompts">
    Continue to System Prompts →
  </a>
</div>