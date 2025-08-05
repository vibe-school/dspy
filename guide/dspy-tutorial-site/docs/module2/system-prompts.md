---
sidebar_position: 3
---

# System Prompts and Instructions in DSPy

System prompts set the context and behavior for language models. DSPy reimagines this concept through signature docstrings, field descriptions, and automatic instruction optimization.

## Traditional System Prompts vs DSPy

### The Traditional Way

In traditional LLM programming, system prompts are static strings:

```python
# Traditional approach - manual system prompt
messages = [
    {
        "role": "system", 
        "content": "You are a helpful assistant that always responds in a professional tone. Follow these rules:..."
    },
    {
        "role": "user",
        "content": "Write a product description"
    }
]
```

### The DSPy Way

DSPy integrates instructions directly into signatures:

```python
import dspy

class ProductDescription(dspy.Signature):
    """You are a professional copywriter creating compelling product descriptions.
    Always highlight key benefits and use persuasive language."""
    
    product_name: str = dspy.InputField()
    features: list[str] = dspy.InputField(desc="Key product features")
    description: str = dspy.OutputField(desc="Compelling 2-3 paragraph description")
```

## Components of DSPy Instructions

### 1. Signature Docstrings

The docstring acts as the primary instruction:

```python
class EmailResponse(dspy.Signature):
    """You are a customer service representative responding to emails.
    Be empathetic, professional, and solution-oriented.
    Always acknowledge the customer's concern before providing solutions."""
    
    customer_email: str = dspy.InputField()
    response: str = dspy.OutputField()

# The docstring becomes part of the prompt
responder = dspy.Predict(EmailResponse)
```

### 2. Field Descriptions

Field descriptions provide specific guidance:

```python
class CodeReview(dspy.Signature):
    """Review code for quality and suggest improvements."""
    
    code: str = dspy.InputField(
        desc="The code to review"
    )
    language: str = dspy.InputField(
        desc="Programming language (python, javascript, etc.)"
    )
    review: str = dspy.OutputField(
        desc="Detailed review covering: 1) Code quality, 2) Best practices, 3) Potential bugs, 4) Performance suggestions"
    )
    rating: int = dspy.OutputField(
        desc="Overall quality rating from 1-10"
    )
```

### 3. Module-Level Instructions

Some modules add their own instructions:

```python
# ChainOfThought automatically adds reasoning instructions
cot_reviewer = dspy.ChainOfThought(CodeReview)

# The module adds: "Let's think step by step..."
result = cot_reviewer(code="def add(a, b): return a + b", language="python")
print(result.reasoning)  # Shows the thinking process
```

## Advanced Instruction Patterns

### 1. Role-Based Instructions

Create specific personas:

```python
class RoleBasedGeneration(dspy.Signature):
    """You are {role} with expertise in {domain}.
    Your communication style is {style}.
    Always consider {considerations}."""
    
    role: str = dspy.InputField()
    domain: str = dspy.InputField()
    style: str = dspy.InputField()
    considerations: str = dspy.InputField()
    query: str = dspy.InputField()
    response: str = dspy.OutputField()

# Dynamic role assignment
expert = dspy.Predict(RoleBasedGeneration)
result = expert(
    role="a senior data scientist",
    domain="machine learning and statistics",
    style="technical but accessible",
    considerations="business impact and implementation feasibility",
    query="Should we use deep learning for customer churn prediction?"
)
```

### 2. Structured Instructions

Break down complex instructions:

```python
class StructuredAnalysis(dspy.Signature):
    """Perform a comprehensive analysis following this structure:
    
    1. CONTEXT: Understand the background and requirements
    2. ANALYSIS: Break down the problem systematically
    3. RECOMMENDATIONS: Provide actionable suggestions
    4. RISKS: Identify potential challenges
    5. NEXT STEPS: Outline implementation plan
    
    Be thorough but concise. Use bullet points where appropriate."""
    
    topic: str = dspy.InputField()
    data: str = dspy.InputField(desc="Relevant data or information")
    analysis: str = dspy.OutputField(desc="Complete structured analysis")
```

### 3. Conditional Instructions

Instructions that adapt based on input:

```python
class AdaptiveInstructions(dspy.Module):
    def __init__(self):
        # Different signatures for different scenarios
        self.technical = dspy.Predict("""
            Provide a highly technical explanation with implementation details.
            Include code examples and performance considerations.
            target_audience, topic -> explanation
        """)
        
        self.simple = dspy.Predict("""
            Explain in simple terms that anyone can understand.
            Use analogies and avoid jargon.
            target_audience, topic -> explanation
        """)
    
    def forward(self, topic, audience_type):
        if audience_type == "technical":
            return self.technical(target_audience=audience_type, topic=topic)
        else:
            return self.simple(target_audience=audience_type, topic=topic)
```

## Automatic Instruction Optimization

DSPy can automatically optimize instructions using COPRO:

```python
from dspy.teleprompt import COPRO

# Define task with basic instructions
class MathWordProblem(dspy.Signature):
    """Solve the math word problem step by step."""
    problem: str = dspy.InputField()
    solution: str = dspy.OutputField()

# Training data
trainset = [
    dspy.Example(
        problem="If John has 5 apples and buys 3 more, how many does he have?",
        solution="8"
    ),
    # ... more examples
]

# Metric to evaluate
def math_accuracy(example, pred, trace=None):
    return example.solution.strip() == pred.solution.strip()

# COPRO optimizes the instruction
optimizer = COPRO(
    metric=math_accuracy,
    breadth=3,  # Number of instruction candidates
    depth=2,    # Refinement iterations
)

# Compile with optimized instructions
optimized_solver = optimizer.compile(
    student=dspy.ChainOfThought(MathWordProblem),
    trainset=trainset
)

# The instructions are now optimized!
print(optimized_solver.predictors()[0].signature.__doc__)
```

## Instruction Strategies

### 1. Layered Instructions

Build complexity gradually:

```python
class LayeredInstructions(dspy.Module):
    def __init__(self):
        # Base layer - general behavior
        self.base = dspy.Predict("""
            You are a helpful assistant. Be accurate and clear.
            input -> output
        """)
        
        # Specific layer - task-specific rules
        self.specific = dspy.Predict("""
            Follow these specific rules for technical writing:
            1. Use active voice
            2. Define acronyms on first use
            3. Include examples for complex concepts
            content -> refined_content
        """)
    
    def forward(self, input_text):
        # Apply layers sequentially
        base_output = self.base(input=input_text)
        refined = self.specific(content=base_output.output)
        return refined
```

### 2. Instruction Templates

Create reusable instruction patterns:

```python
class InstructionTemplate:
    @staticmethod
    def create_analyst_signature(domain, output_format):
        class_name = f"{domain}Analysis"
        docstring = f"""
        You are a {domain} analyst providing expert insights.
        
        Your analysis should:
        - Be data-driven and objective
        - Consider multiple perspectives
        - Provide actionable recommendations
        - Follow {output_format} format
        
        Always support claims with evidence.
        """
        
        # Dynamically create signature
        return type(class_name, (dspy.Signature,), {
            "__doc__": docstring,
            "data": dspy.InputField(desc="Data to analyze"),
            "context": dspy.InputField(desc="Business context"),
            "analysis": dspy.OutputField(desc=f"{output_format} formatted analysis")
        })

# Usage
FinancialAnalysis = InstructionTemplate.create_analyst_signature(
    "financial", "executive summary"
)
analyzer = dspy.Predict(FinancialAnalysis)
```

### 3. Multi-Modal Instructions

Combine different instruction types:

```python
class MultiModalInstructions(dspy.Signature):
    """PRIMARY ROLE: You are a senior software architect.
    
    CONTEXT: You're reviewing code for a critical production system.
    
    GUIDELINES:
    - Security is the top priority
    - Performance is secondary
    - Maintainability is important
    
    OUTPUT REQUIREMENTS:
    1. Identify all security vulnerabilities
    2. Suggest performance optimizations
    3. Rate code quality (1-10)
    4. Provide refactoring suggestions
    
    TONE: Professional but approachable"""
    
    code: str = dspy.InputField()
    language: str = dspy.InputField()
    security_issues: list[str] = dspy.OutputField()
    performance_tips: list[str] = dspy.OutputField()
    quality_score: int = dspy.OutputField()
    refactor_suggestions: str = dspy.OutputField()
```

## Practical Example: Customer Support Bot

Let's build a complete customer support system with sophisticated instructions:

```python
# Define the support signature with detailed instructions
class CustomerSupport(dspy.Signature):
    """You are an experienced customer support specialist for TechCorp.
    
    PERSONALITY TRAITS:
    - Empathetic and patient
    - Solution-oriented
    - Professional yet friendly
    
    RESPONSE STRUCTURE:
    1. Acknowledge the customer's issue
    2. Express understanding/empathy
    3. Provide clear solution steps
    4. Offer additional assistance
    
    POLICIES:
    - Never admit fault without investigation
    - Always offer a ticket number for follow-up
    - Escalate to human agent for: legal issues, threats, or complex technical problems
    
    TONE: Warm and helpful, avoiding corporate jargon"""
    
    customer_message: str = dspy.InputField(
        desc="The customer's message or complaint"
    )
    customer_history: str = dspy.InputField(
        desc="Previous interactions and purchase history"
    )
    
    response: str = dspy.OutputField(
        desc="Complete customer response following all guidelines"
    )
    ticket_number: str = dspy.OutputField(
        desc="Reference ticket number (format: TECH-YYYY-MM-XXXXX)"
    )
    escalation_needed: bool = dspy.OutputField(
        desc="Whether this needs human agent escalation"
    )

# Create the support module
support_bot = dspy.ChainOfThought(CustomerSupport)

# Example usage
result = support_bot(
    customer_message="My laptop crashed and I lost all my work! This is the third time this month!",
    customer_history="Premium customer, 3 previous technical issues this month"
)

print(f"Response: {result.response}")
print(f"Ticket: {result.ticket_number}")
print(f"Escalate: {result.escalation_needed}")
```

## Best Practices

### 1. Clear and Specific Instructions

```python
# ❌ Vague
class VagueSignature(dspy.Signature):
    """Be helpful and nice."""
    input: str = dspy.InputField()
    output: str = dspy.OutputField()

# ✅ Specific
class SpecificSignature(dspy.Signature):
    """As a technical documentation expert:
    1. Use clear, concise language
    2. Include code examples for all concepts
    3. Follow Google's documentation style guide
    4. Assume reader has intermediate programming knowledge"""
    concept: str = dspy.InputField()
    documentation: str = dspy.OutputField()
```

### 2. Testable Instructions

Make instructions verifiable:

```python
class TestableInstructions(dspy.Signature):
    """Generate a summary that:
    - Is exactly 3 sentences long
    - Contains no more than 50 words
    - Mentions the main topic in the first sentence
    - Includes a conclusion in the last sentence"""
    
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

# Easy to verify these constraints programmatically
```

### 3. Avoid Instruction Conflicts

```python
# ❌ Conflicting instructions
class ConflictingSignature(dspy.Signature):
    """Be extremely detailed and thorough.
    Keep responses under 50 words."""  # Conflict!
    
# ✅ Coherent instructions
class CoherentSignature(dspy.Signature):
    """Provide a comprehensive analysis within these constraints:
    - Maximum 200 words
    - Cover: problem, solution, and impact
    - Use bullet points for clarity"""
```

## Hands-On Exercise

Create a code documentation generator with sophisticated instructions:

```python
# Exercise: Build a documentation generator
# 1. Create signature with role-based instructions
# 2. Include multiple output formats (docstring, README, API docs)
# 3. Add style guidelines
# 4. Implement quality checks

class CodeDocGenerator(dspy.Signature):
    """You are a technical writer specializing in developer documentation.
    
    Your task is to create clear, comprehensive documentation that follows these principles:
    - Write for developers with varying experience levels
    - Include practical examples
    - Follow the Diataxis framework (tutorials, how-tos, reference, explanation)
    - Use active voice and present tense
    
    # Your implementation here...
    """
    
    code: str = dspy.InputField()
    doc_type: str = dspy.InputField(desc="docstring, readme, or api")
    # Add output fields...
```

## Common Pitfalls

1. **Over-instructing**: Too many rules can confuse the model
2. **Contradictory instructions**: Ensure all guidelines align
3. **Ignoring model capabilities**: Some models follow instructions better than others
4. **Static instructions**: Not leveraging DSPy's optimization capabilities

## Summary

In this module, you learned:
- ✅ How DSPy handles system prompts through signatures
- ✅ Using docstrings and field descriptions effectively
- ✅ Automatic instruction optimization with COPRO
- ✅ Advanced patterns for complex instructions
- ✅ Best practices for instruction design

Next, we'll explore structured outputs with Pydantic integration.

---

<div style={{textAlign: 'center', marginTop: '2rem'}}>
  <a className="button button--secondary button--lg" href="/dspy/tutorial/module2/few-shot-examples">
    ← Back to Few-Shot Examples
  </a>
  <span style={{margin: '0 1rem'}}></span>
  <a className="button button--primary button--lg" href="/dspy/tutorial/module2/structured-outputs">
    Continue to Structured Outputs →
  </a>
</div>