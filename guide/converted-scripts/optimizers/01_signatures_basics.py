"""
DSPy Signatures - Basic Examples
Demonstrates different types of signatures in DSPy
"""
import dspy
from typing import Literal

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

print("=== 1. Simple Input & Output ===")
# Basic signature with single input/output
qna = dspy.Predict('question -> answer')
response = qna(question="Why is the sky blue?")
print("Response:", response.answer)
print()

# Document summarization
sum_module = dspy.Predict('document -> summary')
document = """
The market for our products is intensely competitive and is characterized by rapid technological change and evolving industry standards. 
We believe that the principal competitive factors in this market are performance, breadth of product offerings, access to customers and partners and distribution channels, software support, conformity to industry standard APIs, manufacturing capabilities, processor pricing, and total system costs.
"""
response = sum_module(document=document)
print("Summary:", response.summary)
print()

print("=== 2. Multiple Inputs and Outputs ===")
# Multiple inputs and outputs
multi = dspy.Predict('question, context -> answer, citation')
question = "What's my name?"
context = "The user you're talking to is Adam Lucek, AI youtuber extraordinaire"
response = multi(question=question, context=context)
print("Answer:", response.answer)
print("Citation:", response.citation)
print()

print("=== 3. Type Hints with Outputs ===")
# Type hints in signatures
emotion = dspy.Predict('input -> sentiment: str, confidence: float, reasoning: str')
text = "I don't quite know, I didn't really like it"
response = emotion(input=text)
print("Sentiment Classification:", response.sentiment)
print("Confidence:", response.confidence)
print("Reasoning:", response.reasoning)
print()

print("=== 4. Class Based Signatures ===")
# Advanced class-based signature
class TextStyleTransfer(dspy.Signature):
    """Transfer text between different writing styles while preserving content."""
    text: str = dspy.InputField()
    source_style: Literal["academic", "casual", "business", "poetic"] = dspy.InputField()
    target_style: Literal["academic", "casual", "business", "poetic"] = dspy.InputField()
    preserved_keywords: list[str] = dspy.OutputField()
    transformed_text: str = dspy.OutputField()
    style_metrics: dict[str, float] = dspy.OutputField(desc="Scores for formality, complexity, emotiveness")

text = "This coffee shop makes the best lattes ever! Their new barista really knows what he's doing with the espresso machine."
style_transfer = dspy.Predict(TextStyleTransfer)

response = style_transfer(
    text=text,
    source_style="casual",
    target_style="poetic"
)

print("Transformed Text:", response.transformed_text)
print("Style Metrics:", response.style_metrics)
print("Preserved Keywords:", response.preserved_keywords)