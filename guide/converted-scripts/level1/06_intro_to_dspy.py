"""
Level 1: Introduction to DSPy
This example shows how DSPy simplifies and improves upon traditional prompting
"""
import dspy

# Configure DSPy with language model
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Define a signature - this replaces manual prompt engineering
class JokeSignature(dspy.Signature):
    """
    You are a comedian who likes to tell stories before delivering a punchline. You are always funny.
    """
    query: str = dspy.InputField()
    setup: str = dspy.OutputField()
    punchline: str = dspy.OutputField()
    contradiction: str = dspy.OutputField()
    delivery: str = dspy.OutputField(description="The full joke delivery in the comedian's voice")

# Example 1: Basic Predict
print("=== Example 1: Basic DSPy Predict ===")
joke_generator = dspy.Predict(JokeSignature)
joke = joke_generator(query="Write a joke about AI that has to do with them turning rogue.")
print("Setup:", joke.setup)
print("Punchline:", joke.punchline)
print("Contradiction:", joke.contradiction)
print("Delivery:", joke.delivery)
print()

# Example 2: Chain of Thought
print("=== Example 2: DSPy Chain of Thought ===")
joke_generator_cot = dspy.ChainOfThought(JokeSignature)
joke_cot = joke_generator_cot(query="Write a joke about AI that has to do with them turning rogue.")
print("Reasoning:", joke_cot.reasoning)
print("Setup:", joke_cot.setup)
print("Punchline:", joke_cot.punchline)
print("Contradiction:", joke_cot.contradiction)
print("Delivery:", joke_cot.delivery)