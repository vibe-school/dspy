"""
DSPy Modules Overview
Demonstrates different modules: ChainOfThought, ProgramOfThought, ReAct, MultiChainComparison
"""
import dspy
from typing import Literal

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

print("=== 1. Chain of Thought Module ===")
# ChainOfThought adds reasoning step
cot_emotion = dspy.ChainOfThought('input -> sentiment: str')
text = "That was phenomenal, but I hated it!"
cot_response = cot_emotion(input=text)
print("Sentiment:", cot_response.sentiment)
print("Reasoning:", cot_response.reasoning)
print()

print("=== 2. Program of Thought Module ===")
# ProgramOfThought generates and executes code
class MathAnalysis(dspy.Signature):
    """Analyze a dataset and compute various statistical metrics."""
    numbers: list[float] = dspy.InputField(desc="List of numerical values to analyze")
    required_metrics: list[str] = dspy.InputField(desc="List of metrics to calculate (e.g. ['mean', 'variance', 'quartiles'])")
    analysis_results: dict[str, float] = dspy.OutputField(desc="Dictionary containing the calculated metrics")

math_analyzer = dspy.ProgramOfThought(MathAnalysis)
data = [1.5, 2.8, 3.2, 4.7, 5.1, 2.3, 3.9]
metrics = ['mean', 'median']

try:
    pot_response = math_analyzer(numbers=data, required_metrics=metrics)
    print("Results:", pot_response.analysis_results)
    if hasattr(pot_response, 'reasoning'):
        print("Reasoning:", pot_response.reasoning)
except Exception as e:
    print(f"ProgramOfThought execution error: {e}")
print()

print("=== 3. ReAct Module ===")
# ReAct combines reasoning with tool usage
def wikipedia_search(query: str) -> list[str]:
    """Simulated Wikipedia search - returns mock results."""
    # In real implementation, this would connect to Wikipedia API
    mock_results = {
        "world series 1983": ["The Baltimore Orioles won the 1983 World Series, defeating the Philadelphia Phillies four games to one."],
        "world cup 1966": ["England won the 1966 FIFA World Cup, beating West Germany 4–2 in the final match."]
    }
    
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return ["No results found"]

# Create ReAct module with tools
react_module = dspy.ReAct('question -> response', tools=[wikipedia_search])
text = "Who won the world series in 1983?"

try:
    react_response = react_module(question=text)
    print("Answer:", react_response.response)
    if hasattr(react_response, 'reasoning'):
        print("Reasoning:", react_response.reasoning)
except Exception as e:
    print(f"ReAct execution note: {e}")
    print("Note: ReAct requires actual tool implementation for full functionality")
print()

print("=== 4. Multi Chain Comparison ===")
# Run multiple completions with different temperatures
text = "That was phenomenal!"
cot_completions = []

for i in range(3):
    # Temperature increases: 0.7, 0.8, 0.9
    temp_config = dict(temperature=0.7 + (0.1 * i))
    completion = cot_emotion(input=text, config=temp_config)
    cot_completions.append(completion)
    print(f"Completion {i+1} (temp={0.7 + (0.1 * i)}):", completion.sentiment)

# Synthesize with MultiChainComparison
mcot_emotion = dspy.MultiChainComparison('input -> sentiment', M=3)
final_result = mcot_emotion(completions=cot_completions, input=text)

print("\nFinal Synthesized Result:")
print("Sentiment:", final_result.sentiment)
if hasattr(final_result, 'rationale'):
    print("Rationale:", final_result.rationale)
print()

print("=== 5. Majority Voting ===")
# Simple majority voting across completions
majority_result = dspy.majority(cot_completions, field='sentiment')
print("Most common sentiment:", majority_result.sentiment)