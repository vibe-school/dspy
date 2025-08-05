"""
DSPy Metrics and Evaluation
Demonstrates how to create and use metrics for evaluation
"""
import dspy
from typing import Literal

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

print("=== 1. Example Data Type ===")
# Create example data
qa_pair = dspy.Example(question="What is my name?", answer="Your name is Adam Lucek")
print(qa_pair)
print("Question:", qa_pair.question)
print("Answer:", qa_pair.answer)
print()

# Example with explicit inputs
article_summary = dspy.Example(
    article="Placeholder for Article", 
    summary="Expected Summary"
).with_inputs("article")
print("Input fields only:", article_summary.inputs())
print("Label fields only:", article_summary.labels())
print()

print("=== 2. Simple Metrics ===")
# Define a simple sentiment classifier
class TwtSentiment(dspy.Signature):
    tweet: str = dspy.InputField(desc="Candidate tweet for classification")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

twt_sentiment = dspy.ChainOfThought(TwtSentiment)

# Create sample data
examples = [
    dspy.Example(tweet="I love this product!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="This is terrible", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="It's okay I guess", sentiment="neutral").with_inputs("tweet"),
]

# Define simple exact match metric
def validate_answer(example, pred, trace=None):
    return example.sentiment.lower() == pred.sentiment.lower()

# Run manual evaluation
print("Running simple evaluation...")
scores = []
for x in examples:
    pred = twt_sentiment(**x.inputs())
    score = validate_answer(x, pred)
    scores.append(score)
    print(f"Tweet: '{x.tweet}' | Predicted: {pred.sentiment} | Correct: {score}")

accuracy = sum(scores) / len(scores)
print(f"\nBaseline Accuracy: {accuracy:.2%}")
print()

print("=== 3. LLM-based Metrics ===")
# Define assessment signature for LLM-based metrics
class Assess(dspy.Signature):
    """Assess the quality of a text along the specified dimension."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

# Example dialog summarization metric
def dialog_metric(gold, pred, trace=None):
    """
    Evaluate dialog summary quality using LLM assessment.
    Returns normalized score (0-1) or binary success for compilation.
    """
    dialogue = gold.dialogue
    gold_summary = gold.summary
    generated_summary = pred.summary
    
    # Define assessment questions
    accurate_question = f"Given this original dialog: '{dialogue}', does the summary accurately represent what was discussed?"
    concise_question = f"Is the generated summary appropriately detailed compared to: '{gold_summary}'?"
    
    # Run assessments
    accurate = dspy.Predict(Assess)(
        assessed_text=generated_summary, 
        assessment_question=accurate_question
    )
    concise = dspy.Predict(Assess)(
        assessed_text=generated_summary, 
        assessment_question=concise_question
    )
    
    # Extract boolean answers
    accurate_score = accurate.assessment_answer
    concise_score = concise.assessment_answer
    
    # Calculate score
    score = (accurate_score + concise_score) if accurate_score else 0
    
    # Return binary success for compilation mode
    if trace is not None:
        return score >= 2
    
    # Return normalized score for evaluation
    return score / 2.0

# Example usage
dialog_sum = dspy.ChainOfThought("dialogue: str -> summary: str")

dialogue_example = dspy.Example(
    dialogue="Alice: Hey, want to grab lunch? Bob: Sure! How about pizza? Alice: Sounds great, see you at noon.",
    summary="Alice and Bob agreed to have pizza for lunch at noon."
).with_inputs('dialogue')

print("Running LLM-based evaluation...")
pred = dialog_sum(**dialogue_example.inputs())
score = dialog_metric(dialogue_example, pred)
print(f"Dialog summary score: {score:.2f}")
print(f"Generated summary: {pred.summary}")

print("\n=== 4. Metrics with Trace (for optimization) ===")
print("During compilation, metrics can access traces to validate intermediate steps.")
print("Example: score = dialog_metric(example, pred, trace=trace)")
print("This allows metrics to inspect Chain of Thought reasoning or other intermediate outputs.")