"""
DSPy Optimizers - Ensemble Methods
Demonstrates how to combine multiple optimized programs
"""
import dspy
from typing import Literal
from collections import Counter

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Define the task signature
class TwtSentiment(dspy.Signature):
    tweet: str = dspy.InputField(desc="Candidate tweet for classification")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

# Create base module
base_twt_sentiment = dspy.Predict(TwtSentiment)

# Create sample data
twitter_train = [
    dspy.Example(tweet="I love this!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="This is terrible", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="It's okay", sentiment="neutral").with_inputs("tweet"),
    dspy.Example(tweet="Amazing experience!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="Worst ever", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="Regular day", sentiment="neutral").with_inputs("tweet"),
]

twitter_test = [
    dspy.Example(tweet="Best day ever!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="I hate this", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="It's fine", sentiment="neutral").with_inputs("tweet"),
]

# Define metric
def validate_answer(example, pred, trace=None):
    return example.sentiment.lower() == pred.sentiment.lower()

print("=== Creating Multiple Optimized Programs ===")

# 1. Create LabeledFewShot program
from dspy.teleprompt import LabeledFewShot
lfs_optimizer = LabeledFewShot(k=2)
lfs_program = lfs_optimizer.compile(base_twt_sentiment, trainset=twitter_train)
print("✓ LabeledFewShot program created")

# 2. Create BootstrapFewShot program
from dspy.teleprompt import BootstrapFewShot
bsfs_optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=1,
    max_labeled_demos=2,
)
bsfs_program = bsfs_optimizer.compile(base_twt_sentiment, trainset=twitter_train)
print("✓ BootstrapFewShot program created")

# 3. Create another variant with different parameters
bsfs2_optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=2,
    max_labeled_demos=1,
)
bsfs2_program = bsfs2_optimizer.compile(base_twt_sentiment, trainset=twitter_train)
print("✓ BootstrapFewShot variant 2 created")

print("\n=== Individual Program Performance ===")

# Test each program individually
programs = {
    "LabeledFewShot": lfs_program,
    "BootstrapFewShot": bsfs_program,
    "BootstrapFewShot v2": bsfs2_program
}

for name, program in programs.items():
    scores = []
    for x in twitter_test:
        pred = program(**x.inputs())
        score = validate_answer(x, pred)
        scores.append(score)
    accuracy = sum(scores) / len(scores)
    print(f"{name}: {accuracy:.2%}")

print("\n=== Manual Ensemble (Majority Voting) ===")

# Create manual ensemble with majority voting
def manual_ensemble(programs, tweet):
    """Run all programs and return majority vote."""
    predictions = []
    for program in programs:
        pred = program(tweet=tweet)
        predictions.append(pred.sentiment)
    
    # Count votes
    vote_counts = Counter(predictions)
    # Return most common prediction
    return vote_counts.most_common(1)[0][0]

# Test manual ensemble
ensemble_scores = []
for x in twitter_test:
    ensemble_pred = manual_ensemble(programs.values(), x.tweet)
    # Create a prediction object for compatibility
    pred_obj = dspy.Prediction(sentiment=ensemble_pred)
    score = validate_answer(x, pred_obj)
    ensemble_scores.append(score)
    print(f"Tweet: '{x.tweet}' | Ensemble: {ensemble_pred} | Correct: {score}")

ensemble_accuracy = sum(ensemble_scores) / len(ensemble_scores)
print(f"\nManual Ensemble Accuracy: {ensemble_accuracy:.2%}")

print("\n=== DSPy Ensemble Optimizer ===")

try:
    from dspy.teleprompt import Ensemble
    
    # Create ensemble using DSPy's built-in optimizer
    ensemble_optimizer = Ensemble(reduce_fn=dspy.majority)
    
    # Compile ensemble with multiple programs
    ensemble_program = ensemble_optimizer.compile(programs=list(programs.values()))
    
    # Test ensemble
    dspy_ensemble_scores = []
    for x in twitter_test:
        pred = ensemble_program(**x.inputs())
        score = validate_answer(x, pred)
        dspy_ensemble_scores.append(score)
    
    dspy_ensemble_accuracy = sum(dspy_ensemble_scores) / len(dspy_ensemble_scores)
    print(f"DSPy Ensemble Accuracy: {dspy_ensemble_accuracy:.2%}")
    
except Exception as e:
    print(f"Note: DSPy Ensemble may require specific setup. Error: {e}")

print("\n=== Advanced Ensemble Strategies ===")
print("1. Weighted Voting: Assign weights based on individual program performance")
print("2. Confidence-based: Use programs' confidence scores to weight votes")
print("3. Conditional Ensemble: Use different programs for different input types")
print("4. Stacking: Use another model to learn how to combine predictions")

# Example: Weighted ensemble based on individual accuracies
print("\n=== Weighted Ensemble Example ===")

# Calculate weights based on individual performance
weights = {}
for name, program in programs.items():
    scores = []
    for x in twitter_train:  # Use training data to calculate weights
        pred = program(**x.inputs())
        score = validate_answer(x, pred)
        scores.append(score)
    weights[name] = sum(scores) / len(scores)

print("Program weights:", {k: f"{v:.2f}" for k, v in weights.items()})

def weighted_ensemble(programs, weights, tweet):
    """Weighted voting ensemble."""
    vote_weights = {}
    
    for (name, program), weight in zip(programs.items(), weights.values()):
        pred = program(tweet=tweet)
        sentiment = pred.sentiment
        
        if sentiment not in vote_weights:
            vote_weights[sentiment] = 0
        vote_weights[sentiment] += weight
    
    # Return sentiment with highest weight
    return max(vote_weights.items(), key=lambda x: x[1])[0]

# Test weighted ensemble
weighted_scores = []
for x in twitter_test:
    weighted_pred = weighted_ensemble(programs, weights, x.tweet)
    pred_obj = dspy.Prediction(sentiment=weighted_pred)
    score = validate_answer(x, pred_obj)
    weighted_scores.append(score)

weighted_accuracy = sum(weighted_scores) / len(weighted_scores)
print(f"Weighted Ensemble Accuracy: {weighted_accuracy:.2%}")