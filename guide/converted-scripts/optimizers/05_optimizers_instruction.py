"""
DSPy Optimizers - Instruction Optimization
Demonstrates COPRO and MIPROv2 for automatic instruction optimization
"""
import dspy
from typing import Literal

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Define the task signature
class TwtSentiment(dspy.Signature):
    tweet: str = dspy.InputField(desc="Candidate tweet for classification")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

# Create base module
base_twt_sentiment = dspy.Predict(TwtSentiment)

# Create sample training data
twitter_train = [
    dspy.Example(tweet="I absolutely love this new update!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="This is the worst experience ever", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="Just another day at work", sentiment="neutral").with_inputs("tweet"),
    dspy.Example(tweet="Amazing customer service!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="I'm so frustrated with this", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="The meeting was productive", sentiment="neutral").with_inputs("tweet"),
    dspy.Example(tweet="Best purchase I've made all year!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="Complete waste of money", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="The weather is cloudy today", sentiment="neutral").with_inputs("tweet"),
    dspy.Example(tweet="I'm thrilled with the results!", sentiment="positive").with_inputs("tweet"),
]

twitter_test = [
    dspy.Example(tweet="This made my day!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="I regret buying this", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="It's Tuesday", sentiment="neutral").with_inputs("tweet"),
    dspy.Example(tweet="Fantastic news!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="This is unacceptable", sentiment="negative").with_inputs("tweet"),
]

# Define metric
def validate_answer(example, pred, trace=None):
    return example.sentiment.lower() == pred.sentiment.lower()

# Example tweet for testing
example_tweet = "Hi! Waking up, and not lazy at all. You would be proud of me!"

print("=== 1. Baseline Performance ===")
baseline_scores = []
for x in twitter_test:
    pred = base_twt_sentiment(**x.inputs())
    score = validate_answer(x, pred)
    baseline_scores.append(score)

base_accuracy = sum(baseline_scores) / len(baseline_scores)
print(f"Baseline Accuracy: {base_accuracy:.2%}")
print(f"Baseline prediction: {base_twt_sentiment(tweet=example_tweet).sentiment}")
print()

print("=== 2. COPRO (Coordinate Prompt Optimization) ===")
print("COPRO iteratively improves instructions using coordinate ascent.")
print("Note: This requires a separate model for prompt generation (e.g., GPT-4)")

try:
    from dspy.teleprompt import COPRO
    
    # Note: In practice, you'd use a more powerful model for prompt generation
    copro_optimizer = COPRO(
        metric=validate_answer,
        prompt_model=dspy.LM('openai/gpt-4o-mini'),  # In practice, use gpt-4
        breadth=3,  # New prompts per iteration
        depth=2,    # Number of improvement rounds
        init_temperature=1.2
    )
    
    print("Compiling with COPRO (this may take a moment)...")
    copro_twt_sentiment = copro_optimizer.compile(
        base_twt_sentiment, 
        trainset=twitter_train,
        eval_kwargs={'num_threads': 1, 'display_progress': False}
    )
    
    # Evaluate
    copro_scores = []
    for x in twitter_test:
        pred = copro_twt_sentiment(**x.inputs())
        score = validate_answer(x, pred)
        copro_scores.append(score)
    
    copro_accuracy = sum(copro_scores) / len(copro_scores)
    print(f"COPRO Accuracy: {copro_accuracy:.2%}")
    print(f"COPRO prediction: {copro_twt_sentiment(tweet=example_tweet).sentiment}")
    
except Exception as e:
    print(f"COPRO optimization note: {e}")
    print("COPRO requires more computational resources and may need adjustment for small datasets")
print()

print("=== 3. MIPROv2 (Multi-prompt Instruction Proposal Optimizer) ===")
print("MIPROv2 uses Bayesian Optimization to search for optimal instructions and demonstrations.")

try:
    from dspy.teleprompt import MIPROv2
    
    mipro_optimizer = MIPROv2(
        metric=validate_answer,
        prompt_model=dspy.LM('openai/gpt-4o-mini'),  # In practice, use gpt-4
        num_candidates=3,  # Instructions to try
    )
    
    print("Compiling with MIPROv2 (this may take a moment)...")
    mipro_twt_sentiment = mipro_optimizer.compile(
        base_twt_sentiment,
        trainset=twitter_train,
        valset=twitter_test  # MIPROv2 uses validation set
    )
    
    # Evaluate
    mipro_scores = []
    for x in twitter_test:
        pred = mipro_twt_sentiment(**x.inputs())
        score = validate_answer(x, pred)
        mipro_scores.append(score)
    
    mipro_accuracy = sum(mipro_scores) / len(mipro_scores)
    print(f"MIPROv2 Accuracy: {mipro_accuracy:.2%}")
    print(f"MIPROv2 prediction: {mipro_twt_sentiment(tweet=example_tweet).sentiment}")
    
except Exception as e:
    print(f"MIPROv2 optimization note: {e}")
    print("MIPROv2 requires significant computational resources and larger datasets")

print("\n=== Key Differences ===")
print("1. COPRO: Uses coordinate ascent to iteratively improve instructions")
print("2. MIPROv2: Uses Bayesian Optimization for more sophisticated search")
print("3. Both generate task-specific instructions rather than relying on examples")
print("4. Best used with larger datasets (50+ examples) and more powerful prompt models")