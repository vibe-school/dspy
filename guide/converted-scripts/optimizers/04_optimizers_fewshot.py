"""
DSPy Optimizers - Automatic Few-Shot Learning
Demonstrates LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch, and KNNFewShot
"""
import dspy
from typing import Literal
import numpy as np

# Configure LLM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Define the task signature
class TwtSentiment(dspy.Signature):
    tweet: str = dspy.InputField(desc="Candidate tweet for classification")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

# Create base module
base_twt_sentiment = dspy.Predict(TwtSentiment)

# Create sample training and test data
twitter_train = [
    dspy.Example(tweet="I love this new feature!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="This app is terrible", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="The weather is nice today", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="I hate waiting in lines", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="Just had lunch", sentiment="neutral").with_inputs("tweet"),
    dspy.Example(tweet="Amazing experience at the concert!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="Worst customer service ever", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="The book was okay", sentiment="neutral").with_inputs("tweet"),
]

twitter_test = [
    dspy.Example(tweet="Best day ever!", sentiment="positive").with_inputs("tweet"),
    dspy.Example(tweet="This is disappointing", sentiment="negative").with_inputs("tweet"),
    dspy.Example(tweet="Regular Tuesday", sentiment="neutral").with_inputs("tweet"),
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
print(f"Baseline prediction for '{example_tweet}': {base_twt_sentiment(tweet=example_tweet).sentiment}")
print()

print("=== 2. LabeledFewShot Optimizer ===")
from dspy.teleprompt import LabeledFewShot

lfs_optimizer = LabeledFewShot(k=3)  # Use 3 examples in prompts
lfs_twt_sentiment = lfs_optimizer.compile(base_twt_sentiment, trainset=twitter_train)

# Evaluate
lfs_scores = []
for x in twitter_test:
    pred = lfs_twt_sentiment(**x.inputs())
    score = validate_answer(x, pred)
    lfs_scores.append(score)

lfs_accuracy = sum(lfs_scores) / len(lfs_scores)
print(f"LabeledFewShot Accuracy: {lfs_accuracy:.2%}")
print(f"LFS prediction for '{example_tweet}': {lfs_twt_sentiment(tweet=example_tweet).sentiment}")
print()

print("=== 3. BootstrapFewShot Optimizer ===")
from dspy.teleprompt import BootstrapFewShot

bsfs_optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=2,
    max_labeled_demos=3,
)

bsfs_twt_sentiment = bsfs_optimizer.compile(base_twt_sentiment, trainset=twitter_train)

# Evaluate
bsfs_scores = []
for x in twitter_test:
    pred = bsfs_twt_sentiment(**x.inputs())
    score = validate_answer(x, pred)
    bsfs_scores.append(score)

bsfs_accuracy = sum(bsfs_scores) / len(bsfs_scores)
print(f"BootstrapFewShot Accuracy: {bsfs_accuracy:.2%}")
print(f"BSFS prediction for '{example_tweet}': {bsfs_twt_sentiment(tweet=example_tweet).sentiment}")
print()

print("=== 4. BootstrapFewShotWithRandomSearch ===")
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

bsfswrs_optimizer = BootstrapFewShotWithRandomSearch(
    metric=validate_answer,
    num_candidate_programs=3,  # Try 3 random programs
    max_bootstrapped_demos=2,
    max_labeled_demos=3
)

bsfswrs_twt_sentiment = bsfswrs_optimizer.compile(base_twt_sentiment, trainset=twitter_train)

# Evaluate
bsfswrs_scores = []
for x in twitter_test:
    pred = bsfswrs_twt_sentiment(**x.inputs())
    score = validate_answer(x, pred)
    bsfswrs_scores.append(score)

bsfswrs_accuracy = sum(bsfswrs_scores) / len(bsfswrs_scores)
print(f"BootstrapFewShotWithRandomSearch Accuracy: {bsfswrs_accuracy:.2%}")
print(f"BSFSWRS prediction for '{example_tweet}': {bsfswrs_twt_sentiment(tweet=example_tweet).sentiment}")
print()

print("=== 5. KNNFewShot (with mock embeddings) ===")
# Mock embedding function for demonstration
def mock_embeddings(texts):
    """Simple mock embedding function for demonstration."""
    if isinstance(texts, str):
        texts = [texts]
    
    # Create fake embeddings based on sentiment keywords
    embeddings = []
    for text in texts:
        text_lower = text.lower()
        # Simple heuristic embeddings
        if any(word in text_lower for word in ['love', 'amazing', 'best', 'great']):
            emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif any(word in text_lower for word in ['hate', 'terrible', 'worst', 'bad']):
            emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            emb = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    return embeddings[0] if len(embeddings) == 1 else embeddings

try:
    from dspy.teleprompt import KNNFewShot
    
    knn_optimizer = KNNFewShot(
        k=2,
        trainset=twitter_train,
        vectorizer=mock_embeddings
    )
    
    knn_twt_sentiment = knn_optimizer.compile(base_twt_sentiment, trainset=twitter_train)
    
    # Evaluate
    knn_scores = []
    for x in twitter_test:
        pred = knn_twt_sentiment(**x.inputs())
        score = validate_answer(x, pred)
        knn_scores.append(score)
    
    knn_accuracy = sum(knn_scores) / len(knn_scores)
    print(f"KNNFewShot Accuracy: {knn_accuracy:.2%}")
    print(f"KNN prediction for '{example_tweet}': {knn_twt_sentiment(tweet=example_tweet).sentiment}")
except Exception as e:
    print(f"Note: KNNFewShot requires proper embedding setup. Error: {e}")

print("\n=== Summary ===")
print(f"Baseline: {base_accuracy:.2%}")
print(f"LabeledFewShot: {lfs_accuracy:.2%}")
print(f"BootstrapFewShot: {bsfs_accuracy:.2%}")
print(f"BootstrapFewShotWithRandomSearch: {bsfswrs_accuracy:.2%}")