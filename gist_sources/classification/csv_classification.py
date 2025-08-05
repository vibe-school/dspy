"""
DSPy CSV-based Classification with Optimization
Based on patterns from cannin's gist and DSPy examples
"""

import dspy
import pandas as pd
from typing import List, Tuple
from pathlib import Path


class TextClassificationSignature(dspy.Signature):
    """Classify text into predefined categories."""
    text: str = dspy.InputField(description="Text to classify")
    category: str = dspy.OutputField(description="Classification category")


class ClassificationWithReasoning(dspy.Signature):
    """Classify text with reasoning."""
    text: str = dspy.InputField()
    category: str = dspy.OutputField()
    reasoning: str = dspy.OutputField(description="Explanation for the classification")


class CSVClassifier(dspy.Module):
    """Basic CSV-based text classifier."""
    
    def __init__(self, use_reasoning: bool = False):
        if use_reasoning:
            self.classifier = dspy.ChainOfThought(ClassificationWithReasoning)
        else:
            self.classifier = dspy.Predict(TextClassificationSignature)
        self.use_reasoning = use_reasoning
    
    def forward(self, text: str):
        return self.classifier(text=text)


class DataLoader:
    """Load and prepare classification data from CSV."""
    
    @staticmethod
    def load_csv(file_path: str, text_column: str = 'text', 
                 label_column: str = 'label', sample_size: int = None) -> List[dspy.Example]:
        """
        Load classification data from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            sample_size: Optional sample size for testing
        
        Returns:
            List of dspy.Example objects
        """
        df = pd.read_csv(file_path)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        examples = []
        for _, row in df.iterrows():
            example = dspy.Example(
                text=row[text_column],
                category=row[label_column]
            ).with_inputs('text')
            examples.append(example)
        
        return examples
    
    @staticmethod
    def split_data(examples: List[dspy.Example], 
                   train_ratio: float = 0.8) -> Tuple[List[dspy.Example], List[dspy.Example]]:
        """Split data into train and test sets."""
        split_idx = int(len(examples) * train_ratio)
        return examples[:split_idx], examples[split_idx:]


# Evaluation metric
def classification_metric(example: dspy.Example, pred, trace=None) -> float:
    """
    Evaluate classification accuracy.
    Returns 1 if correct, 0 if incorrect.
    """
    return float(example.category.lower() == pred.category.lower())


# Advanced metric with partial credit
def classification_metric_advanced(example: dspy.Example, pred, trace=None) -> float:
    """
    Advanced metric that gives partial credit for related categories.
    """
    exact_match = example.category.lower() == pred.category.lower()
    if exact_match:
        return 1.0
    
    # Define related categories for partial credit
    related_categories = {
        'positive': ['joy', 'love', 'happy'],
        'negative': ['sadness', 'anger', 'fear'],
        'neutral': ['surprise', 'neutral']
    }
    
    # Check if prediction is in related category
    for main_cat, related in related_categories.items():
        if example.category.lower() in related and pred.category.lower() in related:
            return 0.5
    
    return 0.0


class OptimizedCSVClassifier:
    """
    CSV Classifier with DSPy optimization capabilities.
    """
    
    def __init__(self, trainset: List[dspy.Example], metric=classification_metric):
        self.trainset = trainset
        self.metric = metric
        self.base_classifier = CSVClassifier(use_reasoning=True)
        self.optimized_classifier = None
    
    def optimize_with_bootstrap(self, max_bootstrapped_demos: int = 4):
        """Optimize using BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot
        
        teleprompter = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_bootstrapped_demos
        )
        
        self.optimized_classifier = teleprompter.compile(
            self.base_classifier,
            trainset=self.trainset
        )
        
        return self.optimized_classifier
    
    def optimize_with_mipro(self, num_candidates: int = 10, init_temperature: float = 1.0):
        """Optimize using MIPRO (if available)."""
        try:
            from dspy.teleprompt import MIPRO
            
            teleprompter = MIPRO(
                metric=self.metric,
                num_candidates=num_candidates,
                init_temperature=init_temperature
            )
            
            self.optimized_classifier = teleprompter.compile(
                self.base_classifier,
                trainset=self.trainset,
                num_trials=20,
                max_bootstrapped_demos=3,
                max_labeled_demos=5
            )
            
            return self.optimized_classifier
        except ImportError:
            print("MIPRO not available, falling back to BootstrapFewShot")
            return self.optimize_with_bootstrap()
    
    def evaluate(self, testset: List[dspy.Example], classifier=None):
        """Evaluate classifier on test set."""
        if classifier is None:
            classifier = self.optimized_classifier or self.base_classifier
        
        from dspy.evaluate import Evaluate
        
        evaluator = Evaluate(
            devset=testset,
            metric=self.metric,
            num_threads=1,
            display_progress=True
        )
        
        return evaluator(classifier)


# Example usage with sample data creation
def create_sample_csv(file_path: str = "sample_classification_data.csv"):
    """Create a sample CSV file for testing."""
    import random
    
    # Sample data
    texts_and_labels = [
        # Positive examples
        ("This product is amazing! Best purchase ever!", "positive"),
        ("I love how easy it is to use", "positive"),
        ("Exceeded my expectations completely", "positive"),
        ("Fantastic quality and great price", "positive"),
        
        # Negative examples
        ("Terrible experience, would not recommend", "negative"),
        ("Complete waste of money", "negative"),
        ("Very disappointed with the quality", "negative"),
        ("Worst customer service ever", "negative"),
        
        # Neutral examples
        ("The product works as described", "neutral"),
        ("It's okay, nothing special", "neutral"),
        ("Average product for the price", "neutral"),
        ("Meets basic requirements", "neutral"),
    ]
    
    # Duplicate and shuffle for larger dataset
    all_data = texts_and_labels * 5
    random.shuffle(all_data)
    
    df = pd.DataFrame(all_data, columns=['text', 'label'])
    df.to_csv(file_path, index=False)
    print(f"Created sample CSV with {len(df)} rows at {file_path}")


if __name__ == "__main__":
    # Example workflow
    print("DSPy CSV Classification Example")
    print("=" * 50)
    
    # Create sample data
    sample_file = "sample_classification_data.csv"
    # create_sample_csv(sample_file)
    
    # Load and prepare data
    # examples = DataLoader.load_csv(sample_file, text_column='text', label_column='label')
    # trainset, testset = DataLoader.split_data(examples)
    
    # print(f"Loaded {len(trainset)} training examples and {len(testset)} test examples")
    
    # Initialize and optimize classifier
    # optimizer = OptimizedCSVClassifier(trainset)
    
    # Optimize with BootstrapFewShot
    # optimized_model = optimizer.optimize_with_bootstrap()
    
    # Evaluate
    # results = optimizer.evaluate(testset)
    # print(f"Test Accuracy: {results}")
    
    # Example prediction
    # test_text = "This is the worst product I've ever used"
    # prediction = optimized_model(text=test_text)
    # print(f"\nPrediction for '{test_text}':")
    # print(f"Category: {prediction.category}")
    # if hasattr(prediction, 'reasoning'):
    #     print(f"Reasoning: {prediction.reasoning}")