"""
DSPy Sentiment and Emotion Classification Examples
Based on recent DSPy patterns from 2024-2025
"""

import dspy
from typing import Literal
from pydantic import BaseModel, Field


# Basic sentiment classification signature
class SentimentSignature(dspy.Signature):
    """Classify sentiment as positive, negative, or neutral."""
    text: str = dspy.InputField(description="Text to analyze for sentiment")
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField(
        description="The sentiment of the text"
    )


# Emotion classification signature
class EmotionSignature(dspy.Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""
    sentence: str = dspy.InputField(description="Sentence to analyze for emotion")
    emotion: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField(
        description="The primary emotion expressed"
    )


# Advanced sentiment with confidence
class SentimentWithConfidence(dspy.Signature):
    """Classify sentiment with confidence score."""
    text: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField(description="Confidence score between 0 and 1")


# Aspect-based sentiment analysis
class AspectSentiment(BaseModel):
    aspect: str = Field(description="The aspect being discussed")
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="Sentiment towards the aspect")
    reasoning: str = Field(description="Brief explanation for the sentiment")


class AspectBasedSentimentSignature(dspy.Signature):
    """Extract aspects and their associated sentiments from text."""
    text: str = dspy.InputField(description="Text to analyze")
    aspects: list[AspectSentiment] = dspy.OutputField(
        description="List of aspects with their sentiments"
    )


# Sentiment classification module
class SentimentClassifier(dspy.Module):
    """A module for sentiment classification using Chain of Thought."""
    
    def __init__(self):
        self.prog = dspy.ChainOfThought(SentimentSignature)
    
    def forward(self, text: str):
        return self.prog(text=text)


# Multi-step sentiment analysis
class DetailedSentimentAnalyzer(dspy.Module):
    """Detailed sentiment analysis with reasoning."""
    
    def __init__(self):
        self.sentiment_classifier = dspy.ChainOfThought(SentimentWithConfidence)
        self.emotion_classifier = dspy.Predict(EmotionSignature)
    
    def forward(self, text: str):
        # Get sentiment with confidence
        sentiment_result = self.sentiment_classifier(text=text)
        
        # Get emotion
        emotion_result = self.emotion_classifier(sentence=text)
        
        return {
            'sentiment': sentiment_result.sentiment,
            'confidence': sentiment_result.confidence,
            'emotion': emotion_result.emotion,
            'reasoning': sentiment_result.reasoning if hasattr(sentiment_result, 'reasoning') else None
        }


# Example usage and evaluation metric
def evaluate_sentiment(example, pred, trace=None) -> bool:
    """Evaluation metric for sentiment classification."""
    return pred.sentiment in ["positive", "negative", "neutral"]


def evaluate_emotion(example, pred, trace=None) -> bool:
    """Evaluation metric for emotion classification."""
    valid_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return pred.emotion in valid_emotions


if __name__ == "__main__":
    # Configure DSPy with your preferred LM
    # dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    
    # Example usage
    examples = [
        "I absolutely love this new feature! It's amazing!",
        "This is terrible, I'm very disappointed.",
        "The product works as expected.",
        "I started feeling a little vulnerable when the giant spotlight started blinding me",
    ]
    
    # Initialize classifier
    classifier = SentimentClassifier()
    
    # Example predictions
    for text in examples:
        print(f"\nText: {text}")
        # result = classifier(text=text)
        # print(f"Sentiment: {result.sentiment}")