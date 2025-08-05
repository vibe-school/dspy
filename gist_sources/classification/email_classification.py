"""
DSPy Email Classification Pipeline
Based on GitHub issue #164 discussion about practical classification examples
"""

import dspy
from typing import Literal, Optional
from pydantic import BaseModel, Field


# Email summarization signature
class EmailSummarizer(dspy.Signature):
    """Summarize long email threads maintaining key information."""
    email_text: str = dspy.InputField(
        description="Full email text including thread history"
    )
    summary: str = dspy.OutputField(
        description="Concise summary preserving key details and intent"
    )


# Email classification signature
class EmailClassifier(dspy.Signature):
    """Classify email into predefined categories."""
    email_summary: str = dspy.InputField(
        description="Summarized email content"
    )
    category: Literal[
        'refund_request',
        'technical_support', 
        'billing_inquiry',
        'feature_request',
        'complaint',
        'general_inquiry'
    ] = dspy.OutputField(
        description="Primary category of the email"
    )
    confidence: float = dspy.OutputField(
        description="Confidence score between 0 and 1"
    )


# Refund reason extraction
class RefundDetails(BaseModel):
    reason: str = Field(description="Primary reason for refund request")
    product: Optional[str] = Field(description="Product or service name if mentioned")
    order_id: Optional[str] = Field(description="Order ID if provided")
    amount: Optional[str] = Field(description="Refund amount if mentioned")


class RefundReasonExtractor(dspy.Signature):
    """Extract refund details from email."""
    email_text: str = dspy.InputField()
    category: str = dspy.InputField(description="Email category")
    refund_details: RefundDetails = dspy.OutputField(
        description="Extracted refund information"
    )


# Technical issue extraction
class TechnicalIssueDetails(BaseModel):
    issue_description: str = Field(description="Description of the technical problem")
    error_messages: list[str] = Field(description="Any error messages mentioned")
    steps_to_reproduce: Optional[str] = Field(description="Steps to reproduce if provided")
    urgency: Literal['low', 'medium', 'high'] = Field(description="Urgency level")


class TechnicalIssueExtractor(dspy.Signature):
    """Extract technical issue details from support emails."""
    email_text: str = dspy.InputField()
    technical_details: TechnicalIssueDetails = dspy.OutputField()


# Complete email processing pipeline
class EmailProcessingPipeline(dspy.Module):
    """
    Multi-step email processing pipeline:
    1. Summarize long emails
    2. Classify into categories
    3. Extract category-specific information
    """
    
    def __init__(self):
        self.summarizer = dspy.ChainOfThought(EmailSummarizer)
        self.classifier = dspy.Predict(EmailClassifier)
        self.refund_extractor = dspy.Predict(RefundReasonExtractor)
        self.tech_extractor = dspy.Predict(TechnicalIssueExtractor)
    
    def forward(self, email_text: str):
        # Step 1: Summarize if email is long
        if len(email_text) > 1000:
            summary_result = self.summarizer(email_text=email_text)
            summary = summary_result.summary
        else:
            summary = email_text
        
        # Step 2: Classify email
        classification = self.classifier(email_summary=summary)
        
        # Step 3: Extract category-specific details
        extracted_info = None
        
        if classification.category == 'refund_request':
            extracted_info = self.refund_extractor(
                email_text=email_text,
                category=classification.category
            ).refund_details
        elif classification.category == 'technical_support':
            extracted_info = self.tech_extractor(
                email_text=email_text
            ).technical_details
        
        return {
            'summary': summary,
            'category': classification.category,
            'confidence': classification.confidence,
            'extracted_info': extracted_info,
            'reasoning': getattr(summary_result, 'reasoning', None)
        }


# Evaluation metrics
def evaluate_classification(example, pred, trace=None) -> bool:
    """Check if classification is valid."""
    valid_categories = [
        'refund_request', 'technical_support', 'billing_inquiry',
        'feature_request', 'complaint', 'general_inquiry'
    ]
    return pred.category in valid_categories


def evaluate_extraction(example, pred, trace=None) -> bool:
    """Check if key information was extracted."""
    if hasattr(pred, 'refund_details') and pred.refund_details:
        return bool(pred.refund_details.reason)
    return True


# Example email templates
EXAMPLE_EMAILS = {
    'refund': """
    Subject: Request for Refund - Order #12345
    
    Hi Support Team,
    
    I purchased your Premium subscription last week (Order #12345) for $99.99, 
    but I'm experiencing constant crashes and the features advertised don't work 
    as expected. I've tried reinstalling multiple times without success.
    
    I would like to request a full refund as the product doesn't meet my needs.
    
    Thanks,
    John Doe
    """,
    
    'technical': """
    Subject: Urgent: Application Crashing on Startup
    
    Hello,
    
    I'm unable to use your application since yesterday. Every time I try to 
    launch it, I get an error message saying "Failed to initialize module XYZ" 
    and then the app crashes immediately.
    
    Steps I've tried:
    1. Restart computer
    2. Reinstall application
    3. Run as administrator
    
    Nothing works. This is affecting my work badly. Please help ASAP!
    
    Error code: ERR_MODULE_INIT_FAILED
    
    Best regards,
    Sarah
    """
}


if __name__ == "__main__":
    # Example usage
    # dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    
    # Initialize pipeline
    pipeline = EmailProcessingPipeline()
    
    # Process example emails
    for email_type, email_text in EXAMPLE_EMAILS.items():
        print(f"\n{'='*50}")
        print(f"Processing {email_type} email...")
        # result = pipeline(email_text=email_text)
        # print(f"Category: {result['category']} (Confidence: {result['confidence']})")
        # if result['extracted_info']:
        #     print(f"Extracted Info: {result['extracted_info']}")