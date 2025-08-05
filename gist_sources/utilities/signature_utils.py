"""
DSPy Signature Utilities
Helper functions for creating and working with DSPy signatures
Based on patterns from seanchatmangpt's gist
"""

import dspy
from pydantic import BaseModel, create_model
from typing import Type, Dict, Any, get_type_hints, Union, List
import inspect


def pydantic_to_dspy_signature(
    pydantic_model: Type[BaseModel],
    signature_name: str = None,
    docstring: str = None
) -> Type[dspy.Signature]:
    """
    Convert a Pydantic model to a DSPy Signature.
    
    Args:
        pydantic_model: The Pydantic model class
        signature_name: Optional name for the signature (defaults to model name + "Signature")
        docstring: Optional docstring for the signature
    
    Returns:
        A DSPy Signature class
    """
    # Generate signature name
    if signature_name is None:
        signature_name = f"{pydantic_model.__name__}Signature"
    
    # Use model's docstring if none provided
    if docstring is None:
        docstring = pydantic_model.__doc__ or f"Signature for {pydantic_model.__name__}"
    
    # Create field definitions
    fields = {}
    for field_name, field_info in pydantic_model.model_fields.items():
        description = field_info.description or f"{field_name} field"
        
        # Determine if it's input or output based on field name convention
        # (you can customize this logic)
        if field_name.startswith('output_') or field_name in ['result', 'answer', 'prediction']:
            fields[field_name] = dspy.OutputField(description=description)
        else:
            fields[field_name] = dspy.InputField(description=description)
    
    # Create the signature class
    signature_class = type(
        signature_name,
        (dspy.Signature,),
        {
            '__doc__': docstring,
            '__annotations__': get_type_hints(pydantic_model),
            **fields
        }
    )
    
    return signature_class


def create_classification_signature(
    categories: List[str],
    input_field_name: str = "text",
    output_field_name: str = "category",
    include_confidence: bool = False,
    include_reasoning: bool = False
) -> Type[dspy.Signature]:
    """
    Create a classification signature with specified categories.
    
    Args:
        categories: List of classification categories
        input_field_name: Name for the input field
        output_field_name: Name for the output field
        include_confidence: Whether to include confidence score
        include_reasoning: Whether to include reasoning
    
    Returns:
        A DSPy Signature class for classification
    """
    from typing import Literal
    
    # Create fields dictionary
    fields = {
        input_field_name: dspy.InputField(description=f"{input_field_name} to classify"),
        output_field_name: dspy.OutputField(description=f"One of: {', '.join(categories)}")
    }
    
    # Create annotations
    annotations = {
        input_field_name: str,
        output_field_name: Literal[tuple(categories)]
    }
    
    if include_confidence:
        fields['confidence'] = dspy.OutputField(description="Confidence score between 0 and 1")
        annotations['confidence'] = float
    
    if include_reasoning:
        fields['reasoning'] = dspy.OutputField(description="Explanation for the classification")
        annotations['reasoning'] = str
    
    # Create the signature class
    signature_class = type(
        'CustomClassificationSignature',
        (dspy.Signature,),
        {
            '__doc__': f"Classify {input_field_name} into one of: {', '.join(categories)}",
            '__annotations__': annotations,
            **fields
        }
    )
    
    return signature_class


def signature_to_pydantic(signature_class: Type[dspy.Signature]) -> Type[BaseModel]:
    """
    Convert a DSPy Signature back to a Pydantic model.
    
    Args:
        signature_class: The DSPy Signature class
    
    Returns:
        A Pydantic BaseModel class
    """
    # Get annotations
    annotations = get_type_hints(signature_class)
    
    # Create field definitions for Pydantic
    field_definitions = {}
    for field_name, field_type in annotations.items():
        # Get the field object
        field_obj = getattr(signature_class, field_name, None)
        if hasattr(field_obj, 'description'):
            field_definitions[field_name] = (field_type, field_obj.description)
        else:
            field_definitions[field_name] = (field_type, ...)
    
    # Create Pydantic model
    model_name = signature_class.__name__.replace('Signature', 'Model')
    pydantic_model = create_model(
        model_name,
        __doc__=signature_class.__doc__,
        **field_definitions
    )
    
    return pydantic_model


def combine_signatures(
    *signatures: Type[dspy.Signature],
    name: str = "CombinedSignature",
    docstring: str = None
) -> Type[dspy.Signature]:
    """
    Combine multiple DSPy signatures into one.
    
    Args:
        *signatures: DSPy Signature classes to combine
        name: Name for the combined signature
        docstring: Optional docstring
    
    Returns:
        A combined DSPy Signature class
    """
    combined_fields = {}
    combined_annotations = {}
    
    for sig in signatures:
        # Get annotations
        annotations = get_type_hints(sig)
        combined_annotations.update(annotations)
        
        # Get fields
        for field_name in annotations:
            if hasattr(sig, field_name):
                field_obj = getattr(sig, field_name)
                combined_fields[field_name] = field_obj
    
    # Create combined signature
    if docstring is None:
        docstring = f"Combined signature from: {', '.join(s.__name__ for s in signatures)}"
    
    combined_signature = type(
        name,
        (dspy.Signature,),
        {
            '__doc__': docstring,
            '__annotations__': combined_annotations,
            **combined_fields
        }
    )
    
    return combined_signature


def inspect_signature(signature_class: Type[dspy.Signature]) -> Dict[str, Any]:
    """
    Inspect a DSPy signature and return information about its fields.
    
    Args:
        signature_class: The DSPy Signature class to inspect
    
    Returns:
        Dictionary with signature information
    """
    info = {
        'name': signature_class.__name__,
        'docstring': signature_class.__doc__,
        'fields': {}
    }
    
    annotations = get_type_hints(signature_class)
    
    for field_name, field_type in annotations.items():
        field_obj = getattr(signature_class, field_name, None)
        
        field_info = {
            'type': str(field_type),
            'is_input': isinstance(field_obj, dspy.InputField),
            'is_output': isinstance(field_obj, dspy.OutputField),
            'description': getattr(field_obj, 'description', None)
        }
        
        info['fields'][field_name] = field_info
    
    return info


# Example usage
if __name__ == "__main__":
    # Example 1: Create a classification signature
    sentiment_sig = create_classification_signature(
        categories=['positive', 'negative', 'neutral'],
        input_field_name='review',
        output_field_name='sentiment',
        include_confidence=True,
        include_reasoning=True
    )
    
    print("Created Classification Signature:")
    print(inspect_signature(sentiment_sig))
    
    # Example 2: Convert Pydantic to DSPy
    class ProductReview(BaseModel):
        """Analyze product reviews."""
        review_text: str
        product_name: str
        output_sentiment: str
        output_score: float
    
    review_sig = pydantic_to_dspy_signature(ProductReview)
    print("\nPydantic to DSPy Signature:")
    print(inspect_signature(review_sig))
    
    # Example 3: Combine signatures
    class FeatureExtraction(dspy.Signature):
        """Extract product features."""
        text: str = dspy.InputField()
        features: List[str] = dspy.OutputField()
    
    combined = combine_signatures(sentiment_sig, FeatureExtraction)
    print("\nCombined Signature:")
    print(inspect_signature(combined))