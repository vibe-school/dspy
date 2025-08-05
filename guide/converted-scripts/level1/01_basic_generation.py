"""
Level 1: Basic Generation with DSPy
This example shows the simplest form of using an LLM through DSPy
"""
import openai
import time

# Initialize OpenAI client
client = openai.Client()

def generate(prompt):
    """Basic generation function using OpenAI"""
    start_time = time.time()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Updated to use a more accessible model
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Response time: {time.time() - start_time:.2f} seconds")
    return response.choices[0].message.content

# Example 1: Simple question
print("=== Example 1: Simple Question ===")
response = generate("Capital of India?")
print(response)
print()

# Example 2: Basic joke generation
print("=== Example 2: Basic Joke ===")
response = generate("Write a joke about AI")
print(response)