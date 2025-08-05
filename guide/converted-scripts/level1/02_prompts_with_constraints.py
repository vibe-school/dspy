"""
Level 1: Prompts with Constraints
This example shows how to add constraints and context to prompts
"""
import openai
import time

client = openai.Client()

def generate(prompt):
    """Basic generation function using OpenAI"""
    start_time = time.time()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Response time: {time.time() - start_time:.2f} seconds")
    return response.choices[0].message.content

# Example 1: Prompt with constraint
print("=== Example 1: Prompt with Constraint ===")
prompt = "Write a joke about AI that has to do with them turning rogue"
response = generate(prompt)
print(response)
print()

# Example 2: Prompt with constraint plus additional context
print("=== Example 2: Prompt with Structure ===")
prompt = """
Write a joke about AI that has to do with them turning rogue

A joke contains 3 sections:
- A setup
- A punchline
- A contradiction

Maintain a jovial tone.
"""
response = generate(prompt)
print(response)
print()

# Example 3: Multiple generations with same prompt
print("=== Example 3: Multiple Generations (showing variability) ===")
for i in range(3):
    response = generate("Write a joke about AI that has to do with them turning rogue")
    print(f"Generation {i+1}:")
    print(response)
    print("---")