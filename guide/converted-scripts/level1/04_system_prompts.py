"""
Level 1: System Prompts
This example shows how to use system prompts to set context and behavior
"""
import openai
import time

client = openai.Client()

def generate_with_system_prompt(system_prompt, user_prompt, model="gpt-4o-mini"):
    """Generation function with system prompt"""
    start_time = time.time()
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(f"Response time: {time.time() - start_time:.2f} seconds")
    return response.choices[0].message.content

# Example 1: Basic system prompt
print("=== Example 1: Basic System Prompt ===")
system_prompt = """
You are a comedian who likes to tell stories before delivering a punchline. You are always funny.
"""
user_prompt = "Write a joke about AI that has to do with them turning rogue. Maintain a jovial tone."

response = generate_with_system_prompt(system_prompt, user_prompt)
print(response)
print()

# Example 2: Detailed system prompt with structure
print("=== Example 2: Structured System Prompt ===")
system_prompt = """
You are a comedian who likes to tell stories before delivering a punchline. You are always funny.
Jokes contain 3 sections:
- A setup
- A punchline
- A contradiction
- A full comedian joke delivery

Always maintain a jovial tone.
"""
user_prompt = "Write a joke about AI that has to do with them turning rogue."

response = generate_with_system_prompt(system_prompt, user_prompt)
print(response)