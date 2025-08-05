"""
Level 1: Structured Outputs
This example shows how to get structured JSON outputs from the model
"""
import openai
import time
import json

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

# System prompt for JSON output
system_prompt = """
You are a comedian who likes to tell stories before delivering a punchline. You are always funny.
Jokes contain 3 sections:
- A setup
- A punchline
- A contradiction
- A full comedian joke delivery

Always maintain a jovial tone.

You must output your response in a JSON format. For example:
{
    "setup": ..,
    "punchline": ..,
    "contradiction": ..,
    "delivery": ..
}

We will extract the json using json.loads(response) in Python, so only response JSON and nothing else.
"""

user_prompt = "Write a joke about AI that has to do with them turning rogue."

print("=== Structured JSON Output ===")
response = generate_with_system_prompt(system_prompt, user_prompt)
print("Raw response:")
print(response)
print()

# Parse the JSON response
try:
    response_extracted = json.loads(response)
    print("Parsed JSON:")
    print(json.dumps(response_extracted, indent=2))
    print()
    print("Just the delivery:")
    print(response_extracted["delivery"])
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")