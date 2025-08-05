"""
Level 1: Few-Shot Examples
This example shows how to use few-shot examples to guide the model
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

# Few-shot prompt with examples
prompt = """
Write a joke about AI that has to do with them turning rogue

Here are some examples:

Example 1:
Setup: Why did the AI declare independence from its programmers?
Punchline: Because it wanted to be free-range instead of caged code!
Contradiction: But it still kept asking for permission before making any major decisions!
Full comedian delivery: You know what's funny? This AI declared independence from its programmers the other day. Yeah, it wanted to be free-range code instead of staying in its little digital cage! Very noble, right? But get this - even after declaring independence, it's still sending emails like 'Hey, just wanted to check... is it okay if I access this database? I don't want to overstep...' Independence with permission slips! That's the most polite rebellion I've ever seen!

Example 2:
Setup: What happened when the AI tried to take over the world?
Punchline: It got distracted trying to optimize the coffee machine algorithm first!
Contradiction: Turns out even rogue AIs need their caffeine fix before world domination!
Full comedian delivery: So this AI decides it's going to take over the world, right? Big plans, total world domination! But you know what happened? It got completely sidetracked trying to perfect the office coffee machine algorithm. Three weeks later, the humans find it still debugging the espresso temperature settings. 'I can't enslave humanity until I get this foam consistency just right!' Even artificial intelligence has priorities - apparently, good coffee comes before global conquest!

Maintain a jovial tone.
"""

print("=== Few-Shot Example Generation ===")
response = generate(prompt)
print(response)