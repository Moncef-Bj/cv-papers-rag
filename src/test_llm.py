"""
First test: Call an LLM (GPT) with our API key.
This file verifies that everything is properly configured.
"""

# ============================================================
# STEP 1: Imports
# ============================================================
import os
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# STEP 2: Load environment variables
# ============================================================
# load_dotenv() reads the .env file and loads variables into memory
# It's like running: export OPENAI_API_KEY=sk-xxx in your terminal
load_dotenv()

# os.getenv() retrieves the value of an environment variable
# If .env contains OPENAI_API_KEY=sk-xxx, then api_key = "sk-xxx"
api_key = os.getenv("OPENAI_API_KEY")

# Verification: make sure the key exists
if not api_key:
    print(" ERROR: OPENAI_API_KEY not found in .env")
    exit()
else:
    print(" API key loaded successfully")

# ============================================================
# STEP 3: Create the OpenAI client
# ============================================================
# The "client" is the object that will communicate with the OpenAI API
client = OpenAI(api_key=api_key)

# ============================================================
# STEP 4: Make a call to the LLM
# ============================================================
print("\n Sending question to the LLM...")

# This is where the magic happens!
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Model to use (mini = cheaper)
    messages=[
        {
            "role": "system",  # General instructions for the model
            "content": "You are a Computer Vision expert. Answer concisely."
        },
        {
            "role": "user",  # The user's question
            "content": "What is the exact Rank-1 accuracy of GAF-Net on the iLIDS-VID dataset? And what loss function does it use?"

        }
    ],
    temperature=0.5,  # Creativity (0 = deterministic, 1 = creative)
    max_tokens=200    # Response length limit
)

# ============================================================
# STEP 5: Display the response
# ============================================================
# The response is a complex object, we just extract the text
answer = response.choices[0].message.content

print("\n" + "="*50)
print(" LLM RESPONSE:")
print("="*50)
print(answer)
print("="*50)

# Bonus: display usage stats (tokens = cost)
print(f"\n Tokens used: {response.usage.total_tokens}")