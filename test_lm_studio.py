"""
Quick test script for LM Studio
Make sure LM Studio is running with a model loaded before running this script.
"""

from qwen_agent.llm import get_chat_model

# Configure for LM Studio (default port is 1234)
llm = get_chat_model({
    'model': 'your-model-name',  # LM Studio will use the loaded model
    'model_server': 'http://localhost:1234/v1',
    'api_key': 'EMPTY',  # LM Studio doesn't require an API key
})

# Test basic chat
messages = [{'role': 'user', 'content': 'Hello! Who are you?'}]

print("Testing LM Studio connection...\n")
print("User: Hello! Who are you?\n")
print("Assistant: ", end='')

for response in llm.chat(messages=messages):
    # Streaming output
    print(response[-1]['content'], end='', flush=True)

print("\n\n[OK] LM Studio is working!")
