import json

import rich
import weave
from dotenv import load_dotenv

from ritellm import openai_completion

load_dotenv()

weave.init(project_name="ritellm")

# Define the messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what Rust is in one sentence."},
]

response_json = weave.op(openai_completion)(
    model="gpt-3.5-turbo", messages=messages, temperature=0.7, max_tokens=100
)

# Parse and display the response
response = json.loads(response_json)
assistant_message = response["choices"][0]["message"]["content"]

rich.print("\nAssistant's response:")
rich.print(assistant_message)

rich.print("\nToken usage:")
rich.print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
rich.print(f"  Completion tokens: {response['usage']['completion_tokens']}")
rich.print(f"  Total tokens: {response['usage']['total_tokens']}")
