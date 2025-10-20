# Quickstart

## Basic Usage

```python
from ritellm import completion

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

response = completion(
    model="openai/gpt-3.5-turbo",
    messages=messages
)

print(response["choices"][0]["message"]["content"])
```

### Streaming

To enable streaming, set `stream=True`:

```python
from ritellm import completion

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short poem."}
]

response = completion(
    model="openai/gpt-3.5-turbo",
    messages=messages,
    stream=True  # Enable streaming
)

# Iterate over chunks as they arrive
for chunk in response:
    if "choices" in chunk and len(chunk["choices"]) > 0:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)

print()  # New line after streaming completes
```

## Response Format

### Non-Streaming Response

When `stream=False` (default), you receive a complete response dictionary:

```python
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-3.5-turbo-0125",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
```

### Streaming Response

When `stream=True`, you receive an iterator of chunk dictionaries:

```python
# First chunk (usually empty or with role)
{
    "id": "chatcmpl-...",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "gpt-3.5-turbo-0125",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": None
        }
    ]
}

# Content chunks
{
    "id": "chatcmpl-...",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "gpt-3.5-turbo-0125",
    "choices": [
        {
            "index": 0,
            "delta": {
                "content": "Hello"
            },
            "finish_reason": None
        }
    ]
}

# Final chunk
{
    "id": "chatcmpl-...",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "gpt-3.5-turbo-0125",
    "choices": [
        {
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }
    ]
}
```

## Complete Example

Here's a complete example that handles streaming responses gracefully:

```python
from ritellm import completion

def stream_completion(messages, model="openai/gpt-3.5-turbo"):
    """Stream a completion and print the response."""
    response = completion(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=500
    )
    
    print("Assistant: ", end="", flush=True)
    full_response = ""
    
    for chunk in response:
        if "choices" not in chunk or len(chunk["choices"]) == 0:
            continue
            
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")
        
        if content:
            print(content, end="", flush=True)
            full_response += content
        
        # Check if streaming is complete
        if choice.get("finish_reason") == "stop":
            break
    
    print()  # New line
    return full_response


# Usage
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response_text = stream_completion(messages)
print(f"\n\nFull response length: {len(response_text)} characters")
```

## Best Practices

1. **Always handle missing content**: Not all chunks will have content, especially the first and last chunks.
   
   ```python
   content = chunk["choices"][0]["delta"].get("content", "")
   if content:
       print(content, end="", flush=True)
   ```

2. **Use flush=True**: When printing streaming content, use `flush=True` to ensure immediate output.

3. **Check finish_reason**: Monitor the `finish_reason` field to know when streaming is complete.

4. **Error handling**: Wrap streaming in try-except blocks to handle network issues gracefully.

   ```python
   try:
       for chunk in response:
           # Process chunk
           pass
   except Exception as e:
       print(f"\nStreaming error: {e}")
   ```

5. **Accumulate response**: If you need the full response text, accumulate it from the chunks:

   ```python
   full_text = ""
   for chunk in response:
       content = chunk["choices"][0]["delta"].get("content", "")
       full_text += content
   ```

## Supported Providers

Currently, streaming is supported for:

- ✅ OpenAI (`openai/` prefix)

More providers will be added in future releases.
