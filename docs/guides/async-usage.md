# Async Usage

RiteLLM provides async support through the `acompletion` function, which allows you to make non-blocking API calls and handle concurrent requests efficiently.

## Basic Async Usage

The simplest way to use async mode is with the `acompletion` function:

```python
import asyncio
from ritellm import acompletion

async def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    response = await acompletion(
        model="openai/gpt-3.5-turbo",
        messages=messages
    )
    
    print(response["choices"][0]["message"]["content"])

asyncio.run(main())
```

## Async with Streaming

Enable streaming in async mode by setting `stream=True`:

```python
import asyncio
from ritellm import acompletion

async def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about Python."}
    ]
    
    response = await acompletion(
        model="openai/gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    
    # Stream the response as it arrives
    for chunk in response:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
    
    print()  # New line after streaming completes

asyncio.run(main())
```

## Concurrent Requests

One of the main benefits of async is handling multiple requests concurrently:

```python
import asyncio
from ritellm import acompletion

async def ask_question(question: str):
    """Ask a single question and return the response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    
    response = await acompletion(
        model="openai/gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )
    
    return response["choices"][0]["message"]["content"]

async def main():
    # Define multiple questions
    questions = [
        "What is Python?",
        "What is Rust?",
        "What is async programming?"
    ]
    
    # Run all questions concurrently
    tasks = [ask_question(q) for q in questions]
    answers = await asyncio.gather(*tasks)
    
    # Print all answers
    for question, answer in zip(questions, answers):
        print(f"Q: {question}")
        print(f"A: {answer}\n")

asyncio.run(main())
```

## Complete Example: Async Chat Application

Here's a complete example showing a simple async chat application:

```python
import asyncio
from ritellm import acompletion

async def chat_streaming(messages: list[dict], model: str = "openai/gpt-3.5-turbo"):
    """Send a chat message and stream the response."""
    response = await acompletion(
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
        
        if choice.get("finish_reason") == "stop":
            break
    
    print("\n")
    return full_response

async def main():
    """Simple chat loop with async streaming."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    # Simulate a conversation
    user_messages = [
        "Hello! What can you help me with?",
        "Tell me about async programming in Python.",
        "Thanks!"
    ]
    
    for user_msg in user_messages:
        print(f"User: {user_msg}\n")
        messages.append({"role": "user", "content": user_msg})
        
        # Get streaming response
        assistant_msg = await chat_streaming(messages)
        messages.append({"role": "assistant", "content": assistant_msg})

asyncio.run(main())
```

## Best Practices

### 1. Use `asyncio.gather()` for Concurrent Requests

When you need to make multiple API calls, use `asyncio.gather()` to run them concurrently:

```python
# Good: Concurrent requests
results = await asyncio.gather(
    acompletion(model="openai/gpt-3.5-turbo", messages=messages1),
    acompletion(model="openai/gpt-3.5-turbo", messages=messages2),
    acompletion(model="openai/gpt-3.5-turbo", messages=messages3)
)

# Bad: Sequential requests (slower)
result1 = await acompletion(model="openai/gpt-3.5-turbo", messages=messages1)
result2 = await acompletion(model="openai/gpt-3.5-turbo", messages=messages2)
result3 = await acompletion(model="openai/gpt-3.5-turbo", messages=messages3)
```

### 2. Handle Errors Gracefully

Wrap async calls in try-except blocks to handle failures:

```python
async def safe_completion(messages):
    try:
        response = await acompletion(
            model="openai/gpt-3.5-turbo",
            messages=messages
        )
        return response
    except Exception as e:
        print(f"Error during completion: {e}")
        return None
```

### 3. Use Streaming for Long Responses

For better user experience with long responses, use streaming:

```python
# Good for long responses: User sees content as it arrives
response = await acompletion(
    model="openai/gpt-3.5-turbo",
    messages=messages,
    stream=True
)

# Less ideal for long responses: User waits for entire response
response = await acompletion(
    model="openai/gpt-3.5-turbo",
    messages=messages,
    stream=False
)
```

### 4. Rate Limiting with Semaphores

Control concurrency to avoid rate limits:

```python
async def main():
    # Limit to 5 concurrent requests
    semaphore = asyncio.Semaphore(5)
    
    async def limited_completion(messages):
        async with semaphore:
            return await acompletion(
                model="openai/gpt-3.5-turbo",
                messages=messages
            )
    
    # Create many tasks but only 5 run at once
    tasks = [limited_completion(msg) for msg in message_list]
    results = await asyncio.gather(*tasks)
```

## Comparison: Sync vs Async

### Synchronous (Blocking)

```python
from ritellm import completion

# Sequential execution - slow for multiple requests
for i in range(10):
    response = completion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Question {i}"}]
    )
    print(response["choices"][0]["message"]["content"])
```

### Asynchronous (Non-blocking)

```python
import asyncio
from ritellm import acompletion

async def main():
    # Concurrent execution - fast!
    tasks = [
        acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Question {i}"}]
        )
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result["choices"][0]["message"]["content"])

asyncio.run(main())
```
