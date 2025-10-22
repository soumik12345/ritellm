import asyncio

from ritellm import acompletion, completion


def test_openai_completion_sync():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = completion(
        model="openai/gpt-3.5-turbo", messages=messages, temperature=0.7, max_tokens=100
    )
    assistant_message = response["choices"][0]["message"]["content"]
    assert "france" in assistant_message.lower()
    assert response["usage"]["prompt_tokens"] > 10
    assert response["usage"]["completion_tokens"] > 0
    assert response["usage"]["total_tokens"] > response["usage"]["prompt_tokens"]


def test_openai_completion_streaming():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming in Rust."},
    ]

    response = completion(
        model="openai/gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        max_tokens=200,
        stream=True,
    )
    assistant_message = ""
    for chunk in response:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            assistant_message += content
    assert "rust" in assistant_message.lower()


def test_openai_completion_async():
    """Test async completion without streaming."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    async def run_async_test():
        response = await acompletion(
            model="openai/gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
        assistant_message = response["choices"][0]["message"]["content"]
        assert "france" in assistant_message.lower()
        assert response["usage"]["prompt_tokens"] > 10
        assert response["usage"]["completion_tokens"] > 0
        assert response["usage"]["total_tokens"] > response["usage"]["prompt_tokens"]

    asyncio.run(run_async_test())


def test_openai_completion_async_streaming():
    """Test async completion with streaming."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming in Rust."},
    ]

    async def run_async_streaming_test():
        response = await acompletion(
            model="openai/gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            max_tokens=200,
            stream=True,
        )

        assistant_message = ""
        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                assistant_message += content

        assert "rust" in assistant_message.lower()

    asyncio.run(run_async_streaming_test())


def test_openai_completion_async_concurrent():
    """Test multiple concurrent async requests."""
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Italy?"},
        ],
    ]

    async def run_concurrent_test():
        # Make 3 concurrent requests
        tasks = [
            acompletion(
                model="openai/gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
            )
            for messages in messages_list
        ]

        responses = await asyncio.gather(*tasks)

        # Verify all responses
        assert len(responses) == 3
        assert "france" in responses[0]["choices"][0]["message"]["content"].lower()
        assert "germany" in responses[1]["choices"][0]["message"]["content"].lower()
        assert "italy" in responses[2]["choices"][0]["message"]["content"].lower()

    asyncio.run(run_concurrent_test())
