from ritellm import completion


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
