import json
from typing import Any, Iterator

from .ritellm import completion_gateway


def completion(
    model: str,
    messages: list[dict[str, str]],
    temperature=None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    stream: bool = False,
    additional_params: str | None = None,
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """
    Clean Python wrapper around the completion_gateway function.

    This function provides a convenient interface to call various LLM providers' chat completion APIs
    through the Rust-backed completion_gateway binding. The model string should include a provider prefix.

    !!! example
        ```python
        from ritellm import completion

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        # Non-streaming
        response = completion(model="openai/gpt-3.5-turbo", messages=messages)
        print(response["choices"][0]["message"]["content"])
        
        # Streaming
        response = completion(model="openai/gpt-3.5-turbo", messages=messages, stream=True)
        for chunk in response:
            print(chunk["choices"][0]["delta"].get("content", ""), end="")
        ```

    Args:
        model (str): The model to use with provider prefix (e.g., "openai/gpt-4", "openai/gpt-3.5-turbo")
        messages (list): A list of message dictionaries with "role" and "content" keys
        temperature (float, optional): Sampling temperature (0.0 to 2.0)
        max_tokens (int, optional): Maximum tokens to generate
        base_url (str, optional): Base URL for the API endpoint
        stream (bool, optional): Enable streaming responses (default: False)
        additional_params (str, optional): Additional parameters as a JSON string

    Returns:
        dict | Iterator[dict]: A dictionary containing the API response, or an iterator of chunks if stream=True

    Raises:
        ValueError: If the provider prefix is not supported

    Environment Variables:
        OPENAI_API_KEY: Required for OpenAI models
    """
    result = completion_gateway(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        stream=stream,
        additional_params=additional_params,
    )
    
    # If result is a string (non-streaming), parse and return the JSON
    if isinstance(result, str):
        return json.loads(result)
    
    # If result is an iterator (streaming), wrap it to parse JSON chunks
    def parse_streaming_chunks():
        for chunk in result:
            yield json.loads(chunk)
    
    return parse_streaming_chunks()
