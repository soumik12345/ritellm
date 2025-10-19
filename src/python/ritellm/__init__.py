import json
from typing import Any

from .ritellm import completion_gateway


def completion(
    model: str,
    messages: list[dict[str, str]],
    temperature=None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    additional_params: str | None = None,
) -> dict[str, Any]:
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
        response = completion(model="openai/gpt-3.5-turbo", messages=messages)

        print(response["choices"][0]["message"]["content"])
        ```

    Args:
        model (str): The model to use with provider prefix (e.g., "openai/gpt-4", "openai/gpt-3.5-turbo")
        messages (list): A list of message dictionaries with "role" and "content" keys
        temperature (float, optional): Sampling temperature (0.0 to 2.0)
        max_tokens (int, optional): Maximum tokens to generate
        base_url (str, optional): Base URL for the API endpoint
        additional_params (str, optional): Additional parameters as a JSON string

    Returns:
        dict: A dictionary containing the API response

    Raises:
        ValueError: If the provider prefix is not supported

    Environment Variables:
        OPENAI_API_KEY: Required for OpenAI models
    """
    response_json = completion_gateway(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        additional_params=additional_params,
    )
    return json.loads(response_json)
