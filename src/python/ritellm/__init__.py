import json
from typing import Any

import weave
from .ritellm import openai_completion

__all__ = ["openai_completion", "completion"]



def completion(
    model: str,
    messages: list[dict[str, str]],
    temperature=None,
    max_tokens: int | None = None,
    additional_params: str | None = None,
) -> dict[str, Any]:
    """
    Clean Python wrapper around the openai_completion function.

    This function provides a convenient interface to call OpenAI's chat completion API
    through the Rust-backed openai_completion binding.

    Args:
        model (str): The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        messages (list): A list of message dictionaries with "role" and "content" keys
        temperature (float, optional): Sampling temperature (0.0 to 2.0)
        max_tokens (int, optional): Maximum tokens to generate
        additional_params (str, optional): Additional parameters as a JSON string

    Returns:
        str: A JSON string containing the API response

    Environment Variables:
        OPENAI_API_KEY: Required - Your OpenAI API key

    Example:
        >>> import json
        >>> from ritellm import completion
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> response_json = completion(model="gpt-3.5-turbo", messages=messages)
        >>> response = json.loads(response_json)
        >>> print(response["choices"][0]["message"]["content"])
    """
    response_json = openai_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        additional_params=additional_params,
    )
    return json.loads(response_json)
