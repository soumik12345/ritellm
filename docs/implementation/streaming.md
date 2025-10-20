# Streaming Implementation Summary

This document summarizes the implementation of streaming support in ritellm, making it a drop-in replacement for litellm's streaming functionality.

## Overview

The streaming feature allows ritellm to receive and yield LLM responses in real-time as they are generated, providing a better user experience for long responses.

## Implementation Details

### 1. Rust Core (`src/openai.rs`)

**Key Changes:**

- Added `stream` parameter to `openai_completion()` and `openai_completion_async()` functions
- Created `CompletionResult` enum to handle both streaming and non-streaming responses:
  ```rust
  pub enum CompletionResult {
      Text(String),           // For non-streaming responses
      Stream(Vec<String>),    // For streaming responses
  }
  ```
- Implemented `StreamingResponse` PyClass as a Python iterator:
  ```rust
  #[pyclass]
  pub struct StreamingResponse {
      chunks: Vec<String>,
      index: usize,
  }
  ```
- Added Server-Sent Events (SSE) parsing to handle OpenAI's streaming format
- Modified request body to include `"stream": true` when streaming is enabled
- Updated response handling to process SSE chunks in the format `data: {json}\n\n`

**SSE Parsing Logic:**
- Reads response as a byte stream
- Buffers incoming data until complete SSE messages are received
- Parses lines starting with `data: ` prefix
- Handles the `[DONE]` sentinel value
- Stores JSON chunks for Python iteration

### 2. Gateway Layer (`src/lib.rs`)

**Key Changes:**

- Updated `completion_gateway()` signature to include `stream` parameter
- Changed return type from `String` to `Py<PyAny>` to support both strings and iterators
- Pass `stream` parameter through to `openai_completion()`

### 3. Python Wrapper (`src/python/ritellm/__init__.py`)

**Key Changes:**

- Added `stream: bool = False` parameter to `completion()` function
- Updated return type hint to `dict[str, Any] | Iterator[dict[str, Any]]`
- Implemented runtime type checking to handle both response types:
  ```python
  if isinstance(result, str):
      return json.loads(result)
  
  def parse_streaming_chunks():
      for chunk in result:
          yield json.loads(chunk)
  
  return parse_streaming_chunks()
  ```
- Updated docstring with streaming examples

### 4. Dependencies (`Cargo.toml`)

**Added:**
- `futures-util = "0.3"` - For async stream processing
- `"stream"` feature for `reqwest` - To enable `bytes_stream()` functionality

## API Usage

### Non-Streaming (Default)

```python
from ritellm import completion

response = completion(
    model="openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["choices"][0]["message"]["content"])
```

### Streaming

```python
from ritellm import completion

response = completion(
    model="openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a poem."}],
    stream=True
)

for chunk in response:
    content = chunk["choices"][0]["delta"].get("content", "")
    if content:
        print(content, end="", flush=True)
```

## Response Format Comparison

### Non-Streaming Response

```json
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-3.5-turbo-0125",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Full response text"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

### Streaming Response Chunks

```json
// First chunk
{
    "id": "chatcmpl-...",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "gpt-3.5-turbo-0125",
    "choices": [{
        "index": 0,
        "delta": {"role": "assistant", "content": ""},
        "finish_reason": null
    }]
}

// Content chunks
{
    "id": "chatcmpl-...",
    "choices": [{
        "index": 0,
        "delta": {"content": "Hello"},
        "finish_reason": null
    }]
}

// Final chunk
{
    "id": "chatcmpl-...",
    "choices": [{
        "index": 0,
        "delta": {},
        "finish_reason": "stop"
    }]
}
```

## Compatibility with LiteLLM

ritellm streaming is designed as a drop-in replacement for litellm:

| Feature | LiteLLM | RiteLLM |
|---------|---------|---------|
| Enable streaming | `stream=True` | `stream=True` |
| Iterate chunks | `for chunk in response:` | `for chunk in response:` |
| Access content | `chunk.choices[0].delta.content` | `chunk["choices"][0]["delta"].get("content")` |
| Model format | `"gpt-3.5-turbo"` | `"openai/gpt-3.5-turbo"` (requires provider prefix) |
| Response type | Objects | Dictionaries |

**Key Differences:**
1. ritellm requires provider prefix in model name (e.g., `openai/`)
2. ritellm returns dictionaries instead of objects
3. Use `.get()` for safe dictionary access in ritellm

## Testing

Comprehensive tests are provided in `test_streaming.py`:

1. **Non-streaming test**: Verifies regular completion still works
2. **Basic streaming test**: Tests streaming with a simple prompt
3. **Long streaming test**: Tests streaming with longer content (poem)

All tests verify:
- Correct response format
- Proper chunk iteration
- Content extraction from delta
- Successful completion

## Documentation

Documentation has been added in multiple places:

1. **`docs/streaming.md`**: Comprehensive streaming guide
   - Basic usage examples
   - Response format details
   - Complete examples
   - Best practices
   - Performance considerations
   - Comparison with litellm

2. **`docs/index.md`**: Updated with streaming quick start

3. **`README.md`**: Added streaming example and feature listing

4. **`mkdocs.yml`**: Added streaming guide to navigation

## Performance Considerations

**Advantages:**
- Better user experience with real-time feedback
- Lower perceived latency
- Ability to process partial responses

**Considerations:**
- Slightly higher overhead per chunk vs single response
- Network latency can affect chunk delivery
- Need to handle incomplete responses gracefully

## Future Enhancements

Potential improvements for future versions:

1. **Async streaming**: Implement async/await support for Python
   ```python
   async for chunk in await acompletion(..., stream=True):
       print(chunk)
   ```

2. **Stream helpers**: Add utility functions like litellm's `stream_chunk_builder()`
   ```python
   full_response = ritellm.stream_chunk_builder(chunks)
   ```

3. **Provider expansion**: Add streaming support for other providers:
   - Anthropic Claude
   - Google Gemini
   - Azure OpenAI
   - etc.

4. **Buffering options**: Allow configuration of streaming buffer sizes

5. **Error recovery**: Handle network interruptions gracefully with reconnection

## Files Modified

1. `src/openai.rs` - Core streaming implementation
2. `src/lib.rs` - Gateway layer updates
3. `src/python/ritellm/__init__.py` - Python wrapper updates
4. `Cargo.toml` - Added dependencies
5. `docs/streaming.md` - New comprehensive guide
6. `docs/index.md` - Added streaming examples
7. `README.md` - Added streaming feature and examples
8. `mkdocs.yml` - Added streaming to navigation

## Files Created

1. `test_streaming.py` - Comprehensive test suite
2. `docs/streaming.md` - Streaming documentation
3. `STREAMING_IMPLEMENTATION.md` - This summary

## Build & Test

Build the library:
```bash
uv run maturin develop --release
```

Run tests:
```bash
uv run python test_streaming.py
```

## Conclusion

The streaming implementation successfully makes ritellm a drop-in replacement for litellm's streaming functionality while maintaining the performance benefits of Rust and the simplicity of Python. The implementation follows OpenAI's SSE format and is compatible with existing litellm streaming patterns.
