# RiteLLM

<a href="https://www.python.org/downloads/">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python Version">
</a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</a>

## Overview

RiteLLM is a high-performance LLM (Large Language Model) gateway that provides a unified interface for interacting with multiple LLM providers. Built with Rust and exposed through elegant Python bindings, RiteLLM combines the speed of compiled systems programming with the ease of Python development.

## ✨ Key Features

- **🚀 Unified LLM Gateway**: Single, consistent API for multiple LLM providers
- **🔌 Provider Support**: Currently supports OpenAI, with more providers coming soon (Anthropic, Google, Cohere, and more)
- **⚡ Rust-Powered Performance**: Core engine built in Rust for maximum speed and efficiency
- **📊 First-Class Observability**: Built-in integration with [Weights & Biases Weave](https://wandb.ai/site/weave) for seamless tracing, monitoring, and debugging
- **🐍 Pythonic Interface**: Clean, intuitive Python API that feels native to the ecosystem
- **🔒 Type-Safe**: Full type hints for better IDE support and code quality

## 🚀 Installation

Install RiteLLM using pip:

```bash
uv pip install git+https://github.com/soumik12345/ritellm.git
```

## 💻 Quick Start

### Basic Usage

```python
from ritellm import completion

# Define your messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what Rust is in one sentence."}
]

# Make a completion request
response = completion(
    model="openai/gpt-3.5-turbo",
    messages=messages,
    temperature=0.7,
    max_tokens=100
)

# Access the response
print(response["choices"][0]["message"]["content"])
print(f"Tokens used: {response['usage']['total_tokens']}")
```

### With Weave Tracing

RiteLLM has first-class support for Weave, enabling automatic tracing and monitoring of your LLM calls:

```python
import weave
from ritellm import completion

# Initialize Weave
weave.init(project_name="my-llm-project")

# Wrap completion with Weave's op decorator for automatic tracing
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = weave.op(completion)(
    model="openai/gpt-3.5-turbo",
    messages=messages,
    temperature=0.7
)

# Your calls are now automatically traced in Weave!
```

<div align="center">
Made with ❤️ and 🦀
</div>
