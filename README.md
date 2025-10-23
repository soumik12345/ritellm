<div align="center">

# RiteLLM

**A blazingly fast LLM gateway built with Rust 🦀**

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## ✨ Features

- 🚀 **Unified API** - Single interface for multiple LLM providers
- 📡 **Streaming Support** - Real-time streaming responses with async/await
- 🔄 **Provider Routing** - Automatic routing based on model prefix (e.g., `openai/gpt-4o`)
- ⚡ **Zero-Copy Streaming** - Efficient stream processing using Rust combinators
- 🛡️ **Type Safety** - Full Rust type safety with comprehensive error handling
- 🎯 **Simple API** - Clean, ergonomic interface for chat completions

## 🚦 Quick Start

### Non-Streaming Completion

```rust
use ritellm::{ChatCompletionRequest, CompletionResponse, Message, completion};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let request = ChatCompletionRequest {
        model: "openai/gpt-4o-mini".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
        }],
        temperature: Some(0.7),
        max_tokens: Some(50),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        n: None,
        stream: None,
    };

    match completion(request).await {
        Ok(CompletionResponse::Response(response)) => {
            println!("Model: {}", response.model);
            println!("Response: {}", response.choices[0].message.content);
            println!("Tokens used: {}", response.usage.total_tokens);
        }
        Ok(CompletionResponse::Stream(_)) => {
            eprintln!("Unexpected stream response");
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
    Ok(())
}
```

### Streaming Completion

```rust
use futures::StreamExt;
use ritellm::{ChatCompletionRequest, CompletionResponse, Message, completion};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let streaming_request = ChatCompletionRequest {
        model: "openai/gpt-4o-mini".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Count from 1 to 20.".to_string(),
        }],
        temperature: Some(0.7),
        max_tokens: Some(50),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        n: None,
        stream: Some(true),
    };

    match completion(streaming_request).await {
        Ok(CompletionResponse::Response(_)) => {
            eprintln!("Unexpected non-streaming response");
        }
        Ok(CompletionResponse::Stream(stream)) => {
            use std::io::Write;

            print!("Response: ");

            // Use stream combinators for cleaner code
            stream
                .filter_map(|result| async move {
                    match result {
                        Ok(chunk) => chunk
                            .choices
                            .first()
                            .and_then(|choice| choice.delta.content.clone()),
                        Err(e) => {
                            eprintln!("\nError: {}", e);
                            None
                        }
                    }
                })
                .for_each(|content| async move {
                    print!("{}", content);
                    std::io::stdout().flush().ok();
                })
                .await;

            println!();
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
    Ok(())
}
```

## 🔑 Environment Setup

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ritellm = { path = "path/to/ritellm" }
tokio = { version = "1.48", features = ["full"] }
anyhow = "1.0"
futures = "0.3"
```

## 🎯 Supported Providers

- ✅ **OpenAI** - Use `openai/` prefix (e.g., `openai/gpt-4o`, `openai/gpt-4o-mini`)
- 🔜 More providers coming soon!
