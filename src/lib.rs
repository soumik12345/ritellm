//! # RiteLLM
//!
//! A blazingly fast LLM gateway built with Rust 🦀
//!
//! ## Features
//!
//! - 🚀 **Unified API** - Single interface for multiple LLM providers
//! - 📡 **Streaming Support** - Real-time streaming responses with async/await
//! - 🔄 **Provider Routing** - Automatic routing based on model prefix
//! - ⚡ **Zero-Copy Streaming** - Efficient stream processing using Rust combinators
//! - 🛡️ **Type Safety** - Full Rust type safety with comprehensive error handling
//!
//! ## Quick Start
//!
//! ### Non-Streaming Completion
//!
//! ```no_run
//! use ritellm::{completion, CompletionResponse, Message};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! match completion(
//!     "openai/gpt-4o-mini".to_string(),
//!     vec![Message {
//!         role: "user".to_string(),
//!         content: "What is 2+2?".to_string(),
//!     }],
//!     Some(0.7),
//!     Some(50),
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//! ).await? {
//!     CompletionResponse::Response(response) => {
//!         println!("Response: {}", response.choices[0].message.content);
//!     }
//!     _ => {}
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Streaming Completion
//!
//! ```no_run
//! use ritellm::{completion, CompletionResponse, Message};
//! use futures::StreamExt;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! match completion(
//!     "openai/gpt-4o-mini".to_string(),
//!     vec![Message {
//!         role: "user".to_string(),
//!         content: "Tell me a story.".to_string(),
//!     }],
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//!     None,
//!     Some(true),
//!     None,
//! ).await? {
//!     CompletionResponse::Stream(stream) => {
//!         // Use stream combinators for elegant processing
//!         stream
//!             .filter_map(|result| async move {
//!                 result.ok()?.choices.first()?.delta.content.clone()
//!             })
//!             .for_each(|content| async move {
//!                 print!("{}", content);
//!             })
//!             .await;
//!     }
//!     _ => {}
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Environment Setup
//!
//! Set your OpenAI API key:
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! ```

pub mod openai;

use anyhow::{Context, Result};

// Re-export commonly used types for convenience
pub use openai::{
    ChatChoiceStream, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseStream,
    ChatCompletionStreamResponse, ChatCompletionStreamResponseDelta, Choice, Message, Usage,
    openai_completion, openai_completion_stream,
};

/// Enum representing either a complete response or a stream of response chunks
pub enum CompletionResponse {
    /// A complete, non-streaming response
    Response(ChatCompletionResponse),
    /// A stream of response chunks
    Stream(ChatCompletionResponseStream),
}

impl std::fmt::Debug for CompletionResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompletionResponse::Response(response) => {
                f.debug_tuple("Response").field(response).finish()
            }
            CompletionResponse::Stream(_) => f.debug_tuple("Stream").field(&"<stream>").finish(),
        }
    }
}

/// Unified completion function that routes to the appropriate provider based on model prefix
///
/// # Arguments
///
/// * `model` - The model to use, specified in format "provider/model" (e.g., "openai/gpt-4o-mini")
/// * `messages` - The conversation messages
/// * `temperature` - Sampling temperature (0.0 to 2.0)
/// * `max_tokens` - Maximum number of tokens to generate
/// * `top_p` - Nucleus sampling parameter
/// * `frequency_penalty` - Frequency penalty (-2.0 to 2.0)
/// * `presence_penalty` - Presence penalty (-2.0 to 2.0)
/// * `stop` - Stop sequences
/// * `n` - Number of completions to generate
/// * `stream` - Whether to stream the response
/// * `base_url` - Custom base URL for the API endpoint
///
/// # Returns
///
/// * `Result<CompletionResponse>` - Either a complete response or a stream, depending on the `stream` parameter
///
/// # Supported Providers
///
/// * `openai/` - Routes to OpenAI API (e.g., "openai/gpt-4o", "openai/gpt-4o-mini")
///
/// # Environment Variables
///
/// * `OPENAI_API_KEY` - Required for OpenAI models
///
/// # Example (Non-streaming)
///
/// ```no_run
/// use ritellm::{completion, CompletionResponse, Message};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     match completion(
///         "openai/gpt-4o-mini".to_string(),
///         vec![Message {
///             role: "user".to_string(),
///             content: "Hello!".to_string(),
///         }],
///         Some(0.7),
///         Some(100),
///         None,
///         None,
///         None,
///         None,
///         None,
///         None,
///         None,
///     ).await? {
///         CompletionResponse::Response(response) => {
///             println!("{}", response.choices[0].message.content);
///         }
///         CompletionResponse::Stream(_) => {
///             // Handle streaming case
///         }
///     }
///     Ok(())
/// }
/// ```
///
/// # Example (Streaming)
///
/// ```no_run
/// use ritellm::{completion, CompletionResponse, Message};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     match completion(
///         "openai/gpt-4o-mini".to_string(),
///         vec![Message {
///             role: "user".to_string(),
///             content: "Tell me a story.".to_string(),
///         }],
///         Some(0.7),
///         Some(100),
///         None,
///         None,
///         None,
///         None,
///         None,
///         Some(true),
///         None,
///     ).await? {
///         CompletionResponse::Response(response) => {
///             println!("{}", response.choices[0].message.content);
///         }
///         CompletionResponse::Stream(mut stream) => {
///             while let Some(result) = stream.next().await {
///                 match result {
///                     Ok(chunk) => {
///                         if let Some(choice) = chunk.choices.first() {
///                             if let Some(content) = &choice.delta.content {
///                                 print!("{}", content);
///                             }
///                         }
///                     }
///                     Err(e) => eprintln!("Error: {}", e),
///                 }
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
pub async fn completion(
    model: String,
    messages: Vec<Message>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    n: Option<u32>,
    stream: Option<bool>,
    base_url: Option<String>,
) -> Result<CompletionResponse> {
    // Create the ChatCompletionRequest from individual parameters
    let mut request = ChatCompletionRequest {
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        n,
        stream,
        base_url,
    };
    // Check if model starts with "openai/"
    if request.model.starts_with("openai/") {
        // Strip the "openai/" prefix
        request.model = request
            .model
            .strip_prefix("openai/")
            .context("Failed to strip openai/ prefix")?
            .to_string();

        // Check if streaming is enabled
        if request.stream.is_some() && request.stream.unwrap() {
            // Return streaming response
            let stream = openai_completion_stream(request).await;
            Ok(CompletionResponse::Stream(stream))
        } else {
            // Return complete response
            let response = openai_completion(request).await?;
            Ok(CompletionResponse::Response(response))
        }
    } else {
        // Return error for unsupported providers
        anyhow::bail!(
            "Unsupported provider in model '{}'. Currently only 'openai/' prefix is supported.",
            request.model
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_completion_unsupported_provider() {
        let result = completion(
            "anthropic/claude-3".to_string(),
            vec![Message {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            }],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported provider")
        );
    }

    #[tokio::test]
    async fn test_completion_no_provider() {
        let result = completion(
            "gpt-4o".to_string(),
            vec![Message {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            }],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported provider")
        );
    }

    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_completion_with_openai_prefix() {
        let result = completion(
            "openai/gpt-4o-mini".to_string(),
            vec![Message {
                role: "user".to_string(),
                content: "Say 'test' and nothing else.".to_string(),
            }],
            Some(0.0),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await;
        assert!(result.is_ok());

        match result.unwrap() {
            CompletionResponse::Response(response) => {
                assert!(!response.choices.is_empty());
                assert!(!response.choices[0].message.content.is_empty());
            }
            CompletionResponse::Stream(_) => {
                panic!("Expected Response but got Stream");
            }
        }
    }

    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_completion_with_streaming() {
        let result = completion(
            "openai/gpt-4o-mini".to_string(),
            vec![Message {
                role: "user".to_string(),
                content: "Say 'test' and nothing else.".to_string(),
            }],
            Some(0.0),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            Some(true),
            None,
        )
        .await;
        assert!(result.is_ok());

        match result.unwrap() {
            CompletionResponse::Response(_) => {
                panic!("Expected Stream but got Response");
            }
            CompletionResponse::Stream(_stream) => {
                // Successfully got a stream
                // Note: We can't easily test the stream without making an actual API call
            }
        }
    }
}
