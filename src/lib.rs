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
/// * `request` - The chat completion request with model specified in format "provider/model"
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
/// use ritellm::{completion, CompletionResponse, ChatCompletionRequest, Message};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let request = ChatCompletionRequest {
///         model: "openai/gpt-4o-mini".to_string(),
///         messages: vec![
///             Message {
///                 role: "user".to_string(),
///                 content: "Hello!".to_string(),
///             }
///         ],
///         temperature: Some(0.7),
///         max_tokens: Some(100),
///         top_p: None,
///         frequency_penalty: None,
///         presence_penalty: None,
///         stop: None,
///         n: None,
///         stream: None,
///     };
///
///     match completion(request).await? {
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
/// use ritellm::{completion, CompletionResponse, ChatCompletionRequest, Message};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let request = ChatCompletionRequest {
///         model: "openai/gpt-4o-mini".to_string(),
///         messages: vec![
///             Message {
///                 role: "user".to_string(),
///                 content: "Tell me a story.".to_string(),
///             }
///         ],
///         temperature: Some(0.7),
///         max_tokens: Some(100),
///         top_p: None,
///         frequency_penalty: None,
///         presence_penalty: None,
///         stop: None,
///         n: None,
///         stream: Some(true),
///     };
///
///     match completion(request).await? {
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
pub async fn completion(mut request: ChatCompletionRequest) -> Result<CompletionResponse> {
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
        let request = ChatCompletionRequest {
            model: "anthropic/claude-3".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            n: None,
            stream: None,
        };

        let result = completion(request).await;
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
        let request = ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            n: None,
            stream: None,
        };

        let result = completion(request).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported provider")
        );
    }

    #[tokio::test]
    async fn test_completion_with_openai_prefix() {
        let request = ChatCompletionRequest {
            model: "openai/gpt-4o-mini".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Say 'test' and nothing else.".to_string(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(10),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            n: None,
            stream: None,
        };

        let result = completion(request).await;
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
    async fn test_completion_with_streaming() {
        let request = ChatCompletionRequest {
            model: "openai/gpt-4o-mini".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Say 'test' and nothing else.".to_string(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(10),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            n: None,
            stream: Some(true),
        };

        let result = completion(request).await;
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
