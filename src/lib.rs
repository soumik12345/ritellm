pub mod openai;

use anyhow::{Context, Result};

// Re-export commonly used types for convenience
pub use openai::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage, openai_completion,
};

/// Unified completion function that routes to the appropriate provider based on model prefix
///
/// # Arguments
///
/// * `request` - The chat completion request with model specified in format "provider/model"
///
/// # Returns
///
/// * `Result<ChatCompletionResponse>` - The response from the provider or an error
///
/// # Supported Providers
///
/// * `openai/` - Routes to OpenAI API (e.g., "openai/gpt-4o", "openai/gpt-4o-mini")
///
/// # Environment Variables
///
/// * `OPENAI_API_KEY` - Required for OpenAI models
///
/// # Example
///
/// ```no_run
/// use ritellm::{completion, ChatCompletionRequest, Message};
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
///     let response = completion(request).await?;
///     println!("{}", response.choices[0].message.content);
///     Ok(())
/// }
/// ```
pub async fn completion(mut request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
    // Check if model starts with "openai/"
    if request.model.starts_with("openai/") {
        // Strip the "openai/" prefix
        request.model = request
            .model
            .strip_prefix("openai/")
            .context("Failed to strip openai/ prefix")?
            .to_string();

        // Call the OpenAI completion function
        openai_completion(request).await
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

        let response = result.unwrap();
        assert!(!response.choices.is_empty());
        assert!(!response.choices[0].message.content.is_empty());
    }
}
