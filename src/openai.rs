use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// Message structure for chat completions
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Request structure for OpenAI chat completions
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Choice structure in the response
#[derive(Debug, Deserialize, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Usage statistics in the response
#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Response structure from OpenAI chat completions
#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Error structure from OpenAI API
#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    pub error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Creates a chat completion using the OpenAI API
///
/// # Arguments
///
/// * `request` - The chat completion request containing model, messages, and optional parameters
///
/// # Returns
///
/// * `Result<ChatCompletionResponse>` - The response from OpenAI API or an error
///
/// # Environment Variables
///
/// * `OPENAI_API_KEY` - Required. Your OpenAI API key
///
/// # Example
///
/// ```no_run
/// use ritellm::openai::{openai_completion, ChatCompletionRequest, Message};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let request = ChatCompletionRequest {
///         model: "gpt-4o".to_string(),
///         messages: vec![
///             Message {
///                 role: "user".to_string(),
///                 content: "Hello, how are you?".to_string(),
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
///     let response = openai_completion(request).await?;
///     println!("Response: {}", response.choices[0].message.content);
///     Ok(())
/// }
/// ```
pub async fn openai_completion(request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
    // Get API key from environment
    let api_key =
        env::var("OPENAI_API_KEY").context("OPENAI_API_KEY environment variable not set")?;

    // API endpoint
    let url = "https://api.openai.com/v1/chat/completions";

    // Create HTTP client
    let client = Client::new();

    // Send POST request
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .context("Failed to send request to OpenAI API")?;

    // Check if request was successful
    if response.status().is_success() {
        let completion_response: ChatCompletionResponse = response
            .json()
            .await
            .context("Failed to parse successful response from OpenAI API")?;
        Ok(completion_response)
    } else {
        // Try to parse error response
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        // Try to parse as OpenAI error format
        if let Ok(error_response) = serde_json::from_str::<OpenAIError>(&error_text) {
            anyhow::bail!(
                "OpenAI API error ({}): {} - {}",
                status,
                error_response.error.error_type,
                error_response.error.message
            );
        } else {
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_openai_completion() {
        let request = ChatCompletionRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Say 'Hello, World!' and nothing else.".to_string(),
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

        let response = openai_completion(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(!response.choices.is_empty());
        assert!(!response.choices[0].message.content.is_empty());
    }
}
