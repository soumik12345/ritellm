use anyhow::{Context, Result};
use futures::Stream;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use std::env;
use std::pin::Pin;

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

// ============= Streaming Types =============

/// Delta structure for streaming responses - contains partial message content
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionStreamResponseDelta {
    /// The role of the author (only present in the first chunk)
    pub role: Option<String>,
    /// The content chunk from the streaming response
    pub content: Option<String>,
}

/// Choice structure for streaming responses
#[derive(Debug, Deserialize, Clone)]
pub struct ChatChoiceStream {
    /// The index of the choice
    pub index: u32,
    /// The delta containing the content chunk
    pub delta: ChatCompletionStreamResponseDelta,
    /// The reason the model stopped generating tokens (only in final chunk)
    pub finish_reason: Option<String>,
}

/// Streaming response chunk from OpenAI chat completions
#[derive(Debug, Deserialize)]
pub struct ChatCompletionStreamResponse {
    /// Unique identifier for the chat completion (same across all chunks)
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Unix timestamp of when the completion was created
    pub created: u64,
    /// The model used
    pub model: String,
    /// Array of choices (usually one element)
    pub choices: Vec<ChatChoiceStream>,
}

/// Type alias for the streaming response
pub type ChatCompletionResponseStream =
    Pin<Box<dyn Stream<Item = Result<ChatCompletionStreamResponse>> + Send>>;

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
    // Check if stream is enabled
    if request.stream.is_some() && request.stream.unwrap() {
        anyhow::bail!("When stream is true, use openai_completion_stream instead");
    }

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

/// Creates a streaming chat completion using the OpenAI API
///
/// Returns a stream of response chunks that can be iterated over to receive
/// partial message deltas as they become available from the API.
///
/// # Arguments
///
/// * `request` - The chat completion request. The `stream` field will be automatically set to `true`.
///
/// # Returns
///
/// * `ChatCompletionResponseStream` - A stream of response chunks
///
/// # Environment Variables
///
/// * `OPENAI_API_KEY` - Required. Your OpenAI API key
///
/// # Example
///
/// ```no_run
/// use ritellm::openai::{openai_completion_stream, ChatCompletionRequest, Message};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut request = ChatCompletionRequest {
///         model: "gpt-4o-mini".to_string(),
///         messages: vec![
///             Message {
///                 role: "user".to_string(),
///                 content: "Tell me a short story.".to_string(),
///             }
///         ],
///         temperature: Some(0.7),
///         max_tokens: Some(100),
///         top_p: None,
///         frequency_penalty: None,
///         presence_penalty: None,
///         stop: None,
///         n: None,
///         stream: None,  // Will be set to true automatically
///     };
///
///     let mut stream = openai_completion_stream(request).await;
///     
///     while let Some(result) = stream.next().await {
///         match result {
///             Ok(response) => {
///                 if let Some(choice) = response.choices.first() {
///                     if let Some(content) = &choice.delta.content {
///                         print!("{}", content);
///                     }
///                 }
///             }
///             Err(e) => eprintln!("Error: {}", e),
///         }
///     }
///     
///     Ok(())
/// }
/// ```
pub async fn openai_completion_stream(
    mut request: ChatCompletionRequest,
) -> ChatCompletionResponseStream {
    // Ensure stream is set to true
    request.stream = Some(true);

    // Get API key from environment
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            return Box::pin(futures::stream::once(async {
                Err(anyhow::anyhow!(
                    "OPENAI_API_KEY environment variable not set"
                ))
            }));
        }
    };

    // API endpoint
    let url = "https://api.openai.com/v1/chat/completions";

    // Create HTTP client
    let client = Client::new();

    // Build the request with EventSource support
    let event_source = match client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .eventsource()
    {
        Ok(es) => es,
        Err(e) => {
            return Box::pin(futures::stream::once(async move {
                Err(anyhow::anyhow!("Failed to create event source: {}", e))
            }));
        }
    };

    // Create the stream processing logic
    create_stream(event_source).await
}

/// Internal helper function to create and process the SSE stream
async fn create_stream(mut event_source: EventSource) -> ChatCompletionResponseStream {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(ev) = futures::StreamExt::next(&mut event_source).await {
            match ev {
                Err(e) => {
                    let error_msg = format!("EventSource error: {}", e);
                    if tx.send(Err(anyhow::anyhow!(error_msg))).is_err() {
                        // Receiver dropped, stop processing
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Message(message) => {
                        // Check for [DONE] message which signals end of stream
                        if message.data == "[DONE]" {
                            break;
                        }

                        // Parse the JSON chunk
                        let response = match serde_json::from_str::<ChatCompletionStreamResponse>(
                            &message.data,
                        ) {
                            Ok(output) => Ok(output),
                            Err(e) => Err(anyhow::anyhow!(
                                "Failed to parse stream response: {} - Data: {}",
                                e,
                                message.data
                            )),
                        };

                        if tx.send(response).is_err() {
                            // Receiver dropped, stop processing
                            break;
                        }
                    }
                    Event::Open => {
                        // Connection opened, continue
                        continue;
                    }
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
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
