use pyo3::prelude::*;
use pyo3::types::PyString;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::env;
use futures_util::stream::StreamExt;

/// Result type for completion functions
pub enum CompletionResult {
    Text(String),
    Stream(Vec<String>),
}

/// A Python iterator that yields streaming response chunks
#[pyclass]
pub struct StreamingResponse {
    chunks: Vec<String>,
    index: usize,
}

#[pymethods]
impl StreamingResponse {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<String> {
        if slf.index < slf.chunks.len() {
            let chunk = slf.chunks[slf.index].clone();
            slf.index += 1;
            Some(chunk)
        } else {
            None
        }
    }
}

/// Sends a POST request to OpenAI's chat completion endpoint and returns the result
///
/// The API key is loaded from the `OPENAI_API_KEY` environment variable.
///
/// # Arguments
/// * `model` - The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
/// * `messages` - A list of message dictionaries with "role" and "content" keys
/// * `temperature` - Optional sampling temperature (0.0 to 2.0)
/// * `max_tokens` - Optional maximum tokens to generate
/// * `base_url` - Optional base URL for the OpenAI API
/// * `stream` - Optional boolean to enable streaming responses
/// * `additional_params` - Optional additional parameters as a JSON string
///
/// # Returns
/// A JSON string containing the API response, or a StreamingResponse iterator if stream=True
///
/// # Environment Variables
/// * `OPENAI_API_KEY` - Required: Your OpenAI API key
#[pyfunction]
#[pyo3(signature = (model, messages, temperature=None, max_tokens=None, base_url=None, stream=None, additional_params=None))]
pub fn openai_completion(
    py: Python,
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    stream: Option<bool>,
    additional_params: Option<String>,
) -> PyResult<Py<PyAny>> {
    // Load API key from environment variable
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "OPENAI_API_KEY environment variable not set",
        )
    })?;

    // Create a Tokio runtime to run async code
    let runtime = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create runtime: {}",
            e
        ))
    })?;

    // Run the async completion function
    let result = runtime.block_on(async {
        openai_completion_async(
            api_key,
            model,
            messages,
            temperature,
            max_tokens,
            base_url,
            stream,
            additional_params,
        )
        .await
    })?;

    // Convert the result to a Python object
    match result {
        CompletionResult::Text(text) => {
            let py_str = PyString::new(py, &text);
            Ok(py_str.unbind().into_any())
        }
        CompletionResult::Stream(chunks) => {
            let streaming_response = StreamingResponse { chunks, index: 0 };
            Ok(Py::new(py, streaming_response)?.into_any())
        }
    }
}

pub async fn openai_completion_async(
    api_key: String,
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    stream: Option<bool>,
    additional_params: Option<String>,
) -> PyResult<CompletionResult> {
    let client = reqwest::Client::new();
    let url = base_url.unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string());
    let is_streaming = stream.unwrap_or(false);

    // Build the request body
    let mut body = json!({
        "model": model,
        "messages": messages,
    });

    // Add optional parameters
    if let Some(temp) = temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(tokens) = max_tokens {
        body["max_tokens"] = json!(tokens);
    }
    if is_streaming {
        body["stream"] = json!(true);
    }

    // Merge additional parameters if provided
    if let Some(params_str) = additional_params {
        let additional: Value = serde_json::from_str(&params_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid JSON in additional_params: {}",
                e
            ))
        })?;

        if let (Some(body_obj), Some(additional_obj)) =
            (body.as_object_mut(), additional.as_object())
        {
            for (key, value) in additional_obj {
                body_obj.insert(key.clone(), value.clone());
            }
        }
    }

    // Send the POST request
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Request failed: {}", e))
        })?;

    // Check if the request was successful
    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    // Handle streaming vs non-streaming responses
    if is_streaming {
        let mut chunks = Vec::new();
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(item) = stream.next().await {
            let bytes = item.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to read stream: {}",
                    e
                ))
            })?;

            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Process SSE format (Server-Sent Events)
            while let Some(pos) = buffer.find("\n\n") {
                let line = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                if line.is_empty() {
                    continue;
                }

                // Parse SSE lines
                for sse_line in line.lines() {
                    if sse_line.starts_with("data: ") {
                        let data = &sse_line[6..];
                        if data == "[DONE]" {
                            break;
                        }
                        // Store the JSON chunk
                        chunks.push(data.to_string());
                    }
                }
            }
        }

        Ok(CompletionResult::Stream(chunks))
    } else {
        // Parse and return the response
        let response_text = response.text().await.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to read response: {}", e))
        })?;

        Ok(CompletionResult::Text(response_text))
    }
}
