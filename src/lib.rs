use pyo3::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::env;

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
/// * `additional_params` - Optional additional parameters as a JSON string
///
/// # Returns
/// A JSON string containing the API response
///
/// # Environment Variables
/// * `OPENAI_API_KEY` - Required: Your OpenAI API key
#[pyfunction]
#[pyo3(signature = (model, messages, temperature=None, max_tokens=None, base_url=None, additional_params=None))]
fn openai_completion(
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    additional_params: Option<String>,
) -> PyResult<String> {
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
    runtime.block_on(async {
        openai_completion_async(
            api_key,
            model,
            messages,
            temperature,
            max_tokens,
            base_url,
            additional_params,
        )
        .await
    })
}

async fn openai_completion_async(
    api_key: String,
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    additional_params: Option<String>,
) -> PyResult<String> {
    let client = reqwest::Client::new();
    let url = base_url.unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string());

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

    // Parse and return the response
    let response_text = response.text().await.map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to read response: {}", e))
    })?;

    Ok(response_text)
}

/// Gateway function that routes completion requests to the appropriate provider
///
/// This function checks the model prefix and routes to the appropriate completion function.
/// Currently supports:
/// - "openai/" prefix: routes to OpenAI completion
///
/// # Arguments
/// * `model` - The model to use with provider prefix (e.g., "openai/gpt-4")
/// * `messages` - A list of message dictionaries with "role" and "content" keys
/// * `temperature` - Optional sampling temperature (0.0 to 2.0)
/// * `max_tokens` - Optional maximum tokens to generate
/// * `base_url` - Optional base URL for the API
/// * `additional_params` - Optional additional parameters as a JSON string
///
/// # Returns
/// A JSON string containing the API response
///
/// # Errors
/// Returns an error if the provider prefix is not supported
#[pyfunction]
#[pyo3(signature = (model, messages, temperature=None, max_tokens=None, base_url=None, additional_params=None))]
fn completion_gateway(
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    additional_params: Option<String>,
) -> PyResult<String> {
    // Check if model starts with "openai/"
    if model.starts_with("openai/") {
        // Strip the "openai/" prefix
        let actual_model = model.strip_prefix("openai/").unwrap().to_string();

        // Call openai_completion with the actual model name
        openai_completion(
            actual_model,
            messages,
            temperature,
            max_tokens,
            base_url,
            additional_params,
        )
    } else {
        // Provider not supported
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Provider not supported for model: {}",
            model
        )))
    }
}

#[pymodule]
fn ritellm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(completion_gateway, m)?)?;
    Ok(())
}
