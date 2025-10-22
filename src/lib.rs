pub mod openai;

use pyo3::prelude::*;
use pyo3::types::PyString;
use std::collections::HashMap;

use openai::{CompletionResult, openai_completion, openai_completion_async};

// Re-export for Rust usage
pub use openai::CompletionResult as CompletionResultExport;

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
/// * `stream` - Optional boolean to enable streaming responses
/// * `additional_params` - Optional additional parameters as a JSON string
///
/// # Returns
/// A JSON string containing the API response, or a StreamingResponse iterator if stream=True
///
/// # Errors
/// Returns an error if the provider prefix is not supported
#[pyfunction]
#[pyo3(signature = (model, messages, temperature=None, max_tokens=None, base_url=None, stream=None, additional_params=None))]
fn completion_gateway(
    py: Python,
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    stream: Option<bool>,
    additional_params: Option<String>,
) -> PyResult<Py<PyAny>> {
    // Check if model starts with "openai/"
    if model.starts_with("openai/") {
        // Strip the "openai/" prefix
        let actual_model = model.strip_prefix("openai/").unwrap().to_string();

        // Call openai_completion with the actual model name
        openai_completion(
            py,
            actual_model,
            messages,
            temperature,
            max_tokens,
            base_url,
            stream,
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

/// Async gateway function that routes completion requests to the appropriate provider
///
/// This is the async version of completion_gateway that can be awaited from Python.
/// It checks the model prefix and routes to the appropriate async completion function.
/// Currently supports:
/// - "openai/" prefix: routes to OpenAI completion
///
/// # Arguments
/// * `model` - The model to use with provider prefix (e.g., "openai/gpt-4")
/// * `messages` - A list of message dictionaries with "role" and "content" keys
/// * `temperature` - Optional sampling temperature (0.0 to 2.0)
/// * `max_tokens` - Optional maximum tokens to generate
/// * `base_url` - Optional base URL for the API
/// * `stream` - Optional boolean to enable streaming responses
/// * `additional_params` - Optional additional parameters as a JSON string
///
/// # Returns
/// A JSON string containing the API response, or a StreamingResponse iterator if stream=True
///
/// # Errors
/// Returns an error if the provider prefix is not supported
#[pyfunction]
#[pyo3(signature = (model, messages, temperature=None, max_tokens=None, base_url=None, stream=None, additional_params=None))]
fn async_completion_gateway(
    py: Python,
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    stream: Option<bool>,
    additional_params: Option<String>,
) -> PyResult<Py<PyAny>> {
    // Check if model starts with "openai/"
    if model.starts_with("openai/") {
        // Strip the "openai/" prefix
        let actual_model = model.strip_prefix("openai/").unwrap().to_string();

        // Load API key from environment variable
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
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
                actual_model,
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
                let streaming_response = openai::StreamingResponse::new(chunks);
                Ok(Py::new(py, streaming_response)?.into_any())
            }
        }
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
    m.add_function(wrap_pyfunction!(async_completion_gateway, m)?)?;
    Ok(())
}
