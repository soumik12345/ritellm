mod openai;

use pyo3::prelude::*;
use std::collections::HashMap;

use openai::openai_completion;

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

#[pymodule]
fn ritellm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(completion_gateway, m)?)?;
    Ok(())
}
