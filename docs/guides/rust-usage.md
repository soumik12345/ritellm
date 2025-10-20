# Using RiteLLM from Rust

## Important Note

**RiteLLM is designed as a Python library with Rust internals.** The `completion_gateway` function and related code are **not directly callable from pure Rust code**. They are Python functions (using PyO3) meant to be called from Python.

## Architecture Overview

The library uses the following architecture:

```
┌─────────────────────────────────────────┐
│         Python User Code                │
│  (imports ritellm, calls completion())  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       PyO3 Python Bindings              │
│  (completion_gateway in src/lib.rs)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       Rust Implementation               │
│  (openai_completion_async in openai.rs) │
└─────────────────────────────────────────┘
```

## Why Can't You Call It Directly from Rust?

The library is built as a **Python extension module** (`cdylib` crate type) with PyO3's `extension-module` feature. This means:

1. **Python symbols are expected at runtime**: The library expects to be loaded by a Python interpreter which provides symbols like `PyObject_CallNoArgs`, `PyUnicode_FromStringAndSize`, etc.

2. **Not a Rust library**: It's compiled as a shared library (`.so`/`.dylib`/`.dll`) for Python to load, not as a Rust library (`.rlib`) that other Rust code can link against.

3. **PyO3 types everywhere**: The functions use PyO3-specific types (`PyResult`, `Python`, `PyAny`) that only make sense in the context of Python integration.

## Understanding the Code

Here's what the `completion_gateway` function does (from `src/lib.rs`):

```rust
#[pyfunction]  // This makes it callable from Python only
#[pyo3(signature = (model, messages, temperature=None, ...))]
fn completion_gateway(
    py: Python,  // Python GIL guard - only available in Python context
    model: String,
    messages: Vec<HashMap<String, String>>,
    // ... more parameters
) -> PyResult<Py<PyAny>> {  // Returns Python object
    if model.starts_with("openai/") {
        let actual_model = model.strip_prefix("openai/").unwrap().to_string();
        openai_completion(py, actual_model, messages, ...)
    } else {
        Err(PyErr::new::<PyValueError, _>(...))
    }
}
```

The key indicators that this is Python-only:
- `#[pyfunction]` macro - marks it as a Python function
- `py: Python` parameter - a handle to the Python interpreter
- `PyResult` return type - wraps errors for Python
- `Py<PyAny>` - a Python object reference

## How to Use RiteLLM from Rust

You have **two options**:

### Option 1: Use the Python API from Rust (Recommended)

If you want to use RiteLLM from a Rust application, call the Python API using a Python embedding library:

```toml
[dependencies]
pyo3 = "0.27"
```

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Import the ritellm module
        let ritellm = py.import("ritellm")?;
        
        // Create messages
        let messages = PyList::new(py, vec![
            {
                let msg = PyDict::new(py);
                msg.set_item("role", "user")?;
                msg.set_item("content", "Hello from Rust!")?;
                msg
            }
        ])?;
        
        // Call completion
        let response = ritellm.call_method1(
            "completion",
            (
                "openai/gpt-3.5-turbo",
                messages,
            )
        )?;
        
        // Process response
        println!("Response: {:?}", response);
        Ok(())
    })
}
```

### Option 2: Reimplement the Logic in Pure Rust

If you need a pure Rust solution without Python, you would need to reimplement the logic using the same dependencies:

```toml
[dependencies]
reqwest = { version = "0.12", features = ["json", "rustls-tls", "stream"] }
serde_json = "1.0"
tokio = { version = "1.40", features = ["full"] }
```

```rust
use serde_json::json;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("OPENAI_API_KEY")?;
    let client = reqwest::Client::new();
    
    let body = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello from pure Rust!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    });
    
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;
    
    let response_text = response.text().await?;
    println!("Response: {}", response_text);
    
    Ok(())
}
```

This approach gives you full control but loses the benefits of RiteLLM's abstraction layer.

## Viewing the Rust Implementation

You can view the Rust implementation to understand how it works:

- **Gateway logic**: `src/lib.rs` - Routes requests to appropriate providers
- **OpenAI implementation**: `src/openai.rs` - Handles OpenAI API calls
  - `openai_completion()` - Python-facing function
  - `openai_completion_async()` - Async implementation

The core async function (`openai_completion_async`) contains the actual HTTP logic:
```rust
pub async fn openai_completion_async(
    api_key: String,
    model: String,
    messages: Vec<HashMap<String, String>>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    base_url: Option<String>,
    stream: Option<bool>,
    additional_params: Option<String>,
) -> PyResult<CompletionResult>
```

However, even this function returns `PyResult`, so it's still tied to Python.

## Future Plans

To make RiteLLM usable from pure Rust, the project would need:

1. **Separate the core logic** into a pure Rust crate (e.g., `ritellm-core`)
2. **Create Python bindings** in a separate crate (e.g., `ritellm-python`)
3. **Use feature flags** to conditionally compile with/without PyO3

This would allow both Rust and Python consumers, but is not currently implemented.

## Summary

- ✅ **Use from Python**: Yes, that's the primary use case
- ❌ **Use directly from Rust**: No, not currently possible
- ⚠️ **Embed Python in Rust**: Yes, possible but adds complexity
- ✅ **Learn from the Rust code**: Yes, you can read and understand the implementation
- ✅ **Reimplement in pure Rust**: Yes, the code shows you how

For most use cases, we recommend using RiteLLM from Python as designed, or creating a pure Rust implementation based on the patterns shown in this codebase.
