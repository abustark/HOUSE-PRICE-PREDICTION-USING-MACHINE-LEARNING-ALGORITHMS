//! WebAssembly bindings for client-side zero-latency Bengaluru House Price Prediction.
//! Compiles to wasm32-unknown-unknown with wasm-bindgen.

use wasm_bindgen::prelude::*;
use hpp_core::{clean_area_type, extract_bhk, parse_availability, parse_sqft, validate_input, ModelBundle, RawListingInput};

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Return Wasm engine version string
#[wasm_bindgen]
pub fn wasm_version() -> String {
    format!("{} (hpp-wasm / Rust Wasm32)", hpp_core::SCHEMA_VERSION)
}

/// Parse area string into clean square feet in the browser (supports ranges and Indian units)
#[wasm_bindgen]
pub fn wasm_parse_sqft(raw: &str) -> f64 {
    parse_sqft(raw).unwrap_or(0.0)
}

/// Extract BHK number from text label
#[wasm_bindgen]
pub fn wasm_extract_bhk(raw: &str) -> f64 {
    extract_bhk(raw).unwrap_or(0.0)
}

/// Parse availability string into completion year
#[wasm_bindgen]
pub fn wasm_parse_availability(raw: &str) -> u32 {
    parse_availability(raw)
}

/// Validate property input against historical training bounds
#[wasm_bindgen]
pub fn wasm_validate(input_json: &str) -> Result<String, JsValue> {
    let input: RawListingInput = serde_json::from_str(input_json)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;
    let res = validate_input(&input);
    serde_json::to_string(&res)
        .map_err(|e| JsValue::from_str(&format!("JSON serialization error: {}", e)))
}

/// Run client-side inference directly inside the browser with 0ms network latency
#[wasm_bindgen]
pub fn wasm_predict(input_json: &str) -> Result<String, JsValue> {
    let input: RawListingInput = serde_json::from_str(input_json)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

    let bundle = ModelBundle::embedded();
    let mut response = bundle.infer(&input);
    response.engine = "rust-wasm32-client".to_string();

    serde_json::to_string(&response)
        .map_err(|e| JsValue::from_str(&format!("JSON serialization error: {}", e)))
}
