//! PyO3 native extension module `hpp_fast` for high-throughput Python ETL and inference.

use hpp_core::{
    clean_area_type as core_clean_area_type, clean_category as core_clean_category,
    extract_bhk as core_extract_bhk, parse_availability as core_parse_availability,
    parse_sqft as core_parse_sqft, validate_input as core_validate_input, ModelBundle,
    RawListingInput,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn parse_sqft(raw: &str) -> Option<f64> {
    core_parse_sqft(raw)
}

#[pyfunction]
fn extract_bhk(raw: &str) -> Option<f64> {
    core_extract_bhk(raw)
}

#[pyfunction]
fn parse_availability(raw: &str) -> u32 {
    core_parse_availability(raw)
}

#[pyfunction]
fn clean_area_type(raw: &str) -> Option<String> {
    core_clean_area_type(raw)
}

#[pyfunction]
fn clean_category(raw: &str) -> String {
    core_clean_category(raw)
}

#[pyfunction]
fn validate_input_json(json_str: &str) -> PyResult<String> {
    let input: RawListingInput = serde_json::from_str(json_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let res = core_validate_input(&input);
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
}

#[pyfunction]
#[pyo3(signature = (total_sqft, bath, balcony, bhk, area_type, completion_year, location, society, model="linear"))]
fn predict_fast(
    total_sqft: f64,
    bath: i32,
    balcony: i32,
    bhk: i32,
    area_type: &str,
    completion_year: i32,
    location: &str,
    society: &str,
    model: &str,
) -> PyResult<f64> {
    let input = RawListingInput {
        area_type: area_type.to_string(),
        location: location.to_string(),
        society: society.to_string(),
        bhk,
        total_sqft,
        bath,
        balcony,
        completion_year,
        model: model.to_string(),
    };

    let bundle = ModelBundle::embedded();
    let res = bundle.infer(&input);
    if !res.success {
        return Err(PyValueError::new_err(res.errors.join("; ")));
    }
    Ok(res.prediction.price_lakh)
}

#[pyfunction]
fn infer_json(json_str: &str) -> PyResult<String> {
    let input: RawListingInput = serde_json::from_str(json_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let bundle = ModelBundle::embedded();
    let mut res = bundle.infer(&input);
    res.engine = "rust-pyo3-native".to_string();
    serde_json::to_string(&res)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
}

#[pymodule]
fn hpp_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_sqft, m)?)?;
    m.add_function(wrap_pyfunction!(extract_bhk, m)?)?;
    m.add_function(wrap_pyfunction!(parse_availability, m)?)?;
    m.add_function(wrap_pyfunction!(clean_area_type, m)?)?;
    m.add_function(wrap_pyfunction!(clean_category, m)?)?;
    m.add_function(wrap_pyfunction!(validate_input_json, m)?)?;
    m.add_function(wrap_pyfunction!(predict_fast, m)?)?;
    m.add_function(wrap_pyfunction!(infer_json, m)?)?;
    m.add("__version__", hpp_core::SCHEMA_VERSION)?;
    Ok(())
}
