//! Core domain models, feature normalization, boundary validation, and fast inference engine
//! for Bengaluru House Price Prediction.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

pub const SCHEMA_VERSION: &str = "2.1.0";
pub const MISSING: &str = "__MISSING__";

pub const SUPPORTED_AREA_TYPES: [&str; 4] = [
    "Built-up Area",
    "Carpet Area",
    "Plot Area",
    "Super built-up Area",
];

pub const FEATURES: [&str; 8] = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "area_type",
    "completion_year",
    "location",
    "society",
];

// Area conversion multipliers to square feet
static SQFT_PER_UNIT: LazyLock<HashMap<&'static str, f64>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("sq. meter", 10.7639);
    m.insert("sq meter", 10.7639);
    m.insert("sqm", 10.7639);
    m.insert("square meter", 10.7639);
    m.insert("sq. yards", 1.19599);
    m.insert("sq yards", 1.19599);
    m.insert("sqyd", 1.19599);
    m.insert("square yard", 1.19599);
    m.insert("perch", 272.25);
    m.insert("acres", 43560.0);
    m.insert("acre", 43560.0);
    m.insert("guntha", 1011.71);
    m.insert("ground", 10890.0);
    m
});

static RE_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^([\d.]+)\s*([- \x{2013}to]+)\s*([\d.]+)$").unwrap()
});

static RE_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^([\d.]+)\s*([A-Za-z. ]+)$").unwrap()
});

static RE_DIGITS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d+)").unwrap()
});

static RE_DATE_AVAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{2}-[A-Za-z]{3}$").unwrap()
});

static RE_WHITESPACE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\s+").unwrap()
});

/// Parse a raw `total_sqft` value into clean square feet.
/// Handles plain floats, hyphen ranges (returns midpoint), and alternate Indian land units.
pub fn parse_sqft(raw: &str) -> Option<f64> {
    let text = raw.trim().replace(',', "");
    if text.is_empty() {
        return None;
    }

    // 1. Check range match (e.g. "1200 - 1400", "2100 to 2850")
    if let Some(caps) = RE_RANGE.captures(&text) {
        if let (Ok(first), Ok(second)) = (caps[1].parse::<f64>(), caps[3].parse::<f64>()) {
            if first > 0.0 && first <= second {
                return Some((first + second) / 2.0);
            }
        }
        return None;
    }

    // 2. Check plain float
    if let Ok(num) = text.parse::<f64>() {
        return if num > 0.0 { Some(num) } else { None };
    }

    // 3. Check unit conversion (e.g. "34.46Sq. Meter", "1000Sq. Yards", "5Acres")
    if let Some(caps) = RE_UNIT.captures(&text) {
        if let Ok(num) = caps[1].parse::<f64>() {
            if num <= 0.0 {
                return None;
            }
            let unit = caps[2].trim().to_lowercase();
            if let Some(&multiplier) = SQFT_PER_UNIT.get(unit.as_str()) {
                return Some(num * multiplier);
            }
        }
    }

    None
}

/// Extract BHK bedroom count from labels like "2 BHK", "4 Bedroom", "Studio", "1 RK".
pub fn extract_bhk(raw: &str) -> Option<f64> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }

    let lower = text.to_lowercase();
    if lower.contains("studio") {
        return Some(0.0);
    }
    if lower.contains("rk") {
        return Some(1.0);
    }

    if let Some(caps) = RE_DIGITS.captures(text) {
        if let Ok(bhk) = caps[1].parse::<f64>() {
            return Some(bhk);
        }
    }

    None
}

/// Parse availability string into expected completion year (0 for Ready To Move).
pub fn parse_availability(raw: &str) -> u32 {
    let text = raw.trim();
    if RE_DATE_AVAIL.is_match(text) {
        if let Ok(yy) = text[..2].parse::<u32>() {
            return 2000 + yy;
        }
    }
    0
}

/// Normalize area type whitespace and validate against supported list.
pub fn clean_area_type(raw: &str) -> Option<String> {
    let cleaned = RE_WHITESPACE.replace_all(raw.trim(), " ").to_string();
    if SUPPORTED_AREA_TYPES.iter().any(|&s| s == cleaned) {
        Some(cleaned)
    } else {
        None
    }
}

/// Clean categorical string: strips spaces, normalizes empty/nan/none to `__MISSING__`.
pub fn clean_category(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("nan")
        || trimmed.eq_ignore_ascii_case("none")
        || trimmed.eq_ignore_ascii_case("null")
    {
        return MISSING.to_string();
    }
    RE_WHITESPACE.replace_all(trimmed, " ").to_string()
}

/// Standardized validation input payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawListingInput {
    pub area_type: String,
    pub location: String,
    #[serde(default)]
    pub society: String,
    pub bhk: i32,
    pub total_sqft: f64,
    pub bath: i32,
    pub balcony: i32,
    #[serde(default)]
    pub completion_year: i32,
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_model() -> String {
    "random_forest".to_string()
}

/// Diagnostic warnings and validation errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validate raw input against training domain constraints.
pub fn validate_input(input: &RawListingInput) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check area type
    if clean_area_type(&input.area_type).is_none() {
        errors.push(format!(
            "Unsupported area type '{}'. Supported: {}",
            input.area_type,
            SUPPORTED_AREA_TYPES.join(", ")
        ));
    }

    // Check total sqft
    if input.total_sqft < 100.0 || input.total_sqft > 10000.0 {
        errors.push(format!(
            "total_sqft ({}) outside valid range [100.0, 10000.0]",
            input.total_sqft
        ));
    } else if input.total_sqft > 4000.0 {
        warnings.push(format!(
            "Large property: total_sqft ({}) is in upper 2% of training distribution",
            input.total_sqft
        ));
    }

    // Check bathrooms
    if input.bath < 1 || input.bath > 12 {
        errors.push(format!(
            "bath ({}) outside valid range [1, 12]",
            input.bath
        ));
    }

    // Check balconies
    if input.balcony < 0 || input.balcony > 5 {
        errors.push(format!(
            "balcony ({}) outside valid range [0, 5]",
            input.balcony
        ));
    }

    // Check BHK
    if input.bhk < 0 || input.bhk > 14 {
        errors.push(format!(
            "bhk ({}) outside valid range [0, 14]",
            input.bhk
        ));
    }

    // Physical sanity: bath > bhk + 2
    if input.bath > input.bhk + 2 && input.bhk > 0 {
        warnings.push(format!(
            "Unusual layout: {} bathrooms for a {} BHK property",
            input.bath, input.bhk
        ));
    }

    ValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
    }
}

// ---------------------------------------------------------------------------
// Model Weights, Feature Scaling & Fast Inference
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalerParams {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearModelParams {
    pub coef: Vec<f64>,
    pub intercept: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    pub schema_version: String,
    pub feature_order: Vec<String>,
    pub bhk_median: i32,
    pub scaler: ScalerParams,
    pub linear: LinearModelParams,
    pub ridge: LinearModelParams,
    pub location_freq: HashMap<String, f64>,
    pub society_freq: HashMap<String, f64>,
}

// Embedded production weights JSON bundle compiled directly into the binary
static EMBEDDED_BUNDLE_JSON: &str = include_str!("../../../models/weights.json");
static EMBEDDED_BUNDLE: LazyLock<ModelBundle> = LazyLock::new(|| {
    serde_json::from_str(EMBEDDED_BUNDLE_JSON).expect("Failed to deserialize embedded weights.json")
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub price_lakh: f64,
    pub price_range_lakh: (f64, f64),
    pub price_per_sqft: f64,
    pub model_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullInferenceResponse {
    pub success: bool,
    pub input_summary: RawListingInput,
    pub prediction: PredictionResult,
    pub linear_lakh: f64,
    pub ridge_lakh: f64,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub engine: String,
}

impl ModelBundle {
    /// Retrieve static reference to embedded weights bundle
    pub fn embedded() -> &'static ModelBundle {
        &EMBEDDED_BUNDLE
    }

    /// Load weights from a raw JSON string
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Encode input features into raw numeric vector [total_sqft, bath, balcony, bhk, area_type, completion_year, location, society]
    pub fn encode_features(&self, input: &RawListingInput) -> [f64; 8] {
        let area_idx = match clean_area_type(&input.area_type).as_deref() {
            Some("Built-up Area") => 0.0,
            Some("Carpet Area") => 1.0,
            Some("Plot Area") => 2.0,
            Some("Super built-up Area") => 3.0,
            _ => 3.0,
        };

        let loc_clean = clean_category(&input.location);
        let loc_val = self.location_freq.get(&loc_clean).copied().unwrap_or(0.0);

        let soc_clean = clean_category(&input.society);
        let soc_val = self.society_freq.get(&soc_clean).copied().unwrap_or(0.0);

        let bhk_val = if input.bhk > 0 {
            input.bhk as f64
        } else {
            self.bhk_median as f64
        };

        [
            input.total_sqft,
            input.bath as f64,
            input.balcony as f64,
            bhk_val,
            area_idx,
            input.completion_year as f64,
            loc_val,
            soc_val,
        ]
    }

    /// Standardize raw features using StandardScaler (mean & scale)
    pub fn scale_features(&self, raw: &[f64; 8]) -> [f64; 8] {
        let mut scaled = [0.0; 8];
        for i in 0..8 {
            scaled[i] = (raw[i] - self.scaler.mean[i]) / self.scaler.scale[i];
        }
        scaled
    }

    /// Predict price in Lakhs using Ridge or Linear regression weights
    pub fn predict_model(&self, scaled: &[f64; 8], model_name: &str) -> f64 {
        let (coef, intercept) = if model_name.eq_ignore_ascii_case("ridge") {
            (&self.ridge.coef, self.ridge.intercept)
        } else {
            (&self.linear.coef, self.linear.intercept)
        };

        let mut dot = intercept;
        for i in 0..8 {
            dot += coef[i] * scaled[i];
        }

        // Training target was y = log(1 + price_lakh)
        let pred_lakh = dot.exp() - 1.0;
        if pred_lakh < 1.0 {
            1.0
        } else {
            pred_lakh
        }
    }

    /// Full inference pipeline matching Python API semantics
    pub fn infer(&self, input: &RawListingInput) -> FullInferenceResponse {
        let v_res = validate_input(input);
        if !v_res.is_valid {
            return FullInferenceResponse {
                success: false,
                input_summary: input.clone(),
                prediction: PredictionResult {
                    price_lakh: 0.0,
                    price_range_lakh: (0.0, 0.0),
                    price_per_sqft: 0.0,
                    model_used: input.model.clone(),
                },
                linear_lakh: 0.0,
                ridge_lakh: 0.0,
                warnings: v_res.warnings,
                errors: v_res.errors,
                engine: "rust-hpp-core".to_string(),
            };
        }

        let raw_feats = self.encode_features(input);
        let scaled_feats = self.scale_features(&raw_feats);

        let lin_lakh = self.predict_model(&scaled_feats, "linear");
        let rdg_lakh = self.predict_model(&scaled_feats, "ridge");

        let active_lakh = if input.model.eq_ignore_ascii_case("ridge") {
            rdg_lakh
        } else {
            lin_lakh
        };

        // Historical MAE is ~0.315 log scale for linear/ridge
        let error_factor = (0.315f64).exp();
        let lower = (active_lakh / error_factor).max(1.0);
        let upper = active_lakh * error_factor;

        let price_per_sqft = if input.total_sqft > 0.0 {
            (active_lakh * 100_000.0) / input.total_sqft
        } else {
            0.0
        };

        FullInferenceResponse {
            success: true,
            input_summary: input.clone(),
            prediction: PredictionResult {
                price_lakh: (active_lakh * 100.0).round() / 100.0,
                price_range_lakh: (
                    (lower * 100.0).round() / 100.0,
                    (upper * 100.0).round() / 100.0,
                ),
                price_per_sqft: (price_per_sqft * 100.0).round() / 100.0,
                model_used: input.model.clone(),
            },
            linear_lakh: (lin_lakh * 100.0).round() / 100.0,
            ridge_lakh: (rdg_lakh * 100.0).round() / 100.0,
            warnings: v_res.warnings,
            errors: Vec::new(),
            engine: "rust-hpp-core".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqft_plain() {
        assert_eq!(parse_sqft("1200"), Some(1200.0));
        assert_eq!(parse_sqft("1,500.50"), Some(1500.5));
    }

    #[test]
    fn test_sqft_range() {
        assert_eq!(parse_sqft("1000 - 1200"), Some(1100.0));
        assert_eq!(parse_sqft("2000 to 3000"), Some(2500.0));
    }

    #[test]
    fn test_sqft_units() {
        let sq_m = parse_sqft("100Sq. Meter").unwrap();
        assert!((sq_m - 1076.39).abs() < 0.1);

        let acres = parse_sqft("1Acres").unwrap();
        assert!((acres - 43560.0).abs() < 0.1);
    }

    #[test]
    fn test_bhk_extraction() {
        assert_eq!(extract_bhk("2 BHK"), Some(2.0));
        assert_eq!(extract_bhk("4 Bedroom"), Some(4.0));
        assert_eq!(extract_bhk("Studio"), Some(0.0));
        assert_eq!(extract_bhk("1 RK"), Some(1.0));
    }

    #[test]
    fn test_availability_parsing() {
        assert_eq!(parse_availability("Ready To Move"), 0);
        assert_eq!(parse_availability("19-Dec"), 2019);
        assert_eq!(parse_availability("18-May"), 2018);
    }

    #[test]
    fn test_embedded_model_inference() {
        let bundle = ModelBundle::embedded();
        let input = RawListingInput {
            area_type: "Super built-up Area".to_string(),
            location: "Whitefield".to_string(),
            society: "".to_string(),
            bhk: 2,
            total_sqft: 1200.0,
            bath: 2,
            balcony: 1,
            completion_year: 0,
            model: "linear".to_string(),
        };

        let res = bundle.infer(&input);
        assert!(res.success);
        assert!(res.errors.is_empty());
        // Expected linear prediction is ~56.81 Lakhs
        assert!((res.linear_lakh - 56.81).abs() < 0.05);
        assert!((res.ridge_lakh - 56.81).abs() < 0.05);
    }
}
