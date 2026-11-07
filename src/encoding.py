"""
Shared feature engineering, validation, and preprocessing rules.

Single source of truth for:
  - Supported feature schema and category definitions
  - Raw value parsing (total_sqft, bhk, availability)
  - Physical boundary validation and out-of-domain warning diagnostics
  - Leakage-safe DataPreprocessor (fits imputation and encodings fold-locally)
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

SCHEMA_VERSION = "2.1.0"

# Sentinel for missing high-cardinality categories. A dedicated category is
# used instead of 0 so that "no data" is never confused with a real value.
MISSING = "__MISSING__"

# Exact column order the models were trained with.
FEATURES = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "area_type",
    "completion_year",
    "location",
    "society",
]

# Supported area type categories.
# In raw dataset, spacing varies (e.g. 'Plot  Area' vs 'Plot Area');
# clean_area_type normalizes internal whitespace.
SUPPORTED_AREA_TYPES = (
    "Built-up Area",
    "Carpet Area",
    "Plot Area",
    "Super built-up Area",
)

AREA_TYPE_TO_INT = {cat: idx for idx, cat in enumerate(SUPPORTED_AREA_TYPES)}
INT_TO_AREA_TYPE = {idx: cat for idx, cat in enumerate(SUPPORTED_AREA_TYPES)}

# Training support bounds (sane physical domain of the historical dataset).
VALIDATION_CONSTRAINTS = {
    "total_sqft": {"min": 100.0, "max": 10000.0, "warning_high": 4000.0},
    "bath": {"min": 1, "max": 12, "warning_high": 6},
    "balcony": {"min": 0, "max": 5},
    "bhk": {"min": 0, "max": 14, "warning_high": 6},
    "completion_year": {"min": 0, "max": 2030},
}

# Multipliers that convert common Indian area units to square feet.
SQFT_PER_UNIT = {
    "sq. meter": 10.7639,
    "sq meter": 10.7639,
    "sqm": 10.7639,
    "square meter": 10.7639,
    "sq. yards": 1.19599,
    "sq yards": 1.19599,
    "sqyd": 1.19599,
    "square yard": 1.19599,
    "perch": 272.25,
    "acres": 43560.0,
    "acre": 43560.0,
    "guntha": 1011.71,
    "ground": 10890.0,
}


def clean_area_type(value: Any) -> Optional[str]:
    """Normalize internal whitespace and check against supported area types."""
    if value is None or pd.isna(value):
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if cleaned in AREA_TYPE_TO_INT else cleaned


def parse_sqft(value: Any) -> float:
    """
    Turn a total_sqft entry into a float number of square feet.

    Handles plain numbers, comma separators, 'a-b' ranges (midpoint) and
    other area units such as '34.46Sq. Meter', '1100Sq. Yards' or '5Acres'.
    Returns np.nan for negative, zero or malformed inputs.
    """
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan

    match = re.fullmatch(r"([\d.]+)\s*([- \u2013to]+)\s*([\d.]+)", text, flags=re.IGNORECASE)
    if match:
        try:
            first, second = float(match.group(1)), float(match.group(3))
            if 0 < first <= second:
                return (first + second) / 2
            return np.nan
        except (ValueError, TypeError):
            return np.nan

    try:
        number = float(text)
        return number if number > 0 else np.nan
    except ValueError:
        unit_match = re.fullmatch(r"([\d.]+)\s*([A-Za-z. ]+)", text, flags=re.IGNORECASE)
        if not unit_match:
            return np.nan
        try:
            number = float(unit_match.group(1))
            if number <= 0:
                return np.nan
            unit = unit_match.group(2).strip().lower()
            factor = SQFT_PER_UNIT.get(unit)
            if factor is None:
                return np.nan
            return number * factor
        except (ValueError, TypeError):
            return np.nan


def extract_bhk(size: Any) -> float:
    """Extract the bedroom count from labels like '2 BHK', '4 Bedroom', 'Studio'."""
    if size is None or pd.isna(size):
        return np.nan
    text = str(size).strip()
    if not text:
        return np.nan
    if re.search(r"\bstudio\b", text, flags=re.IGNORECASE):
        return 0.0
    if re.search(r"\brk\b", text, flags=re.IGNORECASE):
        return 1.0
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def parse_availability(value: Any) -> int:
    """
    Turn an availability label into the expected completion year.

    0 means the house is ready to move.
    'Ready To Move' -> 0
    '18-Dec'        -> 2018
    Anything else   -> 0
    """
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip()
    if re.fullmatch(r"\d{2}-[A-Za-z]{3}", text):
        return 2000 + int(text[:2])
    return 0


def clean_category(value: Any, default: str = MISSING) -> str:
    """Normalise free-text category input; empty/placeholder text becomes default (MISSING)."""
    if value is None or pd.isna(value):
        return default
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.upper() in ("NAN", "NONE", "NULL", "UNKNOWN", "-", "N/A"):
        return default
    return text


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized: Dict[str, Any] = field(default_factory=dict)


def validate_input(row: Dict[str, Any]) -> ValidationResult:
    """
    Strict domain validation for single prediction input.

    Rejects inputs outside the training domain bounds.
    Adds warnings for unusual combinations (extrapolations).
    Never silently clamps or substitutes invalid categories.
    """
    errors: List[str] = []
    warnings: List[str] = []
    sanitized: Dict[str, Any] = {}

    # 1. area_type
    raw_area = row.get("area_type")
    norm_area = clean_area_type(raw_area)
    if not norm_area or norm_area not in AREA_TYPE_TO_INT:
        supported = ", ".join(SUPPORTED_AREA_TYPES)
        errors.append(
            f"Unsupported area type '{raw_area}'. Supported area types are: {supported}."
        )
    else:
        sanitized["area_type"] = norm_area

    # 2. total_sqft
    raw_sqft = row.get("total_sqft")
    parsed_sqft = parse_sqft(raw_sqft)
    if pd.isna(parsed_sqft):
        errors.append(f"Invalid total area value '{raw_sqft}'. Must be a positive number of sqft.")
    else:
        c = VALIDATION_CONSTRAINTS["total_sqft"]
        if parsed_sqft < c["min"] or parsed_sqft > c["max"]:
            errors.append(
                f"Total area must be between {c['min']:,.0f} and {c['max']:,.0f} sqft. "
                f"Received {parsed_sqft:,.1f} sqft."
            )
        else:
            sanitized["total_sqft"] = float(parsed_sqft)
            if parsed_sqft > c["warning_high"]:
                warnings.append(
                    f"Total area ({parsed_sqft:,.0f} sqft) is in the top 2% of historical listings. "
                    "Prediction involves extrapolation beyond typical properties."
                )

    # 3. bath
    raw_bath = row.get("bath")
    try:
        bath_val = int(raw_bath)
        c = VALIDATION_CONSTRAINTS["bath"]
        if bath_val < c["min"] or bath_val > c["max"]:
            errors.append(
                f"Bathrooms must be between {c['min']} and {c['max']}. Received {bath_val}."
            )
        else:
            sanitized["bath"] = bath_val
            if bath_val > c["warning_high"]:
                warnings.append(
                    f"Bathroom count ({bath_val}) is unusually high in this historical dataset."
                )
    except (TypeError, ValueError):
        errors.append(f"Invalid bathroom count '{raw_bath}'. Must be an integer.")

    # 4. balcony
    raw_balcony = row.get("balcony", 0)
    try:
        balcony_val = int(raw_balcony)
        c = VALIDATION_CONSTRAINTS["balcony"]
        if balcony_val < c["min"] or balcony_val > c["max"]:
            errors.append(
                f"Balconies must be between {c['min']} and {c['max']}. Received {balcony_val}."
            )
        else:
            sanitized["balcony"] = balcony_val
    except (TypeError, ValueError):
        errors.append(f"Invalid balcony count '{raw_balcony}'. Must be an integer.")

    # 5. bhk
    raw_bhk = row.get("bhk")
    try:
        bhk_val = int(raw_bhk)
        c = VALIDATION_CONSTRAINTS["bhk"]
        if bhk_val < c["min"] or bhk_val > c["max"]:
            errors.append(
                f"BHK must be between {c['min']} (Studio/RK) and {c['max']}. Received {bhk_val}."
            )
        else:
            sanitized["bhk"] = bhk_val
            if bhk_val > c["warning_high"]:
                warnings.append(
                    f"BHK count ({bhk_val}) is very rare in this dataset (<0.5% of listings)."
                )
    except (TypeError, ValueError):
        errors.append(f"Invalid BHK '{raw_bhk}'. Must be an integer.")

    # 6. completion_year
    raw_year = row.get("completion_year", 0)
    try:
        year_val = int(raw_year)
        if year_val != 0 and (year_val < 2013 or year_val > 2030):
            errors.append(
                f"Expected completion year must be 0 (ready to move) or between 2013 and 2030. "
                f"Received {year_val}."
            )
        else:
            sanitized["completion_year"] = year_val
    except (TypeError, ValueError):
        errors.append(f"Invalid completion year '{raw_year}'. Must be an integer.")

    # 7. location & society
    sanitized["location"] = clean_category(row.get("location"))
    sanitized["society"] = clean_category(row.get("society"))

    # Feature combination / physical plausibility checks
    if not errors:
        sqft = sanitized["total_sqft"]
        bhk = sanitized["bhk"]
        bath = sanitized["bath"]

        if bhk > 0:
            sqft_per_bhk = sqft / bhk
            if sqft_per_bhk < 200:
                warnings.append(
                    f"Area per bedroom is unusually compact ({sqft_per_bhk:.0f} sqft/BHK). "
                    "Predictions may have lower precision."
                )
            elif sqft_per_bhk > 2500:
                warnings.append(
                    f"Area per bedroom is exceptionally large ({sqft_per_bhk:.0f} sqft/BHK). "
                    "Luxury properties in this tier have high price dispersion."
                )

        if bhk > 0 and bath > bhk + 2:
            warnings.append(
                f"Bathrooms ({bath}) significantly exceed bedrooms ({bhk} BHK). "
                "Unusual layout for historical residential listings."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized=sanitized,
    )


class DataPreprocessor:
    """
    Leakage-free preprocessing pipeline.

    Learns imputation statistics (BHK median) and high-cardinality frequency maps
    strictly from the training split/fold provided to `fit()`.
    Transforming validation or test data uses only the fitted statistics.
    """

    def __init__(self):
        self.schema_version = SCHEMA_VERSION
        self.feature_order = list(FEATURES)
        self.bhk_median_: int = 2
        self.location_freq_: Dict[str, float] = {}
        self.society_freq_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessing statistics strictly on training data."""
        # 1. Fit BHK median
        bhk_series = df["bhk"].dropna()
        self.bhk_median_ = int(round(float(bhk_series.median()))) if len(bhk_series) else 2

        # 2. Fit frequency maps from training split
        loc_clean = df["location"].astype(str).map(clean_category)
        soc_clean = df["society"].astype(str).map(clean_category)
        self.location_freq_ = loc_clean.value_counts().to_dict()
        self.society_freq_ = soc_clean.value_counts().to_dict()

        # 3. Fit StandardScaler on encoded training features (for linear models)
        encoded_train = self._encode_features(df, is_train=True)
        self.scaler_ = StandardScaler().fit(encoded_train)
        return self

    def _encode_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Internal helper to convert prepared columns into the numeric feature matrix."""
        out = pd.DataFrame(index=df.index)

        # Numeric measurements
        out["total_sqft"] = df["total_sqft"].astype(float)
        out["bath"] = pd.to_numeric(df["bath"], errors="coerce").fillna(0).astype(float)
        out["balcony"] = pd.to_numeric(df["balcony"], errors="coerce").fillna(0).astype(float)

        # BHK: fill missing with fitted training median
        out["bhk"] = pd.to_numeric(df["bhk"], errors="coerce").fillna(self.bhk_median_).astype(int)

        # area_type: explicit schema mapping (reject or raise on unknown)
        area_types = df["area_type"].map(clean_area_type)
        unknown_mask = ~area_types.isin(AREA_TYPE_TO_INT)
        if unknown_mask.any() and not is_train:
            first_unknown = df.loc[unknown_mask, "area_type"].iloc[0]
            raise ValueError(f"Unknown area_type encountered during transform: '{first_unknown}'")
        out["area_type"] = area_types.map(AREA_TYPE_TO_INT).fillna(0).astype(int)

        # completion_year
        if "completion_year" in df.columns:
            out["completion_year"] = pd.to_numeric(df["completion_year"], errors="coerce").fillna(0).astype(int)
        elif "availability" in df.columns:
            out["completion_year"] = df["availability"].map(parse_availability).astype(int)
        else:
            out["completion_year"] = 0

        # Frequency encoded location & society (unseen -> 0.0)
        loc_series = df["location"].map(clean_category)
        soc_series = df["society"].map(clean_category)
        out["location"] = loc_series.map(self.location_freq_).fillna(0.0).astype(float)
        out["society"] = soc_series.map(self.society_freq_).fillna(0.0).astype(float)

        return out[self.feature_order]

    def transform(self, df: pd.DataFrame, scale_linear: bool = False) -> pd.DataFrame:
        """Transform a DataFrame into the model-ready feature matrix."""
        encoded = self._encode_features(df, is_train=False)
        if scale_linear:
            if self.scaler_ is None:
                raise RuntimeError("Scaler is not fitted yet. Call fit() first.")
            scaled = self.scaler_.transform(encoded)
            return pd.DataFrame(scaled, index=encoded.index, columns=self.feature_order)
        return encoded

    def transform_row(self, row: Dict[str, Any], scale_linear: bool = False) -> pd.DataFrame:
        """Transform a validated single-row dictionary for inference."""
        df = pd.DataFrame([row])
        return self.transform(df, scale_linear=scale_linear)

    def to_bundle(self) -> Dict[str, Any]:
        """Serialize into a bundle compatible with saved joblib format and legacy loaders."""
        le = LabelEncoder().fit(list(SUPPORTED_AREA_TYPES))
        return {
            "schema_version": self.schema_version,
            "feature_order": list(self.feature_order),
            "bhk_median": int(self.bhk_median_),
            "location_freq": dict(self.location_freq_),
            "society_freq": dict(self.society_freq_),
            "scaler": self.scaler_,
            "le_area_type": le,
            "supported_area_types": list(SUPPORTED_AREA_TYPES),
        }

    @classmethod
    def from_bundle(cls, bundle: Dict[str, Any]) -> "DataPreprocessor":
        """Instantiate preprocessor from a loaded joblib dictionary bundle."""
        instance = cls()
        instance.schema_version = bundle.get("schema_version", "2.0.0")
        instance.feature_order = list(bundle.get("feature_order", FEATURES))
        instance.bhk_median_ = int(bundle.get("bhk_median", 2))
        instance.location_freq_ = dict(bundle.get("location_freq", {}))
        instance.society_freq_ = dict(bundle.get("society_freq", {}))
        instance.scaler_ = bundle.get("scaler")
        return instance
