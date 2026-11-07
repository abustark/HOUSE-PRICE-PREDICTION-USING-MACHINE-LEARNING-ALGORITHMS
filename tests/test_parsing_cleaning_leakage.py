"""
Tests for raw data parsing, cleaning, and leakage-free preprocessing.
"""

import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import pytest

from src.encoding import (
    AREA_TYPE_TO_INT,
    FEATURES,
    MISSING,
    SUPPORTED_AREA_TYPES,
    DataPreprocessor,
    clean_area_type,
    clean_category,
    extract_bhk,
    parse_availability,
    parse_sqft,
)


# ---------------------------------------------------------------------------
# Data parsing tests
# ---------------------------------------------------------------------------
def test_parse_sqft_plain_numeric():
    assert parse_sqft("1200") == 1200.0
    assert parse_sqft(1500) == 1500.0
    assert parse_sqft("1,450.50") == 1450.50


def test_parse_sqft_ranges():
    # Midpoint of ranges
    assert parse_sqft("1000 - 1200") == 1100.0
    assert parse_sqft("2100 - 2850") == 2475.0
    assert parse_sqft("1000 to 1200") == 1100.0
    # Inverted range is malformed -> returns NaN
    assert np.isnan(parse_sqft("1500 - 1000"))


def test_parse_sqft_units():
    # Square meters: 1 sqm ≈ 10.7639 sqft
    sqm = parse_sqft("34.46Sq. Meter")
    assert math.isclose(sqm, 34.46 * 10.7639, rel_tol=1e-3)

    # Square yards: 1 sq yard ≈ 1.19599 sqft (or unit factor in dict)
    sqyd = parse_sqft("1100Sq. Yards")
    assert math.isclose(sqyd, 1100 * 1.19599, rel_tol=1e-3)

    # Acres: 1 acre = 43,560 sqft
    acres = parse_sqft("2Acres")
    assert acres == 2 * 43560.0


def test_parse_sqft_malformed_and_missing():
    for val in ("", "   ", None, np.nan, "unrelated text", "-500", "0"):
        assert np.isnan(parse_sqft(val))


def test_extract_bhk():
    assert extract_bhk("2 BHK") == 2.0
    assert extract_bhk("4 Bedroom") == 4.0
    assert extract_bhk("Studio") == 0.0
    assert extract_bhk("1 RK") == 1.0
    assert extract_bhk("10 BHK") == 10.0
    assert np.isnan(extract_bhk(""))
    assert np.isnan(extract_bhk(None))


def test_parse_availability():
    assert parse_availability("Ready To Move") == 0
    assert parse_availability("Immediate") == 0
    assert parse_availability("18-Dec") == 2018
    assert parse_availability("21-May") == 2021
    assert parse_availability(None) == 0


def test_clean_category():
    assert clean_category(" Whitefield ") == "Whitefield"
    assert clean_category("") == MISSING
    assert clean_category("   ") == MISSING
    assert clean_category("None") == MISSING
    assert clean_category("nan") == MISSING
    assert clean_category("N/A") == MISSING


def test_clean_area_type():
    assert clean_area_type("Super built-up Area") == "Super built-up Area"
    assert clean_area_type("Plot  Area") == "Plot Area"
    assert clean_area_type("Carpet  Area") == "Carpet Area"
    assert clean_area_type("Built-up  Area") == "Built-up Area"
    assert clean_area_type("Luxury Penthouse") == "Luxury Penthouse"


# ---------------------------------------------------------------------------
# Leakage-Free Preprocessing Tests
# ---------------------------------------------------------------------------
def test_preprocessor_fits_only_training_rows():
    """Verify that DataPreprocessor statistics depend strictly on training rows."""
    train_data = pd.DataFrame(
        {
            "total_sqft": [1000.0, 1500.0, 2000.0],
            "bhk": [2, 4, np.nan],  # training median is (2+4)/2 = 3
            "bath": [2, 3, 2],
            "balcony": [1, 2, 1],
            "completion_year": [0, 0, 2019],
            "area_type": ["Super built-up Area", "Built-up Area", "Plot Area"],
            "location": ["Whitefield", "Whitefield", "Electronic City"],
            "society": ["SocA", "SocA", "SocB"],
        }
    )

    validation_data = pd.DataFrame(
        {
            "total_sqft": [1200.0],
            "bhk": [np.nan],  # Should be imputed with train_data median (3), NOT validation median
            "bath": [2],
            "balcony": [1],
            "completion_year": [0],
            "area_type": ["Super built-up Area"],
            "location": ["Indiranagar"],  # Unseen location in training -> must be 0.0
            "society": ["SocC"],  # Unseen society in training -> must be 0.0
        }
    )

    prep = DataPreprocessor().fit(train_data)

    # 1. Imputed BHK on validation data uses training median (3)
    val_transformed = prep.transform(validation_data)
    assert val_transformed.loc[0, "bhk"] == 3

    # 2. Unseen location and society get frequency 0.0
    assert val_transformed.loc[0, "location"] == 0.0
    assert val_transformed.loc[0, "society"] == 0.0

    # 3. Known training frequencies
    train_transformed = prep.transform(train_data)
    assert train_transformed.loc[0, "location"] == 2.0  # Whitefield appeared twice
    assert train_transformed.loc[2, "location"] == 1.0  # Electronic City appeared once


def test_validation_rows_do_not_leak_into_frequencies_or_medians():
    """Prove that changing validation rows does NOT alter transformed training rows."""
    train_data = pd.DataFrame(
        {
            "total_sqft": [1000.0, 1200.0],
            "bhk": [2, 2],
            "bath": [2, 2],
            "balcony": [1, 1],
            "completion_year": [0, 0],
            "area_type": ["Super built-up Area", "Super built-up Area"],
            "location": ["Whitefield", "Whitefield"],
            "society": ["SocA", "SocA"],
        }
    )

    prep = DataPreprocessor().fit(train_data)
    tr_encoded_1 = prep.transform(train_data)

    # Introduce validation data with 100 listings in 'Whitefield'
    val_data = pd.DataFrame(
        {
            "total_sqft": [1000.0] * 100,
            "bhk": [10] * 100,
            "bath": [2] * 100,
            "balcony": [1] * 100,
            "completion_year": [0] * 100,
            "area_type": ["Super built-up Area"] * 100,
            "location": ["Whitefield"] * 100,
            "society": ["SocA"] * 100,
        }
    )

    # Transform validation data
    _ = prep.transform(val_data)

    # Re-transform training data: frequency MUST still be 2.0, not 102.0
    tr_encoded_2 = prep.transform(train_data)
    assert tr_encoded_1.loc[0, "location"] == 2.0
    assert tr_encoded_2.loc[0, "location"] == 2.0
    assert tr_encoded_1.equals(tr_encoded_2)


def test_unknown_area_type_raises_error_in_transform():
    train_data = pd.DataFrame(
        {
            "total_sqft": [1000.0],
            "bhk": [2],
            "bath": [2],
            "balcony": [1],
            "completion_year": [0],
            "area_type": ["Super built-up Area"],
            "location": ["Whitefield"],
            "society": ["SocA"],
        }
    )
    prep = DataPreprocessor().fit(train_data)

    bad_data = pd.DataFrame(
        {
            "total_sqft": [1000.0],
            "bhk": [2],
            "bath": [2],
            "balcony": [1],
            "completion_year": [0],
            "area_type": ["Floating Space Villa"],  # Unknown
            "location": ["Whitefield"],
            "society": ["SocA"],
        }
    )
    with pytest.raises(ValueError, match="Unknown area_type"):
        prep.transform(bad_data)
