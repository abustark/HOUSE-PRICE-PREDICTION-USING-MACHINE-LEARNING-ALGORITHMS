"""
Shared feature engineering rules.

Both the training pipeline (src/train.py) and the web app (src/predict.py)
import these helpers so that raw data is always transformed the same way.
"""

import re

import numpy as np
import pandas as pd

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


def parse_sqft(value):
    """
    Turn a total_sqft entry into a float number of square feet.

    Handles plain numbers, comma separators, 'a-b' ranges (midpoint) and
    other area units such as '34.46Sq. Meter', '1100Sq. Yards' or '5Acres'.
    """
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan

    match = re.fullmatch(r"([\d.]+)\s*([- \u2013to]+)\s*([\d.]+)", text, flags=re.IGNORECASE)
    if match:
        first, second = float(match.group(1)), float(match.group(3))
        if first <= second:  # a real range, not two unrelated numbers
            return (first + second) / 2
    try:
        number = float(text)
    except ValueError:
        unit_match = re.fullmatch(r"([\d.]+)\s*([A-Za-z. ]+)", text, flags=re.IGNORECASE)
        if not unit_match:
            return np.nan
        number = float(unit_match.group(1))
        unit = unit_match.group(2).strip().lower()
        factor = SQFT_PER_UNIT.get(unit)
        if factor is None:
            return np.nan
        return number * factor
    return number


def extract_bhk(size):
    """Extract the bedroom count from labels like '2 BHK', '4 Bedroom', 'Studio'."""
    if size is None or pd.isna(size):
        return np.nan
    text = str(size)
    if re.search(r"\bstudio\b", text, flags=re.IGNORECASE):
        return 0
    if re.search(r"\brk\b", text, flags=re.IGNORECASE):
        return 1
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else np.nan


def parse_availability(value):
    """
    Turn an availability label into the expected completion year.

    0 means the house is ready to move.

    'Ready To Move' -> 0
    '18-Dec'        -> 2018       # expected completion year
    Anything else   -> 0          # treated as immediately available
    """
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip()
    if re.fullmatch(r"\d{2}-[A-Za-z]{3}", text):
        return 2000 + int(text[:2])
    return 0


def clean_category(value, default=MISSING):
    """Normalise free-text category input; empty/placeholder text becomes MISSING."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.upper() in ("NAN", "NONE", "NULL", "UNKNOWN", "-"):
        return default
    return text
