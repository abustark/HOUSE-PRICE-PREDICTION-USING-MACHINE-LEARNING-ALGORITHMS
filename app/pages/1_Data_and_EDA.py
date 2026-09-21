"""Data and EDA page - explore the dataset behind the models."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "Bengaluru_House_Data.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

st.set_page_config(page_title="Data & EDA | Bengaluru House Price Prediction", page_icon="\U0001F4CA")

st.title("Dataset & Exploratory Analysis")
st.caption("Bengaluru_House_Data.csv - housing listings for Bengaluru, price in Lakh INR.")

data = pd.read_csv(DATA_PATH)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(data):,}")
c2.metric("Columns", f"{len(data.columns)}")
c3.metric("Locations", f"{data['location'].nunique():,}")
c4.metric("Median price", f"{data['price'].median():.1f} Lakh")

st.subheader("Columns")
st.dataframe(data.head(10), width="stretch")

st.subheader("Missing values")
st.bar_chart(data.isnull().sum()[::-1])
st.markdown(
    """
    - `society` is missing in about 41% of rows (many listings are individual
      houses with no society name).
    - `location`, `size`, `bath` and `balcony` have very few missing entries.
    - The training pipeline fills missing baths/balconies with 0 and gives
      missing locations/societies a dedicated "not available" category.
    """
)

st.subheader("Price")
st.image(os.path.join(ARTIFACTS_DIR, "price_distribution.png"), width="stretch")
st.image(os.path.join(ARTIFACTS_DIR, "price_vs_sqft.png"), width="stretch")

st.subheader("Size and location")
st.image(os.path.join(ARTIFACTS_DIR, "price_by_bhk.png"), width="stretch")
st.image(os.path.join(ARTIFACTS_DIR, "top_locations.png"), width="stretch")

st.subheader("Price statistics by area type")
st.dataframe(
    data.groupby("area_type")["price"].agg(["count", "mean", "median", "min", "max"]).round(2),
    width="stretch",
)

st.divider()
st.caption(
    "The dataset is a historical snapshot of Bengaluru listings (2017-2020). "
    "It is a widely used public dataset for house price prediction practice."
)
