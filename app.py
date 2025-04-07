import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore') # Ignore potential warnings during prediction

# --- Load Saved Objects ---
try:
    # Load the trained RandomForestRegressor model
    model = joblib.load('model.joblib')

    # Load the label encoders (assuming they are in the same directory)
    encoders = {}
    categorical_cols = ['area_type', 'availability', 'location', 'size', 'society']
    for col in categorical_cols:
        encoders[col] = joblib.load(f'le_{col}.joblib')

except FileNotFoundError as e:
    st.error(f"Error loading saved files: {e}. Make sure 'model.joblib' and all 'le_*.joblib' files are in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# --- Prediction Function ---
def predict_house_price(inputs):
    """
    Takes a dictionary of raw inputs, processes them using loaded encoders,
    and returns the prediction.
    """
    try:
        # Create a DataFrame from inputs (important: one row)
        input_df = pd.DataFrame([inputs])

        # Apply label encoding to categorical features
        for col in categorical_cols:
            # Use the specific encoder for the column
            encoder = encoders[col]
            # Get the input value for the column
            value_to_encode = input_df.loc[0, col] # Get the single value

            # Handle unseen labels during prediction
            # Check if the value is in the encoder's known classes
            if value_to_encode not in encoder.classes_:
                # Option 1: Assign a default value (e.g., encode as if it's the most frequent or a special 'other' category if trained that way)
                # This is tricky without knowing the distribution or having an 'other' category.
                # A simpler approach for now is to raise an error or use a placeholder like -1 if your model handles it.
                # Let's try mapping to the first known class as a placeholder, but warn the user.
                st.warning(f"Warning: Category '{value_to_encode}' in column '{col}' was not seen during training. Using '{encoder.classes_[0]}' as a substitute for prediction.")
                input_df.loc[0, col] = encoder.transform([encoder.classes_[0]])[0] # Encode the substitute
            else:
                 # If the value is known, transform it
                 input_df.loc[0, col] = encoder.transform([value_to_encode])[0] # transform expects an array-like

        # Ensure columns are in the same order as training data
        # Based on your notebook: ['area_type', 'availability', 'location', 'size', 'society', 'bath', 'balcony']
        expected_cols_order = ['area_type', 'availability', 'location', 'size', 'society', 'bath', 'balcony']
        input_df = input_df[expected_cols_order]

        # Make prediction
        prediction = model.predict(input_df)
        return prediction[0] # Return the single prediction value

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


# --- Streamlit Interface ---
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="wide")
st.title("🏡 Bengaluru House Price Prediction")
st.write("Enter the details below to predict the house price (in Lakhs INR).")

# --- Get Input Options from Encoders (for selectboxes) ---
# We need the *original* categories known by the encoders
area_type_options = list(encoders['area_type'].classes_)
availability_options = list(encoders['availability'].classes_)
# Location and Size might have many - limit them for the UI or use text input
location_options = list(encoders['location'].classes_) # This list might be huge!
size_options = list(encoders['size'].classes_)
society_options = list(encoders['society'].classes_) # Also huge!

# --- Create Input Fields using Columns for Layout ---
col1, col2 = st.columns(2)

with col1:
    area_type = st.selectbox("Area Type", options=area_type_options, index=0)
    # Limit location options for usability
    location_display_options = location_options[:20] + ["Other"] # Show top 20 + 'Other'
    location = st.selectbox("Location", options=location_display_options, index=0)
    size = st.selectbox("Size (BHK/Bedroom)", options=size_options, index=0) # Use original size strings

with col2:
    availability = st.selectbox("Availability", options=availability_options, index=0)
    bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1, step=1)

# Handle Society separately due to large number of options
society = st.text_input("Society Name (Enter name or leave blank if none/unknown)")

# --- Prediction Button ---
if st.button("Predict Price", key="predict_button"):

    # Handle 'Other' selection for location
    if location == "Other":
        st.warning("Location 'Other' selected. Prediction might be less accurate as it wasn't a specific training category.")
        # Use a placeholder known by the encoder, e.g., the first location
        location_encoded = encoders['location'].classes_[0]
    else:
        location_encoded = location # Use the selected value

    # Handle potentially blank society
    society_input = society if society else "Unknown" # Use 'Unknown' or similar if blank

    # Prepare input dictionary for the prediction function
    input_data = {
        'area_type': area_type,
        'availability': availability,
        'location': location_encoded, # Use potentially adjusted location
        'size': size,
        'society': society_input,    # Use potentially adjusted society
        'bath': float(bath),         # Ensure numeric types
        'balcony': float(balcony)
    }

    # Get prediction
    predicted_price = predict_house_price(input_data)

    # Display result
    if predicted_price is not None:
        st.success(f"Predicted House Price: ₹ {predicted_price:.2f} Lakhs")

st.write("---")
st.write("_Note: This prediction is based on a machine learning model trained on historical data and may not reflect exact current market values._")