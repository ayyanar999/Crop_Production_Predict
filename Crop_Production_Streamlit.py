import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("pipeline_model.pkl")
preprocessor = joblib.load("input_features.pkl")
data = pd.read_csv("Cleaned_FAOSTAT_data.csv")  

st.title("🌾 Crop Production Prediction")

# Dropdowns
area = st.selectbox("Select Area", data["Area"].unique())
item = st.selectbox("Select Crop", data["Item"].unique())
year = st.slider("Select Year", int(data["Year"].min()), int(data["Year"].max()))
area_harvested = st.number_input("Area Harvested (ha)")
yield_val = st.number_input("Yield (kg/ha)")

# Prediction
if st.button("Predict Production"):
    # Create input vector
    input_df = pd.DataFrame([[area, item, year, area_harvested, yield_val]],
                            columns=preprocessor)
    
    # Use pipeline to transform and predict
    prediction = model.predict(input_df)

    st.success(f"🌟 Predicted Production: {prediction[0]:,.2f} tons")
