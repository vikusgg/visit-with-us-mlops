import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Wellness Tourism Purchase Predictor")

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="vikusg/visit-with-us-wellness-model",
    filename="wellness_model.pkl"
)

model = joblib.load(model_path)

st.header("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=70)
income = st.number_input("Monthly Income", min_value=0)
trips = st.number_input("Number of Trips", min_value=0)
passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0,1])

if st.button("Predict"):
    
    input_df = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [income],
        "NumberOfTrips": [trips],
        "Passport": [passport]
    })

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Customer is Likely to Purchase")
    else:
        st.error("Customer is Unlikely to Purchase")
