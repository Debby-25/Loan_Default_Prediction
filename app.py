import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("loan_default_model.pkl")  # Ensure this file exists in your project folder
scaler = joblib.load("scaler.pkl")  # Ensure this file exists in your project folder
encoder = joblib.load("encoder.pkl")  # Ensure this file exists in your project folder

# Title of the app
st.title("Loan Default Prediction App")
st.write("Enter details below to predict loan repayment default.")

# User input fields
def user_input_features():
    # Adjust these fields based on the actual dataset features
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Monthly Income", min_value=0, value=5000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    
    # Create a DataFrame for user inputs
    data = pd.DataFrame({
        "age": [age],
        "income": [income],
        "loan_amount": [loan_amount],
        "employment_type": [employment_type],
        "credit_score": [credit_score]
    })
    
    return data

# Get user input
input_data = user_input_features()

# Preprocess user input
categorical_cols = ["employment_type"]
numerical_cols = ["age", "income", "loan_amount", "credit_score"]

input_data_scaled = scaler.transform(input_data[numerical_cols])
input_data_encoded = encoder.transform(input_data[categorical_cols])

# Combine numerical and categorical features
input_data_final = np.hstack((input_data_scaled, input_data_encoded.toarray()))

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data_final)
    result = "Likely to Default" if prediction[0] == 1 else "Unlikely to Default"
    st.write(f"### Prediction: {result}")



import streamlit as st

st.title("Loan Default Prediction")
st.write("Welcome to the Loan Default Prediction App!")

# Dummy input fields
name = st.text_input("Enter your name")
income = st.number_input("Enter your monthly income")
loan_amount = st.number_input("Enter loan amount")

if st.button("Predict"):
    st.write("Prediction result will be displayed here.")

