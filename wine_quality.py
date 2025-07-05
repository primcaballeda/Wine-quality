import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("wine_model.pkl")

st.set_page_config(page_title="Wine Quality Detector")
st.title("Wine Quality Detector")
st.write("Enter the chemical attributes of a red wine sample:")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, step=0.1, value=7.9)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.6, step=0.01, value=0.32)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, step=0.01, value=0.51)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, step=0.1, value=2.0)
chlorides = st.number_input("Chlorides", 0.01, 0.2, step=0.001, value=0.07)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1, 72, step=1, value=15)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6, 300, step=1, value=54)
density = st.number_input("Density", 0.9900, 1.0050, step=0.0001, value=0.9964)
pH = st.number_input("pH", 2.5, 4.5, step=0.01, value=3.30)
sulphates = st.number_input("Sulphates", 0.3, 2.0, step=0.01, value=0.86)
alcohol = st.number_input("Alcohol", 8.0, 15.0, step=0.1, value=12.8)

# Predict
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                          density, pH, sulphates, alcohol]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] 
    if prediction == 1:
        st.success(f"Good Quality Wine (Confidence: {probability:.2f} or {probability*100:.1f}%)")
    else:
        st.error(f"Not Good Quality Wine (Confidence: {1 - probability:.2f} or {(1 - probability)*100:.1f}%)")
