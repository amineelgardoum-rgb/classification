import streamlit as st
import numpy as np
import joblib
import os
model_path = os.path.join("models", "model.3.pkl")
# Load model
with open(model_path, "rb") as f:
    model = joblib.load(f)
scaler_path=os.path.join("models","scaler","scaler.pkl")
# Load scaler
with open(scaler_path, "rb") as f:
    scaler = joblib.load(f)

st.title("❤️ Heart Attack Prediction Dashboard")
st.write("Enter patient data to predict risk of heart attack")

# -----------------------------
# INPUT FIELDS
# -----------------------------
age = st.number_input("Age", 18, 120, 50)
gender = st.selectbox("Gender (1=Male, 0=Female)", [0, 1])
heart_rate = st.number_input("Heart Rate", 30, 200, 70)
sbp = st.number_input("Systolic BP", 60, 250, 120)
dbp = st.number_input("Diastolic BP", 30, 140, 80)
blood_sugar = st.number_input("Blood Sugar", 40.0, 500.0, 100.0)
ckmb = st.number_input("CK-MB", 0.0, 400.0, 1.0)
troponin = st.number_input("Troponin", 0.0, 10.0, 0.01)

# Prepare data for model
features = np.array([[age, gender, heart_rate, sbp, dbp, blood_sugar, ckmb, troponin]])
scaled_features = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]  # prob of class 1

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"High Risk ⚠️ (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk ✅ (Probability: {probability:.2f})")

    st.write("Model Output:", "1 = High Risk", ", 0 = Low Risk")
