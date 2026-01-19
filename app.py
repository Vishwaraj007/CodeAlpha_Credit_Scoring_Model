import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/credit_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("💳 Credit Risk Prediction")

age = st.slider("Age", 18, 70)
income = st.number_input("Annual Income", min_value=0)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
emp_length = st.slider("Employment Length (years)", 0, 40)
loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.slider("Interest Rate (%)", 0.0, 30.0)
loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0)
default_history = st.selectbox("Previous Default", ["Y", "N"])
cred_hist_length = st.slider("Credit History Length (years)", 0, 30)

# Encoding (MUST MATCH TRAINING)
home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"N": 0, "Y": 1}

if st.button("Predict"):
    input_data = np.array([[age, income, home_map[home_ownership], emp_length,
                             intent_map[loan_intent], grade_map[loan_grade],
                             loan_amnt, loan_int_rate, loan_percent_income,
                             default_map[default_history], cred_hist_length]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("✅ Low Credit Risk (Likely to Repay)")
    else:
        st.error("❌ High Credit Risk (Likely to Default)")

