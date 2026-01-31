
# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path


# LOAD MODEL ASSETS (AWS SAFE)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

model = joblib.load(MODEL_DIR / "churn_model.pkl")
imputer = joblib.load(MODEL_DIR / "imputer.pkl")
model_features = joblib.load(MODEL_DIR / "model_features.pkl")


# APP CONFIG
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("📉 Customer Churn Prediction Dashboard")

st.markdown("""
This app predicts **customer churn probability**
and explains the result using **SHAP (Explainable AI)**.
""")


# USER INPUT (SIDEBAR)
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges", 10.0, 150.0, 70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    value=float(monthly_charges * max(tenure, 1))
)


# CREATE INPUT DATAFRAME
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone],
    "InternetService": [internet],
    "Contract": [contract],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# PREDICTION FUNCTION
def predict_churn(input_df):
    df = input_df.copy()

    # Binary encoding
    binary_map = {"Yes": 1, "No": 0}
    for col in ["Partner", "Dependents", "PhoneService"]:
        df[col] = df[col].map(binary_map)

    # Feature engineering
    df["AvgMonthlyCharges"] = df["TotalCharges"] / max(df["tenure"].iloc[0], 1)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align with training features
    df = df.reindex(columns=model_features, fill_value=0)

    # Imputation
    df_imputed = imputer.transform(df)

    # Prediction
    churn_prob = model.predict_proba(df_imputed)[:, 1][0]

    return churn_prob, df_imputed


# RUN PREDICTION
if st.button("🔍 Predict Churn"):

    prob, processed_input = predict_churn(input_df)

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")


    # SHAP EXPLANATION
    st.subheader("🔎 Why this prediction? (SHAP Explanation)")

    background = pd.DataFrame(
        np.zeros((1, len(model_features))),
        columns=model_features
    )

    background = imputer.transform(background)

    explainer = shap.Explainer(
        model,
        pd.DataFrame(background, columns=model_features)
    )

    shap_values = explainer(
        pd.DataFrame(processed_input, columns=model_features)
    )

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
