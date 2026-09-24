import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_prediction_model.joblib")

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰"
)

st.title("💰 Loan Prediction System")
