import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_prediction_model.joblib")

# Page configuration
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Loan Prediction System")
st.write(
    "Enter the applicant details below to predict "
    "the loan approval status using Logistic Regression."
)

st.divider()

# Applicant Information
st.header("👤 Applicant Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

employee_experience = st.number_input(
    "Employee Experience (Years)",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

# Financial Information
st.header("💵 Financial Information")

person_income = st.number_input(
    "Person Income",
    min_value=0,
    value=50000,
    step=1000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000,
    step=1000
)

loan_interest_rate = st.number_input(
    "Loan Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.1
)

loan_percentage = st.number_input(
    "Loan Percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.1
)

# Loan Information
st.header("🏦 Loan Information")

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

credit_history = st.number_input(
    "Credit History",
    min_value=0,
    max_value=30,
    value=2,
    step=1
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650,
    step=1
)

previous_loan = st.selectbox(
    "Previous Loan",
    ["Yes", "No"]
)

st.divider()

# Prediction
if st.button("🔍 Predict Loan Status", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education": [education],
        "Person Income": [person_income],
        "Employee Experience": [employee_experience],
        "Home Onwership": [home_ownership],
        "Loan Amount": [loan_amount],
        "Loan Intent": [loan_intent],
        "Loan interest Rate": [loan_interest_rate],
        "Loan percentage": [loan_percentage],
        "Credit History": [credit_history],
        "Credit Score": [credit_score],
        "Previous Loan": [previous_loan]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("🎉 Loan Status: APPROVED")
    else:
        st.error("❌ Loan Status: NOT APPROVED")

    st.metric(
        "Loan Approval Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    with st.expander("View Submitted Information"):
        st.dataframe(
            input_data,
            use_container_width=True
        )

st.divider()

st.caption(
    "Loan Prediction System | Logistic Regression | Machine Learning Project"
)
