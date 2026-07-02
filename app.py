from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from inference import CreditScorePredictor, MODEL_PATH, ARTIFACT_DIR
from credit_project import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_CASES_PATH = ARTIFACT_DIR / "sample_cases.csv"

st.set_page_config(page_title="Credit Score Deployment", page_icon="", layout="wide")

st.title("Credit Score Prediction Deployment")
st.caption("Local deployment for the BINUS Model Deployment project using data_A.csv")

if not MODEL_PATH.exists():
    st.error("Model belum ditemukan. Jalankan train_pipeline.py terlebih dahulu untuk membuat artifacts/best_model.pkl.")
    st.stop()

predictor = CreditScorePredictor()

st.sidebar.header("Test Case")
case_mode = st.sidebar.radio("Pilih input", ["Manual", "Sample Good", "Sample Poor", "Sample Standard"], index=1)

sample_row = None
if SAMPLE_CASES_PATH.exists():
    sample_cases = pd.read_csv(SAMPLE_CASES_PATH)
    label_map = {
        "Sample Good": "Good",
        "Sample Poor": "Poor",
        "Sample Standard": "Standard",
    }
    if case_mode in label_map and "Sample_Label" in sample_cases.columns:
        subset = sample_cases[sample_cases["Sample_Label"] == label_map[case_mode]]
        if not subset.empty:
            sample_row = subset.iloc[0].to_dict()

def _float_input(label: str, value: float) -> float:
    return float(st.number_input(label, value=float(value), step=1.0, format="%.4f"))

def _int_input(label: str, value: int) -> int:
    return int(st.number_input(label, value=int(value), step=1))

with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August"], index=2)
        occupation = st.selectbox(
            "Occupation",
            ["Accountant", "Developer", "Engineer", "Entrepreneur", "Journalist", "Lawyer", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer", "_______"],
        )
        credit_mix = st.selectbox("Credit_Mix", ["Good", "Standard", "Bad", "_"], index=1)
        payment_of_min_amount = st.selectbox("Payment_of_Min_Amount", ["Yes", "No", "NM", "_"], index=1)
        payment_behaviour = st.selectbox(
            "Payment_Behaviour",
            [
                "High_spent_Large_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Small_value_payments",
                "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Small_value_payments",
                "!@9#%8",
            ],
            index=4,
        )

    with col2:
        age = _int_input("Age", sample_row.get("Age", 35) if sample_row else 35)
        annual_income = _float_input("Annual_Income", sample_row.get("Annual_Income", 50000) if sample_row else 50000)
        monthly_inhand_salary = _float_input("Monthly_Inhand_Salary", sample_row.get("Monthly_Inhand_Salary", 4000) if sample_row else 4000)
        num_bank_accounts = _int_input("Num_Bank_Accounts", sample_row.get("Num_Bank_Accounts", 4) if sample_row else 4)
        num_credit_card = _int_input("Num_Credit_Card", sample_row.get("Num_Credit_Card", 5) if sample_row else 5)
        interest_rate = _int_input("Interest_Rate", sample_row.get("Interest_Rate", 10) if sample_row else 10)
        num_of_loan = _int_input("Num_of_Loan", sample_row.get("Num_of_Loan", 3) if sample_row else 3)
        delay_from_due_date = _int_input("Delay_from_due_date", sample_row.get("Delay_from_due_date", 10) if sample_row else 10)
        num_delayed_payment = _int_input("Num_of_Delayed_Payment", sample_row.get("Num_of_Delayed_Payment", 5) if sample_row else 5)

    with col3:
        changed_credit_limit = _float_input("Changed_Credit_Limit", sample_row.get("Changed_Credit_Limit", 10) if sample_row else 10)
        num_credit_inquiries = _float_input("Num_Credit_Inquiries", sample_row.get("Num_Credit_Inquiries", 4) if sample_row else 4)
        outstanding_debt = _float_input("Outstanding_Debt", sample_row.get("Outstanding_Debt", 1000) if sample_row else 1000)
        credit_utilization_ratio = _float_input("Credit_Utilization_Ratio", sample_row.get("Credit_Utilization_Ratio", 30) if sample_row else 30)
        credit_history_age_months = _float_input("Credit_History_Age_Months", 120)
        total_emi_per_month = _float_input("Total_EMI_per_month", sample_row.get("Total_EMI_per_month", 100) if sample_row else 100)
        amount_invested_monthly = _float_input("Amount_invested_monthly", sample_row.get("Amount_invested_monthly", 200) if sample_row else 200)
        monthly_balance = _float_input("Monthly_Balance", sample_row.get("Monthly_Balance", 300) if sample_row else 300)
        type_of_loan_count = _float_input("Type_of_Loan_Count", sample_row.get("Type_of_Loan_Count", 3) if sample_row else 3)
        type_of_loan_has_not_specified = _float_input("Type_of_Loan_Has_Not_Specified", sample_row.get("Type_of_Loan_Has_Not_Specified", 1) if sample_row else 1)

    submit_button = st.form_submit_button("Predict Credit Score")

if submit_button:
    payload = {
        "Month": month,
        "Occupation": occupation,
        "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Payment_Behaviour": payment_behaviour,
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Credit_History_Age_Months": credit_history_age_months,
        "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly,
        "Monthly_Balance": monthly_balance,
        "Type_of_Loan_Count": type_of_loan_count,
        "Type_of_Loan_Has_Not_Specified": type_of_loan_has_not_specified,
    }

    result = predictor.predict(payload)
    st.subheader("Prediction Result")
    st.success(f"Predicted Credit Score: {result['prediction']}")

    if "probabilities" in result:
        probability_df = pd.DataFrame([result["probabilities"]]).T.reset_index()
        probability_df.columns = ["Class", "Probability"]
        st.dataframe(probability_df, use_container_width=True)

st.divider()
st.subheader("Feature Set Used")
st.write({"numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES})
