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

@st.cache_resource
def get_predictor():
    return CreditScorePredictor()

# Panggil fungsinya
predictor = get_predictor()

st.sidebar.header("Test Case")
case_mode = st.sidebar.radio("Pilih input", ["Manual", "Sample Good", "Sample Poor", "Sample Standard"], index=1)

fallback_cases = {
    "Sample Good": {
        "Age": 35, "Annual_Income": 850000.0, "Month": "January", "Occupation": "Engineer", 
        "Credit_Mix": "Good", "Payment_of_Min_Amount": "No", "Payment_Behaviour": "High_spent_Large_value_payments", 
        "Num_Bank_Accounts": 2, "Num_Credit_Card": 2, "Interest_Rate": 5, "Num_of_Loan": 1, 
        "Delay_from_due_date": 0, "Num_of_Delayed_Payment": 0, "Changed_Credit_Limit": 2.0, 
        "Num_Credit_Inquiries": 1.0, "Outstanding_Debt": 1500.0, "Credit_Utilization_Ratio": 25.0, 
        "Total_EMI_per_month": 50.0, "Amount_invested_monthly": 500.0, "Monthly_Balance": 1000.0, 
        "Type_of_Loan_Count": 1.0, "Type_of_Loan_Has_Not_Specified": 0.0
    },
    "Sample Poor": {
        "Age": 21, "Annual_Income": 35000.0, "Month": "January", "Occupation": "_______", 
        "Credit_Mix": "Bad", "Payment_of_Min_Amount": "Yes", "Payment_Behaviour": "Low_spent_Small_value_payments", 
        "Num_Bank_Accounts": 7, "Num_Credit_Card": 7, "Interest_Rate": 20, "Num_of_Loan": 7, 
        "Delay_from_due_date": 90, "Num_of_Delayed_Payment": 15, "Changed_Credit_Limit": 20.0, 
        "Num_Credit_Inquiries": 10.0, "Outstanding_Debt": 150000.0, "Credit_Utilization_Ratio": 45.0, 
        "Total_EMI_per_month": 500.0, "Amount_invested_monthly": 10.0, "Monthly_Balance": 10.0, 
        "Type_of_Loan_Count": 7.0, "Type_of_Loan_Has_Not_Specified": 1.0
    },
    "Sample Standard": {
        "Age": 28, "Annual_Income": 150000.0, "Month": "January", "Occupation": "Teacher", 
        "Credit_Mix": "Standard", "Payment_of_Min_Amount": "NM", "Payment_Behaviour": "Low_spent_Medium_value_payments", 
        "Num_Bank_Accounts": 4, "Num_Credit_Card": 4, "Interest_Rate": 10, "Num_of_Loan": 3, 
        "Delay_from_due_date": 14, "Num_of_Delayed_Payment": 4, "Changed_Credit_Limit": 10.0, 
        "Num_Credit_Inquiries": 3.0, "Outstanding_Debt": 5000.0, "Credit_Utilization_Ratio": 30.0, 
        "Total_EMI_per_month": 150.0, "Amount_invested_monthly": 100.0, "Monthly_Balance": 400.0, 
        "Type_of_Loan_Count": 3.0, "Type_of_Loan_Has_Not_Specified": 0.0
    }
}

sample_row = None
if case_mode in ["Sample Good", "Sample Poor", "Sample Standard"]:
    label_map = {"Sample Good": "Good", "Sample Poor": "Poor", "Sample Standard": "Standard"}
    
    # Coba baca dari CSV dulu
    if SAMPLE_CASES_PATH.exists():
        sample_cases = pd.read_csv(SAMPLE_CASES_PATH)
        subset = sample_cases[sample_cases["Sample_Label"] == label_map[case_mode]]
        if not subset.empty:
            sample_row = subset.iloc[0].to_dict()
    
    # Kalau gagal baca CSV (atau subset kosong), pakai data darurat
    if not sample_row and case_mode in fallback_cases:
        sample_row = fallback_cases[case_mode]

def _float_input(label: str, value: float) -> float:
    return float(st.number_input(label, value=float(value), step=1.0, format="%.4f"))

def _int_input(label: str, value: int) -> int:
    return int(st.number_input(label, value=int(value), step=1))

# 2. FUNGSI PENCARI INDEX UNTUK DROPDOWN COL1
def get_idx(options_list, val):
    return options_list.index(val) if val in options_list else 0

with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        month_opts = ["January", "February", "March", "April", "May", "June", "July", "August"]
        month_val = sample_row.get("Month", "March") if sample_row else "March"
        month = st.selectbox("Month", month_opts, index=get_idx(month_opts, month_val))
        
        occ_opts = ["Accountant", "Developer", "Engineer", "Entrepreneur", "Journalist", "Lawyer", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer", "_______"]
        occ_val = sample_row.get("Occupation", "Engineer") if sample_row else "Engineer"
        occupation = st.selectbox("Occupation", occ_opts, index=get_idx(occ_opts, occ_val))
        
        mix_opts = ["Good", "Standard", "Bad", "_"]
        mix_val = sample_row.get("Credit_Mix", "Standard") if sample_row else "Standard"
        credit_mix = st.selectbox("Credit_Mix", mix_opts, index=get_idx(mix_opts, mix_val))
        
        pay_opts = ["Yes", "No", "NM", "_"]
        pay_val = sample_row.get("Payment_of_Min_Amount", "NM") if sample_row else "NM"
        payment_of_min_amount = st.selectbox("Payment_of_Min_Amount", pay_opts, index=get_idx(pay_opts, pay_val))
        
        beh_opts = ["High_spent_Large_value_payments", "High_spent_Medium_value_payments", "High_spent_Small_value_payments", "Low_spent_Large_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments", "!@9#%8"]
        beh_val = sample_row.get("Payment_Behaviour", "!@9#%8") if sample_row else "!@9#%8"
        payment_behaviour = st.selectbox("Payment_Behaviour", beh_opts, index=get_idx(beh_opts, beh_val))

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
