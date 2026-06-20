import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="EMI Prediction", page_icon="🔮", layout="wide")

# ── Load models ──
@st.cache_resource
def load_models():
    classifier = joblib.load('models/best_classifier.pkl')
    regressor  = joblib.load('models/best_regressor.pkl')
    scaler     = joblib.load('models/scaler.pkl')
    with open('data/processed/feature_cols_engineered.json') as f:
        feature_cols = json.load(f)
    return classifier, regressor, scaler, feature_cols

classifier, regressor, scaler, feature_cols = load_models()

# ── Feature engineering ──
def engineer_features(df):
    df = df.copy()
    df['total_expenses'] = (
        df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
        df['travel_expenses'] + df['groceries_utilities'] +
        df['other_monthly_expenses'] + df['current_emi_amount']
    )
    df['debt_to_income_ratio']    = df['current_emi_amount'] / (df['monthly_salary'] + 1)
    df['expense_to_income_ratio'] = df['total_expenses'] / (df['monthly_salary'] + 1)
    df['disposable_income']       = df['monthly_salary'] - df['total_expenses']
    df['affordability_ratio']     = df['disposable_income'] / (df['requested_amount'] + 1)
    df['savings_to_income_ratio'] = df['bank_balance'] / (df['monthly_salary'] + 1)
    df['emergency_fund_months']   = df['emergency_fund'] / (df['total_expenses'] + 1)
    df['credit_score_band']       = pd.cut(
        df['credit_score'], bins=[0, 579, 669, 739, 799, 850],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    df['loan_to_income_ratio']      = df['requested_amount'] / (df['monthly_salary'] * 12 + 1)
    df['risk_score']                = (
        df['debt_to_income_ratio'] * 0.30 +
        df['expense_to_income_ratio'] * 0.25 +
        df['loan_to_income_ratio'] * 0.25 +
        (1 - df['credit_score'] / 850) * 0.20
    )
    df['salary_credit_interaction'] = df['monthly_salary'] * df['credit_score'] / 100000
    df['income_per_dependent']      = df['disposable_income'] / (df['dependents'] + 1)
    df['rent_burden']               = df['monthly_rent'] / (df['monthly_salary'] + 1)
    df['emi_burden']                = df['current_emi_amount'] / (df['monthly_salary'] + 1)
    return df

# ── Page Header ──
st.title("🔮 EMI Prediction")
st.markdown("Fill in the customer details below to get EMI eligibility and maximum EMI amount.")
st.divider()

# ── Input Form ──
with st.form("prediction_form"):

    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age            = st.number_input("Age", min_value=25, max_value=60, value=35)
        gender         = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    with col2:
        education      = st.selectbox("Education",
                            ["High School", "Graduate", "Post Graduate", "Professional"])
        employment_type= st.selectbox("Employment Type",
                            ["Private", "Government", "Self-employed"])
        company_type   = st.selectbox("Company Type",
                            ["Small", "Startup", "Mid-size", "Large Indian", "MNC"])
    with col3:
        years_of_employment = st.number_input("Years of Employment", min_value=0.5,
                                               max_value=36.0, value=5.0, step=0.5)
        house_type     = st.selectbox("House Type", ["Rented", "Family", "Own"])
        dependents     = st.number_input("Dependents", min_value=0, max_value=4, value=1)

    st.divider()
    st.subheader("💰 Financial Information")
    col4, col5, col6 = st.columns(3)
    with col4:
        monthly_salary      = st.number_input("Monthly Salary (₹)", min_value=15000,
                                               max_value=200000, value=50000, step=1000)
        monthly_rent        = st.number_input("Monthly Rent (₹)", min_value=0,
                                               max_value=80000, value=10000, step=500)
        bank_balance        = st.number_input("Bank Balance (₹)", min_value=0,
                                               max_value=2000000, value=100000, step=5000)
    with col5:
        emergency_fund      = st.number_input("Emergency Fund (₹)", min_value=0,
                                               max_value=1000000, value=50000, step=5000)
        credit_score        = st.number_input("Credit Score", min_value=300,
                                               max_value=850, value=700)
        existing_loans      = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount  = st.number_input("Current EMI Amount (₹)", min_value=0,
                                               max_value=60000, value=0, step=500)
    with col6:
        school_fees         = st.number_input("School Fees (₹)", min_value=0,
                                               max_value=15000, value=0, step=500)
        college_fees        = st.number_input("College Fees (₹)", min_value=0,
                                               max_value=25000, value=0, step=500)
        travel_expenses     = st.number_input("Travel Expenses (₹)", min_value=600,
                                               max_value=30000, value=3000, step=200)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=1800,
                                               max_value=71200, value=10000, step=500)
        other_monthly_expenses = st.number_input("Other Expenses (₹)", min_value=600,
                                               max_value=42900, value=3000, step=200)

    st.divider()
    st.subheader("🏦 Loan Request Details")
    col7, col8, col9 = st.columns(3)
    with col7:
        emi_scenario     = st.selectbox("EMI Scenario", [
            "E-commerce Shopping EMI", "Home Appliances EMI",
            "Vehicle EMI", "Personal Loan EMI", "Education EMI"
        ])
    with col8:
        requested_amount = st.number_input("Requested Amount (₹)", min_value=10000,
                                            max_value=1500000, value=200000, step=5000)
    with col9:
        requested_tenure = st.number_input("Requested Tenure (months)",
                                            min_value=3, max_value=84, value=24)

    submitted = st.form_submit_button("🔮 Predict EMI Eligibility", use_container_width=True)

# ── Prediction ──
if submitted:
    encode = {
        'gender'          : {'Female': 0, 'Male': 1},
        'marital_status'  : {'Single': 0, 'Married': 1},
        'education'       : {'High School': 0, 'Graduate': 1,
                             'Post Graduate': 2, 'Professional': 3},
        'employment_type' : {'Private': 0, 'Government': 1, 'Self-employed': 2},
        'company_type'    : {'Small': 0, 'Startup': 1, 'Mid-size': 2,
                             'Large Indian': 3, 'MNC': 4},
        'house_type'      : {'Rented': 0, 'Family': 1, 'Own': 2},
        'existing_loans'  : {'No': 0, 'Yes': 1},
        'emi_scenario'    : {'E-commerce Shopping EMI': 0, 'Home Appliances EMI': 1,
                             'Vehicle EMI': 2, 'Personal Loan EMI': 3, 'Education EMI': 4}
    }

    input_data = pd.DataFrame([{
        'age'                   : age,
        'gender'                : encode['gender'][gender],
        'marital_status'        : encode['marital_status'][marital_status],
        'education'             : encode['education'][education],
        'monthly_salary'        : monthly_salary,
        'employment_type'       : encode['employment_type'][employment_type],
        'years_of_employment'   : years_of_employment,
        'company_type'          : encode['company_type'][company_type],
        'house_type'            : encode['house_type'][house_type],
        'monthly_rent'          : monthly_rent,
        'dependents'            : dependents,
        'school_fees'           : school_fees,
        'college_fees'          : college_fees,
        'travel_expenses'       : travel_expenses,
        'groceries_utilities'   : groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'existing_loans'        : encode['existing_loans'][existing_loans],
        'current_emi_amount'    : current_emi_amount,
        'credit_score'          : credit_score,
        'bank_balance'          : bank_balance,
        'emergency_fund'        : emergency_fund,
        'emi_scenario'          : encode['emi_scenario'][emi_scenario],
        'requested_amount'      : requested_amount,
        'requested_tenure'      : requested_tenure
    }])

    # Engineer features
    input_engineered = engineer_features(input_data)
    input_final = input_engineered[feature_cols]

    # Predict
    eligibility_num = classifier.predict(input_final)[0]
    max_emi         = regressor.predict(input_final)[0]
    proba           = classifier.predict_proba(input_final)[0]

    eligibility_map = {0: 'Not Eligible', 1: 'High Risk', 2: 'Eligible'}
    eligibility     = eligibility_map[eligibility_num]

    st.divider()
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if eligibility == 'Eligible':
            st.success(f"### 🟢 {eligibility}")
        elif eligibility == 'High Risk':
            st.warning(f"### 🟡 {eligibility}")
        else:
            st.error(f"### 🔴 {eligibility}")

    with col2:
        st.metric("💰 Max Monthly EMI", f"₹{max_emi:,.0f}")

    with col3:
        st.metric("📉 Risk Score",
                  f"{input_engineered['risk_score'].values[0]:.3f}")

    st.divider()
    st.subheader("🎯 Prediction Confidence")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Not Eligible", f"{proba[0]*100:.1f}%")
    with col5:
        st.metric("High Risk",    f"{proba[1]*100:.1f}%")
    with col6:
        st.metric("Eligible",     f"{proba[2]*100:.1f}%")

    st.divider()
    st.subheader("📋 Financial Summary")
    total_exp = (monthly_rent + school_fees + college_fees +
                 travel_expenses + groceries_utilities +
                 other_monthly_expenses + current_emi_amount)
    disposable = monthly_salary - total_exp

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        st.metric("Monthly Salary",    f"₹{monthly_salary:,}")
    with col8:
        st.metric("Total Expenses",    f"₹{total_exp:,}")
    with col9:
        st.metric("Disposable Income", f"₹{disposable:,}")
    with col10:
        st.metric("Credit Score",      credit_score)