import pandas as pd
import numpy as np
import joblib
import json

# ── Paths ──
CLASSIFIER_PATH  = 'models/best_classifier.pkl'
REGRESSOR_PATH   = 'models/best_regressor.pkl'
SCALER_PATH      = 'models/scaler.pkl'
FEATURE_PATH     = 'data/processed/feature_cols_engineered.json'
DATA_PATH        = 'data/processed/df_engineered.csv'

# ── Load models once ──
@st.cache_resource
def load_models():
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor  = joblib.load(REGRESSOR_PATH)
    scaler     = joblib.load(SCALER_PATH)
    return classifier, regressor, scaler

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_feature_cols():
    with open(FEATURE_PATH) as f:
        return json.load(f)

# ── Encoding maps ──
ENCODE_MAPS = {
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

ELIGIBILITY_MAP = {0: 'Not Eligible', 1: 'High Risk', 2: 'Eligible'}
ELIGIBILITY_COLOR = {
    'Not Eligible': '🔴',
    'High Risk'   : '🟡',
    'Eligible'    : '🟢'
}

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
    df['loan_to_income_ratio']       = df['requested_amount'] / (df['monthly_salary'] * 12 + 1)
    df['risk_score']                 = (
        df['debt_to_income_ratio'] * 0.30 +
        df['expense_to_income_ratio'] * 0.25 +
        df['loan_to_income_ratio'] * 0.25 +
        (1 - df['credit_score'] / 850) * 0.20
    )
    df['salary_credit_interaction']  = df['monthly_salary'] * df['credit_score'] / 100000
    df['income_per_dependent']       = df['disposable_income'] / (df['dependents'] + 1)
    df['rent_burden']                = df['monthly_rent'] / (df['monthly_salary'] + 1)
    df['emi_burden']                 = df['current_emi_amount'] / (df['monthly_salary'] + 1)

    return df