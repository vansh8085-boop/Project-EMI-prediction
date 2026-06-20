import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Data Management", page_icon="🗄️", layout="wide")

st.title("🗄️ Data Management")
st.markdown("View, add, update and delete financial records.")
st.divider()

# ── Load Data ──
DATA_PATH = 'data/processed/df_engineered.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def save_data(df):
    df.to_csv(DATA_PATH, index=False)
    st.cache_data.clear()

# Decode maps for display
DECODE = {
    'gender'          : {0: 'Female', 1: 'Male'},
    'marital_status'  : {0: 'Single', 1: 'Married'},
    'education'       : {0: 'High School', 1: 'Graduate',
                         2: 'Post Graduate', 3: 'Professional'},
    'employment_type' : {0: 'Private', 1: 'Government', 2: 'Self-employed'},
    'company_type'    : {0: 'Small', 1: 'Startup', 2: 'Mid-size',
                         3: 'Large Indian', 4: 'MNC'},
    'house_type'      : {0: 'Rented', 1: 'Family', 2: 'Own'},
    'existing_loans'  : {0: 'No', 1: 'Yes'},
    'emi_scenario'    : {0: 'E-commerce', 1: 'Home Appliances',
                         2: 'Vehicle', 3: 'Personal Loan', 4: 'Education'},
    'emi_eligibility' : {0: 'Not Eligible', 1: 'High Risk', 2: 'Eligible'}
}

df = load_data()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📋 View Records", "➕ Add Record",
                                   "✏️ Update Record", "🗑️ Delete Record"])

# ════════════════════════════════
# TAB 1 — VIEW
# ════════════════════════════════
with tab1:
    st.subheader("📋 View Financial Records")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{df.shape[1]}")
    with col3:
        eligible_pct = (df['emi_eligibility'] == 2).sum() / len(df) * 100
        st.metric("Eligible %", f"{eligible_pct:.1f}%")

    st.divider()

    # Filter options
    col4, col5 = st.columns(2)
    with col4:
        eligibility_filter = st.selectbox(
            "Filter by Eligibility",
            ["All", "Eligible", "High Risk", "Not Eligible"]
        )
    with col5:
        n_rows = st.slider("Number of rows to display", 10, 200, 50)

    # Apply filter
    df_display = df.copy()
    if eligibility_filter != "All":
        reverse_map = {'Eligible': 2, 'High Risk': 1, 'Not Eligible': 0}
        df_display  = df_display[
            df_display['emi_eligibility'] == reverse_map[eligibility_filter]
        ]

    # Decode for display
    for col, mapping in DECODE.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].map(mapping)

    # Show key columns only
    display_cols = ['age', 'gender', 'education', 'monthly_salary',
                    'employment_type', 'credit_score', 'bank_balance',
                    'emi_scenario', 'requested_amount', 'emi_eligibility',
                    'max_monthly_emi']
    display_cols = [c for c in display_cols if c in df_display.columns]

    st.dataframe(df_display[display_cols].head(n_rows), use_container_width=True)
    st.caption(f"Showing {min(n_rows, len(df_display)):,} of {len(df_display):,} records")

# ════════════════════════════════
# TAB 2 — ADD
# ════════════════════════════════
with tab2:
    st.subheader("➕ Add New Financial Record")

    with st.form("add_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age             = st.number_input("Age", 25, 60, 35)
            gender          = st.selectbox("Gender", ["Female", "Male"])
            marital_status  = st.selectbox("Marital Status", ["Single", "Married"])
            education       = st.selectbox("Education",
                                ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type",
                                ["Private", "Government", "Self-employed"])

        with col2:
            company_type        = st.selectbox("Company Type",
                                    ["Small", "Startup", "Mid-size", "Large Indian", "MNC"])
            years_of_employment = st.number_input("Years of Employment", 0.5, 36.0, 5.0)
            house_type          = st.selectbox("House Type", ["Rented", "Family", "Own"])
            dependents          = st.number_input("Dependents", 0, 4, 1)
            monthly_salary      = st.number_input("Monthly Salary (₹)", 15000, 200000, 50000)

        with col3:
            monthly_rent           = st.number_input("Monthly Rent (₹)", 0, 80000, 0)
            bank_balance           = st.number_input("Bank Balance (₹)", 0, 2000000, 100000)
            emergency_fund         = st.number_input("Emergency Fund (₹)", 0, 1000000, 50000)
            credit_score           = st.number_input("Credit Score", 300, 850, 700)
            existing_loans         = st.selectbox("Existing Loans", ["No", "Yes"])

        col4, col5, col6 = st.columns(3)
        with col4:
            current_emi_amount  = st.number_input("Current EMI (₹)", 0, 60000, 0)
            school_fees         = st.number_input("School Fees (₹)", 0, 15000, 0)
            college_fees        = st.number_input("College Fees (₹)", 0, 25000, 0)
        with col5:
            travel_expenses     = st.number_input("Travel Expenses (₹)", 600, 30000, 3000)
            groceries_utilities = st.number_input("Groceries & Utilities (₹)", 1800, 71200, 10000)
            other_monthly_expenses = st.number_input("Other Expenses (₹)", 600, 42900, 3000)
        with col6:
            emi_scenario        = st.selectbox("EMI Scenario",
                                    ["E-commerce", "Home Appliances", "Vehicle",
                                     "Personal Loan", "Education"])
            requested_amount    = st.number_input("Requested Amount (₹)", 10000, 1500000, 200000)
            requested_tenure    = st.number_input("Tenure (months)", 3, 84, 24)

        add_submitted = st.form_submit_button("➕ Add Record", use_container_width=True)

    if add_submitted:
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
            'emi_scenario'    : {'E-commerce': 0, 'Home Appliances': 1,
                                 'Vehicle': 2, 'Personal Loan': 3, 'Education': 4}
        }

        total_expenses = (monthly_rent + school_fees + college_fees +
                         travel_expenses + groceries_utilities +
                         other_monthly_expenses + current_emi_amount)

        new_row = {
            'age'                    : age,
            'gender'                 : encode['gender'][gender],
            'marital_status'         : encode['marital_status'][marital_status],
            'education'              : encode['education'][education],
            'monthly_salary'         : monthly_salary,
            'employment_type'        : encode['employment_type'][employment_type],
            'years_of_employment'    : years_of_employment,
            'company_type'           : encode['company_type'][company_type],
            'house_type'             : encode['house_type'][house_type],
            'monthly_rent'           : monthly_rent,
            'dependents'             : dependents,
            'school_fees'            : school_fees,
            'college_fees'           : college_fees,
            'travel_expenses'        : travel_expenses,
            'groceries_utilities'    : groceries_utilities,
            'other_monthly_expenses' : other_monthly_expenses,
            'existing_loans'         : encode['existing_loans'][existing_loans],
            'current_emi_amount'     : current_emi_amount,
            'credit_score'           : credit_score,
            'bank_balance'           : bank_balance,
            'emergency_fund'         : emergency_fund,
            'emi_scenario'           : encode['emi_scenario'][emi_scenario],
            'requested_amount'       : requested_amount,
            'requested_tenure'       : requested_tenure,
            'emi_eligibility'        : -1,
            'max_monthly_emi'        : 0.0,
            'total_expenses'         : total_expenses,
            'debt_to_income_ratio'   : current_emi_amount / (monthly_salary + 1),
            'expense_to_income_ratio': total_expenses / (monthly_salary + 1),
            'disposable_income'      : monthly_salary - total_expenses,
            'affordability_ratio'    : (monthly_salary - total_expenses) / (requested_amount + 1),
            'savings_to_income_ratio': bank_balance / (monthly_salary + 1),
            'emergency_fund_months'  : emergency_fund / (total_expenses + 1),
            'credit_score_band'      : 2.0,
            'loan_to_income_ratio'   : requested_amount / (monthly_salary * 12 + 1),
            'risk_score'             : 0.0,
            'salary_credit_interaction': monthly_salary * credit_score / 100000,
            'income_per_dependent'   : (monthly_salary - total_expenses) / (dependents + 1),
            'rent_burden'            : monthly_rent / (monthly_salary + 1),
            'emi_burden'             : current_emi_amount / (monthly_salary + 1)
        }

        df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df_new)
        st.success(f"✅ Record added successfully! Total records: {len(df_new):,}")

# ════════════════════════════════
# TAB 3 — UPDATE
# ════════════════════════════════
with tab3:
    st.subheader("✏️ Update Existing Record")

    row_index = st.number_input("Enter Row Index to Update",
                                 min_value=0, max_value=len(df)-1, value=0)

    if st.button("🔍 Load Record"):
        record = df.iloc[row_index]
        st.write("**Current Record:**")
        st.dataframe(pd.DataFrame([record]).T, use_container_width=True)
        st.info("To update this record, use the Delete tab to remove it and Add tab to add the corrected version.")

# ════════════════════════════════
# TAB 4 — DELETE
# ════════════════════════════════
with tab4:
    st.subheader("🗑️ Delete a Record")

    row_index = st.number_input("Enter Row Index to Delete",
                                 min_value=0, max_value=len(df)-1, value=0)

    record = df.iloc[row_index]
    st.write("**Record to be deleted:**")

    display_cols = ['age', 'monthly_salary', 'credit_score',
                    'emi_eligibility', 'max_monthly_emi']
    st.dataframe(pd.DataFrame([record[display_cols]]).T, use_container_width=True)

    confirm = st.checkbox("✅ I confirm I want to delete this record")

    if confirm:
        if st.button("🗑️ Delete Record", type="primary"):
            df_new = df.drop(index=row_index).reset_index(drop=True)
            save_data(df_new)
            st.success(f"✅ Record deleted. Total records remaining: {len(df_new):,}")