import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 EDA Dashboard")
st.markdown("Explore the dataset visually with interactive charts.")
st.divider()

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/df_engineered.csv')

df = load_data()

# Decode labels for display
df['eligibility_label'] = df['emi_eligibility'].map(
    {0: 'Not Eligible', 1: 'High Risk', 2: 'Eligible'})
df['scenario_label'] = df['emi_scenario'].map(
    {0: 'E-commerce', 1: 'Home Appliances',
     2: 'Vehicle', 3: 'Personal Loan', 4: 'Education'})

# ── Dataset Overview ──
st.subheader("📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",   f"{len(df):,}")
col2.metric("Total Features",  "38")
col3.metric("EMI Scenarios",   "5")
col4.metric("Target Classes",  "3")
st.divider()

# ── Chart 1: Eligibility Distribution ──
st.subheader("🎯 EMI Eligibility Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['eligibility_label'].value_counts()
    colors = ['#e74c3c', '#e67e22', '#2ecc71']
    ax.bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)',
                ha='center', fontsize=9)
    ax.set_title('EMI Eligibility Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(counts.values, labels=counts.index,
           colors=colors, autopct='%1.1f%%',
           startangle=90, wedgeprops={'edgecolor': 'white'})
    ax.set_title('Eligibility Share', fontweight='bold')
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Chart 2: EMI Scenario Distribution ──
st.subheader("🏦 EMI Scenario Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
scenario_counts = df['scenario_label'].value_counts()
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
bars = ax.barh(scenario_counts.index, scenario_counts.values,
               color=colors, edgecolor='white', height=0.5)
for bar, val in zip(bars, scenario_counts.values):
    ax.text(val + 200, bar.get_y() + bar.get_height()/2,
            f'{val:,} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
ax.set_title('EMI Scenario Distribution', fontweight='bold')
ax.set_xlabel('Number of Records')
st.pyplot(fig)
plt.close()
st.divider()

# ── Chart 3: Salary by Eligibility ──
st.subheader("💰 Monthly Salary by Eligibility")
fig, ax = plt.subplots(figsize=(10, 4))
colors_map = {'Not Eligible': '#e74c3c', 'High Risk': '#e67e22', 'Eligible': '#2ecc71'}
for label, color in colors_map.items():
    subset = df[df['eligibility_label'] == label]['monthly_salary']
    ax.hist(subset, bins=50, alpha=0.6, label=label, color=color, edgecolor='white')
ax.set_title('Monthly Salary Distribution by EMI Eligibility', fontweight='bold')
ax.set_xlabel('Monthly Salary (INR)')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)
plt.close()
st.divider()

# ── Chart 4: Credit Score by Eligibility ──
st.subheader("📈 Credit Score by Eligibility")
fig, ax = plt.subplots(figsize=(10, 4))
for label, color in colors_map.items():
    subset = df[df['eligibility_label'] == label]['credit_score']
    ax.hist(subset, bins=50, alpha=0.6, label=label, color=color, edgecolor='white')
ax.set_title('Credit Score Distribution by EMI Eligibility', fontweight='bold')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)
plt.close()
st.divider()

# ── Chart 5: Max EMI by Scenario ──
st.subheader("📊 Average Max Monthly EMI by Scenario")
fig, ax = plt.subplots(figsize=(10, 4))
scenario_emi = df.groupby('scenario_label')['max_monthly_emi'].mean().sort_values(ascending=False)
bars = ax.bar(scenario_emi.index, scenario_emi.values,
              color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, scenario_emi.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            f'₹{val:,.0f}', ha='center', fontsize=9)
ax.set_title('Average Max Monthly EMI by Scenario', fontweight='bold')
ax.set_xlabel('EMI Scenario')
ax.set_ylabel('Average Max Monthly EMI (INR)')
st.pyplot(fig)
plt.close()
st.divider()

# ── Chart 6: Correlation Heatmap ──
st.subheader("🔥 Feature Correlation Heatmap")
key_features = ['monthly_salary', 'credit_score', 'bank_balance',
                'emergency_fund', 'risk_score', 'disposable_income',
                'debt_to_income_ratio', 'loan_to_income_ratio',
                'emi_eligibility', 'max_monthly_emi']
fig, ax = plt.subplots(figsize=(12, 8))
corr = df[key_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, annot_kws={'size': 9}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()
st.markdown("""
**📌 Key Business Insights:**
- 🔴 Not Eligible class dominates — dataset is imbalanced
- 💰 Higher salary strongly linked to Eligible class
- 📈 Credit score is a strong predictor of eligibility
- 🏠 Own house customers have more EMI capacity
- 🚗 Vehicle EMI has highest average max monthly EMI
""")