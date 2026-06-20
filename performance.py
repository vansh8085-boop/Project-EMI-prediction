import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

st.set_page_config(page_title="Model Performance", page_icon="🏆", layout="wide")

st.title("🏆 Model Performance")
st.markdown("Compare all trained models for both classification and regression.")
st.divider()

# ── Classification Results ──
st.subheader("🎯 Classification Models Comparison")

clf_data = {
    'Model'     : ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Val Acc %' : [80.19, 88.81, 90.03, 92.29],
    'Precision %': [92.91, 95.70, 95.56, 96.74],
    'Recall %'  : [80.19, 88.81, 90.03, 92.29],
    'F1 Score %': [85.09, 91.32, 92.04, 93.80],
    'ROC AUC %' : [96.47, 98.46, 99.21, 99.65]
}
clf_df = pd.DataFrame(clf_data)

# Highlight best model
def highlight_best(s):
    is_max = s == s.max()
    return ['background-color: #d5f5e3' if v else '' for v in is_max]

st.dataframe(
    clf_df.style.apply(highlight_best, subset=['Val Acc %', 'F1 Score %', 'ROC AUC %']),
    use_container_width=True
)

st.divider()

# ── Classification Bar Charts ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
metrics_clf = ['Val Acc %', 'F1 Score %', 'ROC AUC %']

for i, metric in enumerate(metrics_clf):
    bars = axes[i].bar(clf_df['Model'], clf_df[metric],
                       color=colors, edgecolor='white', width=0.5)
    axes[i].set_title(metric, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Score (%)')
    axes[i].set_ylim(70, 103)
    axes[i].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, clf_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val}%', ha='center', fontsize=9)

plt.suptitle('Classification Models Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Regression Results ──
st.subheader("📈 Regression Models Comparison")

reg_data = {
    'Model'     : ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Val RMSE'  : [3981.37, 1300.07, 1058.75, 807.94],
    'Val MAE'   : [2826.28, 538.49, 292.64, 310.81],
    'Val R²'    : [0.7301, 0.9712, 0.9809, 0.9889],
    'MAPE %'    : [176.84, 11.47, 7.51, 10.79]
}
reg_df = pd.DataFrame(reg_data)

def highlight_best_reg(s):
    if s.name in ['Val RMSE', 'Val MAE', 'MAPE %']:
        is_best = s == s.min()
    else:
        is_best = s == s.max()
    return ['background-color: #d5f5e3' if v else '' for v in is_best]

st.dataframe(
    reg_df.style.apply(highlight_best_reg,
                       subset=['Val RMSE', 'Val MAE', 'Val R²', 'MAPE %']),
    use_container_width=True
)

st.divider()

# ── Regression Bar Charts ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# RMSE
bars = axes[0].bar(reg_df['Model'], reg_df['Val RMSE'],
                   color=colors, edgecolor='white', width=0.5)
axes[0].set_title('Val RMSE (lower is better)', fontweight='bold')
axes[0].set_ylabel('RMSE (INR)')
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, reg_df['Val RMSE']):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 30,
                 f'{val:,.0f}', ha='center', fontsize=9)

# MAE
bars = axes[1].bar(reg_df['Model'], reg_df['Val MAE'],
                   color=colors, edgecolor='white', width=0.5)
axes[1].set_title('Val MAE (lower is better)', fontweight='bold')
axes[1].set_ylabel('MAE (INR)')
axes[1].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, reg_df['Val MAE']):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 10,
                 f'{val:,.0f}', ha='center', fontsize=9)

# R²
bars = axes[2].bar(reg_df['Model'], reg_df['Val R²'],
                   color=colors, edgecolor='white', width=0.5)
axes[2].set_title('Val R² (higher is better)', fontweight='bold')
axes[2].set_ylabel('R² Score')
axes[2].set_ylim(0.6, 1.02)
axes[2].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, reg_df['Val R²']):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{val}', ha='center', fontsize=9)

plt.suptitle('Regression Models Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Feature Importance ──
st.subheader("🔍 XGBoost Feature Importance")

@st.cache_resource
def load_classifier():
    return joblib.load('models/best_classifier.pkl')

@st.cache_data
def load_features():
    with open('data/processed/feature_cols_engineered.json') as f:
        return json.load(f)

clf     = load_classifier()
feature_cols = load_features()

importance_df = pd.DataFrame({
    'Feature'   : feature_cols,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
               color='#3498db', edgecolor='white')
ax.set_title('Top 15 Feature Importances — XGBoost Classifier',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Best Model Summary ──
st.subheader("🥇 Best Model Selection Summary")

col1, col2 = st.columns(2)
with col1:
    st.success("""
    **✅ Best Classifier: XGBoost**
    - Validation Accuracy : 92.29%
    - F1 Score            : 93.80%
    - ROC AUC             : 99.65%
    - Test Accuracy       : 92.27%
    """)
with col2:
    st.success("""
    **✅ Best Regressor: XGBoost**
    - Validation RMSE : ₹807.94
    - Validation MAE  : ₹310.81
    - Validation R²   : 0.9889
    - Test RMSE       : ₹796.71
    """)