import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            border-left: 4px solid #3498db;
        }
        .stMetric {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ── Home Page ──
st.markdown('<p class="main-header">💰 EMIPredict AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Financial Risk Assessment Platform</p>',
            unsafe_allow_html=True)

st.divider()

# ── Overview cards ──
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="📊 Total Records",    value="4,04,800")
with col2:
    st.metric(label="🎯 Classifier AUC",   value="99.65%")
with col3:
    st.metric(label="📈 Regressor R²",     value="0.9889")
with col4:
    st.metric(label="💡 Features Used",    value="38")

st.divider()

# ── Navigation guide ──
st.subheader("📌 Navigate using the sidebar")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **🔮 EMI Prediction**
    Enter customer financial details and get:
    - EMI Eligibility (Eligible / High Risk / Not Eligible)
    - Maximum safe monthly EMI amount
    """)

    st.success("""
    **📊 EDA Dashboard**
    Explore the dataset visually:
    - Target distributions
    - Feature correlations
    - Scenario analysis
    """)

with col2:
    st.warning("""
    **🏆 Model Performance**
    Compare all trained models:
    - Classification metrics (Accuracy, F1, AUC)
    - Regression metrics (RMSE, MAE, R²)
    - Feature importance charts
    """)

    st.error("""
    **🗄️ Data Management**
    CRUD operations:
    - View financial records
    - Add new customer records
    - Update and delete records
    """)

st.divider()

st.markdown("""
    <p style='text-align:center; color:grey; font-size:0.85rem'>
    Built with Python · XGBoost · MLflow · Streamlit &nbsp;|&nbsp;
    Domain: FinTech & Banking
    </p>
""", unsafe_allow_html=True)