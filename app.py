import streamlit as st
from utils import tickers, run_analysis

# Page settings
st.set_page_config(
    page_title="Irish Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Irish Stock Market Prediction using Machine Learning")

st.markdown("""
This application predicts weekly Buy/Sell signals for Irish stocks using:

- Logistic Regression
- XGBoost
""")

# Sidebar
st.sidebar.header("Analysis Settings")

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    tickers
)

if st.sidebar.button("Run Analysis"):
    run_analysis(selected_ticker)
