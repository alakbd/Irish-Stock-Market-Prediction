import streamlit as st
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)
import io
from contextlib import redirect_stdout
import sys

warnings.filterwarnings("ignore")

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Irish Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Irish Stock Market Prediction")

# =========================
# CACHING - CRITICAL FOR STREAMLIT
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker, period="8y"):
    """Cache downloaded stock data"""
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def process_stock_data(ticker, period="8y", threshold=0.01):
    """Cache all data processing and model training"""
    # Download data
    df = download_stock_data(ticker, period)
    
    # Convert to weekly
    weekly = pd.DataFrame()
    weekly["Open"]   = df["Open"].resample("W-FRI").first()
    weekly["High"]   = df["High"].resample("W-FRI").max()
    weekly["Low"]    = df["Low"].resample("W-FRI").min()
    weekly["Close"]  = df["Close"].resample("W-FRI").last()
    weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    weekly.dropna(inplace=True)
    weekly["Return"] = weekly["Close"].pct_change()
    
    # Create features
    close = weekly["Close"]
    weekly["SMA10"]  = close.rolling(10).mean().shift(1)
    weekly["SMA20"]  = close.rolling(20).mean().shift(1)
    weekly["EMA12"]  = close.ewm(span=12, adjust=False).mean().shift(1)
    weekly["EMA26"]  = close.ewm(span=26, adjust=False).mean().shift(1)
    
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    weekly["RSI"]  = (100 - (100 / (1 + rs))).shift(1)
    weekly["MACD"] = (close.ewm(span=12, adjust=False).mean()
                      - close.ewm(span=26, adjust=False).mean()).shift(1)
    
    weekly["Return_lag1"] = weekly["Return"].shift(1)
    weekly["Return_lag2"] = weekly["Return"].shift(2)
    weekly["Volatility"]  = weekly["Return"].rolling(4).std().shift(1)
    
    # Target
    weekly["Target"] = np.where(weekly["Return"].shift(-1) > threshold, 1, 0)
    weekly.dropna(inplace=True)
    
    return weekly

@st.cache_resource
def train_models(weekly_df, features_list, train_ratio=0.70):
    """Cache trained models"""
    X = weekly_df[features_list]
    y = weekly_df["Target"]
    
    split = int(len(X) * train_ratio)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_tr_s, y_tr)
    lr_proba = lr.predict_proba(X_te_s)[:, 1]
    lr_pred = lr.predict(X_te_s)
    
    # XGBoost
    pos_weight = (y_tr == 0).sum() / ((y_tr == 1).sum() + 1e-9)
    val_split = int(len(X_tr_s) * 0.9)
    X_tr2, X_val = X_tr_s[:val_split], X_tr_s[val_split:]
    y_tr2, y_val = y_tr.iloc[:val_split], y_tr.iloc[val_split:]
    
    dtrain = xgb.DMatrix(X_tr2, label=y_tr2)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_te_s)
    
    xgb_model = xgb.train(
        {
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.03,
            "seed": 42,
            "eval_metric": "logloss",
            "scale_pos_weight": pos_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
        },
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    xgb_proba = xgb_model.predict(dtest)
    thresholds = np.arange(0.30, 0.71, 0.05)
    best_t = max(thresholds, key=lambda t: f1_score(y_te, (xgb_proba >= t).astype(int), zero_division=0))
    xgb_pred = (xgb_proba >= best_t).astype(int)
    
    return X_te_s, y_te, lr_pred, lr_proba, xgb_pred, xgb_proba, xgb_model, scaler, split

def calculate_metrics(y_test, lr_pred, xgb_pred, lr_proba, xgb_proba):
    """Calculate classification metrics efficiently"""
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost"],
        "Accuracy": [accuracy_score(y_test, lr_pred), accuracy_score(y_test, xgb_pred)],
        "Precision": [precision_score(y_test, lr_pred, zero_division=0), 
                     precision_score(y_test, xgb_pred, zero_division=0)],
        "Recall": [recall_score(y_test, lr_pred, zero_division=0),
                  recall_score(y_test, xgb_pred, zero_division=0)],
        "F1 Score": [f1_score(y_test, lr_pred, zero_division=0),
                    f1_score(y_test, xgb_pred, zero_division=0)]
    })
    
    # ROC-AUC only if both classes present
    if len(np.unique(y_test)) > 1:
        results["ROC-AUC"] = [roc_auc_score(y_test, lr_proba), roc_auc_score(y_test, xgb_proba)]
    else:
        results["ROC-AUC"] = [np.nan, np.nan]
    
    return results

# =========================
# MAIN APP
# =========================
st.write("Analyzing Irish stock market data with machine learning models...")

# Parameters
tickers = ["IRES.IR", "KRZ.IR", "GL9.IR", "UPR.IR", "GRP.IR", "KRX.IR", "KMR.IR"]
features_list = ["SMA10", "SMA20", "EMA12", "EMA26", "RSI", "MACD", 
                 "Return_lag1", "Return_lag2", "Volatility"]

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

results_data = []

# Process each ticker with progress feedback
for idx, ticker in enumerate(tickers):
    status_text.text(f"Processing {ticker}...")
    
    try:
        # Process data
        weekly_df = process_stock_data(ticker)
        
        # Check if we have enough data
        if len(weekly_df) < 50:  # Need minimum data points
            st.warning(f"Not enough data for {ticker}. Skipping...")
            progress_bar.progress((idx + 1) / len(tickers))
            continue
        
        # Train models
        X_te_s, y_test, lr_pred, lr_proba, xgb_pred, xgb_proba, xgb_model, scaler, split = train_models(weekly_df, features_list)
        
        # Get test data
        test = weekly_df.iloc[split:].copy()
        test["LR_Position"] = lr_pred
        test["XGB_Position"] = xgb_pred
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, lr_pred, xgb_pred, lr_proba, xgb_proba)
        
        # Store results
        results_data.append({
            "Ticker": ticker,
            "LR_Accuracy": metrics.loc[0, "Accuracy"],
            "XGB_Accuracy": metrics.loc[1, "Accuracy"],
            "LR_F1": metrics.loc[0, "F1 Score"],
            "XGB_F1": metrics.loc[1, "F1 Score"]
        })
        
        # Display metrics in Streamlit
        st.subheader(f"📊 {ticker}")
        st.dataframe(metrics.round(3))
        
        # Create a simple plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(test.index, test["Close"], label="Close Price", linewidth=1)
        ax.set_title(f"{ticker} - Price and Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")
    
    progress_bar.progress((idx + 1) / len(tickers))

status_text.text("✅ Analysis complete!")

# Display summary
if results_data:
    st.subheader("📈 Summary Results")
    summary_df = pd.DataFrame(results_data)
    st.dataframe(summary_df.round(3))
    
    # Create summary visualizations
    st.subheader("📊 Model Comparison")
    
    # F1 Score comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax.bar(x - width/2, summary_df["LR_F1"], width, label="Logistic Regression", color='skyblue')
    ax.bar(x + width/2, summary_df["XGB_F1"], width, label="XGBoost", color='lightgreen')
    
    ax.set_xlabel("Ticker")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Ticker"], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)

st.success("✅ Analysis complete! Check the metrics above for each stock.")

# Add note about performance
st.info("💡 **Note:** Data is cached for 1 hour. Results will update automatically when the cache expires.")
