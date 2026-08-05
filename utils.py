import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import streamlit as st

# Constants
FEATURES_LIST = ["SMA10", "SMA20", "EMA12", "EMA26", "RSI", "MACD", 
                 "Return_lag1", "Return_lag2", "Volatility"]

# Irish stock tickers
IRISH_TICKERS = ["IRES.IR", "KRZ.IR", "GL9.IR", "UPR.IR", "GRP.IR", "KRX.IR", "KMR.IR"]

def get_ticker_list():
    """Return list of Irish stock tickers"""
    return IRISH_TICKERS

@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker, period="8y"):
    """Download stock data with caching"""
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            st.warning(f"No data found for {ticker}")
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error downloading {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def process_stock_data(ticker, period="8y", threshold=0.01):
    """Process stock data and create features with caching"""
    df = download_stock_data(ticker, period)
    if df is None or df.empty:
        return None
    
    # Convert to weekly
    weekly = pd.DataFrame()
    weekly["Open"] = df["Open"].resample("W-FRI").first()
    weekly["High"] = df["High"].resample("W-FRI").max()
    weekly["Low"] = df["Low"].resample("W-FRI").min()
    weekly["Close"] = df["Close"].resample("W-FRI").last()
    weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    weekly.dropna(inplace=True)
    weekly["Return"] = weekly["Close"].pct_change()
    
    # Create features
    close = weekly["Close"]
    weekly["SMA10"] = close.rolling(10).mean().shift(1)
    weekly["SMA20"] = close.rolling(20).mean().shift(1)
    weekly["EMA12"] = close.ewm(span=12, adjust=False).mean().shift(1)
    weekly["EMA26"] = close.ewm(span=26, adjust=False).mean().shift(1)
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    weekly["RSI"] = (100 - (100 / (1 + rs))).shift(1)
    weekly["MACD"] = (close.ewm(span=12, adjust=False).mean() 
                      - close.ewm(span=26, adjust=False).mean()).shift(1)
    
    weekly["Return_lag1"] = weekly["Return"].shift(1)
    weekly["Return_lag2"] = weekly["Return"].shift(2)
    weekly["Volatility"] = weekly["Return"].rolling(4).std().shift(1)
    
    # Target variable
    weekly["Target"] = np.where(weekly["Return"].shift(-1) > threshold, 1, 0)
    weekly.dropna(inplace=True)
    
    return weekly

@st.cache_resource(show_spinner=False)
def train_models(weekly_df, features_list, train_ratio=0.70, random_seed=42):
    """Train models with caching"""
    X = weekly_df[features_list]
    y = weekly_df["Target"]
    
    # Check if both classes exist
    if len(y.unique()) < 2:
        st.warning("Only one class present in target variable")
        return None, None, None, None, None, None, None, None, None
    
    # Split data chronologically
    split = int(len(X) * train_ratio)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    
    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    # =========================
    # Logistic Regression
    # =========================
    lr = LogisticRegression(
        max_iter=1000, 
        random_state=random_seed, 
        class_weight="balanced"
    )
    lr.fit(X_tr_s, y_tr)
    lr_proba = lr.predict_proba(X_te_s)[:, 1]
    lr_pred = lr.predict(X_te_s)
    
    # =========================
    # XGBoost
    # =========================
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
            "seed": random_seed,
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
    
    # Optimize threshold
    thresholds = np.arange(0.30, 0.71, 0.05)
    best_t = max(
        thresholds, 
        key=lambda t: f1_score(y_te, (xgb_proba >= t).astype(int), zero_division=0)
    )
    xgb_pred = (xgb_proba >= best_t).astype(int)
    
    return X_te_s, y_te, lr_pred, lr_proba, xgb_pred, xgb_proba, xgb_model, scaler, split

def calculate_metrics(y_test, lr_pred, xgb_pred, lr_proba, xgb_proba):
    """Calculate classification metrics"""
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
        results["ROC-AUC"] = [
            roc_auc_score(y_test, lr_proba),
            roc_auc_score(y_test, xgb_proba)
        ]
    else:
        results["ROC-AUC"] = [np.nan, np.nan]
    
    return results

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio"""
    excess = returns - risk_free_rate / 52
    if excess.std() == 0:
        return 0
    return (excess.mean() / excess.std()) * np.sqrt(52)

def calculate_max_drawdown(cum_returns):
    """Calculate maximum drawdown"""
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / (peak + 1e-9)
    return dd.min()

def calculate_win_rate(returns):
    """Calculate win rate"""
    active = returns[returns != 0]
    if len(active) == 0:
        return np.nan
    return (active > 0).mean()
