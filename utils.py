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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


tickers = [
    "IRES.IR",
    "KRZ.IR",
    "GL9.IR",
    "UPR.IR",
    "GRP.IR",
    "KRX.IR",
    "KMR.IR"
]

period = "8y"
weekly_threshold = 0.01
transaction_cost = 0.001
train_ratio = 0.70
random_seed = 42
risk_free_rate = 0.0

# =========================
# DATA FUNCTIONS
# =========================
def download_stock_data(ticker):
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)
    return df


def convert_to_weekly(df):
    weekly = pd.DataFrame()
    weekly["Open"]   = df["Open"].resample("W-FRI").first()
    weekly["High"]   = df["High"].resample("W-FRI").max()
    weekly["Low"]    = df["Low"].resample("W-FRI").min()
    weekly["Close"]  = df["Close"].resample("W-FRI").last()
    weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    weekly.dropna(inplace=True)
    weekly["Return"] = weekly["Close"].pct_change()
    return weekly


def create_features(df):
    """
    All features are lagged by 1 period (.shift(1)) so that at prediction
    time we only use information available *before* the current bar closes.
    This eliminates look-ahead / feature leakage.
    """
    close = df["Close"]

    df["SMA10"]  = close.rolling(10).mean().shift(1)
    df["SMA20"]  = close.rolling(20).mean().shift(1)
    df["EMA12"]  = close.ewm(span=12, adjust=False).mean().shift(1)
    df["EMA26"]  = close.ewm(span=26, adjust=False).mean().shift(1)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"]  = (100 - (100 / (1 + rs))).shift(1)
    df["MACD"] = (close.ewm(span=12, adjust=False).mean()
                  - close.ewm(span=26, adjust=False).mean()).shift(1)

    # Extra features that add signal without leaking
    df["Return_lag1"] = df["Return"].shift(1)   # last week's return
    df["Return_lag2"] = df["Return"].shift(2)
    df["Volatility"]  = df["Return"].rolling(4).std().shift(1)  # 4-week vol

    return df

# =========================
# METRICS
# =========================
def sharpe_ratio(r, rf=risk_free_rate):
    """Annualised Sharpe (weekly returns, 52 weeks/year)."""
    excess = r - rf / 52
    return (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(52)


def max_drawdown(cum):
    peak = cum.cummax()
    dd   = (cum - peak) / (peak + 1e-9)
    return dd.min()


def win_rate(r):
    active = r[r != 0]           # exclude flat / no-position weeks
    if len(active) == 0:
        return np.nan
    return (active > 0).mean()

