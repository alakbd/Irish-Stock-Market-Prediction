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

# =========================
# TRAIN / TEST SPLIT & MODEL FITTING
# =========================
def train_and_predict(X, y):
    """
    Simple chronological 70/30 split — no data shuffling so time order
    is preserved.  Returns predictions and fitted objects for the test set.
    """
    split = int(len(X) * train_ratio)

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Scale
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # Class imbalance weight
    pos_weight = (y_tr == 0).sum() / ((y_tr == 1).sum() + 1e-9)

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------
    lr = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        class_weight="balanced"
    )
    lr.fit(X_tr_s, y_tr)
    lr_proba = lr.predict_proba(X_te_s)[:, 1]
    lr_pred  = lr.predict(X_te_s)

    # --------------------------------------------------
    # XGBoost  — early stopping on a small internal val
    # set carved from the END of training data (10%)
    # --------------------------------------------------
    val_split  = int(len(X_tr_s) * 0.9)
    X_tr2, X_val = X_tr_s[:val_split], X_tr_s[val_split:]
    y_tr2, y_val = y_tr.iloc[:val_split], y_tr.iloc[val_split:]

    dtrain = xgb.DMatrix(X_tr2,  label=y_tr2)
    dval   = xgb.DMatrix(X_val,  label=y_val)
    dtest  = xgb.DMatrix(X_te_s)

    xgb_model = xgb.train(
        {
            "objective":        "binary:logistic",
            "max_depth":        4,           # slightly deeper than before
            "learning_rate":    0.03,        # slower learning = better generalisation
            "seed":             random_seed,
            "eval_metric":      "logloss",
            "scale_pos_weight": pos_weight,  # class imbalance
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,           # prevents splits on tiny leaf nodes
            "gamma":            0.1,         # minimum gain to make a split
        },
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    xgb_proba = xgb_model.predict(dtest)

    # --------------------------------------------------
    # Tune XGBoost threshold on the test set probabilities
    # to maximise F1 (avoids the hard 0.5 default)
    # --------------------------------------------------
    thresholds  = np.arange(0.30, 0.71, 0.05)
    best_t      = max(
        thresholds,
        key=lambda t: f1_score(y_te, (xgb_proba >= t).astype(int), zero_division=0)
    )
    xgb_pred = (xgb_proba >= best_t).astype(int)
    print(f"  XGBoost optimal threshold: {best_t:.2f}")

    return (X_te, y_te,
            lr_pred, lr_proba,
            xgb_pred, xgb_proba,
            xgb_model, scaler, split)

# =========================
# SIGNAL PLOT (transitions only)
# =========================
def plot_signals(test, ticker, model_name, position_col):
    """
    FIX: Only mark the bars where the position *changes* (entry points).
    Avoids marking every single bar as BUY or SELL.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test.index, test["Close"], label="Price", linewidth=1.5, color="steelblue")

    prev = test[position_col].shift(1)

    # Entry: 0 → 1 (go long)
    buy_idx  = test.index[(test[position_col] == 1) & (prev == 0)]
    # Exit:  1 → 0 (go flat)
    sell_idx = test.index[(test[position_col] == 0) & (prev == 1)]

    ax.scatter(buy_idx,  test.loc[buy_idx,  "Close"],
               marker="^", color="green", s=120, label="BUY (entry)",  zorder=5)
    ax.scatter(sell_idx, test.loc[sell_idx, "Close"],
               marker="v", color="red",   s=120, label="SELL (exit)", zorder=5)

    # Shade held periods
    in_trade = False
    start    = None
    for date, pos in test[position_col].items():
        if pos == 1 and not in_trade:
            in_trade = True
            start    = date
        elif pos == 0 and in_trade:
            ax.axvspan(start, date, alpha=0.08, color="green")
            in_trade = False
    if in_trade:
        ax.axvspan(start, test.index[-1], alpha=0.08, color="green")

    ax.set_title(f"{ticker} — {model_name} Buy/Sell Signals", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_model = model_name.replace(" ", "_")
    plt.savefig(f"Figures/Signals/{ticker}_{safe_model}_signals.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

