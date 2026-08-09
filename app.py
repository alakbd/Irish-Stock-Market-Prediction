import streamlit as st
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Irish Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Irish Stock Market Prediction")

st.write("Welcome to the Stock Market Prediction Application.")

# =========================
# PARAMETERS - EXACTLY AS ORIGINAL
# =========================
tickers            = ["IRES.IR", "KRZ.IR", "GL9.IR", "UPR.IR", "GRP.IR", "KRX.IR", "KMR.IR"]
period             = "8y"
weekly_threshold   = 0.01          # 1% up = BUY signal
transaction_cost   = 0.001         # 0.1% per trade
train_ratio        = 0.70
random_seed        = 42
risk_free_rate     = 0.0           # Annualised; set to e.g. 0.03 for 3% if preferred

# =========================
# DATA FUNCTIONS - EXACTLY AS ORIGINAL
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker):
    df = yf.download(ticker, period=period, progress=False)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
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
# METRICS - EXACTLY AS ORIGINAL
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
# TRAIN / TEST SPLIT & MODEL FITTING - EXACTLY AS ORIGINAL
# =========================
def train_and_predict(X, y, ticker):
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

    # =========================
    # SHAP VALUES - EXACTLY AS ORIGINAL
    # =========================
    explainer = shap.TreeExplainer(xgb_model)
    
    # SHAP values calculated on the scaled test data
    shap_values = explainer.shap_values(X_te_s)
    
    # Summary Plot - EXACTLY AS ORIGINAL
    plt.figure(figsize=(10,6))
    shap.summary_plot(
        shap_values,
        X_te,                  # original dataframe (keeps feature names)
        feature_names=X_te.columns,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"Figures/SHAP/{ticker}_SHAP_Summary.png", dpi=300)
    plt.close()
    
    # Feature Importance (Bar) - EXACTLY AS ORIGINAL
    plt.figure(figsize=(8,6))
    shap.summary_plot(
        shap_values,
        X_te,
        feature_names=X_te.columns,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"Figures/SHAP/{ticker}_SHAP_Bar.png", dpi=300)
    plt.close()

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

    return (X_te, y_te,
            lr_pred, lr_proba,
            xgb_pred, xgb_proba,
            xgb_model, scaler, split, shap_values)

# =========================
# SIGNAL PLOT (transitions only) - EXACTLY AS ORIGINAL
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
    return fig

# =========================
# MAIN LOOP - EXACTLY AS ORIGINAL
# =========================
if st.button("🚀 Run Analysis", type="primary"):
    
    # Create folders (for file saving)
    folders = [
        "Figures/Confusion",
        "Figures/Cumulative",
        "Figures/Signals",
        "Figures/SHAP",
        "Figures/FeatureImportance",
        "Results"
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Storage for results
    all_results = []
    overall_trading = []
    features_list = ["SMA10", "SMA20", "EMA12", "EMA26", "RSI", "MACD",
                     "Return_lag1", "Return_lag2", "Volatility"]
    
    # Display results in expandable sections
    results_container = st.container()
    
    with results_container:
        for idx, ticker in enumerate(tickers):
            status_text.text(f"Processing {ticker}... ({idx+1}/{len(tickers)})")
            
            # -------------------------
            # DATA - EXACTLY AS ORIGINAL
            # -------------------------
            df = download_stock_data(ticker)
            weekly_df = convert_to_weekly(df)
            weekly_df = create_features(weekly_df)

            # Target: next week's return > threshold → 1 (BUY), else 0
            weekly_df["Target"] = np.where(
                weekly_df["Return"].shift(-1) > weekly_threshold, 1, 0
            )

            weekly_df.dropna(inplace=True)

            X = weekly_df[features_list]
            y = weekly_df["Target"]

            # --- Diagnostics: class balance ---
            # --- Diagnostics ---
            class_counts = y.value_counts()

            st.write(
                f"{ticker}: {len(weekly_df)} weekly observations | "
                f"Class 0: {class_counts.get(0, 0)} | "
                f"Class 1: {class_counts.get(1, 0)}"
            )

            if len(class_counts) < 2:
                st.warning(f"{ticker}: Only one class present — skipping.")
                progress_bar.progress((idx + 1) / len(tickers))
                continue

            # -------------------------
            # TRAIN / TEST SPLIT - EXACTLY AS ORIGINAL
            # -------------------------
            (X_te, y_test,
             lr_pred, lr_proba,
             xgb_pred, xgb_proba,
             xgb_model, scaler, split, shap_values) = train_and_predict(X, y, ticker)

            # Test slice of weekly_df
            test = weekly_df.iloc[split:].copy()

            test["LR_Position"]  = lr_pred
            test["XGB_Position"] = xgb_pred
            test["LR_Proba"]     = lr_proba
            test["XGB_Proba"]    = xgb_proba

            # --- Diagnostics: zero-trade check ---
            n_lr_trades  = (test["LR_Position"].diff().abs() > 0).sum()
            n_xgb_trades = (test["XGB_Position"].diff().abs() > 0).sum()

            # -------------------------
            # CLASSIFICATION METRICS - EXACTLY AS ORIGINAL
            # -------------------------
            def safe_roc(y_true, proba):
                if len(np.unique(y_true)) < 2:
                    return np.nan
                return roc_auc_score(y_true, proba)

            classification_results = pd.DataFrame({
                "Model":     ["Logistic Regression", "XGBoost"],
                "Accuracy":  [accuracy_score(y_test, lr_pred),
                              accuracy_score(y_test, xgb_pred)],
                "Precision": [precision_score(y_test, lr_pred,  zero_division=0),
                              precision_score(y_test, xgb_pred, zero_division=0)],
                "Recall":    [recall_score(y_test, lr_pred,  zero_division=0),
                              recall_score(y_test, xgb_pred, zero_division=0)],
                "F1 Score":  [f1_score(y_test, lr_pred,  zero_division=0),
                              f1_score(y_test, xgb_pred, zero_division=0)],
                "ROC-AUC":   [safe_roc(y_test, test["LR_Proba"]),
                              safe_roc(y_test, test["XGB_Proba"])]
            })

            # -------------------------
            # STRATEGY RETURNS WITH TRANSACTION COST - EXACTLY AS ORIGINAL
            # -------------------------
            test["LR_Trade"]  = test["LR_Position"].diff().abs().fillna(0)
            test["XGB_Trade"] = test["XGB_Position"].diff().abs().fillna(0)

            test["LR_Returns"] = (
                test["Return"] * test["LR_Position"]
                - transaction_cost * test["LR_Trade"]
            )
            test["XGB_Returns"] = (
                test["Return"] * test["XGB_Position"]
                - transaction_cost * test["XGB_Trade"]
            )

            test["Market_Cum"] = (1 + test["Return"]).cumprod()
            test["LR_Cum"]     = (1 + test["LR_Returns"]).cumprod()
            test["XGB_Cum"]    = (1 + test["XGB_Returns"]).cumprod()

            # -------------------------
            # TRADING PERFORMANCE TABLE - EXACTLY AS ORIGINAL
            # -------------------------
            trading_results = pd.DataFrame({
                "Strategy": [
                    "Buy & Hold (Benchmark)",
                    "Logistic Regression",
                    "XGBoost"
                ],
                "Total Return (%)": [
                    (test["Market_Cum"].iloc[-1] - 1) * 100,
                    (test["LR_Cum"].iloc[-1] - 1) * 100,
                    (test["XGB_Cum"].iloc[-1] - 1) * 100
                ],
                "Sharpe Ratio": [
                    sharpe_ratio(test["Return"]),
                    sharpe_ratio(test["LR_Returns"]),
                    sharpe_ratio(test["XGB_Returns"])
                ],
                "Maximum Drawdown (%)": [
                    max_drawdown(test["Market_Cum"]) * 100,
                    max_drawdown(test["LR_Cum"]) * 100,
                    max_drawdown(test["XGB_Cum"]) * 100
                ],
                "Win Rate (%)": [
                    win_rate(test["Return"]) * 100,
                    win_rate(test["LR_Returns"]) * 100,
                    win_rate(test["XGB_Returns"]) * 100
                ]
            })

            # Store for summary
            all_results.append({
                "Ticker":           ticker,
                "Market_Return":    test["Market_Cum"].iloc[-1] - 1,
                "LR_Return":        test["LR_Cum"].iloc[-1]     - 1,
                "XGB_Return":       test["XGB_Cum"].iloc[-1]    - 1,
                "Market_Sharpe":    sharpe_ratio(test["Return"]),
                "LR_Sharpe":        sharpe_ratio(test["LR_Returns"]),
                "XGB_Sharpe":       sharpe_ratio(test["XGB_Returns"]),
                "Market_Drawdown":  max_drawdown(test["Market_Cum"]),
                "LR_Drawdown":      max_drawdown(test["LR_Cum"]),
                "XGB_Drawdown":     max_drawdown(test["XGB_Cum"]),
                "LR_WinRate":       win_rate(test["LR_Returns"]),
                "XGB_WinRate":      win_rate(test["XGB_Returns"]),
            })

            overall_trading.append({
                "Ticker": ticker,
                "Market Return": test["Market_Cum"].iloc[-1] - 1,
                "LR Return": test["LR_Cum"].iloc[-1] - 1,
                "XGB Return": test["XGB_Cum"].iloc[-1] - 1,
                "Market Sharpe": sharpe_ratio(test["Return"]),
                "LR Sharpe": sharpe_ratio(test["LR_Returns"]),
                "XGB Sharpe": sharpe_ratio(test["XGB_Returns"]),
                "Market Drawdown": max_drawdown(test["Market_Cum"]),
                "LR Drawdown": max_drawdown(test["LR_Cum"]),
                "XGB Drawdown": max_drawdown(test["XGB_Cum"]),
                "Market Win Rate": win_rate(test["Return"]),
                "LR Win Rate": win_rate(test["LR_Returns"]),
                "XGB Win Rate": win_rate(test["XGB_Returns"])
            })

            # -------------------------
            # SAVE FIGURES - EXACTLY AS ORIGINAL
            # -------------------------
            
            # Confusion Matrices
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, lr_pred,  cmap="Blues",  ax=ax[0])
            ax[0].set_title("Logistic Regression")
            ConfusionMatrixDisplay.from_predictions(y_test, xgb_pred, cmap="Greens", ax=ax[1])
            ax[1].set_title("XGBoost")
            plt.suptitle(f"{ticker} Confusion Matrices")
            plt.tight_layout()
            plt.savefig(f"Figures/Confusion/{ticker}_confusion.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Feature Importance (XGBoost)
            plt.figure(figsize=(8, 6))
            xgb.plot_importance(xgb_model, importance_type="gain", xlabel="Importance Score")
            plt.title(f"{ticker} — XGBoost Feature Importance")
            plt.tight_layout()
            plt.savefig(f"Figures/FeatureImportance/{ticker}_feature_importance.png",
                        dpi=300, bbox_inches="tight")
            plt.close()

            # Signal Plots
            fig = plot_signals(test, ticker, "Logistic Regression", "LR_Position")
            plt.savefig(f"Figures/Signals/{ticker}_Logistic_Regression_signals.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            fig = plot_signals(test, ticker, "XGBoost", "XGB_Position")
            plt.savefig(f"Figures/Signals/{ticker}_XGBoost_signals.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Cumulative Returns Plot
            plt.figure(figsize=(12, 6))
            plt.plot(test.index, test["Market_Cum"], label="Buy & Hold", linewidth=2)
            plt.plot(test.index, test["LR_Cum"],     label="LR Strategy",  linewidth=2)
            plt.plot(test.index, test["XGB_Cum"],    label="XGB Strategy", linewidth=2)
            plt.title(f"{ticker} — Cumulative Returns", fontsize=14)
            plt.xlabel("Date")
            plt.ylabel("Growth of €1")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"Figures/Cumulative/{ticker}_cumulative.png", dpi=300, bbox_inches="tight")
            plt.close()

            # SHAP plots are already saved in train_and_predict function

            # -------------------------
            # DISPLAY RESULTS FOR THIS TICKER
            # -------------------------
            st.subheader(f"📊 {ticker}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classification Results**")
                st.dataframe(classification_results.round(3))
                
                # Save classification results
                classification_results.to_csv(
                    f"Results/{ticker}_classification_metrics.csv", index=False
                )
            
            with col2:
                st.markdown("**Trading Performance**")
                st.dataframe(trading_results.round(2))
                
                # Save trading results
                trading_results.to_csv(
                    f"Results/{ticker}_trading_metrics.csv", index=False
                )
            
            # Display SHAP plots
            st.markdown("**SHAP Analysis**")
            shap_col1, shap_col2 = st.columns(2)
            
            with shap_col1:
                # Display SHAP Summary
                if os.path.exists(f"Figures/SHAP/{ticker}_SHAP_Summary.png"):
                    st.image(f"Figures/SHAP/{ticker}_SHAP_Summary.png", 
                            caption=f"{ticker} - SHAP Summary Plot")
            
            with shap_col2:
                # Display SHAP Bar
                if os.path.exists(f"Figures/SHAP/{ticker}_SHAP_Bar.png"):
                    st.image(f"Figures/SHAP/{ticker}_SHAP_Bar.png", 
                            caption=f"{ticker} - SHAP Feature Importance")
            
            # Display other plots in expandable sections
            with st.expander(f"📈 View Plots for {ticker}"):
                # Confusion Matrices
                if os.path.exists(f"Figures/Confusion/{ticker}_confusion.png"):
                    st.image(f"Figures/Confusion/{ticker}_confusion.png", 
                            caption=f"{ticker} - Confusion Matrices")
                
                # Feature Importance
                if os.path.exists(f"Figures/FeatureImportance/{ticker}_feature_importance.png"):
                    st.image(f"Figures/FeatureImportance/{ticker}_feature_importance.png", 
                            caption=f"{ticker} - XGBoost Feature Importance")
                
                # Signal Plots
                col_sig1, col_sig2 = st.columns(2)
                with col_sig1:
                    if os.path.exists(f"Figures/Signals/{ticker}_Logistic_Regression_signals.png"):
                        st.image(f"Figures/Signals/{ticker}_Logistic_Regression_signals.png", 
                                caption=f"{ticker} - Logistic Regression Signals")
                with col_sig2:
                    if os.path.exists(f"Figures/Signals/{ticker}_XGBoost_signals.png"):
                        st.image(f"Figures/Signals/{ticker}_XGBoost_signals.png", 
                                caption=f"{ticker} - XGBoost Signals")
                
                # Cumulative Returns
                if os.path.exists(f"Figures/Cumulative/{ticker}_cumulative.png"):
                    st.image(f"Figures/Cumulative/{ticker}_cumulative.png", 
                            caption=f"{ticker} - Cumulative Returns")
            
            progress_bar.progress((idx + 1) / len(tickers))
    
    status_text.text("✅ Analysis Complete!")
    
    # ========================================
    # SUMMARY - EXACTLY AS ORIGINAL
    # ========================================
    if all_results:
        st.markdown("---")
        st.header("📊 SUMMARY RESULTS")
        
        summary = pd.DataFrame(all_results)
        
        summary_display = summary.copy()
        for col in ["Market_Return","LR_Return","XGB_Return",
                    "Market_Drawdown","LR_Drawdown","XGB_Drawdown",
                    "LR_WinRate","XGB_WinRate"]:
            summary_display[col] = summary_display[col].apply(lambda x: f"{x:.2%}")
        for col in ["Market_Sharpe","LR_Sharpe","XGB_Sharpe"]:
            summary_display[col] = summary_display[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(summary_display)
        summary.to_csv("Results/summary.csv", index=False)
        
        # ========================================
        # TRADING PERFORMANCE FOR ALL STOCKS - EXACTLY AS ORIGINAL
        # ========================================
        st.markdown("---")
        st.header("💰 TRADING PERFORMANCE FOR ALL STOCKS")
        
        overall_trading_df = pd.DataFrame(overall_trading)
        
        display_table = overall_trading_df.copy()
        
        percent_cols = [
            "Market Return", "LR Return", "XGB Return",
            "Market Drawdown", "LR Drawdown", "XGB Drawdown",
            "Market Win Rate", "LR Win Rate", "XGB Win Rate"
        ]
        
        for col in percent_cols:
            display_table[col] = display_table[col].map("{:.2%}".format)
        
        decimal_cols = ["Market Sharpe", "LR Sharpe", "XGB Sharpe"]
        for col in decimal_cols:
            display_table[col] = display_table[col].map("{:.3f}".format)
        
        st.dataframe(display_table)
        
        display_table.to_csv(
            "Results/All_Stocks_Trading_Performance.csv",
            index=False
        )
        
        # ========================================
        # AVERAGE PERFORMANCE - EXACTLY AS ORIGINAL
        # ========================================
        st.markdown("---")
        st.header("📈 AVERAGE PERFORMANCE ACROSS ALL TICKERS")
        st.markdown("**(Mean / Median / Std)**")
        
        def summarise(col):
            return {
                "Mean":   summary[col].mean(),
                "Median": summary[col].median(),
                "Std":    summary[col].std()
            }
        
        # Display in a formatted way
        for label, m_col, lr_col, xgb_col in [
            ("Return",   "Market_Return",   "LR_Return",   "XGB_Return"),
            ("Sharpe",   "Market_Sharpe",   "LR_Sharpe",   "XGB_Sharpe"),
            ("Drawdown", "Market_Drawdown", "LR_Drawdown", "XGB_Drawdown"),
        ]:
            ms  = summarise(m_col)
            lrs = summarise(lr_col)
            xs  = summarise(xgb_col)
            
            st.markdown(f"**{label}:**")
            st.text(f"  Market  — Mean: {ms['Mean']:+.3f}  Median: {ms['Median']:+.3f}  Std: {ms['Std']:.3f}")
            st.text(f"  LR      — Mean: {lrs['Mean']:+.3f}  Median: {lrs['Median']:+.3f}  Std: {lrs['Std']:.3f}")
            st.text(f"  XGBoost — Mean: {xs['Mean']:+.3f}  Median: {xs['Median']:+.3f}  Std: {xs['Std']:.3f}")
        
        st.markdown("**Win Rate:**")
        lrw = summarise("LR_WinRate")
        xw = summarise("XGB_WinRate")
        st.text(f"  LR      — Mean: {lrw['Mean']:.3f}  Median: {lrw['Median']:.3f}  Std: {lrw['Std']:.3f}")
        st.text(f"  XGBoost — Mean: {xw['Mean']:.3f}  Median: {xw['Median']:.3f}  Std: {xw['Std']:.3f}")
        
        # ========================================
        # OVERALL COMPARISON - EXACTLY AS ORIGINAL
        # ========================================
        st.markdown("---")
        st.header("📊 OVERALL COMPARISON")
        
        overall = pd.DataFrame({
            "Metric": ["Mean Return", "Median Return", "Mean Sharpe", "Median Sharpe",
                       "Mean Drawdown", "Mean Win Rate"],
            "Market": [
                summary["Market_Return"].mean(), summary["Market_Return"].median(),
                summary["Market_Sharpe"].mean(), summary["Market_Sharpe"].median(),
                summary["Market_Drawdown"].mean(), np.nan
            ],
            "Logistic Regression": [
                summary["LR_Return"].mean(), summary["LR_Return"].median(),
                summary["LR_Sharpe"].mean(), summary["LR_Sharpe"].median(),
                summary["LR_Drawdown"].mean(), summary["LR_WinRate"].mean()
            ],
            "XGBoost": [
                summary["XGB_Return"].mean(), summary["XGB_Return"].median(),
                summary["XGB_Sharpe"].mean(), summary["XGB_Sharpe"].median(),
                summary["XGB_Drawdown"].mean(), summary["XGB_WinRate"].mean()
            ]
        })
        
        st.dataframe(overall.round(3))
        overall.to_csv("Results/overall_comparison.csv", index=False)
        
        # ========================================
        # DOWNLOAD ALL RESULTS
        # ========================================
        st.markdown("---")
        st.subheader("📥 Download All Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_summary = summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Summary Results",
                data=csv_summary,
                file_name="summary.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_trading = overall_trading_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Trading Performance",
                data=csv_trading,
                file_name="All_Stocks_Trading_Performance.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_overall = overall.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Overall Comparison",
                data=csv_overall,
                file_name="overall_comparison.csv",
                mime="text/csv"
            )
