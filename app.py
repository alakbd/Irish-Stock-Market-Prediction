import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    download_stock_data, 
    process_stock_data, 
    train_models,
    calculate_metrics,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    get_ticker_list,
    FEATURES_LIST
)
import warnings
warnings.filterwarnings("ignore")

# Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Irish Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .summary-header {
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Irish Stock Market Prediction</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome to the Irish Stock Market Prediction Application. 
This app uses machine learning to predict weekly stock movements for Irish stocks.
""")

# =========================
# SIDEBAR - Parameters
# =========================
st.sidebar.header("⚙️ Parameters")

# Stock selection
tickers = get_ticker_list()
selected_tickers = st.sidebar.multiselect(
    "Select Stocks to Analyze",
    options=tickers,
    default=tickers  # Select all by default
)

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["Both", "Logistic Regression", "XGBoost"],
    index=0
)

# Additional parameters
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
show_signals = st.sidebar.checkbox("Show Trading Signals", value=True)
show_confusion_matrix = st.sidebar.checkbox("Show Confusion Matrices", value=True)

# Advanced parameters (collapsible)
with st.sidebar.expander("Advanced Parameters"):
    period = st.selectbox("Data Period", ["8y", "5y", "3y", "1y"], index=0)
    threshold = st.slider("Weekly Return Threshold (%)", 0.0, 5.0, 1.0, 0.5) / 100
    train_ratio = st.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7, 0.05)
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100

st.sidebar.info("""
💡 **Tip:** 
- Select fewer stocks for faster analysis
- Data is cached for 1 hour for better performance
""")

# =========================
# MAIN CONTENT
# =========================

if not selected_tickers:
    st.warning("⚠️ Please select at least one stock from the sidebar.")
    st.stop()

# Run analysis button
if st.button("🚀 Run Analysis", type="primary"):
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store all results for summary
    all_classification_results = []
    all_trading_results = []
    overall_trading = []
    
    # Process each selected ticker
    for idx, ticker in enumerate(selected_tickers):
        status_text.text(f"📊 Processing {ticker}... ({idx+1}/{len(selected_tickers)})")
        
        try:
            # Download and process data
            with st.spinner(f"Downloading data for {ticker}..."):
                weekly_df = process_stock_data(ticker, period=period, threshold=threshold)
            
            if weekly_df is None or len(weekly_df) < 30:
                st.warning(f"⚠️ Not enough data for {ticker}. Skipping...")
                progress_bar.progress((idx + 1) / len(selected_tickers))
                continue
            
            # Train models
            with st.spinner(f"Training models for {ticker}..."):
                X_te_s, y_test, lr_pred, lr_proba, xgb_pred, xgb_proba, xgb_model, scaler, split = train_models(
                    weekly_df, FEATURES_LIST, train_ratio=train_ratio
                )
            
            # Get test data
            test = weekly_df.iloc[split:].copy()
            test["LR_Position"] = lr_pred
            test["XGB_Position"] = xgb_pred
            test["LR_Proba"] = lr_proba
            test["XGB_Proba"] = xgb_proba
            
            # Calculate trading returns with transaction costs
            test["LR_Trade"] = test["LR_Position"].diff().abs().fillna(0)
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
            test["LR_Cum"] = (1 + test["LR_Returns"]).cumprod()
            test["XGB_Cum"] = (1 + test["XGB_Returns"]).cumprod()
            
            # =========================
            # CLASSIFICATION METRICS
            # =========================
            metrics = calculate_metrics(y_test, lr_pred, xgb_pred, lr_proba, xgb_proba)
            
            # =========================
            # TRADING PERFORMANCE
            # =========================
            trading_results = pd.DataFrame({
                "Strategy": ["Buy & Hold (Benchmark)", "Logistic Regression", "XGBoost"],
                "Total Return (%)": [
                    (test["Market_Cum"].iloc[-1] - 1) * 100,
                    (test["LR_Cum"].iloc[-1] - 1) * 100,
                    (test["XGB_Cum"].iloc[-1] - 1) * 100
                ],
                "Sharpe Ratio": [
                    calculate_sharpe_ratio(test["Return"]),
                    calculate_sharpe_ratio(test["LR_Returns"]),
                    calculate_sharpe_ratio(test["XGB_Returns"])
                ],
                "Maximum Drawdown (%)": [
                    calculate_max_drawdown(test["Market_Cum"]) * 100,
                    calculate_max_drawdown(test["LR_Cum"]) * 100,
                    calculate_max_drawdown(test["XGB_Cum"]) * 100
                ],
                "Win Rate (%)": [
                    calculate_win_rate(test["Return"]) * 100,
                    calculate_win_rate(test["LR_Returns"]) * 100,
                    calculate_win_rate(test["XGB_Returns"]) * 100
                ]
            })
            
            # Store for overall summary
            overall_trading.append({
                "Ticker": ticker,
                "Market Return": test["Market_Cum"].iloc[-1] - 1,
                "LR Return": test["LR_Cum"].iloc[-1] - 1,
                "XGB Return": test["XGB_Cum"].iloc[-1] - 1,
                "Market Sharpe": calculate_sharpe_ratio(test["Return"]),
                "LR Sharpe": calculate_sharpe_ratio(test["LR_Returns"]),
                "XGB Sharpe": calculate_sharpe_ratio(test["XGB_Returns"]),
                "Market Drawdown": calculate_max_drawdown(test["Market_Cum"]),
                "LR Drawdown": calculate_max_drawdown(test["LR_Cum"]),
                "XGB Drawdown": calculate_max_drawdown(test["XGB_Cum"]),
                "Market Win Rate": calculate_win_rate(test["Return"]),
                "LR Win Rate": calculate_win_rate(test["LR_Returns"]),
                "XGB Win Rate": calculate_win_rate(test["XGB_Returns"])
            })
            
            # Store classification results
            all_classification_results.append({
                "Ticker": ticker,
                "LR_Accuracy": metrics.loc[0, "Accuracy"],
                "XGB_Accuracy": metrics.loc[1, "Accuracy"],
                "LR_F1": metrics.loc[0, "F1 Score"],
                "XGB_F1": metrics.loc[1, "F1 Score"],
                "LR_Precision": metrics.loc[0, "Precision"],
                "XGB_Precision": metrics.loc[1, "Precision"],
                "LR_Recall": metrics.loc[0, "Recall"],
                "XGB_Recall": metrics.loc[1, "Recall"],
                "LR_ROC_AUC": metrics.loc[0, "ROC-AUC"] if "ROC-AUC" in metrics.columns else np.nan,
                "XGB_ROC_AUC": metrics.loc[1, "ROC-AUC"] if "ROC-AUC" in metrics.columns else np.nan
            })
            
            # =========================
            # DISPLAY RESULTS FOR THIS TICKER
            # =========================
            st.markdown("---")
            st.subheader(f"📊 {ticker}")
            
            # Tabs for this ticker
            ticker_tabs = st.tabs(["📈 Classification Metrics", "💰 Trading Performance", "📉 Visualizations"])
            
            with ticker_tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📈 Model Performance**")
                    st.dataframe(metrics.round(3), use_container_width=True)
                with col2:
                    # Class distribution
                    class_counts = y_test.value_counts()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(class_counts.values, labels=['No Trade', 'Trade'], 
                           autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
                    ax.set_title(f"{ticker} - Test Set Class Distribution")
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Confusion Matrices
                if show_confusion_matrix:
                    from sklearn.metrics import ConfusionMatrixDisplay
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ConfusionMatrixDisplay.from_predictions(y_test, lr_pred, cmap="Blues", ax=ax)
                        ax.set_title("Logistic Regression")
                        st.pyplot(fig)
                        plt.close(fig)
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ConfusionMatrixDisplay.from_predictions(y_test, xgb_pred, cmap="Greens", ax=ax)
                        ax.set_title("XGBoost")
                        st.pyplot(fig)
                        plt.close(fig)
            
            with ticker_tabs[1]:
                st.dataframe(trading_results.round(2), use_container_width=True)
                
                # Cumulative returns plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(test.index, test["Market_Cum"], label="Buy & Hold", linewidth=2)
                ax.plot(test.index, test["LR_Cum"], label="LR Strategy", linewidth=2)
                ax.plot(test.index, test["XGB_Cum"], label="XGB Strategy", linewidth=2)
                ax.set_title(f"{ticker} — Cumulative Returns", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("Growth of €1")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            
            with ticker_tabs[2]:
                # Feature Importance
                if show_feature_importance and xgb_model is not None:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    importance = xgb_model.get_score(importance_type='gain')
                    if importance:
                        importance_df = pd.DataFrame({
                            'feature': list(importance.keys()),
                            'importance': list(importance.values())
                        }).sort_values('importance', ascending=True)
                        ax.barh(importance_df['feature'], importance_df['importance'])
                        ax.set_xlabel('Importance Score')
                        ax.set_title(f'{ticker} - XGBoost Feature Importance')
                        st.pyplot(fig)
                    plt.close(fig)
                
                # Signal plots
                if show_signals:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # LR Signals
                    axes[0].plot(test.index, test["Close"], label="Price", linewidth=1.5, color="steelblue")
                    buy_idx = test.index[(test["LR_Position"] == 1) & (test["LR_Position"].shift(1) == 0)]
                    sell_idx = test.index[(test["LR_Position"] == 0) & (test["LR_Position"].shift(1) == 1)]
                    if len(buy_idx) > 0:
                        axes[0].scatter(buy_idx, test.loc[buy_idx, "Close"], marker="^", color="green", s=80, label="BUY")
                    if len(sell_idx) > 0:
                        axes[0].scatter(sell_idx, test.loc[sell_idx, "Close"], marker="v", color="red", s=80, label="SELL")
                    axes[0].set_title(f"{ticker} - Logistic Regression Signals")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # XGB Signals
                    axes[1].plot(test.index, test["Close"], label="Price", linewidth=1.5, color="steelblue")
                    buy_idx = test.index[(test["XGB_Position"] == 1) & (test["XGB_Position"].shift(1) == 0)]
                    sell_idx = test.index[(test["XGB_Position"] == 0) & (test["XGB_Position"].shift(1) == 1)]
                    if len(buy_idx) > 0:
                        axes[1].scatter(buy_idx, test.loc[buy_idx, "Close"], marker="^", color="green", s=80, label="BUY")
                    if len(sell_idx) > 0:
                        axes[1].scatter(sell_idx, test.loc[sell_idx, "Close"], marker="v", color="red", s=80, label="SELL")
                    axes[1].set_title(f"{ticker} - XGBoost Signals")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
        except Exception as e:
            st.error(f"❌ Error processing {ticker}: {str(e)}")
            st.exception(e)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(selected_tickers))
    
    status_text.text("✅ Analysis Complete!")
    
    # ========================================
    # SUMMARY RESULTS SECTION
    # ========================================
    if all_classification_results and overall_trading:
        st.markdown("---")
        st.header("📊 SUMMARY RESULTS")
        st.markdown('<div class="summary-header">📈 Overall Performance Summary</div>', unsafe_allow_html=True)
        
        # Create summary dataframe
        summary_df = pd.DataFrame(all_classification_results)
        
        # Format for display
        summary_display = summary_df.copy()
        for col in summary_display.select_dtypes(include=[np.number]).columns:
            if 'Accuracy' in col or 'Recall' in col or 'Precision' in col:
                summary_display[col] = summary_display[col].apply(lambda x: f"{x:.2%}")
            else:
                summary_display[col] = summary_display[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(summary_display, use_container_width=True)
        
        # ========================================
        # TRADING PERFORMANCE FOR ALL STOCKS
        # ========================================
        st.markdown("---")
        st.header("💰 TRADING PERFORMANCE FOR ALL STOCKS")
        st.markdown('<div class="summary-header">📊 Trading Performance Metrics</div>', unsafe_allow_html=True)
        
        overall_trading_df = pd.DataFrame(overall_trading)
        
        # Format for display
        display_table = overall_trading_df.copy()
        percent_cols = [
            "Market Return", "LR Return", "XGB Return",
            "Market Drawdown", "LR Drawdown", "XGB Drawdown",
            "Market Win Rate", "LR Win Rate", "XGB Win Rate"
        ]
        for col in percent_cols:
            if col in display_table.columns:
                display_table[col] = display_table[col].map("{:.2%}".format)
        
        decimal_cols = ["Market Sharpe", "LR Sharpe", "XGB Sharpe"]
        for col in decimal_cols:
            if col in display_table.columns:
                display_table[col] = display_table[col].map("{:.3f}".format)
        
        st.dataframe(display_table, use_container_width=True)
        
        # Download button for trading performance
        csv_trading = overall_trading_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Trading Performance as CSV",
            data=csv_trading,
            file_name="trading_performance_all_stocks.csv",
            mime="text/csv"
        )
        
        # ========================================
        # AVERAGE PERFORMANCE ACROSS ALL TICKERS
        # ========================================
        st.markdown("---")
        st.header("📈 AVERAGE PERFORMANCE ACROSS ALL TICKERS")
        st.markdown('<div class="summary-header">📊 Mean / Median / Std</div>', unsafe_allow_html=True)
        
        # Calculate summary statistics
        def summarise(df, col):
            if col not in df.columns:
                return {"Mean": np.nan, "Median": np.nan, "Std": np.nan}
            return {
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Std": df[col].std()
            }
        
        # Create summary statistics table
        summary_stats = []
        
        for label, col_name in [
            ("Return", "LR Return"),
            ("Sharpe", "LR Sharpe"),
            ("Drawdown", "LR Drawdown"),
        ]:
            # For Logistic Regression
            lr_stats = summarise(overall_trading_df, f"LR {label}")
            xgb_stats = summarise(overall_trading_df, f"XGB {label}")
            market_stats = summarise(overall_trading_df, f"Market {label}")
            
            summary_stats.append({
                "Metric": f"{label} (Market)",
                "Mean": market_stats["Mean"],
                "Median": market_stats["Median"],
                "Std": market_stats["Std"]
            })
            summary_stats.append({
                "Metric": f"{label} (LR)",
                "Mean": lr_stats["Mean"],
                "Median": lr_stats["Median"],
                "Std": lr_stats["Std"]
            })
            summary_stats.append({
                "Metric": f"{label} (XGB)",
                "Mean": xgb_stats["Mean"],
                "Median": xgb_stats["Median"],
                "Std": xgb_stats["Std"]
            })
        
        # Add Win Rate separately
        lr_win = summarise(overall_trading_df, "LR Win Rate")
        xgb_win = summarise(overall_trading_df, "XGB Win Rate")
        market_win = summarise(overall_trading_df, "Market Win Rate")
        
        summary_stats.append({
            "Metric": "Win Rate (Market)",
            "Mean": market_win["Mean"],
            "Median": market_win["Median"],
            "Std": market_win["Std"]
        })
        summary_stats.append({
            "Metric": "Win Rate (LR)",
            "Mean": lr_win["Mean"],
            "Median": lr_win["Median"],
            "Std": lr_win["Std"]
        })
        summary_stats.append({
            "Metric": "Win Rate (XGB)",
            "Mean": xgb_win["Mean"],
            "Median": xgb_win["Median"],
            "Std": xgb_win["Std"]
        })
        
        stats_df = pd.DataFrame(summary_stats)
        
        # Format for display
        stats_display = stats_df.copy()
        for col in ["Mean", "Median", "Std"]:
            if col in stats_display.columns:
                stats_display[col] = stats_display[col].apply(
                    lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
                )
        
        st.dataframe(stats_display, use_container_width=True)
        
        # ========================================
        # OVERALL COMPARISON TABLE
        # ========================================
        st.markdown("---")
        st.header("📊 OVERALL COMPARISON")
        
        overall_comparison = pd.DataFrame({
            "Metric": ["Mean Return", "Median Return", "Mean Sharpe", "Median Sharpe",
                       "Mean Drawdown", "Mean Win Rate"],
            "Market": [
                overall_trading_df["Market Return"].mean() if "Market Return" in overall_trading_df else np.nan,
                overall_trading_df["Market Return"].median() if "Market Return" in overall_trading_df else np.nan,
                overall_trading_df["Market Sharpe"].mean() if "Market Sharpe" in overall_trading_df else np.nan,
                overall_trading_df["Market Sharpe"].median() if "Market Sharpe" in overall_trading_df else np.nan,
                overall_trading_df["Market Drawdown"].mean() if "Market Drawdown" in overall_trading_df else np.nan,
                overall_trading_df["Market Win Rate"].mean() if "Market Win Rate" in overall_trading_df else np.nan
            ],
            "Logistic Regression": [
                overall_trading_df["LR Return"].mean() if "LR Return" in overall_trading_df else np.nan,
                overall_trading_df["LR Return"].median() if "LR Return" in overall_trading_df else np.nan,
                overall_trading_df["LR Sharpe"].mean() if "LR Sharpe" in overall_trading_df else np.nan,
                overall_trading_df["LR Sharpe"].median() if "LR Sharpe" in overall_trading_df else np.nan,
                overall_trading_df["LR Drawdown"].mean() if "LR Drawdown" in overall_trading_df else np.nan,
                overall_trading_df["LR Win Rate"].mean() if "LR Win Rate" in overall_trading_df else np.nan
            ],
            "XGBoost": [
                overall_trading_df["XGB Return"].mean() if "XGB Return" in overall_trading_df else np.nan,
                overall_trading_df["XGB Return"].median() if "XGB Return" in overall_trading_df else np.nan,
                overall_trading_df["XGB Sharpe"].mean() if "XGB Sharpe" in overall_trading_df else np.nan,
                overall_trading_df["XGB Sharpe"].median() if "XGB Sharpe" in overall_trading_df else np.nan,
                overall_trading_df["XGB Drawdown"].mean() if "XGB Drawdown" in overall_trading_df else np.nan,
                overall_trading_df["XGB Win Rate"].mean() if "XGB Win Rate" in overall_trading_df else np.nan
            ]
        })
        
        # Format for display
        comparison_display = overall_comparison.copy()
        for col in ["Market", "Logistic Regression", "XGBoost"]:
            comparison_display[col] = comparison_display[col].apply(
                lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
            )
        
        st.dataframe(comparison_display, use_container_width=True)
        
        # Visual comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Return comparison
        ax = axes[0, 0]
        metrics = ['Mean Return', 'Median Return']
        x = np.arange(len(metrics))
        width = 0.25
        ax.bar(x - width, overall_comparison['Market'][:2], width, label='Market', color='gray')
        ax.bar(x, overall_comparison['Logistic Regression'][:2], width, label='LR', color='#1E88E5')
        ax.bar(x + width, overall_comparison['XGBoost'][:2], width, label='XGB', color='#43A047')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Return')
        ax.set_title('Return Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sharpe comparison
        ax = axes[0, 1]
        metrics = ['Mean Sharpe', 'Median Sharpe']
        x = np.arange(len(metrics))
        ax.bar(x - width, overall_comparison['Market'][2:4], width, label='Market', color='gray')
        ax.bar(x, overall_comparison['Logistic Regression'][2:4], width, label='LR', color='#1E88E5')
        ax.bar(x + width, overall_comparison['XGBoost'][2:4], width, label='XGB', color='#43A047')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Drawdown comparison
        ax = axes[1, 0]
        metrics = ['Mean Drawdown']
        x = np.arange(len(metrics))
        ax.bar(x - width, overall_comparison['Market'][4:5], width, label='Market', color='gray')
        ax.bar(x, overall_comparison['Logistic Regression'][4:5], width, label='LR', color='#1E88E5')
        ax.bar(x + width, overall_comparison['XGBoost'][4:5], width, label='XGB', color='#43A047')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Win Rate comparison
        ax = axes[1, 1]
        metrics = ['Mean Win Rate']
        x = np.arange(len(metrics))
        ax.bar(x - width, overall_comparison['Market'][5:6], width, label='Market', color='gray')
        ax.bar(x, overall_comparison['Logistic Regression'][5:6], width, label='LR', color='#1E88E5')
        ax.bar(x + width, overall_comparison['XGBoost'][5:6], width, label='XGB', color='#43A047')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Download all results
        st.markdown("---")
        st.subheader("📥 Download All Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_classification = pd.DataFrame(all_classification_results).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Classification Results",
                data=csv_classification,
                file_name="classification_results.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_trading = overall_trading_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Trading Performance",
                data=csv_trading,
                file_name="trading_performance.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_summary = overall_comparison.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Summary Statistics",
                data=csv_summary,
                file_name="summary_statistics.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Select stocks from the sidebar and click 'Run Analysis' to begin.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with using Streamlit, XGBoost, and yfinance</p>
    <p>Data is cached for performance. Results may take a moment to load.</p>
</div>
""", unsafe_allow_html=True)
