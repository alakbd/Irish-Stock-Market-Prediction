import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    download_stock_data, 
    process_stock_data, 
    train_models,
    calculate_metrics,
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
    default=tickers[:3]  # Default to first 3 stocks
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

# Advanced parameters (collapsible)
with st.sidebar.expander("Advanced Parameters"):
    period = st.selectbox("Data Period", ["8y", "5y", "3y", "1y"], index=0)
    threshold = st.slider("Weekly Return Threshold (%)", 0.0, 5.0, 1.0, 0.5) / 100
    train_ratio = st.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7, 0.05)

st.sidebar.info("""
💡 **Tip:** 
- Select fewer stocks for faster analysis
- Use caching to speed up repeated runs
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
    
    # Store all results
    all_results = []
    all_metrics = []
    
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
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, lr_pred, xgb_pred, lr_proba, xgb_proba)
            
            # Display results for this ticker
            st.subheader(f"📊 {ticker}")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Model Performance**")
                st.dataframe(metrics.round(3), use_container_width=True)
            
            with col2:
                # Show class distribution
                class_counts = y_test.value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(class_counts.values, labels=['No Trade', 'Trade'], 
                       autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
                ax.set_title(f"{ticker} - Test Set Class Distribution")
                st.pyplot(fig)
                plt.close(fig)
            
            # Store results
            all_results.append({
                "Ticker": ticker,
                "LR_Accuracy": metrics.loc[0, "Accuracy"],
                "XGB_Accuracy": metrics.loc[1, "Accuracy"],
                "LR_F1": metrics.loc[0, "F1 Score"],
                "XGB_F1": metrics.loc[1, "F1 Score"],
                "LR_Precision": metrics.loc[0, "Precision"],
                "XGB_Precision": metrics.loc[1, "Precision"],
                "LR_Recall": metrics.loc[0, "Recall"],
                "XGB_Recall": metrics.loc[1, "Recall"]
            })
            
            # Feature Importance (if selected)
            if show_feature_importance and xgb_model is not None:
                with st.expander(f"🔍 Feature Importance for {ticker}"):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    importance = xgb_model.get_score(importance_type='gain')
                    importance_df = pd.DataFrame({
                        'feature': list(importance.keys()),
                        'importance': list(importance.values())
                    }).sort_values('importance', ascending=True)
                    
                    ax.barh(importance_df['feature'], importance_df['importance'])
                    ax.set_xlabel('Importance Score')
                    ax.set_title(f'{ticker} - XGBoost Feature Importance')
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Signal plots (if selected)
            if show_signals:
                with st.expander(f"📈 Trading Signals for {ticker}"):
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # Plot 1: LR Signals
                    axes[0].plot(test.index, test["Close"], label="Price", linewidth=1.5, color="steelblue")
                    buy_idx = test.index[(test["LR_Position"] == 1) & (test["LR_Position"].shift(1) == 0)]
                    sell_idx = test.index[(test["LR_Position"] == 0) & (test["LR_Position"].shift(1) == 1)]
                    axes[0].scatter(buy_idx, test.loc[buy_idx, "Close"], marker="^", color="green", s=80, label="BUY")
                    axes[0].scatter(sell_idx, test.loc[sell_idx, "Close"], marker="v", color="red", s=80, label="SELL")
                    axes[0].set_title(f"{ticker} - Logistic Regression Signals")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot 2: XGB Signals
                    axes[1].plot(test.index, test["Close"], label="Price", linewidth=1.5, color="steelblue")
                    buy_idx = test.index[(test["XGB_Position"] == 1) & (test["XGB_Position"].shift(1) == 0)]
                    sell_idx = test.index[(test["XGB_Position"] == 0) & (test["XGB_Position"].shift(1) == 1)]
                    axes[1].scatter(buy_idx, test.loc[buy_idx, "Close"], marker="^", color="green", s=80, label="BUY")
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
    
    # =========================
    # SUMMARY SECTION
    # =========================
    if all_results:
        st.markdown("---")
        st.header("📊 Overall Summary")
        
        summary_df = pd.DataFrame(all_results)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "📊 Model Comparison", "📋 Raw Data"])
        
        with tab1:
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg LR Accuracy", f"{summary_df['LR_Accuracy'].mean():.2%}")
            with col2:
                st.metric("Avg XGB Accuracy", f"{summary_df['XGB_Accuracy'].mean():.2%}")
            with col3:
                st.metric("Avg LR F1", f"{summary_df['LR_F1'].mean():.3f}")
            with col4:
                st.metric("Avg XGB F1", f"{summary_df['XGB_F1'].mean():.3f}")
        
        with tab2:
            # Comparison bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(summary_df))
            width = 0.35
            
            ax.bar(x - width/2, summary_df["LR_F1"], width, label="Logistic Regression", 
                   color='#1E88E5', alpha=0.8)
            ax.bar(x + width/2, summary_df["XGB_F1"], width, label="XGBoost", 
                   color='#43A047', alpha=0.8)
            
            ax.set_xlabel("Stock Ticker")
            ax.set_ylabel("F1 Score")
            ax.set_title("Model Performance Comparison (F1 Score)")
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df["Ticker"], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Accuracy comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(summary_df))
            width = 0.35
            
            ax.bar(x - width/2, summary_df["LR_Accuracy"], width, label="Logistic Regression", 
                   color='#1E88E5', alpha=0.8)
            ax.bar(x + width/2, summary_df["XGB_Accuracy"], width, label="XGBoost", 
                   color='#43A047', alpha=0.8)
            
            ax.set_xlabel("Stock Ticker")
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Performance Comparison (Accuracy)")
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df["Ticker"], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig)
        
        with tab3:
            st.dataframe(summary_df.round(3), use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="stock_prediction_results.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Select stocks from the sidebar and click 'Run Analysis' to begin.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit, XGBoost, and yfinance</p>
    <p>Data is cached for performance. Results may take a moment to load.</p>
</div>
""", unsafe_allow_html=True)
