# Irish-Stock-Market-Prediction
Stock Market Prediction
# 📈 Irish Stock Market Prediction

A machine learning application for predicting weekly stock movements of Irish stocks using Logistic Regression and XGBoost, with SHAP analysis for model interpretability.

## 🎯 Project Overview

This application analyzes seven Irish stocks (IRES.IR, KRZ.IR, GL9.IR, UPR.IR, GRP.IR, KRX.IR, KMR.IR) and generates trading signals based on weekly price movements. The system uses technical indicators as features and compares the performance of Logistic Regression and XGBoost models.

### Key Features

- **Data Processing**: Downloads and processes 8 years of stock data from Yahoo Finance
- **Feature Engineering**: Creates technical indicators (SMA, EMA, RSI, MACD, volatility, lagged returns)
- **Machine Learning Models**: Logistic Regression and XGBoost with hyperparameter tuning
- **SHAP Analysis**: Model interpretability using SHAP values
- **Trading Strategy**: Simulates trading with transaction costs and generates buy/sell signals
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate, and classification metrics
- **Interactive Dashboard**: Built with Streamlit for easy exploration
