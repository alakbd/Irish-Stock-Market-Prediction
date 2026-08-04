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
