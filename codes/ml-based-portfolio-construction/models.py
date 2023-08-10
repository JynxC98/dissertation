"""
Script to feed the data into the machine learning model.
"""
import warnings
from typing import Type
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from scipy.signal import argrelextrema

from xgboost import XGBClassifier


from indicators import TechnicalIndicatorGenerator

warnings.filterwarnings("ignore")


class GeneratePredictions:
    """ """


def calculate_vertical_levels(data, threshold=3):
    """
    Calculates the vertical resistance and support levels based on price data.

    Parameters:
        data (DataFrame): Price data containing 'High', 'Low', and 'Close' columns.
        threshold (int): Number of times the price should touch a level (default: 3).

    Returns:
        tuple: Last resistance and support levels.
    """
    resistance_levels = []
    support_levels = []

    # Calculate local maximas and minimas
    maximas = argrelextrema(data["High"].values, np.greater_equal, order=threshold)[0]
    minimas = argrelextrema(data["Low"].values, np.less_equal, order=threshold)[0]

    # if maximas or minimas found, add them as resistance and support levels
    if maximas.any():
        resistance_levels.extend(data["High"].iloc[maximas].tolist())

    if minimas.any():
        support_levels.extend(data["Low"].iloc[minimas].tolist())

    # Plotting the last resistance and support levels with closing price
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Closing Price")

    if resistance_levels:
        plt.hlines(
            resistance_levels[-1],
            xmin=data.index[0],
            xmax=data.index[-1],
            colors="r",
            label="Resistance",
        )
    if support_levels:
        plt.hlines(
            support_levels[-1],
            xmin=data.index[0],
            xmax=data.index[-1],
            colors="g",
            label="Support",
        )

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Vertical Resistance and Support Levels")
    plt.legend()
    plt.show()

    return (
        resistance_levels[-1] if resistance_levels else None,
        support_levels[-1] if support_levels else None,
    )


params = {
    "colsample_bytree": 0.760869675435,
    "gamma": 0.029041340790919024,
    "learning_rate": 0.0888082370556709,
    "max_depth": 9,
    "n_estimators": 422,
    "subsample": 0.9505754827278782,
}


{
    "subsample": 1.0,
    "n_estimators": 200,
    "min_child_weight": 1,
    "max_depth": 5,
    "learning_rate": 0.01,
    "gamma": 0.1,
    "colsample_bytree": 0.8,
}
