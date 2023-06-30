"""
Scipt used to generate technical indicators and signals 
for buying, selling the stock.

The codes for this section are referred from the following Udemy Course

```
Instructor: Mayank Rasu
Course Title: Algorithmic Trading and Quantitative Analysis using python
url: https://www.udemy.com/course/algorithmic-trading-quantitative-analysis-using-python/

"""
from typing import Type
import pandas as pd


class FeatureEngineering:
    """
    Class to generate several technical indicators and \\
    direction signal based on the stock data. \\
    The class calculates the following indicators: \\
    MACD, ATR, Bollinger Band, RSI, ADX, RENKO 
    
    Input parameters
    ----------------
    data: Historical data of the stocks

    Returns
    -------
    Technical indicators
    
    """

    def __init__(self, data: Type[pd.DataFrame]) -> None:
        self.data = data


def moving_average_convergence_divergence(data, fast=12, slow=26, signal=9):
    """function to calculate MACD
    typical values a(fast moving average) = 12;
                   b(slow moving average) =26;
                   c(signal line ma window) =9"""
    data["ma_fast"] = data["Adj Close"].ewm(span=fast, min_periods=fast).mean()
    data["ma_slow"] = data["Adj Close"].ewm(span=slow, min_periods=slow).mean()
    data["macd"] = data["ma_fast"] - data["ma_slow"]
    data["signal"] = data["macd"].ewm(span=signal, min_periods=signal).mean()
    return data.loc[:, ["macd", "signal"]]


def average_true_range(data, num_days=14):
    "function to calculate True Range and Average True Range"
    data["H-L"] = data["High"] - data["Low"]
    data["H-PC"] = abs(data["High"] - data["Adj Close"].shift(1))
    data["L-PC"] = abs(data["Low"] - data["Adj Close"].shift(1))
    data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
    data["ATR"] = data["TR"].ewm(com=num_days, min_periods=num_days).mean()
    return data["ATR"]


def bollinger_band(data, n=14):
    "function to calculate Bollinger Band"
    data["MB"] = data["Adj Close"].rolling(n).mean()
    data["UB"] = data["MB"] + 2 * data["Adj Close"].rolling(n).std(ddof=0)
    data["LB"] = data["MB"] - 2 * data["Adj Close"].rolling(n).std(ddof=0)
    data["BB_Width"] = data["UB"] - data["LB"]
    return data[["MB", "UB", "LB", "BB_Width"]]
