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
import numpy as np


class TechnicalIndicatorGenerator:
    """
    Class to generate various technical indicators and direction signals based on stock data.
    The class calculates the following technical indicators: MACD, ATR, Bollinger Bands, RSI, ADX, and Renko.
    Parameters
    ----------
    data : pandas.DataFrame
        Historical data of the stock obtained from the yfinance API.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the technical indicators as features for the given stock data.
    Notes
    -----
    This class is designed to perform feature engineering by generating multiple technical indicators
    from the historical stock data. These indicators can be used as input features for training an
    XGBoost classification model or any other machine learning model.

    Example usage:
    --------------
    >>> data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
    >>> indicator_generator = TechnicalIndicatorGenerator()
    >>> indicators = indicator_generator.generate_indicators(data)
    References:
    -----------
    [1] MACD:

    """

    def __init__(self, data: Type[pd.DataFrame]) -> None:
        self.data = data

    def moving_average_convergence_divergence(self, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence) Indicator

        The MACD indicator is a popular technical
        analysis tool used to identify potential buying and selling opportunities
        in financial markets. It is based on the convergence and divergence of two moving averages,
        typically the 12-day Exponential Moving Average (EMA) and the 26-day EMA.

        Parameters
        ----------
        data: Historical price data of the asset.
        fast: 12-day EMA
        slow: 26-day EMA

        Returns
        -------
        DataFrame containing the MACD data.
        """

        data = self.data.copy()
        data["ma_fast"] = data["Close"].ewm(span=fast, min_periods=fast).mean()
        data["ma_slow"] = data["Close"].ewm(span=slow, min_periods=slow).mean()
        data["macd"] = data["ma_fast"] - data["ma_slow"]
        data["signal"] = data["macd"].ewm(span=signal, min_periods=signal).mean()
        return data

    def average_true_range(self, num_days=14):
        "function to calculate True Range and Average True Range"

        data = self.moving_average_convergence_divergence()

        data["H-L"] = data["High"] - data["Low"]
        data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
        data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
        data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
        data["ATR"] = data["TR"].ewm(com=num_days, min_periods=num_days).mean()
        return data

    def bollinger_band(self, num_days=14):
        "function to calculate Bollinger Band"
        data = self.average_true_range()
        data["middle_band"] = data["Close"].rolling(num_days).mean()
        data["upper_band"] = data["middle_band"] + 2 * data["Close"].rolling(
            num_days
        ).std(ddof=0)
        data["lower_band"] = data["middle_band"] - 2 * data["Close"].rolling(
            num_days
        ).std(ddof=0)
        data["BB_Width"] = data["upper_band"] - data["lower_band"]
        return data.drop(["middle_band", "upper_band", "lower_band"], 1)

    def ADX(self, n=20):
        "function to calculate ADX"
        data = self.bollinger_band()
        data = data.copy()
        data["upmove"] = data["High"] - data["High"].shift(1)
        data["downmove"] = data["Low"].shift(1) - data["Low"]
        data["+dm"] = np.where(
            (data["upmove"] > data["downmove"]) & (data["upmove"] > 0),
            data["upmove"],
            0,
        )
        data["-dm"] = np.where(
            (data["downmove"] > data["upmove"]) & (data["downmove"] > 0),
            data["downmove"],
            0,
        )
        data["+di"] = (
            100 * (data["+dm"] / data["ATR"]).ewm(alpha=1 / n, min_periods=n).mean()
        )
        data["-di"] = (
            100 * (data["-dm"] / data["ATR"]).ewm(alpha=1 / n, min_periods=n).mean()
        )
        data["ADX"] = (
            100
            * abs((data["+di"] - data["-di"]) / (data["+di"] + data["-di"]))
            .ewm(alpha=1 / n, min_periods=n)
            .mean()
        )
        return data
