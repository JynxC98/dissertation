"""
Script used to generate technical indicators and signals for buying, selling the stock.

The codes for this section are referred from the following Udemy Course:

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
    """

    def __init__(self, data: Type[pd.DataFrame]) -> None:
        self.data = data

    def calculate_moving_average(self, period=14) -> Type[pd.DataFrame]:
        """
        Calculates the moving average of the stock prices.

        Parameters
        ----------
        period: int, optional
            Default value is 14 days

        Returns
        -------
        pandas.DataFrame
            DataFrame with moving average data added
        """
        data = self.data.copy()
        data["Moving Average"] = data["Close"].rolling(window=period).mean()
        return data

    def moving_average_convergence_divergence(self, fast=12, slow=26, signal=9):
        """
        Calculate the MACD (Moving Average Convergence Divergence) Indicator.

        The MACD indicator is a popular technical analysis tool used to identify potential buying and selling
        opportunities in financial markets. It is based on the convergence and divergence of two moving averages,
        typically the 12-day Exponential Moving Average (EMA) and the 26-day EMA.

        Parameters
        ----------
        fast : int, optional
            Period for the faster EMA, default is 12.
        slow : int, optional
            Period for the slower EMA, default is 26.
        signal : int, optional
            Period for the signal line, default is 9.

        Returns
        -------
        pandas.DataFrame
            DataFrame with market_type related data added.
            market_type: 1 if macd > signal (Indicating bullish phase)
            market_type: 0 if macd < signal (Bearish Phase)

        """
        data = self.calculate_moving_average()
        data["ma_fast"] = data["Close"].ewm(span=fast, min_periods=fast).mean()
        data["ma_slow"] = data["Close"].ewm(span=slow, min_periods=slow).mean()
        data["macd"] = data["ma_fast"] - data["ma_slow"]
        data["signal"] = data["macd"].ewm(span=signal, min_periods=signal).mean()
        data["Market Type"] = np.where(data["macd"] > data["signal"], 1, 0)
        return data.drop(["ma_fast", "ma_slow", "signal", "macd"], axis=1)

    def average_true_range(self, num_days=14):
        """
        Calculate True Range and Average True Range.

        Returns
        -------
        pandas.DataFrame
            DataFrame with ATR data added.
        """
        data = self.moving_average_convergence_divergence()

        data["H-L"] = data["High"] - data["Low"]
        data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
        data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
        data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
        data["ATR"] = data["TR"].ewm(com=num_days, min_periods=num_days).mean()
        return data.drop(["H-L", "H-PC", "L-PC", "TR"], 1)

    def relative_strength_index(self, n=14):
        """
        Calculates RSI

        Returns
        -------
        pandas.DataFrame
            Dataframe with RSI added

        """
        data = self.average_true_range()
        data["change"] = data["Close"] - data["Close"].shift(1)
        data["gain"] = np.where(data["change"] >= 0, data["change"], 0)
        data["loss"] = np.where(data["change"] < 0, -1 * data["change"], 0)
        data["avgGain"] = data["gain"].ewm(alpha=1 / n, min_periods=n).mean()
        data["avgLoss"] = data["loss"].ewm(alpha=1 / n, min_periods=n).mean()
        data["rs"] = data["avgGain"] / data["avgLoss"]
        data["RSI"] = 100 - (100 / (1 + data["rs"]))
        return data.drop(["change", "gain", "loss", "avgGain", "avgLoss", "rs"], 1)

    def average_directional_index(self, n=14):
        """
        Calculate ADX.

        Returns
        -------
        pandas.DataFrame
            DataFrame with ADX data added.
        """
        data = self.relative_strength_index()
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

        return data.drop(["upmove", "downmove", "+dm", "-dm", "+di", "-di"], 1)

    def generate_direction(self):
        """
        Returns
        -------
        pandas.DataFrame
            DataFrame with direction `1` if there is an increase in the stock price with
            respect to the previous price, else `0`.
        """
        data = self.average_directional_index()
        data["Direction"] = np.where(data["Close"].diff() > 0, 1, 0)
        return data

    def return_final_data(self):
        """
        Returns
        -------
        pandas.DataFrame
            The final cleaned tabular data.
        """
        data = self.generate_direction()
        return data.dropna()
