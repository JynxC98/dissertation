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
from sklearn.preprocessing import MinMaxScaler


class TechnicalIndicatorGenerator:
    """
    Class to generate various technical indicators and direction signals based on stock data.
    The class calculates the following technical indicators: MA, MACD, ATR,  RSI, Stochastic Oscillator,
    RSI, Williams%R

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

    def exponential_smoothing(self, alpha=0.2) -> Type[pd.DataFrame]:
        """
        Applies exponential smoothing to the input data.

        Parameters:
            data (Series or DataFrame): Time series data to be smoothed.
            alpha (float): Smoothing factor (default: 0.5).
                           Values closer to 1 give more weight to recent observations.

        Returns:
            Series or DataFrame: Smoothed time series data.
        """
        smoothed_data = self.data.copy()
        smoothed_data["Smoothened Close"] = (
            smoothed_data["Close"].ewm(alpha=alpha).mean()
        )
        smoothed_data["Price Change"] = smoothed_data["Smoothened Close"].pct_change()

        return smoothed_data

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
        data = self.exponential_smoothing()
        data["Moving Average"] = data["Smoothened Close"].rolling(window=period).mean()
        data["Moving Average Change"] = data["Moving Average"].pct_change()
        return data

    def calculate_trend(self) -> Type[pd.DataFrame]:
        """
        Rules for setting trend
        1. The closing value must lead (lag) its 25 day moving average
        2. The 25 day moving average must lead (lag) 65 day moving
        average.
        3. The 25 day moving average must have been rising (falling)
        for at least 5 days.
        4. The 65 day moving average must have been rising (falling)
        for at least 1 day.

        Uptrend is denoted by 1
        Downtrend is denoted by -1
        No trend is denoted by 0

        Returns
        -------
        pandas.DataFrame
            DataFrame with trend data added
        """
        # Calculate moving averages
        data = self.calculate_moving_average()
        data["MA25"] = data["Smoothened Close"].rolling(window=25).mean()
        data["MA65"] = data["Smoothened Close"].rolling(window=65).mean()

        # Calculate the change in moving averages
        data["MA25_Diff"] = data["MA25"].diff(periods=5)
        data["MA65_Diff"] = data["MA65"].diff(periods=1)

        # Initialize the 'Trend' column to 'No Trend'
        data["Trend"] = 0

        # Identify uptrends according to the given conditions
        data.loc[
            (data["Smoothened Close"] > data["MA25"])
            & (data["MA25"] > data["MA65"])
            & (data["MA25_Diff"] > 0)
            & (data["MA65_Diff"] > 0),
            "Trend",
        ] = 1

        # Identify downtrends according to the given conditions
        data.loc[
            (data["Smoothened Close"] < data["MA25"])
            & (data["MA25"] < data["MA65"])
            & (data["MA25_Diff"] < 0)
            & (data["MA65_Diff"] < 0),
            "Trend",
        ] = -1

        data["Trend"] = data["Trend"].astype(int)
        data["Trend"] = data["Trend"].shift(-1)

        return data.drop(["MA25", "MA65", "MA25_Diff", "MA65_Diff"], axis=1)

    def moving_average_convergence_divergence(
        self, fast=12, slow=26, signal=9
    ) -> Type[pd.DataFrame]:
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
        data = self.calculate_trend()
        data["ma_fast"] = (
            data["Smoothened Close"].ewm(span=fast, min_periods=fast).mean()
        )
        data["ma_slow"] = (
            data["Smoothened Close"].ewm(span=slow, min_periods=slow).mean()
        )
        data["macd"] = data["ma_fast"] - data["ma_slow"]
        data["signal"] = data["macd"].ewm(span=signal, min_periods=signal).mean()
        # data["Market Type"] = np.where(data["macd"] > data["signal"], 1, 0)
        return data.drop(["ma_fast", "ma_slow", "signal"], axis=1)

    def average_true_range(self, num_days=14) -> Type[pd.DataFrame]:
        """
        Calculate True Range and Average True Range.

        Returns
        -------
        pandas.DataFrame
            DataFrame with ATR data added.
        """
        data = self.moving_average_convergence_divergence()

        data["H-L"] = data["High"] - data["Low"]
        data["H-PC"] = abs(data["High"] - data["Smoothened Close"].shift(1))
        data["L-PC"] = abs(data["Low"] - data["Smoothened Close"].shift(1))
        data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
        data["ATR"] = data["TR"].ewm(com=num_days, min_periods=num_days).mean()
        return data.drop(["H-L", "H-PC", "L-PC", "TR"], 1)

    def relative_strength_index(self, n=14) -> Type[pd.DataFrame]:
        """
        Calculates RSI

        Returns
        -------
        pandas.DataFrame
            Dataframe with RSI added

        """
        data = self.average_true_range()
        data["change"] = data["Smoothened Close"] - data["Smoothened Close"].shift(1)
        data["gain"] = np.where(data["change"] >= 0, data["change"], 0)
        data["loss"] = np.where(data["change"] < 0, -1 * data["change"], 0)
        data["avgGain"] = data["gain"].ewm(alpha=1 / n, min_periods=n).mean()
        data["avgLoss"] = data["loss"].ewm(alpha=1 / n, min_periods=n).mean()
        data["rs"] = data["avgGain"] / data["avgLoss"]
        data["RSI"] = 100 - (100 / (1 + data["rs"]))
        return data.drop(["change", "gain", "loss", "avgGain", "avgLoss", "rs"], 1)

    def calculate_stochastic_oscillator(self, window=14) -> Type[pd.DataFrame]:
        """
        Returns
        -------
        pandas.DataFrame
            DataFrame with %K and %D added
        """
        # Calculate the lowest and highest prices in the rolling window
        data = self.relative_strength_index()
        data["Lowest Low"] = data["Low"].rolling(window).min()
        data["Highest High"] = data["High"].rolling(window).max()

        # Calculate the %K and %D values
        data["%K"] = (
            (data["Smoothened Close"] - data["Lowest Low"])
            / (data["Highest High"] - data["Lowest Low"])
            * 100
        )
        data["%D"] = (
            data["%K"].rolling(3).mean()
        )  # Using a 3-period moving average for %D

        return data.drop(["Lowest Low", "Highest High"], axis=1)

    def calculate_williams_percent_range(self, window=14) -> Type[pd.DataFrame]:
        """
        Returns
        -------
        pandas.DataFrame
            DataFrame with Williams % Range

        """
        data = self.calculate_stochastic_oscillator()
        highest_high = data["High"].rolling(window=window).max()
        lowest_low = data["Low"].rolling(window=window).min()

        data["W%R"] = (
            (highest_high - data["Smoothened Close"])
            / (highest_high - lowest_low)
            * -100
        )

        return data

    def calculate_obv(self) -> Type[pd.DataFrame]:
        """
        Returns
        -------
        pandas.DataFrame
            DataFrame with on balance volume

        """
        data = self.calculate_williams_percent_range()

        data["OBV"] = (
            (np.sign(data["Smoothened Close"].diff()) * data["Volume"])
            .fillna(0)
            .cumsum()
        )
        return data

    def generate_direction(self, shift_period=-1) -> Type[pd.DataFrame]:
        """
        Returns
        -------
        pandas.DataFrame
            DataFrame with direction `1` if there is an increase in the stock price with
            respect to the previous price, else `0`.
        """
        data = self.calculate_obv()
        # data['Future Return'] = data['Smoothened Close'].shift(shift_period) / data['Smoothened Close'] - 1
        # data['Direction'] = np.where(data['Future Return'] > 0, 1, 0)
        data["Direction"] = np.where(data["Smoothened Close"].diff() > 0, 1, 0)
        return data

    def drop_unnecessary_features(self) -> Type[pd.DataFrame]:
        """
        Returns
        -------
        pandas.DataFrame
            The final cleaned tabular data.
        """
        data = self.calculate_obv()
        return data.drop(
            [
                "Open",
                "High",
                "Low",
                "Stock Splits",
                "Dividends",
                "Close",
                "Moving Average",
                "Volume",
            ],
            axis=1,
        ).dropna()

    def return_final_data(self) -> Type[pd.DataFrame]:
        """
        Returns Scaled data
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = self.drop_unnecessary_features()
        X = data.drop(["Trend"], axis=1)
        y = data["Trend"]
        cols = X.columns
        X = pd.DataFrame(scaler.fit_transform(X), columns=cols, index=data.index)
        final_data = pd.concat([X, y], axis=1)
        return final_data
