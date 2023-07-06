"""
Script for creating the data used to feed into the XGBoost model
"""
# Inbuilt Libraries
import warnings
from typing import Type
from datetime import datetime

import numpy as np
import pandas as pd

import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

from indicators import TechnicalIndicatorGenerator

warnings.filterwarnings("ignore")


class DataPreprocessing:
    """
    The main class through which the raw data is supplied,
    Inputs
    ------
    stocks: The list of tickers.
    market_index: The main market_index
    start_date: The start of the historical date.
    end_date: The end of the historical date.
    investment: Total investment
    """

    def __init__(
        self,
        stocks: Type[list],
        start_date: Type[datetime],
        end_date: Type[datetime],
        investment: Type[int],
    ) -> None:
        """
        Initialisation of class `DataPreprocessing`
        """
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.investment = investment

    def get_data_from_yahoo(self) -> Type[pd.DataFrame]:
        """
        Fetches data using yahoo finance API.

        Returns a multi-indexed dataframe comprising of companies.
        """
        stock_data = {}
        for stock in self.stocks:
            try:
                ticker = yf.Ticker(stock)
                stock_data[stock] = ticker.history(
                    start=self.start_date, end=self.end_date
                )
            except:
                print(f"The data for ticker {stock} not found")

        return pd.DataFrame(stock_data)

    def get_required_features(self) -> Type[pd.DataFrame]:
        """ """
