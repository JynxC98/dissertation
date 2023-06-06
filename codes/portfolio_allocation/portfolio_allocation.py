"""
Script that would select the stocks which would outperform the market returns.
We will be using the NSE market for our stock selection.
"""
from typing import List, Type
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

RISK_FREE_RATE = (
    0.073  # Data provided by https://tradingeconomics.com/india/government-bond-yield
)


class StockSelection:
    """
    - Class used to select the top 15 best performing stocks based on the market
    excess return.
    - Only stocks with a positive excess return and a Sharpe ratio greater than 1
    are considered as outperforming the market.
    - We will be using yahoo finance api to gather the historical data.
    More about the yahoo finance api:  https://algotrading101.com/learn/yahoo-finance-api-guide/

    Input Parameters
    ----------------
    tickers: The list of tickers for which the historical data is required.
    Note that the tickers list must have the market index included.
    start_date: The start of the historical data.
    end_date: The end of the historical data.

    """

    def __init__(
        self, tickers: List, start_date: Type[datetime], end_date: Type[datetime]
    ) -> None:
        """
        Initialisation function of the `StockSelection` class.

        Note, the tickers must have the market index present
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        # These parameters will be stored after calculation

    def get_data_from_yahoo(self) -> Type[pd.DataFrame]:
        """

        Returns
        -------
        Historical data comprising of
        1. Opening price
        2. High
        3. Low
        4. Closing price
        5. Volume
        6. Dividends

        """
        stock_data = {}

        for stock in self.tickers:
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(start=self.start_date, end=self.end_date)
        return pd.DataFrame(stock_data)
