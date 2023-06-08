"""
Script that would select the stocks which would outperform the market returns.
We will be using the NIFTY 50 index for our stock selection.
"""
from typing import List
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from alive_progress import alive_bar

import matplotlib.pyplot as plt
import yfinance as yf

NUM_TRADING_DAYS = 252  # Assumption
RISK_FREE_RATE = (
    0.07
    / NUM_TRADING_DAYS  # Data provided by https://tradingeconomics.com/india/government-bond-yield
)  # Assuming risk free rate to be constant, we calculate daily risk free return


# The function outside the `StockSelection` class are used to feed the stock data.
def calculate_sharpe_and_sortino_ratio(
    returns: np.array, risk_free_rate: float
) -> List:
    """
    Function used to calculate the Sharpe and Sortino Ratio for the stocks. \\
    Sharpe ratio is given as (E[X] - rf)/std(excess returns) \\
    Sortino ratio is given as (E[X] - rf)/ std(negative asset returns)

    Inputs
    ------
    returns: Log returns of the daily stock data
    risk_free_rate: Tresury bond rates, assumed to be constant throughout the time period.

    Returns
    -------
    Sharpe ratio and Sortino ratio respectively

    """

    excess_return = np.mean(returns - risk_free_rate) * NUM_TRADING_DAYS

    # Calculation of Sharpe Ratio
    portfolio_volatility = np.std(returns - risk_free_rate, ddof=1)
    sharpe_ratio = (excess_return) / (portfolio_volatility * np.sqrt(NUM_TRADING_DAYS))

    # Calculation of Sortino Ratio
    downside_returns = np.where(returns < risk_free_rate, returns - risk_free_rate, 0)
    downside_std = np.std(downside_returns, ddof=1)
    sortino_ratio = (
        (excess_return / (downside_std * np.sqrt(NUM_TRADING_DAYS)))
        if np.std(downside_returns) != 0
        else 0
    )

    return [sharpe_ratio, sortino_ratio]


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

    def __init__(self, tickers: List, start_date: datetime, end_date: datetime) -> None:
        """
        Initialisation function of the `StockSelection` class.

        Note, the tickers must have the market index present
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        # These parameters will be stored after calculation
        self.portfolio_data = None
        self.returns = None

    def get_data_from_yahoo(self) -> pd.DataFrame:
        """
        The function generates the historical data of daily stock returns.

        Returns
        -------
        Closing price of the historical data
        """
        stock_data = {}

        for stock in self.tickers:
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(
                start=self.start_date, end=self.end_date
            )["Close"]
        return pd.DataFrame(stock_data)

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculates the percentage return of the data.
        """
        data = self.get_data_from_yahoo()
        returns = data.pct_change().dropna()
        self.returns = pd.DataFrame(returns)  # We save the value of log returns.
        return pd.DataFrame(
            returns
        )  # We skip the first row to eliminate the NaN values.

    def return_top_stocks(self) -> pd.DataFrame:
        """
        Returns the top 15 stocks based on sharpe ratio and sortino ratio.
        """
        stock_data = defaultdict(list)
        returns = self.calculate_returns()
        for stock in returns:  # We start from column 1 as column 0 is the index's data.
            stock_data["ticker"].append(stock)
            stock_data["sharpe ratio"].append(
                calculate_sharpe_and_sortino_ratio(returns[stock], RISK_FREE_RATE)[0]
            )
            stock_data["sortino ratio"].append(
                calculate_sharpe_and_sortino_ratio(returns[stock], RISK_FREE_RATE)[1]
            )

        stock_data = pd.DataFrame(stock_data)
        stock_data["kpi"] = (
            0.7 * stock_data["sharpe ratio"] + 0.3 * stock_data["sortino ratio"]
        )
        stock_data.sort_values(by=["kpi"], ascending=False, inplace=True)
        self.portfolio_data = stock_data[0:15]
        return stock_data[0:15]

    def plot_returns(self) -> None:
        """
        Plots the cumulative returns of the stock returns.
        """
        top_stocks = self.return_top_stocks()

        cumulative_returns = (1 + self.returns[top_stocks["ticker"]]).cumprod()
        fig = px.line(
            cumulative_returns,
            labels={"value": "times compounded", "Date": "year"},
            title="Annual compounding of the stocks",
        )
        fig.show()
