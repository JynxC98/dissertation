"""
Script that would select the stocks which would outperform the market returns.
We will be using the NIFTY 50 index for our stock selection.
"""
from typing import List
from datetime import datetime, timedelta
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from alive_progress import alive_bar
import scipy.optimize as optimize
import yfinance as yf


from helper_functions import (
    calculate_sharpe_and_sortino_ratio,
    minimise_function,
    statistics,
)

NUM_TRADING_DAYS = 252  # Assumption
RISK_FREE_RATE = (
    0.07
    / NUM_TRADING_DAYS  # Data provided by https://tradingeconomics.com/india/government-bond-yield
)  # Assuming risk free rate to be constant, we calculate daily risk free return


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

    NUM_PORTFOLIO = (
        500000  # Number of random portfolios used to generate Efficient Frontier
    )

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
        self.weights = None
        self.top_stocks = None

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
        self.portfolio_data = stock_data[0:10]
        print(stock_data)
        return stock_data[0:10]

    def plot_returns(self) -> pd.DataFrame:
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

        # Code to create a folder for all the generated graphs.
        if not os.path.exists("images"):
            os.mkdir("images")
        fig.write_html("images/wealth_growth.html")
        self.top_stocks = top_stocks
        return top_stocks

    def generate_portfolios(self) -> dict:
        """
        Generates random portfolios for creating an efficient frontier.
        """
        portfolio_data = defaultdict(list)
        top_stocks = self.plot_returns()
        weights = []
        required_returns = self.returns[top_stocks["ticker"].tolist()]
        with alive_bar(self.NUM_PORTFOLIO) as pbar:
            print("Generating portfolios \n")
            for _ in range(self.NUM_PORTFOLIO):
                weight = np.random.random(len(top_stocks["ticker"]))
                weight /= np.sum(weight)
                weights.append(weight)

                portfolio_return = (
                    np.sum(required_returns.mean() * weight) * NUM_TRADING_DAYS
                )
                excess_return = required_returns - RISK_FREE_RATE
                portfolio_data["mean"].append(portfolio_return)
                portfolio_volatility = np.sqrt(
                    np.dot(
                        weight.T,
                        np.dot(excess_return.cov() * NUM_TRADING_DAYS, weight),
                    )
                )
                portfolio_data["risk"].append(portfolio_volatility)
                pbar()

        # Code to create a folder for all the graphs

        fig = px.scatter(
            data_frame=portfolio_data,
            x="risk",
            y="mean",
            color=(np.array(portfolio_data["mean"]) - RISK_FREE_RATE * 252)
            / np.array(portfolio_data["risk"]),
            color_continuous_scale="Viridis",
            labels={"risk": "Expected Volatility", "mean": "Expected Return"},
            title="Portfolio Analysis",
        )

        fig.update_layout(
            xaxis=dict(title="Expected Volatility"),
            yaxis=dict(title="Expected Return"),
            coloraxis_colorbar=dict(title="Sharpe Ratio"),
            showlegend=False,
        )
        fig.show()
        fig.write_html("images/random_portfolios.html")

        self.weights = np.array(weights)
        self.portfolio_data = portfolio_data

        return portfolio_data

    def optimize_portfolio(self) -> np.array:
        """
        Used to optimize the weights with respect to the sharpe ratio.
        It uses the SLSPQ algorithm to find the global minima of the function.
        Here is the documentation: https://docs.scipy.org/doc/scipy/reference/optimize.html

        Returns
        -------
        Optimal weights in which one should invest in the top stocks.
        """
        _ = self.generate_portfolios()
        returns = self.returns[self.top_stocks["ticker"].tolist()]
        func = minimise_function
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # The weights can at the most be 1.
        bounds = tuple((0, 1) for _ in range(len(self.top_stocks["ticker"])))
        random_weights = np.random.random(len(self.top_stocks["ticker"]))
        random_weights /= np.sum(random_weights)
        optimum = optimize.minimize(
            fun=func,
            x0=np.array(
                random_weights
            ),  # We are randomly selecting a weight for optimisation
            args=returns,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return optimum["x"].round(4)

    def display_statistics(self, weights):
        """
        Displays the Sharpe Ratio, Expected return and the volatility of the
        given portfolio.
        """
        return (
            "Expected return, volatility and Sharpe ratio: ",
            statistics(weights.round(3), self.returns[self.top_stocks["ticker"]]),
        )

    def display_and_print_portfolio(self) -> None:
        """
        Generates the point on the efficient portfolio frontier where
        the portfolio shows the optimal return and risk.
        """
        optimal = self.optimize_portfolio()
        portfolio_data = self.portfolio_data
        result = {}
        for stock, optimum_weight in zip(self.top_stocks["ticker"], optimal):
            result[stock] = optimum_weight
        print(self.display_statistics(optimal))
        print(result)
        fig = px.scatter(
            data_frame=portfolio_data,
            x="risk",
            y="mean",
            color=(np.array(portfolio_data["mean"]) - RISK_FREE_RATE * 252)
            / np.array(portfolio_data["risk"]),
            color_continuous_scale="Viridis",
            title="Portfolio Analysis",
        )

        fig.add_trace(
            go.Scatter(
                x=[statistics(optimal, self.returns[self.top_stocks["ticker"]])[1]],
                y=[statistics(optimal, self.returns[self.top_stocks["ticker"]])[0]],
                mode="markers",
                marker=dict(color="red", size=15),
            )
        )

        fig.update_layout(
            xaxis=dict(title="Expected Volatility"),
            yaxis=dict(title="Expected Return"),
            coloraxis_colorbar=dict(title="Sharpe Ratio"),
            showlegend=True,
        )
        fig.write_html("images/efficient_portfolio.html")
        fig.show()


if __name__ == "__main__":
    # Adding file path for the stock data
    path = r"~/dissertation/datasets/required_stocks.csv"
    END_TIME = datetime.now()
    START_TIME = END_TIME - timedelta(365 * 10)  # We take data of 10 years
    data = pd.read_csv(path)
    # sample_stocks = ["AAPL", "AMZN", "MSFT", "JPM", "DB", "NVDA"]
    stocks = data["indices"][1:].tolist()
    portfolio = StockSelection(tickers=stocks, start_date=START_TIME, end_date=END_TIME)
    portfolio.display_and_print_portfolio()
