from typing import Type
import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier
import yfinance as yf

from indicators import TechnicalIndicatorGenerator


def price_difference(price_1, price_2):
    """
    price_1: The base price
    price_2: The current price
    """
    return (price_1 - price_2) / price_1


class AlgorithmicTrading:
    r"""
    The Trade class represents a trading simulation. It encapsulates trading behavior 
    based on a trained machine learning model and applies it to a dataset representing 
    the price movements of a financial instrument over time.

    Attributes:
        capital (float): The available capital for trading. Adjusts dynamically with buying and selling. \
        stocks (list): List of all the stocks. \
        model (XGBoost Classifier): A trained XGBoost classifier for predicting the financial trend. \
        buy_signals (dict): A dictionary containing timestamps and prices of executed buy operations. \
        sell_signals (dict): A dictionary containing timestamps and prices of executed sell operations. \
        current_position (str): Current position of the trader, either 'CASH' (no shares owned) or 'HOLD' (shares owned).
        shares (int): The number of shares currently held. \
        scaler (MinMaxScaler): Scaler used for data normalization. \
        data_scaled (pandas.DataFrame): Scaled version of the input data, suitable for the machine learning model. \

    Methods:
        buy_shares(price: float): Executes a buying operation if the current capital allows it. \
        sell_shares(price: float): Executes a selling operation if there are shares in the current position. \
        execute_trade(): Executes a trading simulation based on the rules defined, historical data, and model predictions. \
        plot_trades(): Visualizes the trading simulation by plotting the prices and indicating buying and selling points. \
    """

    def __init__(
        self,
        capital: Type[int],
        model: Type[XGBClassifier],
        stock: Type[str],
        start: Type[datetime.date],
        end: Type[datetime.date],
    ):
        """
        Initialisation class of the class `Algorithmic Trading`.
        """
        self.capital = capital
        self.model = model
        self.initial_capital = capital
        self.stock = stock
        self.start = start
        self.end = end
        self.shares = 0  # We start with zero shares.
        self.position = "CASH"
        self.buy_signals = {}
        self.sell_signals = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.start_date = None
        self.data_scaled = None
        self.has_bought = (
            False  # flag indicating whether we've bought stocks at least once.
        )

    def get_data_from_yahoo(self):
        start = self.start
        end = self.end
        stock = self.stock
        ticker = yf.Ticker(stock)
        data_raw = ticker.history(start=start, end=end)
        indicator = TechnicalIndicatorGenerator(data_raw)
        required_data = indicator.drop_unnecessary_features()
        required_data.index = required_data.index.to_series().apply(
            lambda x: pd.to_datetime(x).date()
        )
        data_scaled = pd.DataFrame(
            self.scaler.fit_transform(
                required_data.drop(["Smoothened Close", "Trend"], axis=1)
            ),
            columns=required_data.drop(["Smoothened Close", "Trend"], axis=1).columns,
            index=required_data.index,
        )
        required_data = required_data.sort_index()
        data_scaled = data_scaled.sort_index()
        self.data = required_data
        self.data_scaled = data_scaled
        return required_data, data_scaled

    def predict(self, X):
        y_probab = self.model.predict_proba(X)
        return y_probab.argmax(axis=-1) - 1

    def buy_shares(self, price):
        shares_to_buy = self.capital // price
        if shares_to_buy > 0:
            self.capital -= shares_to_buy * price
            self.shares += shares_to_buy
            self.position = "HOLD"
            self.has_bought = True  # Updating the flag after a successful buy operation

    def sell_shares(self, price):
        if self.shares > 0:
            self.capital += self.shares * price
            self.shares = 0
            self.position = "CASH"

    def execute_date(self):
        data, data_scaled = self.get_data_from_yahoo()
        start_date = pd.to_datetime("2022-01-02").date()
        self.start_date = start_date
        observation_period = 21  # Assuming a month has 21 trading days.
        prices = data["Smoothened Close"].loc[start_date:]
        highs = data["Smoothened Close"].rolling(window=observation_period).max()
        lows = data["Smoothened Close"].rolling(window=observation_period).min()
        current_buy, current_sell = 0, 0

        for date, price in prices.iteritems():
            predicted_trend = self.predict(data_scaled.loc[[date]])

            if (
                predicted_trend == 1
                and price_difference(lows.loc[date], price) < 0.05
                and self.position == "CASH"
            ):
                self.buy_shares(price)
                self.buy_signals[date] = price
                current_buy = price

            elif (
                (predicted_trend == -1 or predicted_trend == 0)
                and price_difference(highs.loc[date], price) < 0.05
                and self.position == "HOLD"
                and price > 1.2 * current_buy
                and self.has_bought
            ):  # Check the flag here
                self.sell_shares(price)
                self.sell_signals[date] = price
                current_sell = price

            elif (
                price < 0.9 * current_buy and self.position == "HOLD"
            ):  # Stop loss irrespective of the trend
                self.sell_shares(price)
                self.sell_signals[date] = price
                current_sell = price

            elif (
                predicted_trend == -1
                and price < 0.95 * current_buy
                and self.position == "HOLD"
            ):  # Stop Loss
                self.sell_shares(price)
                self.sell_signals[date] = price
                current_sell = price

        return self.buy_signals, self.sell_signals

    def plot_trades(self):
        self.execute_date()
        start_date = self.start_date

        # Trace for the close price line
        close_trace = go.Scatter(
            x=self.data.loc[start_date:].index,
            y=self.data.loc[start_date:]["Smoothened Close"],
            mode="lines",
            name="Close price",
            line=dict(color="black"),
        )

        # Traces for buy and sell signals
        buy_traces = go.Scatter(
            x=list(self.buy_signals.keys()),
            y=list(self.buy_signals.values()),
            mode="markers",
            name="Buy",
            marker=dict(color="green", symbol="triangle-up", size=10),
        )
        sell_traces = go.Scatter(
            x=list(self.sell_signals.keys()),
            y=list(self.sell_signals.values()),
            mode="markers",
            name="Sell",
            marker=dict(color="red", symbol="triangle-down", size=10),
        )

        # Layout
        layout = go.Layout(
            title=f"Trade Execution Plot for {self.stock}",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            showlegend=True,
        )

        # Combine all traces into a single figure
        fig = go.Figure(data=[close_trace, buy_traces, sell_traces], layout=layout)

        # Show the figure
        fig.show()

    def return_stats(self):
        self.plot_trades()
        total_trades = len(self.buy_signals) + len(self.sell_signals)

        if self.shares > 0:  # If holding any shares
            # Get the current price of the stock
            current_price = self.data["Smoothened Close"][-1]
            # Portfolio value is capital left + (number of shares * current price)
            portfolio_value = self.capital + self.shares * current_price
            profit = (portfolio_value - self.initial_capital) / self.initial_capital

            return {
                "Total Trades": total_trades,
                "Remaining Capital": self.capital,
                "Current Portfolio Value": portfolio_value,
                "Profit": profit * 100,
            }

        else:
            portfolio_value = self.capital
            profit = (portfolio_value - self.initial_capital) / self.initial_capital
            return {
                "Total Trades": total_trades,
                "Remaining Capital": self.capital,
                "Profit Percentage": profit * 100,
            }
