"""
Script to simulate the necessary models.
"""
from typing import List, Type
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


class TradingStrategy:
    def __init__(
        self,
        data,
        initial_investment=10e6,
        transaction_cost=0.001,
        tax_rate=0.15,
        profit_threshold=10e5,
    ):
        self.data = data
        self.initial_investment = initial_investment
        self.transaction_cost = transaction_cost
        self.tax_rate = tax_rate
        self.profit_threshold = profit_threshold
        self.cash = initial_investment
        self.portfolio = {}
        self.trades = []

    def calculate_indicators(self):
        self.data["sma"] = SMAIndicator(
            close=self.data["Close"], window=14
        ).sma_indicator()
        self.data["rsi"] = RSIIndicator(close=self.data["Close"]).rsi()
        self.data["stoch"] = StochasticOscillator(
            high=self.data["High"], low=self.data["Low"], close=self.data["Close"]
        ).stoch()
        self.data["atr"] = AverageTrueRange(
            high=self.data["High"], low=self.data["Low"], close=self.data["Close"]
        ).average_true_range()
        self.data["williams"] = (
            self.data["Close"]
            .rolling(14)
            .apply(lambda x: ((np.max(x) - x[-1]) / (np.max(x) - np.min(x))) * -100)
        )

    def generate_signals(self):
        buy_signals = (
            (self.data["Close"] > self.data["sma"])
            & (self.data["rsi"] < 30)
            & (self.data["stoch"] < 20)
            & (self.data["williams"] > -80)
        )
        sell_signals = (
            (self.data["Close"] < self.data["sma"])
            & (self.data["rsi"] > 70)
            & (self.data["stoch"] > 80)
            & (self.data["williams"] < -20)
        )
        self.data["signal"] = np.where(
            buy_signals, "buy", np.where(sell_signals, "sell", "hold")
        )

    def execute_trades(self):
        for i, row in self.data.iterrows():
            if row["signal"] == "buy":
                shares_to_buy = self.cash // (
                    row["Close"] * (1 + self.transaction_cost)
                )
                self.cash -= shares_to_buy * row["Close"] * (1 + self.transaction_cost)
                self.portfolio[i] = shares_to_buy
                self.trades.append(("buy", i, row["Close"]))

            elif row["signal"] == "sell" and i in self.portfolio:
                shares_to_sell = self.portfolio[i]
                self.cash += shares_to_sell * row["Close"] * (1 - self.transaction_cost)
                del self.portfolio[i]
                self.trades.append(("sell", i, row["Close"]))

            if self.cash - self.initial_investment > self.profit_threshold:
                tax = (self.cash - self.initial_investment) * self.tax_rate
                self.cash -= tax

    def plot_signals(self):
        buys = self.data.loc[self.data["signal"] == "buy"]
        sells = self.data.loc[self.data["signal"] == "sell"]

        plt.figure(figsize=(12, 5))
        plt.plot(self.data["Close"], label="Close Price", color="blue", alpha=0.3)
        plt.scatter(
            buys.index,
            buys["Close"],
            color="green",
            label="Buy Signal",
            marker="^",
            alpha=1,
        )
        plt.scatter(
            sells.index,
            sells["Close"],
            color="red",
            label="Sell Signal",
            marker="v",
            alpha=1,
        )
        plt.title("Stock Price with Buy/Sell Signals")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def simulate(self):
        self.calculate_indicators()
        self.generate_signals()
        self.execute_trades()
        self.plot_signals()
