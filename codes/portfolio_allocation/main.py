"""
Running script for the portfolio allocation.
"""
from datetime import timedelta, datetime
from portfolio_allocation import StockSelection
from modern_portfolio_theory import Portfolio

import pandas as pd

if __name__ == "__main__":
    # Adding file path for the stock data
    path = r"~/dissertation/datasets/required_stocks.csv"
    END_TIME = datetime.now()
    START_TIME = END_TIME - timedelta(365 * 10)  # We take data of 10 years
    data = pd.read_csv(path)
    stocks = data["indices"][1:].tolist()
    portfolio = StockSelection(tickers=stocks, start_date=START_TIME, end_date=END_TIME)
    top_stocks = portfolio.return_top_stocks()
    print(top_stocks)
    portfolio.plot_returns()
    markowitz_model = Portfolio(stocks=stocks, start=START_TIME, end=END_TIME)
    print("generating mean-variance optimisation weights")
    markowitz_model.display_and_print_portfolio()
