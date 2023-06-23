"""
These are the helper functions which would assist 
the `StockSelection` and `Portfolio` classes
"""
from typing import List, Type
import numpy as np
import pandas as pd

NUM_TRADING_DAYS = 252
RISK_FREE_RATE = 0.07 / 252


# The function below is used to feed into the `StockSelection Class`
def calculate_sharpe_and_sortino_ratio(
    returns: np.array, risk_free_rate=RISK_FREE_RATE
) -> List:
    r"""
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
    # portfolio_volatility = np.std(returns - risk_free_rate, ddof=1)
    portfolio_volatility = np.std(returns - risk_free_rate) * np.sqrt(NUM_TRADING_DAYS)
    sharpe_ratio = (excess_return) / portfolio_volatility

    # Calculation of Sortino Ratio
    downside_returns = np.where(returns < risk_free_rate, returns - risk_free_rate, 0)
    downside_std = np.std(downside_returns, ddof=1)
    sortino_ratio = (
        (excess_return / (downside_std * np.sqrt(NUM_TRADING_DAYS)))
        if np.std(downside_returns) != 0
        else 0
    )

    return [sharpe_ratio, sortino_ratio]


# The functions below are used to feed into the `Portfolio` class
def minimise_function(weights, returns) -> np.array:
    """
    Minimisation class for the given function
    """
    return -np.array(
        statistics(weights, returns)[2]
    )  # The maximum of f(x) is the minimum of -f(x)


def statistics(weights, returns, risk_free_rate=RISK_FREE_RATE, n_days=252) -> np.array:
    """
    Calculates the required statistics for optimisation function.
    Parameters
    ----------
    weights: Portfolio weights
    returns: Log daily returns
    n_days: Number of trading days
    """

    portfolio_return = np.sum(np.dot(returns.mean(), weights.T)) * n_days
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov(), weights))
    ) * np.sqrt(n_days)
    # portfolio_volatility = np.sqrt(
    #     np.dot(weights.T, np.dot(excess_return.cov(), weights))
    # ) * np.sqrt(n_days)
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            (portfolio_return - risk_free_rate * n_days) / portfolio_volatility,
        ]
    )


class VaRMonteCarloMulti:
    """
    VaR(Value at Risk) for multiple stocks using Monte Carlo simulation.

    Parameters
    ----------
    investment: The total amount invested in the portfolio.
    weights: list of weights for each stock in the portfolio.
    returns: list of mean returns for each stock.
    sigma: list of standard deviations for each stock.
    conf_int: Confidence interval.
    n_days: Number of days for which the VaR has to be calculated.

    Returns
    -------
    Value at Risk for the given portfolio.
    """

    NUM_ITERATIONS = 10000

    def __init__(
        self,
        investment: Type[int],
        weights: Type[List[float]],
        returns: Type[List[float]],
        sigma: Type[List[float]],
        conf_int: Type[float],
        n_days: Type[int],
    ):
        self.investment = investment
        self.weights = np.array(weights)
        self.returns = np.array(returns)
        self.sigma = np.array(sigma)
        self.conf_int = conf_int
        self.n_days = n_days

    def simulation(self):
        """
        Uses Monte Carlo simulation for calculating VaR
        """
        rand = np.random.normal(0, 1, [len(self.returns), self.NUM_ITERATIONS])

        # Compute the investment in each stock, based on its weight
        individual_investment = self.investment * self.weights

        stock_price = individual_investment * np.exp(
            self.n_days * (self.returns - 0.5 * pow(self.sigma, 2))
            + self.sigma * np.sqrt(self.n_days) * rand
        )

        portfolio_price = np.sum(stock_price, axis=0)

        percentile = np.percentile(portfolio_price, (1 - self.conf_int) * 100)

        return self.investment - percentile
