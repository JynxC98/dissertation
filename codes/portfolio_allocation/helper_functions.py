"""
These are the helper functions which would assist 
the `StockSelection` and `Portfolio` classes
"""
from typing import List
import numpy as np

NUM_TRADING_DAYS = 252
RISK_FREE_RATE = 0.07 / 252


# The function below is used to feed into the `StockSelection Class`
def calculate_sharpe_and_sortino_ratio(
    returns: np.array, risk_free_rate=RISK_FREE_RATE
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
    excess_return = returns - risk_free_rate
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(excess_return.cov(), weights))
    ) * np.sqrt(n_days)
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            (portfolio_return - risk_free_rate * n_days) / portfolio_volatility,
        ]
    )
