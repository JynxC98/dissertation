"""
Script to simulate the necessary models.
"""
from typing import List, Type
from datetime import datetime, timedelta
import pandas as pd


class RollModel:
    """
    Uses the Roll model to calculate the transaction costs behind each trade.
    More about Roll model: https://www.bauer.uh.edu/rsusmel/phd/lecture%204.pdf

    Input parameters
    ----------------
    tickers: Indices of the stock
    start_date: The date from which the data is required.
    end_date: The date upto which the data is required.
    """

    def __init__(
        self, tickers: List, start_date: Type[datetime], end_date: Type[datetime]
    ) -> None:
        """
        Initialisation class of the Roll Model
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date


if __name__ == "__main__":
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(365 * 10)  # We use the data of 10 years
