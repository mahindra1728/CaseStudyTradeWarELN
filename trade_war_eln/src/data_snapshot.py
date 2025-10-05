"""Pull market metrics for the shortlisted equities using yfinance.

This helper regenerates the CSV summaries used to evidence liquidity and
volatility filters when selecting the two-name ELN basket.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

TICKERS = ["LMT", "INTC", "CSCO", "NUE", "SPY"]


def fetch_market_metrics(as_of: datetime | None = None) -> pd.DataFrame:
    if as_of is None:
        as_of = datetime.today()
    start = as_of - timedelta(days=365)
    rows = []
    for ticker in TICKERS:
        data = yf.download(ticker, start=start, end=as_of, auto_adjust=False)
        data = data.xs(ticker, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
        data['Return'] = data['Adj Close'].pct_change()
        one_year_return = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0] - 1) * 100
        realized_vol = data['Return'].tail(63).std() * (252 ** 0.5) * 100
        avg_volume = data['Volume'].tail(63).mean()
        rows.append({
            "Ticker": ticker,
            "1Y Return %": one_year_return,
            "90D Realized Vol %": realized_vol,
            "Average Volume (3M)": avg_volume,
        })
    return pd.DataFrame(rows)


def fetch_correlation(as_of: datetime | None = None) -> pd.DataFrame:
    if as_of is None:
        as_of = datetime.today()
    start = as_of - timedelta(days=365)
    price_panel = []
    index = None
    for ticker in TICKERS[:-1]:  # exclude SPY from correlation matrix for clarity
        data = yf.download(ticker, start=start, end=as_of, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            series = data.xs(ticker, axis=1, level=1)['Close']
        else:
            series = data['Close']
        if index is None:
            index = series.index
        price_panel.append(series.rename(ticker))
    prices = pd.concat(price_panel, axis=1)
    returns = prices.pct_change().dropna()
    return returns.corr()


def main() -> None:
    metrics = fetch_market_metrics()
    metrics.to_csv('outputs/market_metrics.csv', index=False)
    print(metrics.round(2))

    corr = fetch_correlation()
    corr.to_csv('outputs/correlation_matrix.csv')
    print('\nCorrelation matrix (last 1Y):')
    print(corr.round(2))


if __name__ == "__main__":
    main()
