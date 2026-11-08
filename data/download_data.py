"""Download the market data used in the book from Yahoo Finance.

Run once on a computer with internet access:
    pip install yfinance pandas
    python download_data.py
It writes prices.csv (daily adjusted closing prices) in the current folder.
"""
import yfinance as yf

TICKERS = ["^GSPC", "^XU100", "AAPL", "MSFT", "JPM", "XOM"]
START, END = "2010-01-01", "2026-01-01"

data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
prices = data["Close"][TICKERS].dropna(how="all")
prices.index.name = "Date"
prices.to_csv("prices.csv", float_format="%.6f")
print(prices.shape, "rows x columns written to prices.csv")
print(prices.tail())
