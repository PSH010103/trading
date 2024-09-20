import yfinance as yf
import pandas as pd

# Define the ticker symbol for Apple
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Download the stock data for a specific time range (from 2015 to the current date)
for ticker in tickers:
    stock_data = yf.download(ticker, start='2015-01-01', end='2024-09-20')
    stock_data = stock_data.reset_index()
    stock_data = stock_data.rename(columns={'Date': 'date'})
    stock_data = stock_data.set_index('date')
    stock_data.to_csv(f'stock_prices/{ticker}.csv')
