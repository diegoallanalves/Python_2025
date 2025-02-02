# Python

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def analyze_and_save_stock_data(stock_symbol, start_date, end_date, file_name):
    # Fetch historical data for the given stock symbol
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Calculate moving average
    stock_data['Moving Average'] = stock_data['Close'].rolling(window=20).mean()

    # Save data to an Excel file
    stock_data.to_excel(file_name)
    print(f"Data saved to {file_name}")

    # Display summary statistics
    print(stock_data.describe())

    # Plot stock closing prices and moving average
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Closing Price')
    plt.plot(stock_data['Moving Average'], label='20-Day Moving Average', linestyle='--')

    plt.title(f'{stock_symbol} Stock Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
analyze_and_save_stock_data('AAPL', '2022-01-01', '2023-01-01', 'AAPL_stock_analysis.xlsx')
