# Python

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime


def fetch_and_predict_stock_prices(stock_symbol, start_date, end_date):
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Fetch the stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Prepare the features and target variables
    stock_data['Date'] = pd.to_datetime(stock_data.index)
    stock_data['Date_ordinal'] = stock_data['Date'].apply(lambda date: date.toordinal())

    X = stock_data[['Date_ordinal']]
    y = stock_data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the future prices
    predictions = model.predict(X_test)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    plt.plot(X_test, predictions, color='red', linewidth=2, label='Predicted Prices')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Output the model's performance
    score = model.score(X_test, y_test)
    print(f'Model R^2 score: {score:.2f}')


# Example usage
fetch_and_predict_stock_prices('AAPL', '2022-01-01', '2023-01-01')

######################### feature engineering analysis########

# Python

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime


def fetch_and_predict_with_features(stock_symbol, start_date, end_date):
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Fetch the stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Feature engineering
    stock_data['Moving_Average_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['Moving_Average_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Close_Change'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Close'].rolling(window=5).std()

    # Drop any rows with NaN values created by rolling calculations
    stock_data.dropna(inplace=True)

    # Prepare the features and target variables
    X = stock_data[['Moving_Average_5', 'Moving_Average_20', 'Close_Change', 'Volatility']]
    y = stock_data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the future prices
    predictions = model.predict(X_test)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices', color='blue')
    plt.plot(X_test.index, predictions, 'r', label='Predicted Prices', linewidth=2)
    plt.title(f'{stock_symbol} Stock Price Prediction with Features')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # Output the model's performance
    score = model.score(X_test, y_test)
    print(f'Model R^2 score: {score:.2f}')


# Example usage
fetch_and_predict_with_features('AAPL', '2022-01-01', '2023-01-01')

####### time series analysis ################
# Python

# Python

import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime


def arima_forecast(stock_symbol, start_date, end_date):
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Fetch the stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Prepare the data for ARIMA
    stock_data = stock_data['Close'].dropna()

    # Fit the ARIMA model
    model = ARIMA(stock_data, order=(1, 1, 1))  # simplest ARIMA, adjust p, d, q for better models
    fitted_model = model.fit()

    # Forecast future prices
    forecast_steps = 30  # number of days to forecast
    forecast = fitted_model.forecast(steps=forecast_steps)

    # Generate a date range for the forecast
    last_date = stock_data.index[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_steps)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index, stock_data, label='Historical Prices', color='blue')
    plt.plot(forecast_dates, forecast, color='red', linestyle='--', label='Forecasted Prices')
    plt.title(f'{stock_symbol} Stock Price Forecast with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
arima_forecast('AAPL', '2022-01-01', '2023-01-01')


