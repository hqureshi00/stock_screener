import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from utils.fetch_stock_data import fetch_stock_data
import pandas_ta as pa
from strategies.ma_crossover import crossover_signal
from strategies.rsi import generate_rsi_signals
from strategies.ema import ema_strategy
from sklearn.metrics import classification_report, accuracy_score
import time
import sys
import threading
import argparse


def test_model_for_different_parameters(stock_values, rsi_thresholds_range, ma_windows_range, ema_windows_range):
    # Define the ranges of values for RSI, MA, and EMA
    # rsi_thresholds_range = [(20, 80), (30, 70), (25, 75)]  # Example RSI buy/sell thresholds
    # ma_windows_range = [(5, 20), (10, 30), (11, 20)]  # Example MA window sizes
    # ema_windows_range = [(7, 14), (5, 20), (10, 30)]  # Example EMA window sizes

    # Store results for different combinations
    results = []

    for rsi_thresholds in rsi_thresholds_range:
        for ma_windows in ma_windows_range:
            for ema_windows in ema_windows_range:
                # Get data with the current set of indicators
                df = get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows)
                
                # Define the target based on future close price
                df['Future_Close'] = df['close'].shift(-5)
                df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
                
                # Features and target
                X = df[['RSI_signal', 'MA_signal', 'EMA_signal', 'open', 'close', 'high', 'low']]
                y = df['Target']
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Initialize the XGBoost classifier
                model = XGBClassifier()
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Predict on the test set
                y_pred = model.predict(X_test)
                
                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store the result
                results.append({
                    'RSI_thresholds': rsi_thresholds,
                    'MA_windows': ma_windows,
                    'EMA_windows': ema_windows,
                    'accuracy': accuracy
                })
                
                # Print classification report
                print(f"Results for RSI: {rsi_thresholds}, MA: {ma_windows}, EMA: {ema_windows}")
                print(classification_report(y_test, y_pred))
                print(f"Accuracy: {accuracy}\n")
    
    return results


# Spinner function to show progress
def spinner_func():
    spinner = ['-', '\\', '|', '/']
    idx = 0
    while not stop_spinner:
        sys.stdout.write(f"\r{spinner[idx % len(spinner)]} Processing...")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

# Decorator to run the spinner while the function executes
def spinner_decorator(func):
    def wrapper(*args, **kwargs):
        global stop_spinner
        stop_spinner = False

        # Start the spinner in a separate thread
        t = threading.Thread(target=spinner_func)
        t.start()

        # Execute the function
        result = func(*args, **kwargs)

        # Stop the spinner after function finishes
        stop_spinner = True
        t.join()

        sys.stdout.write("\rDone!          \n")
        sys.stdout.flush()

        return result
    return wrapper

# @spinner_decorator
def read_data(stock_name, interval, start_date, end_date):
  data = fetch_stock_data(stock_name, interval, start_date, end_date)

  return data

# @spinner_decorator
def get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows):

  stock, start_date, end_date, interval = stock_values
  df = fetch_stock_data(stock, interval, start_date, end_date)
  # breakpoint()
  
  signals = generate_rsi_signals(df, buy_threshold=rsi_thresholds[0], sell_threshold=rsi_thresholds[1])
  df['RSI_signal'] = signals['Buy_Sell']
  signals = crossover_signal(df, small_win=ma_windows[0], long_win=ma_windows[1])
  df['MA_signal'] = signals['Buy_Sell']
  signals = ema_strategy(df, short_window=ema_windows[0], long_window=ema_windows[1])
  df['EMA_signal'] = signals['Buy_Sell']
  # breakpoint()

  # df.dropna(inplace=True)
  # df.reset_index(drop=True, inplace=True)
  return df


def test_with_single_indicator_values(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-5)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    X = df[['MA_signal', 'EMA_signal', 'open', 'close', 'high', 'low']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
    model = XGBClassifier()

# Train the model
    model.fit(X_train, y_train)

  # Predict on the test set
    y_pred = model.predict(X_test)

# Evaluate the model's performance
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
  


def main():

  parser = argparse.ArgumentParser(
        description="Run Machine Learning Models"
    )
    
  parser.add_argument(
      "stock", 
      type=str, 
      help="The stock symbol (e.g., AAPL for Apple Inc.)"
  )
  parser.add_argument(
      "start_date", 
      type=str, 
      help="The start date in DD-MM-YYYY format (e.g., 01-01-2023)"
  )
  parser.add_argument(
      "end_date", 
      type=str, 
      help="The end date in DD-MM-YYYY format (e.g., 01-01-2023)"
  )

  parser.add_argument(
        "--interval", 
        type=str, 
        choices=["1min", "5min", "15min", "30min"], 
        default="1min", 
        required=True,
        help="The interval for the data. Options: '1min', '5min', '15min', '30min'. Default is '1min'."
    )
  
  parser.add_argument(
    "--benchmark", 
    action="store_true", 
    help="Run the benchmark function instead of the normal functionality."
)
  
  args = parser.parse_args()

  stock, start_date, end_date, interval = args.stock, args.start_date, args.end_date, args.interval
  stock_values = (stock, start_date, end_date, interval)
  # breakpoint()

  if args.benchmark:
     rsi_thresholds_range = [(20, 80), (30, 70), (25, 75)]  # Example RSI buy/sell thresholds
     ma_windows_range = [(5, 20), (10, 30), (11, 20)]  # Example MA window sizes
     ema_windows_range = [(7, 14), (5, 20), (10, 30)]  # Example EMA window sizes
    #  breakpoint()
     test_model_for_different_parameters(stock_values, rsi_thresholds_range, ma_windows_range, ema_windows_range)
  else:
     breakpoint()
     rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
     test_with_single_indicator_values(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

if __name__ == '__main__':
  main()
