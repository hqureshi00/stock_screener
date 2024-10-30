import argparse
import sys
import threading
import time
import matplotlib.pyplot as plt
<<<<<<< HEAD
import seaborn as sns
import numpy as np
import pandas as pd
import pandas_ta as pa
import ta  
from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot
from xgboost.callback import EarlyStopping

from strategies.ema import ema_strategy
from strategies.ma_crossover import crossover_signal
from strategies.rsi import generate_rsi_signals
from utils.fetch_stock_data import fetch_stock_data

=======
import numpy as np
import pandas as pd
import pandas_ta as pa
import seaborn as sns
import ta  # Technical Analysis library

>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
<<<<<<< HEAD
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

def find_correlated_features(df):

    correlation_matrix = df.corr()

    # Set a threshold for high correlation
    threshold = 0.8

    # Find highly correlated features
    highly_correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                highly_correlated_pairs.append((feature1, feature2, correlation_matrix.iloc[i, j]))

    # Convert to DataFrame for better readability
    highly_correlated_df = pd.DataFrame(highly_correlated_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
    
    print(highly_correlated_df)

=======
from xgboost import XGBClassifier

from strategies.ema import ema_strategy
from strategies.ma_crossover import crossover_signal
from strategies.rsi import generate_rsi_signals
from utils.fetch_stock_data import fetch_stock_data
from sklearn.feature_selection import RFE

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5
def calculate_price_increase_accuracy(data, model, X_test):
    # Get predictions from the model
    y_pred = model.predict(X_test)
    
    # Extract the rows where the model predicted `1` (i.e., price increase)
    predicted_increase_indices = X_test.index[y_pred == 1]
    
    # Get actual prices 5 days later for those rows
    actual_future_prices = data.loc[predicted_increase_indices, 'Future_Close']
    current_prices = data.loc[predicted_increase_indices, 'close']
    
    # Check if actual future price is greater than current price
    correct_increase_predictions = (actual_future_prices > current_prices).sum()
    
    # Total number of times model predicted price increase (`1`)
    total_increase_predictions = len(predicted_increase_indices)
    
    # Calculate accuracy as a percentage
    if total_increase_predictions > 0:
        accuracy = correct_increase_predictions / total_increase_predictions
    else:
        accuracy = 0  # Avoid division by zero if no `1` was predicted
    
    print(f"Accuracy for price increase predictions (1): {accuracy * 100:.2f}%")
    return accuracy

# Example function to create sequential data for LSTM
def create_lstm_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :-1])  # Input features
        y.append(data[i + window_size, -1])    # Target variable
    return np.array(X), np.array(y)

def compare_models(stock_values, rsi_thresholds, ma_windows, ema_windows):
    breakpoint()
    # Get data with technical indicators
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows)
    df['Future_Close'] = df['close'].shift(-20)  # Predict 5 days ahead
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)  # Binary target
    df.dropna(inplace=True)

    # Features and target
    # X = df[['RSI_signal', 'MA_signal', 'EMA_signal', 'open', 'close', 'high', 'low']]
    X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month',
<<<<<<< HEAD
    'MA_signal', 'EMA_signal', 'Volatility', 'close_lag_1', 'close_lag_5', 'close_lag_20', 'Normalized_Volume', 'open', 'close', 'high', 'low']
=======
    'MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']
>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5
    ]
    y = df['Target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to evaluate
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'XGBoost': XGBClassifier()
    }

    # Evaluate each model
    results = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        print(f"Results for {model_name}:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy}\n")

    # Plot results
    plt.bar(results.keys(), results.values())
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.show()

def test_model_for_different_horizons(stock_values, rsi_thresholds, ma_windows, ema_windows, horizons):
    
    # Store results for different horizons
    results = []
    
    for horizon in horizons:
        # Get data with the current set of indicators
        df = get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows)
        
        # Define the target based on future close price for the given horizon
        df['Future_Close'] = df['close'].shift(-horizon)
        df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
        
        # Drop NaN rows (due to shifting)
        df.dropna(inplace=True)
        
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
        # breakpoint()
        # accuracy = calculate_price_increase_accuracy(stock_values, model, X_test)
        
        # Store the result
        results.append({
            'horizon': horizon,
            'accuracy': accuracy
        })
        
        # Print classification report for each horizon
        print(f"Results for horizon: {horizon}")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy}\n")
    
    return results

# Plotting the accuracy for different horizons
def plot_accuracy_for_horizons(results):
    horizons = [result['horizon'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, accuracies, marker='o')
    plt.title('Model Accuracy vs Different Horizons')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def calculate_volatility(stock_values, window=14):
    """
    Calculate the rolling volatility (standard deviation of returns) for a given window.
    
    Args:
        stock_values (DataFrame): A DataFrame with at least the 'close' column.
        window (int): The rolling window size for volatility calculation.
    
    Returns:
        DataFrame: The stock data with an additional column 'Volatility'.
    """
    df = stock_values.copy()
    
    # Calculate the daily returns
    df['Returns'] = df['close'].pct_change()
    
    # Calculate rolling volatility (standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    
    # Fill NaN values with 0 for the initial rows where rolling window can't be applied
    df['Volatility'].fillna(0, inplace=True)
    
    return df

def calculate_normalized_volume(stock_values, window=20):
    """
    Calculate normalized volume, which is the volume relative to its moving average.
    
    Args:
        stock_values (DataFrame): A DataFrame with at least the 'volume' column.
        window (int): The rolling window size for normalizing the volume.
    
    Returns:
        DataFrame: The stock data with an additional column 'Normalized_Volume'.
    """
    df = stock_values.copy()
    
    # Calculate the rolling mean of volume
    df['Volume_MA'] = df['volume'].rolling(window=window).mean()
    
    # Calculate normalized volume (current volume / moving average of volume)
    df['Normalized_Volume'] = df['volume'] / df['Volume_MA']
    
    # Fill NaN values with 1 for the initial rows where rolling window can't be applied
    df['Normalized_Volume'].fillna(1, inplace=True)
    
    return df

def test_model_for_different_parameters(stock_values, rsi_thresholds_range, ma_windows_range, ema_windows_range):
    
    # Store results for different combinations
    results = []

    for rsi_thresholds in rsi_thresholds_range:
        for ma_windows in ma_windows_range:
            for ema_windows in ema_windows_range:
                # Get data with the current set of indicators
                df = get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows)
                # Define the target based on future close price
                df['Future_Close'] = df['close'].shift(-10)
                df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
                # So basically if future is greater than one or not (which is the actual price isn't it?)
                # So you calculate the indicators, set the target as actual price
                # Remember we are not predicting the indicators, we are predicting the target, based on RSI
                
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

@spinner_decorator
def read_data(stock_name, interval, start_date, end_date):
  data = fetch_stock_data(stock_name, interval, start_date, end_date)

  return data


def get_data_with_indicators_test(stock_values, rsi_thresholds, ma_windows, ema_windows):

  stock, start_date, end_date, interval = stock_values
  df = fetch_stock_data(stock, interval, start_date, end_date)
<<<<<<< HEAD
  #breakpoint()
=======
#   breakpoint()
#   df = df.sort_values('date')
>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5
  signals = generate_rsi_signals(df, buy_threshold=rsi_thresholds[0], sell_threshold=rsi_thresholds[1])
  df['RSI_signal'] = signals['Buy_Sell']
  signals = crossover_signal(df, small_win=ma_windows[0], long_win=ma_windows[1])
  df['MA_signal'] = signals['Buy_Sell']
  signals = ema_strategy(df, short_window=ema_windows[0], long_window=ema_windows[1])
  df['EMA_signal'] = signals['Buy_Sell']
  signals = calculate_volatility(df)
  df['Volatility'] = signals['Volatility']
  signals = calculate_normalized_volume(df)
  df['Normalized_Volume'] = signals['Normalized_Volume']
<<<<<<< HEAD
=======

  ##########################

  df['bollinger_mavg'] = df['close'].rolling(window=20).mean()
  df['bollinger_std'] = df['close'].rolling(window=20).std()
  df['bollinger_upper'] = df['bollinger_mavg'] + (df['bollinger_std'] * 2)
  df['bollinger_lower'] = df['bollinger_mavg'] - (df['bollinger_std'] * 2)

    # MACD
  df['macd'] = ta.trend.macd(df['close'])
  df['macd_signal'] = ta.trend.macd_signal(df['close'])
  df['macd_diff'] = df['macd'] - df['macd_signal']

    # ATR
  df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

#     # Lagged Price Features
#   for lag in [1, 2, 5, 10]:
#     df[f'close_lag_{lag}'] = df['close'].shift(lag)

    # Rolling Statistics
  df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
  df['rolling_std_20'] = df['close'].rolling(window=20).std()
  df['rolling_volume_mean_20'] = df['volume'].rolling(window=20).mean()

  ## Day of the week
  df['day_of_week'] = df['timestamp'].dt.dayofweek
  df['day_of_month'] = df['timestamp'].dt.day
>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5

  ##########################

  df['bollinger_mavg'] = df['close'].rolling(window=20).mean()
  df['bollinger_std'] = df['close'].rolling(window=20).std()
  df['bollinger_upper'] = df['bollinger_mavg'] + (df['bollinger_std'] * 2)
  df['bollinger_lower'] = df['bollinger_mavg'] - (df['bollinger_std'] * 2)

    # MACD
  df['macd'] = ta.trend.macd(df['close'])
  df['macd_signal'] = ta.trend.macd_signal(df['close'])
  df['macd_diff'] = df['macd'] - df['macd_signal']

    # ATR
  df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

  # Rolling Statistics
  # df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
  # df['rolling_std_20'] = df['close'].rolling(window=20).std()
  df['rolling_volume_mean_20'] = df['volume'].rolling(window=20).mean()

  ## Day of the week
  df['day_of_week'] = df['timestamp'].dt.dayofweek
  df['day_of_month'] = df['timestamp'].dt.day

  # Determine if each day is a Friday
  df['isFridayClosing'] = df['timestamp'].apply(lambda x: 1 if x.weekday() == 4 and x.time() == pd.Timestamp('16:00:00').time() else 0)

  # Create a new column to hold the copied Friday closing values
  df['Prev_Friday_Close'] = df.apply(lambda row: row['close'] if row['isFridayClosing'] == 1 else None, axis=1)

  # Forward-fill the 'Prev_Friday_Close' column to propagate the Friday close value until the next Friday
  df['Prev_Friday_Close'] = df['Prev_Friday_Close'].ffill()

  # Calculate percentage changes from the last Friday's close
  df['percent_open'] = ((df['open'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_close'] = ((df['close'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_high'] = ((df['high'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_low'] = ((df['low'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100

  # Now, calculate Bollinger Bands as percentages of the last Friday's close
  df['percent_bollinger_mavg'] = ((df['bollinger_mavg'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_bollinger_upper'] = ((df['bollinger_upper'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_bollinger_lower'] = ((df['bollinger_lower'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100

  # Calculate the Bollinger Band width
  df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']

  # Calculate the delta (difference) in Bollinger Band widths between consecutive rows
  df['bollinger_width_delta'] = df['bollinger_width'].diff()

  # Calculate the delta (difference) in Bollinger Band widths over 3 rows 
  df['bollinger_width_delta_3'] = df['bollinger_width'].diff(periods=3)
  
  df['rolling_mean_20'] = df['percent_close'].rolling(window=20).mean()
  df['rolling_std_20'] = df['percent_close'].rolling(window=20).std()
  df['rolling_mean_5'] = df['percent_close'].rolling(window=5).mean()
  df['rolling_std_5'] = df['percent_close'].rolling(window=5).std()

    # Ensure that the timestamp column is in datetime format
#   if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
#      df['timestamp'] = pd.to_datetime(df['timestamp'])
#   # Check if timestamps are in strictly increasing order
#   if df['timestamp'].is_monotonic_increasing:
#       print("Timestamps are in strictly increasing order.")
#   else:
#       print("Timestamps are NOT in strictly increasing order.")

  # Filter rows where the time is between 08:00:00 and 16:00:00
  df = df[(df['timestamp'].dt.time >= pd.to_datetime('08:00:00').time()) & 
        (df['timestamp'].dt.time <= pd.to_datetime('16:00:00').time())]

  df.reset_index(drop=True, inplace=True)

  # Ensure the timestamp is in datetime format
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Extract time of day from the timestamp
  df['time_of_day'] = df['timestamp'].dt.time

  # Calculate the average volume for each time of day over the last 5 trading days
  df['avg_volume_last_5_days'] = df.groupby('time_of_day')['volume'].rolling(window=20).mean().reset_index(level=0, drop=True)

  df['volume_spike'] = df['volume'] / df['avg_volume_last_5_days']

  volume_threshold = 2.0
  df['large_volume_indicator'] = df['volume_spike'] > volume_threshold

  df.to_csv('training_data.csv', index=True)


#       breakpoint()

  return df
# @spinner_decorator
def get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows):

  stock, start_date, end_date, interval = stock_values
  df = fetch_stock_data(stock, interval, start_date, end_date)
#   breakpoint()
#   df = df.sort_values('date')
  signals = generate_rsi_signals(df, buy_threshold=rsi_thresholds[0], sell_threshold=rsi_thresholds[1])
  df['RSI_signal'] = signals['Buy_Sell']
  signals = crossover_signal(df, small_win=ma_windows[0], long_win=ma_windows[1])
  df['MA_signal'] = signals['Buy_Sell']
  signals = ema_strategy(df, short_window=ema_windows[0], long_window=ema_windows[1])
  df['EMA_signal'] = signals['Buy_Sell']
  signals = calculate_volatility(df)
  df['Volatility'] = signals['Volatility']
  signals = calculate_normalized_volume(df)
  df['Normalized_Volume'] = signals['Normalized_Volume']

  ##########################

  df['bollinger_mavg'] = df['close'].rolling(window=20).mean()
  df['bollinger_std'] = df['close'].rolling(window=20).std()
  df['bollinger_upper'] = df['bollinger_mavg'] + (df['bollinger_std'] * 2)
  df['bollinger_lower'] = df['bollinger_mavg'] - (df['bollinger_std'] * 2)

    # MACD
  df['macd'] = ta.trend.macd(df['close'])
  df['macd_signal'] = ta.trend.macd_signal(df['close'])
  df['macd_diff'] = df['macd'] - df['macd_signal']

    # ATR
  df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

  df['close_lag_1'] = df['close'].shift(1)
  df['close_lag_5'] = df['close'].shift(5)
  df['close_lag_20'] = df['close'].shift(20)
  df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
  df['rolling_std_20'] = df['close'].rolling(window=20).std()
  df['rolling_volume_mean_20'] = df['volume'].rolling(window=20).mean()

  ## Day of the week
  df['day_of_week'] = df['timestamp'].dt.dayofweek
  df['day_of_month'] = df['timestamp'].dt.day
#   df['timestamp'] = df['timestamp']
#   df.fillna(method='bfill', inplace=True)
# #   df = df.dropna(inplace=True)
#   df_dropped = df.drop('timestamp', axis=1)
#   scaler = StandardScaler()
#   df_scaled = scaler.fit_transform(df_dropped)
#   df_scaled_ = pd.DataFrame(df_scaled, columns=df_dropped.columns)
#   df.fillna(method='bfill', inplace=True)
  # df.reset_index(drop=True, inplace=True)
  find_correlated_features(df)

  
#   plt.figure(figsize=(12, 10))  # Set the figure size
#   sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")

# # Add title and labels
#   plt.title('Correlation Matrix')
#   plt.show()

  return df

def get_technical_features_for_lstm(stock_values):
    rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-5)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]
 
    y = df['Target']

    return 
<<<<<<< HEAD
=======

def test_data_lstm(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-5)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]
    y = df['Target']

    return df

def test_with_rfe_features(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-20)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    # X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]
    breakpoint()
    X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month',
    'MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']
    ]
    y = df['Target']
    print("Class distribution in the dataset:")
    print(y.value_counts())
    # breakpoint()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Class distribution in the training set:")
    print(y_train.value_counts())


   # Initialize the XGBoost classifier
    model = XGBClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

   # Evaluate the model's performance
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    #########RFE###########

    rfe = RFE(estimator=model, n_features_to_select=7)  # You can choose the number of features you want to select

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Transform the data to only include selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Train the model on the reduced dataset
    model.fit(X_train_rfe, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy with RFE selected features: {accuracy}")
    print("Selected features:", X.columns[rfe.support_])



def test_with_single_indicator_values(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-20)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    # X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]
    breakpoint()
    X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month',
    'MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']
    ]
    y = df['Target']
    print("Class distribution in the dataset:")
    print(y.value_counts())
    # breakpoint()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Class distribution in the training set:")
    print(y_train.value_counts())

>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5

def test_data_lstm(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-5)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]
    y = df['Target']

    return df

def test_with_rfe_features(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-20)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    # X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]
    X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month',
    'MA_signal', 'EMA_signal', 'Volatility', 'close_lag_1', 'close_lag_5', 'close_lag_20', 'Normalized_Volume', 'open', 'close', 'high', 'low']
    ]
    y = df['Target']
    print("Class distribution in the dataset:")
    print(y.value_counts())
    # breakpoint()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Class distribution in the training set:")
    print(y_train.value_counts())


   # Initialize the XGBoost classifier
    model = XGBClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

   # Evaluate the model's performance
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    #########RFE###########

    rfe = RFE(estimator=model, n_features_to_select=7)  # You can choose the number of features you want to select

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Transform the data to only include selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Train the model on the reduced dataset
    model.fit(X_train_rfe, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy with RFE selected features: {accuracy}")
    print("Selected features:", X.columns[rfe.support_])

def get_normalized_features(df):

    scaler = StandardScaler()
    df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower']] = scaler.fit_transform(df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower']])

    df[['macd', 'macd_signal', 'macd_diff']] = scaler.fit_transform(df[['macd', 'macd_signal', 'macd_diff']])
    df['atr'] = scaler.fit_transform(df[['atr']])

    df[['rolling_volume_mean_20']] = scaler.fit_transform(df[['rolling_volume_mean_20']])

    
    min_max_scaler = MinMaxScaler()
    # df[['day_of_week', 'day_of_month']] = min_max_scaler.fit_transform(df[['day_of_week', 'day_of_month']])
    df = pd.get_dummies(df, columns=['day_of_week', 'day_of_month'], drop_first=True)

    # df[['MA_signal', 'EMA_signal']] = scaler.fit_transform(df[['MA_signal', 'EMA_signal']])
    df['Volatility'] = scaler.fit_transform(df[['Volatility']])
    # df[['close_lag_1', 'close_lag_5', 'close_lag_20']] = scaler.fit_transform(df[['close_lag_1', 'close_lag_5', 'close_lag_20']])
    df['Normalized_Volume'] = scaler.fit_transform(df[['Normalized_Volume']])
    df[['open', 'close', 'high', 'low']] = scaler.fit_transform(df[['open', 'close', 'high', 'low']])
   

    return df


def plot_cross_validation(model):

    results = model.evals_result()
    breakpoint()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()

def get_train_test_data(X, y):
    # Calculate the split index
    split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing

    # Split the data
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

def test_with_single_indicator_values(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators_test(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

    df['Future_Close'] = df['close'].shift(-20)

  # if Target == 1, its a profitable buy signal
  # if Target == 0, its a profitable sell signal
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    # X = df[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']]

    # X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    # 'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month',
    # 'MA_signal', 'EMA_signal', 'Volatility', 'close_lag_1', 'close_lag_5', 'close_lag_20', 'Normalized_Volume', 'open', 'close', 'high', 'low']
    # ]

    # X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    # 'atr', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month', 'Volatility', 'close_lag_1', 'close_lag_5', 'close_lag_20', 'Normalized_Volume', 'open', 'close', 'high', 'low']



    X = df[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_volume_mean_20', 'day_of_week', 'day_of_month', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']
    ]

    breakpoint()

    X = get_normalized_features(X)
    y = df['Target']
    print("Class distribution in the dataset:")
    d = y.value_counts()
    print(y.value_counts())
    # breakpoint()

    #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # X_train, X_test, y_train, y_test = get_train_test_data(X, y)
    print("Class distribution in the training set:")
    print(y_train.value_counts())
    e = y_train.value_counts()
    f = y_test.value_counts()


# Initialize the XGBoost classifier
    # model = XGBClassifier()
    model = XGBClassifier()
#     model = XGBClassifier(
#     n_estimators=500,        # Reduce estimators if overfitting is observed
#     learning_rate=0.01,      # Lower learning rate to train slower
#     max_depth=3,             # Limit the depth to prevent deep trees
#     subsample=0.8,           # Use only 80% of the data to train each tree
#     colsample_bytree=0.8,    # Use only 80% of features to train each tree
#     reg_lambda=10,           # Stronger L2 regularization (increase if needed)
#     reg_alpha=1,             # Stronger L1 regularization (increase if needed)
#     gamma=5,                 # Further constrain tree splits
#     scale_pos_weight=1       # Adjust if classes are imbalanced
# )

# Train the model
    # eval_set = [(X_test, y_test)]
    # model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)

    eval_set = [(X_train, y_train), (X_test, y_test)]

   # Remove eval_metric from fit
    model.fit(X_train, y_train, 
          eval_set=eval_set, 
          verbose=True)
    # model.fit(X_train, y_train)
    # model.fit(X_train, y_train, 
    #       eval_set=[(X_test, y_test)], 
    #       eval_metric="logloss",
    #       early_stopping_rounds=50, # Early stopping
    #       verbose=True)

  # Predict on the test set
    y_pred = model.predict(X_test)

# Evaluate the model's performance
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

<<<<<<< HEAD

    ### TEst for training

    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Test accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    


# Predict on the test set
    y_pred = model.predict(X_test)

# Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:\n", cm)

=======
>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5
    #####Print importances

# Get feature importance
    importances = model.feature_importances_

# Create a DataFrame for easy plotting
    feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
    }).sort_values(by='Importance', ascending=False)

# Print feature importance
    print(feature_importances)

<<<<<<< HEAD
    breakpoint()

    ### model for cv
    model_for_cv = XGBClassifier(eval_metric=["logloss", "error"])
    

    # eval_set = [(X_train, y_train), (X_test, y_test)]

#     # Create the early stopping callback
#     early_stopping = EarlyStopping(rounds=10, save_best=True, maximize=False, data_name='validation_0', metric_name='logloss')

# # Fit the model with the callback
#     model_for_cv.fit(X_train, y_train,
#                  eval_set=eval_set,
#                  callbacks=[early_stopping],
#                  verbose=True)

   # Remove eval_metric from fit
    model_for_cv.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    tscv = TimeSeriesSplit(n_splits=5)
#     count = 1
#     for train_index, test_index in tscv.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#         eval_set = [(X_train, y_train), (X_test, y_test)]
        
#         # Fit your model here
#         model_for_cv.fit(X_train, y_train, eval_set=eval_set)

#         y_pred = model.predict(X_test)

# # Evaluate the model's performance
#         print(classification_report(y_test, y_pred))
#         print(f"Accuracy {count}: {accuracy_score(y_test, y_pred)}")
#         count += 1

    # tscv = TimeSeriesSplit(n_splits=5)
    cross_val_scores = cross_val_score(model_for_cv, X, y, cv=tscv, scoring='accuracy')
    print(f"Mean accuracy: {cross_val_scores.mean()}")
    print(f"Standard deviation: {cross_val_scores.std()}")

    plot_cross_validation(model_for_cv)

=======
>>>>>>> 85ea2d623823eb33398205b53e63cc5eeca557f5
# Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importance')
    plt.show()

    return accuracy_score(y_test, y_pred)
  
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
  
  parser.add_argument(
    "--accuracy", 
    action="store_true", 
    help="plot the accuracy of the ML model using different future price targets"
)
  parser.add_argument(
    "--compare", 
    action="store_true", 
    help="Compare different machine learning models"
)
  
  parser.add_argument(
    "--lstm", 
    action="store_true", 
    help="for lstm combined model with xgb"
)
  
  parser.add_argument(
    "--rfe", 
    action="store_true", 
    help="for rfe accuracy comparison"
)
  args = parser.parse_args()

  stock, start_date, end_date, interval = args.stock, args.start_date, args.end_date, args.interval
  stock_values = (stock, start_date, end_date, interval)

  if args.benchmark:
     rsi_thresholds_range = [(20, 80), (30, 70), (25, 75)]  # Example RSI buy/sell thresholds
     ma_windows_range = [(5, 20), (10, 30), (11, 20)]  # Example MA window sizes
     ema_windows_range = [(7, 14), (5, 20), (10, 30)]  # Example EMA window sizes
     test_model_for_different_parameters(stock_values, rsi_thresholds_range, ma_windows_range, ema_windows_range)

  if args.accuracy:
     rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
     horizons = [1, 5, 10, 15, 20]  # Different prediction horizons (1 day, 5 days, 10 days, etc.)
     breakpoint()
     results = test_model_for_different_horizons(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds, horizons)
     plot_accuracy_for_horizons(results)

  if args.compare:
     rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
     breakpoint()
     compare_models(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

  if args.rfe:
      rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
      acc = test_with_rfe_features(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

  else:
    rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
    accuracy = test_with_single_indicator_values(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

#   if args.lstm:
#      rsi_thresholds, ma_thresholds, ema_thresholds = [(30, 70), (11, 20), (7, 14)]
#      df = test_data_lstm(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
#      lstm_xgboost_combined_model(df)

# def lstm_xgboost_combined_model(stock_values, window_size=60):
    
#     # Ensure stock_values contains columns: RSI, MA, EMA, open, close, high, low
#     # And target (which is the future movement of the price, 1 for up, 0 for down)
    
#     # Select relevant features and target
#     features = stock_values[['MA_signal', 'EMA_signal', 'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']].values
#     target = stock_values['Target'].values.reshape(-1, 1)  # Target is 1 or 0 (future price movement)

#     # Combine features and target for LSTM input
#     data = np.hstack((features, target))

#     # Create LSTM dataset (with sliding window)
#     X_lstm, y_lstm = create_lstm_dataset(data, window_size)

#     # Train/test split for LSTM
#     X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

#     # Build the LSTM model
#     lstm_model = Sequential()
#     lstm_model.add(LSTM(64, return_sequences=False, input_shape=(window_size, X_train_lstm.shape[2])))
#     lstm_model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

#     lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     # Train the LSTM model
#     lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

#     # Get LSTM output (features) for both train and test data
#     train_lstm_features = lstm_model.predict(X_train_lstm)
#     test_lstm_features = lstm_model.predict(X_test_lstm)

#     # Extract original technical indicator features corresponding to the LSTM outputs
#     X_train_tech = features[window_size:len(train_lstm_features) + window_size]
#     X_test_tech = features[len(train_lstm_features) + window_size:]

#     # Combine LSTM features with technical indicators for XGBoost
#     X_train_combined = np.hstack((X_train_tech, train_lstm_features))
#     X_test_combined = np.hstack((X_test_tech, test_lstm_features))

#     # Initialize XGBoost model
#     xgb_model = XGBClassifier()

#     # Train XGBoost on the combined LSTM and technical indicator features
#     xgb_model.fit(X_train_combined, y_train_lstm)

#     # Make predictions on the test set
#     y_pred_xgb = xgb_model.predict(X_test_combined)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test_lstm, y_pred_xgb)
#     print(f"Accuracy: {accuracy}")
#     print(classification_report(y_test_lstm, y_pred_xgb))

#     return accuracy


if __name__ == '__main__':
  main()
