import argparse
import sys
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest_ml_model import BacktestTrader
import pandas_ta as pa
import seaborn as sns
import ta  # Technical Analysis library
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from strategies.ema import ema_strategy
from strategies.ma_crossover import crossover_signal
from strategies.rsi import generate_rsi_signals
from utils.fetch_stock_data import fetch_stock_data
from sklearn.feature_selection import RFE

def get_train_test_data(X, y):
    # Calculate the split index
    split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing

    # Split the data
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def test_with_single_indicator_values(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-7)

    # Define the class ranges dynamically
    class_ranges = [
        (-50/100, -0.3/100, 0), 
        (-0.3/100, -0.2/100, 1),  
        (-0.2/100, 0, 2),
        (0, 0.2/100, 3),    
        (0.2/100, 0.3/100, 4),   
        (0.3/100, 50/100, 5),    
      ]
    
      # class_ranges = [
      #   (-50/100, 0.3/100, 0),   
      #   (0.3/100, 50/100, 5),    
      # ]
    # Assign target classes based on these ranges
    df = assign_target_class(df, class_ranges)

    df['Binary_Target'] = (df['Target'] >= 4).astype(int)

    X = df[['percent_bollinger_mavg', 'bollinger_std', 'percent_bollinger_upper', 'percent_bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_mean_5', 'rolling_std_5', 'day_of_week', 'day_of_month',
    'Volatility', 'percent_open', 'percent_close', 'percent_high', 'percent_low', 'postmarket_flag', 'premarket_flag', 'avg_volume_last_20_days', 'large_volume_indicator', 'volume_spike', 'hour_of_day',
    'gain_last_close', 'gain_second_last_close', 'gain_third_last_close', 'gain_fifth_last_close']]

    y = df['Binary_Target']

    print("Class distribution in the dataset:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    print("Class distribution in the training set:")
    print(y_train_res.value_counts())

   # Initialize the XGBoost classifier
    model = XGBClassifier()

   # Train the model
    model.fit(X_train_res, y_train_res)

    # Step 1: Get the predicted probabilities for the positive class
    y_probs = model.predict_proba(X_test)[:, 1]  # Assuming the second column corresponds to class 1

    # Step 2: Define a new threshold
    threshold = 0.4  # This is an example; adjust based on your specific needs

    # Step 3: Generate predictions based on the new threshold
    y_pred = (y_probs >= threshold).astype(int)

    # weights = [0.5, 1, 1.5, 2, 5, 10]  # Example scale_pos_weight values to test
    # for weight in weights:
    #     model = XGBClassifier(scale_pos_weight=weight)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     print(f"Weight: {weight}, Precision: {precision_score(y_test, y_pred)}, Recall: {recall_score(y_test, y_pred)}")
    #     print(classification_report(y_test, y_pred))



    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    # ROC-AUC score
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    # Calculate the percentage by dividing by the sum of each column (predicted class)
    cm_percentage = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :] * 100

    # Display the confusion matrix with percentages
    labels = ["0 ----", "1 ---", "2 --", "3 +", "4 ++", "5 +++"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)

    # Set the titles and labels
    ax.set_title("Confusion Matrix with Percentages")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

    plt.show()

# Get feature importance
    importances = model.feature_importances_

# Create a DataFrame for easy plotting
    feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
    }).sort_values(by='Importance', ascending=False)

# Print feature importance
    print(feature_importances)

#Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importance')
    plt.show()

    print("Backtesting...", end=": ")
    # Backtest

    stock, start_date, end_date, interval = stock_values
    backtest_stock_values = (stock, end_date, "01-03-2024", interval)  # Stock symbol, start date, end date, interval

    backtest_df = get_data_with_indicators(backtest_stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

    backtest_trader = BacktestTrader(model, backtest_df, rsi_thresholds, ma_thresholds, ema_thresholds, stock, interval)
    
    backtest_trader.run_backtest()
    backtest_trader.log_trades()
    print("Backtesting done")

    print(backtest_trader.accumulated_profit)

    #breakpoint()
    return accuracy_score(y_test, y_pred)

  
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
    df['Volatility'] = df['Volatility'].fillna(0)
    
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
    df['Normalized_Volume'] = df['Normalized_Volume'].fillna(1)    
    return df

def read_data(stock_name, interval, start_date, end_date):
  data = fetch_stock_data(stock_name, interval, start_date, end_date)

  return data

def get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows):

  stock, start_date, end_date, interval = stock_values
  df = fetch_stock_data(stock, interval, start_date, end_date)
  #breakpoint()
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

  # Filter rows where the time is between 08:00:00 and 16:00:00
  df = df[(df['timestamp'].dt.time >= pd.to_datetime('07:00:00').time()) & #market opens at 9:30 but experimenting with a little bit of pre market here
        (df['timestamp'].dt.time <= pd.to_datetime('18:00:00').time())]
  
  df['premarket_flag'] = (df['timestamp'].dt.time < pd.to_datetime('09:30:00').time())
  df['postmarket_flag'] = (df['timestamp'].dt.time > pd.to_datetime('16:00:00').time())

  # Ensure the timestamp is in datetime format
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Extract time of day from the timestamp
  df['time_of_day'] = df['timestamp'].dt.time
  df.reset_index(drop=True, inplace=True)

  # Calculate the average volume for each time of day over the last 5 trading days
  df['avg_volume_last_20_days'] = df.groupby('time_of_day')['volume'].rolling(window=20).mean().reset_index(level=0, drop=True)

  df['volume_spike'] = df['volume'] / df['avg_volume_last_20_days']

  df['hour_of_day'] = df['timestamp'].dt.hour

  volume_threshold = 2
  df['large_volume_indicator'] =  df['volume_spike'] > volume_threshold

      # Price action direction
  # Calculate the gain compared to the last closing value
  df['gain_last_close'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

  # Calculate the gain compared to the second-last closing value
  df['gain_second_last_close'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
  df['gain_third_last_close'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(2)
  df['gain_fifth_last_close'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(2)


    # Handling potential NaN values that arise from shifting
  df['gain_last_close'] = df['gain_last_close'].fillna(0)  # No gain for the first row
  df['gain_second_last_close'] = df['gain_second_last_close'].fillna(0)  # No gain for the first two rows


  df.reset_index(drop=True, inplace=True)


  #df.to_csv('training_data.csv', index=True)


# breakpoint()

  print(df.describe())
  print(df.dtypes)

  return df

def assign_target_class(df, class_ranges):
    """
    Assign target class based on percentage changes.

    Parameters:
    df (pd.DataFrame): The data frame containing stock data.
    class_ranges (list of tuples): A list of tuples where each tuple defines a class.
                                   Format: (min_threshold, max_threshold, class_label)

    Example:
    class_ranges = [
        (-1.0, -0.004, 0),   # Class 0: Decrease by more than 0.4%
        (-0.004, 0.004, 1),  # Class 1: No significant change (within 0.4% up or down)
        (0.004, 1.0, 2)      # Class 2: Increase by more than 0.4%
    ]
    """
    
    # Default to 'unclassified' in case no class fits (this is optional and can be removed)
    df['Target'] = 3
    
    # Iterate over the class ranges and assign classes dynamically
    for min_threshold, max_threshold, class_label in class_ranges:
        df['Target'] = np.where(
            ((df['Future_Close'] - df['close']) / df['close'] >= min_threshold) &
            ((df['Future_Close'] - df['close']) / df['close'] < max_threshold),
            class_label,
            df['Target']
        )
    
    return df

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

if __name__ == '__main__':
  main()
