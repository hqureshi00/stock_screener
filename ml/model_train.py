import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import BacktestTrader
import pandas_ta as pa
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from constants import CLASS_RANGES, MODEL_THRESHOLD
from data_prep import get_data_with_indicators

def trainXGBoost(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds):
    df = get_data_with_indicators(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)
    df['Future_Close'] = df['close'].shift(-7)

    # Assign target classes based on these ranges
    df = assign_target_class(df, CLASS_RANGES)

    df['Binary_Target'] = (df['Target'] >= 4).astype(int)
    df['Actual_gain'] = df['Future_Close'] - df['close']
    df.dropna(subset=['Actual_gain'], inplace=True)


    X = df[['percent_bollinger_mavg', 'bollinger_std', 'percent_bollinger_upper', 'percent_bollinger_lower', 'macd', 'macd_signal', 'macd_diff',
    'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_mean_5', 'rolling_std_5', 'day_of_week', 'day_of_month',
    'Volatility', 'percent_open', 'percent_close', 'percent_high', 'percent_low', 'postmarket_flag', 'premarket_flag', 'avg_volume_last_20_days', 'large_volume_indicator', 'volume_spike', 'hour_of_day',
    'gain_last_close', 'gain_second_last_close', 'gain_third_last_close', 'gain_fifth_last_close', 'hour_sin', 'hour_cos', 'part_of_day', 'is_peak_hour', 'hour_of_day_scaled', 'hour_x_volume']]

    y = df['Binary_Target']

    # Combine X and y into a new DataFrame for saving
    df_for_saving = X.copy()  # Make a copy of X to avoid modifying the original data
    df_for_saving['Binary_Target'] = y  # Add the target column

    # Save the DataFrame to a CSV file
    df_for_saving.to_csv('full_dataset.csv', index=False)  # Set index=False to avoid saving the row index

    print("Class distribution in the dataset:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    print("Class distribution in the training set:")
    print(y_train_res.value_counts())

   # Initialize the XGBoost classifier
    model = XGBClassifier(n_jobs=-8)

   # Train the model
    model.fit(X_train_res, y_train_res)

    # Step 1: Get the predicted probabilities for the positive class
    y_probs = model.predict_proba(X_test)[:, 1]  # Assuming the second column corresponds to class 1

    # Step 2: Define a new threshold
    threshold = MODEL_THRESHOLD # This is an example; adjust based on your specific needs

    # Step 3: Generate predictions based on the new threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    ####COST CALCULATION####

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
    backtest_stock_values = (stock, "01-08-2024", "12-12-2024", interval)  # Stock symbol, start date, end date, interval

    backtest_df = get_data_with_indicators(backtest_stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

    backtest_trader = BacktestTrader(model, backtest_df, rsi_thresholds, ma_thresholds, ema_thresholds, stock, interval)
    
    backtest_trader.run_backtest()
    backtest_trader.log_trades()
    print("Backtesting done")

    print(backtest_trader.accumulated_profit)

    #breakpoint()
    return accuracy_score(y_test, y_pred)

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
    accuracy = trainXGBoost(stock_values, rsi_thresholds, ma_thresholds, ema_thresholds)

if __name__ == '__main__':
  main()
