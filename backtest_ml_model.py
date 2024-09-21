import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestTrader:
    def __init__(self, model, df, rsi_thresholds, ma_thresholds, ema_thresholds):
        self.model = model
        self.df = df
        self.rsi_thresholds = rsi_thresholds
        self.ma_thresholds = ma_thresholds
        self.ema_thresholds = ema_thresholds
        self.trades = []
        self.open_positions = []  # To track active trades
        self.total_profit = 0

    def place_trade(self, row, current_index):
        """Place a trade if conditions are met."""
     
        trade_entry = {
            'entry_price': row['close'],
            'exit_index': current_index + 5,
        }
        self.open_positions.append(trade_entry)

    def close_trades(self, row, current_index):
        """Close any open trades after 5 candles and calculate profit."""
        trades_to_close = []
        
        for trade in self.open_positions:
            if trade['exit_index'] == current_index:  
                exit_price = row['close']
                profit = exit_price - trade['entry_price']
                self.total_profit += profit - 1  # Assuming a transaction cost of 1
                print(f"Trade closed at index {current_index}. Total Profit: {self.total_profit:.2f}")
                self.trades.append(profit)
                trades_to_close.append(trade)  # Collect trades to close

        # Remove closed trades
        self.open_positions = [trade for trade in self.open_positions if trade not in trades_to_close]

    def extendTradeAtExpiry(self, current_index):
        """Extend the exit index of the trade that is expiring at the current index by 5 more candles.
        Returns True if a trade was extended, False otherwise."""
        for trade in self.open_positions:
            if trade['exit_index'] == current_index:
                trade['exit_index'] += 5  # Extend exit by 5 more candles
                print(f"Trade at index {current_index} extended to {trade['exit_index']}")
                return True  # Return True to indicate trade was extended
        return False  # Return False if no trade was extended


    def run_backtest(self):
        last_date_str = ""
        """Run the backtest on the dataset."""
        for i, row in self.df.iterrows():
            # Format the current date
            date_str = row['timestamp'].strftime('%d %B, %Y')  # Adjust based on your timestamp column
        
            # Print the date only if it's different from the last printed date
            if date_str != last_date_str:
                print(f"Processing day: {date_str}")
                last_date_str = date_str  # Update the last printed date

            features = row[['bollinger_mavg', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 
                            'macd', 'macd_signal', 'macd_diff', 'atr', 
                            'rolling_mean_20', 'rolling_std_20', 'rolling_volume_mean_20', 
                            'day_of_week', 'day_of_month', 'MA_signal', 'EMA_signal', 
                            'Volatility', 'Normalized_Volume', 'open', 'close', 'high', 'low']].values.reshape(1, -1)
            prediction = self.model.predict(features)

            if prediction == 5:  # Highest class (buy signal)
                if not self.extendTradeAtExpiry(i):  # Check for trade extension
                    self.place_trade(row, i)  # Place a new trade
            self.close_trades(row, i)  # Close trades regardless of prediction
        