import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


class BacktestTrader:
    def __init__(self, model, df, rsi_thresholds, ma_thresholds, ema_thresholds, stock_symbol, interval):
        self.model = model
        self.df = df
        self.rsi_thresholds = rsi_thresholds
        self.ma_thresholds = ma_thresholds
        self.ema_thresholds = ema_thresholds
        self.completed_trades = []
        self.open_positions = []  # To track active trades
        self.accumulated_profit = 0
        self.trade_log_file = 'trades.csv'  # File to log trades
        self.stock_symbol = stock_symbol
        self.interval = interval
        self.profit_over_time = []  # List to track profit at each trade for charting
        self.initial_capital = 10000;

    def place_trade(self, row, current_index, prediction):
        """Place a trade if conditions are met."""
        if row['timestamp']=="2024-05-28 07:05:00":
            print("here")
            #breakpoint()

        trade_entry = {
            'entry_price': row['close'],
            'entry_date': row['timestamp'],
            'exit_index': current_index + 10,
            'predicted_class' : prediction,
            'num_stocks' : int((self.initial_capital + self.accumulated_profit) / 500)
        }
        self.open_positions.append(trade_entry)

    def close_trades(self, row, current_index):
        """Close any open trades after 5 candles and calculate profit."""
        trades_to_close = []
        
        for trade in self.open_positions:
            if trade['exit_index'] == current_index:  
                exit_price = row['close']
                num_stocks = float(trade['num_stocks'])
                #breakpoint()
                profit = (exit_price - trade['entry_price']) * num_stocks
                profit_percentage = (profit / (trade['entry_price']*num_stocks)) * 100
                self.accumulated_profit += (profit - (0.005 * num_stocks))  # Assuming a transaction cost of 1
                
                # Log trade details
                #self._log_trade(trade['entry_date'], trade['entry_price'], row['timestamp'], exit_price, profit, profit_percentage)
                
                #print(f"Trade closed at index {current_index}. Total Profit: {self.accumulated_profit:.2f}")
                    # Append the completed trade details
                trade_record = {
                    'predicted_class' : trade['predicted_class'],
                    'start_time': trade['entry_date'],
                    'end_time': row['timestamp'],
                    'entry_price': trade['entry_price'],
                    'num_stocks': trade['num_stocks'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'profit_percent': profit_percentage,
                    'accumulated_profit': self.accumulated_profit

                }
                self.completed_trades.append(trade_record)

               # Append profit for charting
                self.profit_over_time.append((row['timestamp'], self.accumulated_profit))
                trades_to_close.append(trade)  # Collects trades to close

        # Remove closed trades
        self.open_positions = [trade for trade in self.open_positions if trade not in trades_to_close]

    def log_trades(self):
        """Log all completed trades into a CSV file."""
        trades_df = pd.DataFrame(self.completed_trades)

        # Generate a unique filename based on symbol, interval, and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.stock_symbol}_{self.interval}_trades_{timestamp}.csv"

        # Save the trades dataframe to a CSV file
        trades_df.to_csv(filename, index=False)
        print(f"Trades logged to {filename}")

        # Plotting accumulated profit and stock price over time
        self.plot_accumulated_profit()


    def extendTradeAtExpiry(self, current_index):
        """Extend the exit index of the trade that is expiring at the current index by 5 more candles.
        Returns True if a trade was extended, False otherwise."""
        for trade in self.open_positions:
            if trade['exit_index'] == current_index:
                trade['exit_index'] += 10  # Extend exit by 10 more candles
                #print(f"Trade at index {current_index} extended to {trade['exit_index']}")
                return True  # Return True to indicate trade was extended
        return False  # Return False if no trade was extended

    def run_backtest(self):
        last_date_str = ""
        """Run the backtest on the dataset. Index is decreasing"""
        self.df = self.df.iloc[40:].reset_index(drop=True)
        #breakpoint()

        for i in range(len(self.df)):
            row = self.df.iloc[i]  # Access the row by its sequential index
            # Perform operations with row
            # Format the current date
            date_str = row['timestamp'].strftime('%B, %Y')  # Adjust based on your timestamp column
        
            # Print the date only if it's different from the last printed date
            if date_str != last_date_str:
                print(f"Processing: {date_str}")
                last_date_str = date_str  # Update the last printed date

            features = row[['percent_bollinger_mavg', 'bollinger_std', 'percent_bollinger_upper', 'percent_bollinger_lower', 'bollinger_width_delta','bollinger_width_delta_3', 'macd', 'macd_signal', 'macd_diff',
                'atr', 'rolling_mean_20', 'rolling_std_20', 'rolling_mean_5', 'rolling_std_5', 'day_of_week', 'day_of_month',
                 'MA_signal', 'EMA_signal', 'Volatility', 'volume_spike', 'percent_open', 'percent_close', 'percent_high', 'percent_low']].values.reshape(1, -1)
            prediction = self.model.predict(features)
            # if prediction > 3:
            #    print(f"Prediction on {i}: {prediction}")

            #breakpoint()
            if prediction >= 5:  # Highest class (buy signal)
                if not self.extendTradeAtExpiry(i):  # Check for trade extension
                    self.place_trade(row, i, prediction)  # Place a new trade
            self.close_trades(row, i)  # Close trades regardless of prediction

    def plot_accumulated_profit(self):
        """Plot accumulated profit and stock price over time."""
        # Extract profit and timestamps
        profit_df = pd.DataFrame(self.profit_over_time, columns=['timestamp', 'accumulated_profit'])
        #breakpoint();
        # Merge profit_df with stock price data to get the stock price at those timestamps
        merged_df = pd.merge(profit_df, self.df[['timestamp', 'close']], on='timestamp', how='left')

        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot accumulated profit on the left y-axis
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Accumulated Profit', color='tab:blue')
        ax1.plot(merged_df['timestamp'], merged_df['accumulated_profit'], label='Accumulated Profit', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Plot stock price on the right y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Stock Price', color='tab:orange')
        ax2.plot(merged_df['timestamp'], merged_df['close'], label='Stock Price', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Title and layout
        plt.title(f'Accumulated Profit and Stock Price Over Time ({self.stock_symbol})')
        fig.tight_layout()

        # Show plot
        plt.show()