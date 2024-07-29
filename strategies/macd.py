import pandas as pd
import pdb
import numpy as np
import pandas_ta as ta


def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def generate_macd_signals(data, short_window=12, long_window=26, signal_window=9):
    signals = pd.DataFrame(index=data.index)
    signals['close'] = data['close']
    
    # Calculate short-term and long-term EMA
    signals['EMA_short'] = calculate_ema(data['close'], short_window)
    signals['EMA_long'] = calculate_ema(data['close'], long_window)
    
    # Calculate MACD and Signal Line
    signals['MACD'] = signals['EMA_short'] - signals['EMA_long']
    signals['Signal_Line'] = calculate_ema(signals['MACD'], signal_window)
    
    # Generate buy and sell signals
    signals['Signal'] = 0
    signals['Signal'][signal_window:] = np.where(
        signals['MACD'][signal_window:] > signals['Signal_Line'][signal_window:], 1, -1
    )
    signals['Position'] = signals['Signal'].diff()

    # Buy signal: 1, Sell signal: -1
    signals['Buy_Sell'] = np.where(
        signals['Position'] == 1, 1,
        np.where(signals['Position'] == -1, -1, 0)
    )

    count_ones = (signals['Signal'] == 1).sum()
    count_minus = (signals['Signal'] == -1).sum()

    
    return signals

# Example usage
# Assuming you have a DataFrame `df` with a 'Close' column containing the stock prices
# df = pd.read_csv('path_to_your_csv_file.csv')

# Calculate MACD and Signal line
# df_macd = calculate_macd(df)

# # Generate buy and sell signals
# df_signals = generate_signals(df_macd)

# import ace_tools as tools; tools.display_dataframe_to_user(name="MACD Buy and Sell Signals", dataframe=df_signals)