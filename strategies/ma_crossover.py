import pandas as pd
import numpy as np


def moving_average_crossover_signals(data, short_window=7, long_window=14):
   
    # Create short-term and long-term SMAs
    data[f'SMA_{short_window}'] = data['close'].rolling(window=short_window, min_periods=1).mean()
    data[f'SMA_{long_window}'] = data['close'].rolling(window=long_window, min_periods=1).mean()

    # Initialize the Signal column
    data['Signal'] = 0  # 0 means no position, 1 means buy, -1 means sell

    # Generate Buy (1) and Sell (-1) signals
    data['Signal'][short_window:] = np.where(
        data[f'SMA_{short_window}'][short_window:] > data[f'SMA_{long_window}'][short_window:], 1, -1
    )

    # Calculate position by taking the difference between consecutive signals
    data['Buy_Sell'] = data['Signal'].diff()

    return data

def crossover_signal_with_slope(data, small_win=3, long_win=5):

    key_small = f'SMA_{small_win}'
    key_large = f'SMA_{long_win}'

    # Calculate the moving averages
    data[key_small] = data['close'].rolling(window=int(small_win)).mean()
    data[key_large] = data['close'].rolling(window=int(long_win)).mean()

    # Calculate the slope of the smaller moving average
    data['Slope'] = data[key_small].diff()

    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    # Initialize Signals column in the original data
    data['Signals'] = 0

    # Generate signals based on crossover and slope condition
    # Point to Discuss
    # for i in range(int(small_win), len(data)):
    #     if (data[key_small].iloc[i] > data[key_large].iloc[i]) and data['Slope'].iloc[i] > 0:
    #         data['Signals'].iloc[i] = 1
    #     elif (data[key_small].iloc[i] < data[key_large].iloc[i]) and data['Slope'].iloc[i] < 0:
    #         data['Signals'].iloc[i] = -1
    #     else:
    #         data['Signals'].iloc[i] = 0

    # for i in range(int(small_win), len(data)):
    #     if data[key_small].iloc[i] > data[key_large].iloc[i]:
    #         data['Signals'].iloc[i] = 1
    #     elif data[key_small].iloc[i] < data[key_large].iloc[i]:
    #         data['Signals'].iloc[i] = -1
    #     else:
    #         data['Signals'].iloc[i] = 0

    # Calculate the signal changes (1 for buy, -1 for sell)
    # signals['Signal'] = data['Signals'].diff()

    return signals

#TODO: Why does slope greater than zero work?

def crossover_signal(data, small_win=7, long_win=14):

    key_small = f'SMA_{small_win}'
    key_large = f'SMA_{long_win}'
  
    data[key_small] = data['close'].rolling(window=int(small_win)).mean()
    data[key_large] = data['close'].rolling(window=int(long_win)).mean()

    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    
    data['Signals'] = 0

    data['Signals'][int(small_win):] = np.where(data[key_small][int(small_win):] > data[key_large][int(small_win):], 1, 0)

    signals['Signal'] = data['Signals'].diff()

    return signals