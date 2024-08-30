import pandas as pd
import numpy as np
import pdb
from sklearn.linear_model import LinearRegression


def calculate_slope(series, window=5):
    """
    Calculate the slope of a series using the difference between the end and start of the window.
    """
    slopes = pd.Series(index=series.index, dtype=float)
    for i in range(window - 1, len(series)):
        start_value = series.iloc[i-window+1]
        end_value = series.iloc[i]
        slopes.iloc[i] = end_value - start_value
    return slopes

def crossover_signal_with_slope(data, small_win=7, long_win=14, slope_window=5):
    key_small = f'SMA_{small_win}'
    key_large = f'SMA_{long_win}'

    # Calculate Simple Moving Averages
    data[key_small] = calculate_sma(data['close'], window=small_win)  # Fast MA
    data[key_large] = calculate_sma(data['close'], window=long_win)  # Slow MA

    # Calculate slopes of the SMAs
    data[f'Slope_{small_win}'] = calculate_slope(data[key_small], window=slope_window)
    data[f'Slope_{long_win}'] = calculate_slope(data[key_large], window=slope_window)

    # Generate Buy/Sell signals based on MA crossovers and slopes
    buy_signal = (
        (data[key_small] > data[key_large]) &
        (data[key_small].shift(1) <= data[key_large].shift(1)) &
        (data[f'Slope_{small_win}'] > 0) &  # Fast MA should be trending up
        (data[f'Slope_{long_win}'] > 0)    # Slow MA should be trending up
    )
    
    sell_signal = (
        (data[key_small] < data[key_large]) &
        (data[key_small].shift(1) >= data[key_large].shift(1)) &
        (data[f'Slope_{small_win}'] < 0) &  # Fast MA should be trending down
        (data[f'Slope_{long_win}'] < 0)    # Slow MA should be trending down
    )

    # Initialize Buy_Sell column
    data['Buy_Sell'] = 0
    data.loc[buy_signal, 'Buy_Sell'] = 1
    data.loc[sell_signal, 'Buy_Sell'] = -1

    return data

def calculate_sma(data, window):
    sma = data.rolling(window=window).mean()
    return sma

def crossover_signal(data, small_win=7, long_win=14):

    key_small = f'SMA_{small_win}'
    key_large = f'SMA_{long_win}'

    data[key_small] = calculate_sma(data['close'], window=small_win)  # Fast MA (e.g., 5-day)
    data[key_large] = calculate_sma(data['close'], window=long_win) # Slow MA (e.g., 10-day)

    buy_signal = (data[key_small] > data[key_large]) & (data[key_small].shift(1) <= data[key_large].shift(1))
    sell_signal = (data[key_small] < data[key_large]) & (data[key_small].shift(1) >= data[key_large].shift(1))

    data['Buy_Sell'] = 0  # Initialize the column with zeros
    data.loc[buy_signal, 'Buy_Sell'] = 1
    data.loc[sell_signal, 'Buy_Sell'] = -1

    # count_ones = (data['Buy_Sell'] == 1).sum()
    # count_minus = (data['Buy_Sell'] == -1).sum()

    return data

