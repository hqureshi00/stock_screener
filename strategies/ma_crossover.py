import pandas as pd
import numpy as np

def crossover_signal_with_slope(data, small_win, long_win):
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
    for i in range(int(small_win), len(data)):
        if data[key_small].iloc[i] > data[key_large].iloc[i] and data['Slope'].iloc[i] > 0:
            data['Signals'].iloc[i] = 1
        elif data[key_small].iloc[i] < data[key_large].iloc[i] and data['Slope'].iloc[i] < 0:
            data['Signals'].iloc[i] = -1
        else:
            data['Signals'].iloc[i] = 0

    # Calculate the signal changes (1 for buy, -1 for sell)
    signals['Signal'] = data['Signals'].diff()

    return signals

def crossover_signal(data, small_win, long_win):
    key_small = f'SMA_{small_win}'
    key_large = f'SMA_{long_win}'
  
    data[key_small] = data['close'].rolling(window=int(small_win)).mean()
    data[key_large] = data['close'].rolling(window=int(long_win)).mean()

    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    
    # pdb.set_trace()
    data['Signals'] = 0

    data['Signals'][int(small_win):] = np.where(data[key_small][int(small_win):] > data[key_large][int(small_win):], 1, 0)

    signals['Signal'] = data['Signals'].diff()

    return signals