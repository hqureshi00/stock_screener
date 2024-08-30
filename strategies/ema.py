import pandas as pd
import numpy as np

def calculate_ema(data, span):
  return data.ewm(span=span, adjust=False).mean()

def ema_strategy(data, short_window=7, long_window=14):
  signals = pd.DataFrame(index=data.index)
  signals['close'] = data['close']

  key_small = 'EMA_short'
  key_large = 'EMA_long'
  
  # Calculate short-term and long-term EMA
  signals[key_small] = calculate_ema(data['close'], short_window)
  signals[key_large] = calculate_ema(data['close'], long_window)
  
  # Generate buy and sell signals
  # signals['Signal'] = 0
  # signals['Signal'][short_window:] = np.where(signals['EMA_short'][short_window:] > signals['EMA_long'][short_window:], 1, 0)
  # signals['Position'] = signals['Signal'].diff()

  # # Buy signal: 1, Sell signal: -1
  # signals['Buy_Sell'] = np.where(signals['Position'] == 1, 1, np.where(signals['Position'] == -1, -1, 0))

  buy_signal = (data[key_small] > data[key_large]) & (data[key_small].shift(1) <= data[key_large].shift(1))
  sell_signal = (data[key_small] < data[key_large]) & (data[key_small].shift(1) >= data[key_large].shift(1))

  data['Buy_Sell'] = 0  # Initialize the column with zeros
  data.loc[buy_signal, 'Buy_Sell'] = 1
  data.loc[sell_signal, 'Buy_Sell'] = -1
  
  return signals

