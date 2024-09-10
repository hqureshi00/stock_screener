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

    # Buy signal: short EMA crosses above long EMA
    buy_signal = (signals[key_small] > signals[key_large]) & (signals[key_small].shift(1) <= signals[key_large].shift(1))
    # Sell signal: short EMA crosses below long EMA
    sell_signal = (signals[key_small] < signals[key_large]) & (signals[key_small].shift(1) >= signals[key_large].shift(1))

    signals['Buy_Sell'] = 0  # Initialize the column with zeros
    signals.loc[buy_signal, 'Buy_Sell'] = 1
    signals.loc[sell_signal, 'Buy_Sell'] = -1

    return signals

