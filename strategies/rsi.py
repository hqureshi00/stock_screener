import pandas as pd
import pdb

def calculate_rsi(df, window=14):
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi

    return df

def generate_rsi_signals(df, buy_threshold=30, sell_threshold=70):
    # Calculate RSI
    signals = calculate_rsi(df, window=14)
    
    # Initialize the Signal column
    signals['Signal'] = 0
    
    # Generate Buy Signal (1) and Sell Signal (-1)
    signals.loc[signals['RSI'] <= buy_threshold, 'Signal'] = 1
    signals.loc[signals['RSI'] >= sell_threshold, 'Signal'] = -1
    
    # Initialize Buy_Sell column
    signals['Buy_Sell'] = 0
    
    # Check the previous value to determine Buy or Sell action
    for i in range(1, len(signals)):
        # If the current signal is Buy (1) and the previous signal was not Buy (0 or -1)
        if signals.loc[i, 'Signal'] == 1 and signals.loc[i - 1, 'Signal'] != 1:
            signals.loc[i, 'Buy_Sell'] = 1
        # If the current signal is Sell (-1) and the previous signal was not Sell (0 or 1)
        elif signals.loc[i, 'Signal'] == -1 and signals.loc[i - 1, 'Signal'] != -1:
            signals.loc[i, 'Buy_Sell'] = -1
    
    return signals