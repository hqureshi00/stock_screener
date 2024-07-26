import pandas as pd

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    # Calculate the Short Term Exponential Moving Average
    df['ShortEMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    # Calculate the Long Term Exponential Moving Average
    df['LongEMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    # Calculate the MACD line
    df['MACD'] = df['ShortEMA'] - df['LongEMA']
    # Calculate the Signal line
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    return df

def generate_macd_signals(df):
    signals = []
    position = None  # 1 means buy, -1 means sell

    for i in range(len(df)):
        if df['MACD'][i] > df['Signal'][i]:
            if position != 1:
                signals.append(1)
                position = 1
            else:
                signals.append(0)
        elif df['MACD'][i] < df['Signal'][i]:
            if position != -1:
                signals.append(-1)
                position = -1
            else:
                signals.append(0)
        else:
            signals.append(0)
    
    df['Signal'] = signals
    
    return df

# Example usage
# Assuming you have a DataFrame `df` with a 'Close' column containing the stock prices
# df = pd.read_csv('path_to_your_csv_file.csv')

# Calculate MACD and Signal line
# df_macd = calculate_macd(df)

# # Generate buy and sell signals
# df_signals = generate_signals(df_macd)

# import ace_tools as tools; tools.display_dataframe_to_user(name="MACD Buy and Sell Signals", dataframe=df_signals)