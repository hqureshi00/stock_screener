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

    signals = calculate_rsi(df, window=14)
    
    # Initialize the Buy/Sell column
    signals['Buy_Sell'] = 0
    
    # Generate Buy Signal (1) and Sell Signal (-1)
    signals.loc[signals['RSI'] < buy_threshold, 'Buy_Sell'] = 1
    signals.loc[signals['RSI'] > sell_threshold, 'Buy_Sell'] = -1
    
    return signals

# Example usage
# Assuming you have a DataFrame `df` with a 'Close' column containing the stock prices
# df = pd.read_csv('path_to_your_csv_file.csv')

# Calculate RSI
# df_rsi = calculate_rsi(df)

# # Generate RSI buy and sell signals
# df_signals = generate_rsi_signals(df_rsi)

# import ace_tools as tools; tools.display_dataframe_to_user(name="RSI Buy and Sell Signals", dataframe=df_signals)