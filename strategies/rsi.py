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

def generate_rsi_signals_1(df, buy_threshold=30, sell_threshold=70):

    signals = calculate_rsi(df, window=14)
    
    # Initialize the Buy/Sell column
    signals['Buy_Sell'] = 0
    
    # Generate Buy Signal (1) and Sell Signal (-1)
    signals.loc[signals['RSI'] < buy_threshold, 'Buy_Sell'] = 1
    signals.loc[signals['RSI'] > sell_threshold, 'Buy_Sell'] = -1
    
    return signals

def generate_rsi_signals(df, buy_threshold=30, sell_threshold=70):
    signals = calculate_rsi(df, window=14)
    
    # Initialize the Signal column
    signals['Signal'] = 0
    
    # Generate Buy Signal (1) and Sell Signal (-1)
    signals.loc[signals['RSI'] <= buy_threshold, 'Signal'] = 1
    signals.loc[signals['RSI'] >= sell_threshold, 'Signal'] = -1
    
    # Calculate the Position as the difference of consecutive signals
    signals['Position'] = signals['Signal'].diff()
    
    # Translate Position into Buy_Sell actions
    signals['Buy_Sell'] = 0
    signals.loc[signals['Position'] == 1, 'Buy_Sell'] = 1  # Buy signal
    signals.loc[signals['Position'] == -1, 'Buy_Sell'] = -1  # Sell signal
    
    return signals

def generate_rsi_signals_old(df, buy_threshold=30, sell_threshold=70):
    signals = calculate_rsi(df, window=14)
    
    # Initialize the Buy/Sell column
    signals['Buy_Sell'] = 0
    
    # Loop through the DataFrame to set signals only when there is a transition
    for i in range(1, len(signals)):
        if signals['RSI'].iloc[i] < buy_threshold and signals['Buy_Sell'].iloc[i-1] != 1:
            signals.at[signals.index[i], 'Buy_Sell'] = 1
        elif signals['RSI'].iloc[i] > sell_threshold and signals['Buy_Sell'].iloc[i-1] != -1:
            signals.at[signals.index[i], 'Buy_Sell'] = -1
        else:
            signals.at[signals.index[i], 'Buy_Sell'] = 0

    return signals

# Example usage
# Assuming you have a DataFrame `df` with a 'Close' column containing the stock prices
# df = pd.read_csv('path_to_your_csv_file.csv')

# Calculate RSI
# df_rsi = calculate_rsi(df)

# # Generate RSI buy and sell signals
# df_signals = generate_rsi_signals(df_rsi)

# import ace_tools as tools; tools.display_dataframe_to_user(name="RSI Buy and Sell Signals", dataframe=df_signals)