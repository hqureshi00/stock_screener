def calculate_bollinger_bands(df, window=20, std_dev=2):
    df['middle_band'] = df['Close'].rolling(window).mean()
    df['std_dev'] = df['Close'].rolling(window).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * std_dev)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * std_dev)
    
def generate_bb_signals(df):
    calculate_bollinger_bands(df)
    df['Signal'] = 0
    df.loc[df['Close'] < df['lower_band'], 'Signal'] = 1  # Buy signal
    df.loc[df['Close'] > df['upper_band'], 'Signal'] = -1  # Sell signal
    return df