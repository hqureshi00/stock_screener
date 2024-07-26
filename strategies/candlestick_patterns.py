def hammer(df):
    df['Hammer'] = ((df['Close'] > df['Open']) &
                    ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) &
                    ((df['Close'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6) &
                    ((df['Open'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6)).astype(int)

def shooting_star(df):
    df['Shooting_Star'] = ((df['Open'] > df['Close']) &
                           ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) &
                           ((df['High'] - df['Close']) / (.001 + df['High'] - df['Low']) > 0.6) &
                           ((df['High'] - df['Open']) / (.001 + df['High'] - df['Low']) > 0.6)).astype(int)

def engulfing(df):
    df['Bullish_Engulfing'] = ((df['Open'].shift(1) > df['Close'].shift(1)) &
                               (df['Close'] > df['Open']) &
                               (df['Open'] < df['Close'].shift(1)) &
                               (df['Close'] > df['Open'].shift(1))).astype(int)

    df['Bearish_Engulfing'] = ((df['Open'].shift(1) < df['Close'].shift(1)) &
                               (df['Close'] < df['Open']) &
                               (df['Open'] > df['Close'].shift(1)) &
                               (df['Close'] < df['Open'].shift(1))).astype(int)
    

def generate_candlestick_signals(df):
    hammer(df)
    shooting_star(df)
    engulfing(df)
    df['Signal'] = 0
    df.loc[df['Hammer'] | df['Bullish_Engulfing'], 'Signal'] = 1
    df.loc[df['Shooting_Star'] | df['Bearish_Engulfing'], 'Signal'] = -1

    return df
