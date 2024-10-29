from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ta  # Technical Analysis library
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from strategies.ema import ema_strategy
from strategies.ma_crossover import crossover_signal
from strategies.rsi import generate_rsi_signals
from utils.fetch_stock_data import fetch_stock_data

def get_data_with_indicators(stock_values, rsi_thresholds, ma_windows, ema_windows):

  stock, start_date, end_date, interval = stock_values

  d_start_date = datetime.strptime(start_date, "%d-%m-%Y")  # Adjust format as necessary

  # Subtract 30 days - so we cn calculate indicators using past data windows
  adjusted_start_date = d_start_date - timedelta(days=55)
  adjusted_start_date_str = adjusted_start_date.strftime("%d-%m-%Y")  # Format as needed

  df = fetch_stock_data(stock, interval, adjusted_start_date_str, end_date)
  signals = generate_rsi_signals(df, buy_threshold=rsi_thresholds[0], sell_threshold=rsi_thresholds[1])
  df['RSI_signal'] = signals['Buy_Sell']
  signals = crossover_signal(df, small_win=ma_windows[0], long_win=ma_windows[1])
  df['MA_signal'] = signals['Buy_Sell']
  signals = ema_strategy(df, short_window=ema_windows[0], long_window=ema_windows[1])
  df['EMA_signal'] = signals['Buy_Sell']
  signals = calculate_volatility(df)
  df['Volatility'] = signals['Volatility']
  signals = calculate_normalized_volume(df)
  df['Normalized_Volume'] = signals['Normalized_Volume']

  df['bollinger_mavg'] = df['close'].rolling(window=20).mean()
  df['bollinger_std'] = df['close'].rolling(window=20).std()
  df['bollinger_upper'] = df['bollinger_mavg'] + (df['bollinger_std'] * 2)
  df['bollinger_lower'] = df['bollinger_mavg'] - (df['bollinger_std'] * 2)

  df['macd'] = ta.trend.macd(df['close'])
  df['macd_signal'] = ta.trend.macd_signal(df['close'])
  df['macd_diff'] = df['macd'] - df['macd_signal']

  df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

  df['rolling_volume_mean_20'] = df['volume'].rolling(window=20).mean()
  df['day_of_week'] = df['timestamp'].dt.dayofweek
  df['day_of_month'] = df['timestamp'].dt.day
  df['isFridayClosing'] = df['timestamp'].apply(lambda x: 1 if x.weekday() == 4 and x.time() == pd.Timestamp('16:00:00').time() else 0)
  
  df['Prev_Friday_Close'] = df.apply(lambda row: row['close'] if row['isFridayClosing'] == 1 else None, axis=1)
  df['Prev_Friday_Close'] = df['Prev_Friday_Close'].ffill()

  # Calculate percentage changes from the last Friday's close
  df['percent_open'] = ((df['open'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_close'] = ((df['close'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_high'] = ((df['high'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_low'] = ((df['low'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100

  # Now, calculate Bollinger Bands as percentages of the last Friday's close
  df['percent_bollinger_mavg'] = ((df['bollinger_mavg'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_bollinger_upper'] = ((df['bollinger_upper'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100
  df['percent_bollinger_lower'] = ((df['bollinger_lower'] - df['Prev_Friday_Close']) / df['Prev_Friday_Close']) * 100

  df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']
  df['bollinger_width_delta'] = df['bollinger_width'].diff()
  df['bollinger_width_delta_3'] = df['bollinger_width'].diff(periods=3)
  
  df['rolling_mean_20'] = df['percent_close'].rolling(window=20).mean()
  df['rolling_std_20'] = df['percent_close'].rolling(window=20).std()
  df['rolling_mean_5'] = df['percent_close'].rolling(window=5).mean()
  df['rolling_std_5'] = df['percent_close'].rolling(window=5).std()

  # Filter rows where the time is between 08:00:00 and 16:00:00
  df = df[(df['timestamp'].dt.time >= pd.to_datetime('08:00:00').time()) & #market opens at 9:30 but experimenting with a little bit of pre market here
        (df['timestamp'].dt.time <= pd.to_datetime('17:00:00').time())]
  
  df['premarket_flag'] = (df['timestamp'].dt.time < pd.to_datetime('09:30:00').time())
  df['postmarket_flag'] = (df['timestamp'].dt.time > pd.to_datetime('16:00:00').time())

  # Ensure the timestamp is in datetime format
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Extract time of day from the timestamp
  df['time_of_day'] = df['timestamp'].dt.time
  df.reset_index(drop=True, inplace=True)

  # Calculate the average volume for each time of day over the last 20 trading days
  df['avg_volume_last_20_days'] = df.groupby('time_of_day')['volume'].rolling(window=20).mean().reset_index(level=0, drop=True)

  df['volume_spike'] = df['volume'] / df['avg_volume_last_20_days']
  df['hour_of_day'] = df['timestamp'].dt.hour
  df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
  df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
  df['part_of_day'] = df['hour_of_day'].apply(assign_time_of_day)
  df['is_peak_hour'] = df['hour_of_day'].apply(lambda x: 1 if x in range(8, 11) else 0)
  scaler = MinMaxScaler()
  df['hour_of_day_scaled'] = scaler.fit_transform(df[['hour_of_day']])
  df['hour_x_volume'] = df['hour_of_day'] * df['volume']

  volume_threshold = 2
  df['large_volume_indicator'] =  df['volume_spike'] > volume_threshold

  # Calculate the gain compared to the last closing value
  df['gain_last_close'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
  df['gain_second_last_close'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
  df['gain_third_last_close'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(2)
  df['gain_fifth_last_close'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(2)

  #drop 40 days in start so all data is good. 
  
  df['date'] = df['timestamp'].dt.date
  first_40_dates = df['date'].unique()[:50]
  df_filtered = df[~df['date'].isin(first_40_dates)]
  df_filtered = df_filtered.drop(columns=['date'])
  df=df_filtered

  df.reset_index(drop=True, inplace=True)

  validate_data(df, start_date, end_date);

  return df

def validate_data(df, start_date, end_date):

    #Number of Missing Values for each column
    missing_data = df.isnull().sum()
    print(missing_data)

    #check for missing candles
    daily_counts = df.groupby(df['timestamp'].dt.date).size()
    count_of_days = daily_counts.value_counts().sort_index()
    print(count_of_days)

    start_date = datetime.strptime(start_date, '%d-%m-%Y')
    end_date = datetime.strptime(end_date, '%d-%m-%Y')

    breakpoint();

    return

def calculate_volatility(stock_values, window=14):
    """
    Calculate the rolling volatility (standard deviation of returns) for a given window.
    
    Args:
        stock_values (DataFrame): A DataFrame with at least the 'close' column.
        window (int): The rolling window size for volatility calculation.
    
    Returns:
        DataFrame: The stock data with an additional column 'Volatility'.
    """
    df = stock_values.copy()
    
    # Calculate the returns
    df['Returns'] = df['close'].pct_change()
    
    # Calculate rolling volatility (standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    
    # Fill NaN values with 0 for the initial rows where rolling window can't be applied
    df['Volatility'] = df['Volatility'].fillna(0)
    
    return df

def calculate_normalized_volume(stock_values, window=20):
    """
    Calculate normalized volume, which is the volume relative to its moving average.
    
    Args:
        stock_values (DataFrame): A DataFrame with at least the 'volume' column.
        window (int): The rolling window size for normalizing the volume.
    
    Returns:
        DataFrame: The stock data with an additional column 'Normalized_Volume'.
    """
    df = stock_values.copy()
    
    # Calculate the rolling mean of volume
    df['Volume_MA'] = df['volume'].rolling(window=window).mean()
    
    # Calculate normalized volume (current volume / moving average of volume)
    df['Normalized_Volume'] = df['volume'] / df['Volume_MA']
    
    # Fill NaN values with 1 for the initial rows where rolling window can't be applied
    df['Normalized_Volume'] = df['Normalized_Volume'].fillna(1)    
    return df

def assign_time_of_day(hour):
    if 6 <= hour < 12:
        return 0
    elif 12 <= hour < 17:
        return 1
    elif 17 <= hour < 21:
        return 2
    else:
        return 3
