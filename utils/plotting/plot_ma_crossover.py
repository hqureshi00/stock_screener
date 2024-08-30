import pandas as pd
import matplotlib.pyplot as plt


def plot_moving_average_crossover(stock_data, small=7, large=14):

  key_small = f'SMA_{small}'
  key_large = f'SMA_{large}'

  label1 = f'{small}-day SMA'
  label2 = f'{large}-day SMA'
  plt.figure(figsize=(14, 7))
  df = pd.DataFrame(stock_data)
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  df.set_index('timestamp', inplace=True)

  df.dropna(subset=['close', key_small, key_large], inplace=True)
    
  df['index'] = range(len(df))

  plt.plot(df['index'], df['close'], label='Close Price')
  plt.plot(df['index'], df[key_small], label=label1)
  plt.plot(df['index'], df[key_large], label=label2)

  plt.plot(df.loc[df['Signal'] == 1.0]['index'], 
             df[key_small][df['Signal'] == 1.0], 
             '^', markersize=10, color='g', label='Buy Signal')

  for i in df.loc[df['Signal'] == 1.0]['index']:
        plt.text(i, df['close'][df['index'] == i].values[0], f'{df["close"][df["index"] == i].values[0]:.2f}', fontsize=9, ha='center', color='g', va='bottom')

  plt.plot(df.loc[df['Signal'] == -1.0]['index'], 
             df[key_small][df['Signal'] == -1.0], 
             'v', markersize=10, color='r', label='Sell Signal')

  for i in df.loc[df['Signal'] == -1.0]['index']:
        plt.text(i, df['close'][df['index'] == i].values[0], f'{df["close"][df["index"] == i].values[0]:.2f}', fontsize=9, ha='center', color='r', va='bottom')

  plt.title('Stock Price with Buy and Sell Signals')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.grid(True)

  n = max(1, len(df) // 10)  
  plt.xticks(ticks=df['index'][::n], labels=[date.strftime('%Y-%m-%d') for date in df.index][::n], rotation=45)

  plt.tight_layout()
  plt.tight_layout()
  plt.show()

def plot_moving_average_crossover_test(stock_data, small=7, long=14):
  
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

  key_small = f'SMA_{small}'
  key_large = f'SMA_{long}'

  label1 = f'{small}-day SMA'
  label2 = f'{long}-day SMA'

  df = pd.DataFrame(stock_data)
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  df.set_index('timestamp', inplace=True)

  df.dropna(subset=['close', key_small, key_large], inplace=True)
    
  df['index'] = range(len(df))

# Plot Close Prices
  ax1.plot(df.index, df['close'], color='blue', label='Close Price')
#   plt.plot(df['index'], df['close'], label='Close Price')
  ax2.plot(df['index'], df[key_small], label=label1)
  ax2.plot(df['index'], df[key_large], label=label2)

  ax1.plot(df.loc[df['Signal'] == 1.0]['index'], 
             df[key_small][df['Signal'] == 1.0], 
             '^', markersize=10, color='g', label='Buy Signal')

  for i in df.loc[df['Signal'] == 1.0]['index']:
        ax1.text(i, df['close'][df['index'] == i].values[0], f'{df["close"][df["index"] == i].values[0]:.2f}', fontsize=9, ha='center', color='g', va='bottom')

  ax1.plot(df.loc[df['Signal'] == -1.0]['index'], 
             df[key_small][df['Signal'] == -1.0], 
             'v', markersize=10, color='r', label='Sell Signal')

  for i in df.loc[df['Signal'] == -1.0]['index']:
        ax1.text(i, df['close'][df['index'] == i].values[0], f'{df["close"][df["index"] == i].values[0]:.2f}', fontsize=9, ha='center', color='r', va='bottom')


  ax1.set_title('Stock Price with Buy and Sell Signals')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Price')
  ax1.legend()
  ax1.grid(True)

  ax2 = plt.subplot(2, 1, 2)

    
  ax2.set_title('Short and Long Ma')
  ax2.set_xlabel('Date')
 
  ax2.legend()
  ax2.grid(True)

  n = max(1, len(df) // 10)  
  plt.xticks(ticks=df['index'][::n], labels=[date.strftime('%Y-%m-%d') for date in df.index][::n], rotation=45)

  plt.tight_layout()
  plt.show()