import pandas as pd
import matplotlib.pyplot as plt

def plot_moving_average_crossover(stock_data, small=7, long=14):

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