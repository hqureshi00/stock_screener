import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

def plot_ema_plotly(stock_data, small=12, long=26):
    key_small = f'EMA_short'
    key_large = f'EMA_long'

    label1 = f'{small}-day EMA'
    label2 = f'{long}-day EMA'

    df = pd.DataFrame(stock_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.dropna(subset=['close', key_small, key_large], inplace=True)
    df['index'] = range(len(df))

    # Create a figure
    fig = go.Figure()

    # Add the close price line
    fig.add_trace(go.Scatter(x=df['index'], y=df['close'], mode='lines', name='Close Price'))

    # Add the small EMA
    fig.add_trace(go.Scatter(x=df['index'], y=df[key_small], mode='lines', name=label1))

    # Add the large EMA
    fig.add_trace(go.Scatter(x=df['index'], y=df[key_large], mode='lines', name=label2))

    # Add buy signals
    buy_signals = df[df['Signal'] == 1.0]
    fig.add_trace(go.Scatter(x=buy_signals['index'], y=buy_signals[key_small], mode='markers', 
                             marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy Signal'))
    
    # Add sell signals
    sell_signals = df[df['Signal'] == -1.0]
    fig.add_trace(go.Scatter(x=sell_signals['index'], y=sell_signals[key_small], mode='markers', 
                             marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='Sell Signal'))

    # Add annotations for Buy/Sell signals
    for i in buy_signals['index']:
        fig.add_annotation(x=i, y=df['close'][df['index'] == i].values[0],
                           text=f"{df['close'][df['index'] == i].values[0]:.2f}",
                           showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="green"))

    for i in sell_signals['index']:
        fig.add_annotation(x=i, y=df['close'][df['index'] == i].values[0],
                           text=f"{df['close'][df['index'] == i].values[0]:.2f}",
                           showarrow=True, arrowhead=1, ax=0, ay=30, font=dict(color="red"))

    # Update the layout
    fig.update_layout(
        title='Stock Price with Buy and Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            tickmode='array',
            tickvals=df['index'][::max(1, len(df) // 10)],
            ticktext=[date.strftime('%Y-%m-%d') for date in df.index][::max(1, len(df) // 10)],
            tickangle=45
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
        template='plotly_white',
        height=600,
        width=1000
    )

    # Show the plot
    fig.show()

# Example usage:
# plot_ema_plotly(stock_data)

def plot_ema(stock_data, small=12, long=26):

  key_small = f'EMA_short'
  key_large = f'EMA_long'

  label1 = f'{small}-day EMA'
  label2 = f'{long}-day EMA'
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