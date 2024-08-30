import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_rsi(df, buy_threshold=30, sell_threshold=70):
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'],
                                 name='Candlestick'))

    # Add Buy and Sell markers
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                             mode='markers', marker=dict(color='green', size=10),
                             name='Buy Signal'))

    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                             mode='markers', marker=dict(color='red', size=10),
                             name='Sell Signal'))

    # Add RSI plot in a secondary y-axis
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                             line=dict(color='purple', width=2),
                             name='RSI',
                             yaxis='y2'))

    # Add RSI thresholds
    fig.add_trace(go.Scatter(x=df.index, y=[buy_threshold]*len(df),
                             line=dict(color='green', width=1, dash='dash'),
                             name=f'Buy Threshold ({buy_threshold})',
                             yaxis='y2'))

    fig.add_trace(go.Scatter(x=df.index, y=[sell_threshold]*len(df),
                             line=dict(color='red', width=1, dash='dash'),
                             name=f'Sell Threshold ({sell_threshold})',
                             yaxis='y2'))

    # Update layout
    fig.update_layout(
        title="RSI and Closing Price",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(title="RSI", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        template="plotly_white"
    )

    fig.show()

# Example usage with your DataFrame
# df should have columns: 'open', 'high', 'low', 'close', 'RSI', 'Buy_Sell'

def plot_rsi_old(data, buy_threshold=30, sell_threshold=70):
  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    breakpoint()
    # Plot Close Prices
    ax1.plot(data.index, data['close'], color='blue', label='Close Price')

    # Plot Buy Signals
    buy_signals = data[data['Signal'] == 1]
    ax1.plot(buy_signals.index, buy_signals['close'], '^', markersize=10, color='green', label='Buy Signal', lw=0)

    # Plot Sell Signals
    sell_signals = data[data['Signal'] == -1]
    ax1.plot(sell_signals.index, sell_signals['close'], 'v', markersize=10, color='red', label='Sell Signal', lw=0)

    ax1.set_title('Close Prices with Buy/Sell Signals')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot RSI
    ax2.plot(data.index, data['RSI'], color='red', label='RSI')
    ax2.axhline(sell_threshold, color='gray', linestyle='--', label='Overbought (70)')
    ax2.axhline(buy_threshold, color='gray', linestyle='--', label='Oversold (30)')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI Value')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

