import pandas as pd
import matplotlib.pyplot as plt


def plot_rsi(data, buy_threshold=30, sell_threshold=70):
  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

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

