import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Plotting function

def plot_macd(df, short_window=12, long_window=26, signal_window=9):
    # Calculate MACD strategy
    
    plt.figure(figsize=(14, 10))
    # Subplot 1: Close Prices and Buy/Sell Signals
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], label='Close Price')
    
    # Plot Buy signals
    buy_signals = df[df['Signal'] == 1.0]
    ax1.plot(buy_signals.index, df['close'][buy_signals.index], '^', markersize=10, color='g', label='Buy Signal')
    for i in buy_signals.index:
        ax1.text(i, df['close'][i], f'{df["close"][i]:.2f}', fontsize=9, ha='center', color='g', va='bottom')
    
    # Plot Sell signals
    sell_signals = df[df['Signal'] == -1.0]
    ax1.plot(sell_signals.index, df['close'][sell_signals.index], 'v', markersize=10, color='r', label='Sell Signal')
    for i in sell_signals.index:
        ax1.text(i, df['close'][i], f'{df["close"][i]:.2f}', fontsize=9, ha='center', color='r', va='bottom')
    
    ax1.set_title('Stock Price with Buy and Sell Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: MACD and Signal Line
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df.index, df['MACD'], label='MACD', color='b')
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line', color='r')
    
    ax2.set_title('MACD and Signal Line')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    #TODO: why are so many buy and sell signal being generated in the MACD when the signal lines are not crossing
    #TODO: why no trades for MACD