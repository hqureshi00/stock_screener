from utils import fetch_stock_data, simulate_trades
from utils.plotting.plot_ma_crossover import plot_moving_average_crossover
from strategies.ma_crossover import crossover_signal_with_slope, crossover_signal
from strategies.ema import ema_strategy
from utils.plotting.plot_ema import plot_ema
from strategies.macd import generate_macd_signals
from strategies.rsi import generate_rsi_signals
from strategies.bollinger_bands import generate_bb_signals
from strategies.candlestick_patterns import generate_candlestick_signals

import sys


def main():

  if (len(sys.argv[1:]) < 5 or len(sys.argv[1:]) > 5 ):
    print("Please enter arguments: stock, start_date, end_date, strategy")
    print("""
    Strategies Available: 
          - MA CrossOver
          - MACD
          - EMA
    """)
    sys.exit()

  stock, start_date, end_date, interval, strategy_name = sys.argv[1:]  
  stock_data = fetch_stock_data(stock, interval, start_date, end_date)
 
  if strategy_name == 'MA CrossOver':
    signals = crossover_signal_with_slope(stock_data)
    stock_data['Signal'] = signals['Signal']
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)  
    plot_moving_average_crossover(stock_data)

  elif strategy_name == 'EMA':
    signals = ema_strategy(stock_data)
    stock_data['Signal'] = signals['Buy_Sell']
    stock_data['EMA_short'] = signals['EMA_short']
    stock_data['EMA_long'] = signals['EMA_long']
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
    plot_ema(stock_data, small=7, long=14) 

  elif strategy_name == 'MACD':
    signals = generate_macd_signals(stock_data)
    stock_data['Signal'] = signals['Signal']
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)


  elif strategy_name == 'CandlestickPatterns':
    signals = generate_candlestick_signals(stock_data)
    stock_data['Signal'] = signals['Signal']
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)

  elif strategy_name == 'BollingerBands':
    signals = generate_bb_signals(stock_data)
    stock_data['Signal'] = signals['Signal']
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)

  elif strategy_name == 'RSI':
    signals = generate_rsi_signals(stock_data)
    stock_data['Signal'] = signals['Signal']
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
  

if __name__ == "__main__":
    main()