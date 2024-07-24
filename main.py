from utils import fetch_stock_data, simulate_trades
from utils.plotting.plot_ma_crossover import plot_moving_average_crossover
from strategies.ma_crossover import crossover_signal_with_slope, crossover_signal
from strategies.ema import ema_strategy

import sys


def main():

  if (len(sys.argv[1:]) < 6 or len(sys.argv[1:]) > 6 ):
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
    simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date) 

  
  elif strategy_name == 'MACD':
    pass
  

if __name__ == "__main__":
    main()