from utils import fetch_stock_data, simulate_trades
from utils.plotting.plot_ma_crossover import plot_moving_average_crossover
from strategies.ma_crossover import crossover_signal_with_slope, crossover_signal, moving_average_crossover_signals
from strategies.ema import ema_strategy
from utils.plotting.plot_ema import plot_ema
from utils.plotting.plot_macd import plot_macd
from utils.plotting.plot_rsi import plot_rsi
from strategies.macd import generate_macd_signals
from strategies.rsi import generate_rsi_signals
from strategies.bollinger_bands import generate_bb_signals
from strategies.candlestick_patterns import generate_candlestick_signals
import argparse
import pdb


def main():

  parser = argparse.ArgumentParser(
        description="Process stock data using different trading strategies."
    )
    
  parser.add_argument(
      "stock", 
      type=str, 
      help="The stock symbol (e.g., AAPL for Apple Inc.)"
  )
  parser.add_argument(
      "start_date", 
      type=str, 
      help="The start date in DD-MM-YYYY format (e.g., 01-01-2023)"
  )
  parser.add_argument(
      "end_date", 
      type=str, 
      help="The end date in DD-MM-YYYY format (e.g., 01-01-2023)"
  )
  parser.add_argument(
      "strategy", 
      type=str, 
      choices=["MACrossover", "MACD", "EMA", "RSI"], 
      help="The trading strategy to apply. Options: 'MA CrossOver', 'MACD', 'EMA'"
  )

  parser.add_argument(
        "--interval", 
        type=str, 
        choices=["1min", "5min", "15min", "30min"], 
        default="1min", 
        required=True,
        help="The interval for the data. Options: '1min', '5min', '15min', '30min'. Default is '1min'."
    )


  args = parser.parse_args()

  stock, start_date, end_date, interval, strategy_name = args.stock, args.start_date, args.end_date, args.interval, args.strategy
  stock_data = fetch_stock_data(stock, interval, start_date, end_date)
 
  if strategy_name == 'MACrossover':
    signals = moving_average_crossover_signals(stock_data)
    stock_data['Signal'] = signals['Buy_Sell']
    tr, ex_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)  
    plot_moving_average_crossover(stock_data)

  elif strategy_name == 'EMA':
    signals = ema_strategy(stock_data)
    stock_data['Signal'] = signals['Buy_Sell']
    stock_data['EMA_short'] = signals['EMA_short']
    stock_data['EMA_long'] = signals['EMA_long']
    tr, ex_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
    plot_ema(ex_signals, small=7, long=14) 

  elif strategy_name == 'MACD':
    signals = generate_macd_signals(stock_data)
    stock_data['Signal'] = signals['Buy_Sell']
    stock_data['MACD'] = signals['MACD']
    stock_data['Signal_Line'] = signals['Signal_Line']
    tr, ex_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
    plot_macd(ex_signals)

  elif strategy_name == 'RSI':
    signals = generate_rsi_signals(stock_data)
    stock_data['Signal'] = signals['Buy_Sell']
    tr, ex_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
    plot_rsi(ex_signals)


  # elif strategy_name == 'CandlestickPatterns':
  #   signals = generate_candlestick_signals(stock_data)
  #   stock_data['Signal'] = signals['Signal']
  #   simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)

  # elif strategy_name == 'BollingerBands':
  #   signals = generate_bb_signals(stock_data)
  #   stock_data['Signal'] = signals['Signal']
  #   simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)


  

if __name__ == "__main__":
    main()