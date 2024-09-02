from utils import fetch_stock_data, simulate_trades
from utils.benchmarking_scripts.benchmark_strategies import benchmark_strategy
from strategies.ma_crossover import crossover_signal, crossover_signal_with_slope
from strategies.ema import ema_strategy
from utils.plotting.plot_ma_crossover import plot_moving_average_crossover, plot_moving_average_crossover_plotly
from utils.plotting.plot_ema import plot_ema, plot_ema_plotly
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
  
  parser.add_argument(
    "--benchmark", 
    action="store_true", 
    help="Run the benchmark function instead of the normal functionality."
)
  
  args = parser.parse_args()

  stock, start_date, end_date, interval, strategy_name = args.stock, args.start_date, args.end_date, args.interval, args.strategy
  stock_data = fetch_stock_data(stock, interval, start_date, end_date)
 
  if strategy_name == 'MACrossover':
    if args.benchmark:
      benchmark_strategy(stock_data, start_date, end_date, stock, strategy_name, interval)
    else:
      signals = crossover_signal(stock_data, small_win=7, long_win=14)
      stock_data['Signal'] = signals['Buy_Sell']
      total_profit, executed_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
      if executed_signals.empty:
        print("No trades")
      else:
        plot_moving_average_crossover_plotly(stock_data, small=7, large=14)

  elif strategy_name == 'EMA':
    if args.benchmark:
      benchmark_strategy(stock_data, start_date, end_date, stock, strategy_name, interval)
    else:
      signals = ema_strategy(stock_data, short_window=7, long_window=14)
      stock_data['Signal'] = signals['Buy_Sell']
      stock_data['EMA_short'] = signals['EMA_short']
      stock_data['EMA_long'] = signals['EMA_long']
      total_profit, executed_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
      if executed_signals.empty:
        print("No trades")
      else:
        plot_ema_plotly(executed_signals, small=7, long=14) 

  elif strategy_name == 'RSI':
    if args.benchmark:
      benchmark_strategy(stock_data, start_date, end_date, stock, strategy_name, interval)
    else:
      signals = generate_rsi_signals(stock_data, buy_threshold=30, sell_threshold=70)
      stock_data['Signal'] = signals['Buy_Sell']
      total_profit, executed_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
      if executed_signals.empty:
        print("No trades")

      else:
        plot_rsi(executed_signals, buy_threshold=30, sell_threshold=70)

  elif strategy_name == 'MACD':
    signals = generate_macd_signals(stock_data)
    stock_data['Signal'] = signals['Buy_Sell']
    stock_data['MACD'] = signals['MACD']
    stock_data['Signal_Line'] = signals['Signal_Line']
    total_profit, executed_signals = simulate_trades(stock_data, strategy_name, interval, stock, start_date, end_date)
    plot_macd(executed_signals)



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