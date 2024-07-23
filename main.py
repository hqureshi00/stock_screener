from utils import fetch_stock_data, simulate_trades
from utils.plotting.plot_ma_crossover import plot_moving_average_crossover
from strategies.ma_crossover import crossover_signal_with_slope, crossover_signal
import sys


def main():

  if (len(sys.argv[1:]) < 6 or len(sys.argv[1:]) > 6 ):
    print("Please enter arguments: stock, start_date, end_date, interval with spaces")
    sys.exit()

  stock, start_date, end_date, interval, small, large = sys.argv[1:]  
  stock_data = fetch_stock_data(stock, interval, start_date, end_date)
 
  signals = crossover_signal_with_slope(stock_data, small, large)
  stock_data['Signal'] = signals['Signal']

  simulate_trades(stock_data, 'MA CrossOver', interval, stock, start_date, end_date)

  plot_moving_average_crossover(stock_data, small, large)


if __name__ == "__main__":
    main()