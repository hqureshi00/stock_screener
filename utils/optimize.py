import pandas as pd
import numpy as np
from utils import fetch_stock_data, simulate_trades
from strategies.ma_crossover import crossover_signal_with_slope, crossover_signal

def optimize_ema_parameters():
    pass

def optimize_ma_crossover_parameters(data, interval, stock_name, start_date, end_date, borrow_rate=0.01, initial_cash=10000):
    results = []

    # Define small and long window ranges based on interval
    if interval == '30min':
        small_win_range = [3, 5]
        long_win_range = [8, 10]
    elif interval == '15min':
        small_win_range = [5, 7]
        long_win_range = [10, 14]
    elif interval == '5min':
        small_win_range = [5, 7]
        long_win_range = [10, 14]
    elif interval == '1min':
        small_win_range = [3, 5]
        long_win_range = [8, 10]
    else:
        raise ValueError("Unsupported interval")

    for small_win in small_win_range:
        for long_win in long_win_range:
            if small_win >= long_win:
                continue  # Skip invalid combinations where small window is not less than long window

            signals = crossover_signal_with_slope(data.copy(), small_win, long_win)
            data_with_signals = data.copy()
            data_with_signals['Signal'] = signals['Signal']

            trades_df = simulate_trades(data_with_signals, 'SMA_Slope', interval, stock_name, start_date, end_date, borrow_rate, initial_cash)

            if not trades_df.empty:
                final_cash_balance = trades_df['cash_balance'].iloc[-1]
                profit = final_cash_balance - initial_cash
                results.append({
                    'small_win': small_win,
                    'long_win': long_win,
                    'profit': profit
                })

    results_df = pd.DataFrame(results)
    return results_df

def run_optimize_func():
    intervals = ['1min', '5min', '15min', '30min']
    stock_names = ['NVDA', 'MSFT', 'NFLX', 'GOOG', 'AAPL']

    start_date = '2021-01-01'
    end_date = '2021-12-31'

    columns = ['stock_name', 'start_date', 'end_date', 'small_win', 'long_win', 'profit']
    df = pd.DataFrame(columns=columns)

    interval_results = {}
    for interval in intervals:
      for stock in stock_names:
        data = fetch_stock_data(stock, interval, start_date, end_date)
        results_df = optimize_ma_crossover_parameters(data, interval, stock, '2021-01-01', '2021-12-31')
        interval_results[interval] = results_df
        # Find the best parameters
        best_params = results_df.loc[results_df['profit'].idxmax()]
        row = {'stock_name': stock, 'start_date': start_date, 'end_date': end_date, 'small_win': best_params['small_win'] , 'long_win': best_params['profit'], 'profit': best_params['profit']}
        
        df = df.append(row, ignore_index=True)

    csv_file_path = 'optimized_parameters_for_moving_average_crossover.csv'
    df.to_csv(csv_file_path, index=False)


