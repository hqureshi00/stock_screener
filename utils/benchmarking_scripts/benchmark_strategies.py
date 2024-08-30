import pandas as pd
from strategies.ma_crossover import crossover_signal
from strategies.rsi import generate_rsi_signals
from utils import simulate_trades, fetch_stock_data
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pdb

def save_moving_crossover_as_csv(sorted_results_df, start_date, end_date, stock_name, strategy_name, interval, parameter_range):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f'{start_date}_{end_date}_{stock_name}_{interval}_{parameter_range}_{strategy_name}_{timestamp}.csv'

    folder_name = 'test_runs'  # Now it's a subdirectory under the current script's directory
    file_path = os.path.join(os.path.dirname(__file__), folder_name, file_name)  # Save in a subdirectory within the same folder as the script
    
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        
    try:
        sorted_results_df.to_csv(file_path, index=False)
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")


def plot_moving_crossover_heatmap(sorted_results_df, key_names, xlabel, ylabel, title):
    plt.figure(figsize=(12, 6))
    # Create a heatmap to visualize the profit for different fast and slow window combinations
    pivot_table = sorted_results_df.pivot(index=key_names[0], columns=key_names[1], values="total_profit")
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"fontsize": 5})

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def moving_crossover_benchmark(data, start_date, end_date, stock_name, strategy_name, interval):
    parameter1, parameter2 = [], []
    parameter_range = ''
    key_names = ()
    title, xlabel, ylabel = '', '', ''
    

    if strategy_name == 'MACrossover':
        parameter1 = [i for i in range(5,21)] #fast window
        parameter2 = [i for i in range(20, 51)] #slow window
        parameter_range = f'_fast_{parameter1[0]}_{parameter1[-1]}_slow_{parameter2[0]}_{parameter2[-1]}'
        key_names = ('fast_window', 'slow_window')

        title = 'Total Profit for Different Moving Average Combinations'
        xlabel = 'Slow Moving Average Window'
        ylabel = 'Fast Moving Average Window'

    elif strategy_name == 'EMA':
        parameter1 = [i for i in range(5,16)] #fast window
        parameter2 = [i for i in range(15, 46)] #slow window
        parameter_range = f'_fast_{parameter1[0]}_{parameter1[-1]}_slow_{parameter2[0]}_{parameter2[-1]}'
        key_names = ('ema_short', 'ema_long')
        title = 'Total Profit for Different EMA Combinations'
        xlabel = 'Slow Moving EMA Window'
        ylabel = 'Fast Moving EMA Window'

    elif strategy_name == 'RSI':
        buy_thresholds = [i for i in range(20,41)]
        sell_thresholds = [i for i in range(60, 81)]
        parameter_range = f'_ buy_{buy_thresholds[0]}_{buy_thresholds[-1]}_sell_{sell_thresholds[0]}_{sell_thresholds[-1]}'
        key_names = ('buy_threshold', 'sell_threshold')
        title = 'Total Profit for Different Buy and Sell Treshold Combinations'
        xlabel = 'Sell Thresholds'
        ylabel = 'Buy Thresholds'
    
    
    results = []
    for par1 in parameter1:
        for par2 in parameter2:
            if par1 >= par2:
                continue  # Skip invalid combinations where fast window >= slow window
            
            # Apply the moving crossover strategy
            data_with_signals = crossover_signal(data.copy(), par1, par2)
            data['Signal'] = data_with_signals['Buy_Sell']
            
            # Simulate the trades and calculate profit
            total_profit, executed_signals = simulate_trades(data, strategy_name, interval, stock_name, start_date, end_date) 
            
            # Store the results
            results.append({
                key_names[0]: par1,
                key_names[1]: par2,
                'total_profit': total_profit,
            })
    
    # Convert the results to a DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    sorted_results_df = results_df.sort_values(by='total_profit', ascending=False)

    ##plot as a heatmap
    plot_moving_crossover_heatmap(sorted_results_df, key_names, xlabel, ylabel, title)

    ## save as csv
    save_moving_crossover_as_csv(sorted_results_df, start_date, end_date, stock_name, strategy_name, interval, parameter_range)
