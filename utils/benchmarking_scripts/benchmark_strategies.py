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

def plot_macd_heatmap(sorted_results_df, key_names, xlabel, ylabel, title, zlabel):
    unique_values = sorted_results_df[key_names[2]].unique()
    
    for value in unique_values:
        subset_df = sorted_results_df[sorted_results_df[key_names[2]] == value]
        
        plt.figure(figsize=(12, 6))
        pivot_table = subset_df.pivot(index=key_names[0], columns=key_names[1], values="total_profit")
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"fontsize": 5})
        
        plt.title(f"{title} ({zlabel}={value})")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


def plot_moving_crossover_heatmap(sorted_results_df, key_names, xlabel, ylabel, title):
    plt.figure(figsize=(12, 6))
    # Create a heatmap to visualize the profit for different fast and slow window combinations
    pivot_table = sorted_results_df.pivot(index=key_names[0], columns=key_names[1], values="total_profit")
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={"fontsize": 5})

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def get_results(data, start_date, end_date, stock_name, strategy_name, interval, parameter1, parameter2, key_names):
    results = []
    for par1 in parameter1:
        for par2 in parameter2:
            if par1 >= par2:
                continue  # Skip invalid combinations where fast window >= slow window
            
            # Apply the moving crossover strategy
            if strategy_name == 'MACrossover' or strategy_name == 'EMA':
                data_with_signals = crossover_signal(data.copy(), par1, par2)
            elif strategy_name == 'RSI':
                data_with_signals = generate_rsi_signals(data.copy(), par1, par2)
            data['Signal'] = data_with_signals['Buy_Sell']
            
            # Simulate the trades and calculate profit
            total_profit, executed_signals = simulate_trades(data, strategy_name, interval, stock_name, start_date, end_date) 
            # breakpoint()
            # Store the results
            results.append({
                key_names[0]: par1,
                key_names[1]: par2,
                'total_profit': total_profit,
            })

    results_df = pd.DataFrame(results)
    sorted_results_df = results_df.sort_values(by='total_profit', ascending=False)
            
    return sorted_results_df
"""
In main, the flow would be:
1. if the list of strategies is multiple and benchmark flag present, 
then you head straight into the benchmark combined strategies function  

"""

### Verify for 10 and 11
### Data Prep and finding out the best metrics
### NVidia fix past data /10
def benchmark_combined_strategy(data, start_date, end_date, stock_name, strategies, interval):
    ## create a dictionary with strategy name and its possible parameters and keynames

    signals = pd.DataFrame(index=data.index)

    for strategy in strategies:
        strategy_name = strategy['name']
        parameter1, parameter2 = strategy['parameter1'], strategy['parameter2']
        key_names = ()
        
        if strategy_name == 'MACrossover':
            # Calculate MACrossover signals (assuming you have a function to do this)
            signals['MACrossover'] = crossover_signal(data.copy(), parameter1, parameter2)
        
        elif strategy_name == 'EMA':
            # Calculate EMA signals
            signals['EMA'] = crossover_signal(data.copy(), parameter1, parameter2)

        elif strategy_name == 'RSI':
            # Calculate RSI signals
            signals['RSI'] = generate_rsi_signals(data.copy(), parameter1, parameter2)
    # Initialize combined signal
    
    
    # Combine the signals (example using RSI and MACrossover)

    if len(strategies) > 1:
        combined_signal = pd.Series(index=data.index, data=0)
        if 'RSI' in signals.columns and 'MACrossover' in signals.columns:
            buy_condition = (signals['RSI'] == 1) & (signals['MACrossover'] == 1)
            sell_condition = (signals['RSI'] == -1) & (signals['MACrossover'] == -1)
            
            combined_signal[buy_condition] = 1
            combined_signal[sell_condition] = -1

        # Evaluate the combined signal
        # results = evaluate_combined_signal(data, combined_signal, start_date, end_date, stock_name, interval)
        
        # # Plot and save results
        # plot_combined_strategy_results(results, strategies)
        # save_combined_strategy_results_as_csv(results, start_date, end_date, stock_name, interval)

    elif len(strategies) == 1:
        combined_signal = signals[strategies[0]]
        key_names = ()
        parameters = []
        parameter_range = ''
        xlabel = ''
        ylabel = ''
        title = ''
        ## get total profit  
        results = get_results(data, start_date, end_date, stock_name, strategy_name, interval, parameter1, parameter2, key_names)
        ##plot as a heatmap
        plot_moving_crossover_heatmap(results, key_names, xlabel, ylabel, title)
        ## save as csv
        save_moving_crossover_as_csv(results, start_date, end_date, stock_name, strategy_name, interval, parameter_range)

    
def benchmark_strategy(data, start_date, end_date, stock_name, strategy_name, interval):
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
        parameter1 = [i for i in range(20,41)]
        parameter2 = [i for i in range(60, 81)]
        parameter_range = f'_ buy_{parameter1[0]}_{parameter1[-1]}_sell_{parameter2[0]}_{parameter2[-1]}'
        key_names = ('buy_threshold', 'sell_threshold')
        title = 'Total Profit for Different Buy and Sell Treshold Combinations'
        xlabel = 'Sell Thresholds'
        ylabel = 'Buy Thresholds'

    elif strategy_name == 'MACD':
        parameter1 = [i for i in range(12, 27)]  # Fast period
        parameter2 = [i for i in range(26, 51)]  # Slow period
        parameter3 = [i for i in range(9, 19)]   # Signal period
        parameter_range = f'_fast_{parameter1[0]}_{parameter1[-1]}_slow_{parameter2[0]}_{parameter2[-1]}_signal_{parameter3[0]}_{parameter3[-1]}'
        key_names = ('macd_fast', 'macd_slow', 'macd_signal')

        title = 'Total Profit for Different MACD Combinations'
        xlabel = 'Slow MACD Period'
        ylabel = 'Fast MACD Period'
        zlabel = 'Signal Period'
    
    """
    if multiple strategies, then you need to also send in multiple parameters, args: strategy_names, parameter_list for each strategy    
    """

    if strategy_name != 'MACD':
    ## get total profit  
        results = get_results(data, start_date, end_date, stock_name, strategy_name, interval, parameter1, parameter2, key_names)
        ##plot as a heatmap
        plot_moving_crossover_heatmap(results, key_names, xlabel, ylabel, title)
        ## save as csv
        save_moving_crossover_as_csv(results, start_date, end_date, stock_name, strategy_name, interval, parameter_range)

    else:
        pass
