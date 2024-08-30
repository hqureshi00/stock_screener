import pandas as pd
"""
modify this code to combine strategies
1. Obtain signals for each strategy as a column Signal['strategy_name']
"""

def custom_rule_combination(signals):
    combined_signal = pd.Series(0, index=signals.index)  # Initialize with hold (0)
    
    buy_condition = (signals['RSI'] == 1) & (signals['MovingCrossover'] == 1)
    sell_condition = (signals['RSI'] == -1) & (signals['MovingCrossover'] == -1)
    
    combined_signal[buy_condition] = 1
    combined_signal[sell_condition] = -1
    
    return combined_signal

def combine_strategies(data, strategies, **kwargs):
    combined_signals = pd.DataFrame(index=data.index)
    for strategy in strategies:
        signals = strategy(data, **kwargs)
        combined_signals = pd.concat([combined_signals, signals], axis=1)
    # Implement logic to handle combined signals
    return combined_signals