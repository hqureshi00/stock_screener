import pandas as pd
import os

import os
import pandas as pd

def simulate_trades(stock_data, strategy_name, interval, stock_name, start_date, end_date, borrow_rate=0.01, initial_cash=10000):
    
    trades = []
    position = None
    file_name = f'{stock_name}_{interval}_{strategy_name}_{start_date}_{end_date}.csv'
    cash_balance = initial_cash

    # Create a new DataFrame to store executed signals
    executed_signals = stock_data.copy()
    executed_signals['Executed_Buy_Sell'] = 0  # Initialize with 0 (no action)

    for i, row in stock_data.iterrows():
        if row['Signal'] == 1 and (position is None or position['position_type'] == 'short'):  # Buy signal
            if position and position['position_type'] == 'short':  # Cover short position
                position['sell_date'] = row.timestamp
                position['sell_value'] = row['close']
                position['profit_loss'] = (position['buy_value'] - position['sell_value']) * position['num_shares']

                # Calculate borrow cost
                days_held = (row.timestamp - position['buy_date']).days
                borrow_cost = position['borrowed_value'] * borrow_rate * (days_held / 365)

                cash_balance -= (position['num_shares'] * position['sell_value']) + borrow_cost  # Adjust cash balance for covering
                position['cash_balance'] = cash_balance - borrow_cost  # Subtracting the borrow cost
                position['borrow_cost'] = borrow_cost
                trades.append(position)
                executed_signals.at[i, 'Executed_Buy_Sell'] = 1  # Mark this row as an executed buy signal
                position = None  # Clear the position

            # Open new long position
            position = {
                'buy_date': row.timestamp,
                'buy_value': row['close'],
                'num_shares': cash_balance / row['close'],  # Example: Buying shares worth of available cash balance
                'position_type': 'long',
                'cash_balance': cash_balance
            }
            cash_balance -= position['num_shares'] * row['close']  # Adjust cash balance for purchase
            executed_signals.at[i, 'Executed_Buy_Sell'] = 1  # Mark this row as an executed buy signal

        elif row['Signal'] == -1 and (position is None or position['position_type'] == 'long'):  # Sell signal
            if position and position['position_type'] == 'long':  # Close long position
                position['sell_date'] = row.timestamp
                position['sell_value'] = row['close']
                position['profit_loss'] = (position['sell_value'] - position['buy_value']) * position['num_shares']

                cash_balance += position['num_shares'] * position['sell_value']  # Adjust cash balance for sale
                position['cash_balance'] = cash_balance
                trades.append(position)
                executed_signals.at[i, 'Executed_Buy_Sell'] = -1  # Mark this row as an executed sell signal
                position = None  # Clear the position

            # Open new short position
            position = {
                'buy_date': row.timestamp,
                'buy_value': row['close'],
                'num_shares': cash_balance / row['close'],  # Example: Selling short shares worth of available cash balance
                'position_type': 'short',
                'cash_balance': cash_balance,
                'borrowed_value': row['close'] * (cash_balance / row['close'])  # Value of borrowed shares
            }
            cash_balance += position['num_shares'] * row['close']
            executed_signals.at[i, 'Executed_Buy_Sell'] = -1  # Mark this row as an executed sell signal

    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades)

    # Save trades to CSV
    folder_name = f'data/trade_iterations'
    file_path = os.path.join(folder_name, file_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    trades_df.to_csv(file_path, index=False)

    if trades_df.empty:
        print("No trades")
        return None
    else:
        final_cash_balance = trades_df['cash_balance'].iloc[-1]
        total_profit = final_cash_balance - initial_cash
        print(f"Final Profit: ${total_profit}")
        executed_signals = executed_signals[executed_signals['Executed_Buy_Sell'] != 0]
        return total_profit, executed_signals

    # Filter out rows where no trade was executed
    # executed_signals = executed_signals[executed_signals['Executed_Buy_Sell'] != 0]

    # return executed_signals

def simulate_trades_old(stock_data, strategy_name, interval, stock_name, start_date, end_date, borrow_rate=0.01, initial_cash=10000):
    
    trades = []
    position = None
    file_name = f'{stock_name}_{interval}_{strategy_name}_{start_date}_{end_date}.csv'

    cash_balance = initial_cash

    # Create a new DataFrame to store executed signals
    executed_signals = stock_data.copy()
    executed_signals['Executed_Buy_Sell'] = 0  # Initialize with 0 (no action)

    for i, row in stock_data.iterrows():
        # if we have a buy signal and either we have bought short at a prior point or we are buying for the first time
        if row['Signal'] == 1 and (position is None or position['position_type'] == 'short'):  # Buy signal and no current position or covering a short position
            if position and position['position_type'] == 'short':  # Cover short position
                position['sell_date'] = row.timestamp
                position['sell_value'] = row['close']
                position['profit_loss'] = (position['buy_value'] - position['sell_value']) * position['num_shares']

                # Calculate borrow cost
                days_held = (row.timestamp - position['buy_date']).days
                borrow_cost = position['borrowed_value'] * borrow_rate * (days_held / 365)

                cash_balance -= (position['num_shares'] * position['sell_value']) + borrow_cost  # Adjust cash balance for covering
                position['cash_balance'] = cash_balance - borrow_cost  # Subtracting the borrow cost
                position['borrow_cost'] = borrow_cost
                trades.append(position)
                position = None  # Clear the positionx

            # Open new long position
            position = {
                'buy_date': row.timestamp,
                'buy_value': row['close'],
                'num_shares': cash_balance / row['close'],  # Example: Buying shares worth of available cash balance
                'position_type': 'long',
                'cash_balance': cash_balance
            }
            cash_balance -= position['num_shares'] * row['close']  # Adjust cash balance for purchase

        elif row['Signal'] == -1 and (position is None or position['position_type'] == 'long'):  # Sell signal and no current position or opening a short position
            if position and position['position_type'] == 'long':  # Close long position
                position['sell_date'] = row.timestamp
                position['sell_value'] = row['close']
                position['profit_loss'] = (position['sell_value'] - position['buy_value']) * position['num_shares']

                cash_balance += position['num_shares'] * position['sell_value']  # Adjust cash balance for sale
                position['cash_balance'] = cash_balance
                trades.append(position)
                position = None  # Clear the position

            # Open new short position
            position = {
                'buy_date': row.timestamp,
                'buy_value': row['close'],
                'num_shares': cash_balance / row['close'],  # Example: Selling short shares worth of available cash balance
                'position_type': 'short',
                'cash_balance': cash_balance,
                'borrowed_value': row['close'] * (cash_balance / row['close'])  # Value of borrowed shares
            }
            cash_balance += position['num_shares'] * row['close']

    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades)

    # Save to CSV
    folder_name = f'data/trade_iterations'
    file_path = os.path.join(folder_name, file_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    trades_df.to_csv(file_path, index=False)

    if trades_df.empty:
        print("No trades")
    else:
        final_cash_balance = trades_df['cash_balance'].iloc[-1]
        print(f"Final Profit: ${final_cash_balance - initial_cash}")

  
