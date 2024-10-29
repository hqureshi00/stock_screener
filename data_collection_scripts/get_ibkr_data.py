from ctypes import util
from ib_insync import *
import pandas as pd

# Connect to IBKR TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)  # Use 4001 for live trading

# Define the MSFT contract
contract = Stock('MSFT', 'SMART', 'USD')

# Set the date range for September 2024
end_date_time = '20240930 23:59:59'  # End of September 2024
duration_str = '1 M'  # 1 month duration

# Request historical 1-minute data including pre and post market
bars = ib.reqHistoricalData(
    contract,
    endDateTime=end_date_time,
    durationStr=duration_str,
    barSizeSetting='1 min',
    whatToShow='TRADES',  # 'TRADES' for actual trading prices
    useRTH=False,  # Set to False to include pre-market and post-market data
    formatDate=1
)

# Convert the data to a DataFrame for easy manipulation
df = util.df(bars)

# Rename the columns to match the desired format
df.rename(columns={
    'date': 'timestamp',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume'
}, inplace=True)



df = df.sort_values(by='timestamp', ascending=False)
df = df[df['volume'] != 0]

df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')


# Save to CSV with the specified format
df.to_csv('MSFT_Sep2024_1min.csv', columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'], index=False)

# Disconnect from IBKR
ib.disconnect()

print("Data retrieval and storage complete!")
