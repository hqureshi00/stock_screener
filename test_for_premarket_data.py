import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
from datetime import datetime
from utils.fetch_stock_data import fetch_stock_data

def get_premarket_data(symbol, interval, start_date, end_date):
    
    data = fetch_stock_data(symbol, interval, start_date, end_date)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    premarket_data = data.between_time('04:00', '09:30')
    
    return premarket_data
 

symbol = 'AAPL'
premarket_data = get_premarket_data(symbol, '1min', '15-01-2024', '20-02-2024')
print(premarket_data)