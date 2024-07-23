import pandas as pd
import os
from datetime import datetime

def fetch_stock_data(stock_name, interval, start_date, end_date):

    start_date = datetime.strptime(start_date, "%d-%m-%Y").replace(hour=0, minute=0, second=0)
    end_date = datetime.strptime(end_date, "%d-%m-%Y").replace(hour=23, minute=59, second=59)
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


    files_to_read = []
    for year in range(start_date.year, end_date.year + 1):
        for month in months:
            if start_date.year == end_date.year and start_date.month <= int(month) <= end_date.month:
                file_name = f"{stock_name}-{interval}-{month}-{year}.csv"
            elif start_date.year < end_date.year:
                if year == start_date.year and int(month) >= start_date.month:
                    file_name = f"{stock_name}-{interval}-{month}-{year}.csv"
                elif year == end_date.year and int(month) <= end_date.month:
                    file_name = f"{stock_name}-{interval}-{month}-{year}.csv"
                elif start_date.year < year < end_date.year:
                    file_name = f"{stock_name}-{interval}-{month}-{year}.csv"
                else:
                    continue
            else:
                continue
            
            subdirectory = stock_name
            
            current_dir = os.path.dirname(__file__)
            
            file_path = os.path.join(current_dir, '..', 'data', subdirectory, file_name)
            

            if os.path.exists(file_path):
                files_to_read.append(file_path)

  
    df_list = []


    for file in files_to_read:
        df = pd.read_csv(file, parse_dates=['timestamp'])
        df_list.append(df)

    
    all_data = pd.concat(df_list)

 
    filtered_data = all_data[(all_data['timestamp'] >= start_date) & (all_data['timestamp'] <= end_date)]
   
    filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
    filtered_data_sorted = filtered_data.sort_values(by='timestamp')

    return filtered_data_sorted
