import requests
import pandas as pd
import time
import os

def get_data_from_url(stock, month, year, interval):
  # Step 1: Capture JSON data from a URL
  url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock}&interval={interval}&apikey=HXA9JB1IU4NTPEND&month={year}-{month}&outputsize=full'  # Replace with your URL
  response = requests.get(url, timeout=30)

  # Check if the request was successful
  if response.status_code == 200:
      json_data = response.json()  # Step 2: Convert JSON data into a dictionary

  else:
      print(f"Failed to retrieve data: {response.status_code}")
      json_data = None

  return json_data

def convert_to_dataframe(json_data):
   df = pd.DataFrame()
   time_series_key = ''
   for key in json_data.keys():
      if 'Time Series' in key:
         time_series_key = key
    
   data = json_data[time_series_key]
   df = pd.DataFrame.from_dict(data, orient='index')

   df.index = pd.to_datetime(df.index)

   df.columns = ['open', 'high', 'low', 'close', 'volume']
   df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': int})

   df.reset_index(inplace=True)

  # Rename the index column
   df.rename(columns={'index': 'timestamp'}, inplace=True)


   return df
      

def save_to_csv(df, month, year, timeframe, stock):
   csv_filename = f'{stock}-{timeframe}-{month}-{year}.csv'
   folder_name = f'data/{stock}'
   file_path = os.path.join(folder_name, csv_filename)

   if not os.path.exists(folder_name):
    os.makedirs(folder_name)
   df.to_csv(file_path, index=False)

def generate_month_year_combinations(start_month, start_year, end_month, end_year):
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    start_index = months.index(start_month)
    end_index = months.index(end_month)

    month_year_combinations = []

    for year in range(int(start_year), int(end_year) + 1):
        if year == int(start_year):
            start_idx = start_index
        else:
            start_idx = 0

        if year == int(end_year):
            end_idx = end_index + 1
        else:
            end_idx = len(months)

        for idx in range(start_idx, end_idx):
            month_year_combinations.append((months[idx], str(year)))

    return month_year_combinations


def main():
  start_month = '05'
  start_year = '2014'
  end_month = '06'
  end_year = '2024'
  stocks = ['NVDA', 'MSFT', 'AAPL', 'GOOG', 'NFLX', 'TSLA']
  # stocks = ['NVDA', 'MSFT']
  intraday_durations = ['1min', '5min', '15min', '30min', '60min']

  combinations = generate_month_year_combinations(start_month, start_year, end_month, end_year)

  num_requests = 0

  for stock in stocks:
    for interval in intraday_durations:     
      for month, year in combinations:
        print(f'Getting data for {stock}, {interval}, {month}, {year}')
        json_data = get_data_from_url(stock, month, year, interval)
        df = convert_to_dataframe(json_data=json_data)
        save_to_csv(df, month, year, interval, stock)
        num_requests += 1

      if num_requests % 70 == 0:
        time.sleep(60)
        print('###############################################')
        print(f'{num_requests} requests done; sleeping for 60s')
        print('###############################################')

if __name__ == "__main__":
   main()