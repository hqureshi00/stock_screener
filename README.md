# Stock Screener

This project processes and analyses stock data using different trading strategies.

## Requirements

- Python 3.x

## Installation

Clone the repository or download the script.

## Usage

To run the script, use the following command format:

```sh
python script.py [stock] [start_date] [end_date] [strategy] [--interval INTERVAL]
```

### Positional Arguments

1. `stock`: The stock symbol (e.g., AAPL for Apple Inc.)
2. `start_date`: The start date in DD-MM-YYYY format (e.g., 01-01-2023)
3. `end_date`: The end date in DD-MM-YYYY format (e.g., `01-01-2023)
4. `strategy`: The trading strategy to apply. Options:
   - `MACrossOver`
   - `MACD`
   - `EMA`
   
### Other Required Arguments

- `--interval {1min,5min,15min,30min}`: The interval for the data. Options:
  - `1min`
  - `5min`
  - `15min`
  - `30min`
  
  Default is `1min`.


### Optional Arguments

- `-h, --help`: Show this help message and exit

### Example Command

To compute the output profit for stock `AAPL` and `start_date` `01-01-2023` and `end_date` `01-01-2023` with `MACD` as strategy 
and interval of 5min, use the following command:

```sh
python main.py AAPL 01-01-2023 01-02-2023 MACD --interval 5min
```