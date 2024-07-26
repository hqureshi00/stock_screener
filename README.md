# Stock Screener

This project processes and analyses stock data using different trading strategies.

## Requirements

- Python 3.x

## Installation

Clone the repository or download the script.

## Usage

usage: script.py [-h] [--interval {1min,5min,15min,30min}] stock start_date end_date strategy

Process stock data using different trading strategies.

positional arguments:
  stock                 The stock symbol (e.g., AAPL for Apple Inc.)
  start_date            The start date in YYYY-MM-DD format (e.g., 2023-01-01)
  end_date              The end date in YYYY-MM-DD format (e.g., 2023-12-31)
  strategy              The trading strategy to apply. Options: 'MA CrossOver', 'MACD', 'EMA'

optional arguments:
  -h, --help            show this help message and exit
  --interval {1min,5min,15min,30min}
                        The interval for the data. Options: '1min', '5min', '15min', '30min'. Default is '1min'.

Example Command: python script.py [stock] [start_date] [end_date] [strategy] [--interval INTERVAL]